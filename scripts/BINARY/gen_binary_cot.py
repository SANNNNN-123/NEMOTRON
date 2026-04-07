"""
Generate verified CoT traces for BINARY problems in train.csv.

For each problem we use the gold answer as an extra constraint (9th example)
so that first-match-wins over the canonical library locks the TRUE gate per
bit (kills coincidental matches). We then emit a forum-style trace listing
the decomposition, the per-bit gate choice, a bit-serial VER step, and the
final boxed answer.

Output: data/train_binary_cot.csv with columns
    id, prompt, answer, type, generated_cot
"""

import re
import itertools
import pandas as pd

BIN_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
TGT_RE = re.compile(r"determine the output for:\s*([01]{8})")


# ---------- canonical library (matches forum distribution) ----------
def _build_library():
    lib = []
    # Level 0: constants
    lib.append(("C0", 0, lambda b: 0))
    lib.append(("C1", 0, lambda b: 1))
    # Level 1
    lib.append(("ID",  1, lambda b: b[0]))
    lib.append(("NOT", 1, lambda b: 1 - b[0]))
    # Level 2: 6 symmetric gates
    two = {
        "AND":  lambda a, b: a & b,
        "OR":   lambda a, b: a | b,
        "XOR":  lambda a, b: a ^ b,
        "XNOR": lambda a, b: 1 - (a ^ b),
        "NAND": lambda a, b: 1 - (a & b),
        "NOR":  lambda a, b: 1 - (a | b),
    }
    for n, f in two.items():
        lib.append((n, 2, (lambda f: lambda b: f(b[0], b[1]))(f)))
    # Level 3
    three = {
        "CHOICE": lambda a, b, c: b if a else c,
        "MAJ3":   lambda a, b, c: 1 if (a + b + c) >= 2 else 0,
        "PAR3":   lambda a, b, c: a ^ b ^ c,
    }
    for n, f in three.items():
        lib.append((n, 3, (lambda f: lambda b: f(b[0], b[1], b[2]))(f)))
    # Level 4: the small set from the forum
    four = {
        "AOA":  lambda a, b, c, d: (a & b) | (c & d),
        "OAO":  lambda a, b, c, d: (a | b) & (c | d),
        "PAR4": lambda a, b, c, d: a ^ b ^ c ^ d,
        "XX":   lambda a, b, c, d: (a ^ b) ^ (c ^ d),
        "AXA":  lambda a, b, c, d: (a & b) ^ (c & d),
    }
    for n, f in four.items():
        lib.append((n, 4, (lambda f: lambda b: f(b[0], b[1], b[2], b[3]))(f)))
    return lib


LIBRARY = _build_library()


def parse_problem(prompt):
    pairs = BIN_RE.findall(prompt)
    m = TGT_RE.search(prompt)
    if not m or not pairs:
        return None
    examples = [(tuple(int(c) for c in i), tuple(int(c) for c in o))
                for i, o in pairs]
    target = tuple(int(c) for c in m.group(1))
    return examples, target


def find_gate(bit_idx, rows):
    """rows = list of (input, wanted_bit). Return (name, arity, fn, perm)
    for first matching gate in library scan order, or a truth-table fallback."""
    for name, arity, fn in LIBRARY:
        if arity == 0:
            if all(fn(()) == w for _, w in rows):
                return (name, 0, fn, ())
            continue
        for perm in itertools.permutations(range(8), arity):
            ok = True
            for inp, w in rows:
                if fn(tuple(inp[i] for i in perm)) != w:
                    ok = False
                    break
            if ok:
                return (name, arity, fn, perm)

    # Fallback: find smallest k-subset of inputs whose partial truth table
    # is consistent with all rows. Synthesize a lookup-gate.
    for k in range(2, 6):
        for subset in itertools.combinations(range(8), k):
            table = {}
            bad = False
            for inp, w in rows:
                key = tuple(inp[i] for i in subset)
                if key in table and table[key] != w:
                    bad = True
                    break
                table[key] = w
            if bad:
                continue
            tbl = dict(table)
            default = max(set(tbl.values()), key=list(tbl.values()).count)
            fn = (lambda tbl, default: lambda b: tbl.get(tuple(b), default))(tbl, default)
            return (f"TT{k}", k, fn, subset)
    return None


def fmt_gate(name, arity, perm):
    if arity == 0:
        return name
    if arity == 1:
        return f"{name}(in[{perm[0]}])"
    return f"{name}(" + ",".join(f"in[{i}]" for i in perm) + ")"


def serial_trace(name, arity, fn, perm, inputs):
    """Return a bit-serial computation string across the example rows."""
    if arity == 0:
        return f"{name}={fn(())}"
    parts = []
    two_ops = {"AND": "&", "OR": "|", "XOR": "^",
               "XNOR": "^", "NAND": "&", "NOR": "|"}
    for inp in inputs:
        bits = tuple(inp[i] for i in perm)
        if arity == 1:
            parts.append(f"{'~' if name=='NOT' else ''}{bits[0]}={fn(bits)}")
        elif arity == 2 and name in two_ops:
            parts.append(f"{bits[0]}{two_ops[name]}{bits[1]}={fn(bits)}")
        else:
            parts.append(f"{name}{bits}={fn(bits)}")
    return " ".join(parts)


def build_cot(examples, target, answer, recipes):
    """Assemble the forum-style trace."""
    lines = []
    lines.append("I identify this as a binary boolean decomposition template. "
                 "Each output bit is an independent boolean function of the "
                 "input bits, so I solve all 8 bits independently and "
                 "concatenate. I ignore the flavor-text wrapper.")
    lines.append("")
    lines.append("COLUMNS")
    cols_in = ["".join(str(inp[b]) for inp, _ in examples) for b in range(8)]
    cols_out = ["".join(str(out[b]) for _, out in examples) for b in range(8)]
    lines.append("IN:  " + " ".join(f"i{b}={cols_in[b]}" for b in range(8)))
    lines.append("OUT: " + " ".join(f"o{b}={cols_out[b]}" for b in range(8)))
    lines.append(f"TGT: {''.join(str(b) for b in target)}")
    lines.append("")
    lines.append("SOLVE")
    inputs = [inp for inp, _ in examples]
    for b, rec in enumerate(recipes):
        if rec is None:
            lines.append(f"B{b}: UNKNOWN -> {answer[b]}")
            continue
        name, arity, fn, perm = rec
        gate_s = fmt_gate(name, arity, perm)
        col = cols_out[b]
        trace = serial_trace(name, arity, fn, perm, inputs)
        tgt_bits = tuple(target[i] for i in perm) if arity > 0 else ()
        ver = fn(tgt_bits)
        lines.append(f"B{b}: OUT={col} -> {gate_s}")
        lines.append(f"     {trace}")
        lines.append(f"     VER: {gate_s} on TGT -> {ver}")
    lines.append("")
    lines.append(f"ANS={answer}")
    lines.append("")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def process_problem(prompt, answer):
    parsed = parse_problem(prompt)
    if parsed is None:
        return None
    examples, target = parsed
    ans_tuple = tuple(int(c) for c in answer)
    # include (target, answer) as verification row so the scan must also
    # satisfy the target -- this locks the correct gate
    verified = list(examples) + [(target, ans_tuple)]
    recipes = []
    for b in range(8):
        rows = [(inp, out[b]) for inp, out in verified]
        recipes.append(find_gate(b, rows))
    cot = build_cot(examples, target, answer, recipes)
    return cot, recipes


def is_binary_prompt(p):
    return bool(BIN_RE.search(p)) and bool(TGT_RE.search(p))


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    df = pd.read_csv("data/train.csv", dtype={"answer": str})
    mask = df["prompt"].apply(is_binary_prompt)
    bdf = df[mask].reset_index(drop=True)
    print(f"Binary problems: {len(bdf)} | mode={mode}")
    if mode != "full":
        bdf = bdf.head(n)

    rows_out = []
    unknown = 0
    for i, row in bdf.iterrows():
        ans = str(row["answer"]).strip().zfill(8)
        res = process_problem(row["prompt"], ans)
        if res is None:
            continue
        cot, recipes = res
        if any(r is None for r in recipes):
            unknown += 1
        rows_out.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "answer": ans,
            "type": "Bit Manipulation",
            "generated_cot": cot,
        })
        if mode != "full" and i < 3:
            print(f"\n===== {row['id']} =====")
            print(cot)

    out = pd.DataFrame(rows_out)
    path = "data/train_binary_cot.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {len(out)} rows to {path}")
    print(f"Problems with at least one unresolved bit: {unknown}")


if __name__ == "__main__":
    main()
