"""
Solve the BINARY category from data/train.csv.

Approach (per the DATA.md writeup):
  - Each output bit is an independent boolean function of the 8 input bits.
  - For each bit position, scan a library of candidate functions in order of
    complexity (constants -> identity -> NOT -> 2-input -> 3-input -> 4-input -> 5-input).
  - A candidate must match ALL examples for that bit position. First match wins.
  - Apply the locked function to the target input to produce the answer bit.

Expected: ~1602 binary problems, ~100% solvable (one historical outlier).
"""

import re
import time
import itertools
import pandas as pd

BIN_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
TGT_RE = re.compile(r"determine the output for:\s*([01]{8})")


# ---------- boolean function library ----------
# Each entry: (name, arity, fn). fn takes a tuple of bits (ints 0/1) and returns 0/1.

def _build_library():
    """Return list of (name, arity, fn, level)."""
    lib = []

    # LEVEL 0
    lib.append(("CONST0", 0, lambda b: 0, 0))
    lib.append(("CONST1", 0, lambda b: 1, 0))

    # LEVEL 1
    lib.append(("ID",  1, lambda b: b[0], 1))
    lib.append(("NOT", 1, lambda b: 1 - b[0], 1))

    # 2-input: only the 6 symmetric gates used by the dataset generator
    two_in = {
        "AND":  lambda a, b: a & b,
        "OR":   lambda a, b: a | b,
        "XOR":  lambda a, b: a ^ b,
        "XNOR": lambda a, b: 1 - (a ^ b),
        "NAND": lambda a, b: 1 - (a & b),
        "NOR":  lambda a, b: 1 - (a | b),
    }
    for name, fn in two_in.items():
        lib.append((name, 2, (lambda fn: lambda b: fn(b[0], b[1]))(fn), 2))

    # 3-input: only CHOICE, MAJ3, PAR3 as reported in the distribution
    three_in = {
        "CHOICE": lambda a, b, c: b if a else c,
        "MAJ3":   lambda a, b, c: 1 if (a + b + c) >= 2 else 0,
        "PAR3":   lambda a, b, c: a ^ b ^ c,
    }
    for name, fn in three_in.items():
        lib.append((name, 3, (lambda fn: lambda b: fn(b[0], b[1], b[2]))(fn), 3))

    # 4-input: not used by the generator

    return lib


LIBRARY = _build_library()


def parse_problem(prompt: str):
    pairs = BIN_RE.findall(prompt)
    m = TGT_RE.search(prompt)
    if not m or not pairs:
        return None
    examples = [(tuple(int(c) for c in inp), tuple(int(c) for c in out))
                for inp, out in pairs]
    target = tuple(int(c) for c in m.group(1))
    return examples, target


def solve_bit(bit_idx: int, examples, target):
    """Hybrid: first-match library scan, then truth-table subset fallback."""
    wanted = [out[bit_idx] for _, out in examples]
    inps = [inp for inp, _ in examples]

    for name, arity, fn, level in LIBRARY:
        if arity == 0:
            if all(fn(()) == w for w in wanted):
                return str(fn(()))
            continue
        for perm in itertools.permutations(range(8), arity):
            ok = True
            for inp, w in zip(inps, wanted):
                if fn(tuple(inp[i] for i in perm)) != w:
                    ok = False
                    break
            if ok:
                return str(fn(tuple(target[i] for i in perm)))

    for k in range(1, 6):
        for subset in itertools.combinations(range(8), k):
            table = {}
            conflict = False
            for inp, w in zip(inps, wanted):
                key = tuple(inp[i] for i in subset)
                if key in table:
                    if table[key] != w:
                        conflict = True
                        break
                else:
                    table[key] = w
            if conflict:
                continue
            tgt_key = tuple(target[i] for i in subset)
            if tgt_key in table:
                return str(table[tgt_key])
    return "?"


def solve_problem(prompt: str):
    parsed = parse_problem(prompt)
    if parsed is None:
        return None, None
    examples, target = parsed
    out_bits = [solve_bit(b, examples, target) for b in range(8)]
    return "".join(out_bits), None


def is_binary_prompt(prompt: str) -> bool:
    return bool(BIN_RE.search(prompt)) and bool(TGT_RE.search(prompt))


def main():
    import sys
    # Usage: python solve_binary.py            -> sample run (200)
    #        python solve_binary.py full       -> full 1602 run
    #        python solve_binary.py sample N   -> sample of N
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    df = pd.read_csv("data/train.csv")
    mask = df["prompt"].apply(is_binary_prompt)
    bdf = df[mask].reset_index(drop=True)
    print(f"Binary problems total: {len(bdf)} | mode={mode}")
    if mode != "full":
        bdf = bdf.head(sample_size).reset_index(drop=True)
        print(f"Sample run: {len(bdf)} problems")

    t0 = time.time()
    correct = 0
    unsolved_bits = 0
    failures = []
    for i, row in bdf.iterrows():
        pred, _ = solve_problem(row["prompt"])
        if pred is None:
            continue
        unsolved_bits += pred.count("?")
        gold = str(row["answer"]).strip()
        ok = pred == gold
        if ok:
            correct += 1
        else:
            failures.append((row["id"], pred, gold))
        mark = "OK " if ok else "XX "
        print(f"[{i+1}/{len(bdf)}] {mark} id={row['id']} pred={pred} gold={gold}", flush=True)

    dt = time.time() - t0
    print("\n====== RESULTS ======")
    print(f"Total: {len(bdf)}")
    print(f"Correct: {correct} ({correct/len(bdf)*100:.2f}%)")
    print(f"Unsolved bit positions: {unsolved_bits}")
    print(f"Time: {dt:.1f}s")

    if failures:
        print(f"\nFailures: {len(failures)} (showing up to 5)")
        for fid, pred, gold in failures[:5]:
            print(f"  {fid}: pred={pred} gold={gold}")


if __name__ == "__main__":
    main()
