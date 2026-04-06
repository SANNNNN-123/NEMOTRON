"""
Regenerate Bit Manipulation CoT using the canonical DATA.md template.

Format matches the BINARY prompt/completion example exactly:
- RULE 1/2/3/4 preamble (identity anchor + bit-serial rule)
- S1: intro
- S2: COLUMNS (IN/OUT/TGT)
- S3: SOLVE (per bit, showing search trace with real operators & bit-serial spell-out)
- S4: ANS + \\boxed{}

Per-bit search order (first match wins):
  1. Constants C0/C1
  2. Identity id(i) — shown inline with Y/N markers across i-columns
  3. NOT(i) — shown inline with Y/N markers
  4. 2-input gates (AND, OR, XOR) and negation variants — with bit-serial computation
  5. 3-input gates: MAJ, PAR3, CHO
"""
import pandas as pd
import re
from itertools import combinations

INPUT_CSV = "data/train_split_with_cot.csv"
OUTPUT_CSV = "data/train_split_with_cot_v2.csv"

RULE_PREAMBLE = """I am a reasoning model solving this puzzle strictly by following the template.

RULE 1: This is a binary boolean decomposition template. I see 8-bit binary strings mapped to 8-bit binary strings. Each output bit is an independent boolean function of the input bits. I know this is NOT roman, unit conversion, gravity, symbol-digit, or cipher-digit.

RULE 2: None of the flavor text (Alice in Wonderland, etc.) matters. The wrapper is here to trick me. I am only here to solve the problem.

RULE 3: Final answer in \\boxed{} at the end. Wrong format means zero points.

RULE 4: For gate computations, I MUST spell out each bit operation individually. I CANNOT compute multi-bit results in parallel. AND(0,1): 1&0=0 0&1=0 1&1=1 -> one bit at a time.

S1: This is a binary boolean decomposition template. Each output bit is an independent boolean function. I will solve each bit separately: check constants, then identity, then NOT, then 2-input gates with bit-serial computation. I am now going to fill out the template."""


def int_to_bits(b_int, n=8):
    return tuple((b_int >> (n - 1 - i)) & 1 for i in range(n))


# ---------------- Boolean function library ----------------

# 2-input gate definitions with their operator symbol and function
GATES_2 = [
    ("AND", "&", lambda a, b: a & b),
    ("OR",  "|", lambda a, b: a | b),
    ("XOR", "^", lambda a, b: a ^ b),
]

# ---------------- Columns ----------------

def build_columns(examples):
    """examples: list of (in_bits, out_bits). Returns (i_cols, o_cols) as list of 8 strings."""
    i_cols = ["".join(str(ex[0][b]) for ex in examples) for b in range(8)]
    o_cols = ["".join(str(ex[1][b]) for ex in examples) for b in range(8)]
    return i_cols, o_cols


def invert(col):
    return "".join("1" if c == "0" else "0" for c in col)


def apply_gate_to_col(col_i, col_j, gate_fn):
    return "".join(str(gate_fn(int(a), int(b))) for a, b in zip(col_i, col_j))


def serial_str(col_i, col_j, op_sym, gate_fn):
    """Returns spelled-out bit-serial computation like '1&0=0 0&1=0 1&1=1'."""
    return " ".join(f"{a}{op_sym}{b}={gate_fn(int(a), int(b))}" for a, b in zip(col_i, col_j))


# ---------------- Per-bit solver ----------------

def solve_bit(bit_pos, i_cols, o_cols, target_bits):
    """
    Search in canonical order. Returns a dict with:
      - 'trace': list of strings (the per-bit trace lines for S3)
      - 'answer_bit': the bit value for the target
    Returns None if no function found.
    """
    out_col = o_cols[bit_pos]
    lines = []

    # 1. Constants
    if out_col == "0" * len(out_col):
        lines.append(f"B{bit_pos}: OUT={out_col} -> C0 VER:0")
        return {"trace": lines, "answer_bit": 0}
    if out_col == "1" * len(out_col):
        lines.append(f"B{bit_pos}: OUT={out_col} -> C1 VER:1")
        return {"trace": lines, "answer_bit": 1}

    # 2. Identity (inline Y/N scan)
    id_markers = []
    id_hit = None
    for i in range(8):
        marker = "Y" if i_cols[i] == out_col else "N"
        id_markers.append(f"i{i}={i_cols[i]}{marker}")
        if marker == "Y" and id_hit is None:
            id_hit = i
    if id_hit is not None:
        prefix = f"B{bit_pos}: OUT={out_col} " + " ".join(id_markers)
        t_val = target_bits[id_hit]
        lines.append(f"{prefix} -> id({id_hit}) VER:t{id_hit}={t_val}->{t_val}")
        return {"trace": lines, "answer_bit": t_val}

    # 3. NOT (inline scan)
    not_markers = []
    not_hit = None
    for i in range(8):
        inv = invert(i_cols[i])
        marker = "Y" if inv == out_col else "N"
        not_markers.append(f"~{i}={inv}{marker}")
        if marker == "Y" and not_hit is None:
            not_hit = i
    if not_hit is not None:
        prefix = f"B{bit_pos}: OUT={out_col} id:allN " + " ".join(not_markers)
        t_in = target_bits[not_hit]
        t_val = 1 - t_in
        lines.append(f"{prefix} -> NOT({not_hit}) VER:~t{not_hit}=~{t_in}={t_val}->{t_val}")
        return {"trace": lines, "answer_bit": t_val}

    # 4. 2-input gates — show search with bit-serial spell-out
    lines.append(f"B{bit_pos}: OUT={out_col} id:N ~:N")
    MAX_FAILED_ATTEMPTS = 4  # don't spam too many rejected tries
    attempts_shown = 0
    for i, j in combinations(range(8), 2):
        for gname, op_sym, gfn in GATES_2:
            result = apply_gate_to_col(i_cols[i], i_cols[j], gfn)
            serial = serial_str(i_cols[i], i_cols[j], op_sym, gfn)
            if result == out_col:
                lines.append(f"  {gname}({i},{j}): {serial} ={result} vs {out_col} YES -> {gname}({i},{j})")
                t_i, t_j = target_bits[i], target_bits[j]
                t_val = gfn(t_i, t_j)
                lines.append(f"  VER:{t_i}{op_sym}{t_j}={t_val}->{t_val}")
                return {"trace": lines, "answer_bit": t_val}
            else:
                if attempts_shown < MAX_FAILED_ATTEMPTS:
                    lines.append(f"  {gname}({i},{j}): {serial} ={result} vs {out_col} NO")
                    attempts_shown += 1

    # 4b. 2-input with negation — AND(~a,b), AND(a,~b), OR(~a,b), OR(a,~b), XOR(~a,b) == XNOR
    for i, j in combinations(range(8), 2):
        # AND(~i,j)
        result = apply_gate_to_col(invert(i_cols[i]), i_cols[j], lambda a, b: a & b)
        if result == out_col:
            serial = " ".join(f"~{a}&{b}={(1-int(a))&int(b)}" for a, b in zip(i_cols[i], i_cols[j]))
            lines.append(f"  AND(~{i},{j}): {serial} ={result} vs {out_col} YES -> AND(~{i},{j})")
            t_i, t_j = target_bits[i], target_bits[j]
            t_val = (1 - t_i) & t_j
            lines.append(f"  VER:~{t_i}&{t_j}={t_val}->{t_val}")
            return {"trace": lines, "answer_bit": t_val}
        # AND(i,~j)
        result = apply_gate_to_col(i_cols[i], invert(i_cols[j]), lambda a, b: a & b)
        if result == out_col:
            serial = " ".join(f"{a}&~{b}={int(a)&(1-int(b))}" for a, b in zip(i_cols[i], i_cols[j]))
            lines.append(f"  AND({i},~{j}): {serial} ={result} vs {out_col} YES -> AND({i},~{j})")
            t_i, t_j = target_bits[i], target_bits[j]
            t_val = t_i & (1 - t_j)
            lines.append(f"  VER:{t_i}&~{t_j}={t_val}->{t_val}")
            return {"trace": lines, "answer_bit": t_val}
        # OR(~i,j)
        result = apply_gate_to_col(invert(i_cols[i]), i_cols[j], lambda a, b: a | b)
        if result == out_col:
            serial = " ".join(f"~{a}|{b}={(1-int(a))|int(b)}" for a, b in zip(i_cols[i], i_cols[j]))
            lines.append(f"  OR(~{i},{j}): {serial} ={result} vs {out_col} YES -> OR(~{i},{j})")
            t_i, t_j = target_bits[i], target_bits[j]
            t_val = (1 - t_i) | t_j
            lines.append(f"  VER:~{t_i}|{t_j}={t_val}->{t_val}")
            return {"trace": lines, "answer_bit": t_val}
        # OR(i,~j)
        result = apply_gate_to_col(i_cols[i], invert(i_cols[j]), lambda a, b: a | b)
        if result == out_col:
            serial = " ".join(f"{a}|~{b}={int(a)|(1-int(b))}" for a, b in zip(i_cols[i], i_cols[j]))
            lines.append(f"  OR({i},~{j}): {serial} ={result} vs {out_col} YES -> OR({i},~{j})")
            t_i, t_j = target_bits[i], target_bits[j]
            t_val = t_i | (1 - t_j)
            lines.append(f"  VER:{t_i}|~{t_j}={t_val}->{t_val}")
            return {"trace": lines, "answer_bit": t_val}

    # 5. 3-input gates (no spell-out for brevity, just lock-in)
    for i, j, k in combinations(range(8), 3):
        # MAJ
        maj_col = "".join("1" if (int(a) + int(b) + int(c)) >= 2 else "0"
                          for a, b, c in zip(i_cols[i], i_cols[j], i_cols[k]))
        if maj_col == out_col:
            t_i, t_j, t_k = target_bits[i], target_bits[j], target_bits[k]
            t_val = 1 if (t_i + t_j + t_k) >= 2 else 0
            lines.append(f"  MAJ({i},{j},{k}): ={maj_col} vs {out_col} YES -> MAJ({i},{j},{k})")
            lines.append(f"  VER:MAJ({t_i},{t_j},{t_k})={t_val}->{t_val}")
            return {"trace": lines, "answer_bit": t_val}
        # PAR3 (XOR of three)
        par_col = "".join(str(int(a) ^ int(b) ^ int(c)) for a, b, c in zip(i_cols[i], i_cols[j], i_cols[k]))
        if par_col == out_col:
            t_i, t_j, t_k = target_bits[i], target_bits[j], target_bits[k]
            t_val = t_i ^ t_j ^ t_k
            lines.append(f"  PAR3({i},{j},{k}): ={par_col} vs {out_col} YES -> PAR3({i},{j},{k})")
            lines.append(f"  VER:{t_i}^{t_j}^{t_k}={t_val}->{t_val}")
            return {"trace": lines, "answer_bit": t_val}
        # CHO (mux: if i then j else k)
        cho_col = "".join(b if a == "1" else c for a, b, c in zip(i_cols[i], i_cols[j], i_cols[k]))
        if cho_col == out_col:
            t_i, t_j, t_k = target_bits[i], target_bits[j], target_bits[k]
            t_val = t_j if t_i else t_k
            lines.append(f"  CHO({i},{j},{k}): ={cho_col} vs {out_col} YES -> CHO({i},{j},{k})")
            lines.append(f"  VER:CHO({t_i},{t_j},{t_k})={t_val}->{t_val}")
            return {"trace": lines, "answer_bit": t_val}

    return None


# ---------------- Full CoT builder ----------------

def build_cot(examples, target_str, target_bits):
    i_cols, o_cols = build_columns(examples)

    # Solve each bit
    per_bit = []
    for b in range(8):
        sol = solve_bit(b, i_cols, o_cols, target_bits)
        if sol is None:
            return None, None
        per_bit.append(sol)

    answer_str = "".join(str(s["answer_bit"]) for s in per_bit)

    lines = [RULE_PREAMBLE, ""]
    lines.append("S2: COLUMNS")
    lines.append("IN: " + " ".join(f"i{b}={i_cols[b]}" for b in range(8)))
    lines.append("OUT: " + " ".join(f"o{b}={o_cols[b]}" for b in range(8)))
    lines.append(f"TGT: {target_str}")
    lines.append("")
    lines.append("S3: SOLVE")
    for sol in per_bit:
        for line in sol["trace"]:
            lines.append(line)
    lines.append("")
    lines.append(f"S4: ANS={answer_str}")
    lines.append("")
    lines.append(f"\\boxed{{{answer_str}}}")
    return "\n".join(lines), answer_str


# ---------------- Parse prompt ----------------

def parse_prompt(prompt):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    m = re.search(r'output for[:\s]+([01]{8})', prompt)
    if not m:
        m = re.search(r'([01]{8})\s*$', prompt.strip())
    if not m:
        return None, None
    target = m.group(1)
    examples = [(tuple(int(c) for c in a), tuple(int(c) for c in b)) for a, b in pairs]
    target_bits = tuple(int(c) for c in target)
    return examples, (target, target_bits)


# ---------------- Main ----------------

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")

    bit_mask = df['type'] == 'Bit Manipulation'
    bit_df = df[bit_mask].copy()
    print(f"Bit Manipulation rows: {len(bit_df)}")

    results = []
    solved = 0
    failed = 0
    mismatch = 0

    for idx, row in bit_df.iterrows():
        examples, target = parse_prompt(row['prompt'])
        if examples is None or len(examples) < 4:
            results.append((idx, None))
            failed += 1
            continue
        target_str, target_bits = target
        cot, answer_str = build_cot(examples, target_str, target_bits)
        if cot is None:
            results.append((idx, None))
            failed += 1
            continue
        gt = str(row['answer']).strip()
        if answer_str != gt:
            results.append((idx, None))
            mismatch += 1
            continue
        results.append((idx, cot))
        solved += 1

    print(f"\nSolved: {solved}/{len(bit_df)} ({100*solved/len(bit_df):.1f}%)")
    print(f"Failed (no function): {failed}")
    print(f"Answer mismatch: {mismatch}")

    # Build new dataframe
    df_new = df.copy()
    keep_mask = pd.Series([True] * len(df), index=df.index)
    for idx, cot in results:
        if cot is not None:
            df_new.at[idx, 'generated_cot'] = cot
        else:
            keep_mask.at[idx] = False

    df_new = df_new[keep_mask].reset_index(drop=True)
    print(f"\nNew dataset: {len(df_new)} rows")
    print(df_new['type'].value_counts().sort_index())

    df_new.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
