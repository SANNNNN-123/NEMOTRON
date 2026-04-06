"""
Regenerate Equation Transformation (Symbol-Digit) CoT using DATA.md template.

Each puzzle has multiple example lines of form `ABopCD = result`, where the
operator character varies per line (each op stands for its own rule). The
target uses ONE of the demonstrated operators, so the solver must:

  1. Group example lines by operator character
  2. For the TARGET's operator, find a (pairing x op x format) combo that
     explains all of that operator's example lines
  3. Apply the combo to the target digits

Pipeline in the CoT:
  S1: intro
  S2: PARSE (target operator, collect its examples)
  S3: SCAN (brute-force combos in frequency order; each combo is checked
            against ALL of that operator's examples)
  S4: VER (re-verify on the non-first example if one exists)
  S5: LOCK
  S6: APPLY to target
  S7: ANS + \\boxed{}

Reads:  data/train_split_with_cot_v3.csv
Writes: data/train_split_with_cot_v3.csv  (overwrites, replaces Equation Transformation CoT)
"""
import pandas as pd
import re

INPUT_CSV = "data/train_split_with_cot_v3.csv"
OUTPUT_CSV = "data/train_split_with_cot_v3.csv"

RULE_PREAMBLE = """I am a reasoning model solving this puzzle strictly by following the template.

RULE 1: This is a symbol-digit template. I see equations with two-digit pairs separated by an operator symbol, and I need to figure out what transformation the operator performs. Each operator symbol has its own rule. I know this is NOT roman, unit conversion, gravity, binary, or cipher.

RULE 2: None of the flavor text (Alice in Wonderland, etc.) matters. The wrapper is here to trick me. I am only here to solve the problem.

RULE 3: Final answer in \\boxed{} at the end. Wrong format means zero points.

S1: This is a symbol-digit template. I need to identify the target's operator and figure out what transformation it performs by scanning common combinations. I am now going to fill out the template."""


# ---------------- Combo space ----------------

PAIRINGS = ["AB_CD", "BA_DC", "AB_DC", "BA_CD"]

def pair(pairing, A, B, C, D):
    if pairing == "AB_CD": return 10*A+B, 10*C+D
    if pairing == "BA_DC": return 10*B+A, 10*D+C
    if pairing == "AB_DC": return 10*A+B, 10*D+C
    if pairing == "BA_CD": return 10*B+A, 10*C+D
    raise ValueError(pairing)


OPS = [
    ("add",     lambda L, R: L + R),
    ("sub",     lambda L, R: L - R),
    ("mul",     lambda L, R: L * R),
    ("cat",     lambda L, R: int(f"{L}{R}") if L >= 0 and R >= 0 else None),
    ("add1",    lambda L, R: L + R + 1),
    ("addm1",   lambda L, R: L + R - 1),
    ("muladd1", lambda L, R: L * R + 1),
    ("mulsub1", lambda L, R: L * R - 1),
    ("absdiff", lambda L, R: abs(L - R)),
    ("xor",     lambda L, R: L ^ R),
    ("orsum",   lambda L, R: L | R),
    ("andsum",  lambda L, R: L & R),
    ("maxv",    lambda L, R: max(L, R)),
    ("minv",    lambda L, R: min(L, R)),
    ("div",     lambda L, R: L // R if R != 0 else None),
    ("mod",     lambda L, R: L % R if R != 0 else None),
]
OP_DICT = dict(OPS)


def _fmt_raw(n):
    if n is None:
        return None
    return str(n)

def _fmt_rev(n):
    if n is None:
        return None
    s = str(n)
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]

def _fmt_abs(n):
    if n is None:
        return None
    return str(abs(n))

def _fmt_neg(n):
    if n is None:
        return None
    return "-" + str(abs(n))

def _fmt_zpad(width):
    def f(n):
        if n is None or n < 0:
            return None
        return f"{n:0{width}d}"
    return f

def _fmt_dsum(n):
    if n is None:
        return None
    return str(sum(int(c) for c in str(abs(n))))

FORMATS = [
    ("raw",   _fmt_raw),
    ("rev",   _fmt_rev),
    ("abs",   _fmt_abs),
    ("neg",   _fmt_neg),
    ("zpad2", _fmt_zpad(2)),
    ("zpad3", _fmt_zpad(3)),
    ("zpad4", _fmt_zpad(4)),
    ("dsum",  _fmt_dsum),
]
FMT_DICT = dict(FORMATS)

# Frequency-ordered scan list
FREQ_ORDER = [
    ("AB_CD", "cat",     "raw"),
    ("AB_CD", "mul",     "raw"),
    ("AB_CD", "add",     "raw"),
    ("AB_CD", "sub",     "raw"),
    ("AB_CD", "absdiff", "raw"),
    ("BA_DC", "cat",     "rev"),
    ("BA_DC", "mul",     "rev"),
    ("BA_DC", "add",     "rev"),
    ("BA_DC", "cat",     "raw"),
    ("AB_CD", "cat",     "rev"),
    ("AB_CD", "mul",     "rev"),
    ("AB_CD", "add",     "rev"),
    ("AB_CD", "sub",     "neg"),
    ("AB_CD", "sub",     "rev"),
    ("BA_DC", "sub",     "rev"),
    ("AB_CD", "add1",    "raw"),
    ("AB_CD", "addm1",   "raw"),
    ("AB_CD", "muladd1", "raw"),
    ("AB_CD", "mulsub1", "raw"),
    ("AB_CD", "xor",     "raw"),
    ("AB_CD", "mod",     "raw"),
    ("AB_CD", "div",     "raw"),
    ("AB_CD", "maxv",    "raw"),
    ("AB_CD", "minv",    "raw"),
]
# Append all remaining combos for exhaustive fallback
_SEEN = set(FREQ_ORDER)
for p in PAIRINGS:
    for op_name, _ in OPS:
        for fmt_name, _ in FORMATS:
            t = (p, op_name, fmt_name)
            if t not in _SEEN:
                FREQ_ORDER.append(t)
                _SEEN.add(t)


def apply_combo(pairing, op_name, fmt_name, A, B, C, D):
    L, R = pair(pairing, A, B, C, D)
    val = OP_DICT[op_name](L, R)
    if val is None:
        return None, L, R, None
    out = FMT_DICT[fmt_name](val)
    return out, L, R, val


# ---------------- Parse ----------------

def parse_example_line(line):
    """Parse 'AB{op}CD = result' → (A,B,op_sym,C,D,result_str) or None."""
    m = re.match(r'^\s*(\d)(\d)(\D)(\d)(\d)\s*=\s*(\S+)\s*$', line)
    if not m:
        return None
    A, B, op_sym, C, D, res = m.groups()
    return int(A), int(B), op_sym, int(C), int(D), res

def parse_prompt(prompt):
    """
    Returns (examples: list of tuples, target: (A,B,op_sym,C,D)) or (None, None).
    """
    lines = prompt.split("\n")
    examples = []
    for line in lines:
        # skip the "determine the result for: X" line (it has no '=')
        parsed = parse_example_line(line)
        if parsed:
            examples.append(parsed)
    # target
    m = re.search(r'result for[:\s]+(\d)(\d)(\D)(\d)(\d)', prompt)
    if not m:
        return None, None
    tA, tB, top, tC, tD = m.groups()
    target = (int(tA), int(tB), top, int(tC), int(tD))
    if not examples:
        return None, None
    return examples, target


# ---------------- Solver ----------------

def find_combo_for_op(op_examples, target_result_hint=None):
    """
    op_examples: list of (A,B,C,D,result_str) — all examples using the same operator.
    Returns (combo, attempts_trace) where combo is (pairing, op, fmt) or None.
    attempts_trace is a list of (combo, first_example_eval_tuple, matched_all) for S3.
    """
    attempts = []
    for combo in FREQ_ORDER:
        pairing, op_name, fmt_name = combo
        # check ALL examples
        all_match = True
        first_eval = None
        for i, (A, B, C, D, res) in enumerate(op_examples):
            out, L, R, val = apply_combo(pairing, op_name, fmt_name, A, B, C, D)
            if i == 0:
                first_eval = (L, R, val, out)
            if out is None or out != res:
                all_match = False
                break
        attempts.append((combo, first_eval, all_match))
        if all_match:
            return combo, attempts
    return None, attempts


# ---------------- CoT builder ----------------

def build_cot(examples, target, combo, attempts, tgt_out, tgt_L, tgt_R, tgt_val, op_sym, op_examples):
    tA, tB, _, tC, tD = target
    lines = [RULE_PREAMBLE, ""]

    # S2: PARSE
    lines.append("S2: PARSE")
    lines.append(f"Target: {tA}{tB}{op_sym}{tC}{tD}  A={tA},B={tB},C={tC},D={tD}")
    lines.append(f"Operator: '{op_sym}'")
    lines.append(f"Examples using '{op_sym}':")
    for i, (A, B, C, D, res) in enumerate(op_examples, 1):
        lines.append(f"  E{i}: {A}{B}{op_sym}{C}{D} = {res}")
    lines.append("")

    # S3: SCAN — show attempts against E1 until match
    lines.append("S3: SCAN")
    A1, B1, C1, D1, res1 = op_examples[0]
    shown = 0
    MAX_SHOW = 10
    for combo_t, first_eval, matched in attempts:
        p, op_name, fmt_name = combo_t
        if first_eval is None:
            continue
        L, R, val, out = first_eval
        tag = "YES" if matched else "NO"
        lines.append(f"#{shown+1}:{p}|{op_name}|{fmt_name} L={L},R={R} {val}->{out} vs {res1} {tag}")
        shown += 1
        if matched:
            break
        if shown >= MAX_SHOW:
            break
    lines.append("")

    # S4: VER — if more than 1 example, re-verify on E2
    if len(op_examples) > 1:
        A2, B2, C2, D2, res2 = op_examples[1]
        p, op_name, fmt_name = combo
        out2, L2, R2, val2 = apply_combo(p, op_name, fmt_name, A2, B2, C2, D2)
        lines.append("S4: VER")
        lines.append(f"E2: {A2}{B2}{op_sym}{C2}{D2} = {res2}")
        lines.append(f"{p}|{op_name}|{fmt_name} L={L2},R={R2} {val2}->{out2} vs {res2} YES")
        lines.append("")

    # S5: LOCK
    p, op_name, fmt_name = combo
    lines.append(f"S5: LOCK: {p}|{op_name}|{fmt_name}")
    lines.append("")

    # S6: APPLY
    lines.append("S6: APPLY")
    lines.append(f"Target: {tA}{tB}{op_sym}{tC}{tD}  A={tA},B={tB},C={tC},D={tD}")
    lines.append(f"{p}: L={tgt_L},R={tgt_R}")
    lines.append(f"{op_name}({tgt_L},{tgt_R})={tgt_val}  fmt:{fmt_name}->{tgt_out}")
    lines.append("")

    # S7: ANS
    lines.append(f"S7: ANS={tgt_out}")
    lines.append("")
    lines.append(f"\\boxed{{{tgt_out}}}")
    return "\n".join(lines)


# ---------------- Main ----------------

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {INPUT_CSV}: {len(df)} rows")

    eq_mask = df['type'] == 'Equation Transformation'
    eq_df = df[eq_mask]
    print(f"Equation Transformation rows: {len(eq_df)}")

    results = []
    solved = 0
    parse_fail = 0
    no_combo = 0
    mismatch = 0

    for idx, row in eq_df.iterrows():
        examples, target = parse_prompt(row['prompt'])
        if examples is None:
            results.append((idx, None, "parse_fail"))
            parse_fail += 1
            continue

        tA, tB, top, tC, tD = target
        # collect only examples using the target's operator
        op_examples = [(A, B, C, D, res) for (A, B, sym, C, D, res) in examples if sym == top]
        if not op_examples:
            results.append((idx, None, "no_op_examples"))
            parse_fail += 1
            continue

        combo, attempts = find_combo_for_op(op_examples)
        if combo is None:
            results.append((idx, None, "no_combo"))
            no_combo += 1
            continue

        # apply to target
        p, op_name, fmt_name = combo
        tgt_out, tgt_L, tgt_R, tgt_val = apply_combo(p, op_name, fmt_name, tA, tB, tC, tD)
        if tgt_out is None:
            results.append((idx, None, "apply_fail"))
            no_combo += 1
            continue

        gt = str(row['answer']).strip()
        if tgt_out != gt:
            results.append((idx, None, f"mismatch:{tgt_out}!={gt}"))
            mismatch += 1
            continue

        cot = build_cot(examples, target, combo, attempts, tgt_out, tgt_L, tgt_R, tgt_val, top, op_examples)
        results.append((idx, cot, "ok"))
        solved += 1

    print(f"\nSolved: {solved}/{len(eq_df)} ({100*solved/len(eq_df):.1f}%)")
    print(f"Parse fail: {parse_fail}")
    print(f"No combo found: {no_combo}")
    print(f"Answer mismatch: {mismatch}")

    df_new = df.copy()
    keep_mask = pd.Series([True] * len(df), index=df.index)
    for idx, cot, _ in results:
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
