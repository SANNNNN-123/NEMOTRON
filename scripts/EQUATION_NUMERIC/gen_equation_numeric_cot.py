"""
Generate CoT for Equation Numeric (symbol-digit with real digits) using GOLD as oracle.

Why: greedy first-match (solve_prompt) often disagrees with train labels (~63%). CoT must
teach the *labeled* rule. find_combo_oracle scans (pairing|op|fmt) until ALL examples for
the target operator match AND the target output equals the gold answer — same idea as
gen_binary_cot using the answer as constraint.

Rows with no matching combo in FREQ_ORDER (fractions, odd symbols in answer, etc.) are skipped.

Output: data/train_equation_numeric_cot.csv
  columns: id, prompt, answer, type, generated_cot

Run from repo root:
  python scripts/EQUATION_NUMERIC/gen_equation_numeric_cot.py
  python scripts/EQUATION_NUMERIC/gen_equation_numeric_cot.py full
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_equation_numeric import (  # noqa: E402
    apply_combo,
    find_combo_oracle,
    parse_prompt,
    is_equation_numeric_prompt,
)

OUTPUT_PATH = "data/train_equation_numeric_cot.csv"
ROW_TYPE = "Equation Numeric"

INTRO = """I identify this as a symbol-digit (equation numeric) task: two-digit operands built from A,B,C,D, one operator, and a format. I group examples by the target operator, scan (pairing|op|format) until examples match and the target matches the boxed answer, then apply. I ignore flavor text."""


def build_cot(
    op_sym: str,
    op_examples,
    target,
    combo,
    attempts,
    tgt_out: str,
) -> str:
    tA, tB, _, tC, tD = target
    tgt_out2, tgt_L, tgt_R, tgt_v = apply_combo(*combo, tA, tB, tC, tD)
    assert tgt_out2 == tgt_out
    lines = [INTRO, ""]

    lines.append("S2: PARSE")
    lines.append(f"Operator: '{op_sym}'")
    lines.append(f"Target: {tA}{tB}{op_sym}{tC}{tD}  A={tA},B={tB},C={tC},D={tD}")
    lines.append(f"Examples using '{op_sym}':")
    for i, (A, B, C, D, res) in enumerate(op_examples, 1):
        lines.append(f"  E{i}: {A}{B}{op_sym}{C}{D} = {res}")
    lines.append("")

    lines.append("S3: SCAN")
    A1, B1, C1, D1, res1 = op_examples[0]
    shown = 0
    MAX_SHOW = 12
    for combo_t, first_eval, matched in attempts:
        if first_eval is None:
            continue
        p, op_name, fmt_name = combo_t
        L, R, val, out = first_eval
        tag = "YES" if matched else "NO"
        lines.append(f"#{shown + 1}:{p}|{op_name}|{fmt_name} L={L},R={R} {val}->{out} vs {res1} {tag}")
        shown += 1
        if matched:
            break
        if shown >= MAX_SHOW:
            break
    lines.append("")

    if len(op_examples) > 1:
        A2, B2, C2, D2, res2 = op_examples[1]
        p, op_name, fmt_name = combo
        out2, L2, R2, val2 = apply_combo(p, op_name, fmt_name, A2, B2, C2, D2)
        lines.append("S4: VER")
        lines.append(f"E2: {A2}{B2}{op_sym}{C2}{D2} = {res2}")
        lines.append(f"{p}|{op_name}|{fmt_name} L={L2},R={R2} {val2}->{out2} vs {res2} YES")
        lines.append("")

    p, op_name, fmt_name = combo
    lines.append(f"S5: LOCK: {p}|{op_name}|{fmt_name}")
    lines.append("")

    lines.append("S6: APPLY")
    lines.append(f"Target: {tA}{tB}{op_sym}{tC}{tD}")
    lines.append(f"{p}: L={tgt_L},R={tgt_R}")
    lines.append(f"{op_name}({tgt_L},{tgt_R})={tgt_v} {fmt_name}->{tgt_out}")
    lines.append("")

    lines.append(f"S7: ANS={tgt_out}")
    lines.append("")
    lines.append(f"\\boxed{{{tgt_out}}}")
    return "\n".join(lines)


def process_row(prompt: str, gold: str) -> str | None:
    """Return CoT string or None if parse fails or oracle finds no combo."""
    parsed = parse_prompt(prompt)
    if not parsed[0] or not parsed[1]:
        return None
    examples, target = parsed
    tA, tB, top, tC, tD = target
    op_examples = [(A, B, C, D, res) for A, B, sym, C, D, res in examples if sym == top]
    if not op_examples:
        return None
    combo, attempts = find_combo_oracle(op_examples, tA, tB, tC, tD, gold)
    if combo is None:
        return None
    p, on, fn = combo
    tgt_out, _, _, _ = apply_combo(p, on, fn, tA, tB, tC, tD)
    if tgt_out != gold.strip():
        return None
    return build_cot(top, op_examples, target, combo, attempts, tgt_out)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    df = pd.read_csv("data/train.csv", dtype={"answer": str, "id": str})
    cdf = df[df["prompt"].apply(is_equation_numeric_prompt)].reset_index(drop=True)
    print(f"Equation-numeric rows: {len(cdf)} | mode={mode}")
    if mode != "full":
        cdf = cdf.head(int(mode)).reset_index(drop=True)

    out_rows = []
    skipped = 0
    for i, row in cdf.iterrows():
        gold = str(row["answer"]).strip()
        cot = process_row(row["prompt"], gold)
        if cot is None:
            skipped += 1
            continue
        out_rows.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": gold,
                "type": ROW_TYPE,
                "generated_cot": cot,
            }
        )
        if i < 2:
            print(f"\n===== sample {row['id']} =====\n{cot[:900]}...\n")

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH} (skipped {skipped} no oracle / no parse)")


if __name__ == "__main__":
    main()
