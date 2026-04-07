"""
Generate CoT for Equation Symbolic (cipher-digit) using GOLD-locked oracle search.

Uses solve_oracle_detail: same traversal as solve_prompt(prompt, gold) so the trace
matches train labels when the solver can reproduce the answer.

Output: data/train_equation_symbolic_cot.csv
  columns: id, prompt, answer, type, generated_cot

Run from repo root:
  python scripts/EQUATION_SYMBOLIC/gen_equation_symbolic_cot.py
  python scripts/EQUATION_SYMBOLIC/gen_equation_symbolic_cot.py full
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_equation_symbolic import (  # noqa: E402
    OracleDetail,
    apply_combo,
    is_equation_symbolic_prompt,
    solve_oracle_detail,
)

OUTPUT_PATH = "data/train_equation_symbolic_cot.csv"
ROW_TYPE = "Equation Symbolic"

PREAMBLE = """I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a cipher-digit template. I know this because I see equations where ALL characters (including digits) have been replaced by random symbols. Each symbol maps to exactly one digit and vice versa (bijective). The cipher is unique to this problem and will never repeat. I know this is NOT roman, unit conversion, gravity, or bare symbol-digit (which uses real digits).

RULE 2: Since I have identified this is a cipher-digit template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \\boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: I see that this is a cipher-digit template. All characters are encrypted symbols. I need to first CRACK the cipher to recover the actual digits, then SCAN for the operation as a normal digit problem, then ENCODE my answer back to cipher symbols. I am now going to fill out the template."""


def build_cot(result: OracleDetail) -> str:
    combo, mapping, target, op_sym, op_examples, tgt_digits, enc_str, tL, tR, tV = result
    p, on, fn = combo
    lines = [PREAMBLE, ""]

    lines.append("S2: DETECT")
    lines.append(f"OP_SYM: {op_sym} (position 2)")
    lines.append(f"SYMS: {len(mapping)} unique digit symbols")
    lines.append("")

    lines.append("S3: CRACK")
    map_str = " ".join(f"{s}={d}" for s, d in sorted(mapping.items(), key=lambda x: x[1]))
    lines.append(f"MAP: {map_str}")
    s0, s1, s2, s3, rhs = op_examples[0]
    decoded_lhs = f"{mapping[s0]}{mapping[s1]}{op_sym}{mapping[s2]}{mapping[s3]}"
    decoded_rhs = "".join("-" if c == "-" else str(mapping[c]) for c in rhs)
    lines.append(f"CHK: {s0}{s1}{op_sym}{s2}{s3}={rhs} -> {decoded_lhs}={decoded_rhs}")
    lines.append("")

    lines.append("S4: SCAN")
    A1, B1, C1, D1 = mapping[s0], mapping[s1], mapping[s2], mapping[s3]
    out1, L1, R1, v1 = apply_combo(p, on, fn, A1, B1, C1, D1)
    lines.append(f"LOCK: {p}|{on}|{fn} L={L1},R={R1} {v1}->{out1} vs {decoded_rhs} YES")
    lines.append("")

    tA, tB, tC, tD = target[0], target[1], target[3], target[4]
    lines.append("S5: APPLY")
    lines.append(f"TGT: {target} -> DIG:{mapping[tA]}{mapping[tB]}{op_sym}{mapping[tC]}{mapping[tD]}")
    lines.append(f"{p}|L={tL},R={tR}|{on}={tV}|{fn}={tgt_digits}")
    lines.append("")

    lines.append("S6: ENCODE")
    lines.append(f"RES: {tgt_digits}")
    inv = {d: s for s, d in mapping.items()}
    enc_steps = " ".join(
        ("-->-" if dch == "-" else f"{dch}->{inv[int(dch)]}") for dch in tgt_digits
    )
    lines.append(f"ENC: {enc_steps}")
    lines.append(f"OUT: {enc_str}")
    lines.append("")

    lines.append(f"S7: ANS={enc_str}")
    lines.append("")
    lines.append(f"\\boxed{{{enc_str}}}")
    return "\n".join(lines)


def process_row(prompt: str, gold: str) -> str | None:
    res = solve_oracle_detail(prompt, gold)
    if res is None:
        return None
    _, _, _, _, _, _, enc, _, _, _ = res
    if enc != gold.strip():
        return None
    return build_cot(res)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    df = pd.read_csv("data/train.csv", dtype={"answer": str, "id": str})
    cdf = df[df["prompt"].apply(is_equation_symbolic_prompt)].reset_index(drop=True)
    print(f"Equation-symbolic rows: {len(cdf)} | mode={mode}")
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
            print(f"\n===== sample {row['id']} =====\n{cot[:1200]}...\n")

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH} (skipped {skipped} no oracle / no parse)")


if __name__ == "__main__":
    main()
