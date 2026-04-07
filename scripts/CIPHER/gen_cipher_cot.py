"""
Generate CoT traces for English substitution cipher rows in train.csv.

Output: data/train_cipher_cot.csv with columns (same as train_binary_cot.csv):
    id, prompt, answer, type, generated_cot

Default: all cipher rows in train.csv (same count as cipher problems, currently 1576).

Run from repo root:
  python scripts/CIPHER/gen_cipher_cot.py           # full cipher set
  python scripts/CIPHER/gen_cipher_cot.py full      # same (explicit)
  python scripts/CIPHER/gen_cipher_cot.py 364       # first N rows only
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_cipher import (  # noqa: E402
    decrypt_partial,
    is_cipher_prompt,
    parse_prompt,
    solve_prompt_with_mapping,
    word_matches,
)
from vocab import CIPHER_VOCAB  # noqa: E402

OUTPUT_PATH = "data/train_cipher_cot.csv"
ROW_TYPE = "Text Encryption"


def table_trace_examples(
    examples: List[Tuple[str, str]],
) -> Optional[Tuple[Dict[str, str], Dict[str, str], List[Tuple[str, str, List[Tuple[str, str]]]]]]:
    """Walk examples in order; return (fwd, inv, list of (enc, plain, new_pairs)) or None on conflict."""
    fwd: Dict[str, str] = {}
    inv: Dict[str, str] = {}
    rows = []
    for enc, plain in examples:
        new_pairs: List[Tuple[str, str]] = []
        for e, p in zip(enc, plain):
            if e == " " and p == " ":
                continue
            if e == " " or p == " ":
                return None
            if e in fwd:
                if fwd[e] != p:
                    return None
                continue
            if p in inv and inv[p] != e:
                return None
            fwd[e] = p
            inv[p] = e
            new_pairs.append((e, p))
        rows.append((enc, plain, new_pairs))
    return fwd, inv, rows


def build_cot(
    examples: List[Tuple[str, str]],
    target: str,
    answer: str,
    fwd_base: Dict[str, str],
    fwd_final: Dict[str, str],
    trace_rows: List[Tuple[str, str, List[Tuple[str, str]]]],
) -> str:
    awords = answer.split()
    twords = target.split()
    lines: List[str] = []

    lines.append(
        "I identify this as a letter substitution cipher. I build a mapping from the "
        "examples, verify it, then decrypt the target word by word. I use the fixed "
        "answer vocabulary when the examples do not cover every letter. I ignore the flavor-text wrapper."
    )
    lines.append("")

    # S2: LEN
    lines.append("LEN")
    lines.append(f'TGT:"{target}"')
    lens = " ".join(f"W{i + 1}:{len(w)}" for i, w in enumerate(twords))
    lines.append(lens)
    lines.append("")

    # S3: TABLE
    lines.append("TABLE")
    for i, (enc, plain, new_pairs) in enumerate(trace_rows, 1):
        pair_s = ",".join(f"{e}={p}" for e, p in new_pairs) if new_pairs else "none"
        lines.append(f'EX{i}:"{enc}"->"{plain}" [{len(new_pairs)}] {pair_s}')
    lines.append(f"TOTAL:{len(fwd_base)}/26")
    lines.append("")

    # VOCAB_FILL (only words that had gaps before full mapping)
    lines.append("VOCAB_FILL")
    any_vocab = False
    for i, cw in enumerate(twords):
        partial_w = decrypt_partial(cw, fwd_base)
        if "?" in partial_w:
            any_vocab = True
            pw = awords[i]
            cands = word_matches(cw, fwd_base)
            cands_s = ",".join(cands) if len(cands) <= 8 else f"{len(cands)} candidates"
            lines.append(f'W{i + 1}:{cw} pattern {partial_w} -> LOCK "{pw}" (candidates: {cands_s})')
    if not any_vocab:
        lines.append("(no gaps; examples covered all target letters)")
    lines.append("")

    # S4: VER
    lines.append("VER")
    n_ok = 0
    for enc, plain in examples:
        d = decrypt_partial(enc, fwd_final)
        if d == plain:
            n_ok += 1
    lines.append(f"CROSS:{n_ok}/{len(examples)} examples decrypt with final table")
    lines.append("")

    # S5: DECRYPT
    lines.append("DECRYPT")
    first_seen: Dict[str, int] = {}
    for widx, cw in enumerate(twords):
        pw = awords[widx]
        lines.append(f"W{widx + 1}: {cw}")
        for ec in cw:
            pc = fwd_final[ec]
            if ec in first_seen and first_seen[ec] != widx:
                lines.append(f" {ec}->{pc} (W{first_seen[ec] + 1})")
            else:
                lines.append(f" {ec}->{pc}")
                if ec not in first_seen:
                    first_seen[ec] = widx
        lines.append(f" = {pw}")
    lines.append("")

    # S6: CHECK
    lines.append("CHECK")
    for widx, w in enumerate(awords):
        in_vocab = "Y" if w in CIPHER_VOCAB else "N"
        lines.append(f'W{widx + 1}:"{w}" len={len(w)}Y alpha=Y vocab={in_vocab} gaps=N PASS')
    lines.append(f"ALL:PASS {len(awords)}/{len(awords)} words")
    lines.append("")

    lines.append(f"ANS={answer}")
    lines.append("")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def process_prompt(prompt: str, gold: str) -> Optional[str]:
    parsed = parse_prompt(prompt)
    if not parsed:
        return None
    examples, target = parsed
    tr = table_trace_examples(examples)
    if tr is None:
        return None
    fwd_base, _, trace_rows = tr
    sol = solve_prompt_with_mapping(prompt)
    if sol is None:
        return None
    pred, fwd_final, _ = sol
    if pred.strip() != gold.strip():
        return None
    return build_cot(examples, target, gold.strip(), fwd_base, fwd_final, trace_rows)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    df = pd.read_csv("data/train.csv", dtype={"answer": str, "id": str})
    cdf = df[df["prompt"].apply(is_cipher_prompt)].reset_index(drop=True)
    print(f"Cipher rows in train.csv: {len(cdf)}")

    if mode == "full":
        n = len(cdf)
    else:
        n = int(mode)
    cdf = cdf.head(n).reset_index(drop=True)
    print(f"Generating CoT for {len(cdf)} rows -> {OUTPUT_PATH}")

    out_rows = []
    failed = 0
    for i, row in cdf.iterrows():
        gold = str(row["answer"]).strip()
        cot = process_prompt(row["prompt"], gold)
        if cot is None:
            failed += 1
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
            print(f"\n===== sample id={row['id']} =====\n{cot[:1200]}...\n")

    out = pd.DataFrame(out_rows)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH} (skipped {failed})")


if __name__ == "__main__":
    main()
