#!/usr/bin/env python3
"""
Build data/train_split_with_cot_v5.csv from:
  - train_binary_cot.csv, train_cipher_cot.csv,
    train_equation_numeric_cot.csv, train_equation_symbolic_cot.csv
  - train_split_with_cot.csv rows: Gravitational Constant, Numeral Conversion, Unit Conversion

Run from repo root: python scripts/merge_train_split_cot_v5.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

COT_FOUR = [
    "train_binary_cot.csv",
    "train_cipher_cot.csv",
    "train_equation_numeric_cot.csv",
    "train_equation_symbolic_cot.csv",
]

SPLIT_TYPES = [
    "Gravitational Constant",
    "Numeral Conversion",
    "Unit Conversion",
]

OUT = "train_split_with_cot_v5.csv"


def main() -> None:
    parts: list[pd.DataFrame] = []
    for name in COT_FOUR:
        p = DATA / name
        df = pd.read_csv(p, dtype=str, low_memory=False)
        for c in ("id", "prompt", "answer", "type", "generated_cot"):
            if c not in df.columns:
                raise SystemExit(f"{name}: missing column {c}")
        parts.append(df)

    split = pd.read_csv(DATA / "train_split_with_cot.csv", dtype=str, low_memory=False)
    rest = split[split["type"].isin(SPLIT_TYPES)].copy()
    parts.append(rest)

    out = pd.concat(parts, ignore_index=True)
    dup = out["id"].duplicated(keep=False)
    if dup.any():
        dups = out.loc[dup, "id"].unique().tolist()
        raise SystemExit(f"Duplicate id(s) after merge (first 20): {dups[:20]}")

    out_path = DATA / OUT
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path.relative_to(REPO)}")
    print("\nPer type:")
    for t, n in out["type"].value_counts().sort_index().items():
        print(f"  {t!r}: {n}")


if __name__ == "__main__":
    main()
