#!/usr/bin/env python3
"""Build data/viz_manifest.json for visualization.html (slim first-load payload)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = ap.parse_args()
    root: Path = args.repo_root
    train_path = root / "data" / "train.csv"
    prob_path = root / "data" / "problems.jsonl"
    out_path = root / "data" / "viz_manifest.json"

    cat: dict[str, str] = {}
    with prob_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            cat[o["id"]] = o["category"]

    rows: list[dict[str, str]] = []
    with train_path.open(newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            pid = rec["id"]
            rows.append(
                {
                    "id": pid,
                    "category": cat.get(pid, "unknown"),
                    "answer": rec["answer"].strip(),
                }
            )

    payload = {"version": 1, "rows": rows}
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {out_path} ({out_path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
