"""
Build verified bit_manipulation training data using reasoning.py CoTs.

Strategy:
  1. Read pre-generated reasoning/*.txt files (familiar column-hashing format)
  2. Extract the boxed answer from each file
  3. Compare against ground truth from train.csv
  4. Keep ONLY files where reasoning.py answer == ground truth (verified correct)
  5. Merge with other categories from v6 → problem_ids_matched_v10.csv

This gives the model CoTs in the familiar format it already knows,
with guaranteed correct answers — best of both worlds.

Output: error_analysis/88_0/problem_ids_matched_v10.csv
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Set

# ── Paths ─────────────────────────────────────────────────────────────────────

NEMOTRON_REPO  = Path("/home/zuhair/Desktop/project/nemotron")
REASONING_DIR  = NEMOTRON_REPO / "reasoning"
PROBLEMS_JSONL = NEMOTRON_REPO / "problems.jsonl"
TRAIN_CSV      = NEMOTRON_REPO / "train.csv"

BASE           = Path(__file__).resolve().parents[2]
V6_CSV         = BASE / "error_analysis" / "88_0" / "problem_ids_matched_v6.csv"
OUT_CSV        = BASE / "error_analysis" / "88_0" / "problem_ids_matched_v10.csv"

# ── Answer extraction (mirrors reasoning.py extract_answer) ───────────────────

def extract_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load bit_manipulation IDs
    bit_ids: Set[str] = set()
    with PROBLEMS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") == "bit_manipulation":
                bit_ids.add(r["id"])

    print(f"bit_manipulation problems : {len(bit_ids)}")

    # Load ground truth from train.csv
    ground_truth: Dict[str, str] = {}
    prompts:      Dict[str, str] = {}
    with TRAIN_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] in bit_ids:
                ground_truth[row["id"]] = row["answer"].strip()
                prompts[row["id"]]      = row["prompt"]

    print(f"Loaded train.csv entries  : {len(ground_truth)}")

    # Process each bit_manipulation problem
    correct_rows = []
    wrong_rows   = []
    no_ans_rows  = []

    for pid in sorted(bit_ids):
        rfile = REASONING_DIR / f"{pid}.txt"
        if not rfile.exists():
            no_ans_rows.append(pid)
            continue

        text  = rfile.read_text(encoding="utf-8")
        ans   = extract_answer(text)
        truth = ground_truth.get(pid, "")

        if not ans:
            no_ans_rows.append(pid)
            continue

        if ans == truth:
            correct_rows.append({
                "id":            pid,
                "prompt":        prompts.get(pid, ""),
                "answer":        truth,
                "type":          "bit_manipulation",
                "generated_cot": text.strip(),
            })
        else:
            wrong_rows.append((pid, ans, truth))

    print(f"\nReasoning.py results:")
    print(f"  Correct (keeping)  : {len(correct_rows)}")
    print(f"  Wrong   (dropping) : {len(wrong_rows)}")
    print(f"  No answer          : {len(no_ans_rows)}")
    print(f"  Accuracy           : {100*len(correct_rows)/len(bit_ids):.1f}%")

    # Load non-bit rows from v6
    non_bit = []
    with V6_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["type"] != "bit_manipulation":
                non_bit.append(row)

    print(f"\nNon-bit rows from v6.csv  : {len(non_bit)}")

    # Merge and write
    merged     = correct_rows + non_bit
    fieldnames = ["id", "prompt", "answer", "type", "generated_cot"]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    # Summary
    types = Counter(r["type"] for r in merged)
    total = len(merged)
    print(f"\n--- Done ---")
    print(f"\nDataset breakdown:")
    print(f"  {'Category':<30} {'Count':>7} {'%':>7}")
    print(f"  {'-'*44}")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:<30} {c:>7} {100*c/total:>6.1f}%")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<30} {total:>7} {100.0:>6.1f}%")
    print(f"\nOutput: {OUT_CSV}")

if __name__ == "__main__":
    main()
