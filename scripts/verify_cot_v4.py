"""
Verify that the generated_cot template in data/train_split_with_cot_v4.csv
is usable: feed it to Gemini as a few-shot, then test the LLM on Equation
Numeric and Equation Symbolic ground-truth rows from the same CSV.

Reuses FEWSHOT, build_system_prompt, extract_boxed, answers_match from
verify_templates.py so this stays a single source of truth for templates.

Usage:
    python scripts/verify_cot_v4.py
"""
import csv
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from google import genai
from google.genai import types

from verify_templates import (
    extract_boxed,
    answers_match,
)


def build_system_prompt_from_row(shot_row: dict) -> str:
    """Use an actual v4 row (prompt + generated_cot) as the few-shot example."""
    return (
        "You are a reasoning model solving Kaggle competition puzzles. "
        "You MUST follow the exact template shown in the example below. "
        "You MUST put your final answer inside \\boxed{} at the very end. "
        "Do NOT deviate from the template structure. "
        "Ignore all flavor text (Alice in Wonderland, etc.) — focus only on the math/logic.\n\n"
        "=== EXAMPLE 1 ===\n"
        f"PROMPT:\n{shot_row['prompt']}\n\n"
        f"CORRECT RESPONSE:\n{shot_row['generated_cot']}\n\n"
        "Now solve the next problem using the SAME template structure. "
        "Show your work step by step, then put your final answer in \\boxed{}."
    )

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-pro"
CSV_PATH = Path(__file__).parent.parent / "data" / "train_split_with_cot_v4.csv"

# How many rows of each type to test
SAMPLES = {
    "Equation Numeric": 40,
    "Equation Symbolic": 40,
}
SEED = 42
MAX_RETRIES = 3
RETRY_DELAY = 2


def main():
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY in .env")
        return
    client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"Loading {CSV_PATH.name}...")
    rows_by_type: dict[str, list] = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_type.setdefault(row["type"], []).append(row)

    for t in SAMPLES:
        print(f"  {t}: {len(rows_by_type.get(t, []))} rows available")

    random.seed(SEED)
    sampled = {}
    fewshot_rows = {}  # per type: the one row used as the few-shot example
    for t, n in SAMPLES.items():
        avail = [r for r in rows_by_type.get(t, []) if r.get("generated_cot")]
        if not avail:
            print(f"  WARNING: no rows for {t}")
            continue
        pool = random.sample(avail, min(n + 1, len(avail)))
        fewshot_rows[t] = pool[0]  # first becomes the few-shot demo
        sampled[t] = pool[1:]      # rest are test rows
        print(f"  {t}: fewshot id={fewshot_rows[t]['id']}, testing {len(sampled[t])}")

    # Build system prompts from the actual v4 CoT rows
    system_prompts = {t: build_system_prompt_from_row(fewshot_rows[t]) for t in sampled}

    # Interleave for early signal across both types
    queues = {t: list(s) for t, s in sampled.items()}
    interleaved = []
    while any(queues.values()):
        for t in list(queues.keys()):
            if queues[t]:
                interleaved.append((t, queues[t].pop(0)))

    results = {t: {"total": len(s), "correct": 0, "errors": 0, "done": 0, "failures": []}
               for t, s in sampled.items()}

    print(f"\n{'='*60}\nVERIFYING {len(interleaved)} samples\n{'='*60}")

    for idx, (t, row) in enumerate(interleaved):
        results[t]["done"] += 1
        prompt = row["prompt"]
        gt = row["answer"]
        rid = row["id"]
        predicted, raw = None, ""

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompts[t],
                        temperature=0.0,
                        max_output_tokens=16384,
                    ),
                )
                raw = resp.text or ""
                predicted = extract_boxed(raw)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"    retry {attempt+1}: {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"    FAILED {rid}: {e}")
                    results[t]["errors"] += 1

        ok = answers_match(predicted, gt)
        if ok:
            results[t]["correct"] += 1
        else:
            results[t]["failures"].append({"id": rid, "gt": gt, "pred": predicted})

        status = "OK  " if ok else "MISS"
        done = results[t]["done"]
        tot = results[t]["total"]
        acc = results[t]["correct"] / done * 100
        print(f"  [{idx+1:3d}/{len(interleaved)}] {status} | {t:<18} "
              f"[{done:3d}/{tot:3d} {acc:5.1f}%] | id={rid} | gt={gt} | pred={predicted}")
        time.sleep(0.5)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Type':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for t in SAMPLES:
        if t in results:
            r = results[t]
            acc = r["correct"] / r["total"] * 100 if r["total"] else 0
            print(f"{t:<20} {r['correct']:>8} {r['total']:>8} {acc:>9.1f}%")
    print()
    for t in SAMPLES:
        if t in results and results[t]["failures"]:
            print(f"First failures for {t}:")
            for f in results[t]["failures"][:5]:
                print(f"  id={f['id']} gt={f['gt']} pred={f['pred']}")
            print()


if __name__ == "__main__":
    main()
