#!/usr/bin/env python3
"""
Equation Numeric solver — the train rows where equations use REAL digits (0–9), not symbols.

Each line is:  (A)(B)(op)(C)(D) = result   with exactly one character for the operator at index 2,
e.g. 42-16 = -73  →  A=4,B=2, op='-', C=1,D=6, result string from the puzzle.

This is *not* cipher-digit (symbol substitution). It is the same family as DATA.md "symbol-digit"
/ verify_templates `symbol_digit`: scan (pairing × op × format) in frequency order until all
examples for the *target operator* match, then apply to the target.

Logic is aligned with scripts/regenerate_eqnumeric_cot_v4.py.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAIRINGS = ["AB_CD", "BA_DC", "AB_DC", "BA_CD"]


def pair(p: str, A: int, B: int, C: int, D: int) -> Tuple[int, int]:
    if p == "AB_CD":
        return 10 * A + B, 10 * C + D
    if p == "BA_DC":
        return 10 * B + A, 10 * D + C
    if p == "AB_DC":
        return 10 * A + B, 10 * D + C
    if p == "BA_CD":
        return 10 * B + A, 10 * C + D
    raise ValueError(p)


OPS = [
    ("add", lambda L, R: L + R),
    ("sub", lambda L, R: L - R),
    ("mul", lambda L, R: L * R),
    ("cat", lambda L, R: int(f"{L}{R}") if L >= 0 and R >= 0 else None),
    ("add1", lambda L, R: L + R + 1),
    ("addm1", lambda L, R: L + R - 1),
    ("muladd1", lambda L, R: L * R + 1),
    ("mulsub1", lambda L, R: L * R - 1),
    ("absdiff", lambda L, R: abs(L - R)),
    ("xor", lambda L, R: L ^ R),
    ("orsum", lambda L, R: L | R),
    ("andsum", lambda L, R: L & R),
    ("maxv", lambda L, R: max(L, R)),
    ("minv", lambda L, R: min(L, R)),
    ("div", lambda L, R: L // R if R != 0 else None),
    ("mod", lambda L, R: L % R if R != 0 else None),
]
OP_DICT = dict(OPS)


def fmt_raw(n):
    return None if n is None else str(n)


def fmt_rev(n):
    if n is None:
        return None
    s = str(n)
    return "-" + s[1:][::-1] if s.startswith("-") else s[::-1]


def fmt_abs(n):
    return None if n is None else str(abs(n))


def fmt_neg(n):
    return None if n is None else "-" + str(abs(n))


def fmt_zpad(w):
    def f(n):
        if n is None or n < 0:
            return None
        return f"{n:0{w}d}"

    return f


def fmt_dsum(n):
    return None if n is None else str(sum(int(c) for c in str(abs(n))))


FORMATS = [
    ("raw", fmt_raw),
    ("rev", fmt_rev),
    ("abs", fmt_abs),
    ("neg", fmt_neg),
    ("zpad2", fmt_zpad(2)),
    ("zpad3", fmt_zpad(3)),
    ("zpad4", fmt_zpad(4)),
    ("dsum", fmt_dsum),
]
FMT_DICT = dict(FORMATS)

FREQ_ORDER: List[Tuple[str, str, str]] = [
    ("AB_CD", "cat", "raw"),
    ("AB_CD", "mul", "raw"),
    ("AB_CD", "add", "raw"),
    ("AB_CD", "sub", "raw"),
    ("AB_CD", "absdiff", "raw"),
    ("BA_DC", "cat", "rev"),
    ("BA_DC", "mul", "rev"),
    ("BA_DC", "add", "rev"),
    ("BA_DC", "cat", "raw"),
    ("AB_CD", "cat", "rev"),
    ("AB_CD", "mul", "rev"),
    ("AB_CD", "add", "rev"),
    ("AB_CD", "sub", "neg"),
    ("AB_CD", "sub", "rev"),
    ("BA_DC", "sub", "rev"),
    ("AB_CD", "add1", "raw"),
    ("AB_CD", "addm1", "raw"),
    ("AB_CD", "muladd1", "raw"),
    ("AB_CD", "mulsub1", "raw"),
    ("AB_CD", "xor", "raw"),
    ("AB_CD", "mod", "raw"),
    ("AB_CD", "div", "raw"),
    ("AB_CD", "maxv", "raw"),
    ("AB_CD", "minv", "raw"),
]
_seen = set(FREQ_ORDER)
for p in PAIRINGS:
    for on, _ in OPS:
        for fn, _ in FORMATS:
            t = (p, on, fn)
            if t not in _seen:
                FREQ_ORDER.append(t)
                _seen.add(t)


def apply_combo(p, on, fn, A, B, C, D):
    L, R = pair(p, A, B, C, D)
    v = OP_DICT[on](L, R)
    if v is None:
        return None, L, R, None
    return FMT_DICT[fn](v), L, R, v


def parse_prompt(prompt: str):
    lines = prompt.split("\n")
    examples = []
    for line in lines:
        m = re.match(r"^\s*(\d)(\d)(\D)(\d)(\d)\s*=\s*(\S+)\s*$", line)
        if m:
            A, B, sym, C, D, res = m.groups()
            examples.append((int(A), int(B), sym, int(C), int(D), res))
    m = re.search(r"result for[:\s]+(\d)(\d)(\D)(\d)(\d)", prompt, re.I)
    if not m:
        return None, None
    tA, tB, top, tC, tD = m.groups()
    return examples, (int(tA), int(tB), top, int(tC), int(tD))


def find_combo(op_examples):
    attempts = []
    for combo in FREQ_ORDER:
        p, on, fn = combo
        ok = True
        first_eval = None
        for i, (A, B, C, D, res) in enumerate(op_examples):
            out, L, R, v = apply_combo(p, on, fn, A, B, C, D)
            if i == 0:
                first_eval = (L, R, v, out)
            if out is None or out != res:
                ok = False
                break
        attempts.append((combo, first_eval, ok))
        if ok:
            return combo, attempts
    return None, attempts


def find_combo_oracle(
    op_examples: List[Tuple[int, int, int, int, str]],
    tA: int,
    tB: int,
    tC: int,
    tD: int,
    gold: str,
) -> Tuple[Optional[Tuple[str, str, str]], List[Tuple]]:
    """
    Like find_combo, but only accepts a combo if apply_combo on the target equals `gold`.

    Use this for CoT generation / training so the trace matches labels even when the
    greedy first-match rule would pick a different (still example-consistent) combo.
    """
    gold = gold.strip()
    attempts: List[Tuple] = []
    for combo in FREQ_ORDER:
        p, on, fn = combo
        ok_ex = True
        first_eval = None
        for i, (A, B, C, D, res) in enumerate(op_examples):
            out, L, R, v = apply_combo(p, on, fn, A, B, C, D)
            if i == 0:
                first_eval = (L, R, v, out)
            if out is None or out != res:
                ok_ex = False
                break
        pred, _, _, _ = (
            apply_combo(p, on, fn, tA, tB, tC, tD) if ok_ex else (None, None, None, None)
        )
        ok_gold = ok_ex and pred is not None and pred == gold
        attempts.append((combo, first_eval, ok_gold))
        if ok_gold:
            return combo, attempts
    return None, attempts


def solve_prompt(prompt: str) -> Optional[str]:
    parsed = parse_prompt(prompt)
    if not parsed[0] or not parsed[1]:
        return None
    examples, target = parsed
    tA, tB, top, tC, tD = target
    op_examples = [(A, B, C, D, res) for A, B, sym, C, D, res in examples if sym == top]
    if not op_examples:
        return None
    combo, _ = find_combo(op_examples)
    if combo is None:
        return None
    p, on, fn = combo
    out, _, _, _ = apply_combo(p, on, fn, tA, tB, tC, tD)
    return out


def solve_prompt_oracle(prompt: str, gold: str) -> Optional[str]:
    """Return gold if some (pairing,op,fmt) fits all examples for the op and matches gold on target; else None."""
    parsed = parse_prompt(prompt)
    if not parsed[0] or not parsed[1]:
        return None
    examples, target = parsed
    tA, tB, top, tC, tD = target
    op_examples = [(A, B, C, D, res) for A, B, sym, C, D, res in examples if sym == top]
    if not op_examples:
        return None
    combo, _ = find_combo_oracle(op_examples, tA, tB, tC, tD, gold)
    if combo is None:
        return None
    p, on, fn = combo
    out, _, _, _ = apply_combo(p, on, fn, tA, tB, tC, tD)
    return out


def is_equation_numeric_prompt(prompt: str) -> bool:
    """Same idea as verify_templates: equation/transformation + a real digit on the LHS."""
    pl = prompt.lower()
    if "equation" not in pl and "transformation" not in pl:
        return False
    for line in prompt.strip().split("\n"):
        if "=" in line and "Now" not in line and "Determine" not in line:
            lhs = line.split("=")[0].strip()
            if re.search(r"\d", lhs):
                return True
    return False


def main():
    import pandas as pd

    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    df = pd.read_csv("data/train.csv", dtype={"answer": str})
    cdf = df[df["prompt"].apply(is_equation_numeric_prompt)].reset_index(drop=True)
    print(f"Equation-numeric (digit LHS) problems total: {len(cdf)} | mode={mode}")
    if mode != "full":
        cdf = cdf.head(sample_size).reset_index(drop=True)
        print(f"Sample run: {len(cdf)} problems")

    t0 = time.time()
    correct = 0
    failures = []
    for i in range(len(cdf)):
        row = cdf.iloc[i]
        gold = str(row["answer"]).strip()
        pred = solve_prompt(row["prompt"])
        ok = pred == gold
        if ok:
            correct += 1
        else:
            failures.append((row["id"], pred, gold))
        mark = "OK " if ok else "XX "
        print(
            f"[{i + 1}/{len(cdf)}] {mark} id={row['id']} pred={pred} gold={gold}",
            flush=True,
        )

    dt = time.time() - t0
    print("\n====== RESULTS ======")
    print(f"Total: {len(cdf)}")
    print(f"Correct: {correct} ({correct / len(cdf) * 100:.2f}%)")
    print(f"Time: {dt:.1f}s")
    if failures:
        print(f"\nFailures: {len(failures)} (showing up to 8)")
        for fid, p, g in failures[:8]:
            print(f"  {fid}: pred={p!r} gold={g!r}")


if __name__ == "__main__":
    main()
