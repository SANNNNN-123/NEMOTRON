#!/usr/bin/env python3
"""
Brute-force English substitution cipher solver (train distribution).

1) Build cipher<->plain table from all example lines (bijective merge).
2) For each target word, list plaintext words in VOCAB that match known letters (? = wildcard).
3) Greedily lock uniquely determined words; if gaps remain, DFS over remaining words and
   try every compatible vocab word until the substitution stays consistent.

Run from repo root:
  python scripts/CIPHER/solve_cipher.py             # sample 200 (same default as solve_binary.py)
  python scripts/CIPHER/solve_cipher.py full      # all cipher rows in train.csv
  python scripts/CIPHER/solve_cipher.py sample N
"""

from __future__ import annotations

import copy
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab import CIPHER_VOCAB

EXAMPLE_RE = re.compile(r"([a-z ]+?)\s*->\s*([a-z ]+)")
TARGET_RE = re.compile(r"decrypt the following text:\s*(.+?)(?:\n|$)", re.I)


def parse_prompt(prompt: str) -> Optional[Tuple[List[Tuple[str, str]], str]]:
    pairs = EXAMPLE_RE.findall(prompt)
    examples = []
    for enc, plain in pairs:
        enc, plain = enc.strip(), plain.strip()
        if not enc or not plain or "decrypt" in plain.lower():
            continue
        if len(enc.split()) != len(plain.split()) or len(enc) != len(plain):
            continue
        examples.append((enc, plain))
    m = TARGET_RE.search(prompt)
    if not m or not examples:
        return None
    return examples, m.group(1).strip()


def merge_letter(e: str, p: str, fwd: Dict[str, str], inv: Dict[str, str]) -> bool:
    if e in fwd:
        return fwd[e] == p
    if p in inv:
        return inv[p] == e
    fwd[e] = p
    inv[p] = e
    return True


def build_table(examples: List[Tuple[str, str]]) -> Optional[Tuple[Dict[str, str], Dict[str, str]]]:
    fwd: Dict[str, str] = {}
    inv: Dict[str, str] = {}
    for enc, plain in examples:
        for a, b in zip(enc, plain):
            if a == " " and b == " ":
                continue
            if a == " " or b == " ":
                return None
            if not merge_letter(a, b, fwd, inv):
                return None
    return fwd, inv


def decrypt_partial(text: str, fwd: Dict[str, str]) -> str:
    return "".join(fwd.get(c, "?") if c != " " else " " for c in text)


def word_matches(cipher_word: str, fwd: Dict[str, str]) -> List[str]:
    pat = "".join(fwd.get(c, "?") for c in cipher_word)
    if "?" not in pat:
        return [pat] if pat in CIPHER_VOCAB else []
    out = []
    for w in CIPHER_VOCAB:
        if len(w) != len(cipher_word):
            continue
        if all(pat[i] == "?" or pat[i] == w[i] for i in range(len(w))):
            out.append(w)
    return out


def apply_word(cw: str, pw: str, fwd: Dict[str, str], inv: Dict[str, str]) -> bool:
    for e, p in zip(cw, pw):
        if not merge_letter(e, p, fwd, inv):
            return False
    return True


def greedy_unique(target: str, fwd: Dict[str, str], inv: Dict[str, str]) -> Tuple[bool, Dict[str, str], Dict[str, str]]:
    fwd = copy.deepcopy(fwd)
    inv = copy.deepcopy(inv)
    words = target.split()
    for _ in range(300):
        changed = False
        for cw in words:
            cands = word_matches(cw, fwd)
            if not cands:
                return False, fwd, inv
            if len(cands) == 1:
                if not apply_word(cw, cands[0], fwd, inv):
                    return False, fwd, inv
                changed = True
        if not changed:
            break
    return ("?" not in decrypt_partial(target, fwd)), fwd, inv


def dfs_bruteforce(
    words: List[str],
    i: int,
    fwd: Dict[str, str],
    inv: Dict[str, str],
) -> Optional[Dict[str, str]]:
    if i == len(words):
        return fwd
    cw = words[i]
    for pw in word_matches(cw, fwd):
        nf, ni = copy.deepcopy(fwd), copy.deepcopy(inv)
        if not apply_word(cw, pw, nf, ni):
            continue
        r = dfs_bruteforce(words, i + 1, nf, ni)
        if r is not None:
            return r
    return None


def solve_target(target: str, fwd: Dict[str, str], inv: Dict[str, str]) -> Optional[str]:
    ok, f2, i2 = greedy_unique(target, fwd, inv)
    if ok:
        return decrypt_partial(target, f2).strip()
    f3 = dfs_bruteforce(target.split(), 0, copy.deepcopy(fwd), copy.deepcopy(inv))
    if f3 is None or "?" in decrypt_partial(target, f3):
        return None
    return decrypt_partial(target, f3).strip()


def inv_from_fwd(fwd: Dict[str, str]) -> Dict[str, str]:
    return {p: e for e, p in fwd.items()}


def solve_prompt_with_mapping(
    prompt: str,
) -> Optional[Tuple[str, Dict[str, str], Dict[str, str]]]:
    """Return (plaintext_answer, final_fwd, final_inv) after vocab fill, or None."""
    p = parse_prompt(prompt)
    if not p:
        return None
    examples, target = p
    t = build_table(examples)
    if not t:
        return None
    fwd, inv = t
    ok, f2, i2 = greedy_unique(target, copy.deepcopy(fwd), copy.deepcopy(inv))
    if ok:
        return decrypt_partial(target, f2).strip(), f2, i2
    f3 = dfs_bruteforce(target.split(), 0, copy.deepcopy(fwd), copy.deepcopy(inv))
    if f3 is None or "?" in decrypt_partial(target, f3):
        return None
    return decrypt_partial(target, f3).strip(), f3, inv_from_fwd(f3)


def solve(prompt: str) -> Optional[str]:
    r = solve_prompt_with_mapping(prompt)
    return None if r is None else r[0]


def is_cipher_prompt(s: str) -> bool:
    return bool(TARGET_RE.search(s) and EXAMPLE_RE.search(s))


def main() -> None:
    import pandas as pd

    # Usage: python solve_cipher.py            -> sample run (200)
    #        python solve_cipher.py full       -> full cipher rows
    #        python solve_cipher.py sample N   -> sample of N
    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    df = pd.read_csv("data/train.csv", dtype={"answer": str})
    cdf = df[df["prompt"].apply(is_cipher_prompt)].reset_index(drop=True)
    print(f"Cipher problems total: {len(cdf)} | mode={mode}")
    if mode != "full":
        cdf = cdf.head(sample_size).reset_index(drop=True)
        print(f"Sample run: {len(cdf)} problems")

    t0 = time.time()
    correct = 0
    unsolved = 0
    failures: List[Tuple[str, Optional[str], str]] = []
    for i in range(len(cdf)):
        row = cdf.iloc[i]
        pred = solve(row["prompt"])
        gold = str(row["answer"]).strip()
        if pred is None:
            unsolved += 1
        ok = pred == gold
        if ok:
            correct += 1
        else:
            failures.append((str(row["id"]), pred, gold))
        mark = "OK " if ok else "XX "
        print(
            f"[{i + 1}/{len(cdf)}] {mark} id={row['id']} pred={pred} gold={gold}",
            flush=True,
        )

    dt = time.time() - t0
    print("\n====== RESULTS ======")
    print(f"Total: {len(cdf)}")
    print(f"Correct: {correct} ({correct / len(cdf) * 100:.2f}%)")
    print(f"Unsolved (no prediction): {unsolved}")
    print(f"Time: {dt:.1f}s")

    if failures:
        print(f"\nFailures: {len(failures)} (showing up to 5)")
        for fid, pred, gold in failures[:5]:
            print(f"  {fid}: pred={pred} gold={gold}")


if __name__ == "__main__":
    main()
