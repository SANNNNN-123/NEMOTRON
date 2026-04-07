#!/usr/bin/env python3
"""
Equation Symbolic solver (cipher-digit: all LHS/RHS characters are symbols mapping to digits).

Pipeline (DATA.md): CRACK bijection symbol<->digit from examples, SCAN pairing|op|format,
APPLY to decoded target, ENCODE answer back to symbols.

Combo space aligned with scripts/regenerate_eqsymbolic_cot_v4.py, with target completion:
every digit in the numeric output must map to some symbol from the prompt pool.

Run: python scripts/EQUATION_SYMBOLIC/solve_equation_symbolic.py [sample|full] [n]
"""

from __future__ import annotations

import re
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    ("maxv", lambda L, R: max(L, R)),
    ("minv", lambda L, R: min(L, R)),
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


def fmt_absrev(n):
    return None if n is None else str(abs(n))[::-1]


def fmt_neg(n):
    return None if n is None else "-" + str(abs(n))


def fmt_zpad(w):
    def f(n):
        if n is None or n < 0:
            return None
        return f"{n:0{w}d}"

    return f


def fmt_dsum(n):
    if n is None:
        return None
    return str(sum(int(c) for c in str(abs(n))))


def fmt_dprod(n):
    if n is None:
        return None
    p = 1
    for c in str(abs(n)):
        p *= int(c)
    return str(p)


FORMATS = [
    ("raw", fmt_raw),
    ("rev", fmt_rev),
    ("abs", fmt_abs),
    ("absrev", fmt_absrev),
    ("neg", fmt_neg),
    ("zpad2", fmt_zpad(2)),
    ("zpad3", fmt_zpad(3)),
    ("zpad4", fmt_zpad(4)),
    ("dsum", fmt_dsum),
    ("dprod", fmt_dprod),
]
FMT_DICT = dict(FORMATS)

FREQ_ORDER: List[Tuple[str, str, str]] = []
for p in PAIRINGS:
    for on, _ in OPS:
        for fn, _ in FORMATS:
            FREQ_ORDER.append((p, on, fn))


def apply_combo(p, on, fn, A, B, C, D):
    L, R = pair(p, A, B, C, D)
    v = OP_DICT[on](L, R)
    if v is None:
        return None, L, R, None
    return FMT_DICT[fn](v), L, R, v


def try_combo_on_group(combo, op_examples_enc):
    """Return one symbol->digit map that satisfies all op_examples_enc, or None."""
    p, on, fn = combo
    s0, s1, s2, s3, rhs = op_examples_enc[0]
    uniq = []
    for c in (s0, s1, s2, s3):
        if c not in uniq:
            uniq.append(c)
    k = len(uniq)
    for digits in permutations(range(10), k):
        m = dict(zip(uniq, digits))
        A, B, C, D = m[s0], m[s1], m[s2], m[s3]
        out, _, _, _ = apply_combo(p, on, fn, A, B, C, D)
        if out is None or len(out) != len(rhs):
            continue
        mm = dict(m)
        bad = False
        used_digits = set(mm.values())
        for ch, dch in zip(rhs, out):
            if dch == "-":
                if ch != "-":
                    bad = True
                    break
                continue
            if ch == "-":
                bad = True
                break
            d = int(dch)
            if ch in mm:
                if mm[ch] != d:
                    bad = True
                    break
            else:
                if d in used_digits:
                    bad = True
                    break
                mm[ch] = d
                used_digits.add(d)
        if bad:
            continue
        ok = True
        mm2 = dict(mm)
        for t0, t1, t2, t3, trhs in op_examples_enc[1:]:
            used = set(mm2.values())
            needed = [c for c in (t0, t1, t2, t3) if c not in mm2]
            if needed:
                found_ext = None
                uniq_needed = []
                for c in needed:
                    if c not in uniq_needed:
                        uniq_needed.append(c)
                avail = [d for d in range(10) if d not in used]
                for dperm in permutations(avail, len(uniq_needed)):
                    ext = dict(zip(uniq_needed, dperm))
                    mm3 = dict(mm2)
                    mm3.update(ext)
                    A2, B2, C2, D2 = mm3[t0], mm3[t1], mm3[t2], mm3[t3]
                    out2, _, _, _ = apply_combo(p, on, fn, A2, B2, C2, D2)
                    if out2 is None or len(out2) != len(trhs):
                        continue
                    ok_rhs = True
                    used2 = set(mm3.values())
                    for ch, dch in zip(trhs, out2):
                        if dch == "-":
                            if ch != "-":
                                ok_rhs = False
                                break
                            continue
                        if ch == "-":
                            ok_rhs = False
                            break
                        d = int(dch)
                        if ch in mm3:
                            if mm3[ch] != d:
                                ok_rhs = False
                                break
                        else:
                            if d in used2:
                                ok_rhs = False
                                break
                            mm3[ch] = d
                            used2.add(d)
                    if ok_rhs:
                        found_ext = mm3
                        break
                if found_ext is None:
                    ok = False
                    break
                mm2 = found_ext
            else:
                A2, B2, C2, D2 = mm2[t0], mm2[t1], mm2[t2], mm2[t3]
                out2, _, _, _ = apply_combo(p, on, fn, A2, B2, C2, D2)
                if out2 is None or len(out2) != len(trhs):
                    ok = False
                    break
                used2 = set(mm2.values())
                mm3 = dict(mm2)
                ok_rhs = True
                for ch, dch in zip(trhs, out2):
                    if dch == "-":
                        if ch != "-":
                            ok_rhs = False
                            break
                        continue
                    if ch == "-":
                        ok_rhs = False
                        break
                    d = int(dch)
                    if ch in mm3:
                        if mm3[ch] != d:
                            ok_rhs = False
                            break
                    else:
                        if d in used2:
                            ok_rhs = False
                            break
                        mm3[ch] = d
                        used2.add(d)
                if not ok_rhs:
                    ok = False
                    break
                mm2 = mm3
        if ok:
            return mm2
    return None


def parse_prompt(prompt: str):
    lines = prompt.split("\n")
    examples = []
    for line in lines:
        m = re.match(r"^\s*(\S{5})\s*=\s*(\S+)\s*$", line)
        if m:
            examples.append((m.group(1), m.group(2)))
    m = re.search(r"result for[:\s]+(\S{5})", prompt, re.I)
    if not m:
        return None, None
    if not examples:
        return None, None
    return examples, m.group(1)


def collect_symbols(prompt: str) -> Set[str]:
    syms: Set[str] = set()
    for line in prompt.split("\n"):
        mm = re.match(r"^\s*(\S{5})\s*=\s*(\S+)\s*$", line)
        if mm:
            for c in mm.group(1) + mm.group(2):
                syms.add(c)
    m = re.search(r"result for[:\s]+(\S{5})", prompt, re.I)
    if m:
        for c in m.group(1):
            syms.add(c)
    return syms


def encode_output(out_str: str, sym_to_digit: Dict[str, int]) -> Optional[str]:
    inv = {d: s for s, d in sym_to_digit.items()}
    enc = []
    for dch in out_str:
        if dch == "-":
            enc.append("-")
            continue
        d = int(dch)
        if d not in inv:
            return None
        enc.append(inv[d])
    return "".join(enc)


def complete_mapping_for_digits(
    sym_to_digit: Dict[str, int],
    digit_chars_needed: Set[int],
    symbol_pool: Set[str],
) -> Optional[List[Dict[str, int]]]:
    """
    Extend sym_to_digit so every digit in digit_chars_needed appears as a value.
    Uses only symbols from symbol_pool not yet used as keys. Returns list of all valid extensions.
    """
    have_vals = set(sym_to_digit.values())
    missing_vals = [d for d in digit_chars_needed if d not in have_vals]
    if not missing_vals:
        return [dict(sym_to_digit)]
    unused_syms = [s for s in symbol_pool if s not in sym_to_digit]
    if len(unused_syms) < len(missing_vals):
        return []
    out = []
    for perm in permutations(unused_syms, len(missing_vals)):
        ext = dict(sym_to_digit)
        for d, s in zip(missing_vals, perm):
            ext[s] = d
        if len(set(ext.values())) == len(ext):
            out.append(ext)
    return out


def solve_prompt(prompt: str, gold: Optional[str] = None) -> Optional[str]:
    """
    If gold is set, return the first encoding that matches gold (train verification).
    If gold is None, return the first valid encoding found (may not be unique).
    """
    examples, target = parse_prompt(prompt)
    if not examples or target is None:
        return None
    op_sym = target[2]
    op_examples = [
        (lhs[0], lhs[1], lhs[3], lhs[4], rhs)
        for lhs, rhs in examples
        if len(lhs) == 5 and lhs[2] == op_sym
    ]
    if not op_examples:
        return None
    symbol_pool = collect_symbols(prompt)
    tA, tB, tC, tD = target[0], target[1], target[3], target[4]

    for combo in FREQ_ORDER:
        mapping = try_combo_on_group(combo, op_examples)
        if mapping is None:
            continue
        need_lhs = list(dict.fromkeys([c for c in (tA, tB, tC, tD) if c not in mapping]))
        used = set(mapping.values())
        avail = [d for d in range(10) if d not in used]
        if len(avail) < len(need_lhs):
            continue
        for dperm_lhs in permutations(avail, len(need_lhs)):
            ext0 = dict(mapping)
            for s, d in zip(need_lhs, dperm_lhs):
                ext0[s] = d
            A, B, C, D = ext0[tA], ext0[tB], ext0[tC], ext0[tD]
            out, _, _, _ = apply_combo(*combo, A, B, C, D)
            if out is None:
                continue
            digits_needed = {int(c) for c in out if c.isdigit()}
            for ext in complete_mapping_for_digits(ext0, digits_needed, symbol_pool) or []:
                enc = encode_output(out, ext)
                if enc is None:
                    continue
                if gold is not None:
                    if enc == gold.strip():
                        return enc
                else:
                    return enc
    return None


OracleDetail = Tuple[
    Tuple[str, str, str],  # combo
    Dict[str, int],  # full symbol->digit mapping used for encode
    str,  # target LHS (5 chars)
    str,  # op symbol at index 2
    List[Tuple[str, str, str, str, str]],  # op_examples
    str,  # numeric output string
    str,  # encoded answer (symbols)
    int,
    int,
    Optional[int],  # L, R, v
]


def solve_oracle_detail(prompt: str, gold: str) -> Optional[OracleDetail]:
    """
    Same search as solve_prompt(..., gold), but returns full trace for CoT when gold matches.
    """
    gold = gold.strip()
    examples, target = parse_prompt(prompt)
    if not examples or target is None:
        return None
    op_sym = target[2]
    op_examples = [
        (lhs[0], lhs[1], lhs[3], lhs[4], rhs)
        for lhs, rhs in examples
        if len(lhs) == 5 and lhs[2] == op_sym
    ]
    if not op_examples:
        return None
    symbol_pool = collect_symbols(prompt)
    tA, tB, tC, tD = target[0], target[1], target[3], target[4]

    for combo in FREQ_ORDER:
        mapping = try_combo_on_group(combo, op_examples)
        if mapping is None:
            continue
        need_lhs = list(dict.fromkeys([c for c in (tA, tB, tC, tD) if c not in mapping]))
        used = set(mapping.values())
        avail = [d for d in range(10) if d not in used]
        if len(avail) < len(need_lhs):
            continue
        for dperm_lhs in permutations(avail, len(need_lhs)):
            ext0 = dict(mapping)
            for s, d in zip(need_lhs, dperm_lhs):
                ext0[s] = d
            A, B, C, D = ext0[tA], ext0[tB], ext0[tC], ext0[tD]
            out, tL, tR, tV = apply_combo(*combo, A, B, C, D)
            if out is None:
                continue
            digits_needed = {int(c) for c in out if c.isdigit()}
            for ext in complete_mapping_for_digits(ext0, digits_needed, symbol_pool) or []:
                enc = encode_output(out, ext)
                if enc is None:
                    continue
                if enc == gold:
                    return (combo, ext, target, op_sym, op_examples, out, enc, tL, tR, tV)
    return None


def is_equation_symbolic_prompt(prompt: str) -> bool:
    pl = prompt.lower()
    if "equation" not in pl and "transformation" not in pl:
        return False
    for line in prompt.strip().split("\n"):
        if "=" in line and "Now" not in line and "Determine" not in line:
            lhs = line.split("=")[0].strip()
            if not re.search(r"\d", lhs):
                return True
    return False


# Back-compat name (same predicate)
is_cipher_digit_prompt = is_equation_symbolic_prompt


def main():
    import pandas as pd

    mode = sys.argv[1] if len(sys.argv) > 1 else "sample"
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    df = pd.read_csv("data/train.csv", dtype={"answer": str})
    cdf = df[df["prompt"].apply(is_equation_symbolic_prompt)].reset_index(drop=True)
    print(f"Equation-symbolic problems total: {len(cdf)} | mode={mode}")
    if mode != "full":
        cdf = cdf.head(sample_size).reset_index(drop=True)
        print(f"Sample run: {len(cdf)} problems")

    t0 = time.time()
    correct = 0
    failures = []
    for i in range(len(cdf)):
        row = cdf.iloc[i]
        gold = str(row["answer"]).strip()
        pred = solve_prompt(row["prompt"], gold)
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
