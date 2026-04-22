"""
Balanced structured dataset generator for bit_manipulation.

Strategy:
  1. Run solver on all 1602 real train.csv problems → get (rule_rank, expr, td, pred)
  2. Group correct predictions by rule_rank
  3. For each of 20 rules: take up to N_PER_RULE real problems
  4. Fill remaining with SYNTHETIC problems (random inputs, perfect labels)
  5. Output: structured format  RuleID / Rule / Transforms / Output / \\boxed{}
     (no reasoning chains, no FAIL/PASS text — direct classification target)
  6. Merge with other categories from v6 → problem_ids_matched_v9.csv

Output format (generated_cot):
  RuleID: 3
  Rule: XNOR({A},NOT_A_AND_B({B},{C}))
  Transforms: A=shl(1)  B=shr(1)  C=rot(7)
  Output: 10010111

  \\boxed{10010111}
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_PER_RULE    = 80    # target samples per rule (20 rules × 80 = 1,600 total)
N_EXAMPLES    = 9     # input-output examples per problem (matches competition)

# ── 8-bit helpers ─────────────────────────────────────────────────────────────

def bin8(x: Any) -> str:
    s = "".join(c for c in str(x).strip() if c in "01")
    return (s.zfill(8) if s else "00000000")[-8:]

def bits(s: str) -> List[int]:
    return [int(c) for c in bin8(s)]

def bits_to_str(b: List[int]) -> str:
    return "".join(map(str, b))

# ── Transforms ────────────────────────────────────────────────────────────────

TRANSFORMS: List[Tuple[str, int]] = [("rot", 0)]
for _k in range(1, 8):
    TRANSFORMS.extend([("rot", _k), ("shl", _k), ("shr", _k)])

def trans_label(t: Tuple[str, int]) -> str:
    tp, k = t
    if tp == "rot" and k == 0: return "identity"
    if tp == "rot":  return f"rot({k})"
    if tp == "shl":  return f"shl({k})"
    return f"shr({k})"

def get_src(in_bits: List[int], i: int, t: Tuple[str, int]) -> int:
    tp, k = t
    if tp == "rot": return in_bits[(i + k) % 8]
    if tp == "shl":
        idx = i + k; return in_bits[idx] if 0 <= idx < 8 else 0
    idx = i - k; return in_bits[idx] if 0 <= idx < 8 else 0

# ── Operations ────────────────────────────────────────────────────────────────

OPS: Dict[str, Callable] = {
    "AND":         lambda a, b: a & b,
    "OR":          lambda a, b: a | b,
    "XOR":         lambda a, b: a ^ b,
    "NAND":        lambda a, b: ~(a & b),
    "NOR":         lambda a, b: ~(a | b),
    "XNOR":        lambda a, b: ~(a ^ b),
    "NOT_A_AND_B": lambda a, b: (~a) & b,
    "A_AND_NOT_B": lambda a, b: a & (~b),
    "NOT_A_OR_B":  lambda a, b: (~a) | b,
    "A_OR_NOT_B":  lambda a, b: a | (~b),
}

# ── Grammar ───────────────────────────────────────────────────────────────────

def _grammar():
    mask = 255
    l0 = {
        0:          ("C0",  lambda a, b, c, m: 0),
        255:        ("C1",  lambda a, b, c, m: m),
        0b11110000: ("{A}", lambda a, b, c, m: a),
        0b11001100: ("{B}", lambda a, b, c, m: b),
        0b10101010: ("{C}", lambda a, b, c, m: c),
    }
    visited = set(l0.keys()); levels = [l0]
    for tt, (expr, func) in l0.items(): yield tt, expr, func
    for depth in range(1, 4):
        nxt: Dict = {}
        for v, (expr, func) in levels[-1].items():
            nv = (~v) & mask
            if nv not in visited:
                ne = f"NOT({expr})"; nf = lambda a,b,c,m,f=func: (~f(a,b,c,m)) & m
                visited.add(nv); nxt[nv] = (ne, nf); yield nv, ne, nf
        for i in range(depth):
            j = depth - 1
            for v1, (e1, f1) in levels[i].items():
                for v2, (e2, f2) in levels[j].items():
                    for on, op in OPS.items():
                        if i == j and v1 > v2 and on in ("AND","OR","XOR","NAND","NOR","XNOR"):
                            continue
                        val = op(v1, v2) & mask
                        if val not in visited:
                            ne = f"{on}({e1},{e2})"
                            nf = lambda a,b,c,m,f1=f1,f2=f2,op=op: op(f1(a,b,c,m),f2(a,b,c,m)) & m
                            visited.add(val); nxt[val] = (ne, nf); yield val, ne, nf
                        if i != j:
                            val2 = op(v2, v1) & mask
                            if val2 not in visited:
                                ne2 = f"{on}({e2},{e1})"
                                nf2 = lambda a,b,c,m,f1=f1,f2=f2,op=op: op(f2(a,b,c,m),f1(a,b,c,m)) & m
                                visited.add(val2); nxt[val2] = (ne2, nf2); yield val2, ne2, nf2
        levels.append(nxt)

def _used(expr: str) -> List[str]:
    return [v for v in ("{A}", "{B}", "{C}") if v in expr]

# ── Registry ──────────────────────────────────────────────────────────────────

TT_FREQ_ORDER = [
    0x3c, 0xfc, 0x2d, 0xf0, 0xc0, 0x0c, 0xd2, 0xe0, 0x6d, 0xf9,
    0xac, 0xe8, 0xc3, 0xcf, 0xf2, 0x68, 0xff, 0xd0, 0x0d, 0x03,
]

def _build_registry():
    tt_map: Dict[int, Tuple] = {}
    for tt, expr, fn in _grammar():
        if tt not in tt_map:
            tt_map[tt] = (expr, fn)
    return [(tt, *tt_map[tt]) for tt in TT_FREQ_ORDER if tt in tt_map]

REGISTRY = _build_registry()

# ── Transform combos ──────────────────────────────────────────────────────────

def _combos(used: List[str]):
    if len(used) == 0:
        yield {}
    elif len(used) == 1:
        for t in TRANSFORMS:
            yield {used[0]: t}
    elif len(used) == 2:
        for t1 in TRANSFORMS:
            for t2 in TRANSFORMS:
                if t1 != t2:
                    yield {used[0]: t1, used[1]: t2}
    else:
        for t1, t2, t3 in itertools.permutations(TRANSFORMS, 3):
            yield {used[0]: t1, used[1]: t2, used[2]: t3}

def _eval_all(ev, td, ins, outs, n) -> bool:
    for b in range(8):
        for ex in range(n):
            a  = get_src(ins[ex], b, td.get("{A}", ("rot", 0)))
            bv = get_src(ins[ex], b, td.get("{B}", ("rot", 0)))
            cv = get_src(ins[ex], b, td.get("{C}", ("rot", 0)))
            if ev(a, bv, cv, 1) & 1 != outs[ex][b]:
                return False
    return True

def apply_rule(ev, td, in_bits: List[int]) -> List[int]:
    out = []
    for b in range(8):
        a  = get_src(in_bits, b, td.get("{A}", ("rot", 0)))
        bv = get_src(in_bits, b, td.get("{B}", ("rot", 0)))
        cv = get_src(in_bits, b, td.get("{C}", ("rot", 0)))
        out.append(ev(a, bv, cv, 1) & 1)
    return out

# ── Solver ────────────────────────────────────────────────────────────────────

def solve(ins, outs, query_in):
    n = len(ins)
    for rank, (tt, expr, ev) in enumerate(REGISTRY, 1):
        used = _used(expr)
        for td in _combos(used):
            if _eval_all(ev, td, ins, outs, n):
                pred = bits_to_str(apply_rule(ev, td, query_in))
                return rank, expr, td, pred
    return None, None, None, None

# ── Prompt parser ─────────────────────────────────────────────────────────────

_EX  = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_QRY = re.compile(r"(?:output for:|determine the output for:)\s*([01]{8})", re.I)

def parse(prompt: str):
    pairs = _EX.findall(prompt)
    qm    = _QRY.search(prompt)
    if not pairs or not qm:
        return None
    return ([bits(a) for a, _ in pairs],
            [bits(b) for _, b in pairs],
            bits(qm.group(1)))

# ── Structured CoT builder ────────────────────────────────────────────────────

def make_structured_cot(rule_rank: int, expr: str, td: Dict, pred: str) -> str:
    used = _used(expr)
    td_str = "  ".join(f"{u.strip('{}')}={trans_label(td[u])}" for u in used) if used else "no wires"
    L = []
    L.append(f"RuleID: {rule_rank}")
    L.append(f"Rule: {expr}")
    L.append(f"Transforms: {td_str}")
    L.append(f"Output: {pred}")
    L.append("")
    L.append(r"\boxed{" + pred + r"}")
    return "\n".join(L)

# ── Synthetic problem generator ───────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
    "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
    "and possibly majority or choice functions.\n\n"
    "Here are some examples of input -> output:\n"
    "{examples}\n\n"
    "Now, determine the output for: {query}"
)

def make_synthetic_problem(rule_rank: int, expr: str, ev, td: Dict, n_ex: int = N_EXAMPLES):
    """Generate a synthetic competition-style problem for a given rule + transforms."""
    # Random examples
    ins, outs = [], []
    seen = set()
    attempts = 0
    while len(ins) < n_ex and attempts < 200:
        attempts += 1
        in_bits = [random.randint(0, 1) for _ in range(8)]
        key = tuple(in_bits)
        if key in seen:
            continue
        seen.add(key)
        out_bits = apply_rule(ev, td, in_bits)
        ins.append(in_bits); outs.append(out_bits)

    # Random query (different from examples)
    while True:
        q = [random.randint(0, 1) for _ in range(8)]
        if tuple(q) not in seen:
            break
    pred = bits_to_str(apply_rule(ev, td, q))

    # Build prompt
    examples_str = "\n".join(
        f"{bits_to_str(i)} -> {bits_to_str(o)}" for i, o in zip(ins, outs)
    )
    prompt = PROMPT_TEMPLATE.format(examples=examples_str, query=bits_to_str(q))
    answer = pred

    # Synthetic ID
    hash_src = f"syn_{rule_rank}_{bits_to_str(q)}_{examples_str[:32]}"
    pid = "syn" + hashlib.md5(hash_src.encode()).hexdigest()[:6]

    cot = make_structured_cot(rule_rank, expr, td, pred)
    return pid, prompt, answer, cot

def random_td(used: List[str]) -> Dict:
    """Pick a random valid transform assignment."""
    if len(used) == 0:
        return {}
    elif len(used) == 1:
        return {used[0]: random.choice(TRANSFORMS)}
    elif len(used) == 2:
        t1, t2 = random.sample(TRANSFORMS, 2)
        return {used[0]: t1, used[1]: t2}
    else:
        t1, t2, t3 = random.sample(TRANSFORMS, 3)
        return {used[0]: t1, used[1]: t2, used[2]: t3}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).resolve().parents[2]

    # ── Load real train.csv bit_manipulation problems ─────────────────────────
    bit_ids: set = set()
    with (base / "data" / "problems.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") == "bit_manipulation":
                bit_ids.add(r["id"])

    print("Running solver on real train.csv problems...")
    real_by_rank: Dict[int, List] = {i: [] for i in range(1, 21)}

    with (base / "data" / "train.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] not in bit_ids:
                continue
            parsed = parse(row["prompt"])
            if parsed is None:
                continue
            ins, outs, query_in = parsed
            rank, expr, td, pred = solve(ins, outs, query_in)
            if rank is None or pred != bin8(row["answer"]):
                continue
            cot = make_structured_cot(rank, expr, td, pred)
            real_by_rank[rank].append({
                "id": row["id"], "prompt": row["prompt"],
                "answer": bin8(row["answer"]), "type": "bit_manipulation",
                "generated_cot": cot
            })

    for rank in range(1, 21):
        _, rexpr, _ = REGISTRY[rank - 1]
        print(f"  Rank {rank:2}: {len(real_by_rank[rank]):3} real  — {rexpr}")

    # ── Build balanced dataset ────────────────────────────────────────────────
    print(f"\nBuilding balanced dataset ({N_PER_RULE} per rule)...")
    ev_map = {tt: fn for tt, _, fn in REGISTRY}
    all_rows = []

    for rank, (tt, expr, ev) in enumerate(REGISTRY, 1):
        used      = _used(expr)
        real_pool = real_by_rank[rank]
        random.shuffle(real_pool)

        # Use real problems first
        selected = real_pool[:N_PER_RULE]
        n_synth  = N_PER_RULE - len(selected)

        all_rows.extend(selected)

        # Fill with synthetic
        synth_added = 0
        attempts    = 0
        while synth_added < n_synth and attempts < n_synth * 10:
            attempts += 1
            td = random_td(used)
            pid, prompt, answer, cot = make_synthetic_problem(rank, expr, ev, td)
            all_rows.append({
                "id": pid, "prompt": prompt, "answer": answer,
                "type": "bit_manipulation", "generated_cot": cot
            })
            synth_added += 1

        print(f"  Rank {rank:2}: {len(selected):3} real + {synth_added:3} synthetic = {len(selected)+synth_added}")

    print(f"\nTotal bit_manipulation rows: {len(all_rows)}")

    # ── Write bit_manipulation only first ────────────────────────────────────
    out_path = base / "error_analysis" / "88_0" / "problem_ids_matched_v9.csv"
    fieldnames = ["id", "prompt", "answer", "type", "generated_cot"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Merge non-bit rows from v6 ────────────────────────────────────────────
    print("Merging non-bit_manipulation rows from v6.csv...")
    v6_path = base / "error_analysis" / "88_0" / "problem_ids_matched_v6.csv"
    non_bit = []
    with v6_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["type"] != "bit_manipulation":
                non_bit.append(row)

    merged = all_rows + non_bit
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
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
    print(f"\nOutput: {out_path}")

if __name__ == "__main__":
    main()
