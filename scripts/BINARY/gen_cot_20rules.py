"""
Generate SFT CoT for ALL 1602 bit_manipulation problems using the 20-rule
systematic template.

For each problem the CoT:
  1. Lists all examples + query.
  2. Tests each of the 20 known rule types in frequency order.
     - Failed rules: shows a concrete attempt with bit-by-bit steps until
       the first failure, so the model learns HOW to verify and reject.
     - Winning rule: full bit-by-bit verification on example 1, brief check
       on remaining examples, then applies to query step-by-step.
  3. Ends with \\boxed{answer}.

Only rows where pred == ground-truth answer are written.
Output: error_analysis/88_0/problem_ids_matched_v6_20rules.csv
"""
from __future__ import annotations

import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── 8-bit helpers ─────────────────────────────────────────────────────────────

def bin8(x: Any) -> str:
    s = "".join(c for c in str(x).strip() if c in "01")
    return (s.zfill(8) if s else "00000000")[-8:]

def bits(s: str) -> List[int]:
    return [int(c) for c in bin8(s)]

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

def describe_transform(t: Tuple[str, int]) -> str:
    tp, k = t
    if tp == "rot" and k == 0: return "in[i]  (no shift)"
    if tp == "rot":  return f"in[(i+{k}) mod 8]  (rotate right {k})"
    if tp == "shl":  return f"in[i+{k}]  (shift left {k}; 0 if out of range)"
    return f"in[i-{k}]  (shift right {k}; 0 if out of range)"

def get_src(in_bits: List[int], i: int, t: Tuple[str, int]) -> int:
    tp, k = t
    if tp == "rot": return in_bits[(i + k) % 8]
    if tp == "shl":
        idx = i + k; return in_bits[idx] if 0 <= idx < 8 else 0
    idx = i - k; return in_bits[idx] if 0 <= idx < 8 else 0

def _wire_val_str(used: List[str], td: Dict, in_bits: List[int], i: int) -> str:
    parts = []
    for u in used:
        tp, k = td[u]
        v = get_src(in_bits, i, td[u])
        if tp == "rot" and k == 0:
            parts.append(f"{u.strip('{}')}=in[{i}]={v}")
        elif tp == "rot":
            parts.append(f"{u.strip('{}')}=in[{(i+k)%8}]={v}")
        elif tp == "shl":
            si = i + k
            parts.append(f"{u.strip('{}')}=in[{si}]={v}" if 0 <= si < 8 else f"{u.strip('{}')}=OOB=0")
        else:
            si = i - k
            parts.append(f"{u.strip('{}')}=in[{si}]={v}" if 0 <= si < 8 else f"{u.strip('{}')}=OOB=0")
    return "  ".join(parts)

# ── Operations ────────────────────────────────────────────────────────────────

OPS: Dict[str, Callable[[int, int], int]] = {
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

# ── Registry (20 rules in frequency order) ───────────────────────────────────

TT_FREQ_ORDER = [
    0x3c, 0xfc, 0x2d, 0xf0, 0xc0, 0x0c, 0xd2, 0xe0, 0x6d, 0xf9,
    0xac, 0xe8, 0xc3, 0xcf, 0xf2, 0x68, 0xff, 0xd0, 0x0d, 0x03,
]

def _build_registry() -> List[Tuple[int, str, Callable]]:
    tt_map: Dict[int, Tuple[str, Callable]] = {}
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

# ── Best-attempt finder for failed rules ──────────────────────────────────────

def _find_demo_attempt(
    ev: Callable, used: List[str],
    ins: List[List[int]], outs: List[List[int]], n: int
) -> Tuple[Dict, int, int]:
    """
    Find the transform combo that passes the most consecutive bits on example 1
    (scanning only the first 22 combos for speed). Returns (td, bits_passed, fail_bit).
    fail_bit = -1 means failure was on a later example (all 8 bits of ex1 matched).
    """
    best_td    = next(_combos(used))  # fallback
    best_bits  = -1
    best_fail  = 0

    checked = 0
    for td in _combos(used):
        if checked >= 22:   # cap scan at 22 combos — fast enough
            break
        checked += 1

        # Count how many bits pass on example 1
        bits_ok = 0
        fail_bit = -1
        for b in range(8):
            a  = get_src(ins[0], b, td.get("{A}", ("rot", 0)))
            bv = get_src(ins[0], b, td.get("{B}", ("rot", 0)))
            cv = get_src(ins[0], b, td.get("{C}", ("rot", 0)))
            got = ev(a, bv, cv, 1) & 1
            if got != outs[0][b]:
                fail_bit = b
                break
            bits_ok += 1

        if bits_ok > best_bits:
            best_bits = bits_ok
            best_fail = fail_bit
            best_td   = td
        if best_bits == 8:   # can't do better on example 1
            break

    return best_td, max(best_bits, 0), best_fail

# ── Solver with trace ─────────────────────────────────────────────────────────

def solve_20_with_trace(ins, outs, query_in):
    """
    Returns (win_rank, tt, expr, td, pred, failed_list).
    failed_list: [(rank, expr, ev, demo_td, bits_passed, fail_bit), ...] for each
                 rule checked before the winner.
    """
    n = len(ins)
    failed: List[Tuple] = []

    for rank, (tt, expr, ev) in enumerate(REGISTRY, 1):
        used = _used(expr)
        win_td = None

        for td in _combos(used):
            if _eval_all(ev, td, ins, outs, n):
                win_td = td
                break

        if win_td is not None:
            pred_bits = []
            for b in range(8):
                a  = get_src(query_in, b, win_td.get("{A}", ("rot", 0)))
                bv = get_src(query_in, b, win_td.get("{B}", ("rot", 0)))
                cv = get_src(query_in, b, win_td.get("{C}", ("rot", 0)))
                pred_bits.append(str(ev(a, bv, cv, 1) & 1))
            return rank, tt, expr, win_td, "".join(pred_bits), failed

        # Rule failed — find a demo attempt for the CoT
        demo_td, bits_passed, fail_bit = _find_demo_attempt(ev, used, ins, outs, n)
        failed.append((rank, expr, ev, demo_td, bits_passed, fail_bit))

    return None, None, None, None, None, failed

# ── CoT builder ───────────────────────────────────────────────────────────────

def make_cot(
    ins: List[List[int]], outs: List[List[int]], query_in: List[int],
    win_rank: int, win_expr: str, win_td: Dict, pred: str, win_ev: Callable,
    failed: List[Tuple],
) -> str:
    n   = len(ins)
    win_used = _used(win_expr)
    L = []

    # ── Header ────────────────────────────────────────────────────────────────
    L.append("I will solve this bit manipulation problem by testing 20 known rule types in frequency order.")
    L.append("")

    # ── Examples ──────────────────────────────────────────────────────────────
    L.append("Examples (input -> output):")
    for ib, ob in zip(ins, outs):
        L.append(f"  {''.join(map(str,ib))} -> {''.join(map(str,ob))}")
    L.append(f"Query: {''.join(map(str,query_in))}")
    L.append("")

    # ── Rule search ───────────────────────────────────────────────────────────
    L.append("Searching for the rule:")

    # Failed rules — show a concrete counter-example for each
    for (rank, expr, ev, demo_td, bits_passed, fail_bit) in failed:
        used = _used(expr)
        wire_count = len(used)
        wires_str  = f"({wire_count}w)" if wire_count else "(const)"

        # Build transform label for this attempt
        if used:
            td_label = "  ".join(f"{u.strip('{}')}={trans_label(demo_td[u])}" for u in used)
        else:
            td_label = "no wires"

        L.append("")
        L.append(f"  Rule {rank:2}/20 {wires_str}: {expr}")
        L.append(f"    Try {td_label}:")

        if bits_passed == 0 and fail_bit == 0:
            # Fails immediately at bit[0] of example 1
            a  = get_src(ins[0], 0, demo_td.get("{A}", ("rot", 0)))
            bv = get_src(ins[0], 0, demo_td.get("{B}", ("rot", 0)))
            cv = get_src(ins[0], 0, demo_td.get("{C}", ("rot", 0)))
            got = ev(a, bv, cv, 1) & 1
            ws  = _wire_val_str(used, demo_td, ins[0], 0)
            L.append(f"      bit[0]: {ws}  -> {got}  expected {outs[0][0]}  WRONG")
            L.append(f"    Fails at bit[0] on example 1 — no transforms fit. SKIP")
        else:
            # Show passing bits then first failure (cap at 8)
            ex1_in, ex1_out = ins[0], outs[0]
            for b in range(min(bits_passed + 1, 8)):
                a  = get_src(ex1_in, b, demo_td.get("{A}", ("rot", 0)))
                bv = get_src(ex1_in, b, demo_td.get("{B}", ("rot", 0)))
                cv = get_src(ex1_in, b, demo_td.get("{C}", ("rot", 0)))
                got = ev(a, bv, cv, 1) & 1
                ok  = (got == ex1_out[b])
                ws  = _wire_val_str(used, demo_td, ex1_in, b)
                tick = "CORRECT" if ok else "WRONG"
                L.append(f"      bit[{b}]: {ws}  -> {got}  expected {ex1_out[b]}  {tick}")
                if not ok:
                    break

            if fail_bit == -1:
                L.append(f"    Example 1 passes, but fails on another example — no transforms fit. SKIP")
            else:
                L.append(f"    Fails at bit[{fail_bit}] — no transforms fit this rule. SKIP")

    # Winner rule
    used = win_used
    wire_count = len(used)
    wires_str  = f"({wire_count}w)" if wire_count else "(const)"
    td_label   = "  ".join(f"{u.strip('{}')}={trans_label(win_td[u])}" for u in used) if used else "no wires"

    L.append("")
    L.append(f"  Rule {win_rank:2}/20 {wires_str}: {win_expr}")
    L.append(f"    Try {td_label}:")

    # Verify on ALL examples (brief per example)
    all_ex_ok = True
    for ex_idx, (ex_in, ex_out) in enumerate(zip(ins, outs)):
        ex_pass = all(
            (win_ev(
                get_src(ex_in, b, win_td.get("{A}", ("rot", 0))),
                get_src(ex_in, b, win_td.get("{B}", ("rot", 0))),
                get_src(ex_in, b, win_td.get("{C}", ("rot", 0))),
                1) & 1) == ex_out[b]
            for b in range(8)
        )
        if ex_idx == 0:
            # Show full bit-by-bit for example 1
            L.append(f"      Example 1 ({''.join(map(str,ex_in))} -> {''.join(map(str,ex_out))}):")
            for b in range(8):
                a  = get_src(ex_in, b, win_td.get("{A}", ("rot", 0)))
                bv = get_src(ex_in, b, win_td.get("{B}", ("rot", 0)))
                cv = get_src(ex_in, b, win_td.get("{C}", ("rot", 0)))
                got = win_ev(a, bv, cv, 1) & 1
                ws  = _wire_val_str(used, win_td, ex_in, b)
                L.append(f"        bit[{b}]: {ws}  -> {got}  {'CORRECT' if got==ex_out[b] else 'WRONG'}")
        else:
            tick = "CORRECT" if ex_pass else "WRONG"
            L.append(f"      Example {ex_idx+1} ({''.join(map(str,ex_in))} -> {''.join(map(str,ex_out))}): all bits {tick}")
        if not ex_pass:
            all_ex_ok = False

    L.append(f"    {'All examples match. MATCH FOUND' if all_ex_ok else 'MISMATCH — should not happen'}")

    # ── Rule summary ──────────────────────────────────────────────────────────
    L.append("")
    rule_label = win_expr
    for u in used:
        rule_label = rule_label.replace(u, trans_label(win_td[u]))
    L.append(f"Rule confirmed: {rule_label}")
    if used:
        L.append("Wire assignments (per output bit position i, leftmost=0):")
        for u in used:
            L.append(f"  {u.strip('{}')}: {describe_transform(win_td[u])}")
    L.append("")

    # ── Apply to query ────────────────────────────────────────────────────────
    q_str = "".join(map(str, query_in))
    L.append(f"Applying rule to query: {q_str}")
    out_bits = []
    for i in range(8):
        a  = get_src(query_in, i, win_td.get("{A}", ("rot", 0)))
        bv = get_src(query_in, i, win_td.get("{B}", ("rot", 0)))
        cv = get_src(query_in, i, win_td.get("{C}", ("rot", 0)))
        result = win_ev(a, bv, cv, 1) & 1
        out_bits.append(str(result))
        ws = _wire_val_str(used, win_td, query_in, i)
        L.append(f"  bit[{i}]: {ws}  -> {result}")

    pred_out = "".join(out_bits)
    L.append(f"Output: {pred_out}")
    L.append("")
    L.append(f"\\boxed{{{pred_out}}}")
    return "\n".join(L)

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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base = Path(__file__).resolve().parents[2]

    bit_ids: set = set()
    with (base / "data" / "problems.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") == "bit_manipulation":
                bit_ids.add(r["id"])

    train_rows = []
    with (base / "data" / "train.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] in bit_ids:
                train_rows.append(row)

    total = len(train_rows)
    print(f"Processing {total} bit_manipulation problems...")
    print(f"Registry: {len(REGISTRY)} rules loaded\n")

    out_path   = base / "error_analysis" / "88_0" / "problem_ids_matched_v6_20rules.csv"
    fieldnames = ["id", "prompt", "answer", "type", "generated_cot"]
    ev_map     = {tt: fn for tt, _, fn in REGISTRY}

    correct = wrong = skipped = no_rule = 0
    rank_counts: Dict[int, int] = {i: 0 for i in range(1, 21)}

    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(train_rows):
            pid    = row["id"]
            prompt = row["prompt"]
            answer = bin8(row["answer"])

            parsed = parse(prompt)
            if parsed is None:
                skipped += 1; continue

            ins, outs, query_in = parsed
            win_rank, tt, expr, td, pred, failed = solve_20_with_trace(ins, outs, query_in)

            if win_rank is None:
                print(f"[{idx+1}/{total}] {pid}  NO RULE FOUND")
                no_rule += 1; continue

            if pred != answer:
                print(f"[{idx+1}/{total}] {pid}  WRONG  pred={pred}  ans={answer}  rank={win_rank}")
                wrong += 1; continue

            rank_counts[win_rank] += 1
            correct += 1
            cot = make_cot(ins, outs, query_in, win_rank, expr, td, pred,
                           ev_map[tt], failed)
            writer.writerow({
                "id": pid, "prompt": prompt, "answer": answer,
                "type": "bit_manipulation", "generated_cot": cot,
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                print(f"[{idx+1}/{total}]  correct={correct}  wrong={wrong}  no_rule={no_rule}")

    print(f"\n--- Done ---")
    print(f"Written : {correct}/{total}")
    print(f"Wrong   : {wrong}/{total}")
    print(f"No rule : {no_rule}/{total}")
    print(f"Skipped : {skipped}/{total}")
    print(f"\nWinning rule rank distribution:")
    for rank in range(1, 21):
        if rank_counts[rank]:
            _, rexpr, _ = REGISTRY[rank - 1]
            print(f"  Rank {rank:2}: {rank_counts[rank]:4}  {rexpr}")
    print(f"\nOutput: {out_path}")

if __name__ == "__main__":
    main()
