"""
Hybrid CoT generator v2: bit-serial computation + 20-rule logic.

Format:
  - "We need to deduce..." opening  (matches base model's learned phrasing)
  - Failed rules: one line each "Rule X/20: <name>  no match"
  - Winning rule: bit-serial verification on ALL examples, one bit at a time
  - Bit-serial application to query
  - Single clean \\boxed{} closing  (no duplicates)

No parallel array math anywhere — forces one-bit-at-a-time reasoning.
Output: error_analysis/88_0/problem_ids_matched_v7_hybrid.csv
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
    if tp == "rot" and k == 0: return "in[i]"
    if tp == "rot":  return f"in[(i+{k}) mod 8]"
    if tp == "shl":  return f"in[i+{k}] (0 if OOB)"
    return f"in[i-{k}] (0 if OOB)"

def get_src(in_bits: List[int], i: int, t: Tuple[str, int]) -> int:
    tp, k = t
    if tp == "rot": return in_bits[(i + k) % 8]
    if tp == "shl":
        idx = i + k; return in_bits[idx] if 0 <= idx < 8 else 0
    idx = i - k; return in_bits[idx] if 0 <= idx < 8 else 0

def idx_display(i: int, t: Tuple[str, int]) -> str:
    tp, k = t
    if tp == "rot" and k == 0: return f"in[{i}]"
    if tp == "rot":  return f"in[{(i+k)%8}]"
    if tp == "shl":
        si = i + k; return f"in[{si}]" if 0 <= si < 8 else "in[OOB]"
    si = i - k; return f"in[{si}]" if 0 <= si < 8 else "in[OOB]"

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

OPS_1BIT: Dict[str, Callable] = {
    "AND":         lambda a, b: a & b,
    "OR":          lambda a, b: a | b,
    "XOR":         lambda a, b: a ^ b,
    "NAND":        lambda a, b: (~(a & b)) & 1,
    "NOR":         lambda a, b: (~(a | b)) & 1,
    "XNOR":        lambda a, b: (~(a ^ b)) & 1,
    "NOT_A_AND_B": lambda a, b: ((~a) & b) & 1,
    "A_AND_NOT_B": lambda a, b: (a & (~b)) & 1,
    "NOT_A_OR_B":  lambda a, b: ((~a) | b) & 1,
    "A_OR_NOT_B":  lambda a, b: (a | (~b)) & 1,
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

# ── Solver ────────────────────────────────────────────────────────────────────

def solve_with_trace(ins, outs, query_in):
    n = len(ins)
    failed_exprs = []

    for rank, (tt, expr, ev) in enumerate(REGISTRY, 1):
        used   = _used(expr)
        win_td = None
        for td in _combos(used):
            if _eval_all(ev, td, ins, outs, n):
                win_td = td; break

        if win_td is not None:
            pred_bits = []
            for b in range(8):
                a  = get_src(query_in, b, win_td.get("{A}", ("rot", 0)))
                bv = get_src(query_in, b, win_td.get("{B}", ("rot", 0)))
                cv = get_src(query_in, b, win_td.get("{C}", ("rot", 0)))
                pred_bits.append(str(ev(a, bv, cv, 1) & 1))
            return rank, tt, expr, win_td, "".join(pred_bits), failed_exprs

        failed_exprs.append((rank, expr))

    return None, None, None, None, None, []

# ── Recursive bit-serial evaluator ───────────────────────────────────────────

def _split_args(s: str) -> List[str]:
    depth, parts, cur = 0, [], []
    for c in s:
        if c == "(":   depth += 1; cur.append(c)
        elif c == ")": depth -= 1; cur.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur).strip())
    return parts

def eval_display(expr: str, td: Dict, in_bits: List[int], i: int) -> Tuple[int, str]:
    """Recursively evaluate and return (value, spelled_out_string)."""
    for wire in ("{A}", "{B}", "{C}"):
        if expr == wire:
            v   = get_src(in_bits, i, td[wire])
            ref = idx_display(i, td[wire])
            return v, f"{ref}={v}"

    if expr == "C0": return 0, "0"
    if expr == "C1": return 1, "1"

    if expr.startswith("NOT(") and expr.endswith(")"):
        iv, is_ = eval_display(expr[4:-1], td, in_bits, i)
        result  = (~iv) & 1
        return result, f"NOT({is_})={result}"

    paren   = expr.index("(")
    op_name = expr[:paren]
    args    = _split_args(expr[paren+1:-1])

    if len(args) == 2:
        v1, s1 = eval_display(args[0], td, in_bits, i)
        v2, s2 = eval_display(args[1], td, in_bits, i)
        result = OPS_1BIT[op_name](v1, v2)
        return result, f"{op_name}({s1},{s2})={result}"

    return 0, "?"

# ── CoT builder ───────────────────────────────────────────────────────────────

def make_cot(ins, outs, query_in, win_rank, win_expr, win_td, pred, win_ev, failed_exprs):
    n    = len(ins)
    used = _used(win_expr)
    L    = []

    # ── Opening ───────────────────────────────────────────────────────────────
    L.append("We need to deduce the transformation by testing 20 known rule types.")
    L.append(r"I will put my final answer inside \boxed{}.")
    L.append("")

    # ── Examples ──────────────────────────────────────────────────────────────
    for idx, (ib, ob) in enumerate(zip(ins, outs)):
        L.append(f"Input {idx}: {''.join(map(str,ib))}  Output {idx}: {''.join(map(str,ob))}")
    L.append(f"Query:   {''.join(map(str,query_in))}")
    L.append("")

    # ── Rule search ───────────────────────────────────────────────────────────
    L.append("Testing rules in frequency order:")
    L.append("")

    # Failed rules — one line each
    for (rank, expr) in failed_exprs:
        L.append(f"  Rule {rank:2}/20: {expr}  no match")

    # Winning rule — bit-serial verification
    td_str = "  ".join(f"{u.strip('{}')}={trans_label(win_td[u])}" for u in used) if used else "constant"
    L.append(f"  Rule {win_rank:2}/20: {win_expr}  [{td_str}]")
    L.append("")

    # Verify on ALL examples, one bit at a time
    for ex_idx, (ex_in, ex_out) in enumerate(zip(ins, outs)):
        in_str  = "".join(map(str, ex_in))
        out_str = "".join(map(str, ex_out))
        L.append(f"    Checking Input {ex_idx} ({in_str}) -> Output {ex_idx} ({out_str}):")
        all_ok = True
        for b in range(8):
            result, display = eval_display(win_expr, win_td, ex_in, b)
            ok = result == ex_out[b]
            if not ok: all_ok = False
            status = "OK" if ok else "FAIL"
            L.append(f"      bit[{b}]: {display}  expected {ex_out[b]}  {status}")
        L.append(f"    {'All bits OK' if all_ok else 'FAIL'}")
        L.append("")

    L.append("  MATCH FOUND")
    L.append("")

    # ── Rule summary ──────────────────────────────────────────────────────────
    rule_label = win_expr
    for u in used:
        rule_label = rule_label.replace(u, trans_label(win_td[u]))
    L.append(f"Rule confirmed: {rule_label}")
    if used:
        L.append("Wire assignments (per output bit i, leftmost bit = 0):")
        for u in used:
            L.append(f"  {u.strip('{}')}: {describe_transform(win_td[u])}")
    L.append("")

    # ── Bit-serial application ────────────────────────────────────────────────
    q_str = "".join(map(str, query_in))
    L.append(f"Applying rule to query {q_str}:")
    out_bits = []
    for i in range(8):
        result, display = eval_display(win_expr, win_td, query_in, i)
        out_bits.append(str(result))
        L.append(f"  bit[{i}]: {display}")

    pred_out = "".join(out_bits)
    L.append(f"Output: {pred_out}")
    L.append("")

    # ── Single clean closing ──────────────────────────────────────────────────
    L.append("</think>")
    L.append(r"\boxed{" + pred_out + r"}")
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

    total    = len(train_rows)
    out_path = base / "error_analysis" / "88_0" / "problem_ids_matched_v7_hybrid.csv"
    ev_map   = {tt: fn for tt, _, fn in REGISTRY}

    print(f"Processing {total} bit_manipulation problems...")
    print(f"Registry: {len(REGISTRY)} rules\n")

    correct = wrong = skipped = no_rule = 0
    rank_counts: Dict[int, int] = {i: 0 for i in range(1, 21)}

    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["id","prompt","answer","type","generated_cot"])
        writer.writeheader()

        for idx, row in enumerate(train_rows):
            pid    = row["id"]
            prompt = row["prompt"]
            answer = bin8(row["answer"])

            parsed = parse(prompt)
            if parsed is None:
                skipped += 1; continue

            ins, outs, query_in = parsed
            win_rank, tt, expr, td, pred, failed = solve_with_trace(ins, outs, query_in)

            if win_rank is None:
                print(f"[{idx+1}/{total}] {pid}  NO RULE")
                no_rule += 1; continue

            if pred != answer:
                print(f"[{idx+1}/{total}] {pid}  WRONG  pred={pred}  ans={answer}")
                wrong += 1; continue

            rank_counts[win_rank] += 1
            correct += 1
            cot = make_cot(ins, outs, query_in, win_rank, expr, td, pred, ev_map[tt], failed)
            writer.writerow({"id": pid, "prompt": prompt, "answer": answer,
                             "type": "bit_manipulation", "generated_cot": cot})

            if (idx + 1) % 200 == 0 or (idx + 1) == total:
                print(f"[{idx+1}/{total}]  correct={correct}  wrong={wrong}  no_rule={no_rule}")

    print(f"\n--- Done ---")
    print(f"Written : {correct}/{total}")
    print(f"Wrong   : {wrong}/{total}")
    print(f"No rule : {no_rule}/{total}")
    print(f"\nWinning rule rank distribution:")
    for rank in range(1, 21):
        if rank_counts[rank]:
            _, rexpr, _ = REGISTRY[rank - 1]
            print(f"  Rank {rank:2}: {rank_counts[rank]:4}  {rexpr}")
    print(f"\nOutput: {out_path}")

if __name__ == "__main__":
    main()
