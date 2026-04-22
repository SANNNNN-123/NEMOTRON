"""
Generate CoT for the 248 bit_manipulation problems missing from problem_ids_matched.csv.

Method:
  1. Use pre-generated reasoning.py .txt files (familiar column-hashing format)
  2. Verify answer with our 20-rule solver
  3. Keep if reasoning.py answer == ground truth → use as-is (correct CoT)
  4. If reasoning.py wrong → keep everything BEFORE "Applying to" section,
     then replace the full application section with our solver's correct
     bit-serial computation in the same reasoning.py style.
     No contradictory reasoning → answer mismatch.

Output: error_analysis/88_0/problem_ids_matched_v11_missing248.csv
        (bit_manipulation only — to be merged with v10 for full training)
"""
from __future__ import annotations

import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

NEMOTRON_REPO  = Path("/home/zuhair/Desktop/project/nemotron")
REASONING_DIR  = NEMOTRON_REPO / "reasoning"
BASE           = Path(__file__).resolve().parents[2]
OUT_CSV        = BASE / "error_analysis" / "88_0" / "problem_ids_matched_v11_missing248.csv"

# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else ""

# ── Solver helpers ────────────────────────────────────────────────────────────

def bin8(x: Any) -> str:
    s = "".join(c for c in str(x).strip() if c in "01")
    return (s.zfill(8) if s else "00000000")[-8:]

def bits(s: str) -> List[int]:
    return [int(c) for c in bin8(s)]

TRANSFORMS: List[Tuple[str, int]] = [("rot", 0)]
for _k in range(1, 8):
    TRANSFORMS.extend([("rot", _k), ("shl", _k), ("shr", _k)])

def trans_label(t):
    tp, k = t
    if tp == "rot" and k == 0: return "identity"
    if tp == "rot": return f"rot({k})"
    if tp == "shl": return f"shl({k})"
    return f"shr({k})"

def get_src(in_bits, i, t):
    tp, k = t
    if tp == "rot": return in_bits[(i + k) % 8]
    if tp == "shl":
        idx = i + k; return in_bits[idx] if 0 <= idx < 8 else 0
    idx = i - k; return in_bits[idx] if 0 <= idx < 8 else 0

OPS: Dict[str, Callable] = {
    "AND":         lambda a,b: a&b,   "OR":  lambda a,b: a|b,
    "XOR":         lambda a,b: a^b,   "NAND":lambda a,b: ~(a&b),
    "NOR":         lambda a,b: ~(a|b),"XNOR":lambda a,b: ~(a^b),
    "NOT_A_AND_B": lambda a,b: (~a)&b,"A_AND_NOT_B":lambda a,b: a&(~b),
    "NOT_A_OR_B":  lambda a,b: (~a)|b,"A_OR_NOT_B": lambda a,b: a|(~b),
}

def _grammar():
    mask = 255
    l0 = {
        0:("C0",lambda a,b,c,m:0), 255:("C1",lambda a,b,c,m:m),
        0b11110000:("{A}",lambda a,b,c,m:a),
        0b11001100:("{B}",lambda a,b,c,m:b),
        0b10101010:("{C}",lambda a,b,c,m:c),
    }
    visited=set(l0.keys()); levels=[l0]
    for tt,(expr,func) in l0.items(): yield tt,expr,func
    for depth in range(1,4):
        nxt={}
        for v,(expr,func) in levels[-1].items():
            nv=(~v)&mask
            if nv not in visited:
                ne=f"NOT({expr})"; nf=lambda a,b,c,m,f=func:(~f(a,b,c,m))&m
                visited.add(nv); nxt[nv]=(ne,nf); yield nv,ne,nf
        for i in range(depth):
            j=depth-1
            for v1,(e1,f1) in levels[i].items():
                for v2,(e2,f2) in levels[j].items():
                    for on,op in OPS.items():
                        if i==j and v1>v2 and on in("AND","OR","XOR","NAND","NOR","XNOR"): continue
                        val=op(v1,v2)&mask
                        if val not in visited:
                            ne=f"{on}({e1},{e2})"; nf=lambda a,b,c,m,f1=f1,f2=f2,op=op:op(f1(a,b,c,m),f2(a,b,c,m))&m
                            visited.add(val); nxt[val]=(ne,nf); yield val,ne,nf
                        if i!=j:
                            val2=op(v2,v1)&mask
                            if val2 not in visited:
                                ne2=f"{on}({e2},{e1})"; nf2=lambda a,b,c,m,f1=f1,f2=f2,op=op:op(f2(a,b,c,m),f1(a,b,c,m))&m
                                visited.add(val2); nxt[val2]=(ne2,nf2); yield val2,ne2,nf2
        levels.append(nxt)

def _used(expr): return [v for v in("{A}","{B}","{C}") if v in expr]

TT_FREQ_ORDER = [
    0x3c,0xfc,0x2d,0xf0,0xc0,0x0c,0xd2,0xe0,0x6d,0xf9,
    0xac,0xe8,0xc3,0xcf,0xf2,0x68,0xff,0xd0,0x0d,0x03,
]

def _build_registry():
    tt_map = {}
    for tt,expr,fn in _grammar():
        if tt not in tt_map: tt_map[tt]=(expr,fn)
    return [(tt,*tt_map[tt]) for tt in TT_FREQ_ORDER if tt in tt_map]

REGISTRY = _build_registry()

def _combos(used):
    if len(used)==0: yield {}
    elif len(used)==1:
        for t in TRANSFORMS: yield {used[0]:t}
    elif len(used)==2:
        for t1 in TRANSFORMS:
            for t2 in TRANSFORMS:
                if t1!=t2: yield {used[0]:t1,used[1]:t2}
    else:
        for t1,t2,t3 in itertools.permutations(TRANSFORMS,3):
            yield {used[0]:t1,used[1]:t2,used[2]:t3}

def _eval_all(ev,td,ins,outs,n):
    for b in range(8):
        for ex in range(n):
            a=get_src(ins[ex],b,td.get("{A}",("rot",0)))
            bv=get_src(ins[ex],b,td.get("{B}",("rot",0)))
            cv=get_src(ins[ex],b,td.get("{C}",("rot",0)))
            if ev(a,bv,cv,1)&1!=outs[ex][b]: return False
    return True

def solve(ins, outs, query_in):
    """Returns (pred, expr, td, ev_fn) or (None, None, None, None)."""
    n = len(ins)
    for rank, (tt, expr, ev) in enumerate(REGISTRY, 1):
        used = _used(expr)
        for td in _combos(used):
            if _eval_all(ev, td, ins, outs, n):
                pred = "".join(str(ev(
                    get_src(query_in,b,td.get("{A}",("rot",0))),
                    get_src(query_in,b,td.get("{B}",("rot",0))),
                    get_src(query_in,b,td.get("{C}",("rot",0))),1)&1)
                    for b in range(8))
                return pred, expr, td, ev
    return None, None, None, None

# ── Prompt parser ─────────────────────────────────────────────────────────────

_EX  = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_QRY = re.compile(r"(?:output for:|determine the output for:)\s*([01]{8})", re.I)

def parse(prompt):
    pairs=_EX.findall(prompt); qm=_QRY.search(prompt)
    if not pairs or not qm: return None
    return [bits(a) for a,_ in pairs],[bits(b) for _,b in pairs],bits(qm.group(1))

# ── Recursive bit-serial evaluator (for application section) ─────────────────

OPS_1BIT: Dict[str, Callable] = {
    "AND":         lambda a,b: a&b,
    "OR":          lambda a,b: a|b,
    "XOR":         lambda a,b: a^b,
    "NAND":        lambda a,b: (~(a&b))&1,
    "NOR":         lambda a,b: (~(a|b))&1,
    "XNOR":        lambda a,b: (~(a^b))&1,
    "NOT_A_AND_B": lambda a,b: ((~a)&b)&1,
    "A_AND_NOT_B": lambda a,b: (a&(~b))&1,
    "NOT_A_OR_B":  lambda a,b: ((~a)|b)&1,
    "A_OR_NOT_B":  lambda a,b: (a|(~b))&1,
}

def _idx_display(i: int, t: Tuple) -> str:
    tp, k = t
    if tp == "rot" and k == 0: return f"in[{i}]"
    if tp == "rot": return f"in[{(i+k)%8}]"
    if tp == "shl":
        si = i+k; return f"in[{si}]" if 0 <= si < 8 else "in[OOB]"
    si = i-k; return f"in[{si}]" if 0 <= si < 8 else "in[OOB]"

def _split_args(s: str) -> List[str]:
    depth, parts, cur = 0, [], []
    for c in s:
        if c == "(": depth+=1; cur.append(c)
        elif c == ")": depth-=1; cur.append(c)
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip()); cur=[]
        else: cur.append(c)
    if cur: parts.append("".join(cur).strip())
    return parts

def eval_display(expr: str, td: Dict, in_bits: List[int], i: int) -> Tuple[int, str]:
    """Recursively evaluate and return (value, display_string)."""
    for wire in ("{A}", "{B}", "{C}"):
        if expr == wire:
            v   = get_src(in_bits, i, td[wire])
            ref = _idx_display(i, td[wire])
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

# ── Build correct application section in reasoning.py style ──────────────────

def build_application_section(expr: str, td: Dict, query_in: List[int],
                               correct_answer: str, ev_fn: Callable) -> str:
    """Generate the 'Applying to...' section with solver's correct computation."""
    used = _used(expr)
    q_str = "".join(map(str, query_in))
    lines = []
    lines.append(f"Applying to {q_str}")
    lines.append("Input")
    for i, b in enumerate(query_in):
        lines.append(f"{i} {b}")
    lines.append("Output")
    for i in range(8):
        result, display = eval_display(expr, td, query_in, i)
        lines.append(f"{i} {display}")
    lines.append("")
    lines.append(r"I will now return the answer in \boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{correct_answer}}}")
    return "\n".join(lines)

# ── Patch CoT: keep pre-application section, replace application section ──────

def patch_cot_option_b(cot_text: str, expr: str, td: Dict, query_in: List[int],
                        correct_answer: str, ev_fn: Callable) -> str:
    """
    Keep everything before 'Applying to' in the reasoning.py CoT,
    then replace the full application section with solver's correct computation.
    """
    lines = cot_text.strip().splitlines()

    # Find the line index of "Applying to"
    apply_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Applying to "):
            apply_idx = idx
            break

    if apply_idx is None:
        # No Applying to section found — just fix the boxed answer
        new_lines = []
        for line in lines:
            if r"\boxed{" in line and "The answer in" in line:
                new_lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{correct_answer}}}")
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    # Keep everything before "Applying to"
    pre_section = "\n".join(lines[:apply_idx])

    # Build new correct application section
    new_application = build_application_section(expr, td, query_in,
                                                correct_answer, ev_fn)
    return pre_section + "\n" + new_application

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Get 248 missing IDs
    bit_ids = set()
    with (BASE / "data" / "problems.jsonl").open() as f:
        for line in f:
            r=json.loads(line)
            if r.get("category")=="bit_manipulation": bit_ids.add(r["id"])

    orig_ids = set()
    with (BASE / "data" / "problem_ids_matched.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("type")=="bit_manipulation": orig_ids.add(row["id"])

    missing_ids = sorted(bit_ids - orig_ids)
    print(f"Missing IDs: {len(missing_ids)}")

    # Load ground truth + prompts
    truth={}; prompts={}
    with (BASE / "data" / "train.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] in set(missing_ids):
                truth[row["id"]]   = bin8(row["answer"])
                prompts[row["id"]] = row["prompt"]

    result_rows = []
    stats = {"reasoning_correct":0,"solver_patched":0,"solver_failed":0}

    for pid in missing_ids:
        rfile   = REASONING_DIR / f"{pid}.txt"
        cot_raw = rfile.read_text(encoding="utf-8").strip()
        gt      = truth.get(pid,"")

        # Check reasoning.py answer
        reas_ans = extract_answer(cot_raw)

        if reas_ans == gt:
            # reasoning.py is correct → use as-is (full CoT is correct)
            result_rows.append({
                "id":pid,"prompt":prompts[pid],"answer":gt,
                "type":"bit_manipulation","generated_cot":cot_raw,
                "source":"reasoning_correct"
            })
            stats["reasoning_correct"]+=1
        else:
            # reasoning.py wrong → replace "Applying to" section with solver's
            # correct computation (keeps column-analysis, replaces only application)
            parsed = parse(prompts[pid])
            if parsed:
                ins, outs, query_in = parsed
                solver_ans, expr, td, ev_fn = solve(ins, outs, query_in)
                if solver_ans == gt:
                    patched = patch_cot_option_b(cot_raw, expr, td,
                                                  query_in, gt, ev_fn)
                    result_rows.append({
                        "id":pid,"prompt":prompts[pid],"answer":gt,
                        "type":"bit_manipulation","generated_cot":patched,
                        "source":"solver_patched"
                    })
                    stats["solver_patched"]+=1
                else:
                    stats["solver_failed"]+=1
            else:
                stats["solver_failed"]+=1

    print(f"\nResults:")
    print(f"  reasoning.py correct   : {stats['reasoning_correct']}")
    print(f"  solver patched         : {stats['solver_patched']}")
    print(f"  solver failed (skipped): {stats['solver_failed']}")
    print(f"  Total written          : {len(result_rows)}/248")

    fieldnames=["id","prompt","answer","type","generated_cot","source"]
    with OUT_CSV.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    print(f"\nOutput: {OUT_CSV}")

if __name__=="__main__":
    main()
