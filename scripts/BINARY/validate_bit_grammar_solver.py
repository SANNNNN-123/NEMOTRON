#!/usr/bin/env python3
"""
Standalone validator for the dynamic grammar bit-manipulation solver.
Does not import any other solver modules from this repository.

Ensures all reported inputs/outputs are exactly 8-character binary strings
(e.g. "01010101" -> "00010111"), never short forms like "1011" or "1".
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- 8-bit string contract -------------------------------------------------

_BIN8_RE = re.compile(r"^[01]{8}$")


def bin8_normalize(raw: Any) -> str:
    """
    Reduce any model/solver output to exactly 8 binary digits.
    - Strips whitespace; keeps only '0' and '1'.
    - Shorter than 8: left-pad with '0'.
    - Longer than 8: keep the last 8 bits (right-aligned), consistent with zfill semantics on fixed-width fields.
    - Empty after filtering: "00000000".
    """
    s = "".join(ch for ch in str(raw).strip() if ch in "01")
    if not s:
        return "00000000"
    if len(s) <= 8:
        return s.zfill(8)
    return s[-8:]


def assert_bin8(label: str, value: str) -> str:
    out = bin8_normalize(value)
    if not _BIN8_RE.match(out):
        raise ValueError(f"{label}: expected 8 binary chars after normalize, got {out!r}")
    return out


# --- Operations & transforms (solver core) ---------------------------------

OPS: Dict[str, Callable[[int, int], int]] = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "NAND": lambda a, b: ~(a & b),
    "NOR": lambda a, b: ~(a | b),
    "XNOR": lambda a, b: ~(a ^ b),
    "NOT_A_AND_B": lambda a, b: (~a) & b,
    "A_AND_NOT_B": lambda a, b: a & (~b),
    "NOT_A_OR_B": lambda a, b: (~a) | b,
    "A_OR_NOT_B": lambda a, b: a | (~b),
}

TRANSFORMATIONS: List[Tuple[str, int]] = [("rot", 0)]
for k in range(1, 8):
    TRANSFORMATIONS.extend([("rot", k), ("shl", k), ("shr", k)])


def get_used_vars(expr: str) -> List[str]:
    used: List[str] = []
    if "{A}" in expr:
        used.append("{A}")
    if "{B}" in expr:
        used.append("{B}")
    if "{C}" in expr:
        used.append("{C}")
    return used


def get_source_bit(in_bits: List[int], out_idx: int, trans: Tuple[str, int]) -> int:
    ttype, shift_val = trans
    if ttype == "rot":
        return in_bits[(out_idx + shift_val) % 8]
    if ttype == "shl":
        src = out_idx + shift_val
        return in_bits[src] if 0 <= src < 8 else 0
    if ttype == "shr":
        src = out_idx - shift_val
        return in_bits[src] if 0 <= src < 8 else 0
    raise ValueError(f"unknown transform {trans!r}")


def evaluate_bit(
    evaluator: Callable[..., int],
    trans_dict: Dict[str, Tuple[str, int]],
    bit_idx: int,
    in_arrays: List[List[int]],
    out_arrays: List[List[int]],
    num_examples: int,
) -> bool:
    for ex in range(num_examples):
        in_bits = in_arrays[ex]
        expected = out_arrays[ex][bit_idx]
        a_val = get_source_bit(in_bits, bit_idx, trans_dict.get("{A}", ("rot", 0)))
        b_val = get_source_bit(in_bits, bit_idx, trans_dict.get("{B}", ("rot", 0)))
        c_val = get_source_bit(in_bits, bit_idx, trans_dict.get("{C}", ("rot", 0)))
        res = evaluator(a_val, b_val, c_val, 1) & 1
        if res != expected:
            return False
    return True


def generate_grammar_dynamically():
    mask = 255
    l0 = {
        0: ("C0", lambda a, b, c, m: 0),
        255: ("C1", lambda a, b, c, m: m),
        0b11110000: ("{A}", lambda a, b, c, m: a),
        0b11001100: ("{B}", lambda a, b, c, m: b),
        0b10101010: ("{C}", lambda a, b, c, m: c),
    }
    visited = set(l0.keys())
    levels: List[Dict[int, Tuple[str, Callable[..., int]]]] = [l0]

    for tt, (expr, func) in l0.items():
        yield tt, expr, func

    for depth in range(1, 4):
        next_level: Dict[int, Tuple[str, Callable[..., int]]] = {}

        for v, (expr, func) in levels[-1].items():
            not_v = (~v) & mask
            if not_v not in visited:
                new_expr = f"NOT({expr})"
                new_func = lambda a, b, c, m, f=func: (~f(a, b, c, m)) & m
                visited.add(not_v)
                next_level[not_v] = (new_expr, new_func)
                yield not_v, new_expr, new_func

        for i in range(depth):
            j = depth - 1
            for v1, (expr1, func1) in levels[i].items():
                for v2, (expr2, func2) in levels[j].items():
                    for op_name, op_func in OPS.items():
                        if i == j and v1 > v2 and op_name in (
                            "AND",
                            "OR",
                            "XOR",
                            "NAND",
                            "NOR",
                            "XNOR",
                        ):
                            continue

                        val = op_func(v1, v2) & mask
                        if val not in visited:
                            new_expr = f"{op_name}({expr1}, {expr2})"
                            new_func = lambda a, b, c, m, f1=func1, f2=func2, op=op_func: (
                                op(f1(a, b, c, m), f2(a, b, c, m)) & m
                            )
                            visited.add(val)
                            next_level[val] = (new_expr, new_func)
                            yield val, new_expr, new_func

                        if i != j:
                            val2 = op_func(v2, v1) & mask
                            if val2 not in visited:
                                new_expr = f"{op_name}({expr2}, {expr1})"
                                new_func = lambda a, b, c, m, f1=func1, f2=func2, op=op_func: (
                                    op(f2(a, b, c, m), f1(a, b, c, m)) & m
                                )
                                visited.add(val2)
                                next_level[val2] = (new_expr, new_func)
                                yield val2, new_expr, new_func

        levels.append(next_level)


def format_hyp(expr: str, trans_dict: Dict[str, Tuple[str, int]]) -> str:
    s = expr
    for k, v in trans_dict.items():
        s = s.replace(k, str(v))
    return s


def solve_dfs_trace_dynamic(
    in_arrays: List[List[int]],
    out_arrays: List[List[int]],
    num_examples: int,
    time_budget_s: float = 5.0,
) -> Tuple[List[str], Optional[Callable[[List[int]], str]]]:
    trace: List[str] = []
    start_time = time.time()
    grammar_gen = generate_grammar_dynamically()

    for _tt, expr, evaluator in grammar_gen:
        if time.time() - start_time > time_budget_s:
            trace.append("TIMEOUT")
            return trace, None

        used = get_used_vars(expr)
        combinations: List[Dict[str, Tuple[str, int]]] = []
        if len(used) == 0:
            combinations.append({})
        elif len(used) == 1:
            for t1 in TRANSFORMATIONS:
                combinations.append({used[0]: t1})
        elif len(used) == 2:
            for t1 in TRANSFORMATIONS:
                for t2 in TRANSFORMATIONS:
                    if t1 == t2:
                        continue
                    combinations.append({used[0]: t1, used[1]: t2})
        elif len(used) == 3:
            for t1, t2, t3 in itertools.permutations(TRANSFORMATIONS, 3):
                combinations.append({used[0]: t1, used[1]: t2, used[2]: t3})

        for trans_dict in combinations:
            if time.time() - start_time > time_budget_s:
                trace.append("TIMEOUT")
                return trace, None

            if not evaluate_bit(evaluator, trans_dict, 0, in_arrays, out_arrays, num_examples):
                continue

            hyp_str = format_hyp(expr, trans_dict)
            trace.append(f"B0: Testing {hyp_str} -> YES")

            valid_global = True
            for b in range(1, 8):
                if evaluate_bit(evaluator, trans_dict, b, in_arrays, out_arrays, num_examples):
                    trace.append(f"B{b}: Testing {hyp_str} -> YES")
                else:
                    trace.append(f"B{b}: Testing {hyp_str} -> NO. Contradiction, backtracking...")
                    valid_global = False
                    break

            if valid_global:
                trace.append(f"GLOBAL MATCH FOUND: {hyp_str}")

                def predictor(
                    q_in: List[int],
                    trans_dict: Dict[str, Tuple[str, int]] = dict(trans_dict),
                    ev: Callable[..., int] = evaluator,
                ) -> str:
                    bits: List[str] = []
                    for b_idx in range(8):
                        av = get_source_bit(q_in, b_idx, trans_dict.get("{A}", ("rot", 0)))
                        bv = get_source_bit(q_in, b_idx, trans_dict.get("{B}", ("rot", 0)))
                        cv = get_source_bit(q_in, b_idx, trans_dict.get("{C}", ("rot", 0)))
                        bits.append(str(int(ev(av, bv, cv, 1) & 1)))
                    out = "".join(bits)
                    return assert_bin8("predictor", out)

                return trace, predictor

    trace.append("NO MATCH FOUND")
    return trace, None


# --- Prompt parsing ----------------------------------------------------------

_EX_PAIR_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_QUERY_RE = re.compile(r"(?:output for:|determine the output for:)\s*([01]{8})", re.I)


def parse_prompt_bit_task(prompt: str) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    ex_matches = _EX_PAIR_RE.findall(prompt)
    if not ex_matches:
        raise ValueError("no example pairs found in prompt")
    in_arrays: List[List[int]] = []
    out_arrays: List[List[int]] = []
    for a, b in ex_matches:
        in_arrays.append([int(a[j]) for j in range(8)])
        out_arrays.append([int(b[j]) for j in range(8)])
    qm = _QUERY_RE.search(prompt)
    if not qm:
        raise ValueError("no query line found in prompt")
    q = qm.group(1)
    query_in = [int(q[j]) for j in range(8)]
    return in_arrays, out_arrays, query_in


# --- Data loading ------------------------------------------------------------

def load_bit_ids(problems_jsonl: Path) -> set:
    ids: set = set()
    with problems_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("category") == "bit_manipulation":
                ids.add(rec["id"])
    return ids


def load_ids_file(path: Path) -> set:
    """One problem id per line; blank lines and # comments ignored."""
    ids: set = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def iter_train_rows(train_csv: Path, id_filter: Optional[set]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with train_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if id_filter is not None and row["id"] not in id_filter:
                continue
            rows.append(row)
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    default_train = root / "data" / "train.csv"
    default_prob = root / "data" / "problems.jsonl"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        help="CSV with columns id, prompt, answer (default: data/train.csv)",
    )
    ap.add_argument(
        "--problems-jsonl",
        type=Path,
        default=default_prob,
        help="For --bit-only: filter ids with category bit_manipulation",
    )
    ap.add_argument(
        "--bit-only",
        action="store_true",
        help="Only rows whose id has category bit_manipulation in problems.jsonl",
    )
    ap.add_argument(
        "--ids-file",
        type=Path,
        metavar="PATH",
        help="Restrict to these problem ids (one hex id per line; # comments ok). "
        "Combined with --bit-only as intersection.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=30,
        metavar="N",
        help="Max rows to evaluate (default: 30). Use 0 for all rows after filters (e.g. all --bit-only).",
    )
    ap.add_argument("--timeout", type=float, default=5.0, help="Seconds per problem")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Verbose output: sample headers, solver trace (abbreviated unless --trace)",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="With --debug only: print every trace line (no '...' shortening). Ignored without --debug.",
    )
    args = ap.parse_args()

    train_path = args.csv or default_train
    if not train_path.is_file():
        print(f"Missing CSV: {train_path}", file=sys.stderr)
        return 2

    explicit: Optional[set] = None
    if args.ids_file:
        if not args.ids_file.is_file():
            print(f"Missing --ids-file: {args.ids_file}", file=sys.stderr)
            return 2
        explicit = load_ids_file(args.ids_file)

    id_filter: Optional[set] = None
    if args.bit_only:
        if not args.problems_jsonl.is_file():
            print(f"Missing problems.jsonl: {args.problems_jsonl}", file=sys.stderr)
            return 2
        id_filter = load_bit_ids(args.problems_jsonl)
        if explicit is not None:
            id_filter &= explicit
    elif explicit is not None:
        id_filter = explicit

    rows = iter_train_rows(train_path, id_filter)
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        print("No rows to process.", file=sys.stderr)
        return 1

    n = len(rows)
    print(f"samples: {n}")

    num_found = 0
    num_correct = 0
    norm_fail = Counter()
    running_correct = 0

    def progress_suffix(pos: int) -> str:
        """pos = 1-based index in this run; rate = correct so far among first pos rows."""
        pct = (100.0 * running_correct / pos) if pos else 0.0
        return f"  {pos}/{n} ({pct:.2f}%)"

    for idx, row in enumerate(rows):
        pid = row["id"]
        prompt = row["prompt"]
        answer_raw = str(row.get("answer", "")).strip()
        try:
            answer = assert_bin8("ground_truth", answer_raw)
        except ValueError as e:
            pos = idx + 1
            if args.debug:
                print(f"\n[{idx}] id={pid} SKIP bad answer: {e}")
            else:
                print(
                    f"id={pid}  pred=—  ans=—  correct=—  skip=bad_answer ({e})"
                    f"{progress_suffix(pos)}"
                )
            norm_fail["bad_answer"] += 1
            continue

        try:
            in_arrays, out_arrays, query_in = parse_prompt_bit_task(prompt)
        except ValueError as e:
            pos = idx + 1
            if args.debug:
                print(f"\n[{idx}] id={pid} SKIP parse: {e}")
            else:
                print(
                    f"id={pid}  pred=—  ans={answer}  correct=—  skip=parse ({e})"
                    f"{progress_suffix(pos)}"
                )
            norm_fail["parse"] += 1
            continue

        num_examples = len(in_arrays)
        trace, predictor = solve_dfs_trace_dynamic(
            in_arrays, out_arrays, num_examples, time_budget_s=args.timeout
        )

        if args.debug:
            print(f"\nSample {idx} (ID: {pid}):")
            if args.trace:
                for line in trace:
                    print(line)
            elif len(trace) > 10:
                print("\n".join(trace[:5]))
                print("...")
                print("\n".join(trace[-5:]))
            else:
                print("\n".join(trace))

        pos = idx + 1
        if predictor is None:
            if args.debug:
                print(f"[{idx}] id={pid} FAILED TO FIND RULE")
            else:
                if trace and trace[-1] == "TIMEOUT":
                    print(f"id={pid}  status=timeout{progress_suffix(pos)}")
                else:
                    print(
                        f"id={pid}  pred=—  ans={answer}  correct=—  rule_found=no"
                        f"{progress_suffix(pos)}"
                    )
            continue

        num_found += 1
        pred_str = predictor(query_in)
        pred_str = assert_bin8("prediction", pred_str)
        ok = pred_str == answer
        if ok:
            num_correct += 1
            running_correct += 1
        if args.debug:
            print(f"Pred: {pred_str} | Ans: {answer} | Correct: {ok}")
        else:
            yn = "yes" if ok else "no"
            print(
                f"id={pid}  pred={pred_str}  ans={answer}  correct={yn}"
                f"{progress_suffix(pos)}"
            )

    skipped = sum(norm_fail.values())
    attempted = n - skipped
    pct = (100.0 * num_correct / n) if n else 0.0
    wrong = n - num_correct
    pct_wrong = (100.0 * wrong / n) if n else 0.0
    print(f"\n---")
    print(f"correct: {num_correct}/{n} ({pct:.1f}%)")
    print(f"wrong:   {wrong}/{n} ({pct_wrong:.1f}%)")
    print(f"rules_found: {num_found}/{n}")
    if attempted != n:
        print(f"attempted_solver: {attempted}/{n}  skipped: {skipped}")
    if norm_fail:
        print(f"skip_reasons: {dict(norm_fail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
