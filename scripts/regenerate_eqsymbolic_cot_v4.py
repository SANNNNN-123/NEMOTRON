"""
Regenerate Equation Symbolic CoT in data/train_split_with_cot_v4.csv using the
cipher-digit template (crack bijection -> scan combo -> apply -> encode).
"""
import pandas as pd
import re
from itertools import permutations

CSV = "data/train_split_with_cot_v4.csv"

PREAMBLE = """I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a cipher-digit template. I know this because I see equations where ALL characters (including digits) have been replaced by random symbols. Each symbol maps to exactly one digit and vice versa (bijective). The cipher is unique to this problem and will never repeat. I know this is NOT roman, unit conversion, gravity, or bare symbol-digit (which uses real digits).

RULE 2: Since I have identified this is a cipher-digit template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \\boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: I see that this is a cipher-digit template. All characters are encrypted symbols. I need to first CRACK the cipher to recover the actual digits, then SCAN for the operation as a normal digit problem, then ENCODE my answer back to cipher symbols. I am now going to fill out the template."""

# --- combo space (same as numeric solver) ---
PAIRINGS = ["AB_CD", "BA_DC", "AB_DC", "BA_CD"]
def pair(p, A, B, C, D):
    if p == "AB_CD": return 10*A+B, 10*C+D
    if p == "BA_DC": return 10*B+A, 10*D+C
    if p == "AB_DC": return 10*A+B, 10*D+C
    if p == "BA_CD": return 10*B+A, 10*C+D

OPS = [
    ("add", lambda L,R: L+R),
    ("sub", lambda L,R: L-R),
    ("mul", lambda L,R: L*R),
    ("cat", lambda L,R: int(f"{L}{R}") if L>=0 and R>=0 else None),
    ("add1", lambda L,R: L+R+1),
    ("addm1", lambda L,R: L+R-1),
    ("muladd1", lambda L,R: L*R+1),
    ("mulsub1", lambda L,R: L*R-1),
    ("absdiff", lambda L,R: abs(L-R)),
    ("xor", lambda L,R: L^R),
    ("maxv", lambda L,R: max(L,R)),
    ("minv", lambda L,R: min(L,R)),
]
OP_DICT = dict(OPS)

def fmt_raw(n): return None if n is None else str(n)  # may include '-'
def fmt_rev(n):
    if n is None: return None
    s=str(n)
    return "-"+s[1:][::-1] if s.startswith("-") else s[::-1]
def fmt_abs(n): return None if n is None else str(abs(n))
def fmt_absrev(n): return None if n is None else str(abs(n))[::-1]
def fmt_neg(n): return None if n is None else "-"+str(abs(n))
def fmt_zpad(w):
    def f(n):
        if n is None or n<0: return None
        return f"{n:0{w}d}"
    return f
def fmt_dsum(n):
    if n is None: return None
    return str(sum(int(c) for c in str(abs(n))))
def fmt_dprod(n):
    if n is None: return None
    p=1
    for c in str(abs(n)): p*=int(c)
    return str(p)

FORMATS = [("raw",fmt_raw),("rev",fmt_rev),("abs",fmt_abs),("absrev",fmt_absrev),("neg",fmt_neg),
           ("zpad2",fmt_zpad(2)),("zpad3",fmt_zpad(3)),("zpad4",fmt_zpad(4)),
           ("dsum",fmt_dsum),("dprod",fmt_dprod)]
FMT_DICT = dict(FORMATS)

FREQ_ORDER = []
for p in PAIRINGS:
    for on,_ in OPS:
        for fn,_ in FORMATS:
            FREQ_ORDER.append((p,on,fn))

def apply_combo(p, on, fn, A, B, C, D):
    L,R = pair(p,A,B,C,D)
    v = OP_DICT[on](L,R)
    if v is None: return None,L,R,None
    return FMT_DICT[fn](v),L,R,v


# --- parse ---
def parse_prompt(prompt):
    """Return list of (LHS, RHS) strings for 5-char LHS lines, plus target LHS."""
    lines = prompt.split("\n")
    examples = []
    for line in lines:
        m = re.match(r'^\s*(\S{5})\s*=\s*(\S+)\s*$', line)
        if m:
            lhs, rhs = m.group(1), m.group(2)
            examples.append((lhs, rhs))
    m = re.search(r'result for[:\s]+(\S{5})', prompt)
    if not m: return None, None
    return examples, m.group(1)


# --- solver ---
def try_combo_on_group(combo, op_examples_enc):
    """
    op_examples_enc: list of (s0,s1,s2,s3, rhs_str) all using same (decoded) operator.
    Returns mapping dict (symbol->digit) if a consistent assignment exists for ALL of them, else None.
    """
    p, on, fn = combo
    # Start with empty mapping; use example 0 to generate candidates
    s0,s1,s2,s3,rhs = op_examples_enc[0]
    uniq = []
    for c in (s0,s1,s2,s3):
        if c not in uniq: uniq.append(c)
    k = len(uniq)
    for digits in permutations(range(10), k):
        m = dict(zip(uniq, digits))
        A,B,C,D = m[s0],m[s1],m[s2],m[s3]
        out,_,_,_ = apply_combo(p, on, fn, A, B, C, D)
        if out is None or len(out) != len(rhs): continue
        # extend mapping with RHS
        mm = dict(m); bad = False
        used_digits = set(mm.values())
        for ch, dch in zip(rhs, out):
            if dch == '-':
                if ch != '-': bad=True; break
                continue
            if ch == '-': bad=True; break
            d = int(dch)
            if ch in mm:
                if mm[ch] != d: bad=True; break
            else:
                if d in used_digits: bad=True; break
                mm[ch] = d; used_digits.add(d)
        if bad: continue
        # verify other examples in group extend consistently
        ok = True
        mm2 = dict(mm)
        for (t0,t1,t2,t3,trhs) in op_examples_enc[1:]:
            used = set(mm2.values())
            # ensure lhs symbols in mm2 or enumerate — to keep it tractable, require lhs symbols already in mm2
            needed = [c for c in (t0,t1,t2,t3) if c not in mm2]
            if needed:
                # try to extend by enumeration
                avail = [d for d in range(10) if d not in used]
                found_ext = None
                uniq_needed = []
                for c in needed:
                    if c not in uniq_needed: uniq_needed.append(c)
                for dperm in permutations(avail, len(uniq_needed)):
                    ext = dict(zip(uniq_needed, dperm))
                    mm3 = dict(mm2); mm3.update(ext)
                    A2,B2,C2,D2 = mm3[t0],mm3[t1],mm3[t2],mm3[t3]
                    out2,_,_,_ = apply_combo(p,on,fn,A2,B2,C2,D2)
                    if out2 is None or len(out2)!=len(trhs): continue
                    ok_rhs = True; used2 = set(mm3.values())
                    for ch,dch in zip(trhs,out2):
                        if dch=='-':
                            if ch!='-': ok_rhs=False;break
                            continue
                        if ch=='-': ok_rhs=False;break
                        d=int(dch)
                        if ch in mm3:
                            if mm3[ch]!=d: ok_rhs=False;break
                        else:
                            if d in used2: ok_rhs=False;break
                            mm3[ch]=d; used2.add(d)
                    if ok_rhs:
                        found_ext = mm3; break
                if found_ext is None: ok=False; break
                mm2 = found_ext
            else:
                A2,B2,C2,D2 = mm2[t0],mm2[t1],mm2[t2],mm2[t3]
                out2,_,_,_ = apply_combo(p,on,fn,A2,B2,C2,D2)
                if out2 is None or len(out2)!=len(trhs): ok=False;break
                used2 = set(mm2.values())
                mm3 = dict(mm2); ok_rhs=True
                for ch,dch in zip(trhs,out2):
                    if dch=='-':
                        if ch!='-': ok_rhs=False;break
                        continue
                    if ch=='-': ok_rhs=False;break
                    d=int(dch)
                    if ch in mm3:
                        if mm3[ch]!=d: ok_rhs=False;break
                    else:
                        if d in used2: ok_rhs=False;break
                        mm3[ch]=d; used2.add(d)
                if not ok_rhs: ok=False;break
                mm2 = mm3
        if ok:
            return mm2
    return None


def solve(prompt, gt_answer):
    examples, target = parse_prompt(prompt)
    if not examples or target is None: return None
    # Operator symbol in LHS is position 2
    op_sym = target[2]
    # Group examples by their operator symbol
    op_examples = []
    for lhs, rhs in examples:
        if len(lhs)!=5: continue
        if lhs[2] == op_sym:
            op_examples.append((lhs[0],lhs[1],lhs[3],lhs[4], rhs))
    if not op_examples: return None
    # Try combos in order
    for combo in FREQ_ORDER:
        mapping = try_combo_on_group(combo, op_examples)
        if mapping is None: continue
        # Apply to target
        tA,tB,tC,tD = target[0],target[1],target[3],target[4]
        # Need target symbols in mapping; extend if missing
        needed = [c for c in (tA,tB,tC,tD) if c not in mapping]
        if needed:
            used = set(mapping.values())
            avail = [d for d in range(10) if d not in used]
            uniq_needed = []
            for c in needed:
                if c not in uniq_needed: uniq_needed.append(c)
            found = None
            for dperm in permutations(avail, len(uniq_needed)):
                ext = dict(mapping); ext.update(dict(zip(uniq_needed, dperm)))
                A,B,C,D = ext[tA],ext[tB],ext[tC],ext[tD]
                out,L,R,v = apply_combo(*combo, A, B, C, D)
                if out is None: continue
                # encode out with ext; must be bijective
                used2 = set(ext.values())
                enc = []; ext2 = dict(ext); ok=True
                inv = {d:s for s,d in ext2.items()}
                for dch in out:
                    if dch=='-': enc.append('-'); continue
                    d=int(dch)
                    if d in inv:
                        enc.append(inv[d])
                    else:
                        ok=False; break
                if not ok: continue
                enc_str = "".join(enc)
                if enc_str == gt_answer:
                    return (combo, ext2, target, op_sym, op_examples, out, enc_str, L, R, v)
            continue
        else:
            A,B,C,D = mapping[tA],mapping[tB],mapping[tC],mapping[tD]
            out,L,R,v = apply_combo(*combo, A, B, C, D)
            if out is None: continue
            inv = {d:s for s,d in mapping.items()}
            enc=[]; ok=True
            for dch in out:
                if dch=='-': enc.append('-'); continue
                d=int(dch)
                if d in inv: enc.append(inv[d])
                else: ok=False; break
            if not ok: continue
            enc_str = "".join(enc)
            if enc_str == gt_answer:
                return (combo, mapping, target, op_sym, op_examples, out, enc_str, L, R, v)
    return None


def build_cot(result):
    combo, mapping, target, op_sym, op_examples, tgt_digits, enc_str, tL, tR, tV = result
    p, on, fn = combo
    lines = [PREAMBLE, ""]
    # S2
    lines.append("S2: DETECT")
    lines.append(f"OP_SYM: {op_sym} (position 2)")
    lines.append(f"SYMS: {len(mapping)} unique digit symbols")
    lines.append("")
    # S3
    lines.append("S3: CRACK")
    map_str = " ".join(f"{s}={d}" for s,d in sorted(mapping.items(), key=lambda x:x[1]))
    lines.append(f"MAP: {map_str}")
    s0,s1,s2,s3,rhs = op_examples[0]
    decoded_lhs = f"{mapping[s0]}{mapping[s1]}{op_sym}{mapping[s2]}{mapping[s3]}"
    decoded_rhs = "".join('-' if c=='-' else str(mapping[c]) for c in rhs)
    lines.append(f"CHK: {s0}{s1}{op_sym}{s2}{s3}={rhs} -> {decoded_lhs}={decoded_rhs}")
    lines.append("")
    # S4 SCAN (show the winning combo)
    lines.append("S4: SCAN")
    A1,B1,C1,D1 = mapping[s0],mapping[s1],mapping[s2],mapping[s3]
    out1,L1,R1,v1 = apply_combo(p,on,fn,A1,B1,C1,D1)
    lines.append(f"LOCK: {p}|{on}|{fn} L={L1},R={R1} {v1}->{out1} vs {decoded_rhs} YES")
    lines.append("")
    # S5 APPLY
    tA,tB,tC,tD = target[0],target[1],target[3],target[4]
    lines.append("S5: APPLY")
    lines.append(f"TGT: {target} -> DIG:{mapping[tA]}{mapping[tB]}{op_sym}{mapping[tC]}{mapping[tD]}")
    lines.append(f"{p}|L={tL},R={tR}|{on}={tV}|{fn}={tgt_digits}")
    lines.append("")
    # S6 ENCODE
    lines.append("S6: ENCODE")
    lines.append(f"RES: {tgt_digits}")
    inv = {d:s for s,d in mapping.items()}
    enc_steps = " ".join(('-->-'if dch=='-' else f"{dch}->{inv[int(dch)]}") for dch in tgt_digits)
    lines.append(f"ENC: {enc_steps}")
    lines.append(f"OUT: {enc_str}")
    lines.append("")
    lines.append(f"S7: ANS={enc_str}")
    lines.append("")
    lines.append(f"\\boxed{{{enc_str}}}")
    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV)
    mask = df['type']=='Equation Symbolic'
    idxs = df[mask].index.tolist()
    print(f"Equation Symbolic rows: {len(idxs)}")
    solved=0; failed=[]
    for i, idx in enumerate(idxs):
        row = df.loc[idx]
        gt = str(row['answer']).strip()
        res = solve(row['prompt'], gt)
        if res is None:
            failed.append(idx); continue
        cot = build_cot(res)
        df.at[idx,'generated_cot'] = cot
        solved += 1
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(idxs)} solved={solved}")
    print(f"Solved: {solved}/{len(idxs)}")
    print(f"Failed: {len(failed)}")
    if failed:
        df = df.drop(index=failed).reset_index(drop=True)
    print(f"Final rows: {len(df)}")
    print(df['type'].value_counts())
    df.to_csv(CSV, index=False)
    print(f"Wrote {CSV}")

if __name__=='__main__':
    main()
