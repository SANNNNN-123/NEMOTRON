"""
Regenerate Equation Numeric CoT in data/train_split_with_cot_v4.csv using the
exact template format from the user's example (S2 PARSE / S3 SCAN / S4 APPLY / S5 ANS).

Reuses combo space from regenerate_symboldigit_cot.py but emits a tighter trace.
"""
import pandas as pd
import re

CSV = "data/train_split_with_cot_v4.csv"

PREAMBLE = """I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a symbol-digit template. I know this because I see equations with two-digit pairs separated by an operator symbol, and I need to figure out what transformation the operator performs. I know this is NOT roman, unit conversion, gravity, binary, or any other conversion type.

RULE 2: Since I have identified this is a symbol-digit template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \\boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: I see that this is a symbol-digit template. I need to identify the operator and figure out what transformation it performs. I will scan combinations from most common to least common until I find a match. I am now going to fill out the template."""

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
    ("orsum", lambda L,R: L|R),
    ("andsum", lambda L,R: L&R),
    ("maxv", lambda L,R: max(L,R)),
    ("minv", lambda L,R: min(L,R)),
    ("div", lambda L,R: L//R if R!=0 else None),
    ("mod", lambda L,R: L%R if R!=0 else None),
]
OP_DICT = dict(OPS)

def fmt_raw(n): return None if n is None else str(n)
def fmt_rev(n):
    if n is None: return None
    s=str(n)
    return "-"+s[1:][::-1] if s.startswith("-") else s[::-1]
def fmt_abs(n): return None if n is None else str(abs(n))
def fmt_neg(n): return None if n is None else "-"+str(abs(n))
def fmt_zpad(w):
    def f(n):
        if n is None or n<0: return None
        return f"{n:0{w}d}"
    return f
def fmt_dsum(n): return None if n is None else str(sum(int(c) for c in str(abs(n))))

FORMATS = [("raw",fmt_raw),("rev",fmt_rev),("abs",fmt_abs),("neg",fmt_neg),
           ("zpad2",fmt_zpad(2)),("zpad3",fmt_zpad(3)),("zpad4",fmt_zpad(4)),("dsum",fmt_dsum)]
FMT_DICT = dict(FORMATS)

FREQ_ORDER = [
    ("AB_CD","cat","raw"),("AB_CD","mul","raw"),("AB_CD","add","raw"),
    ("AB_CD","sub","raw"),("AB_CD","absdiff","raw"),
    ("BA_DC","cat","rev"),("BA_DC","mul","rev"),("BA_DC","add","rev"),
    ("BA_DC","cat","raw"),("AB_CD","cat","rev"),("AB_CD","mul","rev"),
    ("AB_CD","add","rev"),("AB_CD","sub","neg"),("AB_CD","sub","rev"),
    ("BA_DC","sub","rev"),("AB_CD","add1","raw"),("AB_CD","addm1","raw"),
    ("AB_CD","muladd1","raw"),("AB_CD","mulsub1","raw"),("AB_CD","xor","raw"),
    ("AB_CD","mod","raw"),("AB_CD","div","raw"),("AB_CD","maxv","raw"),
    ("AB_CD","minv","raw"),
]
_seen=set(FREQ_ORDER)
for p in PAIRINGS:
    for on,_ in OPS:
        for fn,_ in FORMATS:
            t=(p,on,fn)
            if t not in _seen:
                FREQ_ORDER.append(t); _seen.add(t)

def apply_combo(p, on, fn, A, B, C, D):
    L,R = pair(p,A,B,C,D)
    v = OP_DICT[on](L,R)
    if v is None: return None,L,R,None
    return FMT_DICT[fn](v),L,R,v

def parse_prompt(prompt):
    lines = prompt.split("\n")
    examples=[]
    for line in lines:
        m = re.match(r'^\s*(\d)(\d)(\D)(\d)(\d)\s*=\s*(\S+)\s*$', line)
        if m:
            A,B,sym,C,D,res = m.groups()
            examples.append((int(A),int(B),sym,int(C),int(D),res))
    m = re.search(r'result for[:\s]+(\d)(\d)(\D)(\d)(\d)', prompt)
    if not m: return None,None
    tA,tB,top,tC,tD = m.groups()
    return examples, (int(tA),int(tB),top,int(tC),int(tD))

def find_combo(op_examples):
    attempts=[]
    for combo in FREQ_ORDER:
        p,on,fn = combo
        ok=True; first_eval=None
        for i,(A,B,C,D,res) in enumerate(op_examples):
            out,L,R,v = apply_combo(p,on,fn,A,B,C,D)
            if i==0: first_eval=(L,R,v,out)
            if out is None or out!=res:
                ok=False; break
        attempts.append((combo,first_eval,ok))
        if ok: return combo,attempts
    return None,attempts

def build_cot(examples, target, combo, attempts, op_examples):
    tA,tB,top,tC,tD = target
    op_sym = top
    A1,B1,C1,D1,res1 = op_examples[0]
    lines=[PREAMBLE,""]
    lines.append("S2: PARSE")
    lines.append(f"Operator: '{op_sym}'")
    lines.append(f"EX1: {A1}{B1}{op_sym}{C1}{D1} = {res1}")
    lines.append(f"EX1 digits: A={A1},B={B1},C={C1},D={D1}")
    lines.append("")
    lines.append("S3: SCAN")
    shown=0
    for combo_t, first_eval, matched in attempts:
        if first_eval is None: continue
        p,on,fn = combo_t
        L,R,v,out = first_eval
        tag="YES" if matched else "NO"
        lines.append(f"#{shown+1}:{p}|{on}|{fn} L={L},R={R} {v}->{out} vs {res1} {tag}")
        shown+=1
        if matched: break
        if shown>=10: break
    if len(op_examples)>1:
        A2,B2,C2,D2,res2 = op_examples[1]
        p,on,fn = combo
        out2,L2,R2,v2 = apply_combo(p,on,fn,A2,B2,C2,D2)
        lines.append(f"VER EX2: {A2}{B2}{op_sym}{C2}{D2} L={L2},R={R2} {v2}->{out2} vs {res2} YES")
    p,on,fn = combo
    lines.append(f"LOCK: {p}|{on}|{fn}")
    lines.append("")
    tgt_out,tgt_L,tgt_R,tgt_v = apply_combo(p,on,fn,tA,tB,tC,tD)
    lines.append("S4: APPLY")
    lines.append(f"Target: {tA}{tB}{op_sym}{tC}{tD} A={tA},B={tB},C={tC},D={tD}")
    lines.append(f"{p}: L={tgt_L},R={tgt_R}")
    lines.append(f"{on}({tgt_L},{tgt_R})={tgt_v} {fn}={tgt_out}")
    lines.append("")
    lines.append(f"S5: ANS={tgt_out}")
    lines.append("")
    lines.append(f"\\boxed{{{tgt_out}}}")
    return "\n".join(lines), tgt_out

def main():
    df = pd.read_csv(CSV)
    mask = df['type']=='Equation Numeric'
    idxs = df[mask].index.tolist()
    print(f"Equation Numeric rows: {len(idxs)}")
    solved=0; failed=[]
    for idx in idxs:
        row = df.loc[idx]
        examples, target = parse_prompt(row['prompt'])
        if examples is None or not examples:
            failed.append((idx,'parse')); continue
        top = target[2]
        op_examples = [(A,B,C,D,res) for (A,B,sym,C,D,res) in examples if sym==top]
        if not op_examples:
            failed.append((idx,'no_op_ex')); continue
        combo, attempts = find_combo(op_examples)
        if combo is None:
            failed.append((idx,'no_combo')); continue
        cot, tgt_out = build_cot(examples, target, combo, attempts, op_examples)
        gt = str(row['answer']).strip()
        if tgt_out != gt:
            failed.append((idx,f'mismatch {tgt_out}!={gt}')); continue
        df.at[idx,'generated_cot'] = cot
        solved+=1
    print(f"Solved: {solved}/{len(idxs)}")
    print(f"Failed: {len(failed)}")
    # Drop failed rows
    drop_idx = [f[0] for f in failed]
    if drop_idx:
        df = df.drop(index=drop_idx).reset_index(drop=True)
    print(f"Final rows: {len(df)}")
    print(df['type'].value_counts())
    df.to_csv(CSV, index=False)
    print(f"Wrote {CSV}")

if __name__=='__main__':
    main()
