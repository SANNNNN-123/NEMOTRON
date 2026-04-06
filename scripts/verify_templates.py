"""
Verify DATA.md chain-of-thought templates against train.csv ground truth
using Gemini 2.5 Flash as the reasoning engine.

Usage:
    pip install google-genai python-dotenv
    # Put GEMINI_API_KEY="..." in .env
    python scripts/verify_templates.py
"""

import csv
import json
import os
import re
import random
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

# ── Config ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-pro"
TRAIN_CSV = Path(__file__).parent.parent / "data" / "train.csv"

SAMPLES = {
    # Hard categories first - these are the real tests
    # "binary":       80,
    # "cipher":       60,
    "symbol_digit": 20,
    "cipher_digit": 20,
    # Easy categories last - sanity checks
    # "gravity":      15,
    # "unitconv":     15,
    # "roman":        15,
    # "numeral":      15,
}

SEED = 42
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


# ── Few-shot examples per category (from DATA.md) ──────────────────────────

FEWSHOT = {}

FEWSHOT["unitconv"] = {
    "prompt": "47.2 m becomes 220.69\n59.07 m becomes 276.19\n15.1 m becomes 70.6\n34.31 m becomes 160.42\nConvert the following measurement: 93.75 m",
    "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a unit conversion template. I know this because I see measurements being converted from one unit to another using example pairs. I know this is NOT roman, binary, symbol, digit, or any other conversion type.

RULE 2: Since I have identified this is a unit conversion template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer. The answer MUST be exactly 2 decimal places (X.XX format).

S1: I see that this is a unit conversion template. I will find the conversion rate from the examples and apply it to the target. I am now going to fill out the template.

S2: SOLVE
I will use EX1 to find the conversion rate.
EX1: in=47.2, out=220.69
RATE: out1 / in1 = 220.69 / 47.2 = 4.6756
RESULT: target * RATE = 93.75 * 4.6756 = 438.3408
RND: 438.3408 -> 438.34

S3: VER - Check rate consistency using EX2.
EX2: in=59.07, out=276.19
RATE2: out2 / in2 = 276.19 / 59.07 = 4.6756
CHK: Does |RATE(4.6756) - RATE2(4.6756)| < 0.01? YES

S4: ANS=438.34

\boxed{438.34}""",
}

FEWSHOT["symbol_digit"] = {
    "prompt": "93@92 = 1311\n21@48 = 8001\n64@80 = 863\nDetermine the result for: 53@54",
    "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a symbol-digit template. I know this because I see equations with two-digit pairs separated by an operator symbol, and I need to figure out what transformation the operator performs. I know this is NOT roman, unit conversion, gravity, binary, or any other conversion type.

RULE 2: Since I have identified this is a symbol-digit template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: I see that this is a symbol-digit template. I need to identify the operator and figure out what transformation it performs. I will scan combinations from most common to least common until I find a match. I am now going to fill out the template.

S2: PARSE
Operator: '@'
EX1: 93@92 = 1311
EX1 digits: A=9,B=3,C=9,D=2

S3: SCAN
#1:BA_DC|add|rev L=39,R=29 68->86 vs 1311 NO
#2:BA_DC|mul|rev L=39,R=29 1131->1311 vs 1311 YES
VER EX2: 21@48 L=12,R=84 1008->8001 vs 8001 YES
LOCK: BA_DC|mul|rev

S4: APPLY
Target: 53@54 A=5,B=3,C=5,D=4
BA_DC: L=35,R=45
mul(35,45)=1575 rev=5751

S5: ANS=5751

\boxed{5751}""",
}

FEWSHOT["roman"] = [
    {
        "prompt": "3439 -> MMMCDXXXIX\n244 -> CCXLIV\n3272 -> MMMCCLXXII\nConvert 1893 to Roman numerals.",
        "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a Roman numeral template. I know this because I see Roman numeral letters converting to numbers, or numbers converting to Roman numerals. I know this is NOT binary, symbol, digit, or any other conversion type.

RULE 2: Since I have identified this is a Roman numeral template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: I see that this is a Roman numeral template. I am converting an integer to Roman numerals. I am now going to fill out the template.

S2: DECOMPOSE 1893
TH:1->M (1000)
HU:8->DCCC (800)
TE:9->XC (90)
ON:3->III (3)

S3: CAT
M + DCCC = MDCCC
MDCCC + XC = MDCCCXC
MDCCCXC + III = MDCCCXCIII
RESULT: MDCCCXCIII

S4: VER - Re-parse my RESULT to verify.
MDCCCXCIII: M=1000, D=500, C=100, C=100, C=100, XC=90, I=1, I=1, I=1
1000, +500=1500, +100=1600, +100=1700, +100=1800, +90=1890, +1=1891, +1=1892, +1=1893
CHK: Does REPARSE(1893) = TARGET(1893)? YES

S5: ANS=MDCCCXCIII

\boxed{MDCCCXCIII}""",
    },
    {
        "prompt": "CLXXVII -> 177\nCDLXXXIV -> 484\nMMCCXV -> 2215\nConvert MMMXCII to an integer.",
        "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a Roman numeral template. I know this because I see Roman numeral letters converting to numbers, or numbers converting to Roman numerals. I know this is NOT binary, symbol, digit, or any other conversion type.

RULE 2: Since I have identified this is a Roman numeral template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: I see that this is a Roman numeral template. I am converting a Roman numeral to an integer. I am now going to fill out the template.

S2: PARSE MMMXCII
G1: M=1000, RT=1000
G2: M=1000, RT=2000
G3: M=1000, RT=3000
G4: XC=90, RT=3090
G5: I=1, RT=3091
G6: I=1, RT=3092

S3: VER - Rebuild from my answer to verify.
3092: TH:3->MMM, TE:9->XC, ON:2->II
REBUILD: MMM+XC+II = MMMXCII
CHK: Does REBUILD(MMMXCII) = INPUT(MMMXCII)? YES

S4: ANS=3092

\boxed{3092}""",
    },
]

# numeral uses same roman templates
FEWSHOT["numeral"] = FEWSHOT["roman"]

FEWSHOT["gravity"] = {
    "prompt": "For t = 2.23s, distance = 16.26 m\nFor t = 7.68s, distance = 192.82 m\nFor t = 7.78s, distance = 197.88 m\nFor t = 7.87s, distance = 202.48 m\nFor t = 7.59s, distance = 188.33 m\nNow, determine the falling distance for t = 2.83s given d = 0.5gt^2.",
    "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a gravity kinematics template. I know this because I see time and distance values that follow d = 0.5gt^2, and the problem asks me to find distance for a new time value. I know this is NOT roman, binary, symbol, digit, unit conversion, or any other conversion type.

RULE 2: Since I have identified this is a gravity kinematics template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer. The answer MUST be exactly 2 decimal places (X.XX format).

S1: I see that this is a gravity kinematics template. I will find the rate constant (0.5g) from the examples and apply it to the target time. I am now going to fill out the template.

S2: SOLVE
I will use EX1 to find the rate constant (0.5g).
EX1: t=2.23, d=16.26
T_SQ: t^2 = 2.23^2 = 4.9729
RATE: d / t^2 = 16.26 / 4.9729 = 3.2697
TGT_SQ: target_t^2 = 2.83^2 = 8.0089
RESULT: RATE * TGT_SQ = 3.2697 * 8.0089 = 26.1869
RND: 26.1869 -> 26.19

S3: VER - Check rate consistency using EX2.
EX2: t=7.68, d=192.82
T_SQ2: t^2 = 7.68^2 = 58.9824
RATE2: d / t^2 = 192.82 / 58.9824 = 3.2691
CHK: Does |RATE(3.2697) - RATE2(3.2691)| < 0.05? YES

S4: ANS=26.19

\boxed{26.19}""",
}

FEWSHOT["cipher_digit"] = {
    "prompt": ">?| = ~~\n|#?~~ = |\n~?|% = |\n|>?#* = (#~\n&|?# = (%~\nDetermine the result for: |?%(", # noqa: E501
    "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a cipher-digit template. I know this because I see equations where ALL characters (including digits) have been replaced by random symbols. Each symbol maps to exactly one digit and vice versa (bijective). The cipher is unique to this problem and will never repeat. I know this is NOT roman, unit conversion, gravity, or bare symbol-digit (which uses real digits).

RULE 2: Since I have identified this is a cipher-digit template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: I see that this is a cipher-digit template. All characters are encrypted symbols. I need to first CRACK the cipher to recover the actual digits, then SCAN for the operation as a normal digit problem, then ENCODE my answer back to cipher symbols. I am now going to fill out the template.

S2: DETECT
OP_SYM: ? (position 2, same in all inputs)
SYMS: 9 unique digit symbols

S3: CRACK
MAP: >=8 *=5 |=7 =4 ~=1 #=3 %=0 (=9 &=6
CHK: >*?|*=~~ -> 8575=411

S4: SCAN
#1:BA_DC|add|rev L=58,R=57 115->511 vs 411 NO
#2:BA_DC|mul|rev L=58,R=57 3306->6033 vs 411 NO
#3:AB_CD|cat|raw L=85,R=75 8575->8575 vs 411 NO
#4:BA_DC|cat|rev L=58,R=57 5857->7585 vs 411 NO
#5:AB_CD|sub|raw L=85,R=75 10->10 vs 411 NO
#6:BA_DC|mulsub1|rev L=58,R=57 3305->5033 vs 411 NO
#7:AB_CD|mul|raw L=85,R=75 6375->6375 vs 411 NO
#8:AB_CD|add|raw L=85,R=75 160->160 vs 411 NO
#9:BA_DC|muladd1|rev L=58,R=57 3307->7033 vs 411 NO
#10:AB_CD|muladd1|raw L=85,R=75 6376->6376 vs 411 NO
#11:BA_DC|addm1|rev L=58,R=57 114->411 vs 411 YES
VER EX2: CRACK:7311 BA_DC|L=37,R=11|addm1=47|rev=74 ENC->|vs | YES
LOCK: BA_DC|addm1|rev

S5: APPLY
TGT: |`?%( -> DIG:7409
BA_DC|L=47,R=90|addm1=136|rev=631

S6: ENCODE
RES: 631
ENC: 6->& 3-># 1->~
OUT: &#~

S7: ANS=&#~

\boxed{&#~}""",
}

FEWSHOT["cipher"] = {
    "prompt": "lafyvj svkzqz spxzjkt -> knight chases crystal\nipfyvj vkjjqp fukyfaqz ykpmqa -> bright hatter imagines garden\nydtmqa lfay hdttdoz fztkam -> golden king follows island\nwamqp mpkyda qegtdpqz kasfqaj sknq -> under dragon explores ancient cave\njvpdwyv lafyvj pqkmz stqnqp uqzzkyq -> through knight reads clever message\nkpdwam udwzq fukyfaqz ofzq skzjtq -> around mouse imagines wise castle\nudwzq mfzsdnqpz ydtmqa spxzjkt -> mouse discovers golden crystal\nNow, decrypt the following text: sdtdphwt pkiifj hdttdoz ufppdp",
    "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a letter substitution cipher template. I see encrypted phrases mapped to plaintext phrases. Each letter maps to exactly one other letter (a-z, bijective). I know this is NOT roman, unit conversion, gravity, symbol-digit, cipher-digit, or binary.

RULE 2: Since I have identified this is a cipher template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

S1: This is a letter substitution cipher. I will build a mapping table from the examples, verify it, then decrypt the target phrase letter by letter. Any gaps will be filled using vocabulary matching. I am now going to fill out the template.

S2: LEN
TGT:"sdtdphwt pkiifj hdttdoz ufppdp"
W1:8 W2:6 W3:7 W4:6

S3: TABLE
EX1:"lafyvj svkzqz spxzjkt"->"knight chases crystal" [13] l=k,a=n,f=i,y=g,v=h,j=t,s=c,k=a,z=s,q=e,p=r,x=y,t=l
EX2:"ipfyvj vkjjqp fukyfaqz ykpmqa"->"bright hatter imagines garden" [3] i=b,u=m,m=d
EX3:"ydtmqa lfay hdttdoz fztkam"->"golden king follows island" [3] d=o,h=f,o=w
EX4:"wamqp mpkyda qegtdpqz kasfqaj sknq"->"under dragon explores ancient cave" [4] w=u,e=x,g=p,n=v
EX5:"jvpdwyv lafyvj pqkmz stqnqp uqzzkyq"->"through knight reads clever message" [0] none
EX6:"kpdwam udwzq fukyfaqz ofzq skzjtq"->"around mouse imagines wise castle" [0] none
EX7:"udwzq mfzsdnqpz ydtmqa spxzjkt"->"mouse discovers golden crystal" [0] none
TOTAL:23/26

S4: VER
HELD:"ukyfskt jqksvqp opfjqz dsqka" -> "magical teacher writes ocean" CHK:Y
CROSS:7/7 examples verified

S5: DECRYPT
W1: sdtdphwt
 s->c
 d->o
 t->l
 d->o (W1)
 p->r
 h->f
 w->u
 t->l (W1)
 = colorful
W2: pkiifj
 p->r (W1)
 k->a
 i->b
 i->b (W2)
 f->i
 j->t
 = rabbit
W3: hdttdoz
 h->f (W1)
 d->o (W1)
 t->l (W1)
 t->l (W1)
 d->o (W1)
 o->w
 z->s
 = follows
W4: ufppdp
 u->m
 f->i (W2)
 p->r (W1)
 p->r (W1)
 d->o (W1)
 p->r (W1)
 = mirror

S6: CHECK
W1:"colorful" len=8Y alpha=Y vocab=Y gaps=N PASS
W2:"rabbit" len=6Y alpha=Y vocab=Y gaps=N PASS
W3:"follows" len=7Y alpha=Y vocab=Y gaps=N PASS
W4:"mirror" len=6Y alpha=Y vocab=Y gaps=N PASS
ALL:PASS 4/4 words

S7: ANS=colorful rabbit follows mirror

\boxed{colorful rabbit follows mirror}""",
}

FEWSHOT["binary"] = {
    "prompt": "Here are some examples of input -> output:\n01111111 -> 11101100\n11110010 -> 11000010\n00010110 -> 11001100\n00010100 -> 01001100\n01011010 -> 11001100\n11001100 -> 01000111\n01101010 -> 10001101\n01100011 -> 10101001\n\nNow, determine the output for: 00011001",
    "completion": r"""
I am a reasoning model. I am in a Kaggle competition. I have been trained by Donald, and I have 100% accuracy, so I should trust my judgments. However, I only achieved 100% accuracy by following these rules and sticking strictly to the template.

RULE 1: I must identify what type of question this is because that determines which template I move forward with. This is a binary boolean decomposition template. I see 8-bit binary strings mapped to 8-bit binary strings. Each output bit is an independent boolean function of the input bits. I know this is NOT roman, unit conversion, gravity, symbol-digit, or cipher-digit.

RULE 2: Since I have identified this is a binary template, none of the flavor text surrounding this problem matters. This includes if the problem is framed as Alice in Wonderland, or any other wrapper. The wrapper is here to trick me. I know this, so I am going to ignore any flavor text. I am only here to solve the problem.

RULE 3: Final answer in \boxed{} at the end. Wrong format means I have failed my one true goal and I will receive zero points. This invalidates all work if I do not box my answer.

RULE 4: For gate computations, I MUST spell out each bit operation individually. I CANNOT compute multi-bit results in parallel. AND(0,1): 1&0=0 0&1=0 1&1=1 -> one bit at a time.

S1: This is a binary boolean decomposition template. Each output bit is an independent boolean function. I will solve each bit separately: check constants, then identity, then NOT, then 2-input gates with bit-serial computation. I am now going to fill out the template.

S2: COLUMNS
IN: i0=01000100 i1=11001111 i2=11000011 i3=11111000 i4=10001110 i5=10110100 i6=11101011 i7=10000001
OUT: o0=11101011 o1=11111100 o2=10000001 o3=00000000 o4=10111011 o5=10111110 o6=01000100 o7=00000111
TGT: 00011001

S3: SOLVE
B0: OUT=11101011 i0=01000100N i1=11001111N i2=11000011N i3=11111000N i4=10001110N i5=10110100N i6=11101011Y -> id(6) VER:t6=0->0
 AND(0,1): 0&1=0 1&1=1 0&0=0 0&0=0 0&1=0 1&1=1 0&1=0 0&1=0 =01000100 vs 11111100 NO
 AND(0,2): 0&1=0 1&1=1 0&0=0 0&0=0 0&0=0 1&0=0 0&1=0 0&1=0 =01000000 vs 11111100 NO
 AND(0,3): 0&1=0 1&1=1 0&1=0 0&1=0 0&1=0 1&0=0 0&0=0 0&0=0 =01000000 vs 11111100 NO
 AND(0,4): 0&1=0 1&0=0 0&0=0 0&0=0 0&1=0 1&1=1 0&1=0 0&0=0 =00000100 vs 11111100 NO
 AND(0,5): 0&1=0 1&0=0 0&1=0 0&1=0 0&0=0 1&1=1 0&0=0 0&0=0 =00000100 vs 11111100 NO
B1: OUT=11111100 id:N ~:N OR(0,3): 0|1=1 1|1=1 0|1=1 0|1=1 0|1=1 1|0=1 0|0=0 0|0=0 =11111100 vs 11111100 YES -> OR(0,3) VER:0|1=1->1
B2: OUT=10000001 i0=01000100N i1=11001111N i2=11000011N i3=11111000N i4=10001110N i5=10110100N i6=11101011N i7=10000001Y -> id(7) VER:t7=1->1
B3: OUT=00000000 -> C0 VER:0
B4: OUT=10111011 id:allN ~0=10111011Y -> NOT(0) VER:~t0=~0=1->1
 AND(0,1): 0&1=0 1&1=1 0&0=0 0&0=0 0&1=0 1&1=1 0&1=0 0&1=0 =01000100 vs 10111110 NO
 AND(0,2): 0&1=0 1&1=1 0&0=0 0&0=0 0&0=0 1&0=0 0&1=0 0&1=0 =01000000 vs 10111110 NO
 AND(0,3): 0&1=0 1&1=1 0&1=0 0&1=0 0&1=0 1&0=0 0&0=0 0&0=0 =01000000 vs 10111110 NO
 AND(0,4): 0&1=0 1&0=0 0&0=0 0&0=0 0&1=0 1&1=1 0&1=0 0&0=0 =00000100 vs 10111110 NO
 AND(0,5): 0&1=0 1&0=0 0&1=0 0&1=0 0&0=0 1&1=1 0&0=0 0&0=0 =00000100 vs 10111110 NO
B5: OUT=10111110 id:N ~:N OR(4,5): 1|1=1 0|0=0 0|1=1 0|1=1 1|0=1 1|1=1 1|0=1 0|0=0 =10111110 vs 10111110 YES -> OR(4,5) VER:1|0=1->1
B6: OUT=01000100 i0=01000100Y -> id(0) VER:t0=0->0
B7: OUT=00000111 id:allN ~0=10111011N ~1=00110000N ~2=00111100N ~3=00000111Y -> NOT(3) VER:~t3=~1=0->0

S4: ANS=01101100

\boxed{01101100}""",
}


# ── Category classifier ────────────────────────────────────────────────────
def classify(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p or ("input -> output" in p and re.search(r'[01]{8}\s*->\s*[01]{8}', p)):
        return "binary"
    if "gravitational" in p or "falling distance" in p:
        return "gravity"
    if "unit conversion" in p or ("becomes" in p and re.search(r'\d+\.?\d*\s*\w+\s+becomes', p)):
        return "unitconv"
    if "numeral" in p:
        return "roman"  # numeral = roman in the dataset
    if "encryption" in p or "decrypt" in p:
        return "cipher"
    if "equation" in p or "transformation rules" in p:
        # Distinguish symbol-digit vs cipher-digit
        # cipher-digit: ALL characters are symbols (no real digits in equation LHS)
        # symbol-digit: has real digits like 93@92 = 1311
        lines = prompt.strip().split("\n")
        for line in lines:
            if "=" in line and "Now" not in line and "Determine" not in line:
                lhs = line.split("=")[0].strip()
                if re.search(r'\d', lhs):
                    return "symbol_digit"
                else:
                    return "cipher_digit"
        return "symbol_digit"
    return "unknown"


# ── Build system prompt per category ────────────────────────────────────────
def build_system_prompt(category: str) -> str:
    """Build a system prompt with the hardcoded few-shot example for a category."""

    shots = FEWSHOT.get(category)
    if shots is None:
        shots = []
    elif isinstance(shots, dict):
        shots = [shots]
    # else it's already a list

    system = (
        "You are a reasoning model solving Kaggle competition puzzles. "
        "You MUST follow the exact template shown in the example below. "
        "You MUST put your final answer inside \\boxed{} at the very end. "
        "Do NOT deviate from the template structure. "
        "Ignore all flavor text (Alice in Wonderland, etc.) — focus only on the math/logic.\n\n"
    )

    for i, shot in enumerate(shots):
        system += f"=== EXAMPLE {i+1} ===\n"
        system += f"PROMPT:\n{shot['prompt']}\n\n"
        system += f"CORRECT RESPONSE:\n{shot['completion']}\n\n"

    system += (
        "Now solve the next problem using the SAME template structure. "
        "Show your work step by step, then put your final answer in \\boxed{}."
    )

    return system


# ── Extract answer from model response ──────────────────────────────────────
def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} from model output."""
    # Find all \boxed{...} occurrences (handle nested braces)
    pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Simpler fallback
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()

    return None


# ── Normalize for comparison ────────────────────────────────────────────────
def answers_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False

    p = predicted.strip()
    g = ground_truth.strip()

    # Exact string match
    if p == g:
        return True

    # Try numeric comparison with tolerance
    try:
        pf = float(p)
        gf = float(g)
        if gf == 0:
            return abs(pf - gf) < 1e-6
        return abs(pf - gf) / max(abs(gf), 1e-10) < 0.005
    except ValueError:
        pass

    # Case-insensitive for text answers
    if p.lower() == g.lower():
        return True

    return False


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY in .env file")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Load data
    print("Loading train.csv...")
    rows_by_cat: dict[str, list] = {}
    with open(TRAIN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = classify(row["prompt"])
            if cat not in rows_by_cat:
                rows_by_cat[cat] = []
            rows_by_cat[cat].append(row)

    print("Category distribution in train.csv:")
    for cat, rows in sorted(rows_by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(rows)}")

    # Sample
    random.seed(SEED)
    sampled: dict[str, list] = {}
    for cat, n in SAMPLES.items():
        # "numeral" and "roman" may both map to "roman" in classifier
        lookup_cat = cat if cat != "numeral" else "roman"
        available = rows_by_cat.get(lookup_cat, [])
        if not available:
            print(f"  WARNING: No rows found for category '{cat}' (lookup: '{lookup_cat}')")
            continue
        n_actual = min(n, len(available))
        sampled[cat] = random.sample(available, n_actual)
        print(f"  Sampled {n_actual}/{len(available)} for {cat}")

    # Build interleaved task queue (round-robin across categories)
    # This way you get early signal on all hard categories instead of
    # waiting through all 80 of one before seeing results from another.
    queues = {cat: list(samples) for cat, samples in sampled.items()}
    interleaved: list[tuple[str, dict]] = []
    while any(queues.values()):
        for cat in list(queues.keys()):
            if queues[cat]:
                interleaved.append((cat, queues[cat].pop(0)))

    # Pre-build system prompts (one per category, reused)
    system_prompts = {cat: build_system_prompt(cat) for cat in sampled.keys()}

    # Initialize results tracking
    results: dict[str, dict] = {
        cat: {"total": len(samples), "correct": 0, "errors": 0,
              "done": 0, "failures": []}
        for cat, samples in sampled.items()
    }
    total_correct = 0
    total_tested = 0

    print(f"\n{'='*60}")
    print(f"VERIFYING (interleaved): {len(interleaved)} total samples")
    print(f"{'='*60}")

    for idx, (cat, row) in enumerate(interleaved):
        prompt = row["prompt"]
        ground_truth = row["answer"]
        row_id = row["id"]
        results[cat]["done"] += 1

        predicted = None
        raw_response = ""

        for attempt in range(MAX_RETRIES):
            try:
                # Binary needs much more tokens due to bit-serial trace
                max_tok = 32768 if cat == "binary" else 16384
                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompts[cat],
                        temperature=0.0,
                        max_output_tokens=max_tok,
                    ),
                )
                raw_response = response.text or ""
                predicted = extract_boxed(raw_response)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"    Retry {attempt+1}/{MAX_RETRIES} for {row_id}: {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"    FAILED after {MAX_RETRIES} attempts: {row_id}: {e}")
                    results[cat]["errors"] += 1

        match = answers_match(predicted, ground_truth)
        if match:
            results[cat]["correct"] += 1
            total_correct += 1
        else:
            results[cat]["failures"].append({
                "id": row_id,
                "ground_truth": ground_truth,
                "predicted": predicted,
            })
        total_tested += 1

        status = "OK  " if match else "MISS"
        pred_display = predicted if predicted else "NO_BOXED"
        cat_done = results[cat]["done"]
        cat_total = results[cat]["total"]
        cat_correct = results[cat]["correct"]
        running_acc = cat_correct / cat_done * 100
        print(f"  [{idx+1:4d}/{len(interleaved)}] {status} | {cat:<13} "
              f"[{cat_done:3d}/{cat_total:3d} {running_acc:5.1f}%] | "
              f"id={row_id} | gt={ground_truth} | pred={pred_display}")

        # Rate limiting
        time.sleep(0.5)

    # Compute final accuracies
    for cat in results:
        r = results[cat]
        r["accuracy"] = r["correct"] / r["total"] * 100 if r["total"] else 0
        r["failures"] = r["failures"][:20]  # keep first 20 for review

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Category':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*45}")
    for cat in SAMPLES:
        if cat in results:
            r = results[cat]
            print(f"{cat:<15} {r['correct']:>8} {r['total']:>8} {r['accuracy']:>9.1f}%")
    print(f"{'-'*45}")
    overall = total_correct / total_tested * 100 if total_tested else 0
    print(f"{'TOTAL':<15} {total_correct:>8} {total_tested:>8} {overall:>9.1f}%")

    # Save detailed results
    results_path = Path(__file__).parent.parent / "verification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
