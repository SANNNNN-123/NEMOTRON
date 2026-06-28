---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 22px;
    line-height: 1.22;
    padding: 34px 46px;
  }

  h1 {
    font-size: 2em;
  }

  h2 {
    font-size: 1.45em;
  }

  h3 {
    font-size: 1.2em;
  }

  h1,
  h2,
  h3,
  p,
  ul,
  ol,
  blockquote,
  table,
  pre {
    margin-top: 0.34em;
    margin-bottom: 0.34em;
  }

  li {
    margin-top: 0.12em;
    margin-bottom: 0.12em;
  }

  table {
    width: 100%;
    table-layout: fixed;
    font-size: 0.72em;
    line-height: 1.16;
  }

  th,
  td {
    padding: 5px 7px;
    vertical-align: middle;
  }

  code {
    font-size: 0.82em;
  }

  td code {
    white-space: normal;
    overflow-wrap: anywhere;
  }

  pre {
    font-size: 0.68em;
    line-height: 1.12;
  }

  blockquote {
    font-size: 0.86em;
    padding-left: 0.75em;
  }
---

# Intro to Finetuning: SFT + LoRA
### From a real Kaggle competition — NVIDIA Nemotron Reasoning Challenge

---

## Slide 1 — Title

**Intro to Finetuning: SFT + LoRA**

*How we took a 30-billion-parameter model from 0.70 → 0.85 on reasoning puzzles — step by step*

- Model: NVIDIA Nemotron-3-Nano-30B
- Competition: NVIDIA Nemotron Model Reasoning Challenge (Kaggle)
- Approach: Supervised Fine-Tuning + LoRA + Chain-of-Thought data

---

## Section 1: Why Do We Need Finetuning?

---

### Slide 2 — Base Model vs Finetuned: What's Missing

A base model is trained to predict the next token on internet text.
It's general-purpose — it doesn't know *your* task.

**The gap:**

| What base model knows | What it doesn't know |
|---|---|
| Language, grammar, facts | Your answer format |
| General reasoning | Your specific reasoning steps |
| Many topics | Your domain's quirks |

**Real example from our competition:**
- Task: decrypt a substitution cipher in Alice's Wonderland
- Base model: answers in wrong format, wrong style
- Finetuned model: follows exact step-by-step decryption format → 98.8% accuracy

> "The model already knows HOW to reason. Finetuning teaches it WHERE to apply that reasoning."

---

### Slide 3 — The Spectrum: When to Use What

```
Prompt Engineering → RAG → Finetuning → Full Training
        ↑                                      ↑
   cheapest, fastest                  expensive, most control
```

**Prompt Engineering** — add instructions in the prompt
- Best for: general tasks, one-off queries
- Limit: can't change reasoning style

**RAG (Retrieval-Augmented Generation)** — give model external documents
- Best for: knowledge-heavy tasks, up-to-date information
- Limit: doesn't change how model thinks

**SFT (Supervised Fine-Tuning)** — train on (question, reasoning, answer) pairs
- Best for: structured tasks with consistent reasoning patterns ← *our case*
- When format + step-by-step logic matters

**Full Training** — train from scratch
- Best for: entirely new domains or languages
- Almost never needed

---

### Slide 4 — Our Problem: Alice's Wonderland Reasoning Tasks

**The competition:**
- Fine-tune a 30B model to solve 9 types of reasoning puzzles
- Evaluation: 950 test questions, weighted accuracy score (0.0 to 1.0)

**The 9 puzzle categories:**

| Category | Example | Val questions |
|---|---|---|
| bit_manipulation | `10001110 → 00100110, what maps 11100000 →?` | 169 |
| cipher | `rxyefwoo → princess, what does teyuci map to?` | 162 |
| gravity | `t=1.92s, d=26.32m → find d at t=4.59s` | 159 |
| numeral | number system deduction | 149 |
| unit_conversion | custom unit scale deduction | 171 |
| cryptarithm_deduce | `97%32=3103, 25/84=109 → 59?46=?` | 71 |
| equation_numeric_deduce | equation rule deduction | 48 |
| cryptarithm_guess | operator not shown in examples | 14 |
| equation_numeric_guess | — | 7 |

**Starting score: 0.70 → Our best: 0.85**

---

## Section 2: What is SFT?

---

### Slide 5 — Supervised Fine-Tuning Explained Simply

**SFT = teaching by example**

You provide the model with thousands of worked examples:

```
INPUT (prompt):
  "In Alice's Wonderland, the gravitational constant has been
   secretly changed. For t=1.92s, distance=26.32m ...
   Find distance at t=4.59s"

OUTPUT (reasoning + answer):
  "We need to find k from d = k*t²
   t=1.92, d=26.32: t²=3.6864, k=26.32/3.6864=7.14
   t=4.47, d=142.63: t²=19.98, k=142.63/19.98=7.14 ✓
   For t=4.59: d = 7.14 × 4.59² = 7.14 × 21.07 = 150.4
   \boxed{150.4}"
```

The model learns:
1. How to structure its reasoning
2. What intermediate steps to show
3. How to format the final answer

> "Show it HOW to think, not just WHAT to answer."

---

### Slide 6 — SFT Is All You Need (for structured reasoning)

**Alternatives and why we didn't use them:**

| Method | How it works | Problem for us |
|---|---|---|
| RLHF | Human preference feedback | Expensive, needs human raters |
| PPO | Reward model + policy gradient | Unstable, complex setup |
| DPO | Preference pairs | Needs (good, bad) response pairs |
| **SFT** | **Correct examples only** | **Simple, stable, fast** |

**Our evidence:**
- 12 experiments, all SFT-only
- Score went from 0.70 → **0.85** purely by improving SFT data quality
- No reward models, no preference learning, no RLHF

**Rule of thumb:** If you have correct answers and a clear reasoning format → SFT is all you need.

---

## Section 3: What is LoRA?

---

### Slide 7 — The Problem: Full Finetuning a 30B Model

**Why you can't just finetune normally:**

```
Model parameters:  31.8 billion
Storage (BF16):    31.8B × 2 bytes = 63.6 GB   (just to load)
Full FP32 copy:    31.8B × 4 bytes = 127.2 GB  (needed for training)
Optimizer states:  127.2 GB × 2   = 254.4 GB   (Adam m and v)
Gradients:         127.2 GB                     (same size as weights)
─────────────────────────────────────────────────────────────
Total needed:                     ≈ 572 GB      ✗
```

**Available:** RTX Pro 6000 with 96 GB VRAM

Full finetuning is **impossible** on a single GPU for this model.

---

### Slide 8 — LoRA: Low Rank Adaptation

**The key insight:** Most of the "update" needed for finetuning is low-rank.

Instead of updating the full weight matrix W (huge), add a small side path:

```
Normal forward pass:
  output = W × input          (W is frozen — never changes)

LoRA forward pass:
  output = W × input  +  (B × A) × input
           ↑ frozen          ↑ trainable
           63.6 GB            3.5 GB
```

- **W**: original pre-trained weights — frozen, not updated
- **A**: small matrix [rank × input_dim] — trainable
- **B**: small matrix [output_dim × rank] — trainable
- **rank=32**: bottleneck that keeps the matrices tiny

```
W: [4096 × 4096]  = 16M params × 4 bytes = 64 MB  (frozen)
A: [  32 × 4096]  = 131K params            (trainable)
B: [4096 ×   32]  = 131K params            (trainable)
```

262K params instead of 16M — **60× smaller update**.

---

### Slide 9 — LoRA in Practice: Our Numbers

**Our LoRA configuration:**

| Setting | Value | Why |
|---|---|---|
| Model | Nemotron-3-Nano-30B | Competition requirement |
| LoRA rank | 32 | Inference constraint ≤ 32 |
| LoRA alpha | 32 | Scaling factor (= rank) |
| Target modules | q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, up_proj, down_proj, lm_head | All attention + MLP layers |
| Trainable params | 877 million | 2.8% of full model |
| Trainable size | 3.5 GB (FP32) | vs 127 GB for full model |

**GPU memory breakdown during training:**

```
Base model (BF16, frozen):    63.6 GB
LoRA weights (FP32):           3.5 GB
LoRA gradients (FP32):         3.5 GB
Adam optimizer states:         7.1 GB
CUDA context + buffers:        3.0 GB
Gradient checkpointing (μ=2):  9.2 GB
Unsloth fused CE (saves):     -8.5 GB
──────────────────────────────────────
Peak VRAM used:               91.5 GB  ✓ (fits in 96 GB)
```

---

## Section 4: The Dataset — Chain of Thought (CoT)

---

### Slide 10 — What is CoT and Why It Matters

**Chain-of-Thought = reasoning steps between question and answer**

Without CoT:
```
Question: "t=1.92s, d=26.32m ... find d at t=4.59s"
Answer:   "150.4"
```
Model just memorises answers — doesn't generalise.

With CoT:
```
Question: "t=1.92s, d=26.32m ... find d at t=4.59s"
Reasoning:
  d = k*t²
  From t=1.92: k = 26.32/1.92² = 26.32/3.6864 = 7.14
  From t=4.47: k = 142.63/4.47² = 142.63/19.98 = 7.14 ✓
  k is consistent → k = 7.14
  At t=4.59: d = 7.14 × 4.59² = 7.14 × 21.07 = 150.4
Answer: "\boxed{150.4}"
```
Model learns the *procedure* → can generalise to new examples.

> "CoT teaches the model to think, not just to recall."

---

### Slide 11 — CoT Format: Before vs After (The Biggest Lesson)

**This was our single biggest insight.**

**Experiment 1 — Custom CoT format:**
```
"I will solve this step by step.
 CORRECT: 00100110 maps to output bit 0=0, 1=0...
 WRONG: trying rule XOR..."
```
→ bit_manipulation accuracy: **10.7%** (18/169)

**Experiment 5 — Familiar solver format (from existing reasoning.py):**
```
"We need to deduce the transformation by matching example outputs.
 I will put my final answer inside \boxed{}.

 Output 0: 00100110
 0  0
 1  0
 2  1
 ..."
```
→ bit_manipulation accuracy: **82.2%** (139/169) — **+71.5pp jump**

**Why?** The model had seen this format during pre-training. SFT reinforces the correct *answers*, not the format. Use a format the model already recognises.

> **Rule: Match your CoT format to what the model already knows.**

---

### Slide 12 — How We Created Our CoT Data

**Step 1: Find an existing solver**
- reasoning.py — already solves bit_manipulation using column-hashing
- Generates CoT in a format the model recognises
- Result: 1,364 verified bit_manipulation rows

**Step 2: Verify with ground truth**
- Run solver on each problem
- Keep only rows where solver answer = competition ground truth
- Discard wrong answers → no noise in training data

**Step 3: Task-specific refinements**
- Cipher: added word-count check instruction
  ```
  "Crucially, ensure the number of words in your \boxed{} answer
   exactly matches the number of words in the encrypted text."
  ```
  → cipher accuracy: 98.8%

- Gravity: added all 1,597 training rows (was only 976)
  → gravity accuracy: 99.4% → 100%

**Step 4: Handle hard categories with brute-force solver**
- Cryptarithm: tried all digit + operator combinations
- 91 verified out of 823 attempts (11% solve rate)
- Used 3x upsample → 273 training rows

---

## Section 5: Dataset Categories and Balance

---

### Slide 13 — Our Final Dataset: Exp 9 (Best Run)

**7,248 training rows across 9 categories:**

```
gravity                  1,597  ████████████████████  22.0%
bit_manipulation         1,590  ████████████████████  21.9%
cipher                   1,576  ████████████████████  21.7%
unit_conversion            990  ████████████          13.7%
numeral                    650  ████████               9.0%
equation_numeric_deduce    540  ███████                7.5%
cryptarithm_deduce         273  ████                   3.8%
equation_numeric_guess      21  ▌                      0.3%
cryptarithm_guess           11  ▌                      0.2%
```

**Key design choices:**
- Top 3 categories balanced (~1,580 each) — they have the most validation weight
- Small categories kept small — not enough verified data to safely upsample
- Cryptarithm at 3.8% — carefully tuned (see next slide)

---

### Slide 14 — Upsampling: How Much Is Too Much?

**Problem:** Only 91 solver-verified cryptarithm puzzles, but category has 7.5% val weight.

**We tried three upsample levels:**

| Multiplier | Training rows | % of dataset | Score | cryptarithm val |
|---|---|---|---|---|
| 12× (Exp 8) | 1,092 | 13.5% | 0.81 ↓ | 11.3% |
| **3× (Exp 9)** | **273** | **3.8%** | **0.85 ↑** | **5.6%** |
| 2× (Exp 11) | 182 | 2.5% | 0.83 | 2.8% |

**What happened at 12×:**
- Model saw the same 91 puzzles 12 times per epoch
- bit_manipulation dropped from 88.2% → 81.7% (diluted signal)
- cipher dropped from 99.4% → 96.3%

**Sweet spot: 3-5% of total dataset for upsampled minority categories**

> "More repetition ≠ better learning. The model starts memorising, not generalising."

---

## Section 6: Training Hyperparameters

---

### Slide 15 — Key Hyperparameters Explained

**Batch size (μ):** How many examples the model sees before updating weights
- Larger batch → more stable gradients, more VRAM
- Smaller batch → noisier updates, less VRAM

**Gradient accumulation:** Simulate a larger batch without extra VRAM
- Accumulate gradients over N steps before updating
- Effective batch = μ × gradient_accumulation_steps

**Learning rate:** How big each weight update is
- Too high → unstable training, loss spikes
- Too low → slow convergence
- Cosine annealing: starts high, smoothly decreases to near zero

**Epochs:** How many full passes through the training data
- 1 epoch for SFT on structured tasks = usually enough
- More epochs → risk of overfitting (memorising training data)

**Sequence length:** Maximum tokens per training example
- Our CoT outputs are long (step-by-step reasoning)
- We use 8,192 tokens to fit full CoT

---

### Slide 16 — Our Best Config (Exp 9)

```python
# LoRA
LORA_RANK      = 32
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.0

# Training
BATCH_SIZE     = 2       # microbatch per GPU
GRAD_ACCUM     = 32      # effective batch = 2 × 32 = 64
MAX_SEQ_LEN    = 8192    # tokens per example
LEARNING_RATE  = 2e-4    # with cosine annealing schedule
EPOCHS         = 1
LOGGING_STEPS  = 10

# Dataset
TOTAL_ROWS     = 7,248
```

**Results:**
- Training time: **3.8 hours**
- Peak VRAM: **90.5 GB** (94.3% of 96 GB)
- Final loss: **0.68** (started at 6.62)
- Competition score: **0.85**

---

### Slide 17 — The Batch Size Tradeoff (VRAM vs Speed)

**We tested three batch sizes — same dataset, same everything else:**

```
μ=1  (BATCH_SIZE=1, GRAD_ACCUM=64):
  Training time: ~8 hours
  Peak VRAM:     70 GB  (safe)
  Score: 0.81

μ=2  (BATCH_SIZE=2, GRAD_ACCUM=32):
  Training time: ~4 hours  ← 1.75× faster
  Peak VRAM:     90.4 GB  (94% of 96 GB)
  Score: 0.81  (same!)

μ=4  (BATCH_SIZE=4, GRAD_ACCUM=16):
  Training time: would be ~2.5 hours
  Peak VRAM:     ~123 GB  ✗ OOM — doesn't fit
```

**Lesson:** Push batch size as high as your VRAM allows.
- μ=2 was 1.75× faster with identical score
- μ=4 would OOM — 96 GB is the hard ceiling
- Gradient checkpointing is what makes μ=2 possible

---

## Section 7: Training Loss and Convergence

---

### Slide 18 — Reading the Loss Curve

**Exp 9 training loss (7,248 rows, 227 steps, 1 epoch):**

```
Loss
6.6 │●
    │  ●
4.0 │     ●
    │       ●
2.5 │         ●
    │           ●
1.2 │              ●
    │                ● ● ● ●
0.8 │                          ● ● ● ● ● ● ● ●
0.68│                                              ─────
    └──────────────────────────────────────────────────
    0  10  20  30  40  50  80  100  150  200  227  Steps
```

**What each phase means:**

| Steps | Loss | What's happening |
|---|---|---|
| 0–40 | 6.62 → 2.4 | Model learning basic output format |
| 40–100 | 2.4 → 0.8 | Model learning reasoning structure |
| 100–227 | 0.8 → 0.68 | Fine-tuning answer precision |

**Why 1 epoch is enough:**
- Loss plateaus around step 100 — model has learned the patterns
- More epochs would just repeat the same data → overfitting
- SFT on structured tasks saturates fast

---

## Section 8: Experiment Results

---

### Slide 19 — Score Progression: The Journey to 0.85

**12 experiments, each changing one variable:**

| Exp | Dataset | Key Change | Score |
|---|---|---|---|
| 1 | v6 | Custom "I will solve..." CoT format | 0.70 |
| 2 | v7_hybrid | "We need to deduce..." + bit-serial steps | 0.70 |
| 3 | v8 | FAIL/PASS labels, new output structure | 0.70 |
| 4 | v9 | RuleID + structured format | 0.70 |
| **5** | **v10** | **Familiar solver format (reasoning.py)** | **0.81** ← big jump |
| 6 | v11 | +226 solver-patched bit_manip rows | 0.82 |
| 7 | v13 | +cipher word-count check, gravity complete | 0.81 |
| **8** | **v14** | **Cryptarithm 12× upsample** | **0.81** ← regression |
| **9** | **v14_reduce** | **Cryptarithm 3× upsample** | **0.85** ← best |
| 10 | v15 | Added orig. crypto_guess + eq_guess rows | 0.82 |
| 11 | v16 | Cryptarithm 2× upsample | 0.83 |
| 12 | v17 | +4 solver-verified crypto_guess rows | 0.84 |

---

### Slide 20 — Validation Results by Category (Exp 9)

**950 test questions, weighted accuracy:**

| Category | Correct | Total | Accuracy | Contribution |
|---|---|---|---|---|
| gravity | 157 | 159 | 98.7% ✓ | 16.5% |
| numeral | 149 | 149 | **100%** ✓ | 15.7% |
| unit_conversion | 171 | 171 | **100%** ✓ | 18.0% |
| cipher | 160 | 162 | 98.8% ✓ | 16.8% |
| bit_manipulation | 143 | 169 | 84.6% | 15.1% |
| equation_numeric_deduce | 41 | 48 | 85.4% | 4.3% |
| cryptarithm_deduce | 4 | 71 | 5.6% ← weak | 0.4% |
| cryptarithm_guess | 0 | 14 | 0.0% ← none | 0.0% |
| equation_numeric_guess | 0 | 7 | 0.0% ← none | 0.0% |
| **TOTAL** | **825** | **950** | **86.8%** | **Score: 0.85** |

**Where points were left on the table:**
- cryptarithm_deduce: only 91 verified training examples (hard to solve)
- cryptarithm_guess / eq_numeric_guess: 0% — no good training data yet

---

### Slide 21 — Key Lessons from 12 Experiments

**1. Data format beats data volume**
- Exp 1–4: 0.70 with custom formats
- Exp 5: 0.81 just by switching to familiar format
- Same data size, same model, same everything — only the format changed

**2. Solver-verified data is essential**
- We only kept examples where solver answer = ground truth
- Bad CoT (wrong answer but plausible reasoning) is worse than no CoT

**3. Balanced upsampling wins over aggressive upsampling**
- 12× cryptarithm → bit_manipulation regressed by -6.5pp
- 3× cryptarithm → bit_manipulation only -3.6pp, score improved by +0.04

**4. One variable per experiment**
- Exp 9 differed from Exp 8 only in upsample factor (3× vs 12×)
- Clear attribution → know exactly what caused the change

**5. Category-level validation is essential**
- Total accuracy hides what's broken
- Track every category every run

---

## Section 9: SVD Rank Compression

---

### Slide 22 — Why Compress? The Inference Constraint

**Competition rule:** submitted LoRA adapter must have rank ≤ 32.

**But higher rank = better training:**

| Training rank | Trainable params | Score potential |
|---|---|---|
| 32 | 877M | 0.85 |
| 64 | 1,754M | higher |
| 96 | 2,631M | even higher |

**Solution:** Train at high rank, compress before submission.

```
Train (rank=64, alpha=128)
         ↓
    SVD compress (5 min)
         ↓
Submit (rank=32, alpha=64) ← fits competition constraint
```

**SVD compression:** decompose the learned weight update (B×A) using Singular Value Decomposition, keep only the top 32 singular values.

---

### Slide 23 — The Alpha Rule (Critical!)

**After compression, you MUST update adapter_config.json.**

**The rule: keep alpha/rank ratio constant**

```
Before compression:   rank=64,  alpha=128  (ratio = 2.0)
After compression:    rank=32,  alpha=64   (ratio = 2.0)  ✓
```

**What happens if you don't:**
```
Before: rank=64, alpha=128
After:  rank=32, alpha=128  ← wrong! (ratio changed from 2.0 to 4.0)
Result: very poor inference, score collapse
```

Alpha controls the scaling of the LoRA update during inference in vLLM.
Changing the ratio changes the effective magnitude of the adapter → garbage output.

**Compression time:**
- Naive SVD (reconstruct full matrix then decompose): 1h 30min
- Optimized Algorithm 2 (decompose in low-rank space): **5 min** (18× faster)

---

## Section 10: Tips for Your Own Project

---

### Slide 24 — Data Checklist

Before training, make sure your CoT data is:

**✅ Verified correct**
- Run each example through a solver/checker
- Discard any example where the CoT leads to a wrong answer
- Wrong-answer CoT is actively harmful — model learns bad reasoning

**✅ Format-familiar**
- Use a reasoning format the model has seen during pretraining
- If unsure: look at how similar tasks are solved in public datasets

**✅ Step-by-step**
- Don't skip intermediate steps
- Show all calculation steps, intermediate values, verification

**✅ Category-balanced**
- Keep minority categories at ~3-5% of total dataset
- Don't over-upsample — 3× is usually the sweet spot

**✅ Task-specific checks added**
- Cipher: word-count verification before final answer
- Math: intermediate result verification
- Format constraints: `\boxed{}` enforcement

---

### Slide 25 — Training Checklist

**Before you start:**
- [ ] Dataset verified (no wrong-answer CoT rows)
- [ ] LoRA rank set to competition/deployment constraint
- [ ] Sequence length covers your longest CoT example
- [ ] VRAM budget checked (use memory analysis)

**During training:**
- [ ] Start with 1 epoch — add more only if loss hasn't converged
- [ ] Push μ as high as VRAM allows (use gradient checkpointing)
- [ ] Use cosine annealing learning rate schedule
- [ ] Log loss every step (not every 10) for visibility

**After training:**
- [ ] Validate on held-out set per category — not just total score
- [ ] Compare to previous experiment — only one variable changed?
- [ ] If using SVD compression: update adapter_config.json with new alpha
- [ ] Record results in experiment log before starting next run

---

## Section 11: Summary

---

### Slide 26 — What We Covered

**The SFT + LoRA finetuning pipeline:**

```
Problem definition
    ↓
Collect verified (prompt, CoT, answer) pairs
    ↓
Choose format that matches model's pretraining style
    ↓
Balance dataset categories (3-5% for minorities)
    ↓
Train: LoRA rank=32, μ=2, grad_accum=32, 1 epoch
    ↓
Validate per category — find weak spots
    ↓
Fix weakest category's data quality → repeat
    ↓
(Optional) SVD compress for rank reduction
    ↓
Submit
```

**Key numbers from our project:**

| Metric | Value |
|---|---|
| Model size | 31.8B parameters |
| LoRA trainable | 877M params (2.8%) |
| Training data | 7,248 rows |
| Training time | 3.8 hours |
| Score progression | 0.70 → **0.85** |
| Best category | numeral, unit_conversion: **100%** |

---

### Slide 27 — The 5 Rules of SFT

**1. Format familiarity > data volume**
Use the format the model already knows. This single change gave us +71.5pp on bit_manipulation.

**2. Verified data > more data**
One wrong-answer CoT row can hurt more than it helps. Verify everything with a solver.

**3. Balance > upsample**
3× on a minority category is better than 12×. Flooding the dataset crowds out what's working.

**4. One experiment at a time**
Change one variable per run. 12 clean experiments beat 3 messy ones.

**5. Validate per category**
Total score hides problems. Track every category every run.

---

### Slide 28 — What's Next?

**From 0.85 → 0.86+:**

- **Higher LoRA rank during training** (rank=64 or 96 → compress to 32)
  - More expressive adapter → better learning
  - Use SVD to compress before submission

- **SeedLoRA ensembling** (train same data with different seeds)
  - Seed 1: 0.86, Seed 2: 0.84, Seed 3: 0.85 → ensemble: 0.86
  - Naive averaging collapses (≤ 0.5) — use SeedLoRA method

- **Better cryptarithm CoT** (currently 5.6% val accuracy)
  - Need more solver-verified examples
  - Brute-force solver currently finds 91/823 (11% success rate)

- **Fix equation_numeric_guess** (currently 0%)
  - Needs new CoT format — guess category where operator never appears in examples

> SFT + LoRA is not a one-shot process. It's an iterative data quality loop.

---

*End of slides*

---

## Appendix: Quick Reference

### LoRA config (Exp 9 — best run)
```python
LORA_RANK      = 32
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "in_proj", "out_proj", "up_proj", "down_proj", "lm_head"]
```

### Training config (Exp 9)
```python
BATCH_SIZE     = 2
GRAD_ACCUM     = 32      # effective batch = 64
MAX_SEQ_LEN    = 8192
LEARNING_RATE  = 2e-4
LR_SCHEDULER   = "cosine"
EPOCHS         = 1
```

### SVD compression rule
```
Training:   rank=R,      alpha=A       (ratio = A/R)
Compressed: rank=32,     alpha=32*(A/R)
```
Example: rank=64, alpha=128 → compressed: rank=32, alpha=64 ✓
