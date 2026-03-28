# NVIDIA Nemotron reasoning challenge (notes)

This document summarizes the competition framing, evaluation, and dataset layout for local reference. Official rules and the exact grading implementation live on the competition platform.

## Challenge overview

Reasoning benchmarks help measure progress on structured tasks. When methods and results are shared openly, the community can compare approaches, reproduce improvements, and iterate faster.

Many efforts use different datasets, prompts, and evaluation setups, which makes direct comparison hard. A **shared benchmark** and **common baseline** let techniques be tested more consistently.

Participants work from a **Nemotron 3 Nano** baseline and a **reasoning benchmark** from NVIDIA Research. Nemotron provides an open foundation: models, datasets, and training recipes you can build on or adapt.

You may experiment with:

- Prompting strategies  
- Data filtering and curation  
- Synthetic data generation  
- Reinforcement learning  
- Lightweight fine-tuning  
- Other approaches you choose  

You may use **any** training stack (Hugging Face, Unsloth, Axolotl, TRL, NVIDIA recipes, etc.). The **only hard requirement** is that the final artifact is a **compatible LoRA adapter** for **Nemotron-3-Nano-30B**. Clear documentation (notebooks, write-ups) is encouraged and **required for prize eligibility**.

## Dataset

The data is a collection of **logical reasoning puzzles**: find and apply hidden **transformation rules** from input–output examples, then solve a held-out instance. Domains include **bit manipulation**, **algebraic equations**, and related puzzle types.

### Files and fields

| File        | Role |
|------------|------|
| `train.csv` | Training puzzles with solutions. |
| `test.csv`  | Sample test prompts for developing your pipeline. At scoring time this is replaced by a **hidden** test set of **several hundred** problems. |

**`train.csv`**

| Column  | Description |
|---------|-------------|
| `id`    | Unique puzzle identifier. |
| `prompt`| Puzzle description: examples plus the instance to solve. |
| `answer`| Ground truth solution. |

**`test.csv`**

| Column  | Description |
|---------|-------------|
| `id`    | Unique puzzle identifier. |
| `prompt`| Same structure as in `train.csv` (no answers released). |

## Evaluation and score

Submissions are judged on **accuracy**: the fraction of test items answered correctly.

- **Inference:** Nemotron-3-Nano-30B is loaded with **your LoRA** (must include `adapter_config.json`) using **vLLM**.
- **Generation:** For each case the model is prompted to put its **final answer** inside a `\boxed{}` LaTeX command.
- **Parsing:** The metric extracts a prediction from the generated text, **preferring** `\boxed{...}`, then falling back to **other heuristics** or the **last numeric** value if needed.
- **Grading:** A prediction is **correct** if it matches the ground truth **exactly as a string**, or **within the official relative numerical tolerance** for numeric answers. (Exact tolerance and parsing edge cases are defined in the **NVIDIA Nemotron Metric** linked from the competition.)

**Final score** = (number correct) / (total test questions).

### Reported vLLM / serving parameters

| Parameter | Value |
|-----------|--------|
| `max_lora_rank` | 32 |
| `max_tokens` | 7680 |
| `top_p` | 1.0 |
| `temperature` | 0.0 |
| `max_num_seqs` | 64 |
| `gpu_memory_utilization` | 0.85 |
| `max_model_len` | 8192 |

Your LoRA must have **rank at most 32**.

## Submitting

Package a valid LoRA for Nemotron-3-Nano-30B into **`submission.zip`**. The NVIDIA submission demo notebook is a practical template for layout and contents (e.g. `adapter_config.json` plus adapter weights).

---

