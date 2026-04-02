# Unsloth / Nemotron reasoning — experiment log

Notes from local discussion: instant notebook runs, Unsloth full-data training, Kaggle time budget, and the **current Unsloth run** configuration.

---

## 1. Instant notebook (`nvidia-nemotron-training-copy-run-instantly.ipynb`) — past runs

Subsampled training (e.g. 600 rows, `seed=42` on sample) with varying hyperparameters. **Public LB scores observed:**

| Run (summary) | Score |
|---------------|-------|
| `NUM_OF_EPOCH=2` | 0.63 |
| `NUM_OF_EPOCH=1` | 0.64 |
| `LORA_RANK=32`, `MAX_SEQ_LEN=2048`, `NUM_OF_EPOCH=2` | 0.55 |
| `SAMPLES=600`, `EPOCH=1` | 0.65 |
| `LR=1e-4`, `lora_alpha=32` | 0.65 |

**Why scores differed**

- Each line is a **different model** (epochs, rank, seq len, samples, LR, alpha all change the outcome).
- **No global seed** for training + **`lora_dropout=0.05`** → stochastic training; CUDA/bf16 can add small run-to-run noise.
- **Public LB** is a subset of test → small gaps (e.g. 0.63 vs 0.64) can be noise + real differences.

**Takeaway:** Prefer **full-data Unsloth** for serious submits; use **600 + same seed** only for fast iteration.

---

## 2. Unsloth full data (`unsloth-training-full-data.ipynb`) — recipe drift

Baseline-style full-data run (~9.4k train after 100-val holdout, 1 epoch, ~1h20 on RTX PRO 6000 class GPU).

- After changing several things at once (**`MAX_SEQ_LEN` 768→1024**, **`warmup_ratio` 0.03→0.1**, **`optim` → `adamw_torch`**, **`LORA_DROPOUT` 0→0.05**), public LB moved **0.64 → 0.62**.
- **Interpretation:** Not proof that “full data is worse”; multiple hyperparameters changed + LB noise. Ablate **one** knob at a time when debugging.

**Wall clock (reported)**

- ~**1h19m** / epoch with `LORA_DROPOUT=0`
- ~**1h21m** / epoch with current settings → ~2 min difference treated as normal variance, not dropout cost.

---

## 3. Recommendations discussed (Unsloth, not all applied)

- **More epochs** (2–3): best if time allows; watch eval loss if a val split exists.
- **`MAX_SEQ_LEN`**: inference allows long context; training at 1024 may truncate — **1536/2048** can help but **slows** each epoch and risks OOM.
- **Completion-only loss** on assistant / `\boxed{}` region: same wall clock, often better signal (requires TRL wiring).
- **Batching:** keep **effective batch** constant; try **higher `BATCH_SIZE`, lower `GRAD_ACCUM`** (e.g. 8×3 vs 6×24) — Unsloth noted NemotronH + grad accum is slightly imperfect.
- **Final submit:** optional **train on all ~9.5k** rows with **no** val holdout when eval is off anyway.

---

## 4. Current run — configuration checklist

### Data

| Item | Setting |
|------|---------|
| `train_test_split` | **Disabled** — use full `train.csv` (~9.5k rows) |
| `val_df` / `val_dataset` | **Not used** |
| Load cell | Print **`train_df`** and **`test_df`** only; **no** `val_df` print |

### LoRA (config cell — confirm values in notebook)

| Item | This run |
|------|----------|
| `LORA_RANK` | 32 |
| `LORA_ALPHA` | 16 |
| `LORA_DROPOUT` | **0** |

### Training (`TrainingArguments` + `SFTTrainer`)

| Item | This run |
|------|----------|
| `NUM_EPOCHS` | 1 (confirm in notebook) |
| `LR` | 2e-4 (confirm) |
| `MAX_SEQ_LEN` | 1024 (confirm) |
| `warmup_ratio` | **0.03** |
| `lr_scheduler_type` | cosine |
| `optim` | **`adamw_8bit`** |
| `per_device_train_batch_size` | **8** |
| `gradient_accumulation_steps` | **3** |
| Effective batch (per device) | **8 × 3 = 24** |
| `bf16` | True |
| `seed` | 42 |
| `logging_steps` | 50 |
| `save_strategy` | epoch |
| `report_to` | none |
| `dataloader_num_workers` | 4 |
| `eval_strategy` | **`no`** |
| `eval_steps` | **Omitted** (not used with `eval_strategy="no"`) |
| `eval_dataset` | **`None`** (Python `None`, not the string `"None"`) |

### Dataset → trainer

| Item | Setting |
|------|---------|
| `train_dataset.map` | Adds **`text`** via `tokenizer.apply_chat_template(...)` |
| `val_dataset.map` | **Commented out** |
| `dataset_text_field` | **`"text"`** |
| `dataset_kwargs` | `{"skip_prepare_dataset": False}` |

### Submission artifact

- Save LoRA adapter and zip per competition rules (`adapter_config.json` + weights in `submission.zip`).

---

## 5. File map

| Notebook | Role |
|----------|------|
| `nvidia-nemotron-training-copy-run-instantly.ipynb` | Fast runs, subsampled data |
| `unsloth-training-full-data.ipynb` | Full-data Unsloth SFT (this log’s focus) |
| `reasoning-challenge-starter-notebook.ipynb` | Alternative / reference pipeline |

---

*Update this file after each Kaggle submit with date, LB score, and any delta from §4.*
