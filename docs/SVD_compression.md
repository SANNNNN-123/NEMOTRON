# SVD Compression & LoRA Adapter Ensembling

Source: Competition discussion forum post

---

## Overview

Two practical techniques for efficient fine-tuning:
- SVD for rank compression
- Ensembling LoRA adapters

> Note: these methods did not improve the author's public leaderboard score, but they also did not degrade performance. They are mainly useful for efficiency and real-world deployment.

---

## SVD Compression for LoRA

Many solutions use Singular Value Decomposition (SVD) to compress high-rank LoRA adapters down to rank ≤ 32 (the inference constraint).

**Training with a higher rank (e.g., 96, 128, 256) is often beneficial:**
- better performance during training
- less constrained optimization

**Then compressing to a lower rank (e.g., 32):**
- keeps similar performance (or slightly worse)
- improves efficiency for inference

A naive SVD implementation is slow and expensive, even on GPU.

### SVD Measurements

| Method                  | Time    | Score |
|-------------------------|---------|-------|
| Naive SVD (GPU)         | ~1h30m  | 0.86  |
| Optimized SVD (Alg. 2)  | ~5m     | 0.86  |

Reference: https://arxiv.org/pdf/2602.10993 (Algorithm 2 is the key part)

### adapter_config.json After Compression

After compression, you must update adapter_config.json.

Example:
- trained:    rank = 96, alpha = 192
- compressed: rank = 32, alpha = 64

**Keep the same ratio between alpha and rank.**

If alpha ≈ 2 × rank before, keep it the same after compression.
In vLLM, alpha affects decoding behavior — not adjusting it leads to inconsistent and very poor results.

---

## LoRA Ensembling

Train multiple LoRA adapters with different random seeds.

### Single Model Scores

| Seed   | Score |
|--------|-------|
| Seed 1 | 0.86  |
| Seed 2 | 0.84  |
| Seed 3 | 0.85  |

### Ensembling / Merging Results

| Method             | Score |
|--------------------|-------|
| Simple averaging   | ≤ 0.5 |
| TIES / DARE merging| ≤ 0.5 |
| SeedLoRA (3 seeds) | 0.86  |

Naive approaches (averaging or merging like TIES/DARE) failed badly in this setup.

Use SeedLoRA instead: https://openreview.net/forum?id=jkCvAAcSDa
Ensembling across seeds is much more stable and avoids collapse.

---

## Takeaways

- SVD compression helps meet rank constraints efficiently
- Implementation matters a lot (huge time savings — Alg. 2 is 18x faster than naive)
- Keep alpha/rank ratio consistent after compression
- Naive LoRA merging can fail badly (≤ 0.5 score)
- Seed-based ensembling (SeedLoRA) is more reliable

---

## Our Implementation

Notebook: `svd-compress-and-submit.ipynb`

Config used:
- Training rank: 64, alpha: 128
- Compressed rank: 32, alpha: 64 (ratio 2.0 preserved)
- SVD target: Algorithm 2 approach (optimized)
- Device: GPU (auto-detects, falls back to CPU)

Key fix: all output tensors must call `.contiguous()` before saving to safetensors
or it raises `ValueError: non contiguous tensor`.
