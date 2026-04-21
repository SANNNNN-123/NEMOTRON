# LoRA Target Module Ablation Study

> Single-module freeze/unfreeze ablation on 450 samples (mini_eval), 50 training steps, 500 cryptarithm-only samples.
> Goal: identify which modules are safe to train without causing baseline decay.

## Results

| freeze_except | Total % | Delta vs Baseline | Cryptarithm Deduce | Cryptarithm Guess | Baseline Decay |
|---------------|:-------:|:-----------------:|:------------------:|:-----------------:|:--------------|
| **shared_experts** | **68.0%** | **0.0%** | 8.0% | 6.0% | **None (perfect preserve)** |
| **out_proj** | **68.0%** | **0.0%** | 8.0% | 6.0% | **None (perfect preserve)** |
| lm_head | 67.6% | -0.4% | 8.0% | 6.0% | Minimal |
| v_proj | 65.1% | -2.9% | 0.0% | 0.0% | Moderate |
| q_proj | 57.6% | -10.4% | 0.0% | 2.0% | Severe (bit_manip 92→24%) |
| down_proj | 32.4% | -35.6% | 0.0% | 0.0% | Catastrophic |
| up_proj | 32.2% | -35.8% | 0.0% | 0.0% | Catastrophic |
| in_proj | 29.6% | -38.4% | 0.0% | 0.0% | Catastrophic |
| o_proj | 25.1% | -42.9% | 0.0% | 0.0% | Catastrophic |
| none (all trainable) | 27.1% | -40.9% | 2.0% | 0.0% | Catastrophic |

## Key Findings

1. **No target improved cryptarithm** — all stay at or near 8%/6% baseline. 50 steps on 500 cryptarithm samples isn't enough to move the needle on any single target.
2. **Safest targets:** `shared_experts` and `out_proj` — zero baseline decay (perfect preserve).
3. **Moderate safety:** `lm_head` (0.4% decay), `v_proj` (2.9% decay but kills cryptarithm).
4. **Catastrophic:** `in_proj`, `up_proj`, `down_proj`, `o_proj`, `none` — all cause 30–43% total collapse.
5. **The real problem:** 50 steps on 500 cryptarithm-only samples isn't enough to learn cryptarithm reasoning regardless of which layer is unfrozen. The model needs either:
   - Significantly more training steps or data
   - A mix of module types (e.g., `shared_experts + out_proj` together)
   - Higher learning rate or different training schedule
   - Training on a broader dataset that includes cryptarithm alongside other domains

## Conclusion

**Layer choice doesn't fix cryptarithm** — a fundamentally different training strategy is needed (more data, more steps, or multi-target unfreezing).

## Implication for Our Setup

Our current `TARGET_MODULES`:
```python
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj", "out_proj", "up_proj", "down_proj",
    "lm_head",
]
```

**`shared_experts` is missing** — the safest and most effective single target per this ablation.
The MoE shared expert layers handle routing which is critical for reasoning tasks.
**Recommendation:** Add `shared_experts` to TARGET_MODULES in the next experiment.

> Note: this ablation used single-module training with very limited data/steps.
> Multi-module training (our setup) is generally more stable — catastrophic results above
> may not apply when all modules are trained together with diverse data.

---

## shared_experts Implementation — Architecture & Loading Notes

### Nemotron-3-Nano Architecture (from paper)
- **Number of Shared Experts:** 2 (always active, non-routed)
- **Expert Dimension:** 1,856
- **Total Routable Experts:** 128 (only 6 activated per token)

### shared_experts internal structure
`shared_experts` is a `NemotronHMLP` container — **not directly targetable by PEFT**.
It contains two linear layers that implement both shared experts packed together:

```
NemotronHMLP(
  (up_proj):  Linear(2688 → 3712)   # 3712 = 2 × 1856 (2 experts × Expert Dimension)
  (down_proj): Linear(3712 → 2688)
)
```

PEFT must target the internal linear layers, not the container. Using `'shared_experts' in name`
as a freeze filter correctly isolates only these layers.

### Verified trainable parameter names
```
base_model.model.backbone.layers.1.mixer.shared_experts.up_proj.lora_A.default.weight
base_model.model.backbone.layers.1.mixer.shared_experts.up_proj.lora_B.default.weight
base_model.model.backbone.layers.1.mixer.shared_experts.down_proj.lora_A.default.weight
base_model.model.backbone.layers.1.mixer.shared_experts.down_proj.lora_B.default.weight
```

**Trainable params:** 9,420,800 (23 MoE layers × 409,600 per layer)
```
Per layer: up_proj (32×2688 + 3712×32) + down_proj (32×3712 + 2688×32) = 409,600
23 × 409,600 = 9,420,800 ✓
```

### Warm-start from 0.86 adapter — key naming fix
The 0.86 adapter was trained with an older Unsloth version that used a different internal path:
```
Old (0.86 adapter): base_model.model.model.layers.X...
New (current):      base_model.model.backbone.layers.X...
```

Fix: remap keys before loading via `str.replace("model.model.", "model.backbone.")`.
After fix: **12010/12010 weights loaded** (full warm-start).

### Warm-start + shared_experts fine-tuning strategy
```
1. Load base Nemotron-3-Nano (31.8B BF16)
2. Apply LoRA on 9 original modules + shared_experts
3. Load 0.86 adapter weights (remapped) → 9 modules warm-started
4. shared_experts LoRA stays randomly initialized (46 missing keys — expected)
5. Freeze all 9 original modules (878,733,312 params frozen)
6. Train only shared_experts (9,420,800 params) on v3 corrected data
```

**Goal:** preserve 0.86 quality in frozen modules, add new capacity via shared_experts.
