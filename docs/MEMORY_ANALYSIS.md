# Memory Analysis: Nemotron-3-Nano-30B-A3B-BF16 with Rank 32 LoRA

> Source: Tong Hui Kang — theoretical memory limits for training on 8192-length sequences.
> Note: these are theoretical estimates; not all configurations have been proven in practice.

## Memory Requirements by Microbatch Size (μ)


| Component                      | Formula           | μ=1         | μ=4          | μ=16         | μ=64         |
| ------------------------------ | ----------------- | ----------- | ------------ | ------------ | ------------ |
| Base model weights (BF16)      | W × 2             | 63.6 GB     | 63.6 GB      | 63.6 GB      | 63.6 GB      |
| LoRA adapter weights (FP32)    | P × 4             | 3.5 GB      | 3.5 GB       | 3.5 GB       | 3.5 GB       |
| LoRA gradients (FP32)          | P × 4             | 3.5 GB      | 3.5 GB       | 3.5 GB       | 3.5 GB       |
| Optimizer m + v (FP32)         | P × 8             | 7.1 GB      | 7.1 GB       | 7.1 GB       | 7.1 GB       |
| CUDA context & buffers         | ~3 GB             | 3.0 GB      | 3.0 GB       | 3.0 GB       | 3.0 GB       |
| Checkpointed layer inputs      | L × μ × S × H × 2 | 2.3 GB      | 9.2 GB       | 36.6 GB      | 146.6 GB     |
| Peak intra-layer intermediates | μ × S × D × 2     | 331 MB      | 1.3 GB       | 5.3 GB       | 21.2 GB      |
| Backward intra-layer gradient  | μ × S × H × 2     | 44 MB       | 176 MB       | 704 MB       | 2.8 GB       |
| Logits — unchunked             | μ × S × V × 2     | 2.1 GB      | 8.6 GB       | 34.4 GB      | 137.4 GB     |
| **TOTAL (unchunked logits)**   |                   | **85.6 GB** | **100.1 GB** | **157.8 GB** | **388.8 GB** |
| Logits — fused CE              | μ × S × 4         | 32 KB       | 128 KB       | 512 KB       | 2 MB         |
| **TOTAL (fused CE)**           |                   | **83.5 GB** | **91.5 GB**  | **123.5 GB** | **251.4 GB** |


> RTX Pro 6000 (Kaggle) has **96 GB VRAM** → μ=1 and μ=4 fit with fused CE; μ=16 does not.

---

## LoRA Parameter Count

### Attention layers (× 6)


| Weight       | Shape        | Params per adapter |
| ------------ | ------------ | ------------------ |
| q_proj       | [2688, 4096] | 217,088            |
| k_proj       | [2688, 256]  | 94,208             |
| v_proj       | [2688, 256]  | 94,208             |
| o_proj       | [4096, 2688] | 217,088            |
| **Subtotal** |              | **3.74M**          |


### Mamba-2 layers (× 23)


| Weight       | Shape         | Params per adapter |
| ------------ | ------------- | ------------------ |
| in_proj      | [2688, 12864] | 497,664            |
| out_proj     | [5376, 2688]  | 258,048            |
| **Subtotal** |               | **17.38M**         |


### MoE routed experts (× 2944)


| Weight       | Shape        | Params per adapter |
| ------------ | ------------ | ------------------ |
| experts.fc1  | [2688, 1856] | 145,408            |
| experts.fc2  | [1856, 2688] | 145,408            |
| **Subtotal** |              | **856.16M**        |


### Other (possible targets)


| Layer      | Weight                 | Params |
| ---------- | ---------------------- | ------ |
| MoE shared | shared_experts.fc1/fc2 | 9.42M  |
| MoE router | gate                   | 2.07M  |
| Output     | lm_head                | 4.28M  |
| Embedding  | embed_tokens           | 4.28M  |



|                  | Typical     | Possible    |
| ---------------- | ----------- | ----------- |
| **TOTAL params** | **877.28M** | **897.33M** |
| **FP32 size**    | **3.51 GB** | **3.59 GB** |


---

## Training Throughput

**Per-sample compute:**

- Forward pass: 3.5B active params × 8192 tokens × 2 = **57 TFLOP**
- Backward pass (with gradient checkpointing): 3× forward = **171 TFLOP**
- Total per sample: **228 TFLOP**


| GPU          | BF16 TFLOPS | HBM Bandwidth | Critical Arithmetic Intensity |
| ------------ | ----------- | ------------- | ----------------------------- |
| H200         | 990         | 4.8 TB/s      | 206 FLOPs/byte                |
| RTX Pro 6000 | 252         | 1.15 TB/s     | 219 FLOPs/byte                |


**Theoretical minimum time per sequence (100% efficiency):**

- H200: 228 / 990 = **0.23 sec**
- RTX Pro 6000: 228 / 252 = **0.90 sec**

> In practice, MFU is 30–40%. End-to-end single-GPU training achieves 8–20% MFU due to non-matmul overhead between kernels.

---

## Glossary


| Symbol | Meaning               | Value                                            |
| ------ | --------------------- | ------------------------------------------------ |
| μ      | Microbatch size       | samples per forward/backward; B = μ × grad_accum |
| S      | Sequence length       | 8,192 tokens                                     |
| V      | Vocabulary size       | 131,072                                          |
| H      | Hidden dimension      | 2,688                                            |
| L      | Number of layers      | 52 (23 Mamba-2 + 23 MoE + 6 GQA attention)       |
| R      | LoRA rank             | 32                                               |
| B      | Global batch size     | 64                                               |
| W      | Base model params     | 31.8B                                            |
| P      | LoRA trainable params | 886.7M                                           |
| D      | MoE intra-layer width | 20,224 = 6 × 1856 + 3712 + 2×2688                |


---

## Key Observations

1. **Kaggle RTX Pro 6000 has 96 GB VRAM** — μ=1 (83.5 GB) and μ=4 (91.5 GB) fit with fused CE; μ=16 (123.5 GB) does not
2. **Unsloth uses fused cross entropy** — avoids storing full logit matrix (μ × S × V × 2), saving ~8.5 GB at μ=4
3. **Adapter weights are FP32** (~3.5 GB) even though base model is BF16
4. **Nemotron supports Flash Attention** — no quadratic memory for attention mechanism
5. **Gradient checkpointing** — trades compute for memory; backward pass costs 3× forward

---

## References

1. [Unsloth — LoRA hyperparameters guide (effective batch size)](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide#effective-batch-size)
2. [Unsloth — Nemotron-3 fine-tuning](https://unsloth.ai/docs/models/nemotron-3#fine-tuning-nemotron-3-and-rl)
3. [Unsloth — Fused cross entropy](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning#unsloth-loss-refactoring-chunk-and-fuse)
4. [Transformers PR — Flash Attention for Nemotron](https://github.com/huggingface/transformers/pull/44390/changes)
5. [JAX scaling book — training FLOPs](https://jax-ml.github.io/scaling-book/transformers/)
6. Gu & Dao, "Mamba" (2023) — [arxiv:2312.00752](https://arxiv.org/abs/2312.00752)
7. Gale et al., "MegaBlocks" (2022) — [arxiv:2211.15841](https://arxiv.org/abs/2211.15841)
8. [ml-engineering — training performance](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md)

