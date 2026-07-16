# Project: Sprout

A language model built from scratch — tokenizer, architecture, training loop, and SFT pipeline — trained entirely on a single consumer GPU (RTX 4070, 12GB VRAM).

## Why

I'd used LLMs for a long time without really understanding what was happening inside them. After watching Andrej Karpathy's NanoGPT and NanoChat series, I wanted to answer two questions for myself: could I build and train a language model completely locally and get something *usable* out of it, and how much do the architectural advances since GPT-2 actually matter in practice — not in a paper's ablation table, but on my own hardware, with my own data.

So Sprout is both a from-scratch implementation and a running experiment: start with a GPT-2-style baseline, swap in the modern component stack one piece at a time, scale up, and see how far a 12GB card can take you.

## The Story: Vanilla → Modern → Sapling

The project moved through three stages, each answering a different question. All models were trained on the same pretraining data (45% FineWeb-Edu, 45% DCLM Baseline, 6% FineMath-Edu, 4% Olmo Mix-StarCoder), with a context length of 1024 and a custom byte-level BPE tokenizer (32,767 tokens).

**Vanilla** — a GPT-2-style baseline (~110M params): standard multi-head attention, learned positional embeddings, GELU FFN, AdamW + cosine schedule. This model acts as a baseline for comparison, and answers the question *"does this work at all?"*

**Modern (untuned)** — the same scale (~100M params), but with the current-generation component stack swapped in: RMSNorm, RoPE, GQA, QK-Norm, ReLU², Muon+AdamW, and a WSD schedule. This isolates the effect of the architecture and optimizer changes alone, independent of scale.

**Sapling** — the flagship model (134M params, matching SmolLM2-135M for comparison). Modern stack, further tuned, trained on 15B tokens with a wider architecture chosen after an ablation study (below), then instruction-tuned via SFT to produce a chat-capable model.

## Results

![Benchmark comparison](sprout/logs/plots/model_comparison.png)

Sapling reaches a 33.5% average across LAMBADA, ARC-Easy, ARC-Challenge, and HellaSwag — ahead of GPT-2 Small (30.8%) at a comparable parameter count, and a clear step up over both the vanilla baseline (27.4%) and the untuned modern model (30.2%). The gap between vanilla and modern-untuned shows what the architecture swap alone bought; the gap between modern-untuned and Sapling shows what additional scale and tuning bought on top of that.

### Architecture ablation: why wide-and-shallow

Conventional wisdom (MobileLLM, SmolLM2) says deeper-and-narrower wins for small language models — but usually that result shows up late in training, after a lot of tokens. Compute-constrained to a single GPU, I cared more about tokens-per-wall-clock-hour than the theoretical optimum, so I ran a short ablation before committing to Sapling's final shape:

![Architecture ablation](sprout/logs/plots/architecture_ablation.png)

Three architectures — wide (8 layers, d=1024), medium (16 layers, d=768), and deep (30 layers, d=576, close to SmolLM2's own shape) — trained for a fixed ~1000 steps. At equal wall-clock time, wide was consistently ahead on benchmark accuracy, and the token-budget panel shows why: in any fixed training window, the wide model simply gets through more data — roughly 3–4B more tokens than deep in the same 24–72 hours. Given the deep model's theoretical advantage mostly pays off over very long horizons I can't afford locally, I took the throughput win and went wide for Sapling. It's an acknowledged tradeoff, not a free lunch — the ablation itself is a bit informal (batch sizes weren't matched across runs), but it was directionally decisive enough to commit to.

### Training dynamics

![Sapling training and eval loss](sprout/logs/plots/sapling/loss.png)

Loss curves for the full 15B-token Sapling pretraining run.

## Talking to Sapling

After SFT, Sapling holds a short multi-turn conversation, follows instructions, and stays roughly on-topic — a genuinely useful sanity check for a 134M-parameter model trained on one GPU:

![Demo conversation with Sapling](sprout/logs/plots/demo_conversation.png)

## Architecture

| | Vanilla | Modern (untuned) | Sapling |
|---|---|---|---|
| **Architecture** | | | |
| Parameters | ~110M | ~100M | 134M |
| `d_model` | 768 | 768 | 1024 |
| `num_layers` | 12 | 12 | 8 |
| `num_heads` (Q) | 12 | 12 | 16 |
| `num_kv_heads` | — (MHA) | 4 (GQA) | 4 (GQA) |
| `mlp_expansion` | 4× (GELU) | 4× (ReLU²) | 4.75× (ReLU²) |
| Position encoding | Learned | RoPE (base 5,000) | RoPE (base 5,000) |
| Normalization | LayerNorm | RMSNorm + QK-Norm | RMSNorm + QK-Norm |
| `max_seq_len` | 1024 | 1024 | 1024 |
| `vocab_size` | 32,768 | 32,768 | 32,768 |
| **Optimizer** | | | |
| Optimizer | AdamW | Muon + AdamW | Muon + AdamW |
| `max_learning_rate` | 6e-4 | 6e-4 | 1e-3 |
| Schedule | Cosine | WSD (2% warmup / 10% decay) | WSD (2% warmup / 10% decay) |
| `weight_decay` | 0.1 | 0.1 | 0.1 |
| `betas` | (0.9, 0.95) | (0.9, 0.95) | (0.9, 0.95) |
| **Training** | | | |
| Pretraining tokens | 2.5B | 2.5B | 15B |
| Effective batch size | ~500,000 | same | same |
| `grad_clip` | 1.0 | 1.0 | 1.0 |

## Pipeline

1. **Tokenizer** — byte-level BPE trained from scratch, including a from-scratch implementation of the training algorithm (linked-list merge tracking + lazy-deletion max-heap for pair counts, since the naive approach doesn't scale to real corpora).
2. **Data processing** — streaming tokenization and sharding of FineWeb-Edu (pretraining) and smol-smoltalk (SFT) into fixed-size shards, with resumable metadata tracking.
3. **Pretraining** — custom training loop: BF16 mixed precision, `torch.compile`, FlashAttention 2, gradient clipping, WSD learning rate schedule, Muon for 2D weight matrices with AdamW for embeddings/head/norms.
4. **Evaluation** — HellaSwag, ARC-Easy, ARC-Challenge, LAMBADA via log-likelihood scoring.
5. **SFT** — ChatML-formatted conversational fine-tuning with role-conditional loss masking and conversation-respecting sequence packing.

## Hardware & Scale

Every training run was done on a single RTX 4070 (12GB VRAM). The 15B-token Sapling pretraining run alone took roughly 87 hours. 

---

Built by Philipp.