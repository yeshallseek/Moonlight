# Source Review: Muon is Scalable for LLM Training scaled reproduction

Mode: reproduction
Initial source: https://arxiv.org/pdf/2502.16982

## Source Map

| Priority | Source | Version/Commit | Why It Matters |
| --- | --- | --- | --- |
| Official paper | https://arxiv.org/abs/2502.16982 / `Moonlight.pdf` | arXiv v1, submitted 2025-02-24 | User-provided paper; defines the claims, algorithmic changes, ablations, and limits. |
| Official repo | https://github.com/MoonshotAI/Moonlight | local commit `c2ad5b20c605086526a179d36901bfc41b52b44b` | Contains README, paper PDF, figures, requirements, and `examples/toy_train.py` with the paper-style Muon toy trainer. |
| Official model card | https://huggingface.co/moonshotai/Moonlight-16B-A3B | accessed 2026-05-25 | Documents model identity, tags, license, and inference path for the released Moonlight checkpoint. |
| Original Muon implementation | https://github.com/KellerJordan/Muon | remote HEAD `f98f1cacc0263b04290753e32be8d498c1efc806` | Defines Muon as an optimizer for hidden-layer matrix parameters, with AdamW for embeddings/heads/gains/biases. |
| Original Muon design writeup | https://kellerjordan.github.io/posts/muon/ | 2024-12-08 | Explains Newton-Schulz orthogonalization and motivation for 2D hidden parameters. |
| Distributed implementation pointer | https://github.com/NVIDIA/Megatron-LM/pull/1428 | PR head `8aee20bdad4a2561c1418ea1ecf09348027effe8` | Linked by Moonlight README as Megatron-LM proof-of-concept for Distributed Muon; useful for implementation context, not required for one-GPU toy runs. |
| Intermediate checkpoints note | `Moonlight_intermediate_checkpoints.pdf` | local repo artifact | Lists Google Drive checkpoint naming and training schedule; checkpoint download/evaluation is out of scope for the first local run. |

## Source Priority

- Official paper/project/repo/model/data sources: arXiv paper, MoonshotAI/Moonlight repo, Moonlight Hugging Face model card, local intermediate-checkpoint PDF.
- Maintainer or dependency docs: KellerJordan/Muon repo and design writeup; Megatron-LM PR linked by Moonlight README; Hugging Face Transformers/Datasets APIs as needed.
- Third-party reproductions, forks, blogs, or forum notes: none used for experiment design.
- Unresolved candidate sources: paper says intermediate checkpoints are released, while README still says "Coming soon"; the local PDF points to a Google Drive folder. We will not depend on those large checkpoints for the first report.

## Objective

Help the user understand why the paper says Muon can scale for LLM training and test the parts that fit locally: matrix update orthogonalization, shape-dependent update RMS, the paper's RMS correction, and whether those choices produce a measurable training-loss/validation-loss difference in a tiny language-model run.

## Key Claims or Features

- Muon updates matrix parameters by applying momentum and then approximate orthogonalization with Newton-Schulz iteration.
- The Moonlight paper identifies two scaling changes: add AdamW-style decoupled weight decay and scale each matrix update by `0.2 * sqrt(max(A, B))`.
- The paper's Lemma 1 says an unscaled full-rank Muon update for an `[A, B]` matrix has theoretical RMS about `1 / sqrt(max(A, B))`; the correction should make RMS approximately `0.2` across matrix shapes.
- Moonshot reports Muon with those changes achieved about 52% of AdamW training FLOPs at compute-optimal scale and trained a 3B-active/16B-total MoE model on 5.7T tokens.
- The official README exposes a toy training command for a Qwen-like dense model on `openwebtext-100k` comparing Muon and AdamW.

## Repo Architecture

- Entry points: `examples/toy_train.py` is the official toy trainer; new local scripts will live under `scripts/` and write outputs under `artifacts/`.
- Important configs: optimizer choices (`adamw`, `muon`, Muon no-weight-decay, Muon no-RMS-scale), learning rate, weight decay, model width/layers, sequence length, max steps, seed.
- Model/data paths: official example uses `Qwen/Qwen2.5-0.5B` tokenizer and `Elriggs/openwebtext-100k`; local scaled runs may cache tokenized data under `artifacts/datasets/`.
- Tests/examples: no upstream automated tests. Verification comes from smoke commands, generated metrics, and reproducible run records.

## Execution Path

Smallest faithful command or code path to run first:

```bash
python3 scripts/probe_muon_mechanics.py --device cuda --out-dir artifacts/probes
```

Then run a tiny LM smoke comparison before scaling steps:

```bash
python3 scripts/train_tiny_qwen_muon.py --optimizer adamw --max-steps 5 --max-texts 64 --out-dir artifacts/smoke/adamw
python3 scripts/train_tiny_qwen_muon.py --optimizer muon --max-steps 5 --max-texts 64 --out-dir artifacts/smoke/muon
```

## Minimal Causal Experiment

- Simplest faithful experiment: run a mechanics probe on random matrices across several shapes, then train the same tiny Qwen-like causal LM for fixed steps/seeds with AdamW and Muon variants.
- Proposed control runs: AdamW baseline; paper-style Muon (`wd=0.1`, RMS scaling enabled); Muon without weight decay; Muon without RMS scaling.
- Effect being isolated: whether Moonlight's two claimed scaling changes actually normalize update size and improve small-run optimization behavior relative to obvious controls.
- What would count as success: probe RMS matches the theoretical `1 / sqrt(max(A, B))` before scaling and approximately `0.2` after scaling; training commands run reproducibly and produce loss curves/validation loss that can be interpreted without crashes or NaNs.
- What would falsify the local mechanism claim: update RMS does not follow shape theory, RMS scaling fails to normalize across shapes, or Muon variants show instability/NaNs under conditions where AdamW remains stable.

## RTX 5090 Fit

- Expected VRAM: mechanics probe <1 GiB; tiny LM run should fit in <8 GiB with width 128-256, 2-4 layers, sequence length 128-256, batch 8-16. Official `hidden_size=896` toy config may be much heavier and is not the first target.
- Precision: FP32 parameters with CUDA autocast BF16 for model forward/backward when available; Muon Newton-Schulz internally uses BF16 like the official example.
- Batch/sequence settings: start with 5-step smoke, then 100-300 training steps per optimizer if runtime is acceptable.
- Runtime: mechanics probe seconds; smoke minutes; full tiny comparison expected tens of minutes depending on tokenizer/dataset download and model size.
- Disk: repo is ~7 MiB before caches; tokenizer/dataset cache can grow into several GiB under Hugging Face cache and `artifacts/datasets/`.
- Scaling changes from paper/upstream: one GPU, tiny dense model, tiny token budget, public small dataset. We will not claim reproduction of the 52% FLOP result or Moonlight benchmark tables.

## Risks and Ambiguities

- The official toy script has no max-step or validation mode and writes a root `openwebtext-100k.bin`; local scripts need bounded runs and artifact paths.
- The current Python environment differs from upstream requirements (`torch 2.8.0+cu128`, `transformers 5.8.0.dev0`, `datasets 4.4.1` instead of pinned torch 2.6.0/transformers 4.49.0/datasets 3.3.2).
- GPU VRAM is currently mostly occupied by a `llama-server`; if training needs GPU memory, it must be stopped and recorded.
- Public toy data and tiny model scale may not show the same long-horizon overtraining behavior as the paper's 800M/100B-token weight-decay ablation.
- Tokenizer/model downloads need network and may fail or change; all versions and cache paths should be recorded.

## Decision

Proceed with caveats. The local experiments can test mechanisms and small-run behavior, but the final report must distinguish mechanism validation from full scaling-law reproduction.
