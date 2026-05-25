# Muon is Scalable for LLM Training scaled reproduction

Mode: reproduction
Source: https://arxiv.org/pdf/2502.16982
Created: 2026-05-25T00:08:55-07:00

## Goal

Understand what the Moonshot AI "Muon is Scalable for LLM Training" report is claiming and run a single-RTX-5090 scaled reproduction that isolates the optimizer mechanics we can test locally:

- Newton-Schulz orthogonalization behavior.
- Muon's shape-dependent update RMS and the paper's `0.2 * sqrt(max(A, B))` correction.
- A small causal language-model training comparison between AdamW, paper-style Muon, Muon without weight decay, and Muon without update-RMS scaling.

The local result is intended to explain the mechanism and look for the same direction of effect at toy scale. It cannot reproduce Moonshot's full 3B/16B MoE, 5.7T-token scaling-law result on one GPU.

## Source Context

- Upstream repo: https://github.com/MoonshotAI/Moonlight
- Local upstream commit: `c2ad5b20c605086526a179d36901bfc41b52b44b`
- Paper: https://arxiv.org/abs/2502.16982 and local `Moonlight.pdf`
- Model card: https://huggingface.co/moonshotai/Moonlight-16B-A3B
- Original Muon repo/blog: https://github.com/KellerJordan/Muon, https://kellerjordan.github.io/posts/muon/
- Distributed Muon proof-of-concept linked by Moonlight README: https://github.com/NVIDIA/Megatron-LM/pull/1428
- License: MIT (`LICENSE`)

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB VRAM
- Driver/CUDA: NVIDIA driver 580.159.03, CUDA runtime reported by `nvidia-smi` 13.0
- Python: 3.13.5 from `/home/ye/miniconda3/bin/python3`
- Framework: torch 2.8.0+cu128, transformers 5.8.0.dev0, datasets 4.4.1
- Initial GPU contention: `llama-server` PID 4111641 used about 27.5 GiB VRAM at setup time.

## Milestones

- [x] Source/context captured
- [x] Environment inspected
- [x] Smoke test completed
- [x] Main experiment completed
- [x] Report written

## Run Log

Append notable runs here and keep machine-readable records in `experiments/runs.jsonl`.

- 2026-05-25 00:08 PDT: cloned `MoonshotAI/Moonlight` at `c2ad5b20c605086526a179d36901bfc41b52b44b`, created branch `reproduce/muon-scaled`, bootstrapped standard experiment artifacts.
- 2026-05-25 00:13 PDT: ran mechanics probe; outputs in `artifacts/probes/` and `reports/figures/muon_rms_probe.png`.
- 2026-05-25 00:14 PDT: ran 300-step OpenWebText optimizer suite; outputs in `artifacts/runs/*_300/`.
- 2026-05-25 00:15-00:18 PDT: ran 3,000-step OpenWebText optimizer suite; outputs in `artifacts/runs/*_3000/`.
- 2026-05-25 00:18 PDT: generated `reports/results-summary.md`, figures, and `reports/final-report.md`.
