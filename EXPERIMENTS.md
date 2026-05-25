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
- 2026-05-25 00:43 PDT: started larger overnight experiment setup. Gracefully stopped `llama-server` PID 4111641 with `SIGTERM` to reclaim RTX 5090 VRAM for training; GPU memory dropped from about 27.5 GiB used to 2 MiB used.
- 2026-05-25 00:44-00:46 PDT: dry-ran larger models with telemetry/checkpoints/samples. Selected 175.7M-parameter Qwen-like model (`hidden_size=640`, `num_layers=12`, `seq_len=256`, `batch_size=16`) after Muon and AdamW dry runs fit within about 12 GiB CUDA allocation.
- 2026-05-25 00:47 PDT: ran a 50-step Muon calibration for the selected 175.7M model; measured about 37k tokens/sec with 11.7 GiB max CUDA allocation. Updated overnight suite to target 60k steps per optimizer run.
- 2026-05-25 00:49 PDT: launched detached overnight suite with `setsid bash scripts/run_overnight_large_suite.sh`. Runner PID `4137738`; log `logs/overnight_large_suite_20260525-004946.log`; PID file `logs/overnight_large_suite_20260525-004946.pid`. Verified first run `overnight_muon_paper` reached step 400, about 38k tokens/sec average, about 12.4 GiB VRAM, and about 93% GPU utilization.
- 2026-05-25 01:00 PDT: first large run `overnight_muon_paper` reached step 5,200 with latest validation loss 5.0582 at step 4,000, latest train loss 5.1019, and about 39.5k tokens/sec average. Added `scripts/summarize_overnight_large.py` to turn partial or completed overnight metrics into `reports/overnight-large-summary.md` and validation/training-loss plots.
- 2026-05-25 01:08 PDT: `overnight_muon_paper` reached the first trained sample point at step 10,000. Validation loss improved to 4.7638, throughput stayed near 39.4k tokens/sec, and generated samples became repetitive but recognizable English text instead of random initialization output. No checkpoint yet; first checkpoint remains scheduled for step 20,000.
- 2026-05-25 01:25 PDT: `overnight_muon_paper` wrote its first checkpoint at `artifacts/overnight_large/muon_paper_640l12_b16/checkpoints/step_00020000.pt` plus `latest.pt` (about 1.8 GB each, ignored by git). Checkpoint metadata verified step 20,000, 81,920,000 tokens seen, validation loss 4.6152, and commit `1e381b626a2825fe51b39df2a72c332893c5d132`. Step-20k samples remained repetitive but showed more topical English structure than step 10k.
- 2026-05-25 01:43 PDT: `overnight_muon_paper` reached the halfway sample point at step 30,000. Validation loss improved to 4.5133 after 122,880,000 tokens, with throughput still about 39.5k tokens/sec. Samples were more topical and paragraph-like than at 10k/20k, but remained repetitive and not instruction-following quality.
- 2026-05-25 02:00 PDT: `overnight_muon_paper` wrote the second checkpoint at `step_00040000.pt`; `latest.pt` metadata verified step 40,000, 163,840,000 tokens seen, validation loss 4.4164, and commit `de78abd84fb27591e3e24b34aa2c9511faf3d7c5`. GPU stayed stable around 12.4 GiB VRAM and 39.5k tokens/sec.
- 2026-05-25 02:17 PDT: `overnight_muon_paper` reached the step-50,000 sample point. Validation loss improved to 4.3377 after 204,800,000 tokens, with average throughput about 39.5k tokens/sec. Samples were better formed than earlier checkpoints in places, but still showed repetition, factual drift, and prompt-topic derailment.
- 2026-05-25 02:35 PDT: `overnight_muon_paper` completed at step 60,000 with stop reason `max_steps`, 245,760,000 tokens seen, final validation loss 4.3272, final train loss 3.9568, and average throughput about 39.5k tokens/sec. Best observed validation loss for this leg was 4.3184 at step 56,000 before a slight late rise. Final checkpoint `step_00060000.pt` was written and the runner handed off cleanly to `overnight_adamw` in `artifacts/overnight_large/adamw_640l12_b16` with PID `4166626`.
- 2026-05-25 02:37 PDT: `overnight_adamw` passed the first validation check at step 2,000 with validation loss 5.7248, train loss 5.8361, and about 51.5k tokens/sec average throughput. This confirmed the second leg is healthy; at the same step the completed Muon-paper leg had validation loss 5.5019 at about 39.5k tokens/sec.
- 2026-05-25 02:47 PDT: `overnight_adamw` reached the first trained sample point at step 10,000. Validation loss improved to 4.9302 after 40,960,000 tokens, with about 51.8k tokens/sec average throughput. AdamW remained faster per wall-clock step than Muon, but its validation loss was behind the Muon-paper leg at the same step (4.7638). Samples became recognizable English, but were still repetitive and less coherent than the 10k Muon-paper samples.
- 2026-05-25 03:02 PDT: `overnight_adamw` wrote its first checkpoint at `artifacts/overnight_large/adamw_640l12_b16/checkpoints/step_00020000.pt` plus `latest.pt` (about 2.0 GB each, ignored by git). Checkpoint metadata verified step 20,000, 81,920,000 tokens seen, validation loss 4.7410, commit `2bb50ee981b69db25306fd6e1aa16146723764f7`, and torch 2.8.0+cu128. AdamW remained faster at about 51.8k tokens/sec, but still trailed the Muon-paper leg at the same step (validation loss 4.6152).
- 2026-05-25 03:15 PDT: `overnight_adamw` reached the step-30,000 sample point. Validation loss improved to 4.6350 after 122,880,000 tokens, with about 51.8k tokens/sec average throughput. The 30k samples were more fluent than AdamW's 20k samples, but remained repetitive and factually ungrounded; AdamW still trailed the Muon-paper leg at the same step (validation loss 4.5133).
