# Muon is Scalable for LLM Training scaled reproduction Report

Mode: reproduction
Source: https://arxiv.org/pdf/2502.16982
Date: 2026-05-25

## Summary

This repo now contains a scaled, single-GPU reproduction study of Moonshot AI's "Muon is Scalable for LLM Training" report. The full paper result is a 3B-active/16B-total MoE trained on 5.7T tokens, so it is not reproducible on one RTX 5090. I instead tested the causal pieces that fit locally:

- Muon update orthogonalization and the paper's update-RMS scaling rule.
- A tiny Qwen-like causal LM trained on a 2,048-document sample of `Elriggs/openwebtext-100k`.
- Four optimizer controls: AdamW, paper-style Muon, Muon without weight decay, and Muon without paper update-RMS scaling.

Main finding: the local evidence strongly supports the update-RMS correction. Paper-style Muon beat AdamW in both 300-step and 3,000-step tiny runs, while no-scale Muon lagged. The weight-decay claim was not reproduced at this toy scale: Muon without weight decay was essentially tied with and slightly ahead of paper-style Muon over 3M training tokens. That is not a contradiction of the paper's long-horizon result; their weight-decay ablation is in a much larger overtrain regime.

## What Was Tested

- Mechanics probe: random full-rank matrices across shapes from `16x16` through `512x128`, comparing exact SVD orthogonalization to the practical 5-step Newton-Schulz update.
- 300-step OpenWebText suite: 20.5M-parameter Qwen-like dense LM, 307,200 training tokens per run.
- 3,000-step OpenWebText suite: same architecture and cached tokens, 3,072,000 training tokens per run.
- Fixed controls: seed `1234`, Qwen2.5-0.5B tokenizer, sequence length 128, batch size 8, hidden size 128, 4 layers, LR `1e-3`, cosine schedule, BF16 autocast on CUDA.

## Commands

```bash
python3 scripts/probe_muon_mechanics.py --device cuda --out-dir artifacts/probes --runs-jsonl experiments/runs.jsonl
./scripts/run_tiny_suite.sh
./scripts/run_tiny_overtrain_suite.sh
python3 scripts/summarize_results.py
```

## Results

### Mechanics Probe

The exact orthogonal update matched the paper's RMS formula: unscaled RMS was `1 / sqrt(max(A, B))`, and applying the paper scale `0.2 * sqrt(max(A, B))` produced RMS `0.2` for every tested shape. The practical 5-step Newton-Schulz update was consistently lower, around `0.17-0.19` after the paper scale, because its mean singular value was below 1 in this probe. This is still shape-normalized and explains why the correction matters.

Artifacts:

- `artifacts/probes/muon_mechanics_summary.json`
- `artifacts/probes/muon_mechanics_records.csv`
- `reports/figures/muon_rms_probe.png`

### 300-Step OpenWebText Suite

| Run | Optimizer | WD | Scale | Final val loss | Final train loss | Tokens/s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `adamw_openwebtext_300` | AdamW | 0.1 | n/a | 7.4717 | 7.0430 | 79,887 |
| `muon_paper_openwebtext_300` | Muon | 0.1 | paper | 7.3777 | 6.9161 | 58,847 |
| `muon_nowd_openwebtext_300` | Muon | 0.0 | paper | 7.3723 | 6.9056 | 58,676 |
| `muon_noscale_openwebtext_300` | Muon | 0.1 | none | 7.7493 | 7.3220 | 59,001 |

### 3,000-Step OpenWebText Suite

| Run | Optimizer | WD | Scale | Final val loss | Final train loss | Tokens/s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `adamw_openwebtext_3000` | AdamW | 0.1 | n/a | 6.3798 | 6.0337 | 86,851 |
| `muon_paper_openwebtext_3000` | Muon | 0.1 | paper | 6.2790 | 5.8828 | 62,603 |
| `muon_nowd_openwebtext_3000` | Muon | 0.0 | paper | 6.2719 | 5.8563 | 62,357 |
| `muon_noscale_openwebtext_3000` | Muon | 0.1 | none | 6.3836 | 5.9906 | 62,440 |

Interpretation:

- The paper-style RMS-scaled Muon run beat AdamW by `0.1008` validation loss in the 3,000-step setting.
- Removing RMS scaling erased that advantage: no-scale Muon finished at `6.3836`, essentially tied with AdamW and `0.1046` worse than paper-style Muon.
- Removing weight decay did not hurt at this scale. It slightly improved final validation loss by `0.0071` versus paper-style Muon. The paper's weight-decay benefit appears tied to longer/larger overtraining than this local run.
- AdamW was faster in this script, mostly because this Muon implementation performs per-matrix Newton-Schulz updates in Python. This experiment measures optimizer behavior, not optimized production throughput.

Plots:

- `reports/figures/val_loss_300.png`
- `reports/figures/val_loss_3000.png`

## Failures and Limitations

- Not reproduced: the paper's approximately 52% training-FLOP scaling-law claim.
- Not reproduced: Moonlight 16B MoE benchmark tables, MMLU/code/math gains, SFT optimizer interchangeability, or distributed ZeRO-1 behavior.
- Not reproduced: the paper's weight-decay-over-vanilla-Muon win. Our run is a tiny dense model over about 3M tokens, while the paper highlights an 800M-parameter, 100B-token overtrain ablation.
- The official toy script has no max-step or validation controls, so I added local bounded scripts rather than running `examples/toy_train.py` directly.
- The current environment differs from upstream pins: Python 3.13.5, torch 2.8.0+cu128, transformers 5.8.0.dev0, datasets 4.4.1.
- The GPU had an existing `llama-server` process using about 27.5 GiB VRAM; these tiny runs fit in the remaining memory and peaked around 2.1 GiB allocated by PyTorch.

## Reproduction Status

- Reproduced: the update-RMS lemma for exact orthogonal updates.
- Partially reproduced: paper-style Muon improves small causal LM validation loss versus AdamW under the local setup.
- Reproduced locally: removing paper update-RMS scaling degrades Muon substantially.
- Not reproduced locally: weight decay improving Muon over no-weight-decay Muon.
- Not attempted: full-scale Moonlight training, distributed Muon, released checkpoint evaluation, and benchmark-table reproduction.

## Reproducibility Notes

- Branch: `reproduce/muon-scaled`
- Upstream commit: `c2ad5b20c605086526a179d36901bfc41b52b44b`
- Run records: `experiments/runs.jsonl`
- Human source review: `experiments/source-review.md`
- Results table and generated plots: `reports/results-summary.md`, `reports/figures/`
- Token cache: `artifacts/datasets/tokens_Elriggs_openwebtext-100k_9ec2a22bcc073eec.pt` (ignored by git)
- Raw per-step metrics:
  - `artifacts/runs/adamw_openwebtext_3000/metrics.jsonl`
  - `artifacts/runs/muon_paper_openwebtext_3000/metrics.jsonl`
  - `artifacts/runs/muon_nowd_openwebtext_3000/metrics.jsonl`
  - `artifacts/runs/muon_noscale_openwebtext_3000/metrics.jsonl`

## Sources Used

Official and maintainer sources:

- Paper: https://arxiv.org/abs/2502.16982 and https://arxiv.org/pdf/2502.16982
- MoonshotAI/Moonlight repo: https://github.com/MoonshotAI/Moonlight at local commit `c2ad5b20c605086526a179d36901bfc41b52b44b`
- Moonlight model card: https://huggingface.co/moonshotai/Moonlight-16B-A3B
- Original Muon repo: https://github.com/KellerJordan/Muon, remote HEAD observed as `f98f1cacc0263b04290753e32be8d498c1efc806`
- Original Muon design writeup: https://kellerjordan.github.io/posts/muon/
- Megatron-LM Distributed Muon PR linked by Moonlight README: https://github.com/NVIDIA/Megatron-LM/pull/1428, PR head observed as `8aee20bdad4a2561c1418ea1ecf09348027effe8`

No third-party reproduction was used to design or interpret the experiments.

## Next Experiments

- Run a longer overtrain ablation, likely 30k+ tiny steps or a larger model/token budget, to look for the weight-decay crossover the paper reports.
- Add weight/output RMS logging by layer to see whether no-weight-decay Muon begins drifting before validation loss does.
- Evaluate the released Moonlight and Moonlight-A checkpoints if the Google Drive artifacts are accessible and storage budget is acceptable.
- Test a more optimized Muon implementation or PyTorch/NVIDIA implementation to separate optimizer math from Python-loop overhead.
