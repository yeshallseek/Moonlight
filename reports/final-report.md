# Muon is Scalable for LLM Training scaled reproduction Report

Mode: reproduction
Source: https://arxiv.org/pdf/2502.16982
Date: 2026-05-25

## Summary

This repo now contains a scaled, single-GPU reproduction study of Moonshot AI's "Muon is Scalable for LLM Training" report. The full paper result is a 3B-active/16B-total MoE trained on 5.7T tokens, so it is not reproducible on one RTX 5090. I instead tested the causal pieces that fit locally:

- Muon update orthogonalization and the paper's update-RMS scaling rule.
- Small Qwen-like causal LMs trained on `Elriggs/openwebtext-100k`, including a 175.7M-parameter overnight run over 245.8M tokens per optimizer.
- Four optimizer controls: AdamW, paper-style Muon, Muon without weight decay, and Muon without paper update-RMS scaling.

Main finding: the local evidence supports the paper-style Muon recipe at the largest scale tested here. In the 175.7M-parameter overnight suite, paper-style Muon finished with the best validation loss (`4.3272`, best observed `4.3184`), ahead of AdamW (`4.3619`), Muon without update-RMS scaling (`4.3817`), and Muon without weight decay (`4.4306`). The tiny 3,000-step run did not show a weight-decay benefit, but the larger 245.8M-token run did.

## What Was Tested

- Mechanics probe: random full-rank matrices across shapes from `16x16` through `512x128`, comparing exact SVD orthogonalization to the practical 5-step Newton-Schulz update.
- 300-step OpenWebText suite: 20.5M-parameter Qwen-like dense LM, 307,200 training tokens per run.
- 3,000-step OpenWebText suite: same architecture and cached tokens, 3,072,000 training tokens per run.
- 60,000-step overnight OpenWebText suite: 175.7M-parameter Qwen-like dense LM, 245,760,000 training tokens per run.
- Fixed controls: seed `1234`, Qwen2.5-0.5B tokenizer, sequence length 128, batch size 8, hidden size 128, 4 layers, LR `1e-3`, cosine schedule, BF16 autocast on CUDA.
- Overnight controls: seed `20260525`, Qwen2.5-0.5B tokenizer, sequence length 256, batch size 16, hidden size 640, 12 layers, LR `1e-3`, cosine schedule, BF16 autocast on CUDA.

## Commands

```bash
python3 scripts/probe_muon_mechanics.py --device cuda --out-dir artifacts/probes --runs-jsonl experiments/runs.jsonl
./scripts/run_tiny_suite.sh
./scripts/run_tiny_overtrain_suite.sh
python3 scripts/summarize_results.py
setsid bash scripts/run_overnight_large_suite.sh > logs/overnight_large_suite_20260525-004946.log 2>&1 &
python3 scripts/summarize_overnight_large.py
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
- `reports/figures/overnight_large_val_loss.png`
- `reports/figures/overnight_large_train_loss.png`

### Overnight 175.7M-Parameter OpenWebText Suite

| Run | Optimizer | WD | Scale | Final val loss | Best val loss | Final train loss | Tokens/s |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `muon_paper_640l12_b16` | Muon | 0.1 | paper | 4.3272 | 4.3184 @ 56k | 3.9568 | 39,484 |
| `adamw_640l12_b16` | AdamW | 0.1 | n/a | 4.3619 | 4.3619 @ 60k | 4.1649 | 51,714 |
| `muon_nowd_640l12_b16` | Muon | 0.0 | paper | 4.4306 | 4.4111 @ 50k | 3.9514 | 39,504 |
| `muon_noscale_640l12_b16` | Muon | 0.1 | none | 4.3817 | 4.3756 @ 56k | 4.1007 | 39,491 |

Interpretation:

- Paper-style Muon was the best local 175.7M run, finishing `0.0347` validation loss better than AdamW, `0.0546` better than no-scale Muon, and `0.1035` better than no-weight-decay Muon.
- No-scale Muon briefly edged AdamW at step 50k (`4.3958` versus `4.3994`) but plateaued and finished worse. Its MLP weight RMS fell to about `0.0056` mean at 60k, consistent with matrix updates being underpowered without the paper's update-RMS scaling.
- The larger run reversed the tiny-run weight-decay result: Muon without weight decay had low train loss but worse validation loss and a late rise after its best 50k checkpoint.
- AdamW was still much faster in this implementation, about `51.7k` tokens/sec versus about `39.5k` for Muon variants. This code path measures optimizer behavior more than production throughput.

Artifacts:

- `reports/overnight-large-summary.md`
- `reports/figures/overnight_large_val_loss.png`
- `reports/figures/overnight_large_train_loss.png`

## Failures and Limitations

- Not reproduced: the paper's approximately 52% training-FLOP scaling-law claim.
- Not reproduced: Moonlight 16B MoE benchmark tables, MMLU/code/math gains, SFT optimizer interchangeability, or distributed ZeRO-1 behavior.
- Only partially reproduced: the paper's weight-decay-over-vanilla-Muon win. It did not appear in the 3M-token tiny run, but did appear in the 245.8M-token 175.7M-parameter run.
- The official toy script has no max-step or validation controls, so I added local bounded scripts rather than running `examples/toy_train.py` directly.
- The current environment differs from upstream pins: Python 3.13.5, torch 2.8.0+cu128, transformers 5.8.0.dev0, datasets 4.4.1.
- The GPU initially had a `llama-server` process using about 27.5 GiB VRAM; it was stopped before the overnight suite. The 175.7M overnight runs peaked around 11.7-12.0 GiB allocated by PyTorch.

## Reproduction Status

- Reproduced: the update-RMS lemma for exact orthogonal updates.
- Partially reproduced: paper-style Muon improves small causal LM validation loss versus AdamW under the local setup, including the larger 175.7M run.
- Reproduced locally: removing paper update-RMS scaling degrades Muon at 300 steps, 3,000 steps, and 60,000 larger-model steps.
- Partially reproduced locally: weight decay improved Muon in the 175.7M overnight run, but not in the tiny 3,000-step run.
- Not attempted: full-scale Moonlight training, distributed Muon, released checkpoint evaluation, and benchmark-table reproduction.

## Reproducibility Notes

- Branch: `reproduce/muon-scaled`
- Upstream commit: `c2ad5b20c605086526a179d36901bfc41b52b44b`
- Run records: `experiments/runs.jsonl`
- Human source review: `experiments/source-review.md`
- Results tables and generated plots: `reports/results-summary.md`, `reports/overnight-large-summary.md`, `reports/figures/`
- Token cache: `artifacts/datasets/tokens_Elriggs_openwebtext-100k_9ec2a22bcc073eec.pt` (ignored by git)
- Overnight token cache: `artifacts/datasets/tokens_Elriggs_openwebtext-100k_17a86c86becbc0fa.pt` (ignored by git)
- Raw per-step metrics:
  - `artifacts/runs/adamw_openwebtext_3000/metrics.jsonl`
  - `artifacts/runs/muon_paper_openwebtext_3000/metrics.jsonl`
  - `artifacts/runs/muon_nowd_openwebtext_3000/metrics.jsonl`
  - `artifacts/runs/muon_noscale_openwebtext_3000/metrics.jsonl`
  - `artifacts/overnight_large/muon_paper_640l12_b16/metrics.jsonl`
  - `artifacts/overnight_large/adamw_640l12_b16/metrics.jsonl`
  - `artifacts/overnight_large/muon_nowd_640l12_b16/metrics.jsonl`
  - `artifacts/overnight_large/muon_noscale_640l12_b16/metrics.jsonl`

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

- Run multiple seeds or a larger token budget to check whether the 175.7M ranking is robust and where no-scale Muon plateaus.
- Add weight/output RMS logging by layer to see whether no-weight-decay Muon begins drifting before validation loss does.
- Evaluate the released Moonlight and Moonlight-A checkpoints if the Google Drive artifacts are accessible and storage budget is acceptable.
- Test a more optimized Muon implementation or PyTorch/NVIDIA implementation to separate optimizer math from Python-loop overhead.
