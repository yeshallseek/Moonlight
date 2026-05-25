# Overnight Large Results

Generated: `2026-05-25T07:23:47-0700`

## Configuration

- Model: 175,747,840 parameters, hidden size 640, 12 layers, 8 attention heads, 8 KV heads
- Data: `Elriggs/openwebtext-100k`, max texts 50,000, seq len 256, batch size 16
- Schedule: LR 0.001, warmup 2,000, max steps 60,000, seed 20260525

## Runs

| Run | Status | Optimizer | WD | Scale | Last step | Latest val loss | Latest train loss | Tokens seen | Tokens/s | Samples | Checkpoints |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| muon_paper_640l12_b16 | max_steps | muon | 0.1 | paper | 60,000 | 4.3272 | 3.9568 | 245,760,000 | 39,484 | 21 | 3 |
| adamw_640l12_b16 | max_steps | adamw | 0.1 | n/a | 60,000 | 4.3619 | 4.1649 | 245,760,000 | 51,714 | 21 | 3 |
| muon_nowd_640l12_b16 | max_steps | muon | 0.0 | paper | 60,000 | 4.4306 | 3.9514 | 245,760,000 | 39,504 | 21 | 3 |
| muon_noscale_640l12_b16 | max_steps | muon-noscale | 0.1 | none | 60,000 | 4.3817 | 4.1007 | 245,760,000 | 39,491 | 21 | 3 |

## Figures

- `reports/figures/overnight_large_val_loss.png`
- `reports/figures/overnight_large_train_loss.png`

