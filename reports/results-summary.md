# Results Summary

## 300-step OpenWebText suite

| Run | Optimizer | WD | Scale | Steps | Final val loss | Final train loss | Tokens/s | Max CUDA MiB |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| adamw_openwebtext_300 | adamw | 0.1 | n/a | 300 | 7.4717 | 7.0430 | 79887 | 2102 |
| muon_paper_openwebtext_300 | muon | 0.1 | paper | 300 | 7.3777 | 6.9161 | 58847 | 2098 |
| muon_nowd_openwebtext_300 | muon | 0.0 | paper | 300 | 7.3723 | 6.9056 | 58676 | 2098 |
| muon_noscale_openwebtext_300 | muon-noscale | 0.1 | none | 300 | 7.7493 | 7.3220 | 59001 | 2098 |

## 3000-step OpenWebText suite

| Run | Optimizer | WD | Scale | Steps | Final val loss | Final train loss | Tokens/s | Max CUDA MiB |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| adamw_openwebtext_3000 | adamw | 0.1 | n/a | 3000 | 6.3798 | 6.0337 | 86851 | 2102 |
| muon_paper_openwebtext_3000 | muon | 0.1 | paper | 3000 | 6.2790 | 5.8828 | 62603 | 2098 |
| muon_nowd_openwebtext_3000 | muon | 0.0 | paper | 3000 | 6.2719 | 5.8563 | 62357 | 2098 |
| muon_noscale_openwebtext_3000 | muon-noscale | 0.1 | none | 3000 | 6.3836 | 5.9906 | 62440 | 2098 |

## Figures

- `reports/figures/val_loss_300.png`
- `reports/figures/val_loss_3000.png`
- `reports/figures/muon_rms_probe.png`
