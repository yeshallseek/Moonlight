# Evaluation Plan

## 2026-05-25 Overnight Larger Muon Run

### Objective

Use the free overnight window to train a meaningfully larger Qwen-like causal language model with Muon, then compare it to AdamW and a no-scale Muon control on the same public OpenWebText sample. The goal is not to reproduce Moonlight's 5.7T-token MoE training, but to make the local experiment more interesting than the earlier 20.5M-parameter, 3M-token toy run.

### Capabilities to Test

- Whether paper-style Muon remains competitive as model size, sequence length, and token budget increase.
- Whether removing paper update-RMS scaling still damages Muon at larger local scale.
- Whether weight/activation telemetry shows drift patterns that may explain the paper's weight-decay discussion.
- Whether intermediate samples become visibly more coherent during the run.

### Planned Runs

All runs use `Elriggs/openwebtext-100k`, Qwen2.5 tokenizer, BF16 autocast on the RTX 5090, and periodic validation, checkpointing, samples, and weight-RMS logs.

Primary overnight suite:

- `overnight_muon_paper`: Muon, weight decay `0.1`, paper update-RMS scaling.
- `overnight_adamw`: AdamW, weight decay `0.1`.
- `overnight_muon_noscale`: Muon, weight decay `0.1`, no paper update-RMS scaling.

Target model and data, subject to dry-run VRAM calibration:

- Hidden size: 384
- Layers: 8
- Attention heads: 8
- KV heads: 8
- Intermediate size: 1536
- Sequence length: 256
- Batch size: 8 initially, reduce if needed
- Dataset slice: 25,000 OpenWebText documents initially, increase if tokenization/runtime allows
- Training steps: 60,000 optimizer steps per run after calibration, with checkpointing every 20,000 steps and validation every 2,000 steps

### Metrics and Outputs

- Training loss and validation loss in `metrics.jsonl`.
- Tokens/sec, max CUDA memory, elapsed time, stop reason, and final loss in `summary.json`.
- Grouped parameter RMS telemetry for embeddings/head, attention, MLP, norms, and other parameters.
- Sample generations in `samples.jsonl`.
- Recoverable checkpoints under each run's `checkpoints/` directory. These are ignored by git.
- Detached runner log under `logs/`.

### Runtime and Resource Expectations

- Expected VRAM after stopping the existing `llama-server`: 8-20 GiB depending on batch size and optimizer.
- Expected disk: token cache could be hundreds of MiB; each large checkpoint may be roughly 1-3 GiB depending on model size and optimizer state. Keep checkpoint interval moderate.
- Expected runtime: calibration measured about 37k tokens/sec for the 175.7M-parameter Muon model at batch 16. The full four-run suite targets about 983M training tokens total, with max-run-seconds guards so partial metrics/checkpoints survive if the suite exceeds the night.

### Stop Conditions

- Stop immediately on non-finite loss.
- Stop or downscale if CUDA OOM occurs during dry run.
- Record any GPU process reclaimed before launch.
- If one run fails but the runner continues, preserve completed summaries and failed logs.

### Success Criteria

- Dry run completes on GPU.
- Detached overnight process is launched with a PID, log path, run directories, and reproducible command.
- At least one larger Muon run produces checkpointed metrics and samples.
- Final report can compare larger-scale Muon, AdamW, and no-scale Muon with explicit caveats.
