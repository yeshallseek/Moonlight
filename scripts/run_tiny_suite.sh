#!/usr/bin/env bash
set -euo pipefail

COMMON_ARGS=(
  --dataset Elriggs/openwebtext-100k
  --max-texts 2048
  --seq-len 128
  --batch-size 8
  --hidden-size 128
  --intermediate-size 512
  --num-layers 4
  --num-heads 4
  --num-kv-heads 4
  --max-steps 300
  --warmup-steps 20
  --eval-interval 50
  --log-interval 10
  --eval-batches 8
  --lr 0.001
  --seed 1234
)

python3 scripts/train_tiny_qwen_muon.py \
  "${COMMON_ARGS[@]}" \
  --optimizer adamw \
  --wd 0.1 \
  --out-dir artifacts/runs/adamw_openwebtext_300

python3 scripts/train_tiny_qwen_muon.py \
  "${COMMON_ARGS[@]}" \
  --optimizer muon \
  --wd 0.1 \
  --out-dir artifacts/runs/muon_paper_openwebtext_300

python3 scripts/train_tiny_qwen_muon.py \
  "${COMMON_ARGS[@]}" \
  --optimizer muon \
  --wd 0.0 \
  --out-dir artifacts/runs/muon_nowd_openwebtext_300

python3 scripts/train_tiny_qwen_muon.py \
  "${COMMON_ARGS[@]}" \
  --optimizer muon-noscale \
  --wd 0.1 \
  --out-dir artifacts/runs/muon_noscale_openwebtext_300
