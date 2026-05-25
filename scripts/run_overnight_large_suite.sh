#!/usr/bin/env bash
set -uo pipefail

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

run_one() {
  local name="$1"
  shift
  echo "[$(timestamp)] START ${name}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 scripts/train_tiny_qwen_muon.py "$@"
  local status=$?
  echo "[$(timestamp)] END ${name} status=${status}"
  return 0
}

COMMON_ARGS=(
  --dataset Elriggs/openwebtext-100k
  --max-texts 50000
  --seq-len 256
  --batch-size 16
  --hidden-size 640
  --intermediate-size 2560
  --num-layers 12
  --num-heads 8
  --num-kv-heads 8
  --warmup-steps 2000
  --eval-interval 2000
  --log-interval 200
  --eval-batches 32
  --weight-rms-interval 2000
  --checkpoint-interval 20000
  --sample-interval 10000
  --sample-max-new-tokens 96
  --lr 0.001
  --seed 20260525
)

echo "[$(timestamp)] Overnight larger Muon suite"
echo "[$(timestamp)] Git commit: $(git rev-parse HEAD)"
nvidia-smi

run_one "overnight_muon_paper" \
  "${COMMON_ARGS[@]}" \
  --optimizer muon \
  --wd 0.1 \
  --max-steps 60000 \
  --max-run-seconds 21600 \
  --out-dir artifacts/overnight_large/muon_paper_640l12_b16

run_one "overnight_adamw" \
  "${COMMON_ARGS[@]}" \
  --optimizer adamw \
  --wd 0.1 \
  --max-steps 60000 \
  --max-run-seconds 21600 \
  --out-dir artifacts/overnight_large/adamw_640l12_b16

run_one "overnight_muon_nowd" \
  "${COMMON_ARGS[@]}" \
  --optimizer muon \
  --wd 0.0 \
  --max-steps 60000 \
  --max-run-seconds 21600 \
  --out-dir artifacts/overnight_large/muon_nowd_640l12_b16

run_one "overnight_muon_noscale" \
  "${COMMON_ARGS[@]}" \
  --optimizer muon-noscale \
  --wd 0.1 \
  --max-steps 60000 \
  --max-run-seconds 21600 \
  --out-dir artifacts/overnight_large/muon_noscale_640l12_b16

echo "[$(timestamp)] Overnight larger Muon suite finished"
nvidia-smi
