#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 7 ]]; then
  echo "usage: $0 EXE CORPUS CORPUS_ID MODEL CONTEXT SEED OUTPUT" >&2
  exit 2
fi
exe=$1
corpus=$2
corpus_id=$3
model=$4
context=$5
seed=$6
output=$7
steps=${LMV1_STEPS:-300}
batch=${LMV1_BATCH_SIZE:-2}
eval_windows=${LMV1_EVAL_WINDOWS:-16}
validation_interval=${LMV1_VALIDATION_INTERVAL:-50}
mkdir -p "$(dirname "$output")"
rm -rf "$output"
"$exe" \
  --corpus "$corpus" --corpus-id "$corpus_id" \
  --model "$model" --context "$context" --steps "$steps" \
  --batch-size "$batch" --eval-windows "$eval_windows" \
  --validation-interval "$validation_interval" --seed "$seed" \
  --output-dir "$output"
(
  cd "$output"
  sha256sum primary.csv interventions.csv loss_curve.csv checkpoint.bin > SHA256SUMS
)
