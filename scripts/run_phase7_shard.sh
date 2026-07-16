#!/usr/bin/env bash
set -euo pipefail

build_dir=${1:?build directory required}
seed_start=${2:?seed start required}
seed_count=${3:?seed count required}
steps=${4:-800}
batch_size=${5:-8}
train_count=${6:-160}
holdout_count=${7:-160}
output_root=${8:-results/phase7-shard}

raw_dir="$output_root/raw"
causal_dir="$output_root/causal"
log_dir="$output_root/logs"
mkdir -p "$raw_dir" "$causal_dir" "$log_dir"

conditions=(
  structural_core_blind
  kv_fixed_position
  kv_first_free
  kv_hard_router
  kv_annealed_router
)

for ((offset=0; offset<seed_count; ++offset)); do
  seed=$((seed_start + offset))
  for condition in "${conditions[@]}"; do
    args=(
      --condition "$condition"
      --seed "$seed"
      --steps "$steps"
      --batch-size "$batch_size"
      --train-count "$train_count"
      --holdout-count "$holdout_count"
      --output "$raw_dir/${condition}_seed${seed}.csv"
    )
    if [[ "$condition" == kv_annealed_router ]]; then
      args+=(--causal-output "$causal_dir/causal_seed${seed}.csv")
    fi
    "./${build_dir}/sgw_train" "${args[@]}" \
      >"$log_dir/${condition}_seed${seed}.log"
  done
done
