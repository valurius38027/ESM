#!/usr/bin/env bash
set -euo pipefail

build_dir=${1:?build directory required}
seed_start=${2:?seed start required}
seed_count=${3:?seed count required}
steps=${4:-400}
batch_size=${5:-8}
train_count=${6:-160}
holdout_count=${7:-48}
output_root=${8:-results/phase2-shard}

raw_dir="$output_root/raw"
causal_dir="$output_root/causal"
log_dir="$output_root/logs"
mkdir -p "$raw_dir" "$causal_dir" "$log_dir"

for ((offset=0; offset<seed_count; ++offset)); do
  seed=$((seed_start + offset))
  for preset in core_small core_compute_matched core_param_matched sgw; do
    args=(
      --preset "$preset"
      --seed "$seed"
      --steps "$steps"
      --batch-size "$batch_size"
      --train-count "$train_count"
      --holdout-count "$holdout_count"
      --output "$raw_dir/${preset}_seed${seed}.csv"
    )
    if [[ "$preset" == sgw ]]; then
      args+=(--causal-output "$causal_dir/causal_seed${seed}.csv")
    fi
    "./${build_dir}/sgw_train" "${args[@]}" \
      >"$log_dir/${preset}_seed${seed}.log"
  done
done
