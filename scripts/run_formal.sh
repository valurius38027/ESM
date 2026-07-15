#!/usr/bin/env bash
set -euo pipefail

build_dir=${1:-build-release}
raw_dir=results/raw
mkdir -p "$raw_dir"
rm -f "$raw_dir"/*_seed*.csv "$raw_dir"/*_seed*.log

for seed in 0 1 2 3 4; do
  for mode in core_only sgw; do
    timeout 110s "./${build_dir}/sgw_train" \
      --mode "$mode" --seed "$seed" --steps 120 --batch-size 6 \
      --train-count 96 --holdout-count 32 \
      --output "$raw_dir/${mode}_seed${seed}.csv" \
      > "$raw_dir/${mode}_seed${seed}.log"
  done
done

python3 scripts/summarize_runs.py --raw-dir "$raw_dir" --output-dir results
