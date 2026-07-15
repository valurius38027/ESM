#!/usr/bin/env bash
set -euo pipefail

source_dir="$1"
build_dir="$2/summary-test"
raw_dir="$build_dir/raw"
causal_dir="$build_dir/causal"
out_a="$build_dir/out-a"
out_b="$build_dir/out-b"
rm -rf "$build_dir"
mkdir -p "$raw_dir" "$causal_dir"

run_header='preset,seed,steps,batch_size,parameters,train_count,holdout_count,sequence_length,chance_nll,initial_train_nll,initial_holdout_nll,final_train_nll,final_holdout_nll,final_train_accuracy,final_holdout_accuracy,mean_estimated_madds_per_token,mean_active_mechanisms_per_token,mean_writers_per_token,mean_recipients_per_token,first_window_loss,last_window_loss,mechanism_load'
causal_header='seed,intervention,holdout_nll,holdout_accuracy,nll_delta_vs_intact,accuracy_delta_vs_intact,mean_active_mechanisms_per_token,mean_writers_per_token,mean_recipients_per_token,mechanism_load'

for seed in 0 1; do
  for preset in core_small core_compute_matched core_param_matched sgw; do
    case "$preset" in
      core_small) params=327; madds=210; nll=$(python3 -c "print(1.5 + $seed * 0.1)") ;;
      core_compute_matched) params=2232; madds=1092; nll=$(python3 -c "print(1.3 + $seed * 0.1)") ;;
      core_param_matched) params=2181; madds=1474; nll=$(python3 -c "print(1.2 + $seed * 0.1)") ;;
      sgw) params=2182; madds=1092; nll=$(python3 -c "print(1.0 + $seed * 0.1)") ;;
    esac
    file="$raw_dir/${preset}_seed${seed}.csv"
    printf '%s\n' "$run_header" > "$file"
    printf '%s\n' "$preset,$seed,10,2,$params,8,4,9,1.609,1.7,1.6,1.1,$nll,0.5,0.6,$madds,2,1,2,1.6,1.1,1;2;3;4;5;6" >> "$file"
  done

  file="$causal_dir/causal_seed${seed}.csv"
  printf '%s\n' "$causal_header" > "$file"
  printf '%s\n' "$seed,intact,1.0,0.6,0.0,0.0,2,1,2,1;2;3;4;5;6" >> "$file"
  printf '%s\n' "$seed,no_broadcast,1.2,0.5,0.2,-0.1,2,1,2,1;2;3;4;5;6" >> "$file"
  printf '%s\n' "$seed,no_workspace_persistence,1.3,0.5,0.3,-0.1,2,1,2,1;2;3;4;5;6" >> "$file"
  printf '%s\n' "$seed,no_workspace_output,1.1,0.55,0.1,-0.05,2,1,2,1;2;3;4;5;6" >> "$file"
  printf '%s\n' "$seed,permuted_recipients,1.25,0.5,0.25,-0.1,2,1,2,1;2;3;4;5;6" >> "$file"
done

for out in "$out_a" "$out_b"; do
  python3 "$source_dir/scripts/aggregate_phase2.py" \
    --raw-dir "$raw_dir" --causal-dir "$causal_dir" --output-dir "$out" \
    --seed-start 0 --seed-count 2 --bootstrap-replicates 1000
  if grep -R -l $'\r' "$out"/*.csv "$out"/*.json "$out"/SHA256SUMS >/dev/null; then
    echo 'phase2 aggregate contains CRLF line endings' >&2
    exit 1
  fi
done

diff -ru "$out_a" "$out_b"
[[ $(wc -l < "$out_a/phase2_runs.csv") -eq 9 ]]
[[ $(wc -l < "$out_a/phase2_causal.csv") -eq 11 ]]
grep -q '"seed_count": 2' "$out_a/phase2_manifest.json"
grep -q '^core_compute_matched_vs_sgw,' "$out_a/phase2_paired.csv"
grep -q '^no_broadcast,' "$out_a/phase2_causal_comparisons.csv"

cp "$raw_dir/core_small_seed0.csv" "$raw_dir/core_small_seed99.csv"
if python3 "$source_dir/scripts/aggregate_phase2.py" \
  --raw-dir "$raw_dir" --causal-dir "$causal_dir" --output-dir "$build_dir/bad" \
  --seed-start 0 --seed-count 2 --bootstrap-replicates 10 >/dev/null 2>&1; then
  echo 'aggregate accepted stale seed shard' >&2
  exit 1
fi
