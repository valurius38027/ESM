#!/usr/bin/env bash
set -euo pipefail

exe="$1"
out_dir="$2/cli-test"
rm -rf "$out_dir"
mkdir -p "$out_dir"
first="$out_dir/first.csv"
second="$out_dir/second.csv"
causal_first="$out_dir/causal-first.csv"
causal_second="$out_dir/causal-second.csv"

common=(
  --preset sgw
  --seed 41
  --steps 3
  --batch-size 2
  --train-count 12
  --holdout-count 6
)

timeout 110s "$exe" "${common[@]}" --output "$first" \
  --causal-output "$causal_first" >/dev/null
timeout 110s "$exe" "${common[@]}" --output "$second" \
  --causal-output "$causal_second" >/dev/null
cmp "$first" "$second"
cmp "$causal_first" "$causal_second"

header='preset,seed,steps,batch_size,parameters,train_count,holdout_count,sequence_length,chance_nll,initial_train_nll,initial_holdout_nll,final_train_nll,final_holdout_nll,final_train_accuracy,final_holdout_accuracy,mean_estimated_madds_per_token,mean_active_mechanisms_per_token,mean_writers_per_token,mean_recipients_per_token,first_window_loss,last_window_loss,mechanism_load,role_mechanism_load,condition,workspace_aux_initial_weight,workspace_aux_anneal_steps,first_primary_window_loss,last_primary_window_loss,last_workspace_aux_window_loss'
[[ $(head -n 1 "$first") == "$header" ]]
[[ $(wc -l < "$first") -eq 2 ]]
grep -q '^sgw,41,3,2,2182,' "$first"
grep -q 'entity:' "$first"
grep -q 'query_entity:' "$first"

causal_header='seed,intervention,holdout_nll,holdout_accuracy,nll_delta_vs_intact,accuracy_delta_vs_intact,mean_active_mechanisms_per_token,mean_writers_per_token,mean_recipients_per_token,mechanism_load,role_mechanism_load,condition'
[[ $(head -n 1 "$causal_first") == "$causal_header" ]]
[[ $(wc -l < "$causal_first") -eq 6 ]]
for intervention in intact no_broadcast no_workspace_persistence no_workspace_output permuted_recipients; do
  grep -q "^41,$intervention," "$causal_first"
done
grep -q 'entity:' "$causal_first"
grep -q ',sgw_redundant_final_only$' "$causal_first"

for entry in 'core_small:327:210' 'core_compute_matched:2232:1092' 'core_param_matched:2181:1474'; do
  IFS=: read -r preset parameters madds <<<"$entry"
  file="$out_dir/$preset.csv"
  timeout 110s "$exe" --preset "$preset" --seed 7 --steps 1 \
    --batch-size 1 --train-count 6 --holdout-count 3 --output "$file" >/dev/null
  grep -q "^$preset,7,1,1,$parameters," "$file"
  awk -F, -v expected="$madds" 'NR==2 { exit !($16 == expected ".000000000000") }' "$file"
done

legacy="$out_dir/legacy.csv"
timeout 110s "$exe" --mode core_only --seed 7 --steps 1 --batch-size 1 \
  --train-count 6 --holdout-count 3 --output "$legacy" >/dev/null
grep -q '^core_small,7,1,1,327,' "$legacy"


forced="$out_dir/forced-aux.csv"
forced_causal="$out_dir/forced-aux-causal.csv"
timeout 110s "$exe" \
  --condition sgw_broadcast_forced_aux_annealed \
  --seed 17 --steps 3 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$forced" --causal-output "$forced_causal" >/dev/null
grep -q '^sgw_broadcast_forced,17,3,2,2182,' "$forced"
grep -q ',sgw_broadcast_forced_aux_annealed,0.500000000000,200,' "$forced"
[[ $(wc -l < "$forced_causal") -eq 9 ]]
for intervention in intact no_broadcast no_workspace_persistence no_workspace_output no_spine_workspace no_mechanism_output workspace_disconnected permuted_recipients; do
  grep -q "^17,$intervention," "$forced_causal"
done
grep -q ',sgw_broadcast_forced_aux_annealed$' "$forced_causal"
