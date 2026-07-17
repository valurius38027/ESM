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

header='preset,seed,steps,batch_size,parameters,train_count,holdout_count,sequence_length,chance_nll,initial_train_nll,initial_holdout_nll,final_train_nll,final_holdout_nll,final_train_accuracy,final_holdout_accuracy,mean_estimated_madds_per_token,mean_active_mechanisms_per_token,mean_writers_per_token,mean_recipients_per_token,first_window_loss,last_window_loss,mechanism_load,role_mechanism_load,condition,workspace_aux_initial_weight,workspace_aux_anneal_steps,first_primary_window_loss,last_primary_window_loss,last_workspace_aux_window_loss,anneal_boundary_primary_window_loss,final_workspace_aux_weight,final_weighted_workspace_aux_window_loss,training_stream_samples,output_logit_bound,initial_train_brier,initial_holdout_brier,final_train_brier,final_holdout_brier,initial_train_ece,initial_holdout_ece,final_train_ece,final_holdout_ece,final_train_max_confidence,final_holdout_max_confidence,final_train_true_class_probability,final_holdout_true_class_probability,read_slot_load,write_slot_load,mean_write_collision_rate,mean_routing_entropy,mean_routing_disagreement_rate,router_initial_temperature,router_final_temperature,router_anneal_steps,final_200_collision_rate,final_200_routing_entropy,final_200_disagreement_rate,retention_write_rate,retention_skip_rate,retention_eviction_rate,relevant_eviction_rate,queried_entity_retention_rate,query_read_hit_rate,mean_retained_age,eviction_slot_load,final_300_retention_write_rate,final_300_retention_skip_rate,final_300_retention_eviction_rate,final_300_relevant_eviction_rate,final_300_query_read_hit_rate,mean_source_query_distance,delay_binding_count,distractor_write_rate,distractor_eviction_rate,relevant_survival_rate,final_450_distractor_write_rate,final_450_distractor_eviction_rate,final_450_relevant_survival_rate,final_450_disagreement_rate,delay0_nll,delay0_accuracy,delay0_query_hit,delay0_survival,delay6_nll,delay6_accuracy,delay6_query_hit,delay6_survival,delay12_nll,delay12_accuracy,delay12_query_hit,delay12_survival,delay18_nll,delay18_accuracy,delay18_query_hit,delay18_survival,delay24_nll,delay24_accuracy,delay24_query_hit,delay24_survival'
[[ $(head -n 1 "$first") == "$header" ]]
[[ $(wc -l < "$first") -eq 2 ]]
grep -q '^sgw,41,3,2,2182,' "$first"
grep -q 'entity:' "$first"
grep -q 'query_entity:' "$first"

causal_header='seed,intervention,holdout_nll,holdout_accuracy,nll_delta_vs_intact,accuracy_delta_vs_intact,mean_active_mechanisms_per_token,mean_writers_per_token,mean_recipients_per_token,mechanism_load,role_mechanism_load,condition,holdout_brier,holdout_ece,holdout_max_confidence,holdout_true_class_probability,brier_delta_vs_intact,ece_delta_vs_intact,max_confidence_delta_vs_intact,true_class_probability_delta_vs_intact,read_slot_load,write_slot_load,mean_write_collision_rate,mean_routing_entropy,mean_routing_disagreement_rate,retention_write_rate,retention_skip_rate,retention_eviction_rate,relevant_eviction_rate,queried_entity_retention_rate,query_read_hit_rate,mean_retained_age,eviction_slot_load,mean_source_query_distance,delay_binding_count,distractor_write_rate,distractor_eviction_rate,relevant_survival_rate'
[[ $(head -n 1 "$causal_first") == "$causal_header" ]]
[[ $(wc -l < "$causal_first") -eq 6 ]]
for intervention in intact no_broadcast no_workspace_persistence no_workspace_output permuted_recipients; do
  grep -q "^41,$intervention," "$causal_first"
done
grep -q 'entity:' "$causal_first"
grep -q ',sgw_redundant_final_only,' "$causal_first"

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
grep -q ',sgw_broadcast_forced_aux_annealed,' "$forced_causal"


mediation="$out_dir/mediation-aux.csv"
mediation_causal="$out_dir/mediation-aux-causal.csv"
timeout 110s "$exe" \
  --condition mediation_aux_annealed \
  --seed 19 --steps 3 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$mediation" --causal-output "$mediation_causal" >/dev/null
grep -q '^mediation_fixed,19,3,2,15650,' "$mediation"
grep -q ',mediation_aux_annealed,0.500000000000,200,' "$mediation"
awk -F, 'NR==2 { exit !($8 == 8 && $17 == 1 && $18 == 0.75 && $19 == 1) }' "$mediation"
[[ $(wc -l < "$mediation_causal") -eq 11 ]]
for intervention in intact no_broadcast no_workspace_persistence no_workspace_output no_spine_workspace no_mechanism_output workspace_disconnected no_workspace_writes zero_reader_inbox permuted_recipients; do
  grep -q "^19,$intervention," "$mediation_causal"
done
grep -q ',mediation_aux_annealed,' "$mediation_causal"

for entry in 'core_full_content:1169:984' 'core_content_blind:725:696'; do
  IFS=: read -r condition parameters madds <<<"$entry"
  file="$out_dir/$condition.csv"
  timeout 110s "$exe" --condition "$condition" --seed 23 --steps 1 \
    --batch-size 1 --train-count 6 --holdout-count 3 --output "$file" >/dev/null
  grep -q "^$condition,23,1,1,$parameters," "$file"
  awk -F, -v expected="$madds" 'NR==2 { exit !($16 == expected ".000000000000") }' "$file"
done


structural_linear="$out_dir/structural-linear.csv"
structural_bounded="$out_dir/structural-bounded.csv"
structural_causal="$out_dir/structural-causal.csv"
timeout 110s "$exe" --condition structural_mediation_linear --seed 29 \
  --steps 3 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$structural_linear" --causal-output "$structural_causal" >/dev/null
timeout 110s "$exe" --condition structural_mediation_bounded --seed 29 \
  --steps 3 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$structural_bounded" >/dev/null
grep -q '^structural_mediation_linear,29,3,2,15650,' "$structural_linear"
grep -q '^structural_mediation_bounded,29,3,2,15650,' "$structural_bounded"
awk -F, 'NR==2 { exit !($33 == 6 && $34 == "0.000000000000") }' "$structural_linear"
awk -F, 'NR==2 { exit !($33 == 6 && $34 == "1.000000000000") }' "$structural_bounded"
[[ $(wc -l < "$structural_causal") -eq 11 ]]


kv_exact="$out_dir/structural-kv-exact.csv"
kv_learned="$out_dir/structural-kv-learned.csv"
kv_learned_second="$out_dir/structural-kv-learned-second.csv"
kv_causal="$out_dir/structural-kv-causal.csv"
kv_causal_second="$out_dir/structural-kv-causal-second.csv"
timeout 110s "$exe" --condition structural_kv_exact --seed 31 \
  --steps 2 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$kv_exact" >/dev/null
timeout 110s "$exe" --condition structural_kv_learned --seed 31 \
  --steps 2 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$kv_learned" --causal-output "$kv_causal" >/dev/null
timeout 110s "$exe" --condition structural_kv_learned --seed 31 \
  --steps 2 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$kv_learned_second" --causal-output "$kv_causal_second" >/dev/null
cmp "$kv_learned" "$kv_learned_second"
cmp "$kv_causal" "$kv_causal_second"
grep -q '^structural_kv_exact,31,2,2,0,' "$kv_exact"
grep -q '^structural_kv_learned,31,2,2,88,' "$kv_learned"
awk -F, 'NR==2 { exit !($33 == 4 && $15 >= 0.99 && $47 != "") }' "$kv_exact"
[[ $(wc -l < "$kv_causal") -eq 13 ]]
for intervention in intact no_broadcast no_workspace_persistence no_workspace_output no_spine_workspace no_mechanism_output workspace_disconnected no_workspace_writes zero_reader_inbox permuted_workspace_keys zero_query_key permuted_recipients; do
  grep -q "^31,$intervention," "$kv_causal"
done


phase7="$out_dir/phase7-annealed.csv"
phase7_second="$out_dir/phase7-annealed-second.csv"
phase7_causal="$out_dir/phase7-annealed-causal.csv"
phase7_causal_second="$out_dir/phase7-annealed-causal-second.csv"
timeout 110s "$exe" --condition kv_annealed_router --seed 37 \
  --steps 4 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$phase7" --causal-output "$phase7_causal" >/dev/null
timeout 110s "$exe" --condition kv_annealed_router --seed 37 \
  --steps 4 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$phase7_second" --causal-output "$phase7_causal_second" >/dev/null
cmp "$phase7" "$phase7_second"
cmp "$phase7_causal" "$phase7_causal_second"
grep -q '^kv_annealed_router,37,4,2,112,' "$phase7"
grep -q ',kv_annealed_router,' "$phase7"
for intervention in randomized_write_slots cleared_writer_assignment allow_write_collisions; do
  grep -q "^37,$intervention," "$phase7_causal"
done


phase8="$out_dir/phase8-annealed.csv"
phase8_second="$out_dir/phase8-annealed-second.csv"
phase8_causal="$out_dir/phase8-annealed-causal.csv"
phase8_causal_second="$out_dir/phase8-annealed-causal-second.csv"
timeout 110s "$exe" --condition kv_annealed_retention --seed 43 \
  --steps 4 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$phase8" --causal-output "$phase8_causal" >/dev/null
timeout 110s "$exe" --condition kv_annealed_retention --seed 43 \
  --steps 4 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$phase8_second" --causal-output "$phase8_causal_second" >/dev/null
cmp "$phase8" "$phase8_second"
cmp "$phase8_causal" "$phase8_causal_second"
grep -q '^kv_annealed_retention,43,4,2,145,' "$phase8"
grep -q ',kv_annealed_retention,' "$phase8"
grep -q 'context:' "$phase8"
awk -F, 'NR==2 { exit !($8 == 15 && $33 == 8 && $61 >= 0 && $66 >= 0) }' "$phase8"
[[ $(wc -l < "$phase8_causal") -eq 11 ]]
for intervention in intact zero_query_context randomized_retention_actions force_fifo_retention force_relevant_eviction permuted_context_labels disable_retention_skip no_workspace_writes no_workspace_persistence no_mechanism_output; do
  grep -q "^43,$intervention," "$phase8_causal"
done

phase9="$out_dir/phase9-curriculum.csv"
phase9_second="$out_dir/phase9-curriculum-second.csv"
phase9_causal="$out_dir/phase9-curriculum-causal.csv"
phase9_causal_second="$out_dir/phase9-curriculum-causal-second.csv"
timeout 110s "$exe" --condition kv_delayed_annealed_curriculum --seed 47 \
  --steps 4 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$phase9" --causal-output "$phase9_causal" >/dev/null
timeout 110s "$exe" --condition kv_delayed_annealed_curriculum --seed 47 \
  --steps 4 --batch-size 2 --train-count 12 --holdout-count 6 \
  --output "$phase9_second" --causal-output "$phase9_causal_second" >/dev/null
cmp "$phase9" "$phase9_second"
cmp "$phase9_causal" "$phase9_causal_second"
grep -q '^kv_delayed_annealed_curriculum,47,4,2,145,' "$phase9"
grep -q ',kv_delayed_annealed_curriculum,' "$phase9"
awk -F, 'NR==2 { exit !($8 == 51 && $33 == 8 && $71 > 0 && $72 == 18 && $85 != "" && $99 != "") }' "$phase9"
[[ $(wc -l < "$phase9_causal") -eq 14 ]]
for intervention in intact zero_query_context randomized_retention_actions force_fifo_retention force_relevant_eviction permuted_context_labels disable_retention_skip no_workspace_writes no_workspace_persistence no_mechanism_output remove_delay_distractors relevant_looking_delay_distractors reverse_delay_block; do
  grep -q "^47,$intervention," "$phase9_causal"
done
