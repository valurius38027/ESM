#include "test_harness.hpp"

#include "sgw/model.hpp"
#include "sgw/presets.hpp"

#include <string_view>

SGW_TEST(model_preset_parsing_is_explicit) {
  SGW_REQUIRE(sgw::parse_model_preset("core_small") ==
              sgw::ModelPreset::core_small);
  SGW_REQUIRE(sgw::parse_model_preset("core_compute_matched") ==
              sgw::ModelPreset::core_compute_matched);
  SGW_REQUIRE(sgw::parse_model_preset("core_param_matched") ==
              sgw::ModelPreset::core_param_matched);
  SGW_REQUIRE(sgw::parse_model_preset("sgw") ==
              sgw::ModelPreset::sgw_redundant);
  SGW_REQUIRE(sgw::parse_model_preset("sgw_redundant") ==
              sgw::ModelPreset::sgw_redundant);
  SGW_REQUIRE(sgw::parse_model_preset("sgw_broadcast_forced") ==
              sgw::ModelPreset::sgw_broadcast_forced);
  SGW_REQUIRE_THROWS(sgw::parse_model_preset("unknown"));
  SGW_REQUIRE(
      sgw::model_preset_name(sgw::ModelPreset::core_compute_matched) ==
      std::string_view("core_compute_matched"));
}

SGW_TEST(compute_matched_core_equals_sgw_active_compute) {
  const auto matched =
      sgw::make_model_config(sgw::ModelPreset::core_compute_matched);
  const auto workspace = sgw::make_model_config(sgw::ModelPreset::sgw_redundant);
  sgw::SgwEsmModel matched_model(matched, 1);
  sgw::SgwEsmModel workspace_model(workspace, 1);

  SGW_REQUIRE(sgw::estimated_step_madds(matched) == 1092);
  SGW_REQUIRE(sgw::estimated_step_madds(workspace) == 1092);
  SGW_REQUIRE(matched_model.parameters().scalar_count() == 2232);
  SGW_REQUIRE(workspace_model.parameters().scalar_count() == 2182);
}

SGW_TEST(parameter_matched_core_equals_sgw_parameter_count_within_one) {
  const auto matched =
      sgw::make_model_config(sgw::ModelPreset::core_param_matched);
  const auto workspace = sgw::make_model_config(sgw::ModelPreset::sgw_redundant);
  sgw::SgwEsmModel matched_model(matched, 1);
  sgw::SgwEsmModel workspace_model(workspace, 1);

  SGW_REQUIRE(matched_model.parameters().scalar_count() == 2181);
  SGW_REQUIRE(workspace_model.parameters().scalar_count() == 2182);
  SGW_REQUIRE(sgw::estimated_step_madds(matched) == 1474);
}

SGW_TEST(broadcast_forced_preset_disables_direct_workspace_bypasses) {
  const auto redundant =
      sgw::make_model_config(sgw::ModelPreset::sgw_redundant);
  const auto forced =
      sgw::make_model_config(sgw::ModelPreset::sgw_broadcast_forced);
  sgw::SgwEsmModel redundant_model(redundant, 1);
  sgw::SgwEsmModel forced_model(forced, 1);

  SGW_REQUIRE(redundant.spine_reads_workspace);
  SGW_REQUIRE(redundant.output_reads_workspace);
  SGW_REQUIRE(redundant.output_reads_mechanism);
  SGW_REQUIRE(!forced.spine_reads_workspace);
  SGW_REQUIRE(!forced.output_reads_workspace);
  SGW_REQUIRE(forced.output_reads_mechanism);
  SGW_REQUIRE(redundant_model.parameters().scalar_count() == 2182);
  SGW_REQUIRE(forced_model.parameters().scalar_count() == 2182);
  SGW_REQUIRE(sgw::estimated_step_madds(redundant) == 1092);
  SGW_REQUIRE(sgw::estimated_step_madds(forced) == 1002);
}

SGW_TEST(core_small_preserves_phase1_reference_budget) {
  const auto config = sgw::make_model_config(sgw::ModelPreset::core_small);
  sgw::SgwEsmModel model(config, 1);
  SGW_REQUIRE(config.core_only);
  SGW_REQUIRE(config.embedding_dim == 6);
  SGW_REQUIRE(config.spine_dim == 10);
  SGW_REQUIRE(model.parameters().scalar_count() == 327);
  SGW_REQUIRE(sgw::estimated_step_madds(config) == 210);
}

SGW_TEST(experiment_conditions_separate_topology_from_training_objective) {
  using sgw::ExperimentCondition;
  SGW_REQUIRE(sgw::parse_experiment_condition("core_small") ==
              ExperimentCondition::core_small);
  SGW_REQUIRE(sgw::parse_experiment_condition(
                  "sgw_redundant_final_only") ==
              ExperimentCondition::sgw_redundant_final_only);
  SGW_REQUIRE(sgw::parse_experiment_condition(
                  "sgw_broadcast_forced_final_only") ==
              ExperimentCondition::sgw_broadcast_forced_final_only);
  SGW_REQUIRE(sgw::parse_experiment_condition(
                  "sgw_broadcast_forced_aux_annealed") ==
              ExperimentCondition::sgw_broadcast_forced_aux_annealed);
  SGW_REQUIRE_THROWS(sgw::parse_experiment_condition("unknown"));

  SGW_REQUIRE(sgw::model_preset_for_condition(
                  ExperimentCondition::sgw_redundant_final_only) ==
              sgw::ModelPreset::sgw_redundant);
  SGW_REQUIRE(sgw::model_preset_for_condition(
                  ExperimentCondition::sgw_broadcast_forced_final_only) ==
              sgw::ModelPreset::sgw_broadcast_forced);
  SGW_REQUIRE(sgw::model_preset_for_condition(
                  ExperimentCondition::sgw_broadcast_forced_aux_annealed) ==
              sgw::ModelPreset::sgw_broadcast_forced);

  SGW_REQUIRE_NEAR(sgw::condition_workspace_aux_weight(
                       ExperimentCondition::sgw_broadcast_forced_final_only),
                   0.0, 0.0);
  SGW_REQUIRE_NEAR(sgw::condition_workspace_aux_weight(
                       ExperimentCondition::sgw_broadcast_forced_aux_annealed),
                   0.5, 0.0);
  SGW_REQUIRE(sgw::condition_workspace_aux_anneal_steps(
                  ExperimentCondition::sgw_broadcast_forced_aux_annealed) ==
              200);
  SGW_REQUIRE(
      sgw::experiment_condition_name(
          ExperimentCondition::sgw_broadcast_forced_aux_annealed) ==
      std::string_view("sgw_broadcast_forced_aux_annealed"));
}

namespace {

bool has_parameter(const sgw::SgwEsmModel& model, std::string_view name) {
  for (const auto& parameter : model.parameters().parameters()) {
    if (parameter->name() == name) {
      return true;
    }
  }
  return false;
}

}  // namespace

SGW_TEST(phase4_conditions_map_to_content_controls_and_fixed_mediation) {
  using sgw::ExperimentCondition;
  SGW_REQUIRE(sgw::parse_experiment_condition("core_full_content") ==
              ExperimentCondition::core_full_content);
  SGW_REQUIRE(sgw::parse_experiment_condition("core_content_blind") ==
              ExperimentCondition::core_content_blind);
  SGW_REQUIRE(sgw::parse_experiment_condition("mediation_final_only") ==
              ExperimentCondition::mediation_final_only);
  SGW_REQUIRE(sgw::parse_experiment_condition("mediation_aux_annealed") ==
              ExperimentCondition::mediation_aux_annealed);
  SGW_REQUIRE(sgw::model_preset_for_condition(
                  ExperimentCondition::mediation_aux_annealed) ==
              sgw::ModelPreset::mediation_fixed);
  SGW_REQUIRE_NEAR(sgw::condition_workspace_aux_weight(
                       ExperimentCondition::mediation_aux_annealed),
                   0.5, 0.0);
  SGW_REQUIRE(sgw::condition_workspace_aux_anneal_steps(
                  ExperimentCondition::mediation_aux_annealed) == 200);
}

SGW_TEST(phase4_presets_remove_content_bypass_and_learned_address_parameters) {
  const auto full =
      sgw::make_model_config(sgw::ModelPreset::core_full_content, 13, 5);
  const auto blind =
      sgw::make_model_config(sgw::ModelPreset::core_content_blind, 13, 5);
  const auto mediation =
      sgw::make_model_config(sgw::ModelPreset::mediation_fixed, 13, 5);
  sgw::SgwEsmModel full_model(full, 1);
  sgw::SgwEsmModel blind_model(blind, 1);
  sgw::SgwEsmModel mediation_model(mediation, 1);

  SGW_REQUIRE(full.spine_reads_embedding);
  SGW_REQUIRE(!blind.spine_reads_embedding);
  SGW_REQUIRE(full_model.parameters().scalar_count() == 1169);
  SGW_REQUIRE(blind_model.parameters().scalar_count() == 725);
  SGW_REQUIRE(mediation.fixed_binding_mediation);
  SGW_REQUIRE(mediation.mechanism_count == 4);
  SGW_REQUIRE(mediation.workspace_slots == 3);
  SGW_REQUIRE(mediation.active_mechanisms == 1);
  SGW_REQUIRE(mediation.workspace_writers == 1);
  SGW_REQUIRE(mediation.broadcast_recipients == 1);
  SGW_REQUIRE(mediation_model.parameters().scalar_count() == 15650);
  SGW_REQUIRE(sgw::estimated_step_madds(mediation) == 6784);
  SGW_REQUIRE(!has_parameter(mediation_model, "router_embedding"));
  SGW_REQUIRE(!has_parameter(mediation_model, "writer_key"));
  SGW_REQUIRE(!has_parameter(mediation_model, "slot_key"));
  SGW_REQUIRE(!has_parameter(mediation_model, "recipient_key"));
}

SGW_TEST(phase5_conditions_preserve_topology_and_bound_without_parameters) {
  using sgw::ExperimentCondition;
  SGW_REQUIRE(sgw::parse_experiment_condition("structural_core_full") ==
              ExperimentCondition::structural_core_full);
  SGW_REQUIRE(sgw::parse_experiment_condition("structural_core_blind") ==
              ExperimentCondition::structural_core_blind);
  SGW_REQUIRE(sgw::parse_experiment_condition("structural_mediation_linear") ==
              ExperimentCondition::structural_mediation_linear);
  SGW_REQUIRE(sgw::parse_experiment_condition("structural_mediation_bounded") ==
              ExperimentCondition::structural_mediation_bounded);

  const auto linear = sgw::make_model_config(
      sgw::ModelPreset::structural_mediation_linear, 13, 5);
  const auto bounded = sgw::make_model_config(
      sgw::ModelPreset::structural_mediation_bounded, 13, 5);
  sgw::SgwEsmModel linear_model(linear, 3);
  sgw::SgwEsmModel bounded_model(bounded, 3);
  SGW_REQUIRE_NEAR(linear.output_logit_bound, 0.0, 0.0);
  SGW_REQUIRE_NEAR(bounded.output_logit_bound, 1.0, 0.0);
  SGW_REQUIRE(linear_model.parameters().scalar_count() ==
              bounded_model.parameters().scalar_count());
  SGW_REQUIRE(sgw::estimated_step_madds(linear) ==
              sgw::estimated_step_madds(bounded));
}

SGW_TEST(phase6_conditions_map_to_exact_and_learned_key_value_presets) {
  using sgw::ExperimentCondition;
  SGW_REQUIRE(sgw::parse_experiment_condition("structural_kv_exact") ==
              ExperimentCondition::structural_kv_exact);
  SGW_REQUIRE(sgw::parse_experiment_condition("structural_kv_learned") ==
              ExperimentCondition::structural_kv_learned);
  SGW_REQUIRE(sgw::model_preset_for_condition(
                  ExperimentCondition::structural_kv_exact) ==
              sgw::ModelPreset::structural_kv_exact);
  SGW_REQUIRE(sgw::model_preset_for_condition(
                  ExperimentCondition::structural_kv_learned) ==
              sgw::ModelPreset::structural_kv_learned);
  SGW_REQUIRE(sgw::experiment_condition_name(
                  ExperimentCondition::structural_kv_learned) ==
              std::string_view("structural_kv_learned"));
}

SGW_TEST(phase6_presets_fix_slot_identity_and_parameter_budget) {
  const auto exact = sgw::make_model_config(
      sgw::ModelPreset::structural_kv_exact, 13, 5);
  const auto learned = sgw::make_model_config(
      sgw::ModelPreset::structural_kv_learned, 13, 5);
  sgw::SgwEsmModel exact_model(exact, 9);
  sgw::SgwEsmModel learned_model(learned, 9);

  SGW_REQUIRE(exact.key_value_mode ==
              sgw::KeyValueMediationMode::symbolic_exact);
  SGW_REQUIRE(learned.key_value_mode ==
              sgw::KeyValueMediationMode::learned_tied);
  SGW_REQUIRE(exact.workspace_slots == 3);
  SGW_REQUIRE(learned.workspace_slots == 3);
  SGW_REQUIRE(learned.key_dim == 8);
  SGW_REQUIRE(learned.value_dim == 8);
  SGW_REQUIRE(learned.workspace_dim == 16);
  SGW_REQUIRE_NEAR(learned.key_value_logit_scale, 6.0, 0.0);
  SGW_REQUIRE(exact_model.parameters().scalar_count() == 0);
  SGW_REQUIRE(learned_model.parameters().scalar_count() == 88);
  SGW_REQUIRE(has_parameter(learned_model, "kv_entity_codebook"));
  SGW_REQUIRE(has_parameter(learned_model, "kv_value_codebook"));
}

SGW_TEST(phase7_conditions_map_to_position_independent_write_presets) {
  using sgw::ExperimentCondition;
  SGW_REQUIRE(sgw::parse_experiment_condition("kv_fixed_position") ==
              ExperimentCondition::kv_fixed_position);
  SGW_REQUIRE(sgw::parse_experiment_condition("kv_first_free") ==
              ExperimentCondition::kv_first_free);
  SGW_REQUIRE(sgw::parse_experiment_condition("kv_hard_router") ==
              ExperimentCondition::kv_hard_router);
  SGW_REQUIRE(sgw::parse_experiment_condition("kv_annealed_router") ==
              ExperimentCondition::kv_annealed_router);
  SGW_REQUIRE(sgw::model_preset_for_condition(
                  ExperimentCondition::kv_annealed_router) ==
              sgw::ModelPreset::kv_annealed_router);
  SGW_REQUIRE(sgw::experiment_condition_name(
                  ExperimentCondition::kv_first_free) ==
              std::string_view("kv_first_free"));
}

SGW_TEST(phase7_router_presets_have_exact_parameter_budgets) {
  const auto fixed = sgw::make_model_config(
      sgw::ModelPreset::kv_fixed_position, 13, 5);
  const auto first_free = sgw::make_model_config(
      sgw::ModelPreset::kv_first_free, 13, 5);
  const auto hard = sgw::make_model_config(
      sgw::ModelPreset::kv_hard_router, 13, 5);
  const auto annealed = sgw::make_model_config(
      sgw::ModelPreset::kv_annealed_router, 13, 5);
  sgw::SgwEsmModel fixed_model(fixed, 17);
  sgw::SgwEsmModel first_free_model(first_free, 17);
  sgw::SgwEsmModel hard_model(hard, 17);
  sgw::SgwEsmModel annealed_model(annealed, 17);

  SGW_REQUIRE(fixed.key_value_write_routing ==
              sgw::KeyValueWriteRoutingMode::fixed_position);
  SGW_REQUIRE(first_free.key_value_write_routing ==
              sgw::KeyValueWriteRoutingMode::first_free);
  SGW_REQUIRE(hard.key_value_write_routing ==
              sgw::KeyValueWriteRoutingMode::hard_learned);
  SGW_REQUIRE(annealed.key_value_write_routing ==
              sgw::KeyValueWriteRoutingMode::annealed_learned);
  SGW_REQUIRE(fixed_model.parameters().scalar_count() == 88);
  SGW_REQUIRE(first_free_model.parameters().scalar_count() == 88);
  SGW_REQUIRE(hard_model.parameters().scalar_count() == 112);
  SGW_REQUIRE(annealed_model.parameters().scalar_count() == 112);
  SGW_REQUIRE_NEAR(annealed.key_value_router_initial_temperature, 2.0, 0.0);
  SGW_REQUIRE_NEAR(annealed.key_value_router_final_temperature, 0.1, 0.0);
  SGW_REQUIRE(annealed.key_value_router_anneal_steps == 600);
}
