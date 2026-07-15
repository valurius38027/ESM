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
