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
  SGW_REQUIRE(sgw::parse_model_preset("sgw") == sgw::ModelPreset::sgw);
  SGW_REQUIRE_THROWS(sgw::parse_model_preset("unknown"));
  SGW_REQUIRE(
      sgw::model_preset_name(sgw::ModelPreset::core_compute_matched) ==
      std::string_view("core_compute_matched"));
}

SGW_TEST(compute_matched_core_equals_sgw_active_compute) {
  const auto matched =
      sgw::make_model_config(sgw::ModelPreset::core_compute_matched);
  const auto workspace = sgw::make_model_config(sgw::ModelPreset::sgw);
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
  const auto workspace = sgw::make_model_config(sgw::ModelPreset::sgw);
  sgw::SgwEsmModel matched_model(matched, 1);
  sgw::SgwEsmModel workspace_model(workspace, 1);

  SGW_REQUIRE(matched_model.parameters().scalar_count() == 2181);
  SGW_REQUIRE(workspace_model.parameters().scalar_count() == 2182);
  SGW_REQUIRE(sgw::estimated_step_madds(matched) == 1474);
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
