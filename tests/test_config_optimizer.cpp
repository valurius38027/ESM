#include "test_harness.hpp"

#include "sgw/config.hpp"
#include "sgw/optimizer.hpp"
#include "sgw/presets.hpp"
#include "sgw/tensor.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

SGW_TEST(xavier_initialization_is_deterministic_and_bounded) {
  std::mt19937_64 first_rng(42);
  std::mt19937_64 second_rng(42);
  sgw::ParameterSet first;
  sgw::ParameterSet second;
  auto& a = first.add_xavier("a", 3, 5, first_rng);
  auto& b = second.add_xavier("b", 3, 5, second_rng);
  SGW_REQUIRE(a.values().size() == b.values().size());
  const double bound = std::sqrt(6.0 / 8.0);
  for (std::size_t index = 0; index < a.size(); ++index) {
    SGW_REQUIRE_NEAR(a.value(index), b.value(index), 0.0);
    SGW_REQUIRE(std::abs(a.value(index)) <= bound);
  }
}

SGW_TEST(parameter_checks_shape_and_indices) {
  SGW_REQUIRE_THROWS(sgw::Parameter("empty", 0, 2));
  sgw::Parameter parameter("matrix", 2, 3);
  parameter.value(1, 2) = 7.0;
  SGW_REQUIRE_NEAR(parameter.value(5), 7.0, 0.0);
  SGW_REQUIRE_THROWS(parameter.value(2, 0));
  SGW_REQUIRE_THROWS(parameter.value(6));
}

SGW_TEST(model_config_rejects_invalid_dimensions_and_budgets) {
  sgw::ModelConfig config;
  config.validate();

  auto invalid = config;
  invalid.vocab_size = 0;
  SGW_REQUIRE_THROWS(invalid.validate());

  invalid = config;
  invalid.active_mechanisms = 0;
  SGW_REQUIRE_THROWS(invalid.validate());

  invalid = config;
  invalid.active_mechanisms = config.mechanism_count + 1;
  SGW_REQUIRE_THROWS(invalid.validate());

  invalid = config;
  invalid.workspace_writers = config.active_mechanisms + 1;
  SGW_REQUIRE_THROWS(invalid.validate());

  invalid = config;
  invalid.broadcast_recipients = config.mechanism_count + 1;
  SGW_REQUIRE_THROWS(invalid.validate());

  invalid = config;
  invalid.output_classes = config.vocab_size + 1;
  SGW_REQUIRE_THROWS(invalid.validate());
}

SGW_TEST(parameter_set_reports_global_gradient_norm) {
  sgw::ParameterSet parameters;
  auto& parameter = parameters.add_zeros("vector", 1, 2);
  parameter.gradient(0) = 3.0;
  parameter.gradient(1) = 4.0;
  SGW_REQUIRE_NEAR(parameters.global_grad_norm(), 5.0, 1.0e-12);
  parameters.zero_grad();
  SGW_REQUIRE_NEAR(parameters.global_grad_norm(), 0.0, 0.0);
}

SGW_TEST(adam_matches_first_step_formula_and_uses_one_global_clip_scale) {
  sgw::ParameterSet parameters;
  auto& first = parameters.add_zeros("first", 1, 1);
  auto& second = parameters.add_zeros("second", 1, 1);
  first.value(0) = 1.0;
  second.value(0) = -2.0;
  first.gradient(0) = 3.0;
  second.gradient(0) = 4.0;

  sgw::AdamConfig config;
  config.learning_rate = 0.1;
  config.beta1 = 0.9;
  config.beta2 = 0.999;
  config.epsilon = 1.0e-8;
  config.max_grad_norm = 1.0;
  sgw::Adam optimizer(config);
  optimizer.step(parameters);

  SGW_REQUIRE_NEAR(optimizer.last_clip_scale(), 0.2, 1.0e-12);
  const double expected_first =
      1.0 - 0.1 * 0.6 / (std::sqrt(0.36) + 1.0e-8);
  const double expected_second =
      -2.0 - 0.1 * 0.8 / (std::sqrt(0.64) + 1.0e-8);
  SGW_REQUIRE_NEAR(first.value(0), expected_first, 1.0e-12);
  SGW_REQUIRE_NEAR(second.value(0), expected_second, 1.0e-12);
  SGW_REQUIRE(optimizer.step_count() == 1);
}

SGW_TEST(adam_config_rejects_invalid_hyperparameters) {
  sgw::AdamConfig config;
  config.validate();
  config.learning_rate = 0.0;
  SGW_REQUIRE_THROWS(config.validate());
  config = {};
  config.beta1 = 1.0;
  SGW_REQUIRE_THROWS(config.validate());
  config = {};
  config.beta2 = -0.1;
  SGW_REQUIRE_THROWS(config.validate());
  config = {};
  config.max_grad_norm = 0.0;
  SGW_REQUIRE_THROWS(config.validate());
}

SGW_TEST(fixed_binding_mediation_rejects_bypasses_and_invalid_budgets) {
  sgw::ModelConfig config;
  config.core_only = false;
  config.spine_reads_embedding = false;
  config.spine_reads_workspace = false;
  config.output_reads_spine = false;
  config.output_reads_workspace = false;
  config.output_reads_mechanism = true;
  config.fixed_binding_mediation = true;
  config.mediation_binding_count = 3;
  config.mechanism_count = 4;
  config.workspace_slots = 3;
  config.active_mechanisms = 1;
  config.workspace_writers = 1;
  config.broadcast_recipients = 1;
  config.validate();

  auto invalid = config;
  invalid.spine_reads_embedding = true;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.output_reads_spine = true;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.active_mechanisms = 2;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.workspace_slots = 2;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.mechanism_count = 3;
  SGW_REQUIRE_THROWS(invalid.validate());
}

SGW_TEST(model_config_rejects_invalid_output_logit_bounds) {
  sgw::ModelConfig config;
  config.output_logit_bound = -1.0;
  SGW_REQUIRE_THROWS(config.validate());
  config.output_logit_bound = std::numeric_limits<double>::infinity();
  SGW_REQUIRE_THROWS(config.validate());
  config.output_logit_bound = 1.0;
  config.validate();
}
#include <limits>

SGW_TEST(key_value_config_rejects_bypasses_and_invalid_shapes) {
  sgw::ModelConfig config;
  config.core_only = false;
  config.vocab_size = 13;
  config.output_classes = 5;
  config.key_value_mode = sgw::KeyValueMediationMode::learned_tied;
  config.entity_count = 6;
  config.value_count = 5;
  config.key_dim = 8;
  config.value_dim = 8;
  config.workspace_slots = 3;
  config.workspace_dim = 16;
  config.mechanism_count = 4;
  config.mediation_binding_count = 3;
  config.active_mechanisms = 1;
  config.workspace_writers = 1;
  config.broadcast_recipients = 1;
  config.spine_reads_embedding = false;
  config.spine_reads_workspace = false;
  config.output_reads_spine = false;
  config.output_reads_workspace = false;
  config.output_reads_mechanism = true;
  config.validate();

  auto invalid = config;
  invalid.workspace_dim = 15;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.entity_count = 0;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.value_count = 4;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.fixed_binding_mediation = true;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.output_reads_spine = true;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.key_value_logit_scale = 0.0;
  SGW_REQUIRE_THROWS(invalid.validate());
}

SGW_TEST(key_value_router_config_rejects_invalid_schedule) {
  auto config = sgw::make_model_config(
      sgw::ModelPreset::kv_annealed_router, 13, 5);
  config.validate();
  auto invalid = config;
  invalid.key_value_router_initial_temperature = 0.0;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.key_value_router_final_temperature = 0.0;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.key_value_router_final_temperature = 3.0;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.key_value_router_anneal_steps = 0;
  SGW_REQUIRE_THROWS(invalid.validate());
}

SGW_TEST(key_value_retention_config_rejects_invalid_capacity_and_context) {
  auto config = sgw::make_model_config(
      sgw::ModelPreset::kv_annealed_retention, 20, 6);
  config.validate();

  auto invalid = config;
  invalid.context_count = 0;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.workspace_slots = invalid.mediation_binding_count;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.mediation_binding_count = 3;
  SGW_REQUIRE_THROWS(invalid.validate());
  invalid = config;
  invalid.key_value_router_anneal_steps = 0;
  SGW_REQUIRE_THROWS(invalid.validate());

  auto full = sgw::make_model_config(
      sgw::ModelPreset::kv_full_capacity, 20, 6);
  full.workspace_slots = 3;
  SGW_REQUIRE_THROWS(full.validate());
}
