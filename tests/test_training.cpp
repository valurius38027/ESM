#include "test_harness.hpp"

#include "sgw/dataset.hpp"
#include "sgw/experiment.hpp"
#include "sgw/model.hpp"
#include "sgw/presets.hpp"

#include <algorithm>
#include <numeric>

#include <cmath>

namespace {

double window_mean(const std::vector<double>& values, std::size_t begin,
                   std::size_t count) {
  return std::accumulate(values.begin() + static_cast<std::ptrdiff_t>(begin),
                         values.begin() +
                             static_cast<std::ptrdiff_t>(begin + count),
                         0.0) /
         static_cast<double>(count);
}

sgw::ModelConfig training_model_config(const sgw::BindingDataset& dataset) {
  sgw::ModelConfig config;
  config.vocab_size = dataset.vocabulary.size();
  config.output_classes = dataset.config.value_count;
  config.embedding_dim = 4;
  config.spine_dim = 6;
  config.mechanism_count = 4;
  config.mechanism_dim = 4;
  config.workspace_slots = 2;
  config.workspace_dim = 4;
  config.active_mechanisms = 2;
  config.workspace_writers = 1;
  config.broadcast_recipients = 2;
  return config;
}

}  // namespace

SGW_TEST(fixed_seed_training_is_deterministic_and_reduces_loss) {
  sgw::BindingTaskConfig task;
  task.entity_count = 4;
  task.value_count = 3;
  task.filler_count = 3;
  task.binding_count = 1;
  task.fillers_per_binding = 2;
  const auto dataset = sgw::make_binding_split(task, 36, 12, 2026);

  const auto config = training_model_config(dataset);
  sgw::SgwEsmModel first(config, 77);
  sgw::SgwEsmModel second(config, 77);

  sgw::AdamConfig adam;
  adam.learning_rate = 0.02;
  adam.max_grad_norm = 5.0;
  sgw::TrainingConfig training;
  training.steps = 80;
  training.batch_size = 6;
  training.shuffle_seed = 909;

  const auto initial = sgw::evaluate(first, dataset.train, false);
  const auto first_history =
      sgw::train_steps(first, dataset.train, adam, training);
  const auto second_history =
      sgw::train_steps(second, dataset.train, adam, training);
  const auto final = sgw::evaluate(first, dataset.train, false);

  SGW_REQUIRE(first_history.batch_loss == second_history.batch_loss);
  SGW_REQUIRE(first_history.gradient_norm == second_history.gradient_norm);
  SGW_REQUIRE(first_history.clip_scale == second_history.clip_scale);
  SGW_REQUIRE(first_history.batch_loss.size() == training.steps);
  SGW_REQUIRE(window_mean(first_history.batch_loss, 70, 10) <
              window_mean(first_history.batch_loss, 0, 10));
  SGW_REQUIRE(final.mean_nll < initial.mean_nll);
  SGW_REQUIRE(first.parameters().all_finite());
  SGW_REQUIRE(second.parameters().all_finite());
}

SGW_TEST(workspace_auxiliary_schedule_anneals_exactly_to_zero) {
  sgw::TrainingConfig config;
  config.steps = 400;
  config.workspace_aux_weight = 0.5;
  config.workspace_aux_anneal_steps = 200;
  SGW_REQUIRE_NEAR(sgw::workspace_aux_weight_at_step(config, 0), 0.5,
                   0.0);
  SGW_REQUIRE_NEAR(sgw::workspace_aux_weight_at_step(config, 100), 0.25,
                   1.0e-12);
  SGW_REQUIRE_NEAR(sgw::workspace_aux_weight_at_step(config, 199), 0.0025,
                   1.0e-12);
  SGW_REQUIRE_NEAR(sgw::workspace_aux_weight_at_step(config, 200), 0.0,
                   0.0);
  SGW_REQUIRE_NEAR(sgw::workspace_aux_weight_at_step(config, 399), 0.0,
                   0.0);
}

SGW_TEST(workspace_auxiliary_training_records_primary_auxiliary_and_schedule) {
  sgw::BindingTaskConfig task;
  task.entity_count = 4;
  task.value_count = 3;
  task.filler_count = 3;
  task.binding_count = 1;
  task.fillers_per_binding = 2;
  const auto dataset = sgw::make_binding_split(task, 18, 6, 2121);

  auto config = training_model_config(dataset);
  config.spine_reads_workspace = false;
  config.output_reads_workspace = false;
  sgw::SgwEsmModel model(config, 99);

  sgw::AdamConfig adam;
  adam.learning_rate = 0.01;
  adam.max_grad_norm = 5.0;
  sgw::TrainingConfig training;
  training.steps = 5;
  training.batch_size = 3;
  training.shuffle_seed = 77;
  training.workspace_aux_weight = 0.5;
  training.workspace_aux_anneal_steps = 3;

  const auto history = sgw::train_steps(model, dataset.train, adam, training);
  SGW_REQUIRE(history.batch_loss.size() == training.steps);
  SGW_REQUIRE(history.primary_batch_loss.size() == training.steps);
  SGW_REQUIRE(history.workspace_aux_batch_loss.size() == training.steps);
  SGW_REQUIRE(history.workspace_aux_weight.size() == training.steps);
  SGW_REQUIRE_NEAR(history.workspace_aux_weight[0], 0.5, 0.0);
  SGW_REQUIRE_NEAR(history.workspace_aux_weight[3], 0.0, 0.0);
  SGW_REQUIRE_NEAR(history.workspace_aux_weight[4], 0.0, 0.0);
  for (std::size_t step = 0; step < training.steps; ++step) {
    SGW_REQUIRE(std::isfinite(history.primary_batch_loss[step]));
    SGW_REQUIRE(std::isfinite(history.workspace_aux_batch_loss[step]));
    SGW_REQUIRE_NEAR(
        history.batch_loss[step],
        history.primary_batch_loss[step] +
            history.workspace_aux_weight[step] *
                history.workspace_aux_batch_loss[step],
        1.0e-12);
  }
}


SGW_TEST(route_evaluation_reports_role_conditioned_mechanism_load) {
  sgw::BindingTaskConfig task;
  task.entity_count = 4;
  task.value_count = 3;
  task.filler_count = 3;
  task.binding_count = 2;
  task.fillers_per_binding = 1;
  const auto dataset = sgw::make_binding_split(task, 12, 6, 3030);

  const auto config = training_model_config(dataset);
  sgw::SgwEsmModel model(config, 55);
  const auto metrics = sgw::evaluate(model, dataset.holdout, true);

  const std::size_t role_count =
      static_cast<std::size_t>(sgw::TokenRole::count);
  SGW_REQUIRE(metrics.role_mechanism_load.size() ==
              role_count * config.mechanism_count);
  const std::size_t total = std::accumulate(
      metrics.role_mechanism_load.begin(),
      metrics.role_mechanism_load.end(), std::size_t{0});
  SGW_REQUIRE(total == dataset.holdout.size() *
                           dataset.config.sequence_length() *
                           config.active_mechanisms);

  auto core_config = config;
  core_config.core_only = true;
  core_config.spine_reads_workspace = false;
  core_config.output_reads_workspace = false;
  core_config.output_reads_mechanism = false;
  sgw::SgwEsmModel core(core_config, 55);
  const auto core_metrics = sgw::evaluate(core, dataset.holdout, true);
  SGW_REQUIRE(core_metrics.role_mechanism_load.size() ==
              role_count * core_config.mechanism_count);
  SGW_REQUIRE(std::accumulate(core_metrics.role_mechanism_load.begin(),
                              core_metrics.role_mechanism_load.end(),
                              std::size_t{0}) == 0);
}

SGW_TEST(uniform_predictions_have_exact_multiclass_calibration_metrics) {
  auto config = sgw::make_model_config(sgw::ModelPreset::core_content_blind,
                                       13, 5);
  sgw::SgwEsmModel model(config, 4);
  for (auto& parameter : model.parameters().parameters()) {
    for (double& value : parameter->mutable_values()) value = 0.0;
  }
  std::vector<sgw::BindingSample> samples;
  for (std::size_t target = 0; target < 5; ++target) {
    sgw::BindingSample sample;
    sample.tokens = {0, 6, 1, 7, 2, 8, 12, 0};
    sample.roles = {sgw::TokenRole::entity, sgw::TokenRole::value,
                    sgw::TokenRole::entity, sgw::TokenRole::value,
                    sgw::TokenRole::entity, sgw::TokenRole::value,
                    sgw::TokenRole::query_marker,
                    sgw::TokenRole::query_entity};
    sample.target_class = target;
    samples.push_back(sample);
  }
  const auto metrics = sgw::evaluate(model, samples, false);
  SGW_REQUIRE_NEAR(metrics.mean_nll, std::log(5.0), 1.0e-12);
  SGW_REQUIRE_NEAR(metrics.accuracy, 0.2, 1.0e-12);
  SGW_REQUIRE_NEAR(metrics.mean_brier, 0.8, 1.0e-12);
  SGW_REQUIRE_NEAR(metrics.ece, 0.0, 1.0e-12);
  SGW_REQUIRE_NEAR(metrics.mean_max_confidence, 0.2, 1.0e-12);
  SGW_REQUIRE_NEAR(metrics.mean_true_class_probability, 0.2, 1.0e-12);
}

SGW_TEST(stream_training_consumes_exact_batch_budget_deterministically) {
  sgw::BindingTaskConfig task;
  task.entity_count = 6;
  task.value_count = 5;
  task.filler_count = 1;
  task.binding_count = 3;
  task.fillers_per_binding = 0;
  const auto dataset = sgw::make_structural_binding_split(task, 32, 24, 81);
  auto config = sgw::make_model_config(
      sgw::ModelPreset::structural_mediation_bounded,
      dataset.vocabulary.size(), dataset.config.value_count);
  sgw::SgwEsmModel first(config, 17);
  sgw::SgwEsmModel second(config, 17);
  sgw::StructuralBindingStream first_stream(task, 1234);
  sgw::StructuralBindingStream second_stream(task, 1234);
  sgw::AdamConfig adam;
  adam.learning_rate = 0.01;
  adam.max_grad_norm = 5.0;
  sgw::TrainingConfig training;
  training.steps = 12;
  training.batch_size = 8;
  training.shuffle_seed = 7;
  const auto first_history =
      sgw::train_steps(first, first_stream, adam, training);
  const auto second_history =
      sgw::train_steps(second, second_stream, adam, training);
  SGW_REQUIRE(first_history.samples_consumed == 96);
  SGW_REQUIRE(second_history.samples_consumed == 96);
  SGW_REQUIRE(first_stream.samples_consumed() == 96);
  SGW_REQUIRE(first_history.batch_loss == second_history.batch_loss);
}

SGW_TEST(exact_key_value_training_consumes_stream_with_zero_parameters) {
  sgw::BindingTaskConfig task;
  task.entity_count = 6;
  task.value_count = 5;
  task.filler_count = 1;
  task.binding_count = 3;
  task.fillers_per_binding = 0;
  const auto dataset = sgw::make_structural_binding_split(task, 16, 12, 5151);
  const auto config = sgw::make_model_config(
      sgw::ModelPreset::structural_kv_exact,
      dataset.vocabulary.size(), dataset.config.value_count);
  sgw::SgwEsmModel model(config, 5);
  sgw::StructuralBindingStream stream(task, 77);
  sgw::AdamConfig adam;
  sgw::TrainingConfig training;
  training.steps = 4;
  training.batch_size = 3;
  const auto history = sgw::train_steps(model, stream, adam, training);
  SGW_REQUIRE(model.parameters().scalar_count() == 0);
  SGW_REQUIRE(history.samples_consumed == 12);
  SGW_REQUIRE(stream.samples_consumed() == 12);
  SGW_REQUIRE(std::all_of(history.gradient_norm.begin(),
                          history.gradient_norm.end(),
                          [](double value) { return value == 0.0; }));
}

SGW_TEST(phase7_training_records_routing_telemetry_and_hard_only_window) {
  sgw::BindingTaskConfig task;
  task.entity_count = 6;
  task.value_count = 5;
  task.filler_count = 1;
  task.binding_count = 3;
  task.fillers_per_binding = 0;
  sgw::StructuralBindingStream stream(task, 1701);
  const auto config = sgw::make_model_config(
      sgw::ModelPreset::kv_annealed_router, 13, 5);
  sgw::SgwEsmModel model(config, 1702);
  sgw::AdamConfig adam;
  adam.learning_rate = 0.01;
  sgw::TrainingConfig training;
  training.steps = config.key_value_router_anneal_steps + 2;
  training.batch_size = 1;
  const auto history = sgw::train_steps(model, stream, adam, training);
  SGW_REQUIRE(history.routing_collision_rate.size() == training.steps);
  SGW_REQUIRE(history.routing_entropy.size() == training.steps);
  SGW_REQUIRE(history.routing_disagreement_rate.size() == training.steps);
  SGW_REQUIRE(history.routing_collision_rate.back() == 0.0);
  SGW_REQUIRE(history.routing_disagreement_rate.back() == 0.0);
  SGW_REQUIRE(model.key_value_routing_step() == training.steps - 1);
  SGW_REQUIRE(!model.key_value_router_uses_surrogate());
}
