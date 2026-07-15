#include "test_harness.hpp"

#include "sgw/dataset.hpp"
#include "sgw/experiment.hpp"
#include "sgw/model.hpp"

#include <numeric>

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
