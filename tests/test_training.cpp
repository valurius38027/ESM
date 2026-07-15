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
