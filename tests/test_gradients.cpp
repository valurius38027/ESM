#include "test_harness.hpp"

#include "sgw/dataset.hpp"
#include "sgw/experiment.hpp"
#include "sgw/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

sgw::ModelConfig gradient_config(const sgw::BindingDataset& dataset) {
  sgw::ModelConfig config;
  config.vocab_size = dataset.vocabulary.size();
  config.output_classes = dataset.config.value_count;
  config.embedding_dim = 3;
  config.spine_dim = 4;
  config.mechanism_count = 3;
  config.mechanism_dim = 3;
  config.workspace_slots = 2;
  config.workspace_dim = 3;
  config.active_mechanisms = 2;
  config.workspace_writers = 1;
  config.broadcast_recipients = 2;
  return config;
}

struct LossTrace {
  double loss;
  std::vector<sgw::StepTrace> traces;
};

LossTrace loss_and_trace(sgw::SgwEsmModel& model,
                         const sgw::BindingSample& sample) {
  sgw::ad::Tape tape;
  auto result = model.forward_sequence(tape, sample.tokens, true);
  const auto loss =
      sgw::cross_entropy_loss(tape, result.logits, sample.target_class);
  return LossTrace{loss.value(), std::move(result.traces)};
}

}  // namespace

SGW_TEST(model_gradients_match_finite_difference_away_from_route_boundaries) {
  sgw::BindingTaskConfig task;
  task.entity_count = 3;
  task.value_count = 3;
  task.filler_count = 2;
  task.binding_count = 1;
  task.fillers_per_binding = 1;
  const auto dataset = sgw::make_binding_split(task, 6, 2, 91);
  sgw::SgwEsmModel model(gradient_config(dataset), 123);
  model.parameters().zero_grad();

  sgw::ad::Tape tape;
  auto result = model.forward_sequence(tape, dataset.train.front().tokens, true);
  const auto loss = sgw::cross_entropy_loss(
      tape, result.logits, dataset.train.front().target_class);
  tape.backward(loss);
  const auto base_traces = result.traces;

  constexpr double epsilon = 1.0e-5;
  std::size_t accepted = 0;
  for (auto& parameter_pointer : model.parameters().parameters()) {
    auto& parameter = *parameter_pointer;
    for (std::size_t index = 0; index < parameter.size(); ++index) {
      const double analytic = parameter.gradient(index);
      if (std::abs(analytic) < 1.0e-7) {
        continue;
      }
      const double original = parameter.value(index);
      parameter.value(index) = original + epsilon;
      const auto plus = loss_and_trace(model, dataset.train.front());
      parameter.value(index) = original - epsilon;
      const auto minus = loss_and_trace(model, dataset.train.front());
      parameter.value(index) = original;

      if (!sgw::same_route_trace(base_traces, plus.traces) ||
          !sgw::same_route_trace(base_traces, minus.traces)) {
        continue;
      }
      const double numeric = (plus.loss - minus.loss) / (2.0 * epsilon);
      const double tolerance =
          2.0e-5 + 2.0e-3 * std::max(std::abs(analytic), std::abs(numeric));
      SGW_REQUIRE(std::abs(analytic - numeric) <= tolerance);
      if (++accepted == 8) {
        break;
      }
    }
    if (accepted == 8) {
      break;
    }
  }
  SGW_REQUIRE(accepted >= 4);
}
