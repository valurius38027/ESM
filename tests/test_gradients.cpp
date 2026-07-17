#include "test_harness.hpp"

#include "sgw/dataset.hpp"
#include "sgw/experiment.hpp"
#include "sgw/model.hpp"
#include "sgw/presets.hpp"

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

SGW_TEST(workspace_auxiliary_logits_depend_on_workspace_projection_only) {
  sgw::BindingTaskConfig task;
  task.entity_count = 3;
  task.value_count = 3;
  task.filler_count = 2;
  task.binding_count = 1;
  task.fillers_per_binding = 1;
  const auto dataset = sgw::make_binding_split(task, 6, 2, 515);
  auto config = gradient_config(dataset);
  config.spine_reads_workspace = false;
  config.output_reads_workspace = false;
  sgw::SgwEsmModel model(config, 17);

  auto aux_values = [&](sgw::SgwEsmModel& current) {
    sgw::ad::Tape tape;
    const auto result =
        current.forward_sequence(tape, dataset.train.front().tokens, false);
    std::vector<double> values;
    for (const auto value : result.workspace_aux_logits) {
      values.push_back(value.value());
    }
    return values;
  };

  const auto original = aux_values(model);
  SGW_REQUIRE(original.size() == config.output_classes);

  for (auto& parameter : model.parameters().parameters()) {
    if (parameter->name() == "output_spine" ||
        parameter->name() == "output_mechanism") {
      parameter->value(0) += 1.0;
    }
  }
  const auto after_non_workspace = aux_values(model);
  SGW_REQUIRE(original == after_non_workspace);

  for (auto& parameter : model.parameters().parameters()) {
    if (parameter->name() == "output_workspace") {
      parameter->value(0) += 1.0;
    }
  }
  const auto after_workspace = aux_values(model);
  SGW_REQUIRE(original != after_workspace);
}

SGW_TEST(annealed_slot_router_receives_gradient_before_hard_only_window) {
  sgw::BindingTaskConfig task;
  task.entity_count = 6;
  task.value_count = 5;
  task.filler_count = 1;
  task.binding_count = 3;
  task.fillers_per_binding = 0;
  const auto dataset = sgw::make_structural_binding_split(task, 24, 8, 818);
  const auto config = sgw::make_model_config(
      sgw::ModelPreset::kv_annealed_router, dataset.vocabulary.size(), 5);
  sgw::SgwEsmModel model(config, 313);
  model.set_key_value_routing_step(0);
  model.parameters().zero_grad();
  sgw::ad::Tape tape;
  const auto result = model.forward_sequence(
      tape, dataset.train.front().tokens, true);
  const auto loss = sgw::cross_entropy_loss(
      tape, result.logits, dataset.train.front().target_class);
  tape.backward(loss);
  double router_gradient = 0.0;
  for (const auto& parameter : model.parameters().parameters()) {
    if (parameter->name() == "kv_slot_codebook") {
      for (const double value : parameter->gradients()) {
        router_gradient += std::abs(value);
      }
    }
  }
  SGW_REQUIRE(router_gradient > 1.0e-12);
}

SGW_TEST(annealed_retention_router_receives_gradient_before_hard_only_window) {
  sgw::RetentionTaskConfig task;
  const auto dataset = sgw::make_retention_binding_split(task, 24, 8, 828);
  const auto config = sgw::make_model_config(
      sgw::ModelPreset::kv_annealed_retention,
      dataset.vocabulary.size(), task.value_count);
  sgw::SgwEsmModel model(config, 317);

  const auto retention_gradient = [&](std::size_t step) {
    model.set_key_value_routing_step(step);
    model.parameters().zero_grad();
    for (std::size_t sample_index = 0; sample_index < 8; ++sample_index) {
      sgw::ad::Tape tape;
      const auto result = model.forward_sequence(
          tape, dataset.train[sample_index].tokens, true);
      const auto loss = sgw::cross_entropy_loss(
          tape, result.logits, dataset.train[sample_index].target_class);
      tape.backward(loss);
    }
    double total = 0.0;
    for (const auto& parameter : model.parameters().parameters()) {
      if (parameter->name() == "kv_context_codebook" ||
          parameter->name() == "kv_retention_age_weight") {
        for (const double value : parameter->gradients()) {
          total += std::abs(value);
        }
      }
    }
    return total;
  };

  SGW_REQUIRE(retention_gradient(0) > 1.0e-12);
  SGW_REQUIRE_NEAR(retention_gradient(900), 0.0, 1.0e-15);
}
