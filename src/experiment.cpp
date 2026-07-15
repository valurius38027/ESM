#include "sgw/experiment.hpp"

#include "sgw/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace sgw {
namespace {

std::size_t bounded_random(std::mt19937_64& generator,
                           std::size_t upper_exclusive) {
  if (upper_exclusive == 0) {
    throw std::invalid_argument("random bound must be positive");
  }
  const std::uint64_t bound = static_cast<std::uint64_t>(upper_exclusive);
  const std::uint64_t threshold = static_cast<std::uint64_t>(-bound) % bound;
  while (true) {
    const std::uint64_t value = generator();
    if (value >= threshold) {
      return static_cast<std::size_t>(value % bound);
    }
  }
}

void deterministic_shuffle(std::vector<std::size_t>& values,
                           std::mt19937_64& generator) {
  for (std::size_t remaining = values.size(); remaining > 1; --remaining) {
    const std::size_t selected = bounded_random(generator, remaining);
    std::swap(values[remaining - 1], values[selected]);
  }
}

std::pair<double, std::size_t> numeric_loss_and_prediction(
    std::span<const ad::Var> logits, std::size_t target_class) {
  if (logits.empty() || target_class >= logits.size()) {
    throw std::invalid_argument("invalid logits or target class");
  }
  double maximum = logits.front().value();
  std::size_t prediction = 0;
  for (std::size_t index = 1; index < logits.size(); ++index) {
    const double value = logits[index].value();
    if (value > maximum) {
      maximum = value;
      prediction = index;
    }
  }
  double exp_sum = 0.0;
  for (const ad::Var logit : logits) {
    exp_sum += std::exp(logit.value() - maximum);
  }
  const double nll = std::log(exp_sum) + maximum - logits[target_class].value();
  if (!std::isfinite(nll)) {
    throw std::runtime_error("evaluation produced non-finite NLL");
  }
  return {nll, prediction};
}

}  // namespace

void TrainingConfig::validate() const {
  if (steps == 0) {
    throw std::invalid_argument("training steps must be positive");
  }
  if (batch_size == 0) {
    throw std::invalid_argument("batch size must be positive");
  }
  if (!std::isfinite(workspace_aux_weight) || workspace_aux_weight < 0.0) {
    throw std::invalid_argument(
        "workspace auxiliary weight must be finite and non-negative");
  }
  if (workspace_aux_weight > 0.0 && workspace_aux_anneal_steps == 0) {
    throw std::invalid_argument(
        "positive workspace auxiliary weight requires anneal steps");
  }
}

double workspace_aux_weight_at_step(const TrainingConfig& config,
                                    std::size_t step) {
  config.validate();
  if (config.workspace_aux_weight == 0.0 ||
      step >= config.workspace_aux_anneal_steps) {
    return 0.0;
  }
  const double fraction =
      1.0 - static_cast<double>(step) /
                static_cast<double>(config.workspace_aux_anneal_steps);
  return config.workspace_aux_weight * fraction;
}

ad::Var cross_entropy_loss(ad::Tape& tape, std::span<const ad::Var> logits,
                           std::size_t target_class) {
  if (logits.empty()) {
    throw std::invalid_argument("cross entropy requires non-empty logits");
  }
  if (target_class >= logits.size()) {
    throw std::out_of_range("target class out of range");
  }
  double maximum = logits.front().value();
  for (const ad::Var logit : logits) {
    maximum = std::max(maximum, logit.value());
  }
  const ad::Var maximum_constant = tape.constant(maximum);
  ad::Var denominator = tape.constant(0.0);
  for (const ad::Var logit : logits) {
    denominator = denominator + ad::exp(logit - maximum_constant);
  }
  return ad::log(denominator) + maximum_constant - logits[target_class];
}

bool same_route_trace(std::span<const StepTrace> first,
                      std::span<const StepTrace> second) noexcept {
  if (first.size() != second.size()) {
    return false;
  }
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (!first[index].same_route(second[index])) {
      return false;
    }
  }
  return true;
}

EvaluationMetrics evaluate(SgwEsmModel& model,
                           std::span<const BindingSample> samples,
                           bool collect_routes,
                           ForwardIntervention intervention) {
  if (samples.empty()) {
    throw std::invalid_argument("evaluation samples must not be empty");
  }
  EvaluationMetrics metrics;
  metrics.mechanism_load.assign(model.config().mechanism_count, 0);
  const std::size_t role_count =
      static_cast<std::size_t>(TokenRole::count);
  metrics.role_mechanism_load.assign(
      role_count * model.config().mechanism_count, 0);
  double total_nll = 0.0;
  std::size_t correct = 0;
  long double total_madds = 0.0L;
  std::size_t total_tokens = 0;
  std::size_t total_active = 0;
  std::size_t total_writers = 0;
  std::size_t total_recipients = 0;

  for (const BindingSample& sample : samples) {
    ad::Tape tape;
    const SequenceResult result =
        model.forward_sequence(tape, sample.tokens, collect_routes, intervention);
    const auto [nll, prediction] =
        numeric_loss_and_prediction(result.logits, sample.target_class);
    total_nll += nll;
    if (prediction == sample.target_class) {
      ++correct;
    }
    total_tokens += sample.tokens.size();
    total_madds += static_cast<long double>(
        estimated_step_madds(model.config()) * sample.tokens.size());
    if (collect_routes) {
      if (sample.roles.size() != result.traces.size()) {
        throw std::logic_error(
            "sample role count must match captured route trace count");
      }
      for (std::size_t step = 0; step < result.traces.size(); ++step) {
        const StepTrace& trace = result.traces[step];
        total_active += trace.active_mechanisms.size();
        total_writers += trace.writers.size();
        total_recipients += trace.recipients.size();
        const std::size_t role =
            static_cast<std::size_t>(sample.roles[step]);
        if (role >= role_count) {
          throw std::logic_error("invalid token role in evaluation sample");
        }
        for (const std::size_t mechanism : trace.active_mechanisms) {
          ++metrics.mechanism_load.at(mechanism);
          ++metrics.role_mechanism_load.at(
              role * model.config().mechanism_count + mechanism);
        }
      }
    }
  }

  metrics.mean_nll = total_nll / static_cast<double>(samples.size());
  metrics.accuracy =
      static_cast<double>(correct) / static_cast<double>(samples.size());
  metrics.mean_estimated_madds_per_token =
      static_cast<double>(total_madds / static_cast<long double>(total_tokens));
  if (collect_routes) {
    const double denominator = static_cast<double>(total_tokens);
    metrics.mean_active_mechanisms_per_token =
        static_cast<double>(total_active) / denominator;
    metrics.mean_writers_per_token =
        static_cast<double>(total_writers) / denominator;
    metrics.mean_recipients_per_token =
        static_cast<double>(total_recipients) / denominator;
  }
  return metrics;
}

TrainingHistory train_steps(SgwEsmModel& model,
                            std::span<const BindingSample> samples,
                            const AdamConfig& adam_config,
                            const TrainingConfig& training_config) {
  if (samples.empty()) {
    throw std::invalid_argument("training samples must not be empty");
  }
  adam_config.validate();
  training_config.validate();
  Adam optimizer(adam_config);
  TrainingHistory history;
  history.batch_loss.reserve(training_config.steps);
  history.primary_batch_loss.reserve(training_config.steps);
  history.workspace_aux_batch_loss.reserve(training_config.steps);
  history.workspace_aux_weight.reserve(training_config.steps);
  history.gradient_norm.reserve(training_config.steps);
  history.clip_scale.reserve(training_config.steps);

  std::vector<std::size_t> order(samples.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::mt19937_64 generator(training_config.shuffle_seed);
  deterministic_shuffle(order, generator);
  std::size_t cursor = 0;

  for (std::size_t step = 0; step < training_config.steps; ++step) {
    model.parameters().zero_grad();
    const double auxiliary_weight =
        workspace_aux_weight_at_step(training_config, step);
    double loss_sum = 0.0;
    double primary_loss_sum = 0.0;
    double auxiliary_loss_sum = 0.0;
    for (std::size_t batch_index = 0;
         batch_index < training_config.batch_size; ++batch_index) {
      if (cursor == order.size()) {
        deterministic_shuffle(order, generator);
        cursor = 0;
      }
      const BindingSample& sample = samples[order[cursor++]];
      ad::Tape tape;
      const SequenceResult result =
          model.forward_sequence(tape, sample.tokens, false);
      const ad::Var primary_loss =
          cross_entropy_loss(tape, result.logits, sample.target_class);
      ad::Var total_loss = primary_loss;
      double auxiliary_loss_value = 0.0;
      if (auxiliary_weight > 0.0) {
        if (result.workspace_aux_logits.empty()) {
          throw std::invalid_argument(
              "workspace auxiliary loss requires an SGW model");
        }
        const ad::Var auxiliary_loss = cross_entropy_loss(
            tape, result.workspace_aux_logits, sample.target_class);
        auxiliary_loss_value = auxiliary_loss.value();
        total_loss = total_loss +
                     tape.constant(auxiliary_weight) * auxiliary_loss;
      }
      primary_loss_sum += primary_loss.value();
      auxiliary_loss_sum += auxiliary_loss_value;
      loss_sum += total_loss.value();
      tape.backward(total_loss);
    }

    const double inverse_batch =
        1.0 / static_cast<double>(training_config.batch_size);
    for (auto& parameter_pointer : model.parameters().parameters()) {
      Parameter& parameter = *parameter_pointer;
      for (std::size_t index = 0; index < parameter.size(); ++index) {
        parameter.gradient(index) *= inverse_batch;
      }
    }
    const double gradient_norm = model.parameters().global_grad_norm();
    optimizer.step(model.parameters());
    history.batch_loss.push_back(loss_sum * inverse_batch);
    history.primary_batch_loss.push_back(primary_loss_sum * inverse_batch);
    history.workspace_aux_batch_loss.push_back(auxiliary_loss_sum * inverse_batch);
    history.workspace_aux_weight.push_back(auxiliary_weight);
    history.gradient_norm.push_back(gradient_norm);
    history.clip_scale.push_back(optimizer.last_clip_scale());
  }
  return history;
}

}  // namespace sgw
