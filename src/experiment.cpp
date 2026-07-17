#include "sgw/experiment.hpp"

#include "sgw/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
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

struct NumericPrediction {
  double nll{0.0};
  std::size_t prediction{0};
  std::vector<double> probabilities;
};

NumericPrediction numeric_prediction(
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
  std::vector<double> probabilities(logits.size(), 0.0);
  double exp_sum = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    probabilities[index] = std::exp(logits[index].value() - maximum);
    exp_sum += probabilities[index];
  }
  for (double& probability : probabilities) probability /= exp_sum;
  const double nll = -std::log(probabilities[target_class]);
  if (!std::isfinite(nll)) {
    throw std::runtime_error("evaluation produced non-finite NLL");
  }
  return NumericPrediction{nll, prediction, std::move(probabilities)};
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
  metrics.read_slot_load.assign(model.config().workspace_slots, 0);
  metrics.write_slot_load.assign(model.config().workspace_slots, 0);
  metrics.eviction_slot_load.assign(model.config().workspace_slots, 0);
  double total_nll = 0.0;
  double total_brier = 0.0;
  double total_max_confidence = 0.0;
  double total_true_probability = 0.0;
  std::size_t correct = 0;
  constexpr std::size_t calibration_bins = 10;
  std::vector<std::size_t> bin_counts(calibration_bins, 0);
  std::vector<double> bin_confidence(calibration_bins, 0.0);
  std::vector<std::size_t> bin_correct(calibration_bins, 0);
  long double total_madds = 0.0L;
  std::size_t total_tokens = 0;
  std::size_t total_active = 0;
  std::size_t total_writers = 0;
  std::size_t total_recipients = 0;
  std::size_t total_write_collisions = 0;
  double total_routing_entropy = 0.0;
  std::size_t total_routing_decisions = 0;
  std::size_t total_routing_disagreements = 0;
  std::size_t total_retention_decisions = 0;
  std::size_t total_retention_writes = 0;
  std::size_t total_retention_skips = 0;
  std::size_t total_retention_evictions = 0;
  std::size_t total_relevant_evictions = 0;
  std::size_t total_query_sequences = 0;
  std::size_t total_queried_retained = 0;
  std::size_t total_query_read_hits = 0;
  double total_retained_age = 0.0;
  std::size_t total_retained_age_count = 0;

  for (const BindingSample& sample : samples) {
    ad::Tape tape;
    const SequenceResult result =
        model.forward_sequence(tape, sample.tokens, collect_routes, intervention);
    const NumericPrediction prediction =
        numeric_prediction(result.logits, sample.target_class);
    total_nll += prediction.nll;
    const bool is_correct = prediction.prediction == sample.target_class;
    if (is_correct) ++correct;
    double brier = 0.0;
    for (std::size_t index = 0; index < prediction.probabilities.size(); ++index) {
      const double expected = index == sample.target_class ? 1.0 : 0.0;
      const double error = prediction.probabilities[index] - expected;
      brier += error * error;
    }
    total_brier += brier;
    const double confidence = *std::max_element(
        prediction.probabilities.begin(), prediction.probabilities.end());
    total_max_confidence += confidence;
    total_true_probability += prediction.probabilities[sample.target_class];
    const std::size_t bin = std::min(
        calibration_bins - 1,
        static_cast<std::size_t>(confidence *
                                 static_cast<double>(calibration_bins)));
    ++bin_counts[bin];
    bin_confidence[bin] += confidence;
    if (is_correct) ++bin_correct[bin];
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
        total_write_collisions += trace.write_collisions;
        total_routing_entropy += trace.routing_entropy;
        total_routing_decisions += trace.routing_decisions;
        total_routing_disagreements += trace.hard_soft_disagreements;
        total_retention_writes += trace.retention_writes;
        total_retention_skips += trace.retention_skips;
        total_retention_evictions += trace.retention_evictions;
        total_relevant_evictions += trace.relevant_evictions;
        total_retention_decisions +=
            trace.retention_writes + trace.retention_skips;
        if (trace.queried_entity_retained || !trace.read_slots.empty()) {
          ++total_query_sequences;
          if (trace.queried_entity_retained) ++total_queried_retained;
          if (trace.query_read_hit) ++total_query_read_hits;
          total_retained_age += trace.retained_age_sum;
          total_retained_age_count += trace.retained_age_count;
        }
        for (const std::size_t slot : trace.writer_slots) {
          ++metrics.write_slot_load.at(slot);
        }
        if (trace.retention_evictions != 0 && !trace.writer_slots.empty()) {
          ++metrics.eviction_slot_load.at(trace.writer_slots.front());
        }
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
        for (const std::size_t slot : trace.read_slots) {
          ++metrics.read_slot_load.at(slot);
        }
      }
    }
  }

  metrics.mean_nll = total_nll / static_cast<double>(samples.size());
  const double sample_count = static_cast<double>(samples.size());
  metrics.accuracy = static_cast<double>(correct) / sample_count;
  metrics.mean_brier = total_brier / sample_count;
  metrics.mean_max_confidence = total_max_confidence / sample_count;
  metrics.mean_true_class_probability = total_true_probability / sample_count;
  for (std::size_t bin = 0; bin < calibration_bins; ++bin) {
    if (bin_counts[bin] == 0) continue;
    const double count = static_cast<double>(bin_counts[bin]);
    const double mean_confidence = bin_confidence[bin] / count;
    const double mean_accuracy = static_cast<double>(bin_correct[bin]) / count;
    metrics.ece += (count / sample_count) *
                   std::abs(mean_accuracy - mean_confidence);
  }
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
    if (total_routing_decisions != 0) {
      const double route_count =
          static_cast<double>(total_routing_decisions);
      metrics.mean_write_collision_rate =
          static_cast<double>(total_write_collisions) / route_count;
      metrics.mean_routing_entropy = total_routing_entropy / route_count;
      metrics.mean_routing_disagreement_rate =
          static_cast<double>(total_routing_disagreements) / route_count;
    }
    if (total_retention_decisions != 0) {
      const double decisions = static_cast<double>(total_retention_decisions);
      metrics.retention_write_rate =
          static_cast<double>(total_retention_writes) / decisions;
      metrics.retention_skip_rate =
          static_cast<double>(total_retention_skips) / decisions;
      metrics.retention_eviction_rate =
          static_cast<double>(total_retention_evictions) / decisions;
      metrics.relevant_eviction_rate =
          static_cast<double>(total_relevant_evictions) / decisions;
    }
    if (total_query_sequences != 0) {
      const double queries = static_cast<double>(total_query_sequences);
      metrics.queried_entity_retention_rate =
          static_cast<double>(total_queried_retained) / queries;
      metrics.query_read_hit_rate =
          static_cast<double>(total_query_read_hits) / queries;
    }
    if (total_retained_age_count != 0) {
      metrics.mean_retained_age = total_retained_age /
          static_cast<double>(total_retained_age_count);
    }
  }
  return metrics;
}

namespace {

TrainingHistory train_with_provider(
    SgwEsmModel& model,
    const AdamConfig& adam_config,
    const TrainingConfig& training_config,
    const std::function<BindingSample()>& next_sample) {
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
  history.routing_collision_rate.reserve(training_config.steps);
  history.routing_entropy.reserve(training_config.steps);
  history.routing_disagreement_rate.reserve(training_config.steps);
  history.retention_write_rate.reserve(training_config.steps);
  history.retention_skip_rate.reserve(training_config.steps);
  history.retention_eviction_rate.reserve(training_config.steps);
  history.relevant_eviction_rate.reserve(training_config.steps);
  history.queried_entity_retention_rate.reserve(training_config.steps);
  history.query_read_hit_rate.reserve(training_config.steps);

  for (std::size_t step = 0; step < training_config.steps; ++step) {
    model.set_key_value_routing_step(step);
    model.parameters().zero_grad();
    const double auxiliary_weight =
        workspace_aux_weight_at_step(training_config, step);
    double loss_sum = 0.0;
    double primary_loss_sum = 0.0;
    double auxiliary_loss_sum = 0.0;
    std::size_t routing_collisions = 0;
    double routing_entropy = 0.0;
    std::size_t routing_decisions = 0;
    std::size_t routing_disagreements = 0;
    std::size_t retention_decisions = 0;
    std::size_t retention_writes = 0;
    std::size_t retention_skips = 0;
    std::size_t retention_evictions = 0;
    std::size_t relevant_evictions = 0;
    std::size_t query_sequences = 0;
    std::size_t queried_retained = 0;
    std::size_t query_read_hits = 0;
    for (std::size_t batch_index = 0;
         batch_index < training_config.batch_size; ++batch_index) {
      const BindingSample sample = next_sample();
      ++history.samples_consumed;
      ad::Tape tape;
      const bool capture_routing =
          model.config().key_value_mode != KeyValueMediationMode::none;
      const SequenceResult result =
          model.forward_sequence(tape, sample.tokens, capture_routing);
      if (capture_routing) {
        for (const StepTrace& trace : result.traces) {
          routing_collisions += trace.write_collisions;
          routing_entropy += trace.routing_entropy;
          routing_decisions += trace.routing_decisions;
          routing_disagreements += trace.hard_soft_disagreements;
          retention_writes += trace.retention_writes;
          retention_skips += trace.retention_skips;
          retention_evictions += trace.retention_evictions;
          relevant_evictions += trace.relevant_evictions;
          retention_decisions +=
              trace.retention_writes + trace.retention_skips;
          if (trace.queried_entity_retained || !trace.read_slots.empty()) {
            ++query_sequences;
            if (trace.queried_entity_retained) ++queried_retained;
            if (trace.query_read_hit) ++query_read_hits;
          }
        }
      }
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
    history.workspace_aux_batch_loss.push_back(
        auxiliary_loss_sum * inverse_batch);
    history.workspace_aux_weight.push_back(auxiliary_weight);
    history.gradient_norm.push_back(gradient_norm);
    history.clip_scale.push_back(optimizer.last_clip_scale());
    if (routing_decisions == 0) {
      history.routing_collision_rate.push_back(0.0);
      history.routing_entropy.push_back(0.0);
      history.routing_disagreement_rate.push_back(0.0);
    } else {
      const double route_count = static_cast<double>(routing_decisions);
      history.routing_collision_rate.push_back(
          static_cast<double>(routing_collisions) / route_count);
      history.routing_entropy.push_back(routing_entropy / route_count);
      history.routing_disagreement_rate.push_back(
          static_cast<double>(routing_disagreements) / route_count);
    }
    if (retention_decisions == 0) {
      history.retention_write_rate.push_back(0.0);
      history.retention_skip_rate.push_back(0.0);
      history.retention_eviction_rate.push_back(0.0);
      history.relevant_eviction_rate.push_back(0.0);
    } else {
      const double decisions = static_cast<double>(retention_decisions);
      history.retention_write_rate.push_back(
          static_cast<double>(retention_writes) / decisions);
      history.retention_skip_rate.push_back(
          static_cast<double>(retention_skips) / decisions);
      history.retention_eviction_rate.push_back(
          static_cast<double>(retention_evictions) / decisions);
      history.relevant_eviction_rate.push_back(
          static_cast<double>(relevant_evictions) / decisions);
    }
    if (query_sequences == 0) {
      history.queried_entity_retention_rate.push_back(0.0);
      history.query_read_hit_rate.push_back(0.0);
    } else {
      const double queries = static_cast<double>(query_sequences);
      history.queried_entity_retention_rate.push_back(
          static_cast<double>(queried_retained) / queries);
      history.query_read_hit_rate.push_back(
          static_cast<double>(query_read_hits) / queries);
    }
  }
  return history;
}

}  // namespace

TrainingHistory train_steps(SgwEsmModel& model,
                            std::span<const BindingSample> samples,
                            const AdamConfig& adam_config,
                            const TrainingConfig& training_config) {
  if (samples.empty()) {
    throw std::invalid_argument("training samples must not be empty");
  }
  std::vector<std::size_t> order(samples.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::mt19937_64 generator(training_config.shuffle_seed);
  deterministic_shuffle(order, generator);
  std::size_t cursor = 0;
  const auto next_sample = [&]() {
    if (cursor == order.size()) {
      deterministic_shuffle(order, generator);
      cursor = 0;
    }
    return samples[order[cursor++]];
  };
  return train_with_provider(model, adam_config, training_config, next_sample);
}

TrainingHistory train_steps(SgwEsmModel& model,
                            RetentionBindingStream& stream,
                            const AdamConfig& adam_config,
                            const TrainingConfig& training_config) {
  const std::size_t requested = training_config.steps * training_config.batch_size;
  if (requested > stream.remaining()) {
    throw std::invalid_argument(
        "retention training stream does not contain enough unique samples");
  }
  const auto next_sample = [&]() { return stream.next(); };
  return train_with_provider(model, adam_config, training_config, next_sample);
}

TrainingHistory train_steps(SgwEsmModel& model,
                            StructuralBindingStream& stream,
                            const AdamConfig& adam_config,
                            const TrainingConfig& training_config) {
  const std::size_t requested = training_config.steps * training_config.batch_size;
  if (requested > stream.remaining()) {
    throw std::invalid_argument(
        "structural training stream does not contain enough unique samples");
  }
  const auto next_sample = [&]() { return stream.next(); };
  return train_with_provider(model, adam_config, training_config, next_sample);
}

}  // namespace sgw
