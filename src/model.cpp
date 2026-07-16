#include "sgw/model.hpp"

#include "sgw/topk.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace sgw {
namespace {

ad::Var parameter_var(ad::Tape& tape, Parameter& parameter,
                      std::size_t index) {
  return tape.parameter(parameter.value(index), &parameter.gradient(index));
}

ad::Var matrix_entry(ad::Tape& tape, Parameter& parameter, std::size_t row,
                     std::size_t column) {
  return parameter_var(tape, parameter, row * parameter.columns() + column);
}

std::vector<ad::Var> zero_vector(ad::Tape& tape, std::size_t size) {
  std::vector<ad::Var> values;
  values.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    values.push_back(tape.constant(0.0));
  }
  return values;
}

std::vector<double> snapshot(std::span<const ad::Var> values) {
  std::vector<double> result;
  result.reserve(values.size());
  for (const ad::Var value : values) {
    result.push_back(value.value());
  }
  return result;
}

ad::Var affine_row(ad::Tape& tape, Parameter& weights, std::size_t row,
                   std::span<const ad::Var> input, ad::Var initial) {
  if (weights.columns() != input.size() || row >= weights.rows()) {
    throw std::logic_error("affine row shape mismatch");
  }
  ad::Var sum = initial;
  for (std::size_t column = 0; column < input.size(); ++column) {
    sum = sum + matrix_entry(tape, weights, row, column) * input[column];
  }
  return sum;
}

ad::Var add_parameter_dot(ad::Tape& tape, ad::Var initial,
                          Parameter& weights, std::size_t row,
                          std::span<const ad::Var> input) {
  return affine_row(tape, weights, row, input, initial);
}

std::vector<ad::Var> pool_workspace(ad::Tape& tape,
                                    std::span<const ad::Var> workspace,
                                    std::size_t slot_count,
                                    std::size_t dimension) {
  if (workspace.size() != slot_count * dimension) {
    throw std::logic_error("workspace shape mismatch");
  }
  std::vector<ad::Var> pooled;
  pooled.reserve(dimension);
  const ad::Var denominator = tape.constant(static_cast<double>(slot_count));
  for (std::size_t component = 0; component < dimension; ++component) {
    ad::Var sum = tape.constant(0.0);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      sum = sum + workspace[slot * dimension + component];
    }
    pooled.push_back(sum / denominator);
  }
  return pooled;
}

std::vector<std::size_t> select_vars(std::span<const ad::Var> scores,
                                     std::size_t count) {
  std::vector<double> values;
  values.reserve(scores.size());
  for (const ad::Var score : scores) {
    values.push_back(score.value());
  }
  return stable_topk(values, count);
}

std::vector<ad::Var> selected_softmax_vars(
    ad::Tape& tape, std::span<const ad::Var> scores,
    std::span<const std::size_t> selected) {
  if (selected.empty()) {
    throw std::invalid_argument("selected route must not be empty");
  }
  double maximum = scores[selected.front()].value();
  for (const std::size_t index : selected) {
    maximum = std::max(maximum, scores[index].value());
  }
  const ad::Var maximum_constant = tape.constant(maximum);
  std::vector<ad::Var> numerators;
  numerators.reserve(selected.size());
  ad::Var denominator = tape.constant(0.0);
  for (const std::size_t index : selected) {
    const ad::Var numerator = ad::exp(scores[index] - maximum_constant);
    numerators.push_back(numerator);
    denominator = denominator + numerator;
  }
  std::vector<ad::Var> weights;
  weights.reserve(selected.size());
  for (const ad::Var numerator : numerators) {
    weights.push_back(numerator / denominator);
  }
  return weights;
}

}  // namespace

std::string_view forward_intervention_name(
    ForwardIntervention intervention) noexcept {
  switch (intervention) {
    case ForwardIntervention::intact:
      return "intact";
    case ForwardIntervention::no_broadcast:
      return "no_broadcast";
    case ForwardIntervention::no_workspace_persistence:
      return "no_workspace_persistence";
    case ForwardIntervention::no_workspace_output:
      return "no_workspace_output";
    case ForwardIntervention::no_spine_workspace:
      return "no_spine_workspace";
    case ForwardIntervention::no_mechanism_output:
      return "no_mechanism_output";
    case ForwardIntervention::workspace_disconnected:
      return "workspace_disconnected";
    case ForwardIntervention::no_workspace_writes:
      return "no_workspace_writes";
    case ForwardIntervention::zero_reader_inbox:
      return "zero_reader_inbox";
    case ForwardIntervention::permuted_recipients:
      return "permuted_recipients";
    case ForwardIntervention::permuted_workspace_keys:
      return "permuted_workspace_keys";
    case ForwardIntervention::zero_query_key:
      return "zero_query_key";
    case ForwardIntervention::randomized_write_slots:
      return "randomized_write_slots";
    case ForwardIntervention::cleared_writer_assignment:
      return "cleared_writer_assignment";
    case ForwardIntervention::allow_write_collisions:
      return "allow_write_collisions";
  }
  return "unknown";
}

bool StepTrace::same_route(const StepTrace& other) const noexcept {
  return active_mechanisms == other.active_mechanisms &&
         writers == other.writers && writer_slots == other.writer_slots &&
         recipients == other.recipients;
}

std::size_t estimated_step_madds(const ModelConfig& config) {
  config.validate();
  const std::size_t e = config.embedding_dim;
  const std::size_t s = config.spine_dim;
  const std::size_t m = config.mechanism_count;
  const std::size_t d = config.mechanism_dim;
  const std::size_t b = config.workspace_slots;
  const std::size_t w = config.workspace_dim;
  const std::size_t k = config.active_mechanisms;
  const std::size_t q = config.workspace_writers;
  const std::size_t c = config.output_classes;

  if (config.key_value_mode == KeyValueMediationMode::symbolic_exact) {
    return c;
  }
  if (config.key_value_mode == KeyValueMediationMode::learned_tied) {
    std::size_t total = b * config.key_dim + c * config.value_dim;
    if (config.key_value_write_routing ==
            KeyValueWriteRoutingMode::hard_learned ||
        config.key_value_write_routing ==
            KeyValueWriteRoutingMode::annealed_learned) {
      total += b * config.key_dim;
    }
    return total;
  }

  std::size_t total = s * s;
  if (config.spine_reads_embedding) {
    total += s * e;
  }
  if (config.core_only) {
    if (config.output_reads_spine) {
      total += c * s;
    }
    return total;
  }
  if (config.spine_reads_workspace) {
    total += s * w;
  }
  if (!config.fixed_binding_mediation) {
    total += m * (e + s + w + d);
  }
  total += k * d * (e + s + d + w);
  total += k * w * d;
  if (!config.fixed_binding_mediation) {
    total += k * w;
    total += q * b * w;
  }
  total += q * (2 * w * w + 2 * w);
  if (!config.fixed_binding_mediation) {
    total += m * w;
  }
  total += w * w;
  if (config.output_reads_spine) {
    total += c * s;
  }
  if (config.output_reads_workspace) {
    total += c * w;
  }
  if (config.output_reads_mechanism) {
    total += c * d;
  }
  return total;
}

SgwEsmModel::SgwEsmModel(ModelConfig config, std::uint64_t seed)
    : config_(config) {
  config_.validate();
  std::mt19937_64 generator(seed);
  const std::size_t e = config_.embedding_dim;
  const std::size_t s = config_.spine_dim;
  const std::size_t m = config_.mechanism_count;
  const std::size_t d = config_.mechanism_dim;
  const std::size_t b = config_.workspace_slots;
  const std::size_t w = config_.workspace_dim;
  const std::size_t c = config_.output_classes;

  if (config_.key_value_mode != KeyValueMediationMode::none) {
    if (config_.key_value_mode == KeyValueMediationMode::learned_tied) {
      kv_entity_codebook_ = &parameters_.add_xavier(
          "kv_entity_codebook", config_.entity_count, config_.key_dim,
          generator);
      kv_value_codebook_ = &parameters_.add_xavier(
          "kv_value_codebook", config_.value_count, config_.value_dim,
          generator);
      if (config_.key_value_write_routing ==
              KeyValueWriteRoutingMode::hard_learned ||
          config_.key_value_write_routing ==
              KeyValueWriteRoutingMode::annealed_learned) {
        kv_slot_codebook_ = &parameters_.add_xavier(
            "kv_slot_codebook", config_.workspace_slots, config_.key_dim,
            generator);
      }
    }
    return;
  }

  if (config_.spine_reads_embedding || !config_.core_only) {
    embedding_ =
        &parameters_.add_xavier("embedding", config_.vocab_size, e, generator);
  }
  if (config_.spine_reads_embedding ||
      (!config_.core_only && !config_.fixed_binding_mediation)) {
    spine_input_ = &parameters_.add_xavier("spine_input", s, e, generator);
  }
  spine_recurrent_ =
      &parameters_.add_xavier("spine_recurrent", s, s, generator);
  spine_bias_ = &parameters_.add_zeros("spine_bias", s, 1);
  if (config_.output_reads_spine ||
      (!config_.core_only && !config_.fixed_binding_mediation)) {
    output_spine_ = &parameters_.add_xavier("output_spine", c, s, generator);
  }
  output_bias_ = &parameters_.add_zeros("output_bias", c, 1);

  if (!config_.core_only) {
    spine_workspace_ =
        &parameters_.add_xavier("spine_workspace", s, w, generator);

    if (!config_.fixed_binding_mediation) {
      router_embedding_ =
          &parameters_.add_xavier("router_embedding", m, e, generator);
      router_spine_ =
          &parameters_.add_xavier("router_spine", m, s, generator);
      router_workspace_ =
          &parameters_.add_xavier("router_workspace", m, w, generator);
      router_state_ =
          &parameters_.add_xavier("router_state", m, d, generator);
      router_bias_ = &parameters_.add_zeros("router_bias", m, 1);
    }

    mechanism_embedding_ =
        &parameters_.add_xavier("mechanism_embedding", m * d, e, generator);
    mechanism_spine_ =
        &parameters_.add_xavier("mechanism_spine", m * d, s, generator);
    mechanism_recurrent_ =
        &parameters_.add_xavier("mechanism_recurrent", m * d, d, generator);
    mechanism_inbox_ =
        &parameters_.add_xavier("mechanism_inbox", m * d, w, generator);
    mechanism_bias_ = &parameters_.add_zeros("mechanism_bias", m, d);

    message_state_ =
        &parameters_.add_xavier("message_state", m * w, d, generator);
    message_bias_ = &parameters_.add_zeros("message_bias", m, w);
    if (!config_.fixed_binding_mediation) {
      writer_key_ = &parameters_.add_xavier("writer_key", m, w, generator);
      writer_bias_ = &parameters_.add_zeros("writer_bias", m, 1);
      slot_key_ = &parameters_.add_xavier("slot_key", b, w, generator);
    }

    workspace_message_ =
        &parameters_.add_xavier("workspace_message", w, w, generator);
    workspace_recurrent_ =
        &parameters_.add_xavier("workspace_recurrent", w, w, generator);
    workspace_bias_ = &parameters_.add_zeros("workspace_bias", 1, w);
    workspace_gate_message_ = &parameters_.add_xavier(
        "workspace_gate_message", 1, w, generator);
    workspace_gate_slot_ =
        &parameters_.add_xavier("workspace_gate_slot", 1, w, generator);
    workspace_gate_bias_ =
        &parameters_.add_zeros("workspace_gate_bias", 1, 1);

    if (!config_.fixed_binding_mediation) {
      recipient_key_ =
          &parameters_.add_xavier("recipient_key", m, w, generator);
      recipient_bias_ = &parameters_.add_zeros("recipient_bias", m, w);
    }
    broadcast_projection_ = &parameters_.add_xavier(
        "broadcast_projection", w, w, generator);
    broadcast_bias_ = &parameters_.add_zeros("broadcast_bias", 1, w);

    output_workspace_ =
        &parameters_.add_xavier("output_workspace", c, w, generator);
    if (config_.output_reads_mechanism) {
      output_mechanism_ =
          &parameters_.add_xavier("output_mechanism", c, d, generator);
    }
  }

  if (!parameters_.all_finite()) {
    throw std::runtime_error("model initialization produced non-finite values");
  }
}

const ModelConfig& SgwEsmModel::config() const noexcept { return config_; }
ParameterSet& SgwEsmModel::parameters() noexcept { return parameters_; }
const ParameterSet& SgwEsmModel::parameters() const noexcept {
  return parameters_;
}

void SgwEsmModel::set_key_value_routing_step(std::size_t step) noexcept {
  key_value_routing_step_ = step;
}

std::size_t SgwEsmModel::key_value_routing_step() const noexcept {
  return key_value_routing_step_;
}

double SgwEsmModel::key_value_router_temperature() const noexcept {
  if (config_.key_value_write_routing !=
          KeyValueWriteRoutingMode::annealed_learned ||
      key_value_routing_step_ >= config_.key_value_router_anneal_steps) {
    return config_.key_value_router_final_temperature;
  }
  if (config_.key_value_router_anneal_steps <= 1) {
    return config_.key_value_router_final_temperature;
  }
  const double fraction = static_cast<double>(key_value_routing_step_) /
      static_cast<double>(config_.key_value_router_anneal_steps - 1);
  return config_.key_value_router_initial_temperature +
      fraction * (config_.key_value_router_final_temperature -
                  config_.key_value_router_initial_temperature);
}

bool SgwEsmModel::key_value_router_uses_surrogate() const noexcept {
  return config_.key_value_write_routing ==
             KeyValueWriteRoutingMode::annealed_learned &&
         key_value_routing_step_ < config_.key_value_router_anneal_steps;
}

SequenceResult SgwEsmModel::forward_key_value_sequence(
    ad::Tape& tape, std::span<const int> tokens, bool capture_trace,
    ForwardIntervention intervention) {
  const std::size_t bindings = config_.mediation_binding_count;
  const std::size_t expected_length = 2 * bindings + 2;
  if (tokens.size() != expected_length) {
    throw std::invalid_argument(
        "key-value mediation requires the configured binding sequence length");
  }
  for (const int token : tokens) {
    if (token < 0 || static_cast<std::size_t>(token) >= config_.vocab_size) {
      throw std::out_of_range("input token out of vocabulary range");
    }
  }

  const std::size_t b = config_.workspace_slots;
  const std::size_t w = config_.workspace_dim;
  const std::size_t key_dim = config_.key_dim;
  const std::size_t value_dim = config_.value_dim;
  const std::size_t reader = bindings;
  const std::size_t missing = config_.vocab_size;
  const bool writes_enabled =
      intervention != ForwardIntervention::no_workspace_writes;
  const bool persists =
      intervention != ForwardIntervention::no_workspace_persistence;
  const bool output_enabled =
      intervention != ForwardIntervention::no_broadcast &&
      intervention != ForwardIntervention::zero_reader_inbox &&
      intervention != ForwardIntervention::workspace_disconnected &&
      intervention != ForwardIntervention::no_workspace_output &&
      intervention != ForwardIntervention::no_mechanism_output;

  ModelState state;
  state.spine = zero_vector(tape, config_.spine_dim);
  state.mechanisms =
      zero_vector(tape, config_.mechanism_count * config_.mechanism_dim);
  state.workspace = zero_vector(tape, b * w);
  state.inboxes = zero_vector(tape, config_.mechanism_count * w);
  state.active_summary = zero_vector(tape, value_dim);
  std::vector<std::size_t> slot_entities(b, missing);
  std::vector<std::size_t> slot_values(b, missing);
  std::vector<bool> occupied(b, false);
  std::vector<std::size_t> binding_slots(bindings, missing);
  std::vector<std::vector<ad::Var>> binding_gates(bindings);
  std::vector<StepTrace> traces;
  if (capture_trace) traces.reserve(tokens.size());

  const auto normalize = [&](std::span<const ad::Var> input) {
    ad::Var sum_squares = tape.constant(1.0e-12);
    for (const ad::Var value : input) {
      sum_squares = sum_squares + value * value;
    }
    const ad::Var norm = ad::exp(tape.constant(0.5) * ad::log(sum_squares));
    std::vector<ad::Var> result;
    result.reserve(input.size());
    for (const ad::Var value : input) result.push_back(value / norm);
    return result;
  };
  const auto parameter_row = [&](Parameter& parameter, std::size_t row) {
    std::vector<ad::Var> values;
    values.reserve(parameter.columns());
    for (std::size_t column = 0; column < parameter.columns(); ++column) {
      values.push_back(matrix_entry(tape, parameter, row, column));
    }
    return values;
  };
  const auto dot = [&](std::span<const ad::Var> lhs,
                       std::span<const ad::Var> rhs) {
    if (lhs.size() != rhs.size()) {
      throw std::logic_error("key-value dot shape mismatch");
    }
    ad::Var result = tape.constant(0.0);
    for (std::size_t index = 0; index < lhs.size(); ++index) {
      result = result + lhs[index] * rhs[index];
    }
    return result;
  };
  const auto hard_gates = [&](std::size_t selected) {
    std::vector<ad::Var> gates;
    gates.reserve(b);
    for (std::size_t slot = 0; slot < b; ++slot) {
      gates.push_back(tape.constant(slot == selected ? 1.0 : 0.0));
    }
    return gates;
  };
  struct RouteDecision {
    std::size_t selected{0};
    std::vector<ad::Var> gates;
    std::size_t collisions{0};
    double entropy{0.0};
    std::size_t disagreements{0};
  };
  const auto choose_write_route = [&](std::size_t binding,
                                      std::size_t entity_token) {
    RouteDecision decision;
    decision.selected = binding;
    if (config_.key_value_write_routing ==
        KeyValueWriteRoutingMode::first_free) {
      const auto found = std::find(occupied.begin(), occupied.end(), false);
      if (found == occupied.end()) {
        throw std::logic_error("key-value workspace has no free write slot");
      }
      decision.selected =
          static_cast<std::size_t>(found - occupied.begin());
      decision.gates = hard_gates(decision.selected);
    } else if (config_.key_value_write_routing ==
                   KeyValueWriteRoutingMode::hard_learned ||
               config_.key_value_write_routing ==
                   KeyValueWriteRoutingMode::annealed_learned) {
      const auto entity_key = normalize(
          parameter_row(*kv_entity_codebook_, entity_token));
      std::vector<std::size_t> available;
      std::vector<ad::Var> scores;
      for (std::size_t slot = 0; slot < b; ++slot) {
        if (occupied[slot] &&
            intervention != ForwardIntervention::allow_write_collisions) {
          continue;
        }
        available.push_back(slot);
        const auto slot_code = normalize(parameter_row(*kv_slot_codebook_, slot));
        scores.push_back(dot(entity_key, slot_code));
      }
      if (available.empty()) {
        throw std::logic_error("key-value workspace has no available route");
      }
      const std::size_t local = select_vars(scores, 1).front();
      decision.selected = available[local];
      decision.gates = hard_gates(decision.selected);

      const double temperature = key_value_router_temperature();
      double maximum = scores.front().value() / temperature;
      for (const ad::Var score : scores) {
        maximum = std::max(maximum, score.value() / temperature);
      }
      const ad::Var maximum_constant = tape.constant(maximum);
      std::vector<ad::Var> probabilities;
      ad::Var denominator = tape.constant(0.0);
      for (const ad::Var score : scores) {
        const ad::Var numerator = ad::exp(
            score / tape.constant(temperature) - maximum_constant);
        probabilities.push_back(numerator);
        denominator = denominator + numerator;
      }
      for (ad::Var& probability : probabilities) {
        probability = probability / denominator;
        const double numeric = probability.value();
        if (numeric > 0.0) decision.entropy -= numeric * std::log(numeric);
      }
      if (key_value_router_uses_surrogate()) {
        for (std::size_t index = 0; index < available.size(); ++index) {
          const ad::Var probability = probabilities[index];
          const std::size_t slot = available[index];
          decision.gates[slot] = decision.gates[slot] + probability -
                                 tape.constant(probability.value());
        }
      }
    } else {
      decision.gates = hard_gates(decision.selected);
    }

    if (intervention == ForwardIntervention::randomized_write_slots) {
      decision.selected = (decision.selected + 1) % b;
      decision.gates = hard_gates(decision.selected);
    } else if (intervention ==
               ForwardIntervention::allow_write_collisions) {
      decision.selected = 0;
      decision.gates = hard_gates(decision.selected);
    }
    if (decision.selected >= b) {
      throw std::logic_error("key-value write route is out of range");
    }
    if (occupied[decision.selected]) {
      decision.collisions = 1;
      if (intervention != ForwardIntervention::allow_write_collisions &&
          intervention != ForwardIntervention::randomized_write_slots) {
        throw std::logic_error("key-value write route collided");
      }
    }
    occupied[decision.selected] = true;
    return decision;
  };

  for (std::size_t step = 0; step < tokens.size(); ++step) {
    if (!persists) {
      state.workspace = zero_vector(tape, b * w);
      std::fill(slot_entities.begin(), slot_entities.end(), missing);
      std::fill(slot_values.begin(), slot_values.end(), missing);
      std::fill(occupied.begin(), occupied.end(), false);
    }
    StepTrace trace;
    if (capture_trace) {
      trace.mechanism_state_before = snapshot(state.mechanisms);
      trace.inbox_before = snapshot(state.inboxes);
      trace.workspace_before = snapshot(state.workspace);
      trace.estimated_madds = estimated_step_madds(config_);
    }
    const std::size_t mechanism = step < 2 * bindings ? step / 2 : reader;
    trace.active_mechanisms.push_back(mechanism);
    trace.recipients.push_back(reader);

    if (step < 2 * bindings) {
      const std::size_t binding = step / 2;
      const std::size_t token = static_cast<std::size_t>(tokens[step]);
      if (step % 2 == 0) {
        if (token >= config_.entity_count) {
          throw std::invalid_argument("key-value entity token is out of range");
        }
        RouteDecision decision = choose_write_route(binding, token);
        const std::size_t slot = decision.selected;
        binding_slots[binding] = slot;
        binding_gates[binding] = std::move(decision.gates);
        trace.writers.push_back(mechanism);
        trace.writer_slots.push_back(slot);
        trace.write_collisions += decision.collisions;
        trace.routing_entropy += decision.entropy;
        trace.routing_decisions += 1;
        trace.hard_soft_disagreements += decision.disagreements;
        if (writes_enabled) {
          slot_entities[slot] = token;
          std::vector<ad::Var> key;
          if (config_.key_value_mode == KeyValueMediationMode::symbolic_exact) {
            key.reserve(key_dim);
            for (std::size_t component = 0; component < key_dim; ++component) {
              key.push_back(tape.constant(component == token ? 1.0 : 0.0));
            }
          } else {
            key = normalize(parameter_row(*kv_entity_codebook_, token));
          }
          for (std::size_t target = 0; target < b; ++target) {
            for (std::size_t component = 0; component < key_dim; ++component) {
              const std::size_t offset = target * w + component;
              state.workspace[offset] = state.workspace[offset] +
                  binding_gates[binding][target] * key[component];
            }
          }
        }
      } else {
        std::size_t slot = binding_slots[binding];
        if (intervention == ForwardIntervention::cleared_writer_assignment) {
          slot = missing;
        } else if (intervention ==
                   ForwardIntervention::randomized_write_slots) {
          slot = (slot + 1) % b;
          binding_gates[binding] = hard_gates(slot);
        }
        if (slot != missing) {
          trace.writers.push_back(mechanism);
          trace.writer_slots.push_back(slot);
        }
        if (token < config_.entity_count ||
            token >= config_.entity_count + config_.value_count) {
          throw std::invalid_argument("key-value value token is out of range");
        }
        const std::size_t value_class = token - config_.entity_count;
        if (writes_enabled && slot != missing) {
          slot_values[slot] = value_class;
          std::vector<ad::Var> value;
          if (config_.key_value_mode == KeyValueMediationMode::symbolic_exact) {
            value.reserve(value_dim);
            for (std::size_t component = 0; component < value_dim; ++component) {
              value.push_back(tape.constant(
                  component == value_class ? 1.0 : 0.0));
            }
          } else {
            value = normalize(parameter_row(*kv_value_codebook_, value_class));
          }
          for (std::size_t target = 0; target < b; ++target) {
            for (std::size_t component = 0; component < value_dim; ++component) {
              const std::size_t offset = target * w + key_dim + component;
              state.workspace[offset] = state.workspace[offset] +
                  binding_gates[binding][target] * value[component];
            }
          }
        }
      }
    }

    if (step + 1 == tokens.size()) {
      const std::size_t query = static_cast<std::size_t>(tokens[step]);
      if (query >= config_.entity_count) {
        throw std::invalid_argument("key-value query token is out of range");
      }
      std::size_t selected_slot = 0;
      if (config_.key_value_mode == KeyValueMediationMode::symbolic_exact) {
        for (std::size_t slot = 0; slot < b; ++slot) {
          const std::size_t key_slot =
              intervention == ForwardIntervention::permuted_workspace_keys
                  ? (slot + 1) % b
                  : slot;
          if (slot_entities[key_slot] == query) {
            selected_slot = slot;
            break;
          }
        }
      } else {
        std::vector<ad::Var> query_key;
        if (intervention == ForwardIntervention::zero_query_key) {
          query_key = zero_vector(tape, key_dim);
        } else {
          query_key = normalize(parameter_row(*kv_entity_codebook_, query));
        }
        std::vector<ad::Var> scores;
        scores.reserve(b);
        for (std::size_t slot = 0; slot < b; ++slot) {
          const std::size_t key_slot =
              intervention == ForwardIntervention::permuted_workspace_keys
                  ? (slot + 1) % b
                  : slot;
          std::vector<ad::Var> slot_key;
          slot_key.reserve(key_dim);
          for (std::size_t component = 0; component < key_dim; ++component) {
            slot_key.push_back(state.workspace[key_slot * w + component]);
          }
          scores.push_back(dot(query_key, slot_key));
        }
        selected_slot = select_vars(scores, 1).front();
      }
      trace.read_slots.push_back(selected_slot);
      std::vector<ad::Var> retrieved;
      retrieved.reserve(value_dim);
      for (std::size_t component = 0; component < value_dim; ++component) {
        retrieved.push_back(output_enabled
            ? state.workspace[selected_slot * w + key_dim + component]
            : tape.constant(0.0));
      }
      state.active_summary = retrieved;
    }

    if (capture_trace) {
      trace.mechanism_state_after = snapshot(state.mechanisms);
      trace.inbox_after = snapshot(state.inboxes);
      trace.workspace_after = snapshot(state.workspace);
      traces.push_back(std::move(trace));
    }
  }

  std::vector<ad::Var> logits;
  logits.reserve(config_.output_classes);
  if (config_.key_value_mode == KeyValueMediationMode::symbolic_exact) {
    std::size_t selected_value = missing;
    const std::size_t query = static_cast<std::size_t>(tokens.back());
    std::size_t selected_slot = 0;
    for (std::size_t slot = 0; slot < b; ++slot) {
      const std::size_t key_slot =
          intervention == ForwardIntervention::permuted_workspace_keys
              ? (slot + 1) % b
              : slot;
      if (slot_entities[key_slot] == query) {
        selected_slot = slot;
        break;
      }
    }
    if (output_enabled) selected_value = slot_values[selected_slot];
    for (std::size_t output = 0; output < config_.output_classes; ++output) {
      logits.push_back(tape.constant(
          output == selected_value ? config_.key_value_logit_scale : 0.0));
    }
  } else {
    const auto retrieved = normalize(state.active_summary);
    for (std::size_t output = 0; output < config_.output_classes; ++output) {
      const auto code = normalize(parameter_row(*kv_value_codebook_, output));
      logits.push_back(tape.constant(config_.key_value_logit_scale) *
                       dot(retrieved, code));
    }
  }
  return SequenceResult{std::move(logits), {}, std::move(state),
                        std::move(traces)};
}

SequenceResult SgwEsmModel::forward_sequence(
    ad::Tape& tape, std::span<const int> tokens, bool capture_trace,
    ForwardIntervention intervention) {
  if (tokens.empty()) {
    throw std::invalid_argument("input sequence must not be empty");
  }
  if (config_.key_value_mode != KeyValueMediationMode::none) {
    return forward_key_value_sequence(tape, tokens, capture_trace, intervention);
  }
  if (config_.fixed_binding_mediation &&
      tokens.size() != 2 * config_.mediation_binding_count + 2) {
    throw std::invalid_argument(
        "fixed_binding_mediation requires the configured binding sequence length");
  }
  for (const int token : tokens) {
    if (token < 0 || static_cast<std::size_t>(token) >= config_.vocab_size) {
      throw std::out_of_range("input token out of vocabulary range");
    }
  }

  const std::size_t e = config_.embedding_dim;
  const std::size_t s = config_.spine_dim;
  const std::size_t m = config_.mechanism_count;
  const std::size_t d = config_.mechanism_dim;
  const std::size_t b = config_.workspace_slots;
  const std::size_t w = config_.workspace_dim;

  const bool workspace_persists =
      intervention != ForwardIntervention::no_workspace_persistence;
  const bool workspace_writes_enabled =
      intervention != ForwardIntervention::no_workspace_writes;
  const bool broadcast_enabled =
      intervention != ForwardIntervention::no_broadcast &&
      intervention != ForwardIntervention::workspace_disconnected;
  const bool spine_workspace_enabled =
      config_.spine_reads_workspace &&
      intervention != ForwardIntervention::no_spine_workspace &&
      intervention != ForwardIntervention::workspace_disconnected;
  const bool workspace_output_enabled =
      config_.output_reads_workspace &&
      intervention != ForwardIntervention::no_workspace_output &&
      intervention != ForwardIntervention::workspace_disconnected;
  const bool mechanism_output_enabled =
      config_.output_reads_mechanism &&
      intervention != ForwardIntervention::no_mechanism_output;

  ModelState state;
  state.spine = zero_vector(tape, s);
  state.mechanisms = zero_vector(tape, m * d);
  state.workspace = zero_vector(tape, b * w);
  state.inboxes = zero_vector(tape, m * w);
  state.active_summary = zero_vector(tape, d);

  std::vector<StepTrace> traces;
  if (capture_trace) {
    traces.reserve(tokens.size());
  }

  for (std::size_t step = 0; step < tokens.size(); ++step) {
    const int token = tokens[step];
    if (!config_.core_only && !workspace_persists) {
      state.workspace = zero_vector(tape, b * w);
    }
    StepTrace trace;
    if (capture_trace) {
      trace.mechanism_state_before = snapshot(state.mechanisms);
      trace.inbox_before = snapshot(state.inboxes);
      trace.workspace_before = snapshot(state.workspace);
      trace.estimated_madds = estimated_step_madds(config_);
    }

    std::vector<ad::Var> embedding;
    embedding.reserve(e);
    const std::size_t token_index = static_cast<std::size_t>(token);
    if (embedding_ != nullptr) {
      for (std::size_t component = 0; component < e; ++component) {
        embedding.push_back(matrix_entry(tape, *embedding_, token_index,
                                         component));
      }
    } else {
      embedding = zero_vector(tape, e);
    }

    const std::vector<ad::Var> old_workspace_pool =
        pool_workspace(tape, state.workspace, b, w);
    std::vector<ad::Var> new_spine;
    new_spine.reserve(s);
    for (std::size_t component = 0; component < s; ++component) {
      ad::Var sum = parameter_var(tape, *spine_bias_, component);
      if (config_.spine_reads_embedding) {
        sum = add_parameter_dot(tape, sum, *spine_input_, component,
                                embedding);
      }
      sum = add_parameter_dot(tape, sum, *spine_recurrent_, component,
                              state.spine);
      if (!config_.core_only && spine_workspace_enabled) {
        sum = add_parameter_dot(tape, sum, *spine_workspace_, component,
                                old_workspace_pool);
      }
      new_spine.push_back(ad::tanh(sum));
    }
    state.spine = std::move(new_spine);

    if (!config_.core_only) {
      std::vector<std::size_t> active;
      std::vector<ad::Var> route_weights;
      if (config_.fixed_binding_mediation) {
        const std::size_t writer_token_count =
            2 * config_.mediation_binding_count;
        const std::size_t mechanism =
            step < writer_token_count ? step / 2
                                      : config_.mediation_binding_count;
        active.push_back(mechanism);
        route_weights.push_back(tape.constant(1.0));
      } else {
        std::vector<ad::Var> router_scores;
        router_scores.reserve(m);
        for (std::size_t mechanism = 0; mechanism < m; ++mechanism) {
          ad::Var score = parameter_var(tape, *router_bias_, mechanism);
          score = add_parameter_dot(tape, score, *router_embedding_, mechanism,
                                    embedding);
          score = add_parameter_dot(tape, score, *router_spine_, mechanism,
                                    state.spine);
          score = add_parameter_dot(tape, score, *router_workspace_, mechanism,
                                    old_workspace_pool);
          const std::span<const ad::Var> mechanism_state(
              state.mechanisms.data() + mechanism * d, d);
          score = add_parameter_dot(tape, score, *router_state_, mechanism,
                                    mechanism_state);
          router_scores.push_back(score);
        }
        active = select_vars(router_scores, config_.active_mechanisms);
        route_weights = selected_softmax_vars(tape, router_scores, active);
      }
      if (capture_trace) {
        trace.active_mechanisms = active;
      }

      std::vector<ad::Var> new_mechanisms = state.mechanisms;
      std::vector<std::vector<ad::Var>> messages;
      std::vector<ad::Var> utilities;
      messages.reserve(active.size());
      utilities.reserve(active.size());

      for (const std::size_t mechanism : active) {
        std::vector<ad::Var> updated;
        updated.reserve(d);
        const std::span<const ad::Var> old_mechanism(
            state.mechanisms.data() + mechanism * d, d);
        std::vector<ad::Var> zero_inbox;
        const bool reader_inbox_zeroed =
            intervention == ForwardIntervention::zero_reader_inbox &&
            config_.fixed_binding_mediation &&
            mechanism == config_.mediation_binding_count;
        std::span<const ad::Var> inbox;
        if (reader_inbox_zeroed) {
          zero_inbox = zero_vector(tape, w);
          inbox = std::span<const ad::Var>(zero_inbox);
        } else {
          inbox = std::span<const ad::Var>(
              state.inboxes.data() + mechanism * w, w);
        }
        for (std::size_t component = 0; component < d; ++component) {
          const std::size_t row = mechanism * d + component;
          ad::Var sum = parameter_var(
              tape, *mechanism_bias_, mechanism * d + component);
          sum = add_parameter_dot(tape, sum, *mechanism_embedding_, row,
                                  embedding);
          sum = add_parameter_dot(tape, sum, *mechanism_spine_, row,
                                  state.spine);
          sum = add_parameter_dot(tape, sum, *mechanism_recurrent_, row,
                                  old_mechanism);
          sum = add_parameter_dot(tape, sum, *mechanism_inbox_, row, inbox);
          updated.push_back(ad::tanh(sum));
        }
        for (std::size_t component = 0; component < d; ++component) {
          new_mechanisms[mechanism * d + component] = updated[component];
        }

        std::vector<ad::Var> message;
        message.reserve(w);
        for (std::size_t component = 0; component < w; ++component) {
          const std::size_t row = mechanism * w + component;
          ad::Var sum = parameter_var(
              tape, *message_bias_, mechanism * w + component);
          sum = add_parameter_dot(tape, sum, *message_state_, row, updated);
          message.push_back(ad::tanh(sum));
        }
        messages.push_back(std::move(message));
        if (!config_.fixed_binding_mediation) {
          ad::Var utility = parameter_var(tape, *writer_bias_, mechanism);
          utility = add_parameter_dot(tape, utility, *writer_key_, mechanism,
                                      messages.back());
          utilities.push_back(utility);
        }
      }

      std::vector<std::size_t> writer_local;
      if (config_.fixed_binding_mediation) {
        if (step < 2 * config_.mediation_binding_count) {
          writer_local.push_back(0);
        }
      } else {
        writer_local = select_vars(utilities, config_.workspace_writers);
      }
      std::vector<ad::Var> new_workspace = state.workspace;
      if (capture_trace) {
        trace.writers.reserve(writer_local.size());
        trace.writer_slots.reserve(writer_local.size());
      }
      for (const std::size_t local_index : writer_local) {
        const std::size_t mechanism = active[local_index];
        const std::vector<ad::Var>& message = messages[local_index];
        std::size_t slot = mechanism;
        if (!config_.fixed_binding_mediation) {
          std::vector<ad::Var> slot_scores;
          slot_scores.reserve(b);
          for (std::size_t candidate_slot = 0; candidate_slot < b;
               ++candidate_slot) {
            ad::Var score = tape.constant(0.0);
            for (std::size_t component = 0; component < w; ++component) {
              const ad::Var slot_content =
                  new_workspace[candidate_slot * w + component];
              const ad::Var key = matrix_entry(tape, *slot_key_,
                                               candidate_slot, component);
              score = score + message[component] * (slot_content + key);
            }
            slot_scores.push_back(score);
          }
          slot = select_vars(slot_scores, 1).front();
        }
        if (capture_trace) {
          trace.writers.push_back(mechanism);
          trace.writer_slots.push_back(slot);
        }
        if (!workspace_writes_enabled) {
          continue;
        }

        std::vector<ad::Var> old_slot;
        old_slot.reserve(w);
        for (std::size_t component = 0; component < w; ++component) {
          old_slot.push_back(new_workspace[slot * w + component]);
        }
        std::vector<ad::Var> candidate;
        candidate.reserve(w);
        for (std::size_t component = 0; component < w; ++component) {
          ad::Var sum = parameter_var(tape, *workspace_bias_, component);
          sum = add_parameter_dot(tape, sum, *workspace_message_, component,
                                  message);
          sum = add_parameter_dot(tape, sum, *workspace_recurrent_, component,
                                  old_slot);
          candidate.push_back(ad::tanh(sum));
        }
        ad::Var gate_score =
            parameter_var(tape, *workspace_gate_bias_, 0);
        gate_score = add_parameter_dot(tape, gate_score,
                                       *workspace_gate_message_, 0, message);
        gate_score = add_parameter_dot(tape, gate_score,
                                       *workspace_gate_slot_, 0, old_slot);
        const ad::Var gate = ad::sigmoid(gate_score);
        const ad::Var one = tape.constant(1.0);
        for (std::size_t component = 0; component < w; ++component) {
          new_workspace[slot * w + component] =
              gate * candidate[component] + (one - gate) * old_slot[component];
        }
      }

      const std::vector<ad::Var> new_workspace_pool =
          pool_workspace(tape, new_workspace, b, w);
      std::vector<std::size_t> recipients;
      if (config_.fixed_binding_mediation) {
        recipients.push_back(config_.mediation_binding_count);
      } else {
        std::vector<ad::Var> recipient_scores;
        recipient_scores.reserve(m);
        for (std::size_t mechanism = 0; mechanism < m; ++mechanism) {
          ad::Var score = parameter_var(
              tape, *recipient_bias_, mechanism * w);
          score = add_parameter_dot(tape, score, *recipient_key_, mechanism,
                                    new_workspace_pool);
          recipient_scores.push_back(score);
        }
        recipients =
            select_vars(recipient_scores, config_.broadcast_recipients);
      }
      if (intervention == ForwardIntervention::permuted_recipients) {
        for (std::size_t& recipient : recipients) {
          recipient = (recipient + 1) % m;
        }
      }
      if (capture_trace) {
        trace.recipients = recipients;
      }

      std::vector<ad::Var> broadcast_base;
      broadcast_base.reserve(w);
      for (std::size_t component = 0; component < w; ++component) {
        ad::Var sum = parameter_var(tape, *broadcast_bias_, component);
        sum = add_parameter_dot(tape, sum, *broadcast_projection_, component,
                                new_workspace_pool);
        broadcast_base.push_back(sum);
      }
      std::vector<ad::Var> new_inboxes = state.inboxes;
      if (broadcast_enabled) {
        for (const std::size_t recipient : recipients) {
          for (std::size_t component = 0; component < w; ++component) {
            ad::Var value = broadcast_base[component];
            if (!config_.fixed_binding_mediation) {
              value = value + parameter_var(
                                  tape, *recipient_bias_,
                                  recipient * w + component);
            }
            new_inboxes[recipient * w + component] = ad::tanh(value);
          }
        }
      }

      std::vector<ad::Var> active_summary = zero_vector(tape, d);
      for (std::size_t local_index = 0; local_index < active.size();
           ++local_index) {
        const std::size_t mechanism = active[local_index];
        for (std::size_t component = 0; component < d; ++component) {
          active_summary[component] =
              active_summary[component] +
              route_weights[local_index] *
                  new_mechanisms[mechanism * d + component];
        }
      }

      state.mechanisms = std::move(new_mechanisms);
      state.workspace = std::move(new_workspace);
      state.inboxes = std::move(new_inboxes);
      state.active_summary = std::move(active_summary);
    }

    if (capture_trace) {
      trace.mechanism_state_after = snapshot(state.mechanisms);
      trace.inbox_after = snapshot(state.inboxes);
      trace.workspace_after = snapshot(state.workspace);
      traces.push_back(std::move(trace));
    }
  }

  const std::vector<ad::Var> final_workspace_pool =
      pool_workspace(tape, state.workspace, b, w);
  std::vector<ad::Var> logits;
  std::vector<ad::Var> workspace_aux_logits;
  logits.reserve(config_.output_classes);
  if (!config_.core_only) {
    workspace_aux_logits.reserve(config_.output_classes);
  }
  for (std::size_t output = 0; output < config_.output_classes; ++output) {
    ad::Var logit = parameter_var(tape, *output_bias_, output);
    if (config_.output_reads_spine) {
      logit = add_parameter_dot(tape, logit, *output_spine_, output,
                                state.spine);
    }
    if (!config_.core_only) {
      ad::Var workspace_logit = tape.constant(0.0);
      workspace_logit = add_parameter_dot(
          tape, workspace_logit, *output_workspace_, output,
          final_workspace_pool);
      workspace_aux_logits.push_back(workspace_logit);
      if (workspace_output_enabled) {
        logit = logit + workspace_logit;
      }
      if (mechanism_output_enabled) {
        logit = add_parameter_dot(tape, logit, *output_mechanism_, output,
                                  state.active_summary);
      }
    }
    if (config_.output_logit_bound > 0.0) {
      const ad::Var bound = tape.constant(config_.output_logit_bound);
      logit = bound * ad::tanh(logit / bound);
    }
    logits.push_back(logit);
  }

  return SequenceResult{std::move(logits), std::move(workspace_aux_logits),
                        std::move(state), std::move(traces)};
}

}  // namespace sgw
