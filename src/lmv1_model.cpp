#include "sgw/lmv1_model.hpp"

#include "sgw/topk.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace sgw::lmv1 {
namespace {

ad::Var parameter_var(ad::Tape& tape, Parameter& parameter,
                      std::size_t index) {
  return tape.parameter(parameter.value(index), &parameter.gradient(index));
}

ad::Var matrix_entry(ad::Tape& tape, Parameter& parameter, std::size_t row,
                     std::size_t column) {
  return parameter_var(tape, parameter, row * parameter.columns() + column);
}

std::vector<ad::Var> zeros(ad::Tape& tape, std::size_t count) {
  std::vector<ad::Var> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    result.push_back(tape.constant(0.0));
  }
  return result;
}

std::vector<ad::Var> embedding(ad::Tape& tape, Parameter& table,
                               std::uint8_t byte) {
  std::vector<ad::Var> result;
  result.reserve(table.columns());
  for (std::size_t column = 0; column < table.columns(); ++column) {
    result.push_back(matrix_entry(tape, table, byte, column));
  }
  return result;
}

ad::Var affine(ad::Tape& tape, Parameter& weight, std::size_t row,
               const std::vector<ad::Var>& input, ad::Var initial) {
  if (row >= weight.rows() || input.size() != weight.columns()) {
    throw std::logic_error("lmv1 affine shape mismatch");
  }
  ad::Var value = initial;
  for (std::size_t column = 0; column < input.size(); ++column) {
    value = value + matrix_entry(tape, weight, row, column) * input[column];
  }
  return value;
}

std::vector<ad::Var> recurrent_step(ad::Tape& tape, Parameter& input_weight,
                                    Parameter& recurrent_weight,
                                    Parameter& bias,
                                    const std::vector<ad::Var>& input,
                                    const std::vector<ad::Var>& hidden) {
  const std::size_t h = hidden.size();
  std::vector<ad::Var> next;
  next.reserve(h);
  for (std::size_t row = 0; row < h; ++row) {
    ad::Var value = parameter_var(tape, bias, row);
    value = affine(tape, input_weight, row, input, value);
    value = affine(tape, recurrent_weight, row, hidden, value);
    next.push_back(ad::tanh(value));
  }
  return next;
}

std::vector<ad::Var> gru_step(
    ad::Tape& tape, const std::vector<ad::Var>& input,
    const std::vector<ad::Var>& hidden, Parameter& update_input,
    Parameter& update_recurrent, Parameter& update_bias,
    Parameter& reset_input, Parameter& reset_recurrent, Parameter& reset_bias,
    Parameter& candidate_input, Parameter& candidate_recurrent,
    Parameter& candidate_bias) {
  const std::size_t h = hidden.size();
  std::vector<ad::Var> update;
  std::vector<ad::Var> reset;
  update.reserve(h);
  reset.reserve(h);
  for (std::size_t row = 0; row < h; ++row) {
    ad::Var z = parameter_var(tape, update_bias, row);
    z = affine(tape, update_input, row, input, z);
    z = affine(tape, update_recurrent, row, hidden, z);
    update.push_back(ad::sigmoid(z));

    ad::Var r = parameter_var(tape, reset_bias, row);
    r = affine(tape, reset_input, row, input, r);
    r = affine(tape, reset_recurrent, row, hidden, r);
    reset.push_back(ad::sigmoid(r));
  }
  std::vector<ad::Var> gated;
  gated.reserve(h);
  for (std::size_t index = 0; index < h; ++index) {
    gated.push_back(reset[index] * hidden[index]);
  }
  std::vector<ad::Var> next;
  next.reserve(h);
  const ad::Var one = tape.constant(1.0);
  for (std::size_t row = 0; row < h; ++row) {
    ad::Var candidate = parameter_var(tape, candidate_bias, row);
    candidate = affine(tape, candidate_input, row, input, candidate);
    candidate = affine(tape, candidate_recurrent, row, gated, candidate);
    candidate = ad::tanh(candidate);
    next.push_back(update[row] * hidden[row] +
                   (one - update[row]) * candidate);
  }
  return next;
}

std::size_t top_slot(ad::Tape& tape, Parameter& slot_key,
                     const std::vector<ad::Var>& message) {
  std::vector<double> scores;
  scores.reserve(slot_key.rows());
  for (std::size_t slot = 0; slot < slot_key.rows(); ++slot) {
    ad::Var score = tape.constant(0.0);
    for (std::size_t dim = 0; dim < message.size(); ++dim) {
      score = score + matrix_entry(tape, slot_key, slot, dim) * message[dim];
    }
    scores.push_back(score.value());
  }
  return stable_topk(scores, 1).front();
}

}  // namespace

std::string_view model_kind_name(ModelKind kind) noexcept {
  switch (kind) {
    case ModelKind::tiny_rnn:
      return "tiny_rnn";
    case ModelKind::tiny_gru:
      return "tiny_gru";
    case ModelKind::tiny_sgw:
      return "tiny_sgw";
  }
  return "unknown";
}

std::string_view communication_mode_name(CommunicationMode mode) noexcept {
  switch (mode) {
    case CommunicationMode::learned_sparse:
      return "learned_sparse";
    case CommunicationMode::no_workspace:
      return "no_workspace";
    case CommunicationMode::message_permuted:
      return "message_permuted";
    case CommunicationMode::local_only:
      return "local_only";
  }
  return "unknown";
}

void ModelConfig::validate() const {
  if (vocabulary_size != 256) {
    throw std::invalid_argument("lmv1 vocabulary must contain 256 bytes");
  }
  if (embedding_dim == 0 || embedding_dim > 64 || hidden_dim == 0 ||
      hidden_dim > 64 || message_dim == 0 || message_dim > 64) {
    throw std::invalid_argument("lmv1 dimensions must be in [1,64]");
  }
  if (kind == ModelKind::tiny_sgw) {
    if (module_count != 4) {
      throw std::invalid_argument("lmv1 SGW requires four modules");
    }
    if (workspace_slots == 0 || workspace_slots > 8) {
      throw std::invalid_argument("lmv1 workspace slots must be in [1,8]");
    }
  }
}

std::size_t estimated_parameter_count(const ModelConfig& config) {
  config.validate();
  const std::size_t v = config.vocabulary_size;
  const std::size_t e = config.embedding_dim;
  const std::size_t h = config.hidden_dim;
  const std::size_t d = config.message_dim;
  const std::size_t head = v * h + v;
  const std::size_t embeddings = v * e;
  switch (config.kind) {
    case ModelKind::tiny_rnn:
      return embeddings + h * e + h * h + h + head;
    case ModelKind::tiny_gru:
      return embeddings + 3 * (h * e + h * h + h) + head;
    case ModelKind::tiny_sgw:
      return embeddings + (h * e + h * h + h) +
             (h * e + h * h + h * d + h) + (d * h + d) +
             config.workspace_slots * d + head;
  }
  throw std::logic_error("unknown lmv1 model kind");
}

Model::Model(ModelConfig config, std::uint64_t seed) : config_(config) {
  config_.validate();
  std::mt19937_64 generator(seed);
  const std::size_t v = config_.vocabulary_size;
  const std::size_t e = config_.embedding_dim;
  const std::size_t h = config_.hidden_dim;
  const std::size_t d = config_.message_dim;

  embedding_ = &parameters_.add_xavier("lmv1.embedding", v, e, generator);
  if (config_.kind == ModelKind::tiny_rnn) {
    rnn_input_ = &parameters_.add_xavier("lmv1.rnn.input", h, e, generator);
    rnn_recurrent_ =
        &parameters_.add_xavier("lmv1.rnn.recurrent", h, h, generator);
    rnn_bias_ = &parameters_.add_zeros("lmv1.rnn.bias", h, 1);
  } else if (config_.kind == ModelKind::tiny_gru) {
    gru_update_input_ =
        &parameters_.add_xavier("lmv1.gru.update_input", h, e, generator);
    gru_update_recurrent_ = &parameters_.add_xavier(
        "lmv1.gru.update_recurrent", h, h, generator);
    gru_update_bias_ = &parameters_.add_zeros("lmv1.gru.update_bias", h, 1);
    gru_reset_input_ =
        &parameters_.add_xavier("lmv1.gru.reset_input", h, e, generator);
    gru_reset_recurrent_ = &parameters_.add_xavier(
        "lmv1.gru.reset_recurrent", h, h, generator);
    gru_reset_bias_ = &parameters_.add_zeros("lmv1.gru.reset_bias", h, 1);
    gru_candidate_input_ = &parameters_.add_xavier(
        "lmv1.gru.candidate_input", h, e, generator);
    gru_candidate_recurrent_ = &parameters_.add_xavier(
        "lmv1.gru.candidate_recurrent", h, h, generator);
    gru_candidate_bias_ =
        &parameters_.add_zeros("lmv1.gru.candidate_bias", h, 1);
  } else {
    specialist_input_ = &parameters_.add_xavier(
        "lmv1.sgw.specialist_input", h, e, generator);
    specialist_recurrent_ = &parameters_.add_xavier(
        "lmv1.sgw.specialist_recurrent", h, h, generator);
    specialist_bias_ =
        &parameters_.add_zeros("lmv1.sgw.specialist_bias", h, 1);
    readout_input_ = &parameters_.add_xavier(
        "lmv1.sgw.readout_input", h, e, generator);
    readout_recurrent_ = &parameters_.add_xavier(
        "lmv1.sgw.readout_recurrent", h, h, generator);
    readout_inbox_ = &parameters_.add_xavier(
        "lmv1.sgw.readout_inbox", h, d, generator);
    readout_bias_ =
        &parameters_.add_zeros("lmv1.sgw.readout_bias", h, 1);
    message_weight_ = &parameters_.add_xavier(
        "lmv1.sgw.message_weight", d, h, generator);
    message_bias_ =
        &parameters_.add_zeros("lmv1.sgw.message_bias", d, 1);
    slot_key_ = &parameters_.add_xavier(
        "lmv1.sgw.slot_key", config_.workspace_slots, d, generator);
  }
  output_weight_ =
      &parameters_.add_xavier("lmv1.output_weight", v, h, generator);
  output_bias_ = &parameters_.add_zeros("lmv1.output_bias", v, 1);
}

const ModelConfig& Model::config() const noexcept { return config_; }
ParameterSet& Model::parameters() noexcept { return parameters_; }
const ParameterSet& Model::parameters() const noexcept { return parameters_; }

std::vector<double> Model::parameter_values() const {
  std::vector<double> values;
  values.reserve(parameters_.scalar_count());
  for (const auto& parameter : parameters_.parameters()) {
    const auto span = parameter->values();
    values.insert(values.end(), span.begin(), span.end());
  }
  return values;
}

void Model::load_parameter_values(std::span<const double> values) {
  if (values.size() != parameters_.scalar_count()) {
    throw std::invalid_argument("lmv1 parameter snapshot size mismatch");
  }
  std::size_t cursor = 0;
  for (auto& parameter : parameters_.parameters()) {
    for (double& value : parameter->mutable_values()) value = values[cursor++];
  }
}

SequenceResult Model::forward(ad::Tape& tape,
                              std::span<const std::uint8_t> input,
                              bool capture_trace, CommunicationMode mode) {
  if (input.empty()) throw std::invalid_argument("lmv1 input is empty");
  if (config_.kind != ModelKind::tiny_sgw &&
      mode != CommunicationMode::learned_sparse) {
    throw std::invalid_argument("lmv1 communication mode requires SGW");
  }
  const std::size_t e = config_.embedding_dim;
  const std::size_t h = config_.hidden_dim;
  const std::size_t d = config_.message_dim;
  SequenceResult result;

  std::vector<ad::Var> output_hidden = zeros(tape, h);
  if (config_.kind == ModelKind::tiny_rnn) {
    for (const std::uint8_t byte : input) {
      output_hidden = recurrent_step(tape, *rnn_input_, *rnn_recurrent_,
                                     *rnn_bias_, embedding(tape, *embedding_, byte),
                                     output_hidden);
      result.estimated_madds += h * e + h * h;
    }
    result.dense_counterfactual_madds = result.estimated_madds;
  } else if (config_.kind == ModelKind::tiny_gru) {
    for (const std::uint8_t byte : input) {
      output_hidden = gru_step(
          tape, embedding(tape, *embedding_, byte), output_hidden,
          *gru_update_input_, *gru_update_recurrent_, *gru_update_bias_,
          *gru_reset_input_, *gru_reset_recurrent_, *gru_reset_bias_,
          *gru_candidate_input_, *gru_candidate_recurrent_,
          *gru_candidate_bias_);
      result.estimated_madds += 3 * (h * e + h * h);
    }
    result.dense_counterfactual_madds = result.estimated_madds;
  } else {
    std::vector<ad::Var> hidden = zeros(tape, config_.module_count * h);
    std::vector<ad::Var> workspace =
        zeros(tape, config_.workspace_slots * d);
    result.slot_load.assign(config_.workspace_slots, 0);
    if (capture_trace) result.traces.reserve(input.size());

    for (std::size_t token = 0; token < input.size(); ++token) {
      const std::uint8_t byte = input[token];
      const std::vector<ad::Var> byte_embedding =
          embedding(tape, *embedding_, byte);
      StepTrace trace;
      trace.token_index = token;
      trace.active_module = 1U + static_cast<std::size_t>(byte % 3U);

      std::vector<ad::Var> delivered = zeros(tape, d);
      if (mode != CommunicationMode::local_only) {
        std::vector<ad::Var> specialist_hidden;
        specialist_hidden.reserve(h);
        const std::size_t base = trace.active_module * h;
        for (std::size_t index = 0; index < h; ++index) {
          specialist_hidden.push_back(hidden[base + index]);
        }
        specialist_hidden = recurrent_step(
            tape, *specialist_input_, *specialist_recurrent_,
            *specialist_bias_, byte_embedding, specialist_hidden);
        for (std::size_t index = 0; index < h; ++index) {
          hidden[base + index] = specialist_hidden[index];
        }
        result.estimated_madds += h * e + h * h;

        std::vector<ad::Var> message;
        message.reserve(d);
        for (std::size_t row = 0; row < d; ++row) {
          ad::Var value = parameter_var(tape, *message_bias_, row);
          value = affine(tape, *message_weight_, row, specialist_hidden, value);
          message.push_back(ad::tanh(value));
        }
        result.estimated_madds += d * h;

        if (mode != CommunicationMode::no_workspace) {
          const std::size_t slot = top_slot(tape, *slot_key_, message);
          trace.write_slot = slot;
          ++result.write_count;
          ++result.slot_load[slot];
          for (std::size_t dim = 0; dim < d; ++dim) {
            workspace[slot * d + dim] = message[dim];
          }
          if (mode == CommunicationMode::message_permuted) {
            for (std::size_t dim = 0; dim < d; ++dim) {
              delivered[dim] = workspace[slot * d + (d - 1 - dim)];
            }
          } else {
            for (std::size_t dim = 0; dim < d; ++dim) {
              delivered[dim] = workspace[slot * d + dim];
            }
          }
          trace.delivered_to_readout = true;
          ++result.delivered_messages;
          result.estimated_madds += config_.workspace_slots * d + h * d;
        }
      }

      std::vector<ad::Var> readout_hidden;
      readout_hidden.reserve(h);
      for (std::size_t index = 0; index < h; ++index) {
        readout_hidden.push_back(hidden[index]);
      }
      std::vector<ad::Var> next;
      next.reserve(h);
      for (std::size_t row = 0; row < h; ++row) {
        ad::Var value = parameter_var(tape, *readout_bias_, row);
        value = affine(tape, *readout_input_, row, byte_embedding, value);
        value = affine(tape, *readout_recurrent_, row, readout_hidden, value);
        if (mode != CommunicationMode::no_workspace &&
            mode != CommunicationMode::local_only) {
          value = affine(tape, *readout_inbox_, row, delivered, value);
        }
        next.push_back(ad::tanh(value));
      }
      for (std::size_t index = 0; index < h; ++index) hidden[index] = next[index];
      output_hidden = next;
      result.estimated_madds += h * e + h * h;
      if (mode != CommunicationMode::no_workspace &&
          mode != CommunicationMode::local_only) {
        result.estimated_madds += h * d;
      }
      if (capture_trace) result.traces.push_back(std::move(trace));
    }
    result.dense_counterfactual_madds =
        result.estimated_madds + input.size() *
                                     (config_.module_count - 1) * h * d;
  }

  result.logits.reserve(config_.vocabulary_size);
  for (std::size_t row = 0; row < config_.vocabulary_size; ++row) {
    ad::Var value = parameter_var(tape, *output_bias_, row);
    value = affine(tape, *output_weight_, row, output_hidden, value);
    result.logits.push_back(value);
  }
  result.estimated_madds += config_.vocabulary_size * h;
  result.dense_counterfactual_madds += config_.vocabulary_size * h;
  return result;
}

}  // namespace sgw::lmv1
