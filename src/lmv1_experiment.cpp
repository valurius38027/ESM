#include "sgw/lmv1_experiment.hpp"

#include "sgw/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace sgw::lmv1 {
namespace {

ad::Var cross_entropy(ad::Tape& tape, const std::vector<ad::Var>& logits,
                      std::uint8_t target) {
  double maximum = logits.front().value();
  for (const auto logit : logits) maximum = std::max(maximum, logit.value());
  const ad::Var max_value = tape.constant(maximum);
  ad::Var denominator = tape.constant(0.0);
  for (const auto logit : logits) {
    denominator = denominator + ad::exp(logit - max_value);
  }
  return ad::log(denominator) + max_value - logits[target];
}

std::vector<double> probabilities(const std::vector<ad::Var>& logits) {
  double maximum = logits.front().value();
  for (const auto logit : logits) maximum = std::max(maximum, logit.value());
  std::vector<double> result;
  result.reserve(logits.size());
  double sum = 0.0;
  for (const auto logit : logits) {
    const double value = std::exp(logit.value() - maximum);
    result.push_back(value);
    sum += value;
  }
  for (double& value : result) value /= sum;
  return result;
}

std::vector<std::size_t> shuffled_offsets(const WindowDataset& data,
                                          std::uint64_t seed) {
  std::vector<std::size_t> offsets(data.size());
  std::iota(offsets.begin(), offsets.end(), 0);
  std::mt19937_64 generator(seed);
  std::shuffle(offsets.begin(), offsets.end(), generator);
  return offsets;
}

template <typename T>
void write_pod(std::ofstream& output, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!output) throw std::runtime_error("unable to write lmv1 checkpoint");
}

template <typename T>
T read_pod(std::ifstream& input) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) throw std::runtime_error("unable to read lmv1 checkpoint");
  return value;
}

}  // namespace

void TrainingConfig::validate() const {
  if (steps == 0 || steps > 500 || batch_size == 0 || batch_size > 32) {
    throw std::invalid_argument("lmv1 training budget outside Phase 25 bounds");
  }
  if (context_length < 8 || context_length > 128) {
    throw std::invalid_argument("lmv1 context must be in [8,128]");
  }
  if (validation_interval == 0 || steps % validation_interval != 0 ||
      evaluation_windows == 0) {
    throw std::invalid_argument("invalid lmv1 validation schedule");
  }
  optimizer.validate();
}

Metrics evaluate(Model& model, const WindowDataset& data,
                 std::size_t window_count, std::uint64_t offset_seed,
                 CommunicationMode mode) {
  if (window_count == 0 || window_count > data.size()) {
    throw std::invalid_argument("invalid lmv1 evaluation window count");
  }
  const auto before = model.parameter_values();
  const auto offsets = shuffled_offsets(data, offset_seed);
  Metrics metrics;
  metrics.sample_count = window_count;
  std::size_t correct = 0;
  for (std::size_t index = 0; index < window_count; ++index) {
    const Window window = data.at_offset(offsets[index]);
    ad::Tape tape;
    const SequenceResult result = model.forward(tape, window.inputs, false, mode);
    const auto probs = probabilities(result.logits);
    const std::uint8_t target = window.targets.back();
    metrics.mean_nll -= std::log(std::max(probs[target], 1.0e-300));
    const auto best = static_cast<std::size_t>(std::distance(
        probs.begin(), std::max_element(probs.begin(), probs.end())));
    correct += best == target ? 1U : 0U;
    metrics.mean_estimated_madds_per_predicted_byte +=
        static_cast<double>(result.estimated_madds);
    metrics.mean_dense_counterfactual_madds_per_predicted_byte +=
        static_cast<double>(result.dense_counterfactual_madds);
    metrics.mean_writes_per_predicted_byte +=
        static_cast<double>(result.write_count) /
        static_cast<double>(window.inputs.size());
    metrics.mean_delivered_per_predicted_byte +=
        static_cast<double>(result.delivered_messages) /
        static_cast<double>(window.inputs.size());
  }
  const double count = static_cast<double>(window_count);
  metrics.mean_nll /= count;
  metrics.bits_per_byte = metrics.mean_nll / std::log(2.0);
  metrics.accuracy = static_cast<double>(correct) / count;
  metrics.mean_estimated_madds_per_predicted_byte /= count;
  metrics.mean_dense_counterfactual_madds_per_predicted_byte /= count;
  metrics.mean_writes_per_predicted_byte /= count;
  metrics.mean_delivered_per_predicted_byte /= count;
  if (model.parameter_values() != before) {
    throw std::logic_error("lmv1 evaluation mutated parameters");
  }
  return metrics;
}

TrainingReport train(Model& model, const WindowDataset& training_data,
                     const WindowDataset& validation_data,
                     const TrainingConfig& config) {
  config.validate();
  if (training_data.context_length() != config.context_length ||
      validation_data.context_length() != config.context_length) {
    throw std::invalid_argument("lmv1 data context mismatch");
  }
  auto offsets = shuffled_offsets(training_data, config.shuffle_seed);
  std::mt19937_64 generator(config.shuffle_seed);
  std::size_t cursor = 0;
  Adam optimizer(config.optimizer);
  TrainingReport report;
  const std::size_t eval_count =
      std::min(config.evaluation_windows, validation_data.size());
  report.initial_validation_nll =
      evaluate(model, validation_data, eval_count,
               config.shuffle_seed ^ 0xa5a5a5a5ULL)
          .mean_nll;
  report.loss_curve.push_back(
      LossPoint{0, report.initial_validation_nll,
                report.initial_validation_nll});
  double madd_sum = 0.0;

  for (std::size_t step = 1; step <= config.steps; ++step) {
    model.parameters().zero_grad();
    ad::Tape tape;
    ad::Var loss = tape.constant(0.0);
    std::size_t batch_madds = 0;
    for (std::size_t item = 0; item < config.batch_size; ++item) {
      if (cursor == offsets.size()) {
        std::shuffle(offsets.begin(), offsets.end(), generator);
        cursor = 0;
      }
      const Window window = training_data.at_offset(offsets[cursor++]);
      const SequenceResult result = model.forward(tape, window.inputs);
      loss = loss + cross_entropy(tape, result.logits, window.targets.back());
      batch_madds += result.estimated_madds;
    }
    const ad::Var mean_loss =
        loss / tape.constant(static_cast<double>(config.batch_size));
    report.final_train_nll = mean_loss.value();
    tape.backward(mean_loss);
    if (!model.parameters().all_finite()) {
      throw std::runtime_error("nonfinite lmv1 parameter or gradient");
    }
    optimizer.step(model.parameters());
    madd_sum += static_cast<double>(batch_madds) /
                static_cast<double>(config.batch_size);
    if (step % config.validation_interval == 0) {
      const double validation_nll =
          evaluate(model, validation_data, eval_count,
                   config.shuffle_seed ^ 0xa5a5a5a5ULL)
              .mean_nll;
      report.loss_curve.push_back(
          LossPoint{step, report.final_train_nll, validation_nll});
    }
  }
  report.windows_consumed = config.steps * config.batch_size;
  report.predicted_bytes = report.windows_consumed;
  report.final_validation_nll = report.loss_curve.back().validation_nll;
  report.mean_estimated_madds_per_predicted_byte =
      madd_sum / static_cast<double>(config.steps);
  return report;
}

void save_checkpoint(const std::filesystem::path& path, const Model& model) {
  const auto temporary = path.string() + ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("unable to open lmv1 checkpoint");
  output.write("LMV1CP01", 8);
  const ModelConfig& config = model.config();
  write_pod(output, static_cast<std::uint32_t>(config.kind));
  write_pod(output, static_cast<std::uint64_t>(config.vocabulary_size));
  write_pod(output, static_cast<std::uint64_t>(config.embedding_dim));
  write_pod(output, static_cast<std::uint64_t>(config.hidden_dim));
  write_pod(output, static_cast<std::uint64_t>(config.module_count));
  write_pod(output, static_cast<std::uint64_t>(config.message_dim));
  write_pod(output, static_cast<std::uint64_t>(config.workspace_slots));
  const auto values = model.parameter_values();
  write_pod(output, static_cast<std::uint64_t>(values.size()));
  for (const double value : values) write_pod(output, value);
  output.close();
  if (!output) throw std::runtime_error("unable to close lmv1 checkpoint");
  std::filesystem::rename(temporary, path);
}

Checkpoint load_checkpoint(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("unable to open lmv1 checkpoint");
  char magic[8]{};
  input.read(magic, 8);
  if (!input || std::string_view(magic, 8) != "LMV1CP01") {
    throw std::runtime_error("invalid lmv1 checkpoint magic");
  }
  Checkpoint result;
  result.config.kind = static_cast<ModelKind>(read_pod<std::uint32_t>(input));
  result.config.vocabulary_size = read_pod<std::uint64_t>(input);
  result.config.embedding_dim = read_pod<std::uint64_t>(input);
  result.config.hidden_dim = read_pod<std::uint64_t>(input);
  result.config.module_count = read_pod<std::uint64_t>(input);
  result.config.message_dim = read_pod<std::uint64_t>(input);
  result.config.workspace_slots = read_pod<std::uint64_t>(input);
  result.config.validate();
  const std::size_t count = read_pod<std::uint64_t>(input);
  if (count != estimated_parameter_count(result.config)) {
    throw std::runtime_error("invalid lmv1 checkpoint parameter count");
  }
  result.parameter_values.resize(count);
  for (double& value : result.parameter_values) value = read_pod<double>(input);
  if (input.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("trailing lmv1 checkpoint data");
  }
  return result;
}

}  // namespace sgw::lmv1
