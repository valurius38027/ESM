#include "sgw/config.hpp"
#include "sgw/dataset.hpp"
#include "sgw/experiment.hpp"
#include "sgw/model.hpp"
#include "sgw/presets.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  sgw::ModelPreset preset{sgw::ModelPreset::sgw};
  std::uint64_t seed{1};
  std::size_t steps{80};
  std::size_t batch_size{6};
  std::size_t train_count{48};
  std::size_t holdout_count{24};
  std::filesystem::path output{"results/run.csv"};
  std::optional<std::filesystem::path> causal_output;
};

[[noreturn]] void usage_error(const std::string& message) {
  throw std::invalid_argument(
      message +
      "\nUsage: sgw_train [--preset core_small|core_compute_matched|"
      "core_param_matched|sgw] [--mode core_only|sgw] [--seed N] "
      "[--steps N] [--batch-size N] [--train-count N] "
      "[--holdout-count N] [--output PATH] [--causal-output PATH]");
}

template <typename Integer>
Integer parse_integer(std::string_view text, const char* option) {
  Integer value{};
  const char* begin = text.data();
  const char* end = text.data() + text.size();
  const auto [position, error] = std::from_chars(begin, end, value);
  if (error != std::errc{} || position != end) {
    usage_error(std::string("invalid integer for ") + option);
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      std::cout
          << "Usage: sgw_train [--preset core_small|core_compute_matched|"
             "core_param_matched|sgw] [--mode core_only|sgw] [--seed N] "
             "[--steps N] [--batch-size N] [--train-count N] "
             "[--holdout-count N] [--output PATH] "
             "[--causal-output PATH]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      usage_error("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--preset") {
      options.preset = sgw::parse_model_preset(value);
    } else if (argument == "--mode") {
      options.preset = sgw::parse_model_preset(value);
    } else if (argument == "--seed") {
      options.seed = parse_integer<std::uint64_t>(value, "--seed");
    } else if (argument == "--steps") {
      options.steps = parse_integer<std::size_t>(value, "--steps");
    } else if (argument == "--batch-size") {
      options.batch_size =
          parse_integer<std::size_t>(value, "--batch-size");
    } else if (argument == "--train-count") {
      options.train_count =
          parse_integer<std::size_t>(value, "--train-count");
    } else if (argument == "--holdout-count") {
      options.holdout_count =
          parse_integer<std::size_t>(value, "--holdout-count");
    } else if (argument == "--output") {
      options.output = std::filesystem::path(value);
    } else if (argument == "--causal-output") {
      options.causal_output = std::filesystem::path(value);
    } else {
      usage_error("unknown option: " + std::string(argument));
    }
  }
  if (options.steps == 0 || options.batch_size == 0 ||
      options.train_count == 0 || options.holdout_count == 0) {
    usage_error("steps, batch size and split sizes must be positive");
  }
  if (options.causal_output.has_value() &&
      options.preset != sgw::ModelPreset::sgw) {
    usage_error("--causal-output requires --preset sgw");
  }
  return options;
}

double window_mean(const std::vector<double>& values, bool first) {
  const std::size_t count = std::min<std::size_t>(10, values.size());
  const std::size_t begin = first ? 0 : values.size() - count;
  double sum = 0.0;
  for (std::size_t index = begin; index < begin + count; ++index) {
    sum += values[index];
  }
  return sum / static_cast<double>(count);
}

sgw::BindingTaskConfig task_config() {
  sgw::BindingTaskConfig config;
  config.entity_count = 6;
  config.value_count = 5;
  config.filler_count = 5;
  config.binding_count = 2;
  config.fillers_per_binding = 2;
  return config;
}

std::string mechanism_load_text(const std::vector<std::size_t>& load) {
  std::string text;
  for (std::size_t index = 0; index < load.size(); ++index) {
    if (index != 0) {
      text.push_back(';');
    }
    text += std::to_string(load[index]);
  }
  return text;
}

void prepare_parent(const std::filesystem::path& path) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

void write_primary_csv(const Options& options,
                       const sgw::BindingDataset& dataset,
                       const sgw::SgwEsmModel& model,
                       const sgw::EvaluationMetrics& initial_train,
                       const sgw::EvaluationMetrics& initial_holdout,
                       const sgw::EvaluationMetrics& final_train,
                       const sgw::EvaluationMetrics& final_holdout,
                       const sgw::TrainingHistory& history) {
  prepare_parent(options.output);
  std::ofstream output(options.output);
  if (!output) {
    throw std::runtime_error("unable to open output CSV");
  }
  output
      << "preset,seed,steps,batch_size,parameters,train_count,holdout_count,"
         "sequence_length,chance_nll,initial_train_nll,initial_holdout_nll,"
         "final_train_nll,final_holdout_nll,final_train_accuracy,"
         "final_holdout_accuracy,mean_estimated_madds_per_token,"
         "mean_active_mechanisms_per_token,mean_writers_per_token,"
         "mean_recipients_per_token,first_window_loss,last_window_loss,"
         "mechanism_load\n";
  output << std::fixed << std::setprecision(12)
         << sgw::model_preset_name(options.preset) << ',' << options.seed << ','
         << options.steps << ',' << options.batch_size << ','
         << model.parameters().scalar_count() << ',' << options.train_count
         << ',' << options.holdout_count << ','
         << dataset.config.sequence_length() << ','
         << std::log(static_cast<double>(dataset.config.value_count)) << ','
         << initial_train.mean_nll << ',' << initial_holdout.mean_nll << ','
         << final_train.mean_nll << ',' << final_holdout.mean_nll << ','
         << final_train.accuracy << ',' << final_holdout.accuracy << ','
         << final_holdout.mean_estimated_madds_per_token << ','
         << final_holdout.mean_active_mechanisms_per_token << ','
         << final_holdout.mean_writers_per_token << ','
         << final_holdout.mean_recipients_per_token << ','
         << window_mean(history.batch_loss, true) << ','
         << window_mean(history.batch_loss, false) << ','
         << mechanism_load_text(final_holdout.mechanism_load) << '\n';
}

void write_causal_csv(const std::filesystem::path& path, std::uint64_t seed,
                      sgw::SgwEsmModel& model,
                      const sgw::BindingDataset& dataset) {
  constexpr std::array interventions{
      sgw::ForwardIntervention::intact,
      sgw::ForwardIntervention::no_broadcast,
      sgw::ForwardIntervention::no_workspace_persistence,
      sgw::ForwardIntervention::no_workspace_output,
      sgw::ForwardIntervention::permuted_recipients,
  };
  std::array<sgw::EvaluationMetrics, interventions.size()> metrics;
  for (std::size_t index = 0; index < interventions.size(); ++index) {
    metrics[index] =
        sgw::evaluate(model, dataset.holdout, true, interventions[index]);
  }
  const auto& intact = metrics.front();

  prepare_parent(path);
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("unable to open causal output CSV");
  }
  output
      << "seed,intervention,holdout_nll,holdout_accuracy,"
         "nll_delta_vs_intact,accuracy_delta_vs_intact,"
         "mean_active_mechanisms_per_token,mean_writers_per_token,"
         "mean_recipients_per_token,mechanism_load\n";
  output << std::fixed << std::setprecision(12);
  for (std::size_t index = 0; index < interventions.size(); ++index) {
    const auto& current = metrics[index];
    output << seed << ',' << sgw::forward_intervention_name(interventions[index])
           << ',' << current.mean_nll << ',' << current.accuracy << ','
           << current.mean_nll - intact.mean_nll << ','
           << current.accuracy - intact.accuracy << ','
           << current.mean_active_mechanisms_per_token << ','
           << current.mean_writers_per_token << ','
           << current.mean_recipients_per_token << ','
           << mechanism_load_text(current.mechanism_load) << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto started = std::chrono::steady_clock::now();
    const sgw::BindingDataset dataset = sgw::make_binding_split(
        task_config(), options.train_count, options.holdout_count,
        options.seed ^ 0x9e3779b97f4a7c15ULL);
    const sgw::ModelConfig config = sgw::make_model_config(
        options.preset, dataset.vocabulary.size(), dataset.config.value_count);
    sgw::SgwEsmModel model(config, options.seed);

    sgw::AdamConfig adam;
    adam.learning_rate = 0.01;
    adam.max_grad_norm = 5.0;
    sgw::TrainingConfig training;
    training.steps = options.steps;
    training.batch_size = options.batch_size;
    training.shuffle_seed = options.seed ^ 0xd1b54a32d192ed03ULL;

    const auto initial_train = sgw::evaluate(model, dataset.train, false);
    const auto initial_holdout = sgw::evaluate(model, dataset.holdout, false);
    const auto history =
        sgw::train_steps(model, dataset.train, adam, training);
    const auto final_train = sgw::evaluate(model, dataset.train, true);
    const auto final_holdout = sgw::evaluate(model, dataset.holdout, true);
    write_primary_csv(options, dataset, model, initial_train, initial_holdout,
                      final_train, final_holdout, history);
    if (options.causal_output.has_value()) {
      write_causal_csv(*options.causal_output, options.seed, model, dataset);
    }

    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    std::cout << std::fixed << std::setprecision(6)
              << "preset=" << sgw::model_preset_name(options.preset)
              << " initial_holdout_nll=" << initial_holdout.mean_nll
              << " final_holdout_nll=" << final_holdout.mean_nll
              << " final_holdout_accuracy=" << final_holdout.accuracy
              << " parameters=" << model.parameters().scalar_count()
              << " madds_per_token="
              << final_holdout.mean_estimated_madds_per_token
              << " wall_seconds=" << elapsed << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sgw_train: " << error.what() << '\n';
    return 1;
  }
}
