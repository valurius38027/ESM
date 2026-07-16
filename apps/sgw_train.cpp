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
  sgw::ExperimentCondition condition{
      sgw::ExperimentCondition::sgw_redundant_final_only};
  sgw::ModelPreset preset{sgw::ModelPreset::sgw_redundant};
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

sgw::ExperimentCondition default_condition_for_preset(
    sgw::ModelPreset preset) noexcept {
  switch (preset) {
    case sgw::ModelPreset::core_small:
      return sgw::ExperimentCondition::core_small;
    case sgw::ModelPreset::core_compute_matched:
      return sgw::ExperimentCondition::core_compute_matched;
    case sgw::ModelPreset::core_param_matched:
      return sgw::ExperimentCondition::core_param_matched;
    case sgw::ModelPreset::sgw_redundant:
      return sgw::ExperimentCondition::sgw_redundant_final_only;
    case sgw::ModelPreset::sgw_broadcast_forced:
      return sgw::ExperimentCondition::sgw_broadcast_forced_final_only;
    case sgw::ModelPreset::core_full_content:
      return sgw::ExperimentCondition::core_full_content;
    case sgw::ModelPreset::core_content_blind:
      return sgw::ExperimentCondition::core_content_blind;
    case sgw::ModelPreset::mediation_fixed:
      return sgw::ExperimentCondition::mediation_final_only;
  }
  return sgw::ExperimentCondition::core_small;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      std::cout
          << "Usage: sgw_train [--condition core_small|"
             "core_compute_matched|core_param_matched|"
             "sgw_redundant_final_only|sgw_broadcast_forced_final_only|"
             "sgw_broadcast_forced_aux_annealed|core_full_content|"
             "core_content_blind|mediation_final_only|"
             "mediation_aux_annealed] "
             "[--preset core_small|core_compute_matched|core_param_matched|"
             "sgw|sgw_broadcast_forced|core_full_content|"
             "core_content_blind|mediation_fixed] [--mode core_only|sgw] [--seed N] "
             "[--steps N] [--batch-size N] [--train-count N] "
             "[--holdout-count N] [--output PATH] "
             "[--causal-output PATH]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      usage_error("missing value for " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--condition") {
      options.condition = sgw::parse_experiment_condition(value);
      options.preset = sgw::model_preset_for_condition(options.condition);
    } else if (argument == "--preset") {
      options.preset = sgw::parse_model_preset(value);
      options.condition = default_condition_for_preset(options.preset);
    } else if (argument == "--mode") {
      options.preset = sgw::parse_model_preset(value);
      options.condition = default_condition_for_preset(options.preset);
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
      (options.preset == sgw::ModelPreset::core_small ||
       options.preset == sgw::ModelPreset::core_compute_matched ||
       options.preset == sgw::ModelPreset::core_param_matched ||
       options.preset == sgw::ModelPreset::core_full_content ||
       options.preset == sgw::ModelPreset::core_content_blind)) {
    usage_error("--causal-output requires an SGW condition");
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

double window_mean_ending_at(const std::vector<double>& values,
                             std::size_t end_exclusive) {
  if (values.empty() || end_exclusive == 0) {
    return 0.0;
  }
  end_exclusive = std::min(end_exclusive, values.size());
  const std::size_t count = std::min<std::size_t>(10, end_exclusive);
  const std::size_t begin = end_exclusive - count;
  double sum = 0.0;
  for (std::size_t index = begin; index < end_exclusive; ++index) {
    sum += values[index];
  }
  return sum / static_cast<double>(count);
}

double final_weighted_aux_window(const sgw::TrainingHistory& history) {
  const std::size_t count = std::min<std::size_t>(
      10, history.workspace_aux_batch_loss.size());
  if (count == 0) {
    return 0.0;
  }
  const std::size_t begin = history.workspace_aux_batch_loss.size() - count;
  double sum = 0.0;
  for (std::size_t index = begin;
       index < history.workspace_aux_batch_loss.size(); ++index) {
    sum += history.workspace_aux_batch_loss[index] *
           history.workspace_aux_weight[index];
  }
  return sum / static_cast<double>(count);
}

sgw::BindingTaskConfig task_config(sgw::ExperimentCondition condition) {
  sgw::BindingTaskConfig config;
  config.entity_count = 6;
  config.value_count = 5;
  if (condition == sgw::ExperimentCondition::core_full_content ||
      condition == sgw::ExperimentCondition::core_content_blind ||
      condition == sgw::ExperimentCondition::mediation_final_only ||
      condition == sgw::ExperimentCondition::mediation_aux_annealed) {
    config.filler_count = 1;
    config.binding_count = 3;
    config.fillers_per_binding = 0;
  } else {
    config.filler_count = 5;
    config.binding_count = 2;
    config.fillers_per_binding = 2;
  }
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

std::string role_mechanism_load_text(
    const std::vector<std::size_t>& load,
    std::size_t mechanism_count) {
  const std::size_t role_count =
      static_cast<std::size_t>(sgw::TokenRole::count);
  if (mechanism_count == 0 || load.size() != role_count * mechanism_count) {
    throw std::invalid_argument(
        "role mechanism load shape does not match model configuration");
  }
  std::string text;
  for (std::size_t role = 0; role < role_count; ++role) {
    if (role != 0) {
      text.push_back('|');
    }
    text += sgw::token_role_name(static_cast<sgw::TokenRole>(role));
    text.push_back(':');
    for (std::size_t mechanism = 0; mechanism < mechanism_count;
         ++mechanism) {
      if (mechanism != 0) {
        text.push_back(';');
      }
      text += std::to_string(load[role * mechanism_count + mechanism]);
    }
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
         "mechanism_load,role_mechanism_load,condition,"
         "workspace_aux_initial_weight,workspace_aux_anneal_steps,"
         "first_primary_window_loss,last_primary_window_loss,"
         "last_workspace_aux_window_loss,"
         "anneal_boundary_primary_window_loss,final_workspace_aux_weight,"
         "final_weighted_workspace_aux_window_loss\n";
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
         << mechanism_load_text(final_holdout.mechanism_load) << ','
         << role_mechanism_load_text(final_holdout.role_mechanism_load,
                                     model.config().mechanism_count)
         << ',' << sgw::experiment_condition_name(options.condition) << ','
         << sgw::condition_workspace_aux_weight(options.condition) << ','
         << sgw::condition_workspace_aux_anneal_steps(options.condition) << ','
         << window_mean(history.primary_batch_loss, true) << ','
         << window_mean(history.primary_batch_loss, false) << ','
         << window_mean(history.workspace_aux_batch_loss, false) << ','
         << window_mean_ending_at(
                history.primary_batch_loss,
                sgw::condition_workspace_aux_anneal_steps(options.condition))
         << ','
         << (history.workspace_aux_weight.empty()
                 ? 0.0
                 : history.workspace_aux_weight.back())
         << ',' << final_weighted_aux_window(history) << '\n';
}

void write_causal_csv(const Options& options, sgw::SgwEsmModel& model,
                      const sgw::BindingDataset& dataset) {
  const std::filesystem::path& path = *options.causal_output;
  const std::uint64_t seed = options.seed;
  std::vector<sgw::ForwardIntervention> interventions{
      sgw::ForwardIntervention::intact,
      sgw::ForwardIntervention::no_broadcast,
      sgw::ForwardIntervention::no_workspace_persistence,
      sgw::ForwardIntervention::no_workspace_output,
  };
  if (!model.config().spine_reads_workspace &&
      !model.config().output_reads_workspace) {
    interventions.push_back(sgw::ForwardIntervention::no_spine_workspace);
    interventions.push_back(sgw::ForwardIntervention::no_mechanism_output);
    interventions.push_back(sgw::ForwardIntervention::workspace_disconnected);
  }
  if (model.config().fixed_binding_mediation) {
    interventions.push_back(sgw::ForwardIntervention::no_workspace_writes);
    interventions.push_back(sgw::ForwardIntervention::zero_reader_inbox);
  }
  interventions.push_back(sgw::ForwardIntervention::permuted_recipients);
  std::vector<sgw::EvaluationMetrics> metrics(interventions.size());
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
         "mean_recipients_per_token,mechanism_load,role_mechanism_load,"
         "condition\n";
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
           << mechanism_load_text(current.mechanism_load) << ','
           << role_mechanism_load_text(current.role_mechanism_load,
                                       model.config().mechanism_count)
           << ',' << sgw::experiment_condition_name(options.condition)
           << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto started = std::chrono::steady_clock::now();
    const sgw::BindingDataset dataset = sgw::make_binding_split(
        task_config(options.condition), options.train_count, options.holdout_count,
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
    training.workspace_aux_weight =
        sgw::condition_workspace_aux_weight(options.condition);
    training.workspace_aux_anneal_steps =
        sgw::condition_workspace_aux_anneal_steps(options.condition);

    const auto initial_train = sgw::evaluate(model, dataset.train, false);
    const auto initial_holdout = sgw::evaluate(model, dataset.holdout, false);
    const auto history =
        sgw::train_steps(model, dataset.train, adam, training);
    const auto final_train = sgw::evaluate(model, dataset.train, true);
    const auto final_holdout = sgw::evaluate(model, dataset.holdout, true);
    write_primary_csv(options, dataset, model, initial_train, initial_holdout,
                      final_train, final_holdout, history);
    if (options.causal_output.has_value()) {
      write_causal_csv(options, model, dataset);
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
