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

constexpr std::array<std::size_t, 5> kPhase9EvaluationDelays{0, 6, 12, 18, 24};

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
    case sgw::ModelPreset::structural_core_full:
      return sgw::ExperimentCondition::structural_core_full;
    case sgw::ModelPreset::structural_core_blind:
      return sgw::ExperimentCondition::structural_core_blind;
    case sgw::ModelPreset::structural_mediation_linear:
      return sgw::ExperimentCondition::structural_mediation_linear;
    case sgw::ModelPreset::structural_mediation_bounded:
      return sgw::ExperimentCondition::structural_mediation_bounded;
    case sgw::ModelPreset::structural_kv_exact:
      return sgw::ExperimentCondition::structural_kv_exact;
    case sgw::ModelPreset::structural_kv_learned:
      return sgw::ExperimentCondition::structural_kv_learned;
    case sgw::ModelPreset::kv_fixed_position:
      return sgw::ExperimentCondition::kv_fixed_position;
    case sgw::ModelPreset::kv_first_free:
      return sgw::ExperimentCondition::kv_first_free;
    case sgw::ModelPreset::kv_hard_router:
      return sgw::ExperimentCondition::kv_hard_router;
    case sgw::ModelPreset::kv_annealed_router:
      return sgw::ExperimentCondition::kv_annealed_router;
    case sgw::ModelPreset::kv_full_capacity:
      return sgw::ExperimentCondition::kv_full_capacity;
    case sgw::ModelPreset::kv_oracle_retention:
      return sgw::ExperimentCondition::kv_oracle_retention;
    case sgw::ModelPreset::kv_fifo_eviction:
      return sgw::ExperimentCondition::kv_fifo_eviction;
    case sgw::ModelPreset::kv_reservoir:
      return sgw::ExperimentCondition::kv_reservoir;
    case sgw::ModelPreset::kv_hard_retention:
      return sgw::ExperimentCondition::kv_hard_retention;
    case sgw::ModelPreset::kv_annealed_retention:
      return sgw::ExperimentCondition::kv_annealed_retention;
    case sgw::ModelPreset::kv_delayed_oracle:
      return sgw::ExperimentCondition::kv_delayed_oracle;
    case sgw::ModelPreset::kv_delayed_fifo:
      return sgw::ExperimentCondition::kv_delayed_fifo;
    case sgw::ModelPreset::kv_delayed_reservoir:
      return sgw::ExperimentCondition::kv_delayed_reservoir;
    case sgw::ModelPreset::kv_delayed_hard:
      return sgw::ExperimentCondition::kv_delayed_hard;
    case sgw::ModelPreset::kv_delayed_annealed_direct:
      return sgw::ExperimentCondition::kv_delayed_annealed_direct;
    case sgw::ModelPreset::kv_delayed_annealed_curriculum:
      return sgw::ExperimentCondition::kv_delayed_annealed_curriculum;
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
             "mediation_aux_annealed|structural_core_full|"
             "structural_core_blind|structural_mediation_linear|"
             "structural_mediation_bounded|structural_kv_exact|"
             "structural_kv_learned|kv_fixed_position|kv_first_free|"
             "kv_hard_router|kv_annealed_router|kv_full_capacity|"
             "kv_oracle_retention|kv_fifo_eviction|kv_reservoir|"
             "kv_hard_retention|kv_annealed_retention|kv_delayed_oracle|"
             "kv_delayed_fifo|kv_delayed_reservoir|kv_delayed_hard|"
             "kv_delayed_annealed_direct|"
             "kv_delayed_annealed_curriculum] "
             "[--preset core_small|core_compute_matched|core_param_matched|"
             "sgw|sgw_broadcast_forced|core_full_content|"
             "core_content_blind|mediation_fixed|structural_core_full|"
             "structural_core_blind|structural_mediation_linear|"
             "structural_mediation_bounded|structural_kv_exact|"
             "structural_kv_learned|kv_fixed_position|kv_first_free|"
             "kv_hard_router|kv_annealed_router|kv_full_capacity|"
             "kv_oracle_retention|kv_fifo_eviction|kv_reservoir|"
             "kv_hard_retention|kv_annealed_retention|kv_delayed_oracle|"
             "kv_delayed_fifo|kv_delayed_reservoir|kv_delayed_hard|"
             "kv_delayed_annealed_direct|"
             "kv_delayed_annealed_curriculum] "
             "[--mode core_only|sgw] [--seed N] "
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
       options.preset == sgw::ModelPreset::core_content_blind ||
       options.preset == sgw::ModelPreset::structural_core_full ||
       options.preset == sgw::ModelPreset::structural_core_blind)) {
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

double tail_mean(const std::vector<double>& values, std::size_t count) {
  if (values.empty() || count == 0) return 0.0;
  count = std::min(count, values.size());
  const std::size_t begin = values.size() - count;
  double sum = 0.0;
  for (std::size_t index = begin; index < values.size(); ++index) {
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

bool is_structural_condition(sgw::ExperimentCondition condition) noexcept {
  return condition == sgw::ExperimentCondition::structural_core_full ||
         condition == sgw::ExperimentCondition::structural_core_blind ||
         condition == sgw::ExperimentCondition::structural_mediation_linear ||
         condition == sgw::ExperimentCondition::structural_mediation_bounded ||
         condition == sgw::ExperimentCondition::structural_kv_exact ||
         condition == sgw::ExperimentCondition::structural_kv_learned ||
         condition == sgw::ExperimentCondition::kv_fixed_position ||
         condition == sgw::ExperimentCondition::kv_first_free ||
         condition == sgw::ExperimentCondition::kv_hard_router ||
         condition == sgw::ExperimentCondition::kv_annealed_router;
}

bool is_retention_condition(sgw::ExperimentCondition condition) noexcept {
  return condition == sgw::ExperimentCondition::kv_full_capacity ||
         condition == sgw::ExperimentCondition::kv_oracle_retention ||
         condition == sgw::ExperimentCondition::kv_fifo_eviction ||
         condition == sgw::ExperimentCondition::kv_reservoir ||
         condition == sgw::ExperimentCondition::kv_hard_retention ||
         condition == sgw::ExperimentCondition::kv_annealed_retention ||
         condition == sgw::ExperimentCondition::kv_delayed_oracle ||
         condition == sgw::ExperimentCondition::kv_delayed_fifo ||
         condition == sgw::ExperimentCondition::kv_delayed_reservoir ||
         condition == sgw::ExperimentCondition::kv_delayed_hard ||
         condition == sgw::ExperimentCondition::kv_delayed_annealed_direct ||
         condition ==
             sgw::ExperimentCondition::kv_delayed_annealed_curriculum;
}

bool is_phase9_condition(sgw::ExperimentCondition condition) noexcept {
  return condition == sgw::ExperimentCondition::kv_delayed_oracle ||
         condition == sgw::ExperimentCondition::kv_delayed_fifo ||
         condition == sgw::ExperimentCondition::kv_delayed_reservoir ||
         condition == sgw::ExperimentCondition::kv_delayed_hard ||
         condition == sgw::ExperimentCondition::kv_delayed_annealed_direct ||
         condition ==
             sgw::ExperimentCondition::kv_delayed_annealed_curriculum;
}

sgw::RetentionTaskConfig retention_task_config(
    sgw::ExperimentCondition condition,
    std::size_t delay_override = static_cast<std::size_t>(-1)) {
  sgw::RetentionTaskConfig config;
  config.delay_binding_count = is_phase9_condition(condition) ? 18 : 0;
  if (delay_override != static_cast<std::size_t>(-1)) {
    config.delay_binding_count = delay_override;
  }
  return config;
}

sgw::BindingTaskConfig task_config(sgw::ExperimentCondition condition) {
  sgw::BindingTaskConfig config;
  config.entity_count = 6;
  config.value_count = 5;
  if (condition == sgw::ExperimentCondition::core_full_content ||
      condition == sgw::ExperimentCondition::core_content_blind ||
      condition == sgw::ExperimentCondition::mediation_final_only ||
      condition == sgw::ExperimentCondition::mediation_aux_annealed ||
      is_structural_condition(condition)) {
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
                       std::size_t sequence_length,
                       std::size_t value_count,
                       std::size_t train_count,
                       std::size_t holdout_count,
                       const sgw::SgwEsmModel& model,
                       const sgw::EvaluationMetrics& initial_train,
                       const sgw::EvaluationMetrics& initial_holdout,
                       const sgw::EvaluationMetrics& final_train,
                       const sgw::EvaluationMetrics& final_holdout,
                       const sgw::TrainingHistory& history,
                       const std::array<sgw::EvaluationMetrics, 5>*
                           delay_evaluations = nullptr) {
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
         "final_weighted_workspace_aux_window_loss,training_stream_samples,"
         "output_logit_bound,initial_train_brier,initial_holdout_brier,"
         "final_train_brier,final_holdout_brier,initial_train_ece,"
         "initial_holdout_ece,final_train_ece,final_holdout_ece,"
         "final_train_max_confidence,final_holdout_max_confidence,"
         "final_train_true_class_probability,"
         "final_holdout_true_class_probability,read_slot_load,"
         "write_slot_load,mean_write_collision_rate,mean_routing_entropy,"
         "mean_routing_disagreement_rate,router_initial_temperature,"
         "router_final_temperature,router_anneal_steps,"
         "final_200_collision_rate,final_200_routing_entropy,"
         "final_200_disagreement_rate,retention_write_rate,"
         "retention_skip_rate,retention_eviction_rate,relevant_eviction_rate,"
         "queried_entity_retention_rate,query_read_hit_rate,mean_retained_age,"
         "eviction_slot_load,final_300_retention_write_rate,"
         "final_300_retention_skip_rate,final_300_retention_eviction_rate,"
         "final_300_relevant_eviction_rate,final_300_query_read_hit_rate,"
         "mean_source_query_distance,delay_binding_count,"
         "distractor_write_rate,distractor_eviction_rate,"
         "relevant_survival_rate,final_450_distractor_write_rate,"
         "final_450_distractor_eviction_rate,"
         "final_450_relevant_survival_rate,final_450_disagreement_rate,"
         "delay0_nll,delay0_accuracy,delay0_query_hit,delay0_survival,"
         "delay6_nll,delay6_accuracy,delay6_query_hit,delay6_survival,"
         "delay12_nll,delay12_accuracy,delay12_query_hit,delay12_survival,"
         "delay18_nll,delay18_accuracy,delay18_query_hit,delay18_survival,"
         "delay24_nll,delay24_accuracy,delay24_query_hit,delay24_survival\n";
  output << std::fixed << std::setprecision(12)
         << sgw::model_preset_name(options.preset) << ',' << options.seed << ','
         << options.steps << ',' << options.batch_size << ','
         << model.parameters().scalar_count() << ',' << train_count
         << ',' << holdout_count << ','
         << sequence_length << ','
         << std::log(static_cast<double>(value_count)) << ','
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
         << ',' << final_weighted_aux_window(history) << ','
         << history.samples_consumed << ',' << model.config().output_logit_bound
         << ',' << initial_train.mean_brier << ',' << initial_holdout.mean_brier
         << ',' << final_train.mean_brier << ',' << final_holdout.mean_brier
         << ',' << initial_train.ece << ',' << initial_holdout.ece
         << ',' << final_train.ece << ',' << final_holdout.ece
         << ',' << final_train.mean_max_confidence << ','
         << final_holdout.mean_max_confidence << ','
         << final_train.mean_true_class_probability << ','
         << final_holdout.mean_true_class_probability << ','
         << mechanism_load_text(final_holdout.read_slot_load) << ','
         << mechanism_load_text(final_holdout.write_slot_load) << ','
         << final_holdout.mean_write_collision_rate << ','
         << final_holdout.mean_routing_entropy << ','
         << final_holdout.mean_routing_disagreement_rate << ','
         << model.config().key_value_router_initial_temperature << ','
         << model.config().key_value_router_final_temperature << ','
         << model.config().key_value_router_anneal_steps << ','
         << tail_mean(history.routing_collision_rate, 200) << ','
         << tail_mean(history.routing_entropy, 200) << ','
         << tail_mean(history.routing_disagreement_rate, 200) << ','
         << final_holdout.retention_write_rate << ','
         << final_holdout.retention_skip_rate << ','
         << final_holdout.retention_eviction_rate << ','
         << final_holdout.relevant_eviction_rate << ','
         << final_holdout.queried_entity_retention_rate << ','
         << final_holdout.query_read_hit_rate << ','
         << final_holdout.mean_retained_age << ','
         << mechanism_load_text(final_holdout.eviction_slot_load) << ','
         << tail_mean(history.retention_write_rate, 300) << ','
         << tail_mean(history.retention_skip_rate, 300) << ','
         << tail_mean(history.retention_eviction_rate, 300) << ','
         << tail_mean(history.relevant_eviction_rate, 300) << ','
         << tail_mean(history.query_read_hit_rate, 300) << ','
         << final_holdout.mean_source_query_distance << ','
         << final_holdout.mean_delay_binding_count << ','
         << final_holdout.distractor_write_rate << ','
         << final_holdout.distractor_eviction_rate << ','
         << final_holdout.relevant_survival_rate << ','
         << tail_mean(history.distractor_write_rate, 450) << ','
         << tail_mean(history.distractor_eviction_rate, 450) << ','
         << tail_mean(history.relevant_survival_rate, 450) << ','
         << tail_mean(history.routing_disagreement_rate, 450);
  for (std::size_t index = 0; index < kPhase9EvaluationDelays.size(); ++index) {
    const sgw::EvaluationMetrics metrics =
        delay_evaluations == nullptr
            ? sgw::EvaluationMetrics{}
            : delay_evaluations->at(index);
    output << ',' << metrics.mean_nll << ',' << metrics.accuracy << ','
           << metrics.query_read_hit_rate << ','
           << metrics.relevant_survival_rate;
  }
  output << '\n';
}

void write_causal_csv(const Options& options, sgw::SgwEsmModel& model,
                      std::span<const sgw::BindingSample> holdout) {
  const std::filesystem::path& path = *options.causal_output;
  const std::uint64_t seed = options.seed;
  std::vector<sgw::ForwardIntervention> interventions;
  if (model.config().key_value_retention !=
      sgw::KeyValueRetentionMode::none) {
    interventions = {
        sgw::ForwardIntervention::intact,
        sgw::ForwardIntervention::zero_query_context,
        sgw::ForwardIntervention::randomized_retention_actions,
        sgw::ForwardIntervention::force_fifo_retention,
        sgw::ForwardIntervention::force_relevant_eviction,
        sgw::ForwardIntervention::permuted_context_labels,
        sgw::ForwardIntervention::disable_retention_skip,
        sgw::ForwardIntervention::no_workspace_writes,
        sgw::ForwardIntervention::no_workspace_persistence,
        sgw::ForwardIntervention::no_mechanism_output,
    };
    if (is_phase9_condition(options.condition)) {
      interventions.push_back(
          sgw::ForwardIntervention::remove_delay_distractors);
      interventions.push_back(
          sgw::ForwardIntervention::relevant_looking_delay_distractors);
      interventions.push_back(
          sgw::ForwardIntervention::reverse_delay_block);
    }
  } else {
    interventions = {
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
    if (model.config().fixed_binding_mediation ||
        model.config().key_value_mode != sgw::KeyValueMediationMode::none) {
      interventions.push_back(sgw::ForwardIntervention::no_workspace_writes);
      interventions.push_back(sgw::ForwardIntervention::zero_reader_inbox);
    }
    if (model.config().key_value_mode != sgw::KeyValueMediationMode::none) {
      interventions.push_back(
          sgw::ForwardIntervention::permuted_workspace_keys);
      interventions.push_back(sgw::ForwardIntervention::zero_query_key);
      if (model.config().key_value_write_routing !=
          sgw::KeyValueWriteRoutingMode::fixed_position) {
        interventions.push_back(
            sgw::ForwardIntervention::randomized_write_slots);
        interventions.push_back(
            sgw::ForwardIntervention::cleared_writer_assignment);
        interventions.push_back(
            sgw::ForwardIntervention::allow_write_collisions);
      }
    }
    interventions.push_back(sgw::ForwardIntervention::permuted_recipients);
  }
  std::vector<sgw::EvaluationMetrics> metrics(interventions.size());
  for (std::size_t index = 0; index < interventions.size(); ++index) {
    metrics[index] =
        sgw::evaluate(model, holdout, true, interventions[index]);
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
         "condition,holdout_brier,holdout_ece,holdout_max_confidence,"
         "holdout_true_class_probability,brier_delta_vs_intact,"
         "ece_delta_vs_intact,max_confidence_delta_vs_intact,"
         "true_class_probability_delta_vs_intact,read_slot_load,"
         "write_slot_load,mean_write_collision_rate,mean_routing_entropy,"
         "mean_routing_disagreement_rate,retention_write_rate,"
         "retention_skip_rate,retention_eviction_rate,relevant_eviction_rate,"
         "queried_entity_retention_rate,query_read_hit_rate,mean_retained_age,"
         "eviction_slot_load,mean_source_query_distance,delay_binding_count,"
         "distractor_write_rate,distractor_eviction_rate,"
         "relevant_survival_rate\n";
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
           << ',' << current.mean_brier << ',' << current.ece << ','
           << current.mean_max_confidence << ','
           << current.mean_true_class_probability << ','
           << current.mean_brier - intact.mean_brier << ','
           << current.ece - intact.ece << ','
           << current.mean_max_confidence - intact.mean_max_confidence << ','
           << current.mean_true_class_probability -
                  intact.mean_true_class_probability
           << ',' << mechanism_load_text(current.read_slot_load) << ','
           << mechanism_load_text(current.write_slot_load) << ','
           << current.mean_write_collision_rate << ','
           << current.mean_routing_entropy << ','
           << current.mean_routing_disagreement_rate << ','
           << current.retention_write_rate << ','
           << current.retention_skip_rate << ','
           << current.retention_eviction_rate << ','
           << current.relevant_eviction_rate << ','
           << current.queried_entity_retention_rate << ','
           << current.query_read_hit_rate << ','
           << current.mean_retained_age << ','
           << mechanism_load_text(current.eviction_slot_load) << ','
           << current.mean_source_query_distance << ','
           << current.mean_delay_binding_count << ','
           << current.distractor_write_rate << ','
           << current.distractor_eviction_rate << ','
           << current.relevant_survival_rate << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto started = std::chrono::steady_clock::now();
    sgw::EvaluationMetrics initial_train;
    sgw::EvaluationMetrics initial_holdout;
    sgw::EvaluationMetrics final_train;
    sgw::EvaluationMetrics final_holdout;
    sgw::TrainingHistory history;
    std::optional<sgw::SgwEsmModel> model;

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

    if (is_retention_condition(options.condition)) {
      const sgw::RetentionTaskConfig task =
          retention_task_config(options.condition);
      const sgw::RetentionDataset dataset = sgw::make_retention_binding_split(
          task, options.train_count, options.holdout_count,
          options.seed ^ 0x9e3779b97f4a7c15ULL);
      const sgw::ModelConfig config = sgw::make_model_config(
          options.preset, dataset.vocabulary.size(), dataset.config.value_count);
      model.emplace(config, options.seed);
      initial_train = sgw::evaluate(*model, dataset.train, false);
      initial_holdout = sgw::evaluate(*model, dataset.holdout, false);
      if (options.condition ==
          sgw::ExperimentCondition::kv_delayed_annealed_curriculum) {
        std::vector<std::size_t> schedule;
        schedule.reserve(options.steps * options.batch_size);
        for (std::size_t step = 0; step < options.steps; ++step) {
          const std::size_t delay =
              sgw::phase9_curriculum_delay(step, options.steps);
          for (std::size_t item = 0; item < options.batch_size; ++item) {
            schedule.push_back(delay);
          }
        }
        sgw::RetentionBindingStream stream(
            task, options.seed ^ 0xd1b54a32d192ed03ULL, schedule);
        history = sgw::train_steps(*model, stream, adam, training);
      } else {
        sgw::RetentionBindingStream stream(
            task, options.seed ^ 0xd1b54a32d192ed03ULL,
            options.steps * options.batch_size);
        history = sgw::train_steps(*model, stream, adam, training);
      }
      final_train = sgw::evaluate(*model, dataset.train, true);
      final_holdout = sgw::evaluate(*model, dataset.holdout, true);
      std::array<sgw::EvaluationMetrics, 5> delay_evaluations{};
      const bool phase9 = is_phase9_condition(options.condition);
      if (phase9) {
        for (std::size_t index = 0;
             index < kPhase9EvaluationDelays.size(); ++index) {
          const auto evaluation_task = retention_task_config(
              options.condition, kPhase9EvaluationDelays[index]);
          const auto evaluation_dataset = sgw::make_retention_binding_split(
              evaluation_task, options.train_count, options.holdout_count,
              options.seed ^ 0x9e3779b97f4a7c15ULL);
          delay_evaluations[index] =
              sgw::evaluate(*model, evaluation_dataset.holdout, true);
        }
      }
      write_primary_csv(options, dataset.config.sequence_length(),
                        dataset.config.value_count, dataset.train.size(),
                        dataset.holdout.size(), *model, initial_train,
                        initial_holdout, final_train, final_holdout, history,
                        phase9 ? &delay_evaluations : nullptr);
      if (options.causal_output.has_value()) {
        write_causal_csv(options, *model, dataset.holdout);
      }
    } else {
      const sgw::BindingTaskConfig task = task_config(options.condition);
      const sgw::BindingDataset dataset =
          is_structural_condition(options.condition)
              ? sgw::make_structural_binding_split(
                    task, options.train_count, options.holdout_count,
                    options.seed ^ 0x9e3779b97f4a7c15ULL)
              : sgw::make_binding_split(
                    task, options.train_count, options.holdout_count,
                    options.seed ^ 0x9e3779b97f4a7c15ULL);
      const sgw::ModelConfig config = sgw::make_model_config(
          options.preset, dataset.vocabulary.size(), dataset.config.value_count);
      model.emplace(config, options.seed);
      initial_train = sgw::evaluate(*model, dataset.train, false);
      initial_holdout = sgw::evaluate(*model, dataset.holdout, false);
      if (is_structural_condition(options.condition)) {
        sgw::StructuralBindingStream stream(
            task, options.seed ^ 0xd1b54a32d192ed03ULL);
        history = sgw::train_steps(*model, stream, adam, training);
      } else {
        history = sgw::train_steps(*model, dataset.train, adam, training);
      }
      final_train = sgw::evaluate(*model, dataset.train, true);
      final_holdout = sgw::evaluate(*model, dataset.holdout, true);
      write_primary_csv(options, dataset.config.sequence_length(),
                        dataset.config.value_count, dataset.train.size(),
                        dataset.holdout.size(), *model, initial_train,
                        initial_holdout, final_train, final_holdout, history);
      if (options.causal_output.has_value()) {
        write_causal_csv(options, *model, dataset.holdout);
      }
    }

    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    std::cout << std::fixed << std::setprecision(6)
              << "preset=" << sgw::model_preset_name(options.preset)
              << " initial_holdout_nll=" << initial_holdout.mean_nll
              << " final_holdout_nll=" << final_holdout.mean_nll
              << " final_holdout_accuracy=" << final_holdout.accuracy
              << " parameters=" << model->parameters().scalar_count()
              << " madds_per_token="
              << final_holdout.mean_estimated_madds_per_token
              << " wall_seconds=" << elapsed << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sgw_train: " << error.what() << '\n';
    return 1;
  }
}
