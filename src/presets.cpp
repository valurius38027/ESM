#include "sgw/presets.hpp"

#include <stdexcept>
#include <string>

namespace sgw {

ExperimentCondition parse_experiment_condition(std::string_view value) {
  if (value == "core_small") return ExperimentCondition::core_small;
  if (value == "core_compute_matched") return ExperimentCondition::core_compute_matched;
  if (value == "core_param_matched") return ExperimentCondition::core_param_matched;
  if (value == "sgw_redundant_final_only") return ExperimentCondition::sgw_redundant_final_only;
  if (value == "sgw_broadcast_forced_final_only") return ExperimentCondition::sgw_broadcast_forced_final_only;
  if (value == "sgw_broadcast_forced_aux_annealed") return ExperimentCondition::sgw_broadcast_forced_aux_annealed;
  if (value == "core_full_content") return ExperimentCondition::core_full_content;
  if (value == "core_content_blind") return ExperimentCondition::core_content_blind;
  if (value == "mediation_final_only") return ExperimentCondition::mediation_final_only;
  if (value == "mediation_aux_annealed") return ExperimentCondition::mediation_aux_annealed;
  throw std::invalid_argument("unknown experiment condition: " + std::string(value));
}

std::string_view experiment_condition_name(ExperimentCondition condition) noexcept {
  switch (condition) {
    case ExperimentCondition::core_small: return "core_small";
    case ExperimentCondition::core_compute_matched: return "core_compute_matched";
    case ExperimentCondition::core_param_matched: return "core_param_matched";
    case ExperimentCondition::sgw_redundant_final_only: return "sgw_redundant_final_only";
    case ExperimentCondition::sgw_broadcast_forced_final_only: return "sgw_broadcast_forced_final_only";
    case ExperimentCondition::sgw_broadcast_forced_aux_annealed: return "sgw_broadcast_forced_aux_annealed";
    case ExperimentCondition::core_full_content: return "core_full_content";
    case ExperimentCondition::core_content_blind: return "core_content_blind";
    case ExperimentCondition::mediation_final_only: return "mediation_final_only";
    case ExperimentCondition::mediation_aux_annealed: return "mediation_aux_annealed";
  }
  return "unknown";
}

ModelPreset model_preset_for_condition(ExperimentCondition condition) noexcept {
  switch (condition) {
    case ExperimentCondition::core_small: return ModelPreset::core_small;
    case ExperimentCondition::core_compute_matched: return ModelPreset::core_compute_matched;
    case ExperimentCondition::core_param_matched: return ModelPreset::core_param_matched;
    case ExperimentCondition::sgw_redundant_final_only: return ModelPreset::sgw_redundant;
    case ExperimentCondition::sgw_broadcast_forced_final_only:
    case ExperimentCondition::sgw_broadcast_forced_aux_annealed:
      return ModelPreset::sgw_broadcast_forced;
    case ExperimentCondition::core_full_content: return ModelPreset::core_full_content;
    case ExperimentCondition::core_content_blind: return ModelPreset::core_content_blind;
    case ExperimentCondition::mediation_final_only:
    case ExperimentCondition::mediation_aux_annealed:
      return ModelPreset::mediation_fixed;
  }
  return ModelPreset::core_small;
}

double condition_workspace_aux_weight(ExperimentCondition condition) noexcept {
  return condition == ExperimentCondition::sgw_broadcast_forced_aux_annealed ||
                 condition == ExperimentCondition::mediation_aux_annealed
             ? 0.5
             : 0.0;
}

std::size_t condition_workspace_aux_anneal_steps(ExperimentCondition condition) noexcept {
  return condition == ExperimentCondition::sgw_broadcast_forced_aux_annealed ||
                 condition == ExperimentCondition::mediation_aux_annealed
             ? 200
             : 0;
}

ModelPreset parse_model_preset(std::string_view value) {
  if (value == "core_small" || value == "core_only") return ModelPreset::core_small;
  if (value == "core_compute_matched") return ModelPreset::core_compute_matched;
  if (value == "core_param_matched") return ModelPreset::core_param_matched;
  if (value == "sgw" || value == "sgw_redundant") return ModelPreset::sgw_redundant;
  if (value == "sgw_broadcast_forced") return ModelPreset::sgw_broadcast_forced;
  if (value == "core_full_content") return ModelPreset::core_full_content;
  if (value == "core_content_blind") return ModelPreset::core_content_blind;
  if (value == "mediation_fixed") return ModelPreset::mediation_fixed;
  throw std::invalid_argument("unknown model preset: " + std::string(value));
}

std::string_view model_preset_name(ModelPreset preset) noexcept {
  switch (preset) {
    case ModelPreset::core_small: return "core_small";
    case ModelPreset::core_compute_matched: return "core_compute_matched";
    case ModelPreset::core_param_matched: return "core_param_matched";
    case ModelPreset::sgw_redundant: return "sgw";
    case ModelPreset::sgw_broadcast_forced: return "sgw_broadcast_forced";
    case ModelPreset::core_full_content: return "core_full_content";
    case ModelPreset::core_content_blind: return "core_content_blind";
    case ModelPreset::mediation_fixed: return "mediation_fixed";
  }
  return "unknown";
}

ModelConfig make_model_config(ModelPreset preset, std::size_t vocab_size,
                              std::size_t output_classes) {
  ModelConfig config;
  config.vocab_size = vocab_size;
  config.output_classes = output_classes;
  config.mechanism_count = 6;
  config.mechanism_dim = 6;
  config.workspace_slots = 3;
  config.workspace_dim = 6;
  config.active_mechanisms = 2;
  config.workspace_writers = 1;
  config.broadcast_recipients = 2;
  switch (preset) {
    case ModelPreset::core_small:
      config.embedding_dim = 6;
      config.spine_dim = 10;
      config.core_only = true;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = false;
      break;
    case ModelPreset::core_compute_matched:
      config.embedding_dim = 66;
      config.spine_dim = 13;
      config.core_only = true;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = false;
      break;
    case ModelPreset::core_param_matched:
      config.embedding_dim = 40;
      config.spine_dim = 22;
      config.core_only = true;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = false;
      break;
    case ModelPreset::sgw_redundant:
      config.embedding_dim = 6;
      config.spine_dim = 10;
      break;
    case ModelPreset::sgw_broadcast_forced:
      config.embedding_dim = 6;
      config.spine_dim = 10;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      break;
    case ModelPreset::core_full_content:
      config.embedding_dim = 12;
      config.spine_dim = 24;
      config.core_only = true;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = false;
      break;
    case ModelPreset::core_content_blind:
      config.embedding_dim = 12;
      config.spine_dim = 24;
      config.core_only = true;
      config.spine_reads_embedding = false;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = false;
      break;
    case ModelPreset::mediation_fixed:
      config.embedding_dim = 12;
      config.spine_dim = 4;
      config.mechanism_count = 4;
      config.mechanism_dim = 24;
      config.workspace_slots = 3;
      config.workspace_dim = 36;
      config.active_mechanisms = 1;
      config.workspace_writers = 1;
      config.broadcast_recipients = 1;
      config.spine_reads_embedding = false;
      config.spine_reads_workspace = false;
      config.output_reads_spine = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = true;
      config.fixed_binding_mediation = true;
      config.mediation_binding_count = 3;
      break;
  }
  config.validate();
  return config;
}

}  // namespace sgw
