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
  if (value == "structural_core_full") return ExperimentCondition::structural_core_full;
  if (value == "structural_core_blind") return ExperimentCondition::structural_core_blind;
  if (value == "structural_mediation_linear") return ExperimentCondition::structural_mediation_linear;
  if (value == "structural_mediation_bounded") return ExperimentCondition::structural_mediation_bounded;
  if (value == "structural_kv_exact") return ExperimentCondition::structural_kv_exact;
  if (value == "structural_kv_learned") return ExperimentCondition::structural_kv_learned;
  if (value == "kv_fixed_position") return ExperimentCondition::kv_fixed_position;
  if (value == "kv_first_free") return ExperimentCondition::kv_first_free;
  if (value == "kv_hard_router") return ExperimentCondition::kv_hard_router;
  if (value == "kv_annealed_router") return ExperimentCondition::kv_annealed_router;
  if (value == "kv_full_capacity") return ExperimentCondition::kv_full_capacity;
  if (value == "kv_oracle_retention") return ExperimentCondition::kv_oracle_retention;
  if (value == "kv_fifo_eviction") return ExperimentCondition::kv_fifo_eviction;
  if (value == "kv_reservoir") return ExperimentCondition::kv_reservoir;
  if (value == "kv_hard_retention") return ExperimentCondition::kv_hard_retention;
  if (value == "kv_annealed_retention") return ExperimentCondition::kv_annealed_retention;
  if (value == "kv_delayed_oracle") return ExperimentCondition::kv_delayed_oracle;
  if (value == "kv_delayed_fifo") return ExperimentCondition::kv_delayed_fifo;
  if (value == "kv_delayed_reservoir") return ExperimentCondition::kv_delayed_reservoir;
  if (value == "kv_delayed_hard") return ExperimentCondition::kv_delayed_hard;
  if (value == "kv_delayed_annealed_direct") return ExperimentCondition::kv_delayed_annealed_direct;
  if (value == "kv_delayed_annealed_curriculum") return ExperimentCondition::kv_delayed_annealed_curriculum;
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
    case ExperimentCondition::structural_core_full: return "structural_core_full";
    case ExperimentCondition::structural_core_blind: return "structural_core_blind";
    case ExperimentCondition::structural_mediation_linear: return "structural_mediation_linear";
    case ExperimentCondition::structural_mediation_bounded: return "structural_mediation_bounded";
    case ExperimentCondition::structural_kv_exact: return "structural_kv_exact";
    case ExperimentCondition::structural_kv_learned: return "structural_kv_learned";
    case ExperimentCondition::kv_fixed_position: return "kv_fixed_position";
    case ExperimentCondition::kv_first_free: return "kv_first_free";
    case ExperimentCondition::kv_hard_router: return "kv_hard_router";
    case ExperimentCondition::kv_annealed_router: return "kv_annealed_router";
    case ExperimentCondition::kv_full_capacity: return "kv_full_capacity";
    case ExperimentCondition::kv_oracle_retention: return "kv_oracle_retention";
    case ExperimentCondition::kv_fifo_eviction: return "kv_fifo_eviction";
    case ExperimentCondition::kv_reservoir: return "kv_reservoir";
    case ExperimentCondition::kv_hard_retention: return "kv_hard_retention";
    case ExperimentCondition::kv_annealed_retention: return "kv_annealed_retention";
    case ExperimentCondition::kv_delayed_oracle: return "kv_delayed_oracle";
    case ExperimentCondition::kv_delayed_fifo: return "kv_delayed_fifo";
    case ExperimentCondition::kv_delayed_reservoir: return "kv_delayed_reservoir";
    case ExperimentCondition::kv_delayed_hard: return "kv_delayed_hard";
    case ExperimentCondition::kv_delayed_annealed_direct: return "kv_delayed_annealed_direct";
    case ExperimentCondition::kv_delayed_annealed_curriculum: return "kv_delayed_annealed_curriculum";
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
    case ExperimentCondition::structural_core_full:
      return ModelPreset::structural_core_full;
    case ExperimentCondition::structural_core_blind:
      return ModelPreset::structural_core_blind;
    case ExperimentCondition::structural_mediation_linear:
      return ModelPreset::structural_mediation_linear;
    case ExperimentCondition::structural_mediation_bounded:
      return ModelPreset::structural_mediation_bounded;
    case ExperimentCondition::structural_kv_exact:
      return ModelPreset::structural_kv_exact;
    case ExperimentCondition::structural_kv_learned:
      return ModelPreset::structural_kv_learned;
    case ExperimentCondition::kv_fixed_position:
      return ModelPreset::kv_fixed_position;
    case ExperimentCondition::kv_first_free:
      return ModelPreset::kv_first_free;
    case ExperimentCondition::kv_hard_router:
      return ModelPreset::kv_hard_router;
    case ExperimentCondition::kv_annealed_router:
      return ModelPreset::kv_annealed_router;
    case ExperimentCondition::kv_full_capacity:
      return ModelPreset::kv_full_capacity;
    case ExperimentCondition::kv_oracle_retention:
      return ModelPreset::kv_oracle_retention;
    case ExperimentCondition::kv_fifo_eviction:
      return ModelPreset::kv_fifo_eviction;
    case ExperimentCondition::kv_reservoir:
      return ModelPreset::kv_reservoir;
    case ExperimentCondition::kv_hard_retention:
      return ModelPreset::kv_hard_retention;
    case ExperimentCondition::kv_annealed_retention:
      return ModelPreset::kv_annealed_retention;
    case ExperimentCondition::kv_delayed_oracle:
      return ModelPreset::kv_delayed_oracle;
    case ExperimentCondition::kv_delayed_fifo:
      return ModelPreset::kv_delayed_fifo;
    case ExperimentCondition::kv_delayed_reservoir:
      return ModelPreset::kv_delayed_reservoir;
    case ExperimentCondition::kv_delayed_hard:
      return ModelPreset::kv_delayed_hard;
    case ExperimentCondition::kv_delayed_annealed_direct:
      return ModelPreset::kv_delayed_annealed_direct;
    case ExperimentCondition::kv_delayed_annealed_curriculum:
      return ModelPreset::kv_delayed_annealed_curriculum;
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
  if (value == "structural_core_full") return ModelPreset::structural_core_full;
  if (value == "structural_core_blind") return ModelPreset::structural_core_blind;
  if (value == "structural_mediation_linear") return ModelPreset::structural_mediation_linear;
  if (value == "structural_mediation_bounded") return ModelPreset::structural_mediation_bounded;
  if (value == "structural_kv_exact") return ModelPreset::structural_kv_exact;
  if (value == "structural_kv_learned") return ModelPreset::structural_kv_learned;
  if (value == "kv_fixed_position") return ModelPreset::kv_fixed_position;
  if (value == "kv_first_free") return ModelPreset::kv_first_free;
  if (value == "kv_hard_router") return ModelPreset::kv_hard_router;
  if (value == "kv_annealed_router") return ModelPreset::kv_annealed_router;
  if (value == "kv_full_capacity") return ModelPreset::kv_full_capacity;
  if (value == "kv_oracle_retention") return ModelPreset::kv_oracle_retention;
  if (value == "kv_fifo_eviction") return ModelPreset::kv_fifo_eviction;
  if (value == "kv_reservoir") return ModelPreset::kv_reservoir;
  if (value == "kv_hard_retention") return ModelPreset::kv_hard_retention;
  if (value == "kv_annealed_retention") return ModelPreset::kv_annealed_retention;
  if (value == "kv_delayed_oracle") return ModelPreset::kv_delayed_oracle;
  if (value == "kv_delayed_fifo") return ModelPreset::kv_delayed_fifo;
  if (value == "kv_delayed_reservoir") return ModelPreset::kv_delayed_reservoir;
  if (value == "kv_delayed_hard") return ModelPreset::kv_delayed_hard;
  if (value == "kv_delayed_annealed_direct") return ModelPreset::kv_delayed_annealed_direct;
  if (value == "kv_delayed_annealed_curriculum") return ModelPreset::kv_delayed_annealed_curriculum;
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
    case ModelPreset::structural_core_full: return "structural_core_full";
    case ModelPreset::structural_core_blind: return "structural_core_blind";
    case ModelPreset::structural_mediation_linear: return "structural_mediation_linear";
    case ModelPreset::structural_mediation_bounded: return "structural_mediation_bounded";
    case ModelPreset::structural_kv_exact: return "structural_kv_exact";
    case ModelPreset::structural_kv_learned: return "structural_kv_learned";
    case ModelPreset::kv_fixed_position: return "kv_fixed_position";
    case ModelPreset::kv_first_free: return "kv_first_free";
    case ModelPreset::kv_hard_router: return "kv_hard_router";
    case ModelPreset::kv_annealed_router: return "kv_annealed_router";
    case ModelPreset::kv_full_capacity: return "kv_full_capacity";
    case ModelPreset::kv_oracle_retention: return "kv_oracle_retention";
    case ModelPreset::kv_fifo_eviction: return "kv_fifo_eviction";
    case ModelPreset::kv_reservoir: return "kv_reservoir";
    case ModelPreset::kv_hard_retention: return "kv_hard_retention";
    case ModelPreset::kv_annealed_retention: return "kv_annealed_retention";
    case ModelPreset::kv_delayed_oracle: return "kv_delayed_oracle";
    case ModelPreset::kv_delayed_fifo: return "kv_delayed_fifo";
    case ModelPreset::kv_delayed_reservoir: return "kv_delayed_reservoir";
    case ModelPreset::kv_delayed_hard: return "kv_delayed_hard";
    case ModelPreset::kv_delayed_annealed_direct: return "kv_delayed_annealed_direct";
    case ModelPreset::kv_delayed_annealed_curriculum: return "kv_delayed_annealed_curriculum";
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
    case ModelPreset::structural_core_full:
    case ModelPreset::core_full_content:
      config.embedding_dim = 12;
      config.spine_dim = 24;
      config.core_only = true;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = false;
      break;
    case ModelPreset::structural_core_blind:
    case ModelPreset::core_content_blind:
      config.embedding_dim = 12;
      config.spine_dim = 24;
      config.core_only = true;
      config.spine_reads_embedding = false;
      config.spine_reads_workspace = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = false;
      break;
    case ModelPreset::kv_full_capacity:
    case ModelPreset::kv_oracle_retention:
    case ModelPreset::kv_fifo_eviction:
    case ModelPreset::kv_reservoir:
    case ModelPreset::kv_hard_retention:
    case ModelPreset::kv_delayed_oracle:
    case ModelPreset::kv_delayed_fifo:
    case ModelPreset::kv_delayed_reservoir:
    case ModelPreset::kv_delayed_hard:
    case ModelPreset::kv_delayed_annealed_direct:
    case ModelPreset::kv_delayed_annealed_curriculum:
    case ModelPreset::kv_annealed_retention: {
      config.embedding_dim = 1;
      config.spine_dim = 1;
      config.mechanism_count = 7;
      config.mechanism_dim = 1;
      config.workspace_slots =
          preset == ModelPreset::kv_full_capacity ? 6 : 3;
      config.key_dim = 8;
      config.value_dim = 8;
      config.workspace_dim = config.key_dim + config.value_dim;
      config.active_mechanisms = 1;
      config.workspace_writers = 1;
      config.broadcast_recipients = 1;
      config.spine_reads_embedding = false;
      config.spine_reads_workspace = false;
      config.output_reads_spine = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = true;
      config.mediation_binding_count = 6;
      config.entity_count = 9;
      config.value_count = output_classes;
      config.context_count = 3;
      config.key_value_logit_scale = 6.0;
      config.key_value_mode = KeyValueMediationMode::learned_tied;
      config.key_value_write_routing = KeyValueWriteRoutingMode::first_free;
      const bool phase9 =
          preset == ModelPreset::kv_delayed_oracle ||
          preset == ModelPreset::kv_delayed_fifo ||
          preset == ModelPreset::kv_delayed_reservoir ||
          preset == ModelPreset::kv_delayed_hard ||
          preset == ModelPreset::kv_delayed_annealed_direct ||
          preset == ModelPreset::kv_delayed_annealed_curriculum;
      config.key_value_router_anneal_steps = phase9 ? 1350 : 900;
      if (preset == ModelPreset::kv_full_capacity) {
        config.key_value_retention = KeyValueRetentionMode::full_capacity;
      } else if (preset == ModelPreset::kv_oracle_retention ||
                 preset == ModelPreset::kv_delayed_oracle) {
        config.key_value_retention = KeyValueRetentionMode::oracle;
      } else if (preset == ModelPreset::kv_fifo_eviction ||
                 preset == ModelPreset::kv_delayed_fifo) {
        config.key_value_retention = KeyValueRetentionMode::fifo;
      } else if (preset == ModelPreset::kv_reservoir ||
                 preset == ModelPreset::kv_delayed_reservoir) {
        config.key_value_retention = KeyValueRetentionMode::reservoir;
      } else if (preset == ModelPreset::kv_hard_retention ||
                 preset == ModelPreset::kv_delayed_hard) {
        config.key_value_retention = KeyValueRetentionMode::hard_learned;
      } else {
        config.key_value_retention = KeyValueRetentionMode::annealed_learned;
      }
      break;
    }
    case ModelPreset::structural_kv_exact:
    case ModelPreset::structural_kv_learned:
    case ModelPreset::kv_fixed_position:
    case ModelPreset::kv_first_free:
    case ModelPreset::kv_hard_router:
    case ModelPreset::kv_annealed_router:
      config.embedding_dim = 1;
      config.spine_dim = 1;
      config.mechanism_count = 4;
      config.mechanism_dim = 1;
      config.workspace_slots = 3;
      config.key_dim = 8;
      config.value_dim = 8;
      config.workspace_dim = config.key_dim + config.value_dim;
      config.active_mechanisms = 1;
      config.workspace_writers = 1;
      config.broadcast_recipients = 1;
      config.spine_reads_embedding = false;
      config.spine_reads_workspace = false;
      config.output_reads_spine = false;
      config.output_reads_workspace = false;
      config.output_reads_mechanism = true;
      config.mediation_binding_count = 3;
      config.entity_count = 6;
      config.value_count = output_classes;
      config.key_value_logit_scale = 6.0;
      config.key_value_mode =
          preset == ModelPreset::structural_kv_exact
              ? KeyValueMediationMode::symbolic_exact
              : KeyValueMediationMode::learned_tied;
      if (preset == ModelPreset::kv_first_free) {
        config.key_value_write_routing = KeyValueWriteRoutingMode::first_free;
      } else if (preset == ModelPreset::kv_hard_router) {
        config.key_value_write_routing = KeyValueWriteRoutingMode::hard_learned;
      } else if (preset == ModelPreset::kv_annealed_router) {
        config.key_value_write_routing =
            KeyValueWriteRoutingMode::annealed_learned;
      } else {
        config.key_value_write_routing =
            KeyValueWriteRoutingMode::fixed_position;
      }
      break;
    case ModelPreset::structural_mediation_linear:
    case ModelPreset::structural_mediation_bounded:
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
      if (preset == ModelPreset::structural_mediation_bounded) {
        config.output_logit_bound = 1.0;
      }
      break;
  }
  config.validate();
  return config;
}

}  // namespace sgw
