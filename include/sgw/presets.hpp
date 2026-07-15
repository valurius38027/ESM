#pragma once

#include "sgw/config.hpp"

#include <cstddef>
#include <string_view>

namespace sgw {

enum class ExperimentCondition {
  core_small,
  core_compute_matched,
  core_param_matched,
  sgw_redundant_final_only,
  sgw_broadcast_forced_final_only,
  sgw_broadcast_forced_aux_annealed,
};

enum class ModelPreset {
  core_small,
  core_compute_matched,
  core_param_matched,
  sgw_redundant,
  sgw_broadcast_forced,
  sgw = sgw_redundant,
};

[[nodiscard]] ExperimentCondition parse_experiment_condition(
    std::string_view value);
[[nodiscard]] std::string_view experiment_condition_name(
    ExperimentCondition condition) noexcept;
[[nodiscard]] ModelPreset model_preset_for_condition(
    ExperimentCondition condition) noexcept;
[[nodiscard]] double condition_workspace_aux_weight(
    ExperimentCondition condition) noexcept;
[[nodiscard]] std::size_t condition_workspace_aux_anneal_steps(
    ExperimentCondition condition) noexcept;

[[nodiscard]] ModelPreset parse_model_preset(std::string_view value);
[[nodiscard]] std::string_view model_preset_name(ModelPreset preset) noexcept;
[[nodiscard]] ModelConfig make_model_config(ModelPreset preset,
                                            std::size_t vocab_size = 17,
                                            std::size_t output_classes = 5);

}  // namespace sgw
