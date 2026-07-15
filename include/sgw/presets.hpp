#pragma once

#include "sgw/config.hpp"

#include <cstddef>
#include <string_view>

namespace sgw {

enum class ModelPreset {
  core_small,
  core_compute_matched,
  core_param_matched,
  sgw,
};

[[nodiscard]] ModelPreset parse_model_preset(std::string_view value);
[[nodiscard]] std::string_view model_preset_name(ModelPreset preset) noexcept;
[[nodiscard]] ModelConfig make_model_config(ModelPreset preset,
                                            std::size_t vocab_size = 17,
                                            std::size_t output_classes = 5);

}  // namespace sgw
