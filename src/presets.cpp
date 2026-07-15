#include "sgw/presets.hpp"

#include <stdexcept>
#include <string>

namespace sgw {

ModelPreset parse_model_preset(std::string_view value) {
  if (value == "core_small" || value == "core_only") {
    return ModelPreset::core_small;
  }
  if (value == "core_compute_matched") {
    return ModelPreset::core_compute_matched;
  }
  if (value == "core_param_matched") {
    return ModelPreset::core_param_matched;
  }
  if (value == "sgw") {
    return ModelPreset::sgw;
  }
  throw std::invalid_argument("unknown model preset: " + std::string(value));
}

std::string_view model_preset_name(ModelPreset preset) noexcept {
  switch (preset) {
    case ModelPreset::core_small:
      return "core_small";
    case ModelPreset::core_compute_matched:
      return "core_compute_matched";
    case ModelPreset::core_param_matched:
      return "core_param_matched";
    case ModelPreset::sgw:
      return "sgw";
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
      break;
    case ModelPreset::core_compute_matched:
      config.embedding_dim = 66;
      config.spine_dim = 13;
      config.core_only = true;
      break;
    case ModelPreset::core_param_matched:
      config.embedding_dim = 40;
      config.spine_dim = 22;
      config.core_only = true;
      break;
    case ModelPreset::sgw:
      config.embedding_dim = 6;
      config.spine_dim = 10;
      config.core_only = false;
      break;
  }
  config.validate();
  return config;
}

}  // namespace sgw
