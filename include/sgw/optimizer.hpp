#pragma once

#include "sgw/config.hpp"
#include "sgw/tensor.hpp"

#include <cstddef>

namespace sgw {

class Adam {
 public:
  explicit Adam(AdamConfig config);

  void step(ParameterSet& parameters);
  [[nodiscard]] std::size_t step_count() const noexcept;
  [[nodiscard]] double last_clip_scale() const noexcept;

 private:
  AdamConfig config_;
  std::size_t step_count_{0};
  double last_clip_scale_{1.0};
};

}  // namespace sgw
