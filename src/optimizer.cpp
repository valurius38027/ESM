#include "sgw/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sgw {

Adam::Adam(AdamConfig config) : config_(config) { config_.validate(); }

void Adam::step(ParameterSet& parameters) {
  const double norm = parameters.global_grad_norm();
  last_clip_scale_ =
      norm > config_.max_grad_norm ? config_.max_grad_norm / norm : 1.0;
  ++step_count_;

  const double beta1_power =
      std::pow(config_.beta1, static_cast<double>(step_count_));
  const double beta2_power =
      std::pow(config_.beta2, static_cast<double>(step_count_));
  const double first_correction = 1.0 - beta1_power;
  const double second_correction = 1.0 - beta2_power;

  for (auto& parameter_pointer : parameters.parameters()) {
    Parameter& parameter = *parameter_pointer;
    for (std::size_t index = 0; index < parameter.size(); ++index) {
      const double gradient = parameter.gradient(index) * last_clip_scale_;
      double& first = parameter.first_moment(index);
      double& second = parameter.second_moment(index);
      first = config_.beta1 * first + (1.0 - config_.beta1) * gradient;
      second = config_.beta2 * second +
               (1.0 - config_.beta2) * gradient * gradient;
      const double corrected_first = first / first_correction;
      const double corrected_second = second / second_correction;
      parameter.value(index) -=
          config_.learning_rate * corrected_first /
          (std::sqrt(corrected_second) + config_.epsilon);
    }
  }

  if (!parameters.all_finite()) {
    throw std::runtime_error("Adam update produced non-finite parameter state");
  }
}

std::size_t Adam::step_count() const noexcept { return step_count_; }
double Adam::last_clip_scale() const noexcept { return last_clip_scale_; }

}  // namespace sgw
