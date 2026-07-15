#include "sgw/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace sgw {

Parameter::Parameter(std::string name, std::size_t rows, std::size_t columns)
    : name_(std::move(name)),
      rows_(rows),
      columns_(columns),
      values_(rows * columns, 0.0),
      gradients_(rows * columns, 0.0),
      first_moments_(rows * columns, 0.0),
      second_moments_(rows * columns, 0.0) {
  if (name_.empty()) {
    throw std::invalid_argument("parameter name must not be empty");
  }
  if (rows_ == 0 || columns_ == 0) {
    throw std::invalid_argument("parameter dimensions must be positive");
  }
  if (rows_ > std::numeric_limits<std::size_t>::max() / columns_) {
    throw std::overflow_error("parameter shape overflows size_t");
  }
}

const std::string& Parameter::name() const noexcept { return name_; }
std::size_t Parameter::rows() const noexcept { return rows_; }
std::size_t Parameter::columns() const noexcept { return columns_; }
std::size_t Parameter::size() const noexcept { return values_.size(); }

std::size_t Parameter::offset(std::size_t row, std::size_t column) const {
  if (row >= rows_ || column >= columns_) {
    throw std::out_of_range("parameter matrix index out of range");
  }
  return row * columns_ + column;
}

double& Parameter::value(std::size_t row, std::size_t column) {
  return values_.at(offset(row, column));
}

const double& Parameter::value(std::size_t row, std::size_t column) const {
  return values_.at(offset(row, column));
}

double& Parameter::value(std::size_t index) { return values_.at(index); }
const double& Parameter::value(std::size_t index) const {
  return values_.at(index);
}
double& Parameter::gradient(std::size_t index) {
  return gradients_.at(index);
}
const double& Parameter::gradient(std::size_t index) const {
  return gradients_.at(index);
}
double& Parameter::first_moment(std::size_t index) {
  return first_moments_.at(index);
}
double& Parameter::second_moment(std::size_t index) {
  return second_moments_.at(index);
}

std::span<const double> Parameter::values() const noexcept { return values_; }
std::span<double> Parameter::mutable_values() noexcept { return values_; }
std::span<const double> Parameter::gradients() const noexcept {
  return gradients_;
}

void Parameter::zero_grad() noexcept {
  std::fill(gradients_.begin(), gradients_.end(), 0.0);
}

bool Parameter::all_finite() const noexcept {
  const auto finite = [](double value) { return std::isfinite(value); };
  return std::all_of(values_.begin(), values_.end(), finite) &&
         std::all_of(gradients_.begin(), gradients_.end(), finite) &&
         std::all_of(first_moments_.begin(), first_moments_.end(), finite) &&
         std::all_of(second_moments_.begin(), second_moments_.end(), finite);
}

Parameter& ParameterSet::add_zeros(std::string name, std::size_t rows,
                                   std::size_t columns) {
  for (const auto& parameter : parameters_) {
    if (parameter->name() == name) {
      throw std::invalid_argument("duplicate parameter name: " + name);
    }
  }
  parameters_.push_back(
      std::make_unique<Parameter>(std::move(name), rows, columns));
  return *parameters_.back();
}

Parameter& ParameterSet::add_xavier(std::string name, std::size_t rows,
                                    std::size_t columns,
                                    std::mt19937_64& generator) {
  Parameter& parameter = add_zeros(std::move(name), rows, columns);
  const double denominator = static_cast<double>(rows + columns);
  const double bound = std::sqrt(6.0 / denominator);
  std::uniform_real_distribution<double> distribution(-bound, bound);
  for (double& value : parameter.mutable_values()) {
    value = distribution(generator);
  }
  return parameter;
}

std::span<std::unique_ptr<Parameter>> ParameterSet::parameters() noexcept {
  return parameters_;
}

std::span<const std::unique_ptr<Parameter>> ParameterSet::parameters()
    const noexcept {
  return parameters_;
}

void ParameterSet::zero_grad() noexcept {
  for (auto& parameter : parameters_) {
    parameter->zero_grad();
  }
}

double ParameterSet::global_grad_norm() const {
  long double sum = 0.0L;
  for (const auto& parameter : parameters_) {
    for (const double gradient : parameter->gradients()) {
      if (!std::isfinite(gradient)) {
        throw std::runtime_error("non-finite gradient in parameter set");
      }
      const long double promoted = static_cast<long double>(gradient);
      sum += promoted * promoted;
    }
  }
  return std::sqrt(static_cast<double>(sum));
}

bool ParameterSet::all_finite() const noexcept {
  return std::all_of(parameters_.begin(), parameters_.end(),
                     [](const auto& parameter) {
                       return parameter->all_finite();
                     });
}

std::size_t ParameterSet::scalar_count() const noexcept {
  std::size_t count = 0;
  for (const auto& parameter : parameters_) {
    count += parameter->size();
  }
  return count;
}

}  // namespace sgw
