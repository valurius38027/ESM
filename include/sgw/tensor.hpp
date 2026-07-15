#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace sgw {

class Parameter {
 public:
  Parameter(std::string name, std::size_t rows, std::size_t columns);

  [[nodiscard]] const std::string& name() const noexcept;
  [[nodiscard]] std::size_t rows() const noexcept;
  [[nodiscard]] std::size_t columns() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

  [[nodiscard]] double& value(std::size_t row, std::size_t column);
  [[nodiscard]] const double& value(std::size_t row,
                                    std::size_t column) const;
  [[nodiscard]] double& value(std::size_t index);
  [[nodiscard]] const double& value(std::size_t index) const;
  [[nodiscard]] double& gradient(std::size_t index);
  [[nodiscard]] const double& gradient(std::size_t index) const;
  [[nodiscard]] double& first_moment(std::size_t index);
  [[nodiscard]] double& second_moment(std::size_t index);

  [[nodiscard]] std::span<const double> values() const noexcept;
  [[nodiscard]] std::span<double> mutable_values() noexcept;
  [[nodiscard]] std::span<const double> gradients() const noexcept;
  void zero_grad() noexcept;
  [[nodiscard]] bool all_finite() const noexcept;

 private:
  [[nodiscard]] std::size_t offset(std::size_t row,
                                   std::size_t column) const;

  std::string name_;
  std::size_t rows_;
  std::size_t columns_;
  std::vector<double> values_;
  std::vector<double> gradients_;
  std::vector<double> first_moments_;
  std::vector<double> second_moments_;
};

class ParameterSet {
 public:
  Parameter& add_zeros(std::string name, std::size_t rows,
                       std::size_t columns);
  Parameter& add_xavier(std::string name, std::size_t rows,
                        std::size_t columns, std::mt19937_64& generator);

  [[nodiscard]] std::span<std::unique_ptr<Parameter>> parameters() noexcept;
  [[nodiscard]] std::span<const std::unique_ptr<Parameter>> parameters()
      const noexcept;
  void zero_grad() noexcept;
  [[nodiscard]] double global_grad_norm() const;
  [[nodiscard]] bool all_finite() const noexcept;
  [[nodiscard]] std::size_t scalar_count() const noexcept;

 private:
  std::vector<std::unique_ptr<Parameter>> parameters_;
};

}  // namespace sgw
