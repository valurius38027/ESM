#pragma once

#include <cstddef>
#include <vector>

namespace sgw::ad {

class Tape;

class Var {
 public:
  Var() = default;

  [[nodiscard]] double value() const;
  [[nodiscard]] bool valid() const noexcept;

 private:
  friend class Tape;
  friend Var operator+(Var, Var);
  friend Var operator-(Var, Var);
  friend Var operator*(Var, Var);
  friend Var operator/(Var, Var);
  friend Var operator-(Var);
  friend Var exp(Var);
  friend Var log(Var);
  friend Var tanh(Var);
  friend Var sigmoid(Var);
  friend Var relu(Var);

  Var(Tape* tape, std::size_t index) : tape_(tape), index_(index) {}

  Tape* tape_{nullptr};
  std::size_t index_{0};
};

class Tape {
 public:
  Tape();
  ~Tape();
  Tape(const Tape&) = delete;
  Tape& operator=(const Tape&) = delete;
  Tape(Tape&&) noexcept;
  Tape& operator=(Tape&&) noexcept;

  [[nodiscard]] Var constant(double value);
  [[nodiscard]] Var parameter(double value, double* external_gradient);
  void backward(Var loss);
  [[nodiscard]] std::size_t node_count() const noexcept;

 private:
  friend class Var;
  friend Var operator+(Var, Var);
  friend Var operator-(Var, Var);
  friend Var operator*(Var, Var);
  friend Var operator/(Var, Var);
  friend Var operator-(Var);
  friend Var exp(Var);
  friend Var log(Var);
  friend Var tanh(Var);
  friend Var sigmoid(Var);
  friend Var relu(Var);

  enum class Op {
    constant,
    parameter,
    add,
    subtract,
    multiply,
    divide,
    negate,
    exponential,
    logarithm,
    hyperbolic_tangent,
    logistic,
    rectified_linear,
  };

  struct Node {
    double value{0.0};
    double gradient{0.0};
    Op op{Op::constant};
    std::size_t lhs{0};
    std::size_t rhs{0};
    double* external_gradient{nullptr};
  };

  [[nodiscard]] Var unary(Op op, Var input, double value);
  [[nodiscard]] Var binary(Op op, Var lhs, Var rhs, double value);
  [[nodiscard]] double value_of(Var variable) const;
  void require_owned(Var variable) const;

  std::vector<Node> nodes_;
};

[[nodiscard]] Var operator+(Var lhs, Var rhs);
[[nodiscard]] Var operator-(Var lhs, Var rhs);
[[nodiscard]] Var operator*(Var lhs, Var rhs);
[[nodiscard]] Var operator/(Var lhs, Var rhs);
[[nodiscard]] Var operator-(Var input);
[[nodiscard]] Var exp(Var input);
[[nodiscard]] Var log(Var input);
[[nodiscard]] Var tanh(Var input);
[[nodiscard]] Var sigmoid(Var input);
[[nodiscard]] Var relu(Var input);

}  // namespace sgw::ad
