#include "sgw/autodiff.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sgw::ad {
namespace {

void require_finite(double value, const char* context) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(context);
  }
}

}  // namespace

Tape::Tape() = default;
Tape::~Tape() = default;
Tape::Tape(Tape&&) noexcept = default;
Tape& Tape::operator=(Tape&&) noexcept = default;

bool Var::valid() const noexcept { return tape_ != nullptr; }

double Var::value() const {
  if (tape_ == nullptr) {
    throw std::logic_error("cannot read an invalid autodiff variable");
  }
  return tape_->value_of(*this);
}

Var Tape::constant(double value) {
  require_finite(value, "constant value must be finite");
  nodes_.push_back(Node{value, 0.0, Op::constant, 0, 0, nullptr});
  return Var(this, nodes_.size() - 1);
}

Var Tape::parameter(double value, double* external_gradient) {
  require_finite(value, "parameter value must be finite");
  if (external_gradient == nullptr) {
    throw std::invalid_argument("parameter gradient pointer must not be null");
  }
  nodes_.push_back(
      Node{value, 0.0, Op::parameter, 0, 0, external_gradient});
  return Var(this, nodes_.size() - 1);
}

std::size_t Tape::node_count() const noexcept { return nodes_.size(); }

void Tape::require_owned(Var variable) const {
  if (variable.tape_ != this || variable.index_ >= nodes_.size()) {
    throw std::invalid_argument("autodiff variable belongs to another tape");
  }
}

double Tape::value_of(Var variable) const {
  require_owned(variable);
  return nodes_[variable.index_].value;
}

Var Tape::unary(Op op, Var input, double value) {
  require_owned(input);
  require_finite(value, "autodiff unary operation produced a non-finite value");
  nodes_.push_back(Node{value, 0.0, op, input.index_, 0, nullptr});
  return Var(this, nodes_.size() - 1);
}

Var Tape::binary(Op op, Var lhs, Var rhs, double value) {
  require_owned(lhs);
  require_owned(rhs);
  require_finite(value, "autodiff binary operation produced a non-finite value");
  nodes_.push_back(
      Node{value, 0.0, op, lhs.index_, rhs.index_, nullptr});
  return Var(this, nodes_.size() - 1);
}

void Tape::backward(Var loss) {
  require_owned(loss);
  for (Node& node : nodes_) {
    node.gradient = 0.0;
  }
  nodes_[loss.index_].gradient = 1.0;

  for (std::size_t reverse = nodes_.size(); reverse > 0; --reverse) {
    Node& node = nodes_[reverse - 1];
    const double gradient = node.gradient;
    require_finite(gradient, "non-finite gradient during backward pass");

    switch (node.op) {
      case Op::constant:
        break;
      case Op::parameter:
        *node.external_gradient += gradient;
        require_finite(*node.external_gradient,
                       "non-finite external parameter gradient");
        break;
      case Op::add:
        nodes_[node.lhs].gradient += gradient;
        nodes_[node.rhs].gradient += gradient;
        break;
      case Op::subtract:
        nodes_[node.lhs].gradient += gradient;
        nodes_[node.rhs].gradient -= gradient;
        break;
      case Op::multiply:
        nodes_[node.lhs].gradient += gradient * nodes_[node.rhs].value;
        nodes_[node.rhs].gradient += gradient * nodes_[node.lhs].value;
        break;
      case Op::divide: {
        const double denominator = nodes_[node.rhs].value;
        nodes_[node.lhs].gradient += gradient / denominator;
        nodes_[node.rhs].gradient -=
            gradient * nodes_[node.lhs].value /
            (denominator * denominator);
        break;
      }
      case Op::negate:
        nodes_[node.lhs].gradient -= gradient;
        break;
      case Op::exponential:
        nodes_[node.lhs].gradient += gradient * node.value;
        break;
      case Op::logarithm:
        nodes_[node.lhs].gradient += gradient / nodes_[node.lhs].value;
        break;
      case Op::hyperbolic_tangent:
        nodes_[node.lhs].gradient += gradient * (1.0 - node.value * node.value);
        break;
      case Op::logistic:
        nodes_[node.lhs].gradient += gradient * node.value * (1.0 - node.value);
        break;
      case Op::rectified_linear:
        if (nodes_[node.lhs].value > 0.0) {
          nodes_[node.lhs].gradient += gradient;
        }
        break;
    }
  }
}

Var operator+(Var lhs, Var rhs) {
  if (lhs.tape_ == nullptr || lhs.tape_ != rhs.tape_) {
    throw std::invalid_argument("addition requires variables on one tape");
  }
  return lhs.tape_->binary(Tape::Op::add, lhs, rhs,
                           lhs.value() + rhs.value());
}

Var operator-(Var lhs, Var rhs) {
  if (lhs.tape_ == nullptr || lhs.tape_ != rhs.tape_) {
    throw std::invalid_argument("subtraction requires variables on one tape");
  }
  return lhs.tape_->binary(Tape::Op::subtract, lhs, rhs,
                           lhs.value() - rhs.value());
}

Var operator*(Var lhs, Var rhs) {
  if (lhs.tape_ == nullptr || lhs.tape_ != rhs.tape_) {
    throw std::invalid_argument("multiplication requires variables on one tape");
  }
  return lhs.tape_->binary(Tape::Op::multiply, lhs, rhs,
                           lhs.value() * rhs.value());
}

Var operator/(Var lhs, Var rhs) {
  if (lhs.tape_ == nullptr || lhs.tape_ != rhs.tape_) {
    throw std::invalid_argument("division requires variables on one tape");
  }
  if (rhs.value() == 0.0) {
    throw std::domain_error("division by zero");
  }
  return lhs.tape_->binary(Tape::Op::divide, lhs, rhs,
                           lhs.value() / rhs.value());
}

Var operator-(Var input) {
  if (input.tape_ == nullptr) {
    throw std::invalid_argument("negation requires a valid variable");
  }
  return input.tape_->unary(Tape::Op::negate, input, -input.value());
}

Var exp(Var input) {
  if (input.tape_ == nullptr) {
    throw std::invalid_argument("exp requires a valid variable");
  }
  return input.tape_->unary(Tape::Op::exponential, input,
                            std::exp(input.value()));
}

Var log(Var input) {
  if (input.tape_ == nullptr) {
    throw std::invalid_argument("log requires a valid variable");
  }
  if (input.value() <= 0.0) {
    throw std::domain_error("log input must be positive");
  }
  return input.tape_->unary(Tape::Op::logarithm, input,
                            std::log(input.value()));
}

Var tanh(Var input) {
  if (input.tape_ == nullptr) {
    throw std::invalid_argument("tanh requires a valid variable");
  }
  return input.tape_->unary(Tape::Op::hyperbolic_tangent, input,
                            std::tanh(input.value()));
}

Var sigmoid(Var input) {
  if (input.tape_ == nullptr) {
    throw std::invalid_argument("sigmoid requires a valid variable");
  }
  const double value = input.value();
  const double output = value >= 0.0
                            ? 1.0 / (1.0 + std::exp(-value))
                            : std::exp(value) / (1.0 + std::exp(value));
  return input.tape_->unary(Tape::Op::logistic, input, output);
}

Var relu(Var input) {
  if (input.tape_ == nullptr) {
    throw std::invalid_argument("relu requires a valid variable");
  }
  return input.tape_->unary(Tape::Op::rectified_linear, input,
                            input.value() > 0.0 ? input.value() : 0.0);
}

}  // namespace sgw::ad
