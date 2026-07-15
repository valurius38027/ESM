#include "test_harness.hpp"

#include "sgw/autodiff.hpp"

#include <cmath>

namespace {

double reference_expression(double x, double y) {
  const double inner = std::exp(x * y) + 1.0 / (1.0 + std::exp(-x)) -
                       std::tanh(y) + 2.0;
  return std::log(inner);
}

}  // namespace

SGW_TEST(autodiff_matches_finite_difference_for_composed_expression) {
  constexpr double x_value = 0.7;
  constexpr double y_value = -0.4;
  double x_gradient = 0.0;
  double y_gradient = 0.0;

  sgw::ad::Tape tape;
  const auto x = tape.parameter(x_value, &x_gradient);
  const auto y = tape.parameter(y_value, &y_gradient);
  const auto two = tape.constant(2.0);
  const auto loss = sgw::ad::log(sgw::ad::exp(x * y) + sgw::ad::sigmoid(x) -
                                 sgw::ad::tanh(y) + two);
  tape.backward(loss);

  constexpr double epsilon = 1.0e-6;
  const double numeric_x =
      (reference_expression(x_value + epsilon, y_value) -
       reference_expression(x_value - epsilon, y_value)) /
      (2.0 * epsilon);
  const double numeric_y =
      (reference_expression(x_value, y_value + epsilon) -
       reference_expression(x_value, y_value - epsilon)) /
      (2.0 * epsilon);

  SGW_REQUIRE_NEAR(loss.value(), reference_expression(x_value, y_value),
                   1.0e-12);
  SGW_REQUIRE_NEAR(x_gradient, numeric_x, 1.0e-7);
  SGW_REQUIRE_NEAR(y_gradient, numeric_y, 1.0e-7);
}

SGW_TEST(autodiff_accumulates_repeated_parameter_references) {
  constexpr double value = 1.25;
  double gradient = 0.0;
  sgw::ad::Tape tape;
  const auto first = tape.parameter(value, &gradient);
  const auto second = tape.parameter(value, &gradient);
  const auto loss = first * second + first;
  tape.backward(loss);
  SGW_REQUIRE_NEAR(gradient, 2.0 * value + 1.0, 1.0e-12);
}

SGW_TEST(autodiff_supports_relu_division_and_negation) {
  double x_gradient = 0.0;
  double y_gradient = 0.0;
  sgw::ad::Tape tape;
  const auto x = tape.parameter(3.0, &x_gradient);
  const auto y = tape.parameter(2.0, &y_gradient);
  const auto loss = sgw::ad::relu(x / y) + (-y);
  tape.backward(loss);
  SGW_REQUIRE_NEAR(loss.value(), -0.5, 1.0e-12);
  SGW_REQUIRE_NEAR(x_gradient, 0.5, 1.0e-12);
  SGW_REQUIRE_NEAR(y_gradient, -1.75, 1.0e-12);
}

SGW_TEST(autodiff_rejects_cross_tape_operations_and_invalid_log) {
  sgw::ad::Tape first_tape;
  sgw::ad::Tape second_tape;
  const auto first = first_tape.constant(1.0);
  const auto second = second_tape.constant(2.0);
  SGW_REQUIRE_THROWS(first + second);
  SGW_REQUIRE_THROWS(sgw::ad::log(first_tape.constant(0.0)));
}
