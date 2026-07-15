#include "test_harness.hpp"

#include "sgw/topk.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

SGW_TEST(stable_topk_orders_descending_scores) {
  constexpr std::array<double, 5> scores{0.1, 0.9, -0.2, 0.7, 0.4};
  const auto selected = sgw::stable_topk(scores, 3);
  SGW_REQUIRE((selected == std::vector<std::size_t>{1, 3, 4}));
}

SGW_TEST(stable_topk_breaks_ties_by_lower_index) {
  constexpr std::array<double, 4> scores{0.5, 0.8, 0.8, 0.8};
  const auto selected = sgw::stable_topk(scores, 2);
  SGW_REQUIRE((selected == std::vector<std::size_t>{1, 2}));
}

SGW_TEST(stable_topk_rejects_invalid_budget_and_nonfinite_scores) {
  constexpr std::array<double, 2> scores{0.1, 0.2};
  SGW_REQUIRE_THROWS(sgw::stable_topk(scores, 0));
  SGW_REQUIRE_THROWS(sgw::stable_topk(scores, 3));

  constexpr std::array<double, 2> invalid{
      0.1, std::numeric_limits<double>::infinity()};
  SGW_REQUIRE_THROWS(sgw::stable_topk(invalid, 1));
}

SGW_TEST(selected_softmax_normalizes_only_selected_entries) {
  constexpr std::array<double, 4> scores{1.0, 2.0, 3.0, 4.0};
  constexpr std::array<std::size_t, 2> selected{1, 3};
  const auto weights = sgw::selected_softmax(scores, selected);
  SGW_REQUIRE(weights.size() == 2);
  SGW_REQUIRE_NEAR(weights[0] + weights[1], 1.0, 1.0e-12);
  SGW_REQUIRE_NEAR(weights[0], 1.0 / (1.0 + std::exp(2.0)), 1.0e-12);
  SGW_REQUIRE_NEAR(weights[1], std::exp(2.0) / (1.0 + std::exp(2.0)),
                   1.0e-12);
}

SGW_TEST(selected_softmax_rejects_duplicate_or_invalid_indices) {
  constexpr std::array<double, 3> scores{1.0, 2.0, 3.0};
  constexpr std::array<std::size_t, 2> duplicate{1, 1};
  constexpr std::array<std::size_t, 1> invalid{3};
  constexpr std::array<std::size_t, 0> empty{};
  SGW_REQUIRE_THROWS(sgw::selected_softmax(scores, duplicate));
  SGW_REQUIRE_THROWS(sgw::selected_softmax(scores, invalid));
  SGW_REQUIRE_THROWS(sgw::selected_softmax(scores, empty));
}
