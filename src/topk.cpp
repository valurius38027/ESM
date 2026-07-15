#include "sgw/topk.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace sgw {

std::vector<std::size_t> stable_topk(std::span<const double> scores,
                                     std::size_t k) {
  if (k == 0 || k > scores.size()) {
    throw std::invalid_argument("k must be in [1, scores.size()]");
  }
  for (const double score : scores) {
    if (!std::isfinite(score)) {
      throw std::invalid_argument("scores must be finite");
    }
  }

  std::vector<std::size_t> indices(scores.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  std::stable_sort(indices.begin(), indices.end(),
                   [scores](std::size_t lhs, std::size_t rhs) {
                     if (scores[lhs] == scores[rhs]) {
                       return lhs < rhs;
                     }
                     return scores[lhs] > scores[rhs];
                   });
  indices.resize(k);
  return indices;
}

std::vector<double> selected_softmax(
    std::span<const double> scores,
    std::span<const std::size_t> selected) {
  if (selected.empty()) {
    throw std::invalid_argument("selected indices must be non-empty");
  }

  std::unordered_set<std::size_t> seen;
  double maximum = -std::numeric_limits<double>::infinity();
  for (const std::size_t index : selected) {
    if (index >= scores.size()) {
      throw std::out_of_range("selected index out of range");
    }
    if (!seen.insert(index).second) {
      throw std::invalid_argument("selected indices must be unique");
    }
    if (!std::isfinite(scores[index])) {
      throw std::invalid_argument("selected scores must be finite");
    }
    maximum = std::max(maximum, scores[index]);
  }

  std::vector<double> weights;
  weights.reserve(selected.size());
  double sum = 0.0;
  for (const std::size_t index : selected) {
    const double weight = std::exp(scores[index] - maximum);
    weights.push_back(weight);
    sum += weight;
  }
  if (!std::isfinite(sum) || sum <= 0.0) {
    throw std::runtime_error("softmax normalization is not finite and positive");
  }
  for (double& weight : weights) {
    weight /= sum;
  }
  return weights;
}

}  // namespace sgw
