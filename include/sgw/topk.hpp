#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace sgw {

[[nodiscard]] std::vector<std::size_t> stable_topk(
    std::span<const double> scores,
    std::size_t k);

[[nodiscard]] std::vector<double> selected_softmax(
    std::span<const double> scores,
    std::span<const std::size_t> selected);

}  // namespace sgw
