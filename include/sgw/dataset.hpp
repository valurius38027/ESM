#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgw {

struct BindingTaskConfig {
  std::size_t entity_count{6};
  std::size_t value_count{6};
  std::size_t filler_count{4};
  std::size_t binding_count{3};
  std::size_t fillers_per_binding{1};

  void validate() const;
  [[nodiscard]] std::size_t sequence_length() const;
};

struct BindingVocabulary {
  std::size_t entity_count{0};
  std::size_t value_count{0};
  std::size_t filler_count{0};

  [[nodiscard]] int entity_token(std::size_t entity) const;
  [[nodiscard]] int value_token(std::size_t value) const;
  [[nodiscard]] int filler_token(std::size_t filler) const;
  [[nodiscard]] int query_token() const;
  [[nodiscard]] std::size_t size() const;
};

struct BindingSample {
  std::vector<int> tokens;
  std::size_t target_class{0};
  std::size_t queried_entity{0};
  std::size_t source_entity_position{0};
  std::size_t source_value_position{0};
  std::size_t query_entity_position{0};
};

struct BindingDataset {
  BindingTaskConfig config;
  BindingVocabulary vocabulary;
  std::vector<BindingSample> train;
  std::vector<BindingSample> holdout;
};

[[nodiscard]] BindingDataset make_binding_split(
    const BindingTaskConfig& config,
    std::size_t train_count,
    std::size_t holdout_count,
    std::uint64_t seed);

}  // namespace sgw
