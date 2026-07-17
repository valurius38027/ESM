#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace sgw {

enum class TokenRole : std::size_t {
  entity = 0,
  value = 1,
  filler = 2,
  query_marker = 3,
  query_entity = 4,
  context = 5,
  count = 6,
};

[[nodiscard]] std::string_view token_role_name(TokenRole role) noexcept;

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
  std::size_t context_count{0};

  [[nodiscard]] int entity_token(std::size_t entity) const;
  [[nodiscard]] int value_token(std::size_t value) const;
  [[nodiscard]] int filler_token(std::size_t filler) const;
  [[nodiscard]] int query_token() const;
  [[nodiscard]] int context_token(std::size_t context) const;
  [[nodiscard]] std::size_t size() const;
};

struct BindingSample {
  std::vector<int> tokens;
  std::vector<TokenRole> roles;
  std::size_t target_class{0};
  std::size_t queried_entity{0};
  std::size_t source_entity_position{0};
  std::size_t source_value_position{0};
  std::size_t query_entity_position{0};
  std::size_t context_class{0};
  std::vector<bool> binding_relevance;
  std::vector<bool> binding_is_delay_distractor;
  std::size_t delay_binding_count{0};
  std::size_t source_query_distance{0};
};

struct BindingDataset {
  BindingTaskConfig config;
  BindingVocabulary vocabulary;
  std::vector<BindingSample> train;
  std::vector<BindingSample> holdout;
};

struct RetentionTaskConfig {
  std::size_t context_count{3};
  std::size_t entity_count{9};
  std::size_t value_count{6};
  std::size_t binding_count{6};
  std::size_t relevant_binding_count{3};
  std::size_t workspace_slots{3};
  std::size_t delay_binding_count{0};

  void validate() const;
  [[nodiscard]] std::size_t total_binding_count() const;
  [[nodiscard]] std::size_t sequence_length() const;
  [[nodiscard]] bool is_relevant(std::size_t context,
                                 std::size_t entity) const;
  [[nodiscard]] BindingTaskConfig binding_config() const;
};

struct RetentionDataset {
  RetentionTaskConfig config;
  BindingVocabulary vocabulary;
  std::vector<BindingSample> train;
  std::vector<BindingSample> holdout;
};

class RetentionBindingStream {
 public:
  RetentionBindingStream(const RetentionTaskConfig& config,
                         std::uint64_t seed,
                         std::size_t sample_count);
  RetentionBindingStream(const RetentionTaskConfig& config,
                         std::uint64_t seed,
                         std::span<const std::size_t> delay_schedule);

  [[nodiscard]] BindingSample next();
  [[nodiscard]] std::size_t samples_consumed() const noexcept;
  [[nodiscard]] std::size_t remaining() const noexcept;
  [[nodiscard]] std::size_t total_samples() const noexcept;

 private:
  RetentionTaskConfig config_;
  BindingVocabulary vocabulary_;
  std::vector<BindingSample> samples_;
  std::size_t cursor_{0};
};

[[nodiscard]] RetentionDataset make_retention_binding_split(
    const RetentionTaskConfig& config,
    std::size_t train_count,
    std::size_t holdout_count,
    std::uint64_t seed);

[[nodiscard]] bool is_structural_holdout_pair(
    const BindingTaskConfig& config,
    std::size_t entity,
    std::size_t value);

class StructuralBindingStream {
 public:
  StructuralBindingStream(const BindingTaskConfig& config,
                          std::uint64_t seed);

  [[nodiscard]] BindingSample next();
  [[nodiscard]] std::size_t samples_consumed() const noexcept;
  [[nodiscard]] std::size_t remaining() const noexcept;
  [[nodiscard]] std::size_t total_samples() const noexcept;

 private:
  BindingTaskConfig config_;
  BindingVocabulary vocabulary_;
  std::vector<BindingSample> samples_;
  std::size_t cursor_{0};
};

[[nodiscard]] BindingDataset make_structural_binding_split(
    const BindingTaskConfig& config,
    std::size_t train_count,
    std::size_t holdout_count,
    std::uint64_t seed);

[[nodiscard]] BindingDataset make_binding_split(
    const BindingTaskConfig& config,
    std::size_t train_count,
    std::size_t holdout_count,
    std::uint64_t seed);

}  // namespace sgw
