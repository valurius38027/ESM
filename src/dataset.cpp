#include "sgw/dataset.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace sgw {
namespace {

void require_positive(std::size_t value, const char* name) {
  if (value == 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs,
                        const char* context) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw std::overflow_error(context);
  }
  return lhs + rhs;
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs,
                             const char* context) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(context);
  }
  return lhs * rhs;
}

std::size_t bounded_random(std::mt19937_64& generator,
                           std::size_t upper_exclusive) {
  if (upper_exclusive == 0) {
    throw std::invalid_argument("random bound must be positive");
  }
  const std::uint64_t bound = static_cast<std::uint64_t>(upper_exclusive);
  const std::uint64_t threshold =
      static_cast<std::uint64_t>(-bound) % bound;
  while (true) {
    const std::uint64_t value = generator();
    if (value >= threshold) {
      return static_cast<std::size_t>(value % bound);
    }
  }
}

void deterministic_shuffle(std::vector<std::size_t>& values,
                           std::mt19937_64& generator) {
  for (std::size_t remaining = values.size(); remaining > 1; --remaining) {
    const std::size_t selected = bounded_random(generator, remaining);
    std::swap(values[remaining - 1], values[selected]);
  }
}

std::string literal_key(const std::vector<int>& tokens) {
  std::ostringstream stream;
  for (const int token : tokens) {
    stream << token << ',';
  }
  return stream.str();
}

BindingSample make_sample(const BindingTaskConfig& config,
                          const BindingVocabulary& vocabulary,
                          std::mt19937_64& generator) {
  std::vector<std::size_t> entities(config.entity_count);
  std::iota(entities.begin(), entities.end(), std::size_t{0});
  deterministic_shuffle(entities, generator);
  entities.resize(config.binding_count);

  const std::size_t queried_binding =
      bounded_random(generator, config.binding_count);
  BindingSample sample;
  sample.tokens.reserve(config.sequence_length());
  sample.roles.reserve(config.sequence_length());
  sample.queried_entity = entities[queried_binding];

  const std::size_t stride = 2 + config.fillers_per_binding;
  for (std::size_t binding = 0; binding < config.binding_count; ++binding) {
    const std::size_t entity_position = sample.tokens.size();
    sample.tokens.push_back(vocabulary.entity_token(entities[binding]));
    sample.roles.push_back(TokenRole::entity);
    const std::size_t value = bounded_random(generator, config.value_count);
    sample.tokens.push_back(vocabulary.value_token(value));
    sample.roles.push_back(TokenRole::value);
    for (std::size_t filler = 0; filler < config.fillers_per_binding;
         ++filler) {
      const std::size_t filler_id =
          bounded_random(generator, config.filler_count);
      sample.tokens.push_back(vocabulary.filler_token(filler_id));
      sample.roles.push_back(TokenRole::filler);
    }
    if (binding == queried_binding) {
      sample.target_class = value;
      sample.source_entity_position = entity_position;
      sample.source_value_position = entity_position + 1;
    }
  }

  if (sample.tokens.size() != config.binding_count * stride) {
    throw std::logic_error("binding sample body has incorrect length");
  }
  sample.tokens.push_back(vocabulary.query_token());
  sample.roles.push_back(TokenRole::query_marker);
  sample.query_entity_position = sample.tokens.size();
  sample.tokens.push_back(vocabulary.entity_token(sample.queried_entity));
  sample.roles.push_back(TokenRole::query_entity);
  if (sample.roles.size() != sample.tokens.size()) {
    throw std::logic_error("binding sample role sequence has incorrect length");
  }
  return sample;
}


BindingSample make_explicit_sample(
    const BindingTaskConfig& config,
    const BindingVocabulary& vocabulary,
    const std::vector<std::size_t>& entities,
    const std::vector<std::size_t>& values,
    std::size_t queried_binding) {
  if (entities.size() != config.binding_count ||
      values.size() != config.binding_count ||
      queried_binding >= config.binding_count) {
    throw std::invalid_argument("invalid explicit binding sample");
  }
  BindingSample sample;
  sample.tokens.reserve(config.sequence_length());
  sample.roles.reserve(config.sequence_length());
  sample.queried_entity = entities[queried_binding];
  for (std::size_t binding = 0; binding < config.binding_count; ++binding) {
    const std::size_t entity_position = sample.tokens.size();
    sample.tokens.push_back(vocabulary.entity_token(entities[binding]));
    sample.roles.push_back(TokenRole::entity);
    sample.tokens.push_back(vocabulary.value_token(values[binding]));
    sample.roles.push_back(TokenRole::value);
    for (std::size_t filler = 0; filler < config.fillers_per_binding;
         ++filler) {
      sample.tokens.push_back(vocabulary.filler_token(
          (binding + filler) % config.filler_count));
      sample.roles.push_back(TokenRole::filler);
    }
    if (binding == queried_binding) {
      sample.target_class = values[binding];
      sample.source_entity_position = entity_position;
      sample.source_value_position = entity_position + 1;
    }
  }
  sample.tokens.push_back(vocabulary.query_token());
  sample.roles.push_back(TokenRole::query_marker);
  sample.query_entity_position = sample.tokens.size();
  sample.tokens.push_back(vocabulary.entity_token(sample.queried_entity));
  sample.roles.push_back(TokenRole::query_entity);
  return sample;
}

void enumerate_entity_orders_recursive(
    const BindingTaskConfig& config,
    std::vector<std::size_t>& current,
    std::vector<bool>& used,
    std::vector<std::vector<std::size_t>>& orders) {
  if (current.size() == config.binding_count) {
    orders.push_back(current);
    return;
  }
  for (std::size_t entity = 0; entity < config.entity_count; ++entity) {
    if (used[entity]) continue;
    used[entity] = true;
    current.push_back(entity);
    enumerate_entity_orders_recursive(config, current, used, orders);
    current.pop_back();
    used[entity] = false;
  }
}

std::vector<std::vector<std::size_t>> enumerate_entity_orders(
    const BindingTaskConfig& config) {
  std::vector<std::vector<std::size_t>> orders;
  std::vector<std::size_t> current;
  std::vector<bool> used(config.entity_count, false);
  enumerate_entity_orders_recursive(config, current, used, orders);
  return orders;
}

void enumerate_training_values_recursive(
    const BindingTaskConfig& config,
    const std::vector<std::size_t>& entities,
    std::vector<std::size_t>& values,
    std::vector<std::vector<std::size_t>>& assignments) {
  const std::size_t binding = values.size();
  if (binding == config.binding_count) {
    assignments.push_back(values);
    return;
  }
  for (std::size_t value = 0; value < config.value_count; ++value) {
    if (value == entities[binding] % config.value_count) continue;
    values.push_back(value);
    enumerate_training_values_recursive(config, entities, values, assignments);
    values.pop_back();
  }
}

std::vector<BindingSample> enumerate_structural_samples(
    const BindingTaskConfig& config,
    const BindingVocabulary& vocabulary,
    bool holdout) {
  std::vector<BindingSample> samples;
  for (const auto& entities : enumerate_entity_orders(config)) {
    if (!holdout) {
      std::vector<std::vector<std::size_t>> assignments;
      std::vector<std::size_t> values;
      enumerate_training_values_recursive(config, entities, values, assignments);
      for (const auto& assignment : assignments) {
        for (std::size_t query = 0; query < config.binding_count; ++query) {
          samples.push_back(make_explicit_sample(
              config, vocabulary, entities, assignment, query));
        }
      }
      continue;
    }

    for (std::size_t query = 0; query < config.binding_count; ++query) {
      std::vector<std::vector<std::size_t>> assignments;
      std::vector<std::size_t> values;
      values.reserve(config.binding_count);
      const auto recurse = [&](auto&& self, std::size_t binding) -> void {
        if (binding == config.binding_count) {
          assignments.push_back(values);
          return;
        }
        if (binding == query) {
          values.push_back(entities[binding] % config.value_count);
          self(self, binding + 1);
          values.pop_back();
          return;
        }
        for (std::size_t value = 0; value < config.value_count; ++value) {
          if (value == entities[binding] % config.value_count) continue;
          values.push_back(value);
          self(self, binding + 1);
          values.pop_back();
        }
      };
      recurse(recurse, 0);
      for (const auto& assignment : assignments) {
        samples.push_back(make_explicit_sample(
            config, vocabulary, entities, assignment, query));
      }
    }
  }
  return samples;
}


BindingSample make_retention_sample(
    const RetentionTaskConfig& config,
    const BindingVocabulary& vocabulary,
    std::mt19937_64& generator,
    bool holdout) {
  const std::size_t context = bounded_random(generator, config.context_count);
  std::vector<std::size_t> relevant;
  std::vector<std::size_t> irrelevant;
  for (std::size_t entity = 0; entity < config.entity_count; ++entity) {
    (config.is_relevant(context, entity) ? relevant : irrelevant)
        .push_back(entity);
  }
  deterministic_shuffle(irrelevant, generator);
  irrelevant.resize(config.binding_count - config.relevant_binding_count);
  std::vector<std::size_t> entities = relevant;
  entities.insert(entities.end(), irrelevant.begin(), irrelevant.end());
  deterministic_shuffle(entities, generator);

  std::vector<std::size_t> relevant_bindings;
  for (std::size_t binding = 0; binding < entities.size(); ++binding) {
    if (config.is_relevant(context, entities[binding])) {
      relevant_bindings.push_back(binding);
    }
  }
  const std::size_t queried_binding = relevant_bindings.at(
      bounded_random(generator, relevant_bindings.size()));

  BindingSample sample;
  sample.context_class = context;
  sample.queried_entity = entities[queried_binding];
  sample.tokens.reserve(config.sequence_length());
  sample.roles.reserve(config.sequence_length());
  sample.binding_relevance.reserve(config.binding_count);
  sample.binding_is_delay_distractor.reserve(config.total_binding_count());
  sample.delay_binding_count = config.delay_binding_count;
  sample.tokens.push_back(vocabulary.context_token(context));
  sample.roles.push_back(TokenRole::context);

  for (std::size_t binding = 0; binding < config.binding_count; ++binding) {
    const std::size_t entity = entities[binding];
    const std::size_t entity_position = sample.tokens.size();
    sample.tokens.push_back(vocabulary.entity_token(entity));
    sample.roles.push_back(TokenRole::entity);
    sample.binding_relevance.push_back(config.is_relevant(context, entity));
    sample.binding_is_delay_distractor.push_back(false);

    std::size_t value = bounded_random(generator, config.value_count - 1);
    const std::size_t held_out = entity % config.value_count;
    if (value >= held_out) ++value;
    if (holdout && binding == queried_binding) value = held_out;
    sample.tokens.push_back(vocabulary.value_token(value));
    sample.roles.push_back(TokenRole::value);
    if (binding == queried_binding) {
      sample.target_class = value;
      sample.source_entity_position = entity_position;
      sample.source_value_position = entity_position + 1;
    }
  }
  for (std::size_t delay = 0; delay < config.delay_binding_count; ++delay) {
    const std::size_t entity = irrelevant.at(
        bounded_random(generator, irrelevant.size()));
    sample.tokens.push_back(vocabulary.entity_token(entity));
    sample.roles.push_back(TokenRole::entity);
    sample.binding_relevance.push_back(false);
    sample.binding_is_delay_distractor.push_back(true);
    std::size_t value = bounded_random(generator, config.value_count - 1);
    const std::size_t held_out = entity % config.value_count;
    if (value >= held_out) ++value;
    sample.tokens.push_back(vocabulary.value_token(value));
    sample.roles.push_back(TokenRole::value);
  }
  sample.tokens.push_back(vocabulary.query_token());
  sample.roles.push_back(TokenRole::query_marker);
  sample.query_entity_position = sample.tokens.size();
  sample.tokens.push_back(vocabulary.entity_token(sample.queried_entity));
  sample.roles.push_back(TokenRole::query_entity);
  sample.source_query_distance =
      sample.query_entity_position - sample.source_value_position;
  return sample;
}

std::vector<BindingSample> make_unique_retention_samples(
    const RetentionTaskConfig& config,
    const BindingVocabulary& vocabulary,
    std::size_t count,
    std::uint64_t seed,
    bool holdout,
    std::unordered_set<std::string>* forbidden = nullptr) {
  std::mt19937_64 generator(seed);
  std::unordered_set<std::string> seen;
  std::vector<BindingSample> samples;
  samples.reserve(count);
  const std::size_t maximum_attempts = checked_multiply(
      std::max<std::size_t>(count, 1), 1000, "retention generation overflow");
  for (std::size_t attempt = 0;
       samples.size() < count && attempt < maximum_attempts; ++attempt) {
    BindingSample sample = make_retention_sample(
        config, vocabulary, generator, holdout);
    const std::string key = literal_key(sample.tokens);
    if (seen.contains(key) || (forbidden != nullptr && forbidden->contains(key))) {
      continue;
    }
    seen.insert(key);
    samples.push_back(std::move(sample));
  }
  if (samples.size() != count) {
    throw std::runtime_error("could not generate enough unique retention samples");
  }
  if (forbidden != nullptr) {
    forbidden->insert(seen.begin(), seen.end());
  }
  return samples;
}

void deterministic_shuffle_samples(std::vector<BindingSample>& samples,
                                   std::uint64_t seed) {
  std::mt19937_64 generator(seed);
  for (std::size_t remaining = samples.size(); remaining > 1; --remaining) {
    const std::size_t selected = bounded_random(generator, remaining);
    std::swap(samples[remaining - 1], samples[selected]);
  }
}

}  // namespace


std::string_view token_role_name(TokenRole role) noexcept {
  switch (role) {
    case TokenRole::entity:
      return "entity";
    case TokenRole::value:
      return "value";
    case TokenRole::filler:
      return "filler";
    case TokenRole::query_marker:
      return "query_marker";
    case TokenRole::query_entity:
      return "query_entity";
    case TokenRole::context:
      return "context";
    case TokenRole::count:
      return "count";
  }
  return "unknown";
}

void BindingTaskConfig::validate() const {
  require_positive(entity_count, "entity_count");
  require_positive(value_count, "value_count");
  require_positive(filler_count, "filler_count");
  require_positive(binding_count, "binding_count");
  if (binding_count > entity_count) {
    throw std::invalid_argument(
        "binding_count must not exceed entity_count");
  }
  static_cast<void>(sequence_length());

  const std::size_t vocabulary_size =
      checked_add(checked_add(entity_count, value_count,
                              "vocabulary size overflow"),
                  checked_add(filler_count, 1, "vocabulary size overflow"),
                  "vocabulary size overflow");
  if (vocabulary_size >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("vocabulary does not fit in int tokens");
  }
}

std::size_t BindingTaskConfig::sequence_length() const {
  const std::size_t stride =
      checked_add(2, fillers_per_binding, "binding stride overflow");
  return checked_add(checked_multiply(binding_count, stride,
                                      "binding sequence length overflow"),
                     2, "binding sequence length overflow");
}

int BindingVocabulary::entity_token(std::size_t entity) const {
  if (entity >= entity_count) {
    throw std::out_of_range("entity id out of range");
  }
  return static_cast<int>(entity);
}

int BindingVocabulary::value_token(std::size_t value) const {
  if (value >= value_count) {
    throw std::out_of_range("value id out of range");
  }
  return static_cast<int>(entity_count + value);
}

int BindingVocabulary::filler_token(std::size_t filler) const {
  if (filler >= filler_count) {
    throw std::out_of_range("filler id out of range");
  }
  return static_cast<int>(entity_count + value_count + filler);
}

int BindingVocabulary::query_token() const {
  return static_cast<int>(entity_count + value_count + filler_count);
}

int BindingVocabulary::context_token(std::size_t context) const {
  if (context >= context_count) {
    throw std::out_of_range("context id out of range");
  }
  return query_token() + 1 + static_cast<int>(context);
}

std::size_t BindingVocabulary::size() const {
  return entity_count + value_count + filler_count + 1 + context_count;
}


void RetentionTaskConfig::validate() const {
  require_positive(context_count, "context_count");
  require_positive(entity_count, "entity_count");
  require_positive(value_count, "value_count");
  require_positive(binding_count, "binding_count");
  require_positive(relevant_binding_count, "relevant_binding_count");
  require_positive(workspace_slots, "workspace_slots");
  if (entity_count % context_count != 0) {
    throw std::invalid_argument(
        "entity_count must be divisible by context_count");
  }
  if (entity_count / context_count != relevant_binding_count) {
    throw std::invalid_argument(
        "each context must map to relevant_binding_count entities");
  }
  if (binding_count != 2 * relevant_binding_count) {
    throw std::invalid_argument(
        "retention task requires equal relevant and irrelevant bindings");
  }
  if (binding_count > entity_count || workspace_slots > binding_count) {
    throw std::invalid_argument("invalid retention capacity");
  }
  static_cast<void>(sequence_length());
  const BindingVocabulary vocabulary{entity_count, value_count, 1,
                                     context_count};
  if (vocabulary.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("retention vocabulary does not fit int");
  }
}

std::size_t RetentionTaskConfig::total_binding_count() const {
  return checked_add(binding_count, delay_binding_count,
                     "retention binding count overflow");
}

std::size_t RetentionTaskConfig::sequence_length() const {
  return checked_add(checked_multiply(total_binding_count(), 2,
                                      "retention sequence length overflow"),
                     3, "retention sequence length overflow");
}

bool RetentionTaskConfig::is_relevant(std::size_t context,
                                      std::size_t entity) const {
  if (context >= context_count || entity >= entity_count) {
    throw std::out_of_range("retention context or entity out of range");
  }
  return entity % context_count == context;
}

BindingTaskConfig RetentionTaskConfig::binding_config() const {
  BindingTaskConfig config;
  config.entity_count = entity_count;
  config.value_count = value_count;
  config.filler_count = 1;
  config.binding_count = binding_count;
  config.fillers_per_binding = 0;
  return config;
}

RetentionBindingStream::RetentionBindingStream(
    const RetentionTaskConfig& config,
    std::uint64_t seed,
    std::size_t sample_count)
    : config_(config),
      vocabulary_{config.entity_count, config.value_count, 1,
                  config.context_count} {
  config_.validate();
  require_positive(sample_count, "sample_count");
  samples_ = make_unique_retention_samples(
      config_, vocabulary_, sample_count, seed, false);
}

RetentionBindingStream::RetentionBindingStream(
    const RetentionTaskConfig& config,
    std::uint64_t seed,
    std::span<const std::size_t> delay_schedule)
    : config_(config),
      vocabulary_{config.entity_count, config.value_count, 1,
                  config.context_count} {
  config_.validate();
  require_positive(delay_schedule.size(), "delay_schedule");
  std::mt19937_64 generator(seed);
  std::unordered_set<std::string> seen;
  samples_.reserve(delay_schedule.size());
  for (const std::size_t delay : delay_schedule) {
    RetentionTaskConfig sample_config = config_;
    sample_config.delay_binding_count = delay;
    sample_config.validate();
    bool inserted = false;
    for (std::size_t attempt = 0; attempt < 10000 && !inserted; ++attempt) {
      BindingSample sample = make_retention_sample(
          sample_config, vocabulary_, generator, false);
      const std::string key = literal_key(sample.tokens);
      if (!seen.insert(key).second) continue;
      samples_.push_back(std::move(sample));
      inserted = true;
    }
    if (!inserted) {
      throw std::runtime_error(
          "could not generate unique scheduled retention sample");
    }
  }
}

BindingSample RetentionBindingStream::next() {
  if (cursor_ >= samples_.size()) {
    throw std::out_of_range("retention stream exhausted");
  }
  return samples_[cursor_++];
}

std::size_t RetentionBindingStream::samples_consumed() const noexcept {
  return cursor_;
}

std::size_t RetentionBindingStream::remaining() const noexcept {
  return samples_.size() - cursor_;
}

std::size_t RetentionBindingStream::total_samples() const noexcept {
  return samples_.size();
}

RetentionDataset make_retention_binding_split(
    const RetentionTaskConfig& config,
    std::size_t train_count,
    std::size_t holdout_count,
    std::uint64_t seed) {
  config.validate();
  require_positive(train_count, "train_count");
  require_positive(holdout_count, "holdout_count");
  RetentionDataset dataset;
  dataset.config = config;
  dataset.vocabulary = BindingVocabulary{
      config.entity_count, config.value_count, 1, config.context_count};
  std::unordered_set<std::string> used;
  dataset.train = make_unique_retention_samples(
      config, dataset.vocabulary, train_count, seed, false, &used);
  dataset.holdout = make_unique_retention_samples(
      config, dataset.vocabulary, holdout_count,
      seed ^ 0x9e3779b97f4a7c15ULL, true, &used);
  return dataset;
}

BindingDataset make_binding_split(const BindingTaskConfig& config,
                                  std::size_t train_count,
                                  std::size_t holdout_count,
                                  std::uint64_t seed) {
  config.validate();
  require_positive(train_count, "train_count");
  require_positive(holdout_count, "holdout_count");

  BindingDataset dataset;
  dataset.config = config;
  dataset.vocabulary = BindingVocabulary{config.entity_count,
                                         config.value_count,
                                         config.filler_count};
  dataset.train.reserve(train_count);
  dataset.holdout.reserve(holdout_count);

  const std::size_t requested =
      checked_add(train_count, holdout_count, "sample count overflow");
  const std::size_t maximum_attempts =
      checked_add(checked_multiply(requested, 10000,
                                   "sample attempt count overflow"),
                  1000, "sample attempt count overflow");
  std::mt19937_64 generator(seed);
  std::unordered_set<std::string> seen;
  seen.reserve(requested * 2);

  std::size_t attempts = 0;
  while (dataset.train.size() + dataset.holdout.size() < requested) {
    if (++attempts > maximum_attempts) {
      throw std::runtime_error(
          "unable to generate enough unique binding sequences");
    }
    BindingSample sample = make_sample(config, dataset.vocabulary, generator);
    const std::string key = literal_key(sample.tokens);
    if (!seen.insert(key).second) {
      continue;
    }
    if (dataset.train.size() < train_count) {
      dataset.train.push_back(std::move(sample));
    } else {
      dataset.holdout.push_back(std::move(sample));
    }
  }
  return dataset;
}


bool is_structural_holdout_pair(const BindingTaskConfig& config,
                                std::size_t entity,
                                std::size_t value) {
  config.validate();
  if (entity >= config.entity_count || value >= config.value_count) {
    throw std::out_of_range("structural pair index out of range");
  }
  return value == entity % config.value_count;
}

StructuralBindingStream::StructuralBindingStream(
    const BindingTaskConfig& config, std::uint64_t seed)
    : config_(config),
      vocabulary_{config.entity_count, config.value_count, config.filler_count} {
  config_.validate();
  samples_ = enumerate_structural_samples(config_, vocabulary_, false);
  deterministic_shuffle_samples(samples_, seed);
  if (samples_.empty()) {
    throw std::invalid_argument("structural stream has no valid samples");
  }
}

BindingSample StructuralBindingStream::next() {
  if (cursor_ >= samples_.size()) {
    throw std::out_of_range("structural training stream exhausted");
  }
  return samples_[cursor_++];
}

std::size_t StructuralBindingStream::samples_consumed() const noexcept {
  return cursor_;
}

std::size_t StructuralBindingStream::remaining() const noexcept {
  return samples_.size() - cursor_;
}

std::size_t StructuralBindingStream::total_samples() const noexcept {
  return samples_.size();
}

BindingDataset make_structural_binding_split(
    const BindingTaskConfig& config, std::size_t train_count,
    std::size_t holdout_count, std::uint64_t seed) {
  config.validate();
  require_positive(train_count, "train_count");
  require_positive(holdout_count, "holdout_count");
  BindingDataset dataset;
  dataset.config = config;
  dataset.vocabulary = BindingVocabulary{config.entity_count,
                                         config.value_count,
                                         config.filler_count};
  auto train = enumerate_structural_samples(config, dataset.vocabulary, false);
  auto holdout = enumerate_structural_samples(config, dataset.vocabulary, true);
  deterministic_shuffle_samples(train, seed ^ 0x243f6a8885a308d3ULL);
  deterministic_shuffle_samples(holdout, seed ^ 0x13198a2e03707344ULL);
  if (train_count > train.size() || holdout_count > holdout.size()) {
    throw std::invalid_argument(
        "requested structural split exceeds unique sample capacity");
  }
  dataset.train.assign(train.begin(),
                       train.begin() + static_cast<std::ptrdiff_t>(train_count));
  dataset.holdout.assign(
      holdout.begin(),
      holdout.begin() + static_cast<std::ptrdiff_t>(holdout_count));
  return dataset;
}

}  // namespace sgw
