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
  sample.queried_entity = entities[queried_binding];

  const std::size_t stride = 2 + config.fillers_per_binding;
  for (std::size_t binding = 0; binding < config.binding_count; ++binding) {
    const std::size_t entity_position = sample.tokens.size();
    sample.tokens.push_back(vocabulary.entity_token(entities[binding]));
    const std::size_t value = bounded_random(generator, config.value_count);
    sample.tokens.push_back(vocabulary.value_token(value));
    for (std::size_t filler = 0; filler < config.fillers_per_binding;
         ++filler) {
      const std::size_t filler_id =
          bounded_random(generator, config.filler_count);
      sample.tokens.push_back(vocabulary.filler_token(filler_id));
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
  sample.query_entity_position = sample.tokens.size();
  sample.tokens.push_back(vocabulary.entity_token(sample.queried_entity));
  return sample;
}

}  // namespace

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

std::size_t BindingVocabulary::size() const {
  return entity_count + value_count + filler_count + 1;
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

}  // namespace sgw
