#include "test_harness.hpp"

#include "sgw/dataset.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <string>

namespace {

std::string serialize(const sgw::BindingSample& sample) {
  std::ostringstream stream;
  for (const int token : sample.tokens) {
    stream << token << ',';
  }
  stream << '|' << sample.target_class;
  return stream.str();
}

void verify_sample(const sgw::BindingDataset& dataset,
                   const sgw::BindingSample& sample) {
  SGW_REQUIRE(sample.tokens.size() == dataset.config.sequence_length());
  SGW_REQUIRE(sample.roles.size() == sample.tokens.size());
  SGW_REQUIRE(sample.target_class < dataset.config.value_count);
  SGW_REQUIRE(sample.queried_entity < dataset.config.entity_count);
  SGW_REQUIRE(sample.source_value_position == sample.source_entity_position + 1);
  SGW_REQUIRE(sample.query_entity_position + 1 == sample.tokens.size());
  SGW_REQUIRE(sample.tokens[sample.source_entity_position] ==
              dataset.vocabulary.entity_token(sample.queried_entity));
  SGW_REQUIRE(sample.tokens[sample.source_value_position] ==
              dataset.vocabulary.value_token(sample.target_class));
  SGW_REQUIRE(sample.tokens[sample.tokens.size() - 2] ==
              dataset.vocabulary.query_token());
  SGW_REQUIRE(sample.tokens[sample.query_entity_position] ==
              dataset.vocabulary.entity_token(sample.queried_entity));

  std::size_t binding_occurrences = 0;
  const std::size_t stride = 2 + dataset.config.fillers_per_binding;
  for (std::size_t binding = 0; binding < dataset.config.binding_count;
       ++binding) {
    const std::size_t entity_position = binding * stride;
    SGW_REQUIRE(sample.roles[entity_position] == sgw::TokenRole::entity);
    SGW_REQUIRE(sample.roles[entity_position + 1] == sgw::TokenRole::value);
    if (sample.tokens[entity_position] ==
        dataset.vocabulary.entity_token(sample.queried_entity)) {
      ++binding_occurrences;
    }
    for (std::size_t filler = 0;
         filler < dataset.config.fillers_per_binding; ++filler) {
      const std::size_t filler_position = entity_position + 2 + filler;
      const int token = sample.tokens[filler_position];
      SGW_REQUIRE(sample.roles[filler_position] == sgw::TokenRole::filler);
      SGW_REQUIRE(token >= dataset.vocabulary.filler_token(0));
      SGW_REQUIRE(token < dataset.vocabulary.query_token());
    }
  }
  SGW_REQUIRE(binding_occurrences == 1);
  SGW_REQUIRE(sample.roles[sample.tokens.size() - 2] ==
              sgw::TokenRole::query_marker);
  SGW_REQUIRE(sample.roles[sample.query_entity_position] ==
              sgw::TokenRole::query_entity);
  SGW_REQUIRE(sgw::token_role_name(sgw::TokenRole::query_entity) ==
              std::string_view("query_entity"));

  for (const int token : sample.tokens) {
    SGW_REQUIRE(token >= 0);
    SGW_REQUIRE(static_cast<std::size_t>(token) < dataset.vocabulary.size());
  }
}

}  // namespace

SGW_TEST(binding_split_is_identifiable_unique_and_disjoint) {
  sgw::BindingTaskConfig config;
  config.entity_count = 5;
  config.value_count = 4;
  config.filler_count = 3;
  config.binding_count = 3;
  config.fillers_per_binding = 2;
  const auto dataset = sgw::make_binding_split(config, 40, 20, 9123);

  SGW_REQUIRE(dataset.train.size() == 40);
  SGW_REQUIRE(dataset.holdout.size() == 20);
  SGW_REQUIRE(dataset.vocabulary.size() == 13);

  std::set<std::string> train_sequences;
  std::set<std::string> holdout_sequences;
  for (const auto& sample : dataset.train) {
    verify_sample(dataset, sample);
    SGW_REQUIRE(train_sequences.insert(serialize(sample)).second);
  }
  for (const auto& sample : dataset.holdout) {
    verify_sample(dataset, sample);
    SGW_REQUIRE(holdout_sequences.insert(serialize(sample)).second);
    SGW_REQUIRE(!train_sequences.contains(serialize(sample)));
  }
}

SGW_TEST(binding_split_is_byte_deterministic_for_fixed_seed) {
  sgw::BindingTaskConfig config;
  const auto first = sgw::make_binding_split(config, 12, 7, 44);
  const auto second = sgw::make_binding_split(config, 12, 7, 44);
  SGW_REQUIRE(first.train.size() == second.train.size());
  SGW_REQUIRE(first.holdout.size() == second.holdout.size());
  for (std::size_t index = 0; index < first.train.size(); ++index) {
    SGW_REQUIRE(serialize(first.train[index]) == serialize(second.train[index]));
  }
  for (std::size_t index = 0; index < first.holdout.size(); ++index) {
    SGW_REQUIRE(serialize(first.holdout[index]) ==
                serialize(second.holdout[index]));
  }
}

SGW_TEST(binding_task_rejects_impossible_or_empty_configuration) {
  sgw::BindingTaskConfig config;
  config.validate();

  auto invalid = config;
  invalid.entity_count = 0;
  SGW_REQUIRE_THROWS(invalid.validate());

  invalid = config;
  invalid.binding_count = config.entity_count + 1;
  SGW_REQUIRE_THROWS(invalid.validate());

  invalid = config;
  invalid.filler_count = 0;
  SGW_REQUIRE_THROWS(invalid.validate());

  SGW_REQUIRE_THROWS(sgw::make_binding_split(config, 0, 2, 1));
  SGW_REQUIRE_THROWS(sgw::make_binding_split(config, 2, 0, 1));
}

SGW_TEST(mediation_task_has_three_dense_bindings_and_no_filler_tokens) {
  sgw::BindingTaskConfig config;
  config.entity_count = 6;
  config.value_count = 5;
  config.filler_count = 1;
  config.binding_count = 3;
  config.fillers_per_binding = 0;
  const auto dataset = sgw::make_binding_split(config, 24, 12, 4040);
  SGW_REQUIRE(dataset.config.sequence_length() == 8);
  for (const auto& sample : dataset.train) {
    SGW_REQUIRE(sample.tokens.size() == 8);
    SGW_REQUIRE(std::count(sample.roles.begin(), sample.roles.end(),
                           sgw::TokenRole::filler) == 0);
  }
  for (const auto& sample : dataset.holdout) {
    SGW_REQUIRE(sample.tokens.size() == 8);
    SGW_REQUIRE(std::count(sample.roles.begin(), sample.roles.end(),
                           sgw::TokenRole::filler) == 0);
  }
}

SGW_TEST(structural_split_holds_out_entity_value_pairs_without_leakage) {
  sgw::BindingTaskConfig task;
  task.entity_count = 6;
  task.value_count = 5;
  task.filler_count = 1;
  task.binding_count = 3;
  task.fillers_per_binding = 0;
  const auto dataset = sgw::make_structural_binding_split(task, 160, 160, 77);

  for (const auto& sample : dataset.train) {
    for (std::size_t position = 0; position < 6; position += 2) {
      const std::size_t entity = static_cast<std::size_t>(sample.tokens[position]);
      const std::size_t value = static_cast<std::size_t>(
          sample.tokens[position + 1] - static_cast<int>(task.entity_count));
      SGW_REQUIRE(!sgw::is_structural_holdout_pair(task, entity, value));
    }
  }
  for (const auto& sample : dataset.holdout) {
    SGW_REQUIRE(sgw::is_structural_holdout_pair(
        task, sample.queried_entity, sample.target_class));
    for (std::size_t position = 0; position < 6; position += 2) {
      const std::size_t entity = static_cast<std::size_t>(sample.tokens[position]);
      const std::size_t value = static_cast<std::size_t>(
          sample.tokens[position + 1] - static_cast<int>(task.entity_count));
      if (entity != sample.queried_entity) {
        SGW_REQUIRE(!sgw::is_structural_holdout_pair(task, entity, value));
      }
    }
  }
}

SGW_TEST(structural_training_stream_is_deterministic_and_nonrepeating) {
  sgw::BindingTaskConfig task;
  task.entity_count = 6;
  task.value_count = 5;
  task.filler_count = 1;
  task.binding_count = 3;
  task.fillers_per_binding = 0;
  sgw::StructuralBindingStream first(task, 991);
  sgw::StructuralBindingStream second(task, 991);
  std::set<std::vector<int>> seen;
  for (std::size_t index = 0; index < 6400; ++index) {
    const auto lhs = first.next();
    const auto rhs = second.next();
    SGW_REQUIRE(lhs.tokens == rhs.tokens);
    SGW_REQUIRE(lhs.target_class == rhs.target_class);
    SGW_REQUIRE(seen.insert(lhs.tokens).second);
  }
  SGW_REQUIRE(first.samples_consumed() == 6400);
  SGW_REQUIRE(first.remaining() >= 1);
}

SGW_TEST(structural_samples_cover_every_binding_position_for_each_entity) {
  sgw::BindingTaskConfig task;
  task.entity_count = 6;
  task.value_count = 5;
  task.filler_count = 1;
  task.binding_count = 3;
  task.fillers_per_binding = 0;
  const auto dataset = sgw::make_structural_binding_split(task, 640, 160, 707);
  std::vector<std::vector<bool>> seen(
      task.entity_count, std::vector<bool>(task.binding_count, false));
  for (const auto& sample : dataset.train) {
    for (std::size_t binding = 0; binding < task.binding_count; ++binding) {
      const auto entity = static_cast<std::size_t>(sample.tokens[2 * binding]);
      seen.at(entity).at(binding) = true;
    }
  }
  for (const auto& positions : seen) {
    SGW_REQUIRE(std::all_of(positions.begin(), positions.end(),
                            [](bool value) { return value; }));
  }
}

SGW_TEST(retention_task_encodes_context_six_bindings_and_relevance) {
  sgw::RetentionTaskConfig task;
  task.validate();
  const auto dataset = sgw::make_retention_binding_split(task, 96, 48, 8128);
  SGW_REQUIRE(dataset.config.sequence_length() == 15);
  SGW_REQUIRE(dataset.vocabulary.context_count == 3);
  SGW_REQUIRE(dataset.vocabulary.query_token() == 16);
  SGW_REQUIRE(dataset.vocabulary.context_token(0) == 17);
  SGW_REQUIRE(dataset.vocabulary.context_token(2) == 19);
  SGW_REQUIRE(dataset.vocabulary.size() == 20);

  const auto verify = [&](const sgw::BindingSample& sample, bool holdout) {
    SGW_REQUIRE(sample.tokens.size() == 15);
    SGW_REQUIRE(sample.roles.size() == sample.tokens.size());
    SGW_REQUIRE(sample.roles.front() == sgw::TokenRole::context);
    SGW_REQUIRE(sample.context_class < task.context_count);
    SGW_REQUIRE(sample.tokens.front() ==
                dataset.vocabulary.context_token(sample.context_class));
    SGW_REQUIRE(sample.binding_relevance.size() == task.binding_count);
    SGW_REQUIRE(static_cast<std::size_t>(std::count(
                    sample.binding_relevance.begin(),
                    sample.binding_relevance.end(), true)) ==
                task.relevant_binding_count);
    SGW_REQUIRE(task.is_relevant(sample.context_class, sample.queried_entity));
    SGW_REQUIRE(sample.tokens[sample.tokens.size() - 2] ==
                dataset.vocabulary.query_token());
    SGW_REQUIRE(sample.tokens.back() ==
                dataset.vocabulary.entity_token(sample.queried_entity));
    SGW_REQUIRE(sample.query_entity_position + 1 == sample.tokens.size());
    SGW_REQUIRE(holdout == sgw::is_structural_holdout_pair(
                               task.binding_config(), sample.queried_entity,
                               sample.target_class));

    std::set<std::size_t> entities;
    for (std::size_t binding = 0; binding < task.binding_count; ++binding) {
      const std::size_t position = 1 + 2 * binding;
      const auto entity = static_cast<std::size_t>(sample.tokens[position]);
      const auto value = static_cast<std::size_t>(
          sample.tokens[position + 1] - static_cast<int>(task.entity_count));
      SGW_REQUIRE(entities.insert(entity).second);
      SGW_REQUIRE(sample.roles[position] == sgw::TokenRole::entity);
      SGW_REQUIRE(sample.roles[position + 1] == sgw::TokenRole::value);
      SGW_REQUIRE(sample.binding_relevance[binding] ==
                  task.is_relevant(sample.context_class, entity));
      if (!holdout || entity != sample.queried_entity) {
        SGW_REQUIRE(!sgw::is_structural_holdout_pair(
            task.binding_config(), entity, value));
      }
    }
  };
  for (const auto& sample : dataset.train) verify(sample, false);
  for (const auto& sample : dataset.holdout) verify(sample, true);
}

SGW_TEST(retention_training_stream_is_deterministic_and_nonrepeating) {
  sgw::RetentionTaskConfig task;
  sgw::RetentionBindingStream first(task, 9917, 9600);
  sgw::RetentionBindingStream second(task, 9917, 9600);
  std::set<std::vector<int>> seen;
  for (std::size_t index = 0; index < 9600; ++index) {
    const auto lhs = first.next();
    const auto rhs = second.next();
    SGW_REQUIRE(lhs.tokens == rhs.tokens);
    SGW_REQUIRE(lhs.target_class == rhs.target_class);
    SGW_REQUIRE(lhs.context_class == rhs.context_class);
    SGW_REQUIRE(lhs.binding_relevance == rhs.binding_relevance);
    SGW_REQUIRE(seen.insert(lhs.tokens).second);
  }
  SGW_REQUIRE(first.samples_consumed() == 9600);
  SGW_REQUIRE(first.remaining() == 0);
  SGW_REQUIRE_THROWS(first.next());
}
