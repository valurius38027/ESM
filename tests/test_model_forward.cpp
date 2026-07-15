#include "test_harness.hpp"

#include "sgw/model.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace {

sgw::ModelConfig tiny_config() {
  sgw::ModelConfig config;
  config.vocab_size = 12;
  config.output_classes = 4;
  config.embedding_dim = 3;
  config.spine_dim = 4;
  config.mechanism_count = 4;
  config.mechanism_dim = 3;
  config.workspace_slots = 2;
  config.workspace_dim = 3;
  config.active_mechanisms = 2;
  config.workspace_writers = 1;
  config.broadcast_recipients = 2;
  return config;
}

void require_unique(const std::vector<std::size_t>& values) {
  const std::set<std::size_t> unique(values.begin(), values.end());
  SGW_REQUIRE(unique.size() == values.size());
}

}  // namespace

SGW_TEST(model_forward_respects_all_sparse_budgets_and_mutation_boundaries) {
  const auto config = tiny_config();
  sgw::SgwEsmModel model(config, 17);
  sgw::ad::Tape tape;
  const std::vector<int> tokens{0, 5, 10, 2};
  const auto result = model.forward_sequence(tape, tokens, true);

  SGW_REQUIRE(result.logits.size() == config.output_classes);
  SGW_REQUIRE(result.traces.size() == tokens.size());
  SGW_REQUIRE(model.parameters().scalar_count() > 0);
  for (const auto& logit : result.logits) {
    SGW_REQUIRE(std::isfinite(logit.value()));
  }

  for (const auto& trace : result.traces) {
    SGW_REQUIRE(trace.active_mechanisms.size() == config.active_mechanisms);
    SGW_REQUIRE(trace.writers.size() == config.workspace_writers);
    SGW_REQUIRE(trace.writer_slots.size() == config.workspace_writers);
    SGW_REQUIRE(trace.recipients.size() == config.broadcast_recipients);
    SGW_REQUIRE(trace.estimated_madds == sgw::estimated_step_madds(config));
    require_unique(trace.active_mechanisms);
    require_unique(trace.writers);
    require_unique(trace.recipients);

    SGW_REQUIRE(trace.mechanism_state_before.size() ==
                config.mechanism_count * config.mechanism_dim);
    SGW_REQUIRE(trace.mechanism_state_after.size() ==
                trace.mechanism_state_before.size());
    SGW_REQUIRE(trace.inbox_before.size() ==
                config.mechanism_count * config.workspace_dim);
    SGW_REQUIRE(trace.inbox_after.size() == trace.inbox_before.size());

    for (std::size_t mechanism = 0; mechanism < config.mechanism_count;
         ++mechanism) {
      const bool active =
          std::find(trace.active_mechanisms.begin(),
                    trace.active_mechanisms.end(), mechanism) !=
          trace.active_mechanisms.end();
      if (!active) {
        for (std::size_t dimension = 0; dimension < config.mechanism_dim;
             ++dimension) {
          const std::size_t index =
              mechanism * config.mechanism_dim + dimension;
          SGW_REQUIRE(trace.mechanism_state_before[index] ==
                      trace.mechanism_state_after[index]);
        }
      }

      const bool recipient =
          std::find(trace.recipients.begin(), trace.recipients.end(),
                    mechanism) != trace.recipients.end();
      if (!recipient) {
        for (std::size_t dimension = 0; dimension < config.workspace_dim;
             ++dimension) {
          const std::size_t index =
              mechanism * config.workspace_dim + dimension;
          SGW_REQUIRE(trace.inbox_before[index] == trace.inbox_after[index]);
        }
      }
    }
  }
}

SGW_TEST(model_forward_is_deterministic_for_seed_and_input) {
  const auto config = tiny_config();
  sgw::SgwEsmModel first(config, 99);
  sgw::SgwEsmModel second(config, 99);
  const std::vector<int> tokens{1, 6, 11, 3};
  sgw::ad::Tape first_tape;
  sgw::ad::Tape second_tape;
  const auto first_result = first.forward_sequence(first_tape, tokens, true);
  const auto second_result = second.forward_sequence(second_tape, tokens, true);

  SGW_REQUIRE(first_result.logits.size() == second_result.logits.size());
  for (std::size_t index = 0; index < first_result.logits.size(); ++index) {
    SGW_REQUIRE_NEAR(first_result.logits[index].value(),
                     second_result.logits[index].value(), 0.0);
  }
  SGW_REQUIRE(first_result.traces.size() == second_result.traces.size());
  for (std::size_t step = 0; step < first_result.traces.size(); ++step) {
    SGW_REQUIRE(first_result.traces[step].same_route(
        second_result.traces[step]));
    SGW_REQUIRE(first_result.traces[step].mechanism_state_after ==
                second_result.traces[step].mechanism_state_after);
    SGW_REQUIRE(first_result.traces[step].inbox_after ==
                second_result.traces[step].inbox_after);
  }
}

SGW_TEST(model_forward_rejects_empty_or_out_of_range_sequences) {
  const auto config = tiny_config();
  sgw::SgwEsmModel model(config, 1);
  sgw::ad::Tape tape;
  const std::vector<int> empty;
  SGW_REQUIRE_THROWS(model.forward_sequence(tape, empty));
  const std::vector<int> invalid{-1, 0};
  SGW_REQUIRE_THROWS(model.forward_sequence(tape, invalid));
  const std::vector<int> too_large{static_cast<int>(config.vocab_size)};
  SGW_REQUIRE_THROWS(model.forward_sequence(tape, too_large));
}

SGW_TEST(core_only_mode_skips_sparse_workspace_execution) {
  auto config = tiny_config();
  config.core_only = true;
  sgw::SgwEsmModel model(config, 3);
  auto full_config = config;
  full_config.core_only = false;
  sgw::SgwEsmModel full_model(full_config, 3);
  SGW_REQUIRE(model.parameters().scalar_count() <
              full_model.parameters().scalar_count());
  SGW_REQUIRE(sgw::estimated_step_madds(config) <
              sgw::estimated_step_madds(full_config));
  sgw::ad::Tape tape;
  const std::vector<int> tokens{0, 1, 2};
  const auto result = model.forward_sequence(tape, tokens, true);
  for (const auto& trace : result.traces) {
    SGW_REQUIRE(trace.active_mechanisms.empty());
    SGW_REQUIRE(trace.writers.empty());
    SGW_REQUIRE(trace.writer_slots.empty());
    SGW_REQUIRE(trace.recipients.empty());
    SGW_REQUIRE(trace.estimated_madds == sgw::estimated_step_madds(config));
  }
}
