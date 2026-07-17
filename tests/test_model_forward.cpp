#include "test_harness.hpp"

#include "sgw/model.hpp"
#include "sgw/presets.hpp"

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


std::vector<double> forward_logits(const sgw::SequenceResult& result) {
  std::vector<double> values;
  values.reserve(result.logits.size());
  for (const auto value : result.logits) values.push_back(value.value());
  return values;
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

SGW_TEST(fixed_mediation_uses_one_writer_per_binding_and_one_reader) {
  const auto config =
      sgw::make_model_config(sgw::ModelPreset::mediation_fixed, 13, 5);
  sgw::SgwEsmModel model(config, 71);
  const std::vector<int> tokens{0, 6, 1, 7, 2, 8, 12, 1};
  sgw::ad::Tape tape;
  const auto result = model.forward_sequence(tape, tokens, true);

  SGW_REQUIRE(result.traces.size() == 8);
  const std::vector<std::size_t> expected_active{0, 0, 1, 1, 2, 2, 3, 3};
  for (std::size_t step = 0; step < result.traces.size(); ++step) {
    const auto& trace = result.traces[step];
    SGW_REQUIRE(trace.active_mechanisms ==
                std::vector<std::size_t>{expected_active[step]});
    SGW_REQUIRE(trace.recipients == std::vector<std::size_t>{3});
    if (step < 6) {
      SGW_REQUIRE(trace.writers ==
                  std::vector<std::size_t>{expected_active[step]});
      SGW_REQUIRE(trace.writer_slots ==
                  std::vector<std::size_t>{expected_active[step]});
    } else {
      SGW_REQUIRE(trace.writers.empty());
      SGW_REQUIRE(trace.writer_slots.empty());
    }
  }
  for (const auto value : result.final_state.spine) {
    SGW_REQUIRE_NEAR(value.value(), 0.0, 0.0);
  }
}

SGW_TEST(fixed_mediation_rejects_noncanonical_sequence_length) {
  const auto config =
      sgw::make_model_config(sgw::ModelPreset::mediation_fixed, 13, 5);
  sgw::SgwEsmModel model(config, 73);
  const std::vector<int> too_short{0, 6, 1, 7, 2, 8, 12};
  sgw::ad::Tape tape;
  SGW_REQUIRE_THROWS(model.forward_sequence(tape, too_short, false));
}

SGW_TEST(bounded_output_head_limits_every_logit_without_extra_parameters) {
  auto config = sgw::make_model_config(
      sgw::ModelPreset::structural_mediation_bounded, 13, 5);
  sgw::SgwEsmModel model(config, 9);
  for (auto& parameter : model.parameters().parameters()) {
    if (parameter->name() == "output_bias") {
      for (double& value : parameter->mutable_values()) value = 100.0;
    }
  }
  const std::vector<int> tokens{0, 6, 1, 7, 2, 8, 12, 0};
  sgw::ad::Tape tape;
  const auto result = model.forward_sequence(tape, tokens, false);
  for (const auto logit : result.logits) {
    SGW_REQUIRE(logit.value() <= 1.0);
    SGW_REQUIRE(logit.value() >= -1.0);
  }
}

SGW_TEST(exact_key_value_path_preserves_slot_and_value_identity) {
  const auto config = sgw::make_model_config(
      sgw::ModelPreset::structural_kv_exact, 13, 5);
  sgw::SgwEsmModel model(config, 101);
  const std::vector<int> tokens{0, 6, 1, 7, 2, 8, 12, 1};
  sgw::ad::Tape tape;
  const auto result = model.forward_sequence(tape, tokens, true);

  SGW_REQUIRE(result.logits.size() == 5);
  for (std::size_t output = 0; output < 5; ++output) {
    SGW_REQUIRE_NEAR(result.logits[output].value(),
                     output == 1 ? 6.0 : 0.0, 0.0);
  }
  const std::vector<std::size_t> active{0, 0, 1, 1, 2, 2, 3, 3};
  const std::vector<std::size_t> slots{0, 0, 1, 1, 2, 2};
  for (std::size_t step = 0; step < result.traces.size(); ++step) {
    SGW_REQUIRE(result.traces[step].active_mechanisms ==
                std::vector<std::size_t>{active[step]});
    if (step < slots.size()) {
      SGW_REQUIRE(result.traces[step].writer_slots ==
                  std::vector<std::size_t>{slots[step]});
    }
  }
  SGW_REQUIRE(result.traces.back().read_slots ==
              std::vector<std::size_t>{1});
}

SGW_TEST(learned_key_value_path_uses_tied_codebooks_deterministically) {
  const auto config = sgw::make_model_config(
      sgw::ModelPreset::structural_kv_learned, 13, 5);
  sgw::SgwEsmModel first(config, 103);
  sgw::SgwEsmModel second(config, 103);
  const std::vector<int> tokens{0, 6, 1, 7, 2, 8, 12, 2};
  sgw::ad::Tape first_tape;
  sgw::ad::Tape second_tape;
  const auto first_result = first.forward_sequence(first_tape, tokens, true);
  const auto second_result = second.forward_sequence(second_tape, tokens, true);

  SGW_REQUIRE(forward_logits(first_result) == forward_logits(second_result));
  const auto best = std::max_element(first_result.logits.begin(),
                                     first_result.logits.end(),
                                     [](const auto lhs, const auto rhs) {
                                       return lhs.value() < rhs.value();
                                     });
  SGW_REQUIRE(static_cast<std::size_t>(best - first_result.logits.begin()) == 2);
  SGW_REQUIRE(first_result.traces.back().read_slots ==
              std::vector<std::size_t>{2});
}

SGW_TEST(phase7_first_free_and_learned_routes_write_one_available_slot) {
  const std::vector<int> tokens{4, 7, 0, 9, 2, 6, 12, 0};
  for (const auto preset : {sgw::ModelPreset::kv_first_free,
                            sgw::ModelPreset::kv_hard_router,
                            sgw::ModelPreset::kv_annealed_router}) {
    const auto config = sgw::make_model_config(preset, 13, 5);
    sgw::SgwEsmModel model(config, 211);
    model.set_key_value_routing_step(0);
    sgw::ad::Tape tape;
    const auto result = model.forward_sequence(tape, tokens, true);
    std::vector<std::size_t> entity_slots;
    for (std::size_t step = 0; step < 6; step += 2) {
      SGW_REQUIRE(result.traces[step].writer_slots.size() == 1);
      SGW_REQUIRE(result.traces[step + 1].writer_slots ==
                  result.traces[step].writer_slots);
      entity_slots.push_back(result.traces[step].writer_slots.front());
    }
    std::sort(entity_slots.begin(), entity_slots.end());
    SGW_REQUIRE(entity_slots == std::vector<std::size_t>({0, 1, 2}));
    SGW_REQUIRE(result.traces.back().read_slots.size() == 1);
  }
}

namespace {

std::size_t argmax_logits(const sgw::SequenceResult& result) {
  std::size_t best = 0;
  for (std::size_t index = 1; index < result.logits.size(); ++index) {
    if (result.logits[index].value() > result.logits[best].value()) best = index;
  }
  return best;
}

std::vector<int> scarce_retention_example() {
  // Context 0: relevant entities are 0, 3, and 6. Query entity 0 is early,
  // so FIFO evicts it while oracle retention preserves it.
  return {17, 0, 11, 1, 10, 3, 12, 2, 13,
          6, 14, 4, 9, 16, 0};
}

}  // namespace

SGW_TEST(phase8_full_capacity_and_oracle_preserve_queried_binding) {
  const auto tokens = scarce_retention_example();
  auto full_config = sgw::make_model_config(
      sgw::ModelPreset::kv_full_capacity, 20, 6);
  auto oracle_config = sgw::make_model_config(
      sgw::ModelPreset::kv_oracle_retention, 20, 6);
  sgw::SgwEsmModel full(full_config, 41);
  sgw::SgwEsmModel oracle(oracle_config, 41);
  sgw::ad::Tape full_tape;
  sgw::ad::Tape oracle_tape;
  const auto full_result = full.forward_sequence(full_tape, tokens, true);
  const auto oracle_result = oracle.forward_sequence(oracle_tape, tokens, true);

  SGW_REQUIRE(argmax_logits(full_result) == 2);
  SGW_REQUIRE(argmax_logits(oracle_result) == 2);
  SGW_REQUIRE(full_result.traces.size() == tokens.size());
  SGW_REQUIRE(oracle_result.traces.size() == tokens.size());
  std::size_t oracle_writes = 0;
  std::size_t oracle_skips = 0;
  for (const auto& trace : oracle_result.traces) {
    oracle_writes += trace.retention_writes;
    oracle_skips += trace.retention_skips;
  }
  SGW_REQUIRE(oracle_writes == 3);
  SGW_REQUIRE(oracle_skips == 3);
  SGW_REQUIRE(oracle_result.traces.back().queried_entity_retained);
  SGW_REQUIRE(oracle_result.traces.back().query_read_hit);
}

SGW_TEST(phase8_fifo_eviction_can_remove_an_early_relevant_binding) {
  const auto tokens = scarce_retention_example();
  auto config = sgw::make_model_config(
      sgw::ModelPreset::kv_fifo_eviction, 20, 6);
  sgw::SgwEsmModel model(config, 43);
  sgw::ad::Tape tape;
  const auto result = model.forward_sequence(tape, tokens, true);
  SGW_REQUIRE(argmax_logits(result) != 2);
  std::size_t evictions = 0;
  std::size_t relevant_evictions = 0;
  for (const auto& trace : result.traces) {
    evictions += trace.retention_evictions;
    relevant_evictions += trace.relevant_evictions;
  }
  SGW_REQUIRE(evictions == 3);
  SGW_REQUIRE(relevant_evictions >= 1);
  SGW_REQUIRE(!result.traces.back().queried_entity_retained);
}

SGW_TEST(phase8_reservoir_policy_is_seed_deterministic) {
  const auto tokens = scarce_retention_example();
  auto config = sgw::make_model_config(
      sgw::ModelPreset::kv_reservoir, 20, 6);
  sgw::SgwEsmModel first(config, 47);
  sgw::SgwEsmModel second(config, 47);
  sgw::ad::Tape first_tape;
  sgw::ad::Tape second_tape;
  const auto lhs = first.forward_sequence(first_tape, tokens, true);
  const auto rhs = second.forward_sequence(second_tape, tokens, true);
  SGW_REQUIRE(lhs.traces.size() == rhs.traces.size());
  for (std::size_t index = 0; index < lhs.traces.size(); ++index) {
    SGW_REQUIRE(lhs.traces[index].retention_actions ==
                rhs.traces[index].retention_actions);
  }
  SGW_REQUIRE(forward_logits(lhs) == forward_logits(rhs));
}

SGW_TEST(phase9_retention_forward_accepts_variable_delay_lengths) {
  const std::vector<int> prefix{17, 0, 11, 1, 10, 3, 12, 2, 13,
                                6, 14, 4, 9};
  const std::vector<std::pair<int, int>> distractors{
      {1, 12}, {2, 11}, {4, 13}, {5, 10}, {7, 9}, {8, 12}};
  for (const std::size_t delay : {std::size_t{0}, std::size_t{6},
                                  std::size_t{12}, std::size_t{24}}) {
    std::vector<int> tokens = prefix;
    for (std::size_t index = 0; index < delay; ++index) {
      const auto [entity, value] = distractors[index % distractors.size()];
      tokens.push_back(entity);
      tokens.push_back(value);
    }
    tokens.push_back(16);
    tokens.push_back(0);
    auto config = sgw::make_model_config(
        sgw::ModelPreset::kv_oracle_retention, 20, 6);
    sgw::SgwEsmModel model(config, 991 + delay);
    sgw::ad::Tape tape;
    const auto result = model.forward_sequence(tape, tokens, true);
    SGW_REQUIRE(result.traces.size() == tokens.size());
    SGW_REQUIRE(result.traces.back().queried_entity_retained);
    SGW_REQUIRE(result.traces.back().query_read_hit);
    SGW_REQUIRE(result.traces.back().relevant_survival_count == 3);
    SGW_REQUIRE(result.traces.back().relevant_survival_total == 3);
  }
}
