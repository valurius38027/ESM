#include "test_harness.hpp"

#include "sgw/experiment.hpp"
#include "sgw/model.hpp"
#include "sgw/presets.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace {

std::vector<double> logits_of(const sgw::SequenceResult& result) {
  std::vector<double> values;
  values.reserve(result.logits.size());
  for (const auto logit : result.logits) {
    values.push_back(logit.value());
  }
  return values;
}

bool any_difference(const std::vector<double>& first,
                    const std::vector<double>& second) {
  if (first.size() != second.size()) {
    return true;
  }
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (std::abs(first[index] - second[index]) > 1.0e-12) {
      return true;
    }
  }
  return false;
}

}  // namespace

SGW_TEST(intact_intervention_is_identical_to_default_forward) {
  const auto config = sgw::make_model_config(sgw::ModelPreset::sgw);
  sgw::SgwEsmModel model(config, 23);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape default_tape;
  sgw::ad::Tape explicit_tape;
  const auto default_result = model.forward_sequence(default_tape, tokens, true);
  const auto explicit_result = model.forward_sequence(
      explicit_tape, tokens, true, sgw::ForwardIntervention::intact);
  SGW_REQUIRE(logits_of(default_result) == logits_of(explicit_result));
  SGW_REQUIRE(sgw::same_route_trace(default_result.traces,
                                    explicit_result.traces));
}

SGW_TEST(no_broadcast_preserves_every_inbox_while_retaining_selection_trace) {
  const auto config = sgw::make_model_config(sgw::ModelPreset::sgw);
  sgw::SgwEsmModel model(config, 29);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape intact_tape;
  sgw::ad::Tape ablated_tape;
  const auto intact = model.forward_sequence(intact_tape, tokens, true);
  const auto ablated = model.forward_sequence(
      ablated_tape, tokens, true, sgw::ForwardIntervention::no_broadcast);
  SGW_REQUIRE(intact.traces.size() == ablated.traces.size());
  SGW_REQUIRE(intact.traces.front().recipients ==
              ablated.traces.front().recipients);
  for (const auto& trace : ablated.traces) {
    SGW_REQUIRE(trace.recipients.size() == config.broadcast_recipients);
    SGW_REQUIRE(trace.inbox_before == trace.inbox_after);
  }
  SGW_REQUIRE(any_difference(logits_of(intact), logits_of(ablated)));
}

SGW_TEST(no_workspace_persistence_zeros_effective_workspace_before_each_token) {
  const auto config = sgw::make_model_config(sgw::ModelPreset::sgw);
  sgw::SgwEsmModel model(config, 31);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape tape;
  const auto result = model.forward_sequence(
      tape, tokens, true,
      sgw::ForwardIntervention::no_workspace_persistence);
  for (const auto& trace : result.traces) {
    SGW_REQUIRE(trace.workspace_before.size() ==
                config.workspace_slots * config.workspace_dim);
    SGW_REQUIRE(std::all_of(trace.workspace_before.begin(),
                            trace.workspace_before.end(),
                            [](double value) { return value == 0.0; }));
  }
}

SGW_TEST(no_workspace_output_changes_only_final_readout_path) {
  const auto config = sgw::make_model_config(sgw::ModelPreset::sgw);
  sgw::SgwEsmModel model(config, 37);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape intact_tape;
  sgw::ad::Tape ablated_tape;
  const auto intact = model.forward_sequence(intact_tape, tokens, true);
  const auto ablated = model.forward_sequence(
      ablated_tape, tokens, true,
      sgw::ForwardIntervention::no_workspace_output);
  SGW_REQUIRE(sgw::same_route_trace(intact.traces, ablated.traces));
  SGW_REQUIRE(intact.final_state.workspace.size() ==
              ablated.final_state.workspace.size());
  for (std::size_t index = 0; index < intact.final_state.workspace.size();
       ++index) {
    SGW_REQUIRE_NEAR(intact.final_state.workspace[index].value(),
                     ablated.final_state.workspace[index].value(), 0.0);
  }
  SGW_REQUIRE(any_difference(logits_of(intact), logits_of(ablated)));
}

SGW_TEST(permuted_recipients_cyclically_shift_effective_addresses) {
  const auto config = sgw::make_model_config(sgw::ModelPreset::sgw);
  sgw::SgwEsmModel model(config, 41);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape intact_tape;
  sgw::ad::Tape permuted_tape;
  const auto intact = model.forward_sequence(intact_tape, tokens, true);
  const auto permuted = model.forward_sequence(
      permuted_tape, tokens, true,
      sgw::ForwardIntervention::permuted_recipients);
  for (std::size_t step = 0; step < intact.traces.size(); ++step) {
    SGW_REQUIRE(intact.traces[step].recipients.size() ==
                permuted.traces[step].recipients.size());
    for (std::size_t index = 0;
         index < intact.traces[step].recipients.size(); ++index) {
      SGW_REQUIRE(permuted.traces[step].recipients[index] ==
                  (intact.traces[step].recipients[index] + 1) %
                      config.mechanism_count);
    }
  }
}

SGW_TEST(no_spine_workspace_matches_equivalent_static_configuration) {
  const auto redundant =
      sgw::make_model_config(sgw::ModelPreset::sgw_redundant);
  auto static_disabled = redundant;
  static_disabled.spine_reads_workspace = false;
  sgw::SgwEsmModel intervention_model(redundant, 43);
  sgw::SgwEsmModel static_model(static_disabled, 43);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape intervention_tape;
  sgw::ad::Tape static_tape;
  const auto intervention = intervention_model.forward_sequence(
      intervention_tape, tokens, true,
      sgw::ForwardIntervention::no_spine_workspace);
  const auto configured = static_model.forward_sequence(static_tape, tokens,
                                                        true);
  SGW_REQUIRE(logits_of(intervention) == logits_of(configured));
  SGW_REQUIRE(sgw::same_route_trace(intervention.traces,
                                    configured.traces));
}

SGW_TEST(no_mechanism_output_matches_equivalent_static_configuration) {
  const auto redundant =
      sgw::make_model_config(sgw::ModelPreset::sgw_redundant);
  auto static_disabled = redundant;
  static_disabled.output_reads_mechanism = false;
  sgw::SgwEsmModel intervention_model(redundant, 47);
  sgw::SgwEsmModel static_model(static_disabled, 47);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape intervention_tape;
  sgw::ad::Tape static_tape;
  const auto intervention = intervention_model.forward_sequence(
      intervention_tape, tokens, true,
      sgw::ForwardIntervention::no_mechanism_output);
  const auto configured = static_model.forward_sequence(static_tape, tokens,
                                                        true);
  SGW_REQUIRE(logits_of(intervention) == logits_of(configured));
  SGW_REQUIRE(sgw::same_route_trace(intervention.traces,
                                    configured.traces));
}

SGW_TEST(workspace_disconnected_matches_all_workspace_reads_disabled) {
  const auto redundant =
      sgw::make_model_config(sgw::ModelPreset::sgw_redundant);
  auto static_disabled = redundant;
  static_disabled.spine_reads_workspace = false;
  static_disabled.output_reads_workspace = false;
  sgw::SgwEsmModel disconnected_model(redundant, 53);
  sgw::SgwEsmModel static_model(static_disabled, 53);
  const std::vector<int> tokens{0, 6, 12, 4, 7, 13, 1};
  sgw::ad::Tape disconnected_tape;
  sgw::ad::Tape static_tape;
  const auto disconnected = disconnected_model.forward_sequence(
      disconnected_tape, tokens, true,
      sgw::ForwardIntervention::workspace_disconnected);
  const auto configured = static_model.forward_sequence(
      static_tape, tokens, true, sgw::ForwardIntervention::no_broadcast);
  SGW_REQUIRE(logits_of(disconnected) == logits_of(configured));
  SGW_REQUIRE(sgw::same_route_trace(disconnected.traces,
                                    configured.traces));
  for (const auto& trace : disconnected.traces) {
    SGW_REQUIRE(trace.inbox_before == trace.inbox_after);
  }
}

SGW_TEST(no_workspace_writes_preserves_fixed_writer_trace_but_keeps_workspace_zero) {
  const auto config =
      sgw::make_model_config(sgw::ModelPreset::mediation_fixed, 13, 5);
  sgw::SgwEsmModel model(config, 79);
  const std::vector<int> tokens{0, 6, 1, 7, 2, 8, 12, 1};
  sgw::ad::Tape intact_tape;
  sgw::ad::Tape ablated_tape;
  const auto intact = model.forward_sequence(intact_tape, tokens, true);
  const auto ablated = model.forward_sequence(
      ablated_tape, tokens, true,
      sgw::ForwardIntervention::no_workspace_writes);
  SGW_REQUIRE(sgw::same_route_trace(intact.traces, ablated.traces));
  for (const auto value : ablated.final_state.workspace) {
    SGW_REQUIRE_NEAR(value.value(), 0.0, 0.0);
  }
  SGW_REQUIRE(any_difference(logits_of(intact), logits_of(ablated)));
}

SGW_TEST(zero_reader_inbox_matches_no_broadcast_for_fixed_mediation) {
  const auto config =
      sgw::make_model_config(sgw::ModelPreset::mediation_fixed, 13, 5);
  sgw::SgwEsmModel model(config, 83);
  const std::vector<int> tokens{0, 6, 1, 7, 2, 8, 12, 1};
  sgw::ad::Tape broadcast_tape;
  sgw::ad::Tape inbox_tape;
  const auto no_broadcast = model.forward_sequence(
      broadcast_tape, tokens, true,
      sgw::ForwardIntervention::no_broadcast);
  const auto zero_inbox = model.forward_sequence(
      inbox_tape, tokens, true,
      sgw::ForwardIntervention::zero_reader_inbox);
  SGW_REQUIRE(logits_of(no_broadcast) == logits_of(zero_inbox));
  SGW_REQUIRE(sgw::same_route_trace(no_broadcast.traces,
                                    zero_inbox.traces));
}

SGW_TEST(phase4_intervention_names_are_explicit) {
  SGW_REQUIRE(sgw::forward_intervention_name(
                  sgw::ForwardIntervention::no_workspace_writes) ==
              std::string_view("no_workspace_writes"));
  SGW_REQUIRE(sgw::forward_intervention_name(
                  sgw::ForwardIntervention::zero_reader_inbox) ==
              std::string_view("zero_reader_inbox"));
}
