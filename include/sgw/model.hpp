#pragma once

#include "sgw/autodiff.hpp"
#include "sgw/config.hpp"
#include "sgw/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace sgw {

enum class ForwardIntervention {
  intact,
  no_broadcast,
  no_workspace_persistence,
  no_workspace_output,
  no_spine_workspace,
  no_mechanism_output,
  workspace_disconnected,
  no_workspace_writes,
  zero_reader_inbox,
  permuted_recipients,
  permuted_workspace_keys,
  zero_query_key,
  randomized_write_slots,
  cleared_writer_assignment,
  allow_write_collisions,
  zero_query_context,
  randomized_retention_actions,
  force_fifo_retention,
  force_relevant_eviction,
  permuted_context_labels,
  disable_retention_skip,
  remove_delay_distractors,
  relevant_looking_delay_distractors,
  reverse_delay_block,
};

[[nodiscard]] std::string_view forward_intervention_name(
    ForwardIntervention intervention) noexcept;

struct ModelState {
  std::vector<ad::Var> spine;
  std::vector<ad::Var> mechanisms;
  std::vector<ad::Var> workspace;
  std::vector<ad::Var> inboxes;
  std::vector<ad::Var> active_summary;
};

struct StepTrace {
  std::vector<std::size_t> active_mechanisms;
  std::vector<std::size_t> writers;
  std::vector<std::size_t> writer_slots;
  std::vector<std::size_t> recipients;
  std::vector<std::size_t> read_slots;
  std::vector<double> mechanism_state_before;
  std::vector<double> mechanism_state_after;
  std::vector<double> inbox_before;
  std::vector<double> inbox_after;
  std::vector<double> workspace_before;
  std::vector<double> workspace_after;
  std::size_t estimated_madds{0};
  std::size_t write_collisions{0};
  double routing_entropy{0.0};
  std::size_t routing_decisions{0};
  std::size_t hard_soft_disagreements{0};
  std::vector<std::size_t> retention_actions;
  std::size_t retention_writes{0};
  std::size_t retention_skips{0};
  std::size_t retention_evictions{0};
  std::size_t relevant_writes{0};
  std::size_t irrelevant_writes{0};
  std::size_t relevant_evictions{0};
  std::size_t irrelevant_evictions{0};
  std::size_t distractor_writes{0};
  std::size_t distractor_evictions{0};
  std::size_t delay_distractor_decisions{0};
  bool queried_entity_retained{false};
  bool query_read_hit{false};
  std::size_t relevant_survival_count{0};
  std::size_t relevant_survival_total{0};
  double retained_age_sum{0.0};
  std::size_t retained_age_count{0};

  [[nodiscard]] bool same_route(const StepTrace& other) const noexcept;
};

struct SequenceResult {
  std::vector<ad::Var> logits;
  std::vector<ad::Var> workspace_aux_logits;
  ModelState final_state;
  std::vector<StepTrace> traces;
};

[[nodiscard]] std::size_t estimated_step_madds(const ModelConfig& config);

class SgwEsmModel {
 public:
  explicit SgwEsmModel(ModelConfig config, std::uint64_t seed);

  [[nodiscard]] const ModelConfig& config() const noexcept;
  [[nodiscard]] ParameterSet& parameters() noexcept;
  [[nodiscard]] const ParameterSet& parameters() const noexcept;
  void set_key_value_routing_step(std::size_t step) noexcept;
  [[nodiscard]] std::size_t key_value_routing_step() const noexcept;
  [[nodiscard]] double key_value_router_temperature() const noexcept;
  [[nodiscard]] bool key_value_router_uses_surrogate() const noexcept;

  [[nodiscard]] SequenceResult forward_sequence(
      ad::Tape& tape,
      std::span<const int> tokens,
      bool capture_trace = false,
      ForwardIntervention intervention = ForwardIntervention::intact);

 private:
  [[nodiscard]] SequenceResult forward_retention_sequence(
      ad::Tape& tape, std::span<const int> tokens, bool capture_trace,
      ForwardIntervention intervention);

  [[nodiscard]] SequenceResult forward_key_value_sequence(
      ad::Tape& tape, std::span<const int> tokens, bool capture_trace,
      ForwardIntervention intervention);

  ModelConfig config_;
  ParameterSet parameters_;

  Parameter* embedding_{nullptr};
  Parameter* spine_input_{nullptr};
  Parameter* spine_recurrent_{nullptr};
  Parameter* spine_workspace_{nullptr};
  Parameter* spine_bias_{nullptr};

  Parameter* router_embedding_{nullptr};
  Parameter* router_spine_{nullptr};
  Parameter* router_workspace_{nullptr};
  Parameter* router_state_{nullptr};
  Parameter* router_bias_{nullptr};

  Parameter* mechanism_embedding_{nullptr};
  Parameter* mechanism_spine_{nullptr};
  Parameter* mechanism_recurrent_{nullptr};
  Parameter* mechanism_inbox_{nullptr};
  Parameter* mechanism_bias_{nullptr};

  Parameter* message_state_{nullptr};
  Parameter* message_bias_{nullptr};
  Parameter* writer_key_{nullptr};
  Parameter* writer_bias_{nullptr};
  Parameter* slot_key_{nullptr};

  Parameter* workspace_message_{nullptr};
  Parameter* workspace_recurrent_{nullptr};
  Parameter* workspace_bias_{nullptr};
  Parameter* workspace_gate_message_{nullptr};
  Parameter* workspace_gate_slot_{nullptr};
  Parameter* workspace_gate_bias_{nullptr};

  Parameter* recipient_key_{nullptr};
  Parameter* recipient_bias_{nullptr};
  Parameter* broadcast_projection_{nullptr};
  Parameter* broadcast_bias_{nullptr};

  Parameter* output_spine_{nullptr};
  Parameter* output_workspace_{nullptr};
  Parameter* output_mechanism_{nullptr};
  Parameter* output_bias_{nullptr};

  Parameter* kv_entity_codebook_{nullptr};
  Parameter* kv_value_codebook_{nullptr};
  Parameter* kv_slot_codebook_{nullptr};
  Parameter* kv_context_codebook_{nullptr};
  Parameter* kv_retention_age_weight_{nullptr};
  std::uint64_t model_seed_{0};
  std::size_t key_value_routing_step_{0};
};

}  // namespace sgw
