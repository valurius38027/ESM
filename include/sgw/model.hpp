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
  permuted_recipients,
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
  std::vector<double> mechanism_state_before;
  std::vector<double> mechanism_state_after;
  std::vector<double> inbox_before;
  std::vector<double> inbox_after;
  std::vector<double> workspace_before;
  std::vector<double> workspace_after;
  std::size_t estimated_madds{0};

  [[nodiscard]] bool same_route(const StepTrace& other) const noexcept;
};

struct SequenceResult {
  std::vector<ad::Var> logits;
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

  [[nodiscard]] SequenceResult forward_sequence(
      ad::Tape& tape,
      std::span<const int> tokens,
      bool capture_trace = false,
      ForwardIntervention intervention = ForwardIntervention::intact);

 private:
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
};

}  // namespace sgw
