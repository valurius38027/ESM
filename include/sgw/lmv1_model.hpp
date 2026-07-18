#pragma once

#include "sgw/autodiff.hpp"
#include "sgw/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace sgw::lmv1 {

enum class ModelKind {
  tiny_rnn,
  tiny_gru,
  tiny_sgw,
};

[[nodiscard]] std::string_view model_kind_name(ModelKind kind) noexcept;

enum class CommunicationMode {
  learned_sparse,
  no_workspace,
  message_permuted,
  local_only,
};

[[nodiscard]] std::string_view communication_mode_name(
    CommunicationMode mode) noexcept;

struct ModelConfig {
  ModelKind kind{ModelKind::tiny_rnn};
  std::size_t vocabulary_size{256};
  std::size_t embedding_dim{16};
  std::size_t hidden_dim{32};
  std::size_t module_count{4};
  std::size_t message_dim{8};
  std::size_t workspace_slots{2};

  void validate() const;
  bool operator==(const ModelConfig&) const = default;
};

struct StepTrace {
  std::size_t token_index{0};
  std::size_t active_module{0};
  std::optional<std::size_t> write_slot;
  bool delivered_to_readout{false};

  bool operator==(const StepTrace&) const = default;
};

struct SequenceResult {
  std::vector<ad::Var> logits;
  std::vector<StepTrace> traces;
  std::size_t estimated_madds{0};
  std::size_t dense_counterfactual_madds{0};
  std::size_t write_count{0};
  std::size_t delivered_messages{0};
  std::vector<std::size_t> slot_load;
};

[[nodiscard]] std::size_t estimated_parameter_count(
    const ModelConfig& config);

class Model {
 public:
  explicit Model(ModelConfig config, std::uint64_t seed);

  [[nodiscard]] const ModelConfig& config() const noexcept;
  [[nodiscard]] ParameterSet& parameters() noexcept;
  [[nodiscard]] const ParameterSet& parameters() const noexcept;
  [[nodiscard]] std::vector<double> parameter_values() const;
  void load_parameter_values(std::span<const double> values);

  [[nodiscard]] SequenceResult forward(
      ad::Tape& tape, std::span<const std::uint8_t> input,
      bool capture_trace = false,
      CommunicationMode mode = CommunicationMode::learned_sparse);

 private:
  ModelConfig config_;
  ParameterSet parameters_;

  Parameter* embedding_{nullptr};
  Parameter* output_weight_{nullptr};
  Parameter* output_bias_{nullptr};

  Parameter* rnn_input_{nullptr};
  Parameter* rnn_recurrent_{nullptr};
  Parameter* rnn_bias_{nullptr};

  Parameter* gru_update_input_{nullptr};
  Parameter* gru_update_recurrent_{nullptr};
  Parameter* gru_update_bias_{nullptr};
  Parameter* gru_reset_input_{nullptr};
  Parameter* gru_reset_recurrent_{nullptr};
  Parameter* gru_reset_bias_{nullptr};
  Parameter* gru_candidate_input_{nullptr};
  Parameter* gru_candidate_recurrent_{nullptr};
  Parameter* gru_candidate_bias_{nullptr};

  Parameter* specialist_input_{nullptr};
  Parameter* specialist_recurrent_{nullptr};
  Parameter* specialist_bias_{nullptr};
  Parameter* readout_input_{nullptr};
  Parameter* readout_recurrent_{nullptr};
  Parameter* readout_inbox_{nullptr};
  Parameter* readout_bias_{nullptr};
  Parameter* message_weight_{nullptr};
  Parameter* message_bias_{nullptr};
  Parameter* slot_key_{nullptr};
};

}  // namespace sgw::lmv1
