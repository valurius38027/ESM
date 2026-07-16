#pragma once

#include <cstddef>

namespace sgw {

struct ModelConfig {
  std::size_t vocab_size{16};
  std::size_t output_classes{4};
  std::size_t embedding_dim{8};
  std::size_t spine_dim{8};
  std::size_t mechanism_count{4};
  std::size_t mechanism_dim{6};
  std::size_t workspace_slots{2};
  std::size_t workspace_dim{6};
  std::size_t active_mechanisms{2};
  std::size_t workspace_writers{1};
  std::size_t broadcast_recipients{2};
  bool core_only{false};
  bool spine_reads_embedding{true};
  bool spine_reads_workspace{true};
  bool output_reads_spine{true};
  bool output_reads_workspace{true};
  bool output_reads_mechanism{true};
  bool fixed_binding_mediation{false};
  std::size_t mediation_binding_count{0};
  double output_logit_bound{0.0};

  void validate() const;
};

struct AdamConfig {
  double learning_rate{1.0e-3};
  double beta1{0.9};
  double beta2{0.999};
  double epsilon{1.0e-8};
  double max_grad_norm{1.0};

  void validate() const;
};

}  // namespace sgw
