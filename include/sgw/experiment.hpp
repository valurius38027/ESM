#pragma once

#include "sgw/autodiff.hpp"
#include "sgw/config.hpp"
#include "sgw/dataset.hpp"
#include "sgw/model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sgw {

struct TrainingConfig {
  std::size_t steps{100};
  std::size_t batch_size{8};
  std::uint64_t shuffle_seed{1};
  double workspace_aux_weight{0.0};
  std::size_t workspace_aux_anneal_steps{0};

  void validate() const;
};

struct EvaluationMetrics {
  double mean_nll{0.0};
  double accuracy{0.0};
  double mean_brier{0.0};
  double ece{0.0};
  double mean_max_confidence{0.0};
  double mean_true_class_probability{0.0};
  double mean_estimated_madds_per_token{0.0};
  double mean_active_mechanisms_per_token{0.0};
  double mean_writers_per_token{0.0};
  double mean_recipients_per_token{0.0};
  std::vector<std::size_t> mechanism_load;
  std::vector<std::size_t> role_mechanism_load;
  std::vector<std::size_t> read_slot_load;
  std::vector<std::size_t> write_slot_load;
  double mean_write_collision_rate{0.0};
  double mean_routing_entropy{0.0};
  double mean_routing_disagreement_rate{0.0};
  double retention_write_rate{0.0};
  double retention_skip_rate{0.0};
  double retention_eviction_rate{0.0};
  double relevant_eviction_rate{0.0};
  double queried_entity_retention_rate{0.0};
  double query_read_hit_rate{0.0};
  double mean_retained_age{0.0};
  std::vector<std::size_t> eviction_slot_load;
};

struct TrainingHistory {
  std::vector<double> batch_loss;
  std::vector<double> primary_batch_loss;
  std::vector<double> workspace_aux_batch_loss;
  std::vector<double> workspace_aux_weight;
  std::vector<double> gradient_norm;
  std::vector<double> clip_scale;
  std::vector<double> routing_collision_rate;
  std::vector<double> routing_entropy;
  std::vector<double> routing_disagreement_rate;
  std::vector<double> retention_write_rate;
  std::vector<double> retention_skip_rate;
  std::vector<double> retention_eviction_rate;
  std::vector<double> relevant_eviction_rate;
  std::vector<double> queried_entity_retention_rate;
  std::vector<double> query_read_hit_rate;
  std::size_t samples_consumed{0};
};

[[nodiscard]] double workspace_aux_weight_at_step(
    const TrainingConfig& config,
    std::size_t step);

[[nodiscard]] ad::Var cross_entropy_loss(
    ad::Tape& tape,
    std::span<const ad::Var> logits,
    std::size_t target_class);

[[nodiscard]] bool same_route_trace(
    std::span<const StepTrace> first,
    std::span<const StepTrace> second) noexcept;

[[nodiscard]] EvaluationMetrics evaluate(
    SgwEsmModel& model,
    std::span<const BindingSample> samples,
    bool collect_routes = true,
    ForwardIntervention intervention = ForwardIntervention::intact);

[[nodiscard]] TrainingHistory train_steps(
    SgwEsmModel& model,
    std::span<const BindingSample> samples,
    const AdamConfig& adam_config,
    const TrainingConfig& training_config);

[[nodiscard]] TrainingHistory train_steps(
    SgwEsmModel& model,
    RetentionBindingStream& stream,
    const AdamConfig& adam_config,
    const TrainingConfig& training_config);

[[nodiscard]] TrainingHistory train_steps(
    SgwEsmModel& model,
    StructuralBindingStream& stream,
    const AdamConfig& adam_config,
    const TrainingConfig& training_config);

}  // namespace sgw
