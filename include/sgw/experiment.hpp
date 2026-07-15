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
  double mean_estimated_madds_per_token{0.0};
  double mean_active_mechanisms_per_token{0.0};
  double mean_writers_per_token{0.0};
  double mean_recipients_per_token{0.0};
  std::vector<std::size_t> mechanism_load;
  std::vector<std::size_t> role_mechanism_load;
};

struct TrainingHistory {
  std::vector<double> batch_loss;
  std::vector<double> primary_batch_loss;
  std::vector<double> workspace_aux_batch_loss;
  std::vector<double> workspace_aux_weight;
  std::vector<double> gradient_norm;
  std::vector<double> clip_scale;
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

}  // namespace sgw
