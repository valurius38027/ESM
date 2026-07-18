#pragma once

#include "sgw/config.hpp"
#include "sgw/lmv1_data.hpp"
#include "sgw/lmv1_model.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace sgw::lmv1 {

struct TrainingConfig {
  std::size_t steps{300};
  std::size_t batch_size{2};
  std::size_t context_length{16};
  std::size_t validation_interval{50};
  std::size_t evaluation_windows{16};
  std::uint64_t shuffle_seed{9000};
  AdamConfig optimizer{0.003, 0.9, 0.999, 1.0e-8, 1.0};

  void validate() const;
  bool operator==(const TrainingConfig&) const = default;
};

struct LossPoint {
  std::size_t step{0};
  double train_nll{0.0};
  double validation_nll{0.0};
  bool operator==(const LossPoint&) const = default;
};

struct TrainingReport {
  std::vector<LossPoint> loss_curve;
  std::size_t windows_consumed{0};
  std::size_t predicted_bytes{0};
  double initial_validation_nll{0.0};
  double final_train_nll{0.0};
  double final_validation_nll{0.0};
  double mean_estimated_madds_per_predicted_byte{0.0};
  bool operator==(const TrainingReport&) const = default;
};

struct Metrics {
  std::size_t sample_count{0};
  double mean_nll{0.0};
  double bits_per_byte{0.0};
  double accuracy{0.0};
  double mean_estimated_madds_per_predicted_byte{0.0};
  double mean_dense_counterfactual_madds_per_predicted_byte{0.0};
  double mean_writes_per_predicted_byte{0.0};
  double mean_delivered_per_predicted_byte{0.0};
  bool operator==(const Metrics&) const = default;
};

struct Checkpoint {
  ModelConfig config;
  std::vector<double> parameter_values;
  bool operator==(const Checkpoint&) const = default;
};

[[nodiscard]] TrainingReport train(Model& model,
                                   const WindowDataset& training_data,
                                   const WindowDataset& validation_data,
                                   const TrainingConfig& config);
[[nodiscard]] Metrics evaluate(
    Model& model, const WindowDataset& data, std::size_t window_count,
    std::uint64_t offset_seed,
    CommunicationMode mode = CommunicationMode::learned_sparse);
void save_checkpoint(const std::filesystem::path& path, const Model& model);
[[nodiscard]] Checkpoint load_checkpoint(const std::filesystem::path& path);

}  // namespace sgw::lmv1
