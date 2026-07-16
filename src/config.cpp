#include "sgw/config.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace sgw {
namespace {

void require_positive(std::size_t value, const char* name) {
  if (value == 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

void require_finite_positive(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
}

}  // namespace

void ModelConfig::validate() const {
  require_positive(vocab_size, "vocab_size");
  require_positive(output_classes, "output_classes");
  require_positive(embedding_dim, "embedding_dim");
  require_positive(spine_dim, "spine_dim");
  require_positive(mechanism_count, "mechanism_count");
  require_positive(mechanism_dim, "mechanism_dim");
  require_positive(workspace_slots, "workspace_slots");
  require_positive(workspace_dim, "workspace_dim");
  require_positive(active_mechanisms, "active_mechanisms");
  require_positive(workspace_writers, "workspace_writers");
  require_positive(broadcast_recipients, "broadcast_recipients");
  if (!std::isfinite(output_logit_bound) || output_logit_bound < 0.0) {
    throw std::invalid_argument(
        "output_logit_bound must be finite and non-negative");
  }

  if (output_classes > vocab_size) {
    throw std::invalid_argument("output_classes must not exceed vocab_size");
  }
  if (active_mechanisms > mechanism_count) {
    throw std::invalid_argument(
        "active_mechanisms must not exceed mechanism_count");
  }
  if (workspace_writers > active_mechanisms) {
    throw std::invalid_argument(
        "workspace_writers must not exceed active_mechanisms");
  }
  if (broadcast_recipients > mechanism_count) {
    throw std::invalid_argument(
        "broadcast_recipients must not exceed mechanism_count");
  }
  if (fixed_binding_mediation) {
    require_positive(mediation_binding_count, "mediation_binding_count");
    if (core_only) {
      throw std::invalid_argument(
          "fixed_binding_mediation requires a non-core model");
    }
    if (active_mechanisms != 1 || workspace_writers != 1 ||
        broadcast_recipients != 1) {
      throw std::invalid_argument(
          "fixed_binding_mediation requires k=q=m=1");
    }
    if (mechanism_count < mediation_binding_count + 1) {
      throw std::invalid_argument(
          "fixed_binding_mediation requires one writer per binding and one reader");
    }
    if (workspace_slots < mediation_binding_count) {
      throw std::invalid_argument(
          "fixed_binding_mediation requires one workspace slot per binding");
    }
    if (spine_reads_embedding || spine_reads_workspace || output_reads_spine ||
        output_reads_workspace || !output_reads_mechanism) {
      throw std::invalid_argument(
          "fixed_binding_mediation requires the mechanism broadcast path to be the only content path");
    }
  }
}

void AdamConfig::validate() const {
  require_finite_positive(learning_rate, "learning_rate");
  require_finite_positive(epsilon, "epsilon");
  require_finite_positive(max_grad_norm, "max_grad_norm");
  if (!std::isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0) {
    throw std::invalid_argument("beta1 must be finite and in [0, 1)");
  }
  if (!std::isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0) {
    throw std::invalid_argument("beta2 must be finite and in [0, 1)");
  }
}

}  // namespace sgw
