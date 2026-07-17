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
  if (key_value_mode != KeyValueMediationMode::none) {
    require_positive(entity_count, "entity_count");
    require_positive(value_count, "value_count");
    require_positive(key_dim, "key_dim");
    require_positive(value_dim, "value_dim");
    require_positive(mediation_binding_count, "mediation_binding_count");
    require_finite_positive(key_value_logit_scale, "key_value_logit_scale");
    require_finite_positive(key_value_router_initial_temperature,
                            "key_value_router_initial_temperature");
    require_finite_positive(key_value_router_final_temperature,
                            "key_value_router_final_temperature");
    if (key_value_router_final_temperature >
        key_value_router_initial_temperature) {
      throw std::invalid_argument(
          "key-value router final temperature must not exceed initial temperature");
    }
    if (key_value_write_routing ==
            KeyValueWriteRoutingMode::annealed_learned &&
        key_value_router_anneal_steps == 0) {
      throw std::invalid_argument(
          "annealed key-value routing requires positive anneal steps");
    }
    if (fixed_binding_mediation || core_only) {
      throw std::invalid_argument(
          "key-value mediation requires its dedicated non-core path");
    }
    if (value_count != output_classes) {
      throw std::invalid_argument(
          "key-value mediation requires one value code per output class");
    }
    if (workspace_dim != key_dim + value_dim) {
      throw std::invalid_argument(
          "key-value workspace dimension must equal key_dim + value_dim");
    }
    if (key_value_retention == KeyValueRetentionMode::none) {
      if (workspace_slots != mediation_binding_count) {
        throw std::invalid_argument(
            "key-value mediation requires one slot per binding");
      }
    } else {
      require_positive(context_count, "context_count");
      if (mediation_binding_count != 6) {
        throw std::invalid_argument(
            "retention experiments require six bindings");
      }
      if (key_value_retention == KeyValueRetentionMode::full_capacity) {
        if (workspace_slots != mediation_binding_count) {
          throw std::invalid_argument(
              "full-capacity retention requires one slot per binding");
        }
      } else if (workspace_slots != 3) {
        throw std::invalid_argument(
            "scarce retention experiments require three slots");
      }
      if ((key_value_retention == KeyValueRetentionMode::hard_learned ||
           key_value_retention == KeyValueRetentionMode::annealed_learned) &&
          key_value_mode != KeyValueMediationMode::learned_tied) {
        throw std::invalid_argument(
            "learned retention requires learned tied key-value mediation");
      }
      if (key_value_retention == KeyValueRetentionMode::annealed_learned &&
          key_value_router_anneal_steps == 0) {
        throw std::invalid_argument(
            "annealed retention requires positive anneal steps");
      }
    }
    if (mechanism_count < mediation_binding_count + 1) {
      throw std::invalid_argument(
          "key-value mediation requires writer traces and one reader trace");
    }
    if (active_mechanisms != 1 || workspace_writers != 1 ||
        broadcast_recipients != 1) {
      throw std::invalid_argument(
          "key-value mediation requires k=q=m=1");
    }
    if (spine_reads_embedding || spine_reads_workspace || output_reads_spine ||
        output_reads_workspace || !output_reads_mechanism) {
      throw std::invalid_argument(
          "key-value mediation requires the sparse retrieved-value path to be the only content path");
    }
    const std::size_t required_vocab = entity_count + value_count + 1 +
        (key_value_retention == KeyValueRetentionMode::none ? 0 : context_count);
    if (vocab_size < required_vocab) {
      throw std::invalid_argument(
          "key-value vocabulary does not contain all entity and value tokens");
    }
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
