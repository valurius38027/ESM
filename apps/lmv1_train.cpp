#include "sgw/lmv1_data.hpp"
#include "sgw/lmv1_experiment.hpp"
#include "sgw/lmv1_model.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using sgw::lmv1::CommunicationMode;
using sgw::lmv1::ModelKind;

std::size_t parse_size(const std::string& value, std::string_view name) {
  std::size_t position = 0;
  const auto parsed = std::stoull(value, &position);
  if (position != value.size()) throw std::invalid_argument(std::string(name));
  return static_cast<std::size_t>(parsed);
}

ModelKind parse_model(const std::string& value) {
  if (value == "tiny_rnn") return ModelKind::tiny_rnn;
  if (value == "tiny_gru") return ModelKind::tiny_gru;
  if (value == "tiny_sgw") return ModelKind::tiny_sgw;
  throw std::invalid_argument("unknown lmv1 model");
}

void write_atomic(const std::filesystem::path& path, const std::string& text) {
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("unable to open lmv1 output");
    output << text;
  }
  std::filesystem::rename(temporary, path);
}

std::string metric_row(const sgw::lmv1::Metrics& metrics,
                       CommunicationMode mode) {
  std::ostringstream out;
  out << std::setprecision(17) << sgw::lmv1::communication_mode_name(mode)
      << ',' << metrics.sample_count << ',' << metrics.mean_nll << ','
      << metrics.bits_per_byte << ',' << metrics.accuracy << ','
      << metrics.mean_estimated_madds_per_predicted_byte << ','
      << metrics.mean_dense_counterfactual_madds_per_predicted_byte << ','
      << metrics.mean_writes_per_predicted_byte << ','
      << metrics.mean_delivered_per_predicted_byte << '\n';
  return out.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> arguments;
    for (int index = 1; index < argc; index += 2) {
      if (index + 1 >= argc || std::string_view(argv[index]).substr(0, 2) != "--") {
        throw std::invalid_argument("lmv1 arguments require --name value pairs");
      }
      const std::string key = std::string(argv[index]).substr(2);
      if (!arguments.emplace(key, argv[index + 1]).second) {
        throw std::invalid_argument("duplicate lmv1 argument");
      }
    }
    const std::vector<std::string> allowed{
        "corpus", "corpus-id", "model", "context", "steps", "batch-size",
        "eval-windows", "validation-interval", "embedding-dim", "hidden-dim",
        "message-dim", "workspace-slots", "seed", "output-dir"};
    for (const auto& [key, value] : arguments) {
      static_cast<void>(value);
      if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
        throw std::invalid_argument("unknown lmv1 argument: " + key);
      }
    }
    auto required = [&](const char* key) -> const std::string& {
      const auto position = arguments.find(key);
      if (position == arguments.end()) {
        throw std::invalid_argument(std::string("missing --") + key);
      }
      return position->second;
    };

    const std::filesystem::path corpus_path = required("corpus");
    const std::filesystem::path output_dir = required("output-dir");
    const ModelKind kind = parse_model(required("model"));
    const std::uint64_t seed = parse_size(arguments.contains("seed") ? arguments["seed"] : "9000", "seed");
    const std::string corpus_id = arguments.contains("corpus-id") ? arguments["corpus-id"] : "unspecified";

    sgw::lmv1::ModelConfig model_config;
    model_config.kind = kind;
    model_config.embedding_dim = parse_size(arguments.contains("embedding-dim") ? arguments["embedding-dim"] : "16", "embedding-dim");
    model_config.hidden_dim = parse_size(arguments.contains("hidden-dim") ? arguments["hidden-dim"] : (kind == ModelKind::tiny_sgw ? "16" : "32"), "hidden-dim");
    model_config.message_dim = parse_size(arguments.contains("message-dim") ? arguments["message-dim"] : "8", "message-dim");
    model_config.workspace_slots = parse_size(arguments.contains("workspace-slots") ? arguments["workspace-slots"] : "2", "workspace-slots");
    model_config.validate();

    sgw::lmv1::TrainingConfig training_config;
    training_config.context_length = parse_size(arguments.contains("context") ? arguments["context"] : "16", "context");
    training_config.steps = parse_size(arguments.contains("steps") ? arguments["steps"] : "300", "steps");
    training_config.batch_size = parse_size(arguments.contains("batch-size") ? arguments["batch-size"] : "2", "batch-size");
    training_config.evaluation_windows = parse_size(arguments.contains("eval-windows") ? arguments["eval-windows"] : "16", "eval-windows");
    training_config.validation_interval = parse_size(arguments.contains("validation-interval") ? arguments["validation-interval"] : "50", "validation-interval");
    training_config.shuffle_seed = seed;
    training_config.validate();

    const auto splits = sgw::lmv1::load_corpus_splits(corpus_path);
    const sgw::lmv1::WindowDataset training_data(splits.train,
                                                 training_config.context_length);
    const sgw::lmv1::WindowDataset validation_data(
        splits.validation, training_config.context_length);
    sgw::lmv1::Model model(model_config, seed);
    std::filesystem::create_directories(output_dir);

    const auto started = std::chrono::steady_clock::now();
    const auto report = sgw::lmv1::train(model, training_data, validation_data,
                                         training_config);
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const std::size_t eval_count =
        std::min(training_config.evaluation_windows, validation_data.size());
    const auto learned = sgw::lmv1::evaluate(
        model, validation_data, eval_count, seed ^ 0x55aa55aaULL,
        CommunicationMode::learned_sparse);

    std::ostringstream primary;
    primary << "schema_version,model,seed,context_length,steps,batch_size,corpus_id,parameter_count,initial_validation_nll,final_train_nll,final_validation_nll,final_validation_bpb,estimated_madds_per_predicted_byte,dense_counterfactual_madds_per_predicted_byte,writes_per_predicted_byte,delivered_per_predicted_byte\n";
    primary << std::setprecision(17) << 1 << ','
            << sgw::lmv1::model_kind_name(kind) << ',' << seed << ','
            << training_config.context_length << ',' << training_config.steps
            << ',' << training_config.batch_size << ',' << corpus_id << ','
            << model.parameters().scalar_count() << ','
            << report.initial_validation_nll << ',' << report.final_train_nll
            << ',' << learned.mean_nll << ',' << learned.bits_per_byte << ','
            << learned.mean_estimated_madds_per_predicted_byte << ','
            << learned.mean_dense_counterfactual_madds_per_predicted_byte << ','
            << learned.mean_writes_per_predicted_byte << ','
            << learned.mean_delivered_per_predicted_byte << '\n';
    write_atomic(output_dir / "primary.csv", primary.str());

    std::ostringstream interventions;
    interventions << "mode,sample_count,mean_nll,bits_per_byte,accuracy,estimated_madds_per_predicted_byte,dense_counterfactual_madds_per_predicted_byte,writes_per_predicted_byte,delivered_per_predicted_byte\n";
    interventions << metric_row(learned, CommunicationMode::learned_sparse);
    if (kind == ModelKind::tiny_sgw) {
      for (const auto mode : {CommunicationMode::no_workspace,
                              CommunicationMode::message_permuted,
                              CommunicationMode::local_only}) {
        interventions << metric_row(sgw::lmv1::evaluate(
            model, validation_data, eval_count, seed ^ 0x55aa55aaULL, mode), mode);
      }
    }
    write_atomic(output_dir / "interventions.csv", interventions.str());

    std::ostringstream curve;
    curve << "step,train_nll,validation_nll\n";
    curve << std::setprecision(17);
    for (const auto& point : report.loss_curve) {
      curve << point.step << ',' << point.train_nll << ','
            << point.validation_nll << '\n';
    }
    write_atomic(output_dir / "loss_curve.csv", curve.str());
    sgw::lmv1::save_checkpoint(output_dir / "checkpoint.bin", model);

    std::ostringstream runtime;
    runtime << "wall_seconds,predicted_bytes_per_second\n" << std::setprecision(17)
            << elapsed << ','
            << static_cast<double>(report.predicted_bytes) / elapsed << '\n';
    write_atomic(output_dir / "runtime.csv", runtime.str());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "lmv1_train: " << error.what() << '\n';
    return 2;
  }
}
