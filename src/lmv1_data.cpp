#include "sgw/lmv1_data.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace sgw::lmv1 {

CorpusSplits split_corpus(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < kRequiredCorpusBytes) {
    throw std::invalid_argument("lmv1 corpus must contain at least 160 KiB");
  }
  CorpusSplits result;
  result.train.assign(bytes.begin(), bytes.begin() +
                                         static_cast<std::ptrdiff_t>(kTrainBytes));
  result.validation.assign(
      bytes.begin() + static_cast<std::ptrdiff_t>(kTrainBytes),
      bytes.begin() + static_cast<std::ptrdiff_t>(kTrainBytes +
                                                  kValidationBytes));
  result.test.assign(
      bytes.begin() + static_cast<std::ptrdiff_t>(kTrainBytes +
                                                  kValidationBytes),
      bytes.begin() + static_cast<std::ptrdiff_t>(kRequiredCorpusBytes));
  return result;
}

CorpusSplits load_corpus_splits(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("unable to open lmv1 corpus");
  input.seekg(0, std::ios::end);
  const auto end = input.tellg();
  if (end < 0) throw std::runtime_error("unable to size lmv1 corpus");
  input.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  if (!bytes.empty()) {
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }
  if (!input || input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error("unable to read complete lmv1 corpus");
  }
  return split_corpus(bytes);
}

std::vector<std::size_t> make_window_offsets(std::size_t byte_count,
                                             std::size_t context_length,
                                             std::uint64_t seed) {
  if (context_length == 0 || byte_count <= context_length) {
    throw std::invalid_argument("invalid lmv1 byte count or context length");
  }
  std::vector<std::size_t> offsets(byte_count - context_length);
  std::iota(offsets.begin(), offsets.end(), 0);
  std::mt19937_64 generator(seed);
  std::shuffle(offsets.begin(), offsets.end(), generator);
  return offsets;
}

WindowDataset::WindowDataset(std::span<const std::uint8_t> bytes,
                             std::size_t context_length)
    : bytes_(bytes.begin(), bytes.end()), context_length_(context_length) {
  if (context_length_ == 0 || bytes_.size() <= context_length_) {
    throw std::invalid_argument("invalid lmv1 window dataset");
  }
}

std::size_t WindowDataset::size() const noexcept {
  return bytes_.size() - context_length_;
}

std::size_t WindowDataset::context_length() const noexcept {
  return context_length_;
}

Window WindowDataset::at_offset(std::size_t offset) const {
  if (offset >= size()) throw std::out_of_range("lmv1 window offset");
  Window result;
  result.inputs.assign(bytes_.begin() + static_cast<std::ptrdiff_t>(offset),
                       bytes_.begin() + static_cast<std::ptrdiff_t>(
                                            offset + context_length_));
  result.targets.assign(bytes_.begin() + static_cast<std::ptrdiff_t>(offset + 1),
                        bytes_.begin() + static_cast<std::ptrdiff_t>(
                                             offset + context_length_ + 1));
  return result;
}

}  // namespace sgw::lmv1
