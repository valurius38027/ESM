#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace sgw::lmv1 {

inline constexpr std::size_t kTrainBytes = 128U * 1024U;
inline constexpr std::size_t kValidationBytes = 16U * 1024U;
inline constexpr std::size_t kTestBytes = 16U * 1024U;
inline constexpr std::size_t kRequiredCorpusBytes =
    kTrainBytes + kValidationBytes + kTestBytes;

struct CorpusSplits {
  std::vector<std::uint8_t> train;
  std::vector<std::uint8_t> validation;
  std::vector<std::uint8_t> test;

  bool operator==(const CorpusSplits&) const = default;
};

struct Window {
  std::vector<std::uint8_t> inputs;
  std::vector<std::uint8_t> targets;

  bool operator==(const Window&) const = default;
};

[[nodiscard]] CorpusSplits split_corpus(std::span<const std::uint8_t> bytes);
[[nodiscard]] CorpusSplits load_corpus_splits(
    const std::filesystem::path& path);
[[nodiscard]] std::vector<std::size_t> make_window_offsets(
    std::size_t byte_count, std::size_t context_length, std::uint64_t seed);

class WindowDataset {
 public:
  WindowDataset(std::span<const std::uint8_t> bytes,
                std::size_t context_length);

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t context_length() const noexcept;
  [[nodiscard]] Window at_offset(std::size_t offset) const;

 private:
  std::vector<std::uint8_t> bytes_;
  std::size_t context_length_{0};
};

}  // namespace sgw::lmv1
