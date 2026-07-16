#include "test_harness.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

std::size_t parse_env_index(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  std::size_t parsed = 0;
  for (const char* cursor = value; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') {
      throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    parsed = parsed * 10 + static_cast<std::size_t>(*cursor - '0');
  }
  return parsed;
}

}  // namespace

int main() {
  const std::size_t shard_count = parse_env_index("SGW_TEST_SHARD_COUNT", 1);
  const std::size_t shard_index = parse_env_index("SGW_TEST_SHARD_INDEX", 0);
  if (shard_count == 0 || shard_index >= shard_count) {
    std::cerr << "invalid SGW test shard configuration\n";
    return 2;
  }

  std::size_t failures = 0;
  std::size_t executed = 0;
  const auto& tests = sgw::test::registry();
  for (std::size_t index = 0; index < tests.size(); ++index) {
    if (index % shard_count != shard_index) continue;
    const auto& test = tests[index];
    ++executed;
    try {
      test.function();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
    }
  }
  std::cout << "Executed " << executed << " of " << tests.size()
            << " tests in shard " << shard_index << '/' << shard_count << "; "
            << failures << " failed.\n";
  return failures == 0 ? 0 : 1;
}
