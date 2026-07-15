#include "test_harness.hpp"

#include <exception>
#include <iostream>

int main() {
  std::size_t failures = 0;
  for (const auto& test : sgw::test::registry()) {
    try {
      test.function();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
    }
  }
  std::cout << "Executed " << sgw::test::registry().size() << " tests; "
            << failures << " failed.\n";
  return failures == 0 ? 0 : 1;
}
