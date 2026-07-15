#pragma once

#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sgw::test {

using TestFunction = void (*)();

struct TestCase {
  std::string name;
  TestFunction function;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(std::string name, TestFunction function) {
    registry().push_back(TestCase{std::move(name), function});
  }
};

inline void fail(const char* expression, const char* file, int line,
                 const std::string& detail = {}) {
  std::ostringstream stream;
  stream << file << ':' << line << ": requirement failed: " << expression;
  if (!detail.empty()) {
    stream << " (" << detail << ')';
  }
  throw std::runtime_error(stream.str());
}

inline void require_near(double actual, double expected, double tolerance,
                         const char* expression, const char* file, int line) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::ostringstream detail;
    detail << "actual=" << actual << ", expected=" << expected
           << ", tolerance=" << tolerance;
    fail(expression, file, line, detail.str());
  }
}

}  // namespace sgw::test

#define SGW_TEST(name)                                                        \
  static void name();                                                         \
  static ::sgw::test::Registrar name##_registrar{#name, &name};              \
  static void name()

#define SGW_REQUIRE(expression)                                               \
  do {                                                                        \
    if (!(expression)) {                                                      \
      ::sgw::test::fail(#expression, __FILE__, __LINE__);                    \
    }                                                                         \
  } while (false)

#define SGW_REQUIRE_NEAR(actual, expected, tolerance)                         \
  ::sgw::test::require_near((actual), (expected), (tolerance),               \
                            #actual " ~= " #expected, __FILE__, __LINE__)

#define SGW_REQUIRE_THROWS(expression)                                        \
  do {                                                                        \
    bool sgw_threw = false;                                                   \
    try {                                                                     \
      static_cast<void>(expression);                                          \
    } catch (const std::exception&) {                                         \
      sgw_threw = true;                                                       \
    }                                                                         \
    if (!sgw_threw) {                                                         \
      ::sgw::test::fail("expected exception: " #expression, __FILE__,       \
                        __LINE__);                                             \
    }                                                                         \
  } while (false)
