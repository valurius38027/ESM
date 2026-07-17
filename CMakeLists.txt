cmake_minimum_required(VERSION 3.20)
project(sgw_esm_cpp VERSION 0.1.0 LANGUAGES CXX)

option(SGW_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CompilerWarnings.cmake)

add_library(sgw_core STATIC
  src/topk.cpp
  src/autodiff.cpp
  src/tensor.cpp
  src/config.cpp
  src/presets.cpp
  src/optimizer.cpp
  src/dataset.cpp
  src/model.cpp
  src/experiment.cpp
)
target_include_directories(sgw_core PUBLIC include)
sgw_enable_warnings(sgw_core)

if(SGW_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(sgw_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(sgw_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()


add_executable(sgw_train apps/sgw_train.cpp)
target_link_libraries(sgw_train PRIVATE sgw_core)
sgw_enable_warnings(sgw_train)

enable_testing()
add_executable(sgw_tests
  tests/test_main.cpp
  tests/test_topk.cpp
  tests/test_autodiff.cpp
  tests/test_config_optimizer.cpp
  tests/test_dataset.cpp
  tests/test_model_forward.cpp
  tests/test_gradients.cpp
  tests/test_training.cpp
  tests/test_presets.cpp
  tests/test_interventions.cpp
)
target_link_libraries(sgw_tests PRIVATE sgw_core)
target_include_directories(sgw_tests PRIVATE tests)
sgw_enable_warnings(sgw_tests)
add_test(NAME sgw_tests COMMAND sgw_tests)

add_test(
  NAME sgw_cli_determinism
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_cli.sh
          ${CMAKE_BINARY_DIR}/sgw_train ${CMAKE_BINARY_DIR}
)

add_test(
  NAME sgw_summary_lf_endings
  COMMAND bash ${CMAKE_SOURCE_DIR}/tests/test_summary.sh
          ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
)

add_test(
  NAME sgw_phase3_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test_phase3_summary.py
          ${CMAKE_SOURCE_DIR}
)

add_test(
  NAME sgw_phase4_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test_phase4_summary.py
          ${CMAKE_SOURCE_DIR}
)

add_test(
  NAME sgw_phase5_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test_phase5_summary.py
          ${CMAKE_SOURCE_DIR}
)

add_test(
  NAME sgw_phase6_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test_phase6_summary.py
          ${CMAKE_SOURCE_DIR}
)

add_test(
  NAME sgw_phase7_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test_phase7_summary.py
          ${CMAKE_SOURCE_DIR}
)

add_test(
  NAME sgw_phase8_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test_phase8_summary.py
          ${CMAKE_SOURCE_DIR}
)

add_test(
  NAME sgw_phase9_summary
  COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test_phase9_summary.py
          ${CMAKE_SOURCE_DIR}
)
