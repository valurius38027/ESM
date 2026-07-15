#!/usr/bin/env bash
set -euo pipefail

profile=${1:-}
case "$profile" in
  gcc)
    build_dir=build
    rm -rf "$build_dir"
    timeout 110s cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Debug
    ;;
  clang)
    build_dir=build-clang
    rm -rf "$build_dir"
    timeout 110s env CC=clang CXX=clang++ cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Debug
    ;;
  sanitize)
    build_dir=build-sanitize
    rm -rf "$build_dir"
    timeout 110s cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSGW_ENABLE_SANITIZERS=ON
    ;;
  release)
    build_dir=build-release
    rm -rf "$build_dir"
    timeout 110s cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release
    ;;
  *)
    echo "Usage: scripts/run_validation.sh gcc|clang|sanitize|release" >&2
    exit 2
    ;;
esac

timeout 110s cmake --build "$build_dir"
timeout 110s ctest --test-dir "$build_dir" --output-on-failure
