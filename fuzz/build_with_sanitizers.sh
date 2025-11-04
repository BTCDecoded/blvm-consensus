#!/bin/bash
# Build fuzz targets with sanitizers for better bug detection
# Usage: ./build_with_sanitizers.sh [asan|ubsan|msan|all]

set -e

FUZZ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FUZZ_DIR/.."

SANITIZER=${1:-asan}

case "$SANITIZER" in
    asan)
        echo "Building with AddressSanitizer (ASAN)..."
        export RUSTFLAGS="-Zsanitizer=address"
        export ASAN_OPTIONS="detect_leaks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1"
        cargo +nightly fuzz build --all-targets
        ;;
    ubsan)
        echo "Building with UndefinedBehaviorSanitizer (UBSAN)..."
        export RUSTFLAGS="-Zsanitizer=undefined"
        export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:report_error_type=1"
        cargo +nightly fuzz build --all-targets
        ;;
    msan)
        echo "Building with MemorySanitizer (MSAN)..."
        echo "Note: MSAN requires instrumented libstd and may need additional setup"
        export RUSTFLAGS="-Zsanitizer=memory"
        export MSAN_OPTIONS="print_stats=1"
        cargo +nightly fuzz build --all-targets
        ;;
    all)
        echo "Building with all sanitizers (ASAN + UBSAN)..."
        export RUSTFLAGS="-Zsanitizer=address -Zsanitizer=undefined"
        export ASAN_OPTIONS="detect_leaks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1"
        export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1:report_error_type=1"
        cargo +nightly fuzz build --all-targets
        ;;
    *)
        echo "Usage: $0 [asan|ubsan|msan|all]"
        exit 1
        ;;
esac

echo "Build complete! Run with: cargo +nightly fuzz run <target>"



