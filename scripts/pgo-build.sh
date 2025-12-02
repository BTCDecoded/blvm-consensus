#!/bin/bash
# Profile-Guided Optimization (PGO) Build Script
#
# This script automates the PGO build process for bllvm-consensus:
# 1. Build with instrumentation to collect profile data
# 2. Run benchmarks/tests to generate profile
# 3. Build optimized binary using profile data
#
# Usage: ./scripts/pgo-build.sh [clean]
#   clean: Remove existing profile data before building

set -e

PROFILE_DIR="/tmp/pgo-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Clean profile data if requested
if [ "$1" = "clean" ]; then
    echo "Cleaning existing profile data..."
    rm -rf "$PROFILE_DIR"
    echo "Profile data cleaned."
fi

# Create profile directory
mkdir -p "$PROFILE_DIR"

echo "=========================================="
echo "Profile-Guided Optimization (PGO) Build"
echo "=========================================="
echo ""

# Step 1: Build with instrumentation
echo "Step 1/3: Building with PGO instrumentation..."
echo "Profile data will be written to: $PROFILE_DIR"
echo ""

RUSTFLAGS="-C profile-generate=$PROFILE_DIR" \
    cargo build --release

if [ $? -ne 0 ]; then
    echo "Error: Build with instrumentation failed"
    exit 1
fi

echo ""
echo "✓ Build with instrumentation complete"
echo ""

# Step 2: Generate profile data
echo "Step 2/3: Running benchmarks to generate profile data..."
echo "This may take several minutes..."
echo ""

# Try to run benchmarks if available
if cargo bench --help > /dev/null 2>&1; then
    echo "Running benchmarks..."
    RUSTFLAGS="-C profile-generate=$PROFILE_DIR" \
        cargo bench --release || echo "Warning: Benchmarks failed or not available, continuing..."
else
    echo "Benchmarks not available, running tests instead..."
    RUSTFLAGS="-C profile-generate=$PROFILE_DIR" \
        cargo test --release --lib || echo "Warning: Tests failed, continuing..."
fi

# Also run a representative workload if possible
echo ""
echo "Running representative workload..."
# You can add specific test commands here that exercise hot paths
# For example: cargo test --release --test integration_tests

echo ""
echo "✓ Profile data generation complete"
echo "Profile data size: $(du -sh $PROFILE_DIR | cut -f1)"
echo ""

# Step 3: Build optimized binary
echo "Step 3/3: Building optimized binary with profile data..."
echo ""

RUSTFLAGS="-C profile-use=$PROFILE_DIR" \
    cargo build --release

if [ $? -ne 0 ]; then
    echo "Error: Build with profile data failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ PGO build complete!"
echo "=========================================="
echo ""
echo "Optimized binary is in: target/release/"
echo "Profile data is in: $PROFILE_DIR"
echo ""
echo "To clean profile data and rebuild: ./scripts/pgo-build.sh clean"
echo ""


