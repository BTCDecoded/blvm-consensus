#!/bin/bash
# Coverage Analysis Script for bllvm-consensus
# Analyzes Kani proofs and test coverage without running tests

set -e

REPO_ROOT="/home/user/src/BTCDecoded/bllvm-consensus"
cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  bllvm-consensus Coverage Analysis                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Count Kani proofs by module
echo "## Kani Formal Proofs by Module"
echo ""
echo "| Module | Proofs | Functions with Proofs |"
echo "|--------|--------|----------------------|"

total_proofs=0
declare -A module_proofs
declare -A module_functions_with_proofs

for file in $(find src -name "*.rs" -type f | sort); do
    module=$(echo "$file" | sed 's|src/||' | sed 's|\.rs||' | sed 's|/|::|g')
    proofs=$(grep -c "#\[kani::proof\]" "$file" 2>/dev/null || echo "0")
    if [ "$proofs" -gt 0 ]; then
        # Count unique functions that have proofs
        functions_with_proofs=$(grep -B 5 "#\[kani::proof\]" "$file" | grep -E "^    (pub )?fn " | wc -l || echo "0")
        module_proofs["$module"]=$proofs
        module_functions_with_proofs["$module"]=$functions_with_proofs
        total_proofs=$((total_proofs + proofs))
        printf "| %-40s | %6s | %21s |\n" "$module" "$proofs" "$functions_with_proofs"
    fi
done

# Also check test files
for file in $(find tests -name "*.rs" -type f | sort); do
    module=$(echo "$file" | sed 's|tests/||' | sed 's|\.rs||' | sed 's|/|::|g')
    proofs=$(grep -c "#\[kani::proof\]" "$file" 2>/dev/null || echo "0")
    if [ "$proofs" -gt 0 ]; then
        functions_with_proofs=$(grep -B 5 "#\[kani::proof\]" "$file" | grep -E "^    (pub )?fn " | wc -l || echo "0")
        module_proofs["$module"]=$proofs
        module_functions_with_proofs["$module"]=$functions_with_proofs
        total_proofs=$((total_proofs + proofs))
        printf "| %-40s | %6s | %21s |\n" "tests::$module" "$proofs" "$functions_with_proofs"
    fi
done

echo ""
echo "**Total Kani Proofs**: $total_proofs"
echo ""

# Count test functions by category
echo "## Test Functions by Category"
echo ""
echo "| Category | Test Files | Test Functions |"
echo "|----------|------------|----------------|"

total_test_files=0
total_test_functions=0

# Count unit tests
unit_tests=$(find tests/unit -name "*.rs" -type f 2>/dev/null | wc -l || echo "0")
unit_functions=$(find tests/unit -name "*.rs" -type f -exec grep -h "^#\[test\]\|^fn test_\|^fn.*test" {} \; 2>/dev/null | wc -l || echo "0")
if [ "$unit_tests" -gt 0 ]; then
    printf "| %-30s | %11s | %15s |\n" "Unit Tests" "$unit_tests" "$unit_functions"
    total_test_files=$((total_test_files + unit_tests))
    total_test_functions=$((total_test_functions + unit_functions))
fi

# Count integration tests
integration_tests=$(find tests/integration -name "*.rs" -type f 2>/dev/null | wc -l || echo "0")
integration_functions=$(find tests/integration -name "*.rs" -type f -exec grep -h "^#\[test\]\|^fn test_\|^fn.*test" {} \; 2>/dev/null | wc -l || echo "0")
if [ "$integration_tests" -gt 0 ]; then
    printf "| %-30s | %11s | %15s |\n" "Integration Tests" "$integration_tests" "$integration_functions"
    total_test_files=$((total_test_files + integration_tests))
    total_test_functions=$((total_test_functions + integration_functions))
fi

# Count engineering tests
engineering_tests=$(find tests/engineering -name "*.rs" -type f 2>/dev/null | wc -l || echo "0")
engineering_functions=$(find tests/engineering -name "*.rs" -type f -exec grep -h "^#\[test\]\|^fn test_\|^fn.*test" {} \; 2>/dev/null | wc -l || echo "0")
if [ "$engineering_tests" -gt 0 ]; then
    printf "| %-30s | %11s | %15s |\n" "Engineering Tests" "$engineering_tests" "$engineering_functions"
    total_test_files=$((total_test_files + engineering_tests))
    total_test_functions=$((total_test_functions + engineering_functions))
fi

# Count other tests
other_tests=$(find tests -maxdepth 1 -name "*.rs" -type f 2>/dev/null | wc -l || echo "0")
other_functions=$(find tests -maxdepth 1 -name "*.rs" -type f -exec grep -h "^#\[test\]\|^fn test_\|^fn.*test" {} \; 2>/dev/null | wc -l || echo "0")
if [ "$other_tests" -gt 0 ]; then
    printf "| %-30s | %11s | %15s |\n" "Other Tests" "$other_tests" "$other_functions"
    total_test_files=$((total_test_files + other_tests))
    total_test_functions=$((total_test_functions + other_functions))
fi

echo ""
echo "**Total Test Files**: $total_test_files"
echo "**Total Test Functions**: $total_test_functions"
echo ""

# Count source functions
echo "## Source Code Statistics"
echo ""

source_files=$(find src -name "*.rs" -type f | wc -l)
source_functions=$(grep -r "^pub fn\|^fn" src --include="*.rs" | grep -v "test\|kani" | wc -l || echo "0")
source_public_functions=$(grep -r "^pub fn" src --include="*.rs" | grep -v "test\|kani" | wc -l || echo "0")

echo "| Metric | Count |"
echo "|--------|-------|"
echo "| Source Files | $source_files |"
echo "| Total Functions | $source_functions |"
echo "| Public Functions | $source_public_functions |"
echo ""

# Estimate coverage
echo "## Coverage Estimates"
echo ""

# Functions with Kani proofs (approximate - each proof typically covers 1 function)
functions_with_kani_proofs=$total_proofs

# Estimate: assume each test function tests 1-3 source functions on average
# Use conservative estimate of 1.5 functions per test
functions_covered_by_tests=$(echo "$total_test_functions * 1.5" | bc)

# Estimate total unique functions covered
# Some functions may have both Kani proofs and tests
# Conservative estimate: 70% overlap
total_functions_covered=$(echo "scale=0; $functions_with_kani_proofs + ($functions_covered_by_tests * 0.3)" | bc)

if [ "$source_functions" -gt 0 ]; then
    kani_coverage_pct=$(echo "scale=1; ($functions_with_kani_proofs * 100) / $source_functions" | bc)
    test_coverage_pct=$(echo "scale=1; ($functions_covered_by_tests * 100) / $source_functions" | bc)
    combined_coverage_pct=$(echo "scale=1; ($total_functions_covered * 100) / $source_functions" | bc)
    
    echo "| Coverage Type | Functions | Percentage |"
    echo "|---------------|-----------|------------|"
    echo "| Kani Proofs | $functions_with_kani_proofs | ${kani_coverage_pct}% |"
    echo "| Test Coverage (est.) | $(echo "$functions_covered_by_tests" | cut -d. -f1) | ${test_coverage_pct}% |"
    echo "| Combined Coverage (est.) | $(echo "$total_functions_covered" | cut -d. -f1) | ${combined_coverage_pct}% |"
    echo ""
fi

# Module-by-module breakdown
echo "## Module Coverage Breakdown"
echo ""
echo "| Module | Functions | Kani Proofs | Estimated Test Coverage |"
echo "|--------|-----------|-------------|------------------------|"

for module_file in $(find src -name "*.rs" -type f | sort); do
    module=$(echo "$module_file" | sed 's|src/||' | sed 's|\.rs||' | sed 's|/|::|g')
    module_functions=$(grep -c "^pub fn\|^fn" "$module_file" 2>/dev/null | grep -v "test\|kani" || echo "0")
    module_kani=$(grep -c "#\[kani::proof\]" "$module_file" 2>/dev/null || echo "0")
    
    if [ "$module_functions" -gt 0 ]; then
        # Estimate test coverage for this module (rough heuristic)
        module_test_estimate=$(echo "scale=0; $module_functions * 0.6" | bc)
        printf "| %-30s | %9s | %11s | %23s |\n" "$module" "$module_functions" "$module_kani" "$module_test_estimate"
    fi
done

echo ""
echo "**Note**: Coverage estimates are approximate based on static analysis."
echo "Actual coverage may vary based on test depth and proof scope."

