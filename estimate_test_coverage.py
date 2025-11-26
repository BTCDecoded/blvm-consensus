#!/usr/bin/env python3
"""
Incremental Test Coverage Estimation for bllvm-consensus

This script estimates test coverage by analyzing:
1. Test functions (fn test_*, #[test], proptest!)
2. Source functions (pub fn, fn)
3. Module-by-module coverage
4. Test types (unit, integration, property, fuzz)
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

BLLVM_ROOT = Path("/home/user/src/BTCDecoded/bllvm-consensus")
SRC_DIR = BLLVM_ROOT / "src"
TESTS_DIR = BLLVM_ROOT / "tests"

def count_functions_in_file(file_path: Path) -> Tuple[int, int, List[str]]:
    """Count public and private functions in a Rust file."""
    if not file_path.exists():
        return 0, 0, []
    
    content = file_path.read_text()
    functions = []
    
    # Match function definitions: pub fn, fn, pub async fn, etc.
    # Exclude test functions and Kani proofs
    func_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        # Skip test functions, Kani proofs, and helper macros
        if (func_name.startswith('test_') or 
            func_name.startswith('kani_') or
            func_name.startswith('prop_') or
            func_name == 'main' or
            'test' in func_name.lower() and 'helper' not in func_name.lower()):
            continue
        functions.append(func_name)
    
    # Count public vs private
    pub_funcs = len(re.findall(r'pub\s+(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', content))
    total_funcs = len(functions)
    private_funcs = total_funcs - pub_funcs
    
    return pub_funcs, private_funcs, functions

def count_tests_in_file(file_path: Path) -> Tuple[int, Dict[str, int]]:
    """Count test functions and categorize by type."""
    if not file_path.exists():
        return 0, {}
    
    content = file_path.read_text()
    
    # Count different test types
    test_types = {
        'unit': 0,      # #[test] or fn test_*
        'property': 0,  # proptest! or prop_*
        'integration': 0,  # In integration/ directory
        'fuzz': 0,      # fuzz_target! or fuzz_*
        'kani': 0,      # #[kani::proof] (already counted separately)
    }
    
    # Count #[test] functions
    unit_tests = len(re.findall(r'#\[test\]', content))
    test_types['unit'] += unit_tests
    
    # Count fn test_* functions (may overlap with #[test])
    test_functions = len(re.findall(r'fn\s+test_[a-zA-Z0-9_]*\s*\(', content))
    test_types['unit'] = max(test_types['unit'], test_functions)
    
    # Count proptest! macros
    prop_tests = len(re.findall(r'proptest!\s*\{', content))
    test_types['property'] += prop_tests
    
    # Count prop_* functions
    prop_functions = len(re.findall(r'fn\s+prop_[a-zA-Z0-9_]*\s*\(', content))
    test_types['property'] += prop_functions
    
    # Count fuzz targets
    fuzz_tests = len(re.findall(r'fuzz_target!\s*\(|#\[fuzz\]', content))
    test_types['fuzz'] += fuzz_tests
    
    # Check if integration test
    if 'integration' in str(file_path):
        test_types['integration'] = unit_tests + prop_functions
    
    total = sum(test_types.values())
    return total, test_types

def analyze_module(module_path: Path, module_name: str) -> Dict:
    """Analyze a single module for functions and tests."""
    result = {
        'module': module_name,
        'source_file': module_path,
        'public_functions': 0,
        'private_functions': 0,
        'total_functions': 0,
        'test_files': [],
        'test_count': 0,
        'test_types': defaultdict(int),
        'coverage_estimate': 0.0,
    }
    
    # Count functions in source file
    if module_path.exists():
        pub_funcs, priv_funcs, func_list = count_functions_in_file(module_path)
        result['public_functions'] = pub_funcs
        result['private_functions'] = priv_funcs
        result['total_functions'] = pub_funcs + priv_funcs
    
    # Find related test files
    test_patterns = [
        f"*{module_name}*test*.rs",
        f"*test*{module_name}*.rs",
        f"{module_name}_tests.rs",
    ]
    
    for pattern in test_patterns:
        for test_file in TESTS_DIR.rglob(pattern):
            if test_file.is_file():
                test_count, test_types = count_tests_in_file(test_file)
                result['test_files'].append(str(test_file.relative_to(BLLVM_ROOT)))
                result['test_count'] += test_count
                for test_type, count in test_types.items():
                    result['test_types'][test_type] += count
    
    # Estimate coverage (rough heuristic)
    if result['total_functions'] > 0:
        # Assume each test covers ~1-2 functions on average
        # Public functions are more likely to be tested
        estimated_covered = min(
            result['test_count'] * 1.5,  # Each test covers ~1.5 functions
            result['total_functions']
        )
        result['coverage_estimate'] = (estimated_covered / result['total_functions']) * 100
    
    return result

def main():
    print("=" * 80)
    print("bllvm-consensus Test Coverage Estimation")
    print("=" * 80)
    print()
    
    # Find all source modules
    modules = {}
    for src_file in SRC_DIR.rglob("*.rs"):
        if src_file.name == "lib.rs" or src_file.name == "mod.rs":
            continue
        
        module_name = src_file.stem
        modules[module_name] = src_file
    
    # Analyze each module
    results = []
    total_functions = 0
    total_tests = 0
    
    for module_name, module_path in sorted(modules.items()):
        result = analyze_module(module_path, module_name)
        results.append(result)
        total_functions += result['total_functions']
        total_tests += result['test_count']
    
    # Print summary
    print("## Module-by-Module Coverage Estimate")
    print()
    print("| Module | Functions | Tests | Coverage % | Test Types |")
    print("|--------|-----------|-------|------------|------------|")
    
    for result in sorted(results, key=lambda x: x['coverage_estimate'], reverse=True):
        test_types_str = ", ".join([f"{k}:{v}" for k, v in result['test_types'].items() if v > 0])
        if not test_types_str:
            test_types_str = "none"
        
        print(f"| {result['module']} | "
              f"{result['total_functions']} ({result['public_functions']} pub) | "
              f"{result['test_count']} | "
              f"{result['coverage_estimate']:.1f}% | "
              f"{test_types_str} |")
    
    print()
    print("## Overall Statistics")
    print()
    print(f"Total Functions: {total_functions}")
    print(f"Total Tests: {total_tests}")
    print(f"Estimated Coverage: {(total_tests * 1.5 / total_functions * 100) if total_functions > 0 else 0:.1f}%")
    print()
    
    # Count test files by category
    test_file_counts = defaultdict(int)
    for test_file in TESTS_DIR.rglob("*.rs"):
        rel_path = str(test_file.relative_to(TESTS_DIR))
        if 'integration' in rel_path:
            test_file_counts['integration'] += 1
        elif 'unit' in rel_path:
            test_file_counts['unit'] += 1
        elif 'fuzzing' in rel_path or 'fuzz' in rel_path:
            test_file_counts['fuzz'] += 1
        elif 'property' in rel_path or 'proptest' in rel_path:
            test_file_counts['property'] += 1
        else:
            test_file_counts['other'] += 1
    
    print("## Test Files by Category")
    print()
    for category, count in sorted(test_file_counts.items()):
        print(f"- {category}: {count} files")
    print()
    
    # Find modules with low/no coverage
    print("## Modules with Low Coverage (< 50%)")
    print()
    low_coverage = [r for r in results if r['coverage_estimate'] < 50 and r['total_functions'] > 0]
    if low_coverage:
        for result in sorted(low_coverage, key=lambda x: x['coverage_estimate']):
            print(f"- {result['module']}: {result['coverage_estimate']:.1f}% "
                  f"({result['test_count']} tests for {result['total_functions']} functions)")
    else:
        print("None found!")
    print()

if __name__ == "__main__":
    main()

