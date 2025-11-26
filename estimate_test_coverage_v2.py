#!/usr/bin/env python3
"""
Improved Test Coverage Estimation for bllvm-consensus

This version:
1. Counts inline tests in source files
2. Better matches test files to modules
3. Analyzes by consensus-critical vs infrastructure
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

BLLVM_ROOT = Path("/home/user/src/BTCDecoded/bllvm-consensus")
SRC_DIR = BLLVM_ROOT / "src"
TESTS_DIR = BLLVM_ROOT / "tests"

# Consensus-critical modules (highest priority)
CONSENSUS_CRITICAL = {
    'block', 'transaction', 'script', 'pow', 'economic', 
    'reorganization', 'bip_validation', 'bip113', 'bip119',
    'segwit', 'taproot', 'locktime', 'sequence_locks', 'sigop',
    'transaction_hash', 'witness'
}

# Infrastructure modules (lower priority)
INFRASTRUCTURE = {
    'network', 'mempool', 'mining', 'types', 'error', 'constants',
    'serialization', 'optimizations', 'utxo_commitments'
}

def count_functions_in_file(file_path: Path) -> Tuple[int, int, List[str], int]:
    """Count public and private functions, and inline tests."""
    if not file_path.exists():
        return 0, 0, [], 0
    
    content = file_path.read_text()
    functions = []
    
    # Match function definitions
    func_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        # Skip Kani proofs and helper macros
        if (func_name.startswith('kani_') or
            func_name == 'main'):
            continue
        functions.append(func_name)
    
    # Count public vs private
    pub_funcs = len(re.findall(r'pub\s+(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', content))
    total_funcs = len(functions)
    private_funcs = total_funcs - pub_funcs
    
    # Count inline tests
    inline_tests = len(re.findall(r'#\[test\]', content))
    inline_tests += len(re.findall(r'fn\s+test_[a-zA-Z0-9_]*\s*\(', content))
    inline_tests += len(re.findall(r'proptest!\s*\{', content))
    
    return pub_funcs, private_funcs, functions, inline_tests

def find_tests_for_module(module_name: str) -> Tuple[int, Dict[str, int]]:
    """Find all test files that test a given module."""
    test_types = defaultdict(int)
    total_tests = 0
    
    # Search for test files that might test this module
    # Pattern 1: module_name_tests.rs or test_module_name.rs
    patterns = [
        f"*{module_name}*test*.rs",
        f"*test*{module_name}*.rs",
        f"{module_name}_tests.rs",
        f"test_{module_name}.rs",
    ]
    
    # Also check for module name in file content
    for test_file in TESTS_DIR.rglob("*.rs"):
        if not test_file.is_file():
            continue
        
        file_name = test_file.stem.lower()
        content = test_file.read_text().lower()
        
        # Check if this test file is related to the module
        is_related = (
            module_name.lower() in file_name or
            f"use.*{module_name}" in content or
            f"mod {module_name}" in content or
            f"crate::{module_name}" in content
        )
        
        if is_related:
            # Count tests in this file
            file_content = test_file.read_text()
            unit = len(re.findall(r'#\[test\]', file_content))
            unit += len(re.findall(r'fn\s+test_[a-zA-Z0-9_]*\s*\(', file_content))
            prop = len(re.findall(r'proptest!\s*\{', file_content))
            prop += len(re.findall(r'fn\s+prop_[a-zA-Z0-9_]*\s*\(', file_content))
            fuzz = len(re.findall(r'fuzz_target!\s*\(', file_content))
            
            test_types['unit'] += unit
            test_types['property'] += prop
            test_types['fuzz'] += fuzz
            total_tests += unit + prop + fuzz
    
    return total_tests, test_types

def analyze_module(module_path: Path, module_name: str) -> Dict:
    """Analyze a single module."""
    result = {
        'module': module_name,
        'source_file': module_path,
        'public_functions': 0,
        'private_functions': 0,
        'total_functions': 0,
        'inline_tests': 0,
        'external_tests': 0,
        'total_tests': 0,
        'test_types': defaultdict(int),
        'coverage_estimate': 0.0,
        'category': 'other',
    }
    
    # Determine category
    if module_name in CONSENSUS_CRITICAL:
        result['category'] = 'consensus-critical'
    elif module_name in INFRASTRUCTURE:
        result['category'] = 'infrastructure'
    
    # Count functions and inline tests
    if module_path.exists():
        pub_funcs, priv_funcs, func_list, inline_tests = count_functions_in_file(module_path)
        result['public_functions'] = pub_funcs
        result['private_functions'] = priv_funcs
        result['total_functions'] = pub_funcs + priv_funcs
        result['inline_tests'] = inline_tests
    
    # Find external tests
    external_tests, test_types = find_tests_for_module(module_name)
    result['external_tests'] = external_tests
    result['total_tests'] = inline_tests + external_tests
    result['test_types'] = test_types
    if inline_tests > 0:
        result['test_types']['inline'] = inline_tests
    
    # Estimate coverage
    if result['total_functions'] > 0:
        # More conservative estimate: each test covers ~1 function
        # Public functions are more likely tested
        estimated_covered = min(
            result['total_tests'] * 1.0,  # 1 test = 1 function
            result['total_functions']
        )
        # Boost for public functions (more likely to be tested)
        pub_coverage_boost = min(result['public_functions'] * 0.3, result['total_tests'] * 0.5)
        estimated_covered = min(estimated_covered + pub_coverage_boost, result['total_functions'])
        result['coverage_estimate'] = (estimated_covered / result['total_functions']) * 100
    
    return result

def main():
    print("=" * 80)
    print("bllvm-consensus Test Coverage Estimation (v2)")
    print("=" * 80)
    print()
    
    # Find all source modules
    modules = {}
    for src_file in SRC_DIR.rglob("*.rs"):
        if src_file.name == "lib.rs":
            continue
        # Skip mod.rs files unless they're the main module file
        if src_file.name == "mod.rs" and src_file.parent != SRC_DIR:
            continue
        
        module_name = src_file.stem
        if module_name not in ['lib', 'main']:
            modules[module_name] = src_file
    
    # Analyze each module
    results = []
    consensus_results = []
    infra_results = []
    other_results = []
    
    for module_name, module_path in sorted(modules.items()):
        result = analyze_module(module_path, module_name)
        results.append(result)
        
        if result['category'] == 'consensus-critical':
            consensus_results.append(result)
        elif result['category'] == 'infrastructure':
            infra_results.append(result)
        else:
            other_results.append(result)
    
    # Print consensus-critical modules
    print("## Consensus-Critical Modules Coverage")
    print()
    print("| Module | Functions | Tests | Coverage % | Test Types |")
    print("|--------|-----------|-------|------------|------------|")
    
    for result in sorted(consensus_results, key=lambda x: x['coverage_estimate']):
        test_types_str = ", ".join([f"{k}:{v}" for k, v in result['test_types'].items() if v > 0])
        if not test_types_str:
            test_types_str = "none"
        
        print(f"| {result['module']} | "
              f"{result['total_functions']} ({result['public_functions']} pub) | "
              f"{result['total_tests']} ({result['inline_tests']} inline) | "
              f"{result['coverage_estimate']:.1f}% | "
              f"{test_types_str} |")
    
    # Summary statistics
    consensus_funcs = sum(r['total_functions'] for r in consensus_results)
    consensus_tests = sum(r['total_tests'] for r in consensus_results)
    consensus_avg = sum(r['coverage_estimate'] for r in consensus_results) / len(consensus_results) if consensus_results else 0
    
    print()
    print(f"**Consensus-Critical Summary**: {consensus_funcs} functions, {consensus_tests} tests, "
          f"avg coverage: {consensus_avg:.1f}%")
    print()
    
    # Infrastructure modules
    print("## Infrastructure Modules Coverage")
    print()
    print("| Module | Functions | Tests | Coverage % |")
    print("|--------|-----------|-------|------------|")
    
    for result in sorted(infra_results, key=lambda x: x['coverage_estimate'], reverse=True):
        print(f"| {result['module']} | "
              f"{result['total_functions']} ({result['public_functions']} pub) | "
              f"{result['total_tests']} | "
              f"{result['coverage_estimate']:.1f}% |")
    
    print()
    
    # Low coverage consensus-critical modules
    print("## ⚠️ Consensus-Critical Modules with Low Coverage (< 70%)")
    print()
    low_coverage = [r for r in consensus_results if r['coverage_estimate'] < 70 and r['total_functions'] > 0]
    if low_coverage:
        for result in sorted(low_coverage, key=lambda x: x['coverage_estimate']):
            print(f"- **{result['module']}**: {result['coverage_estimate']:.1f}% "
                  f"({result['total_tests']} tests for {result['total_functions']} functions)")
    else:
        print("✅ All consensus-critical modules have good coverage!")
    print()
    
    # Overall stats
    total_funcs = sum(r['total_functions'] for r in results)
    total_tests = sum(r['total_tests'] for r in results)
    
    print("## Overall Statistics")
    print()
    print(f"Total Functions: {total_funcs}")
    print(f"Total Tests: {total_tests}")
    print(f"Consensus-Critical Functions: {consensus_funcs}")
    print(f"Consensus-Critical Tests: {consensus_tests}")
    print(f"Consensus-Critical Coverage: {consensus_avg:.1f}%")
    print()

if __name__ == "__main__":
    main()

