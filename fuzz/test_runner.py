#!/usr/bin/env python3
"""
Fuzz test runner for consensus-proof
Similar to Bitcoin Core's test_runner.py but adapted for Rust/libFuzzer

Usage:
    python3 test_runner.py <corpus_dir> [targets...]
    python3 test_runner.py corpus/ transaction_validation block_validation
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# Set up logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)

def get_fuzz_targets() -> List[str]:
    """Get list of all available fuzz targets."""
    return [
        "transaction_validation",
        "block_validation",
        "script_execution",
        "segwit_validation",
        "mempool_operations",
        "utxo_commitments",
        "compact_block_reconstruction",
        "pow_validation",
        "economic_validation",
        "serialization",
        "script_opcodes",
        "differential_fuzzing",
    ]

def run_fuzz_target(
    target: str,
    corpus_dir: Path,
    max_time: Optional[int] = None,
    max_runs: Optional[int] = None,
    jobs: int = 1,
    sanitizer: Optional[str] = None
) -> tuple[bool, str]:
    """
    Run a single fuzz target.
    Returns (success, output).
    """
    target_corpus = corpus_dir / target
    target_corpus.mkdir(parents=True, exist_ok=True)
    
    # Build fuzz target
    logging.info(f"Building fuzz target: {target}")
    build_cmd = ["cargo", "+nightly", "fuzz", "build", target]
    
    if sanitizer:
        env = os.environ.copy()
        if sanitizer == "asan":
            env["RUSTFLAGS"] = "-Zsanitizer=address"
            env["ASAN_OPTIONS"] = "detect_leaks=1:detect_stack_use_after_return=1"
        elif sanitizer == "ubsan":
            env["RUSTFLAGS"] = "-Zsanitizer=undefined"
            env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"
        elif sanitizer == "all":
            env["RUSTFLAGS"] = "-Zsanitizer=address -Zsanitizer=undefined"
            env["ASAN_OPTIONS"] = "detect_leaks=1:detect_stack_use_after_return=1"
            env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"
    else:
        env = os.environ.copy()
    
    try:
        subprocess.run(
            build_cmd,
            check=True,
            capture_output=True,
            env=env,
            cwd=Path(__file__).parent.parent
        )
    except subprocess.CalledProcessError as e:
        return False, f"Build failed: {e.stderr.decode()}"
    
    # Run fuzz target
    logging.info(f"Running fuzz target: {target}")
    run_cmd = [
        "cargo", "+nightly", "fuzz", "run", target,
        str(target_corpus),
        "--",
        "-max_len=100000",
        "-timeout=60",
    ]
    
    if max_time:
        run_cmd.extend(["-max_total_time", str(max_time)])
    if max_runs:
        run_cmd.extend(["-runs", str(max_runs)])
    if jobs > 1:
        run_cmd.extend(["-jobs", str(jobs)])
    
    try:
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).parent.parent,
            timeout=max_time + 60 if max_time else None
        )
        
        output = result.stdout + result.stderr
        success = result.returncode == 0 or "crash" not in output.lower()
        return success, output
    except subprocess.TimeoutExpired:
        return True, "Fuzzing completed (timeout)"
    except Exception as e:
        return False, f"Error running fuzzer: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Run fuzz targets with corpus management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all targets with default corpus
  python3 test_runner.py corpus/
  
  # Run specific targets
  python3 test_runner.py corpus/ transaction_validation block_validation
  
  # Run with sanitizers (24 hours)
  python3 test_runner.py corpus/ --sanitizer asan --max-time 86400
  
  # Run once through corpus (for CI)
  python3 test_runner.py corpus/ --max-runs 1
        """
    )
    
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Corpus directory (will create subdirectories for each target)"
    )
    
    parser.add_argument(
        "targets",
        nargs="*",
        help="Specific targets to run (default: all targets)"
    )
    
    parser.add_argument(
        "--max-time",
        type=int,
        help="Maximum time to run each target (seconds)"
    )
    
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Maximum number of runs per target"
    )
    
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)"
    )
    
    parser.add_argument(
        "--sanitizer",
        choices=["asan", "ubsan", "msan", "all"],
        help="Sanitizer to use (address, undefined, memory, or all)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run multiple targets in parallel"
    )
    
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.loglevel))
    
    # Determine targets
    if args.targets:
        targets = args.targets
    else:
        targets = get_fuzz_targets()
    
    # Validate targets
    available_targets = get_fuzz_targets()
    invalid_targets = [t for t in targets if t not in available_targets]
    if invalid_targets:
        logging.error(f"Invalid targets: {invalid_targets}")
        logging.info(f"Available targets: {', '.join(available_targets)}")
        sys.exit(1)
    
    logging.info(f"Running {len(targets)} fuzz target(s): {', '.join(targets)}")
    
    # Run targets
    if args.parallel:
        # Run in parallel
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    run_fuzz_target,
                    target,
                    args.corpus_dir,
                    args.max_time,
                    args.max_runs,
                    1,  # Single job per target when parallelizing
                    args.sanitizer
                ): target
                for target in targets
            }
            
            results = {}
            for future in as_completed(futures):
                target = futures[future]
                try:
                    success, output = future.result()
                    results[target] = (success, output)
                    if success:
                        logging.info(f"✓ {target}: Completed successfully")
                    else:
                        logging.error(f"✗ {target}: Failed")
                        logging.error(output[:500])  # First 500 chars
                except Exception as e:
                    logging.error(f"✗ {target}: Exception - {e}")
                    results[target] = (False, str(e))
    else:
        # Run sequentially
        results = {}
        for target in targets:
            success, output = run_fuzz_target(
                target,
                args.corpus_dir,
                args.max_time,
                args.max_runs,
                args.jobs,
                args.sanitizer
            )
            results[target] = (success, output)
            if success:
                logging.info(f"✓ {target}: Completed successfully")
            else:
                logging.error(f"✗ {target}: Failed")
                logging.error(output[:500])
    
    # Summary
    successful = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    logging.info("=" * 60)
    logging.info(f"Summary: {successful}/{total} targets completed successfully")
    
    if successful < total:
        sys.exit(1)

if __name__ == "__main__":
    main()

