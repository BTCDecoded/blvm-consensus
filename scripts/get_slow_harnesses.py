#!/usr/bin/env python3
"""Identify slow Kani proofs (unwind >= 10) and output harness names."""
import re
import os
import sys

# Slow proofs identified by unwind bounds >= 10
slow_proofs = set()

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.rs'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    # Find all proof functions with unwind >= 10
                    # Pattern: fn kani_... followed by kani::unwind(10+) within reasonable distance
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'kani::unwind(' in line:
                            match = re.search(r'unwind\((\d+)\)', line)
                            if match:
                                unwind = int(match.group(1))
                                if unwind >= 10:
                                    # Look backwards for function name
                                    for j in range(max(0, i-10), i):
                                        func_match = re.search(r'fn\s+(kani_\w+)', lines[j])
                                        if func_match:
                                            slow_proofs.add(func_match.group(1))
                                            break
            except Exception:
                pass

# Output as space-separated list for shell script
print(' '.join(sorted(slow_proofs)))

