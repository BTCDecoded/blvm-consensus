#!/usr/bin/env python3
"""Identify Kani proofs by tier (strong/fast/medium/slow) based on criticality and unwind bounds.

Strong tier: Critical consensus proofs that MUST run on every push.
Fast tier: unwind <= 3 or no unwind
Medium tier: unwind 4-9
Slow tier: unwind >= 10
"""
import re
import os
import sys

# STRONG TIER: Critical proofs that run on EVERY push
# These are the MINIMUM set for "formally verified" status (10 proofs)
# Any change to these would break Bitcoin consensus
# Selected for: Criticality + Speed (all fast except one unwind 7)
STRONG_TIER_PROOFS = {
    # Economic Security (prevents inflation) - CRITICAL
    'kani_supply_limit_respected',  # 21M BTC cap (no unwind - fastest)
    'kani_conservation_of_value',  # No money creation (unwind 3 - fast)
    'kani_bip30_duplicate_coinbase_prevention',  # Prevents duplicate coinbase inflation (no unwind - fastest)
    
    # Double-Spending Prevention - CRITICAL
    'kani_no_double_spending',  # Each UTXO can only be spent once (unwind 7 - slow but critical)
    
    # Determinism (required for consensus) - CRITICAL
    'kani_calculate_tx_id_deterministic',  # TX IDs must be deterministic (no unwind - fastest)
    
    # Core Validation - CRITICAL
    'kani_is_coinbase_correct',  # Coinbase identification (no unwind - fastest)
    'kani_validate_block_header_complete',  # Block header validation (no unwind - fastest)
    
    # Constants (verify Bitcoin compatibility) - CRITICAL
    'kani_monetary_constants_match_orange_paper',  # 21M, halving interval, etc. (no unwind - fastest)
    
    # Basic Transaction Structure - CRITICAL
    'kani_check_transaction_structure',  # Core transaction validation (unwind 3 - fast)
    
    # Economic Model - CRITICAL
    'kani_get_block_subsidy_halving_schedule',  # Subsidy calculation (no unwind - fastest)
}

fast_proofs = []  # unwind <= 3 or no unwind
medium_proofs = []  # unwind 4-9
slow_proofs = []  # unwind >= 10

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.rs'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if '#[kani::proof]' in line:
                            # Look for function name in next few lines
                            for j in range(i, min(len(lines), i+10)):
                                if 'fn kani_' in lines[j]:
                                    func_match = re.search(r'fn\s+(kani_\w+)', lines[j])
                                    if func_match:
                                        proof_name = func_match.group(1)
                                        
                                        # Skip if this is a strong tier proof (handled separately)
                                        if proof_name in STRONG_TIER_PROOFS:
                                            break
                                        
                                        # Check unwind bound (look ahead up to 15 lines from proof)
                                        unwind = None
                                        for k in range(i, min(len(lines), i+15)):
                                            if 'kani::unwind(' in lines[k]:
                                                unwind_match = re.search(r'unwind\((\d+)\)', lines[k])
                                                if unwind_match:
                                                    unwind = int(unwind_match.group(1))
                                                    break
                                        
                                        if unwind is None:
                                            fast_proofs.append(proof_name)
                                        elif unwind <= 3:
                                            fast_proofs.append(proof_name)
                                        elif unwind <= 9:
                                            medium_proofs.append(proof_name)
                                        else:
                                            slow_proofs.append(proof_name)
                                        break
            except Exception:
                pass

tier = sys.argv[1] if len(sys.argv) > 1 else 'all'

if tier == 'strong':
    # Strong tier: Critical proofs only (always run)
    proofs = sorted(STRONG_TIER_PROOFS)
elif tier == 'fast':
    proofs = fast_proofs
elif tier == 'fast_medium':
    proofs = fast_proofs + medium_proofs
elif tier == 'all':
    # All tier includes strong tier + fast + medium + slow
    proofs = sorted(STRONG_TIER_PROOFS) + fast_proofs + medium_proofs + slow_proofs
else:
    proofs = []

# Output as space-separated list for shell script
print(' '.join(sorted(proofs)))
