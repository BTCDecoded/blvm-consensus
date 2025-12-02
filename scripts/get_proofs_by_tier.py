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
# These are the minimum set for "formally verified" status
# Any change to these would break Bitcoin consensus
STRONG_TIER_PROOFS = {
    # Constants (Orange Paper locking)
    'kani_monetary_constants_match_orange_paper',
    'kani_bip_activation_heights_match_bitcoin_core',
    'kani_genesis_block_constants_match_bitcoin',
    
    # Economic Model (21M cap, halving schedule)
    'kani_get_block_subsidy_halving_schedule',
    'kani_supply_limit_respected',
    'kani_get_block_subsidy_boundary_correctness',
    
    # Transaction Validation (core structure rules)
    'kani_check_transaction_invariants',
    'kani_check_transaction_no_duplicates',
    'kani_check_transaction_total_output_sum',
    'kani_check_transaction_structure',  # Core transaction structure validation
    'kani_is_coinbase_correct',  # Coinbase identification (critical)
    'kani_conservation_of_value',  # Value conservation (prevents inflation)
    
    # Block Validation (UTXO consistency)
    'kani_connect_block_utxo_consistency',
    'kani_connect_block_coinbase',
    'kani_no_double_spending',  # Double-spend prevention (critical)
    'kani_validate_block_header_complete',  # Block header validation (critical)
    'kani_calculate_tx_id_deterministic',  # TX ID determinism (critical)
    'kani_block_weight_bounded_by_max',  # Block weight limits (DoS prevention)
    
    # Inflation Prevention (critical for economic security)
    'kani_bip30_duplicate_coinbase_prevention',  # Prevents duplicate coinbase (inflation)
    'kani_coinbase_maturity_enforcement',  # Prevents premature coinbase spending
    
    # Consensus Critical BIPs
    'kani_bip34_height_encoding_correctness',  # Height encoding in coinbase (critical)
    
    # DoS Prevention (size limits)
    'kani_transaction_size_consistency',  # Transaction size limits (DoS prevention)
    'kani_script_size_limit',  # Script size limits (DoS prevention)
    
    # PoW Validation
    'kani_expand_target_valid_range',  # Target expansion validation (critical for PoW)
    
    # Script Execution (determinism and correctness)
    'kani_verify_script_correctness',
    'kani_verify_script_deterministic',  # Script determinism (critical)
    
    # Proof of Work (difficulty validation)
    'kani_check_proof_of_work_correctness',
    'kani_check_proof_of_work_deterministic',
    
    # Medium Tier - Critical Atomic Operations (unwind 5-7)
    'kani_apply_transaction_consistency',  # Atomic UTXO update operation (foundation of block validation)
    'kani_apply_transaction_mathematical_correctness',  # Mathematical correctness per Orange Paper
    'kani_connect_block_fee_subsidy_validation',  # Economic rule enforcement (prevents inflation)
    'kani_total_supply_monotonic',  # Supply only increases (economic security)
    'kani_should_reorganize_max_work',  # Core consensus rule (maximum work chain selection)
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
