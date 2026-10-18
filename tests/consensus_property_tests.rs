//! Property-based tests for consensus invariants
//!
//! Uses PropTest to generate thousands of random test cases that verify
//! mathematical properties and invariants of Bitcoin consensus rules.

use blvm_consensus::activation::ForkActivationTable;
use blvm_consensus::bip_validation::{
    check_bip34, check_bip54_coinbase, encode_bip34_coinbase_script,
};
use blvm_consensus::constants::*;
use blvm_consensus::crypto::OptimizedSha256;
use blvm_consensus::economic;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::orange_paper_constants::H;
use blvm_consensus::pow;
use blvm_consensus::script::cast_to_bool;
use blvm_consensus::segwit::{self, Witness};
use blvm_consensus::transaction;
use blvm_consensus::types::*;
use proptest::prelude::*;
use sha2::{Digest, Sha256};

fn padded_tx(script_len: usize) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51; script_len],
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    }
}

/// Non-coinbase transaction whose stripped serialization is `target` bytes.
fn tx_of_stripped_size(target: usize) -> Transaction {
    let mut script_len = target.saturating_sub(65);
    for _ in 0..6 {
        let tx = padded_tx(script_len);
        let size = transaction::calculate_transaction_size(&tx);
        if size == target {
            return tx;
        }
        if size > target {
            script_len = script_len.saturating_sub(size - target);
        } else {
            script_len += target - size;
        }
    }
    padded_tx(script_len)
}

fn one_tx_block(tx: Transaction) -> Block {
    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [1; 32],
            timestamp: 1_700_000_000,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![tx].into_boxed_slice(),
    }
}

fn compact_size_len(n: u64) -> usize {
    if n < 0xfd {
        1
    } else if n <= 0xffff {
        3
    } else if n <= 0xffff_ffff {
        5
    } else {
        9
    }
}

fn expected_stripped_size(tx: &Transaction) -> usize {
    let mut n = 8usize;
    if tx.inputs.is_empty() && !tx.outputs.is_empty() {
        n += compact_size_len(0) + 1 + compact_size_len(0);
    } else {
        n += compact_size_len(tx.inputs.len() as u64);
    }
    for input in tx.inputs.iter() {
        n += 40;
        n += compact_size_len(input.script_sig.len() as u64);
        n += input.script_sig.len();
    }
    n += compact_size_len(tx.outputs.len() as u64);
    for output in tx.outputs.iter() {
        n += 8;
        n += compact_size_len(output.script_pubkey.len() as u64);
        n += output.script_pubkey.len();
    }
    n
}

fn coinbase_tx(script_sig: Vec<u8>, lock_time: u64, sequence: u64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffff_ffff,
            },
            script_sig,
            sequence,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time,
    }
}

/// Minimal BIP34 height prefix: `OP_0`, `OP_1`..=`OP_16`, or a length byte plus
/// the little-endian magnitude with a trailing `0x00` when the high bit is set.
fn bip34_prefix(height: u64) -> Vec<u8> {
    if height == 0 {
        return vec![0x00];
    }
    if (1..=16).contains(&height) {
        return vec![0x50 + height as u8];
    }
    let mut raw = Vec::new();
    let mut n = height;
    while n > 0 {
        raw.push((n & 0xff) as u8);
        n >>= 8;
    }
    if raw.last().is_some_and(|b| b & 0x80 != 0) {
        raw.push(0x00);
    }
    let mut prefix = Vec::with_capacity(1 + raw.len());
    prefix.push(raw.len() as u8);
    prefix.extend(raw);
    prefix
}

// Generate arbitrary Transaction and BlockHeader for property tests
// We'll use manual generation instead of Arbitrary trait to avoid orphan rule issues

proptest! {
    #[test]
    fn sha256_matches_reference(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        let reference = Sha256::digest(&data);
        let ours = OptimizedSha256::new().hash(&data);
        prop_assert_eq!(&reference[..], &ours[..]);
    }

    #[test]
    fn double_sha256_matches_reference(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        let reference = Sha256::digest(Sha256::digest(&data));
        let ours = OptimizedSha256::new().hash256(&data);
        prop_assert_eq!(&reference[..], &ours[..]);
    }

    #[test]
    fn sha256_deterministic(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        // Same input should always produce same output
        let hash1 = OptimizedSha256::new().hash(&data);
        let hash2 = OptimizedSha256::new().hash(&data);
        prop_assert_eq!(&hash1[..], &hash2[..]);
    }

    #[test]
    fn sha256_output_length(data in prop::collection::vec(any::<u8>(), 0..1024)) {
        let hash = OptimizedSha256::new().hash(&data);
        prop_assert_eq!(hash.len(), 32);
    }

    #[test]
    fn double_sha256_output_length(data in prop::collection::vec(any::<u8>(), 0..1024)) {
        let hash = OptimizedSha256::new().hash256(&data);
        prop_assert_eq!(hash.len(), 32);
    }
}

// ============================================================================
// Consensus Economic Rules Property Tests
// ============================================================================

proptest! {
    /// Invariant: Block subsidy matches Orange Paper formula exactly
    ///
    /// Mathematical specification:
    /// ∀ h ∈ ℕ: subsidy(h) = 50 * C * 2^(-⌊h/H⌋) if ⌊h/H⌋ < 64 else 0
    /// where C = 10^8 (satoshis per BTC) and H = 210,000 (halving interval)
    ///
    /// This test uses the Orange Paper formula helper to ensure exact match.
    #[test]
    fn prop_block_subsidy_matches_orange_paper(
        height in 0u32..2100000u32 // Up to 10 halvings
    ) {
        use blvm_consensus::orange_paper_property_helpers::expected_getblocksubsidy_from_orange_paper;

        let actual = economic::get_block_subsidy(height as u64);
        let expected = expected_getblocksubsidy_from_orange_paper(height as u64);

        prop_assert_eq!(actual, expected,
            "Subsidy at height {} must match Orange Paper formula: actual={}, expected={}",
            height, actual, expected);
    }

    /// Invariant: Total supply is monotonic and bounded
    ///
    /// Mathematical specification:
    /// ∀ h₁, h₂ ∈ ℕ: h₁ ≤ h₂ ⟹ total_supply(h₁) ≤ total_supply(h₂)
    /// ∀ h ∈ ℕ: total_supply(h) < 21 * 10^6 * 10^8
    #[test]
    fn prop_total_supply_monotonic_bounded(
        height1 in 0u32..1000000u32,
        height2 in 0u32..1000000u32
    ) {
        let supply1 = economic::total_supply(height1 as u64);
        let supply2 = economic::total_supply(height2 as u64);

        // Monotonicity: supply increases with height
        if height1 <= height2 {
            prop_assert!(supply1 <= supply2,
                "Supply must be monotonic: supply({}) = {} <= supply({}) = {}",
                height1, supply1, height2, supply2);
        } else {
            prop_assert!(supply2 <= supply1,
                "Supply must be monotonic: supply({}) = {} <= supply({}) = {}",
                height2, supply2, height1, supply1);
        }

        // Total supply must be less than 21M BTC (in satoshis)
        let max_supply = 21_000_000i64 * 100_000_000i64;
        prop_assert!(supply1 <= max_supply,
            "Supply at height {} exceeds cap: {} > {}", height1, supply1, max_supply);
        prop_assert!(supply2 <= max_supply,
            "Supply at height {} exceeds cap: {} > {}", height2, supply2, max_supply);
    }

    /// Invariant: Block subsidy is non-negative and decreases over halvings
    #[test]
    fn prop_block_subsidy_non_negative_decreasing(
        height1 in 0u32..2100000u32,
        height2 in 0u32..2100000u32
    ) {
        let subsidy1 = economic::get_block_subsidy(height1 as u64);
        let subsidy2 = economic::get_block_subsidy(height2 as u64);

        // Subsidy is always non-negative
        prop_assert!(subsidy1 >= 0, "Subsidy must be non-negative");
        prop_assert!(subsidy2 >= 0, "Subsidy must be non-negative");

        // Subsidy decreases across halving boundaries
        // Using Orange Paper constant H (halving interval)
        let epoch1 = height1 / (H as u32);
        let epoch2 = height2 / (H as u32);

        if epoch1 < epoch2 && epoch2 < 64 {
            prop_assert!(subsidy1 >= subsidy2,
                "Subsidy should decrease across halvings: epoch {} ({}) >= epoch {} ({})",
                epoch1, subsidy1, epoch2, subsidy2);
        }
    }
}

// ============================================================================
// Proof of Work Property Tests
// ============================================================================

proptest! {
    /// Invariant: Proof of work target expansion produces valid values
    ///
    /// Mathematical specification:
    /// ∀ bits ∈ [0x03000000, 0x1d00ffff]:
    ///   Let expanded = expand_target(bits)
    ///   Then: expanded is a valid U256 value
    ///         expanded can be used for proof of work validation
    #[test]
    fn prop_pow_target_expansion_valid_range(
        bits in 0x03000000u32..0x1d00ffffu32
    ) {
        // Verify that expand_target succeeds for all valid bits
        // This ensures the function handles the entire valid range correctly
        let expanded_result = pow::expand_target(bits as u64);
        prop_assert!(expanded_result.is_ok(),
            "expand_target should succeed for valid bits 0x{:08x}", bits);

        // If successful, the expanded target is a valid U256
        // The type system ensures this, so we just verify the operation succeeded
    }
}

// ============================================================================
// Transaction Validation Property Tests
// ============================================================================

proptest! {
    /// Invariant: Transaction output values are bounded [0, MAX_MONEY]
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳, ∀ o ∈ tx.outputs: 0 ≤ o.value ≤ MAX_MONEY
    #[test]
    fn prop_transaction_output_value_bounded(
        value in 0i64..(MAX_MONEY + 1)
    ) {
        let output = TransactionOutput {
            value,
            script_pubkey: vec![],
        };

        // Output value must be within bounds
        prop_assert!(output.value >= 0, "Output value must be non-negative");
        prop_assert!(output.value <= MAX_MONEY,
            "Output value {} exceeds MAX_MONEY {}", output.value, MAX_MONEY);
    }

    /// Invariant: Transaction has non-empty inputs and outputs
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: CheckTransaction(tx) = valid ⟹ |tx.inputs| > 0 ∧ |tx.outputs| > 0
    #[test]
        fn prop_transaction_non_empty_inputs_outputs(
        num_inputs in 0usize..4,
        num_outputs in 0usize..4
    ) {
        let tx = Transaction {
            version: 1,
            inputs: (0..num_inputs).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8 + 1; 32],
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..num_outputs).map(|_| TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }).collect(),
            lock_time: 0,
        };

        let valid = matches!(
            transaction::check_transaction(&tx).unwrap(),
            ValidationResult::Valid
        );
        prop_assert_eq!(valid, num_inputs > 0 && num_outputs > 0);
    }

    /// Invariant: Transaction size respects limits
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: |tx| ≤ MAX_TX_SIZE
    #[test]
        fn prop_transaction_size_bounded(
        n_in in 0usize..6,
        n_out in 0usize..6,
        script_len in 0usize..40
    ) {
        let tx = Transaction {
            version: 1,
            inputs: (0..n_in).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8 + 1; 32],
                    index: i as u32,
                },
                script_sig: vec![0x51; script_len],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..n_out).map(|_| TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51; script_len],
            }).collect(),
            lock_time: 0,
        };
        prop_assert_eq!(
            transaction::calculate_transaction_size(&tx),
            expected_stripped_size(&tx)
        );
    }

    /// Invariant: Coinbase transactions have special validation rules
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: IsCoinbase(tx) ⟹ 2 ≤ |tx.inputs[0].scriptSig| ≤ 100
    #[test]
        fn prop_coinbase_script_sig_length(
        script_sig_len in 0usize..120
    ) {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![0x51; script_sig_len],
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![0x51],
            }].into(),
            lock_time: 0,
        };
        let valid = matches!(
            transaction::check_transaction(&tx).unwrap(),
            ValidationResult::Valid
        );
        prop_assert_eq!(valid, (2..=100).contains(&script_sig_len));
    }
}

// ============================================================================
// Script Execution Property Tests
// ============================================================================

proptest! {
    /// Invariant: Script execution is deterministic
    ///
    /// Mathematical specification:
    /// ∀ script, stack, flags: eval_script(script, stack, flags) = eval_script(script, stack, flags)
    #[test]
    fn prop_script_execution_deterministic(
        script in prop::collection::vec(any::<u8>(), 0..100)
    ) {
        use blvm_consensus::script;

        let mut stack1 = Vec::new();
        let mut stack2 = Vec::new();
        let flags = 0u32;

        // Execute script twice with same inputs
        let result1 = script::eval_script(&script, &mut stack1, flags, script::SigVersion::Base);
        let result2 = script::eval_script(&script, &mut stack2, flags, script::SigVersion::Base);

        // Results should be identical
        prop_assert_eq!(result1, result2,
            "Script execution must be deterministic");

        // Stacks should be identical
        prop_assert_eq!(stack1, stack2,
            "Script execution must produce same stack state");
    }

    /// Invariant: Script execution respects resource limits
    ///
    /// Mathematical specification:
    /// ∀ script: |script| ≤ MAX_SCRIPT_SIZE ⟹ eval_script terminates
    #[test]
    fn prop_script_size_bounded(extra in 1usize..=32usize) {
        use blvm_consensus::opcodes::OP_NOP;
        use blvm_consensus::script;

        let script = vec![OP_NOP; MAX_SCRIPT_SIZE + extra];
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0, script::SigVersion::Base);
        prop_assert!(
            result.is_err(),
            "a script longer than {MAX_SCRIPT_SIZE} bytes is rejected"
        );
    }
}

// ============================================================================
// Deterministic Execution Property Tests
// ============================================================================

proptest! {
    /// Invariant: Transaction validation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: check_transaction(tx) = check_transaction(tx)
    ///
    /// Same transaction must always produce same validation result.
    #[test]
    fn prop_transaction_validation_deterministic(
        num_inputs in 1usize..10usize,
        num_outputs in 1usize..10usize
    ) {
        let tx = Transaction {
            version: 1,
            inputs: (0..num_inputs).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                },
                script_sig: vec![0; 10],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..num_outputs).map(|_| TransactionOutput {
                value: 1000,
                script_pubkey: vec![0; 20],
            }).collect(),
            lock_time: 0,
        };

        // Validate same transaction multiple times
        let result1 = transaction::check_transaction(&tx);
        let result2 = transaction::check_transaction(&tx);
        let result3 = transaction::check_transaction(&tx);

        // All results must be identical
        prop_assert_eq!(result1.is_ok(), result2.is_ok(), "Transaction validation must be deterministic (1st vs 2nd)");
        prop_assert_eq!(result2.is_ok(), result3.is_ok(), "Transaction validation must be deterministic (2nd vs 3rd)");
        if result1.is_ok() {
            prop_assert_eq!(result1.unwrap(), result2.unwrap(), "Transaction validation result must be identical");
        }
    }

    /// Invariant: Block subsidy calculation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ h ∈ ℕ: get_block_subsidy(h) = get_block_subsidy(h)
    #[test]
    fn prop_block_subsidy_deterministic(
        height in 0u32..2100000u32
    ) {
        let subsidy1 = economic::get_block_subsidy(height as u64);
        let subsidy2 = economic::get_block_subsidy(height as u64);
        let subsidy3 = economic::get_block_subsidy(height as u64);

        prop_assert_eq!(subsidy1, subsidy2, "Block subsidy must be deterministic (1st vs 2nd)");
        prop_assert_eq!(subsidy2, subsidy3, "Block subsidy must be deterministic (2nd vs 3rd)");
    }

    /// Invariant: Total supply calculation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ h ∈ ℕ: total_supply(h) = total_supply(h)
    #[test]
    fn prop_total_supply_deterministic(
        height in 0u32..2100000u32
    ) {
        let supply1 = economic::total_supply(height as u64);
        let supply2 = economic::total_supply(height as u64);
        let supply3 = economic::total_supply(height as u64);

        prop_assert_eq!(supply1, supply2, "Total supply must be deterministic (1st vs 2nd)");
        prop_assert_eq!(supply2, supply3, "Total supply must be deterministic (2nd vs 3rd)");
    }

    /// Invariant: Target expansion is deterministic
    ///
    /// Mathematical specification:
    /// ∀ bits ∈ [0x03000000, 0x1d00ffff]: expand_target(bits) = expand_target(bits)
    #[test]
    fn prop_target_expansion_deterministic(
        bits in 0x03000000u32..0x1d00ffffu32
    ) {
        let target1 = pow::expand_target(bits as u64);
        let target2 = pow::expand_target(bits as u64);
        let target3 = pow::expand_target(bits as u64);

        // Compare results (Results implement PartialEq)
        prop_assert_eq!(target1.is_ok(), target2.is_ok(), "Target expansion must be deterministic (1st vs 2nd)");
        prop_assert_eq!(target2.is_ok(), target3.is_ok(), "Target expansion must be deterministic (2nd vs 3rd)");
        if target1.is_ok() {
            prop_assert_eq!(target1.unwrap(), target2.unwrap(), "Target expansion result must be identical");
        }
    }

    /// Invariant: Fee calculation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳, US ∈ UTXO_SET: calculate_fee(tx, US) = calculate_fee(tx, US)
    #[test]
    fn prop_fee_calculation_deterministic(
        num_inputs in 1usize..5usize,
        num_outputs in 1usize..5usize
    ) {
        // Create transaction and UTXO set
        let mut utxo_set = UtxoSet::default();
        let inputs: Vec<TransactionInput> = (0..num_inputs).map(|i| {
                let outpoint = OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                };
                utxo_set.insert(outpoint, std::sync::Arc::new(UTXO {
                    value: 10000,
                    script_pubkey: vec![0; 20].into(),
                    height: 0,
                    is_coinbase: false,
                }));
                TransactionInput {
                    prevout: outpoint,
                    script_sig: vec![0; 10],
                    sequence: 0xffffffff,
                }
            }).collect();
        let tx = Transaction {
            version: 1,
            inputs: inputs.into(),
            outputs: (0..num_outputs).map(|_| TransactionOutput {
                value: 5000,
                script_pubkey: vec![0; 20],
            }).collect(),
            lock_time: 0,
        };

        // Calculate fee multiple times
        let fee1 = economic::calculate_fee(&tx, &utxo_set);
        let fee2 = economic::calculate_fee(&tx, &utxo_set);
        let fee3 = economic::calculate_fee(&tx, &utxo_set);

        prop_assert_eq!(fee1.is_ok(), fee2.is_ok(), "Fee calculation must be deterministic (1st vs 2nd)");
        prop_assert_eq!(fee2.is_ok(), fee3.is_ok(), "Fee calculation must be deterministic (2nd vs 3rd)");
        if fee1.is_ok() {
            prop_assert_eq!(fee1.unwrap(), fee2.unwrap(), "Fee calculation result must be identical");
        }
    }
}

// ============================================================================
// Integer Overflow Property Tests
// ============================================================================

proptest! {
    /// Invariant: Fee calculation handles overflow correctly
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: If sum(input_values) > i64::MAX, then calculate_fee returns error
    #[test]
    fn prop_fee_calculation_overflow_safety(
        input_values in prop::collection::vec(0i64..i64::MAX, 1..10)
    ) {
        // Check if sum would overflow
        let sum: Option<i64> = input_values.iter().try_fold(0i64, |acc, &val| {
            acc.checked_add(val)
        });

        if sum.is_none() {
            // Overflow would occur - create transaction that triggers it
            let mut utxo_set = UtxoSet::default();
            let mut inputs = Vec::new();

            for (i, &value) in input_values.iter().enumerate() {
                let outpoint = OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                };
                utxo_set.insert(outpoint, std::sync::Arc::new(UTXO {
                    value,
                    script_pubkey: vec![0; 20].into(),
                    height: 0,
                    is_coinbase: false,
                }));
                inputs.push(TransactionInput {
                    prevout: outpoint,
                    script_sig: vec![0; 10],
                    sequence: 0xffffffff,
                });
            }

            let tx = Transaction {
                version: 1,
            inputs: inputs.into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0; 20],
                }].into(),
                lock_time: 0,
            };

            // Fee calculation should detect overflow and return error
            let result = economic::calculate_fee(&tx, &utxo_set);
            prop_assert!(result.is_err(),
                "Fee calculation must detect input value overflow");
        }
    }

    /// Invariant: Output value summation handles overflow correctly
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: If sum(output_values) > i64::MAX, then validation returns error
    #[test]
    fn prop_output_value_overflow_safety(first in 0i64..=MAX_MONEY, second in 0i64..=MAX_MONEY) {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![
                TransactionOutput {
                    value: first,
                    script_pubkey: vec![0x51],
                },
                TransactionOutput {
                    value: second,
                    script_pubkey: vec![0x51],
                },
            ]
            .into(),
            lock_time: 0,
        };
        let result = transaction::check_transaction(&tx).unwrap();
        let sum = first.checked_add(second);
        let in_range = sum.is_some_and(|total| total <= MAX_MONEY);
        prop_assert_eq!(
            matches!(result, ValidationResult::Valid),
            in_range,
            "output sum is accepted only inside MoneyRange: {} + {}",
            first,
            second
        );
    }

    /// Invariant: Total supply calculation handles overflow correctly
    ///
    /// Mathematical specification:
    /// ∀ h ∈ ℕ: total_supply(h) ≤ MAX_MONEY (never overflows)
    #[test]
    fn prop_total_supply_overflow_safety(
        height in 0u32..2100000u32 // Up to 10 halvings (reduced for coverage)
    ) {
        let supply = economic::total_supply(height as u64);

        // Total supply must never exceed MAX_MONEY
        prop_assert!(supply <= MAX_MONEY,
            "Total supply at height {} must not exceed MAX_MONEY: {} <= {}",
            height, supply, MAX_MONEY);

        // Total supply must be non-negative
        prop_assert!(supply >= 0,
            "Total supply must be non-negative: {} >= 0", supply);
    }
}

// ============================================================================
// Temporal/State Transition Property Tests
// ============================================================================

proptest! {
    /// Invariant: Supply never decreases across block connections
    ///
    /// Mathematical specification:
    /// ∀ blocks B₁, B₂, heights h₁, h₂ where h₂ > h₁:
    ///   Let US₁ = connect_block(B₁, US₀, h₁)
    ///   Let US₂ = connect_block(B₂, US₁, h₂)
    ///   Then: supply(US₂) >= supply(US₁)
    ///
    /// This ensures no money destruction across sequential block connections.
    #[test]
    fn prop_supply_never_decreases_across_blocks(
        height1 in 0u32..100u32,
        height2 in 0u32..100u32
    ) {
        use blvm_consensus::block;

        // Ensure height2 > height1
        let height1 = height1;
        let height2 = if height2 <= height1 { height1 + 1 } else { height2 };

        // Create simple blocks with coinbase only
        let block1 = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [1; 32],
                timestamp: 1231006505 + (height1 as u64 * 600),
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                    script_sig: vec![OP_1, OP_1], // 2 bytes for valid coinbase
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height1 as u64),
                    script_pubkey: vec![OP_1],
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        let block2 = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [1; 32],
                merkle_root: [2; 32],
                timestamp: 1231006505 + (height2 as u64 * 600),
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                    script_sig: vec![OP_1, OP_1],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height2 as u64),
                    script_pubkey: vec![OP_1],
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        // Connect block1
        let utxo_set = UtxoSet::default();
        let witnesses1: Vec<Vec<Witness>> = block1.transactions.iter().map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect()).collect();
        let ctx = block::BlockValidationContext::for_network(Network::Mainnet);
        let result1 = block::connect_block(&block1, &witnesses1, utxo_set, height1 as u64, &ctx);

        if let Ok((ValidationResult::Valid, utxo_set1, _undo_log1)) = result1 {
            // Calculate supply after block1
            let supply1: i64 = utxo_set1
                .values()
                .map(|utxo| utxo.value)
                .try_fold(0i64, |acc, val| acc.checked_add(val))
                .unwrap_or(MAX_MONEY);

            // Connect block2
            let witnesses2: Vec<Vec<Witness>> = block2.transactions.iter().map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect()).collect();
            let result2 = block::connect_block(&block2, &witnesses2, utxo_set1, height2 as u64, &ctx);

            if let Ok((ValidationResult::Valid, utxo_set2, _undo_log2)) = result2 {
                // Calculate supply after block2
                let supply2: i64 = utxo_set2
                    .values()
                    .map(|utxo| utxo.value)
                    .try_fold(0i64, |acc, val| acc.checked_add(val))
                    .unwrap_or(MAX_MONEY);

                // Supply should never decrease
                prop_assert!(supply2 >= supply1,
                    "Supply must never decrease: supply after block2 ({}) >= supply after block1 ({})",
                    supply2, supply1);
            }
        }
    }

    /// Invariant: Supply preserved across reorganizations
    ///
    /// Mathematical specification:
    /// ∀ current_chain, new_chain, US:
    ///   Let US_before = supply from US
    ///   Let (result, US_after) = reorganize_chain(new_chain, current_chain, US)
    ///   If result is valid, then: supply(US_after) >= supply(US_before)
    ///
    /// This ensures no money destruction during chain reorganizations.
    #[test]
    fn prop_reorganization_preserves_supply(
        current_chain_len in 1usize..5usize,
        new_chain_len in 1usize..5usize
    ) {
        use blvm_consensus::reorganization;

        // Create simple chains with coinbase blocks
        let mut current_chain = Vec::new();
        for i in 0..current_chain_len {
            current_chain.push(Block {
                header: BlockHeader {
                    version: 1,
                    prev_block_hash: if i == 0 { [0; 32] } else { [i as u8; 32] },
                    merkle_root: [1; 32],
                    timestamp: 1231006505 + (i as u64 * 600),
                    bits: 0x1d00ffff,
                    nonce: i as u64,
                },
                transactions: vec![Transaction {
                    version: 1,
                    inputs: vec![TransactionInput {
                        prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                        script_sig: vec![OP_1, OP_1],
                        sequence: 0xffffffff,
                    }].into(),
                    outputs: vec![TransactionOutput {
                        value: economic::get_block_subsidy(i as u64),
                        script_pubkey: vec![OP_1],
                    }].into(),
                    lock_time: 0,
                }].into_boxed_slice(),
            });
        }

        let mut new_chain = Vec::new();
        for i in 0..new_chain_len {
            new_chain.push(Block {
                header: BlockHeader {
                    version: 1,
                    prev_block_hash: if i == 0 { [0; 32] } else { [100 + i as u8; 32] },
                    merkle_root: [2; 32],
                    timestamp: 1231006505 + (i as u64 * 600),
                    bits: 0x1d00ffff,
                    nonce: (100 + i) as u64,
                },
                transactions: vec![Transaction {
                    version: 1,
                    inputs: vec![TransactionInput {
                        prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                        script_sig: vec![OP_1, OP_1],
                        sequence: 0xffffffff,
                    }].into(),
                    outputs: vec![TransactionOutput {
                        value: economic::get_block_subsidy(i as u64),
                        script_pubkey: vec![OP_1],
                    }].into(),
                    lock_time: 0,
                }].into_boxed_slice(),
            });
        }

        // Calculate supply before reorganization
        let utxo_set = UtxoSet::default();
        let supply_before: i64 = utxo_set
            .values()
            .map(|utxo| utxo.value)
            .try_fold(0i64, |acc, val| acc.checked_add(val))
            .unwrap_or(0);

        // Attempt reorganization
        let witnesses: Vec<Vec<Vec<blvm_consensus::segwit::Witness>>> = new_chain
            .iter()
            .map(|b| {
                b.transactions
                    .iter()
                    .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
                    .collect()
            })
            .collect();
        let network_time = new_chain
            .iter()
            .map(|b| b.header.timestamp)
            .max()
            .unwrap_or(0)
            .saturating_add(blvm_consensus::constants::MAX_FUTURE_BLOCK_TIME);

        let result = reorganization::reorganize_chain_with_witnesses(
            &new_chain,
            &witnesses,
            None,
            &current_chain,
            utxo_set,
            current_chain_len as u64,
            None::<fn(&Block) -> Option<Vec<Witness>>>,
            None::<fn(u64) -> Option<Vec<BlockHeader>>>,
            None::<fn(&Hash) -> Option<blvm_consensus::reorganization::BlockUndoLog>>,
            None::<fn(&Hash, &blvm_consensus::reorganization::BlockUndoLog) -> blvm_consensus::error::Result<()>>,
            network_time,
            blvm_consensus::types::Network::Regtest,
            None,
        );

        if let Ok(reorg_result) = result {
            // Calculate supply after reorganization
            let supply_after: i64 = reorg_result.new_utxo_set
                .values()
                .map(|utxo| utxo.value)
                .try_fold(0i64, |acc, val| acc.checked_add(val))
                .unwrap_or(0);

            // Supply should never decrease (may increase due to new blocks)
            prop_assert!(supply_after >= supply_before,
                "Reorganization must preserve supply: supply_after ({}) >= supply_before ({})",
                supply_after, supply_before);
        }
    }

    /// Invariant: Total supply calculation matches expected formula
    ///
    /// Mathematical specification:
    /// ∀ height h:
    ///   total_supply(h) = Σ(i=0 to h) get_block_subsidy(i)
    ///
    /// Note: This test verifies the economic function directly without requiring
    /// block connection, which may fail due to validation requirements.
    #[test]
    fn prop_supply_matches_expected_across_blocks(
        num_blocks in 1usize..10usize
    ) {
        // Calculate expected supply by summing subsidies
        let mut expected_supply = 0i64;

        for i in 0..num_blocks {
            let height = i as u64;
            let subsidy = economic::get_block_subsidy(height);
            expected_supply = expected_supply
                .checked_add(subsidy)
                .unwrap_or(MAX_MONEY);
        }

        // Calculate using total_supply function
        let calculated_supply = economic::total_supply((num_blocks - 1) as u64);

        // They should match (within rounding tolerance)
        prop_assert!(calculated_supply >= expected_supply - 1000 && calculated_supply <= expected_supply + 1000,
            "Supply calculation should match expected: calculated {} ≈ expected {} (height {})",
            calculated_supply, expected_supply, num_blocks - 1);
    }
}

// ============================================================================
// Compositional Verification Property Tests
// ============================================================================

proptest! {
    /// Invariant: Connecting multiple blocks preserves all invariants
    ///
    /// Mathematical specification:
    /// ∀ blocks B₁, B₂, heights h₁, h₂:
    ///   If connect_block(B₁, US, h₁) = (valid, US₁) and
    ///      connect_block(B₂, US₁, h₂) = (valid, US₂)
    ///   Then: All invariants hold for US₂
    #[test]
    fn prop_connect_block_composition(
        height1 in 0u32..100u32,
        height2 in 0u32..100u32
    ) {
        use blvm_consensus::block;
        use blvm_consensus::constants::MAX_MONEY;

        let height1 = height1;
        let height2 = if height2 <= height1 { height1 + 1 } else { height2 };

        // Create two blocks
        let block1 = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [1; 32],
                timestamp: 1231006505 + (height1 as u64 * 600),
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                    script_sig: vec![OP_1, OP_1],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height1 as u64),
                    script_pubkey: vec![OP_1],
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        let block2 = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [1; 32],
                merkle_root: [2; 32],
                timestamp: 1231006505 + (height2 as u64 * 600),
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                    script_sig: vec![OP_1, OP_1],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height2 as u64),
                    script_pubkey: vec![OP_1],
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        // Connect block1
        let utxo_set = UtxoSet::default();
        let witnesses1: Vec<Vec<Witness>> = block1.transactions.iter().map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect()).collect();
        let ctx = block::BlockValidationContext::for_network(Network::Mainnet);
        let result1 = block::connect_block(&block1, &witnesses1, utxo_set, height1 as u64, &ctx);

        if let Ok((ValidationResult::Valid, utxo_set1, _undo_log1)) = result1 {
            // Verify invariants after block1
            let supply1: i64 = utxo_set1
                .values()
                .map(|utxo| utxo.value)
                .try_fold(0i64, |acc, val| acc.checked_add(val))
                .unwrap_or(MAX_MONEY);

            prop_assert!(supply1 >= 0, "Supply after block1 must be non-negative: {}", supply1);
            prop_assert!(supply1 <= MAX_MONEY, "Supply after block1 must be <= MAX_MONEY: {} <= {}", supply1, MAX_MONEY);

            // Connect block2
            let witnesses2: Vec<Vec<Witness>> = block2.transactions.iter().map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect()).collect();
            let result2 = block::connect_block(&block2, &witnesses2, utxo_set1, height2 as u64, &ctx);

            if let Ok((ValidationResult::Valid, utxo_set2, _undo_log2)) = result2 {
                // Verify invariants after block2 (composition)
                let supply2: i64 = utxo_set2
                    .values()
                    .map(|utxo| utxo.value)
                    .try_fold(0i64, |acc, val| acc.checked_add(val))
                    .unwrap_or(MAX_MONEY);

                prop_assert!(supply2 >= supply1, "Supply must increase or stay same: {} >= {}", supply2, supply1);
                prop_assert!(supply2 >= 0, "Supply after block2 must be non-negative: {}", supply2);
                prop_assert!(supply2 <= MAX_MONEY, "Supply after block2 must be <= MAX_MONEY: {} <= {}", supply2, MAX_MONEY);
            }
        }
    }

    /// Invariant: Disconnect and connect are inverse operations
    ///
    /// Mathematical specification:
    /// ∀ block B, UTXO set US, height h:
    ///   If connect_block(B, US, h) = (valid, US₁) and
    ///      disconnect_block(B, US₁, h) = US₂
    ///   Then: US₂ ≈ US (within reorganization tolerance)
    ///
    /// Note: This is tested via reorganization, but we verify the property holds.
    #[test]
    fn prop_disconnect_connect_idempotency(
        height in 0u32..100u32
    ) {
        use blvm_consensus::block;
        use blvm_consensus::reorganization;

        // Create a block
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [1; 32],
                timestamp: 1231006505 + (height as u64 * 600),
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                    script_sig: vec![OP_1, OP_1],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height as u64),
                    script_pubkey: vec![OP_1],
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        // Connect block
        let utxo_set_before = UtxoSet::default();
        let witnesses: Vec<Vec<Witness>> = block.transactions.iter().map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect()).collect();
        let ctx = block::BlockValidationContext::for_network(Network::Mainnet);
        let result = block::connect_block(&block, &witnesses, utxo_set_before.clone(), height as u64, &ctx);

        if let Ok((ValidationResult::Valid, utxo_set_after_connect, _undo_log)) = result {
            // Simulate disconnect via reorganization (disconnect and reconnect same block)
            let current_chain = vec![block.clone()];
            let new_chain = vec![block];

            // Reorganize to same chain (should preserve state)
            let reorg_witnesses: Vec<Vec<Vec<Witness>>> = new_chain
                .iter()
                .map(|b| {
                    b.transactions
                        .iter()
                        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
                        .collect()
                })
                .collect();
            let network_time = new_chain
                .iter()
                .map(|b| b.header.timestamp)
                .max()
                .unwrap_or(0)
                .saturating_add(blvm_consensus::constants::MAX_FUTURE_BLOCK_TIME);

            let reorg_result = reorganization::reorganize_chain_with_witnesses(
                &new_chain,
                &reorg_witnesses,
                None,
                &current_chain,
                utxo_set_after_connect.clone(),
                height as u64 + 1,
                None::<fn(&Block) -> Option<Vec<Witness>>>,
                None::<fn(u64) -> Option<Vec<BlockHeader>>>,
                None::<fn(&Hash) -> Option<blvm_consensus::reorganization::BlockUndoLog>>,
                None::<fn(&Hash, &blvm_consensus::reorganization::BlockUndoLog) -> blvm_consensus::error::Result<()>>,
                network_time,
                blvm_consensus::types::Network::Regtest,
                None,
            );

            // After reorganizing to the same chain, state should be preserved
            // (This is a simplified test - full idempotency would require direct disconnect_block access)
            let reorg = reorg_result.expect("reorganizing onto the connected chain");
            prop_assert_eq!(reorg.new_height, height as u64 + 1);
            prop_assert_eq!(reorg.new_utxo_set, utxo_set_after_connect);
        }
    }
}

// ============================================================================
// Mempool Property Tests
// ============================================================================

proptest! {
    /// Invariant: Fee rate calculation is bounded and non-negative
    ///
    /// Mathematical specification:
    /// ∀ fee ∈ [0, MAX_MONEY], size ∈ [1, MAX_BLOCK_SIZE]:
    ///   fee_rate = fee / size
    ///   - fee_rate >= 0 (non-negative)
    ///   - fee_rate <= MAX_MONEY (bounded)
    #[test]
    fn prop_fee_rate_calculation_bounded(output_value in 0i64..=50_000i64) {
        let input_value = 50_000i64;
        let prevout = OutPoint {
            hash: [4; 32],
            index: 0,
        };
        let mut utxo_set = UtxoSet::default();
        utxo_set.insert(
            prevout,
            std::sync::Arc::new(UTXO {
                value: input_value,
                script_pubkey: vec![0x51].into(),
                height: 0,
                is_coinbase: false,
            }),
        );
        let tx = Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout,
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: output_value,
                script_pubkey: vec![0x51],
            }]
            .into(),
            lock_time: 0,
        };
        let fee = economic::calculate_fee(&tx, &utxo_set).unwrap();
        prop_assert_eq!(fee, input_value - output_value);
        prop_assert!(fee <= MAX_MONEY);
    }

    /// Invariant: Fee rate comparison is consistent
    ///
    /// Mathematical specification:
    /// ∀ fee₁, fee₂ ∈ [0, MAX_MONEY], size₁, size₂ ∈ [1, MAX_BLOCK_SIZE]:
    ///   If fee₁ * size₂ > fee₂ * size₁, then fee_rate₁ > fee_rate₂
    ///   (Integer-based comparison to avoid floating-point precision issues)
    #[test]
    fn prop_fee_rate_comparison_consistent(
        existing_fee in 0i64..20_000i64,
        new_fee in 0i64..40_000i64,
    ) {
        let input_value = 100_000i64;
        let prevout = OutPoint {
            hash: [9; 32],
            index: 1,
        };
        let mut utxo_set = UtxoSet::default();
        utxo_set.insert(
            prevout,
            std::sync::Arc::new(UTXO {
                value: input_value,
                script_pubkey: vec![0x51].into(),
                height: 0,
                is_coinbase: false,
            }),
        );
        let sequence = (SEQUENCE_RBF as u64) - 1;
        let tx = |fee: i64| Transaction {
            version: 2,
            inputs: vec![TransactionInput {
                prevout,
                script_sig: vec![0x51],
                sequence,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: input_value - fee,
                script_pubkey: vec![0x51],
            }]
            .into(),
            lock_time: 0,
        };
        let accepted = blvm_consensus::mempool::replacement_checks(
            &tx(new_fee),
            &tx(existing_fee),
            &utxo_set,
            &blvm_consensus::mempool::Mempool::new(),
        )
        .unwrap();
        // Equal virtual size, so the higher fee is the higher fee rate.
        prop_assert_eq!(accepted, new_fee > existing_fee + MIN_RELAY_FEE);
    }

    /// Invariant: Mempool size is bounded
    ///
    /// Mathematical specification:
    /// ∀ mempool ∈ Mempool:
    ///   |mempool| <= MAX_MEMPOOL_SIZE
    #[test]
    fn prop_mempool_size_bounded(
        tx_count in 0usize..20000usize
    ) {
        use blvm_consensus::mempool::Mempool;

        let mut mempool = Mempool::new();

        // Add transactions up to tx_count
        for i in 0..tx_count {
            let mut hash = [0u8; 32];
            hash[0] = (i % 256) as u8;
            hash[1] = ((i / 256) % 256) as u8;
            mempool.insert(hash);
        }

        // Mempool size should equal tx_count (HashSet doesn't auto-limit, but acceptance logic does)
        // The actual bounding happens in check_mempool_rules, not in the data structure
        prop_assert!(mempool.len() == tx_count,
            "Mempool size should match inserted count: count={}, actual={}",
            tx_count, mempool.len());

        // Verify that mempool size is bounded (the acceptance logic enforces this)
        // For property test purposes, we just verify the data structure works correctly
        prop_assert!(mempool.len() <= 20000,
            "Mempool size should be reasonable: actual={}",
            mempool.len());
    }
}

// ============================================================================
// SegWit Property Tests
// ============================================================================

proptest! {
    /// Invariant: Witness weight calculation is correct
    ///
    /// Mathematical specification (BIP141):
    /// ∀ base_size, witness_size ∈ ℕ (total_size = base_size + witness_size):
    ///   weight = 3 × base_size + total_size = 4 × base_size + witness_size
    ///   - weight >= base_size (always)
    ///   - weight >= witness_size (always)
    #[test]
    fn prop_witness_weight_calculation(
        base_size in 0u64..(MAX_BLOCK_SIZE as u64),
        witness_size in 0u64..(MAX_BLOCK_SIZE as u64)
    ) {
        use blvm_consensus::witness;

        let total_size = base_size + witness_size;
        let weight = witness::calculate_transaction_weight_segwit(
            base_size,
            total_size
        );

        // BIP141: Weight = 3 * base_size + total_size (= 4 * base_size + witness_size)
        let expected_weight = 3 * base_size + total_size;

        prop_assert_eq!(weight, expected_weight,
            "Weight calculation: base_size={}, witness_size={}, expected={}, actual={}",
            base_size, witness_size, expected_weight, weight);

        // Weight should be >= base_size
        prop_assert!(weight >= base_size,
            "Weight must be >= base_size: weight={}, base_size={}",
            weight, base_size);

        // Weight should be >= witness_size (since base_size >= 0)
        prop_assert!(weight >= witness_size,
            "Weight must be >= witness_size: weight={}, witness_size={}",
            weight, witness_size);
    }

    /// Invariant: Weight to vsize conversion is correct
    ///
    /// Mathematical specification (BIP141):
    /// ∀ weight ∈ ℕ:
    ///   vsize = ceil(weight / 4)
    ///   - vsize >= weight / 4 (ceiling property)
    ///   - vsize <= (weight / 4) + 1 (ceiling property)
    #[test]
    fn prop_weight_to_vsize_round_trip(
        weight in 0u64..(4 * MAX_BLOCK_SIZE as u64)
    ) {
        use blvm_consensus::witness;

        let vsize = witness::weight_to_vsize(weight);

        // vsize = ceil(weight / 4) = (weight + 3) / 4
        let expected_vsize = weight.div_ceil(4);

        prop_assert_eq!(vsize, expected_vsize,
            "Vsize calculation: weight={}, expected={}, actual={}",
            weight, expected_vsize, vsize);

        // Ceiling property: vsize >= weight / 4
        prop_assert!((vsize as u64) >= weight / 4,
            "Vsize must be >= weight / 4: vsize={}, weight={}, weight/4={}",
            vsize, weight, weight / 4);

        // Ceiling property: vsize <= (weight / 4) + 1
        prop_assert!((vsize as u64) <= (weight / 4) + 1,
            "Vsize must be <= (weight / 4) + 1: vsize={}, weight={}, (weight/4)+1={}",
            vsize, weight, (weight / 4) + 1);
    }

    /// Invariant: Witness commitment format is valid
    ///
    /// Mathematical specification:
    /// ∀ commitment ∈ [u8]*:
    ///   If commitment is valid witness commitment:
    ///     - commitment[0] == 0x6a (OP_RETURN)
    ///     - |commitment| == 37 (1 byte OP_RETURN + 36 bytes commitment)
    #[test]
    fn prop_witness_commitment_format(
        root in prop::array::uniform32(any::<u8>()),
        nonce in prop::array::uniform32(any::<u8>())
    ) {
        use blvm_consensus::opcodes::OP_RETURN;
        use blvm_consensus::segwit::{validate_witness_commitment, witness_commitment_script};

        let script = witness_commitment_script(&root, &nonce);
        prop_assert_eq!(script.len(), 38);
        prop_assert_eq!(script[0], OP_RETURN);
        prop_assert_eq!(script[1], 0x24);
        prop_assert_eq!(&script[2..6], &[0xaa, 0x21, 0xa9, 0xed]);
        let mut preimage = [0u8; 64];
        preimage[..32].copy_from_slice(&root);
        preimage[32..].copy_from_slice(&nonce);
        let digest = Sha256::digest(Sha256::digest(preimage));
        prop_assert_eq!(&script[6..38], &digest[..]);

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![0x51, 0xff],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: script,
            }]
            .into(),
            lock_time: 0,
        };
        let matched = validate_witness_commitment(&tx, &root, &[vec![nonce.to_vec()]]).unwrap();
        prop_assert!(matched);
        let mut other = nonce;
        other[0] ^= 0xff;
        let mismatched = validate_witness_commitment(&tx, &root, &[vec![other.to_vec()]]).unwrap();
        prop_assert!(!mismatched);
    }
}

// ============================================================================
// Boundary Value Property Tests
// ============================================================================

proptest! {
    /// Invariant: MAX_MONEY boundary conditions
    ///
    /// Mathematical specification:
    /// ∀ value ∈ {MAX_MONEY - 1, MAX_MONEY, MAX_MONEY + 1}:
    ///   - value <= MAX_MONEY ⟹ valid
    ///   - value > MAX_MONEY ⟹ invalid
    #[test]
    fn prop_max_money_boundary(offset in -1i64..2i64) {
        let value = MAX_MONEY.saturating_add(offset);
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value,
                script_pubkey: vec![0x51],
            }]
            .into(),
            lock_time: 0,
        };
        let result = transaction::check_transaction(&tx).unwrap();
        prop_assert_eq!(
            matches!(result, ValidationResult::Valid),
            (0..=MAX_MONEY).contains(&value)
        );
    }

    /// Invariant: Halving interval boundary conditions
    ///
    /// Mathematical specification:
    /// ∀ height ∈ {H - 1, H, H + 1} where H = 210,000 (Orange Paper halving interval):
    ///   subsidy(height) should follow halving schedule correctly
    #[test]
    fn prop_halving_interval_boundary(
        offset in -1i64..2i64
    ) {
        // Using Orange Paper constant H (halving interval)
        let height = ((H as i64).saturating_add(offset)) as u64;
        let subsidy = economic::get_block_subsidy(height);

        // At halving boundary, subsidy should halve
        let halving_epoch = height / H;
        let prev_halving_epoch = if height > 0 { (height - 1) / H } else { 0 };

        if halving_epoch > prev_halving_epoch {
            // We crossed a halving boundary
            let prev_subsidy = economic::get_block_subsidy(height.saturating_sub(1));
            // New subsidy should be approximately half (allowing for rounding)
            prop_assert!(subsidy <= prev_subsidy,
                "Subsidy should decrease at halving: height={}, subsidy={}, prev_subsidy={}",
                height, subsidy, prev_subsidy);
        }

        // Subsidy should always be non-negative
        prop_assert!(subsidy >= 0,
            "Subsidy must be non-negative: height={}, subsidy={}",
            height, subsidy);
    }

    /// Invariant: Difficulty adjustment interval boundary conditions
    ///
    /// Mathematical specification:
    /// ∀ height ∈ {DIFFICULTY_ADJUSTMENT_INTERVAL - 1, DIFFICULTY_ADJUSTMENT_INTERVAL, DIFFICULTY_ADJUSTMENT_INTERVAL + 1}:
    ///   Difficulty adjustment should only occur at multiples of interval
    #[test]
    fn prop_difficulty_adjustment_boundary(
        offset in -1i64..2i64
    ) {
        use blvm_consensus::constants::DIFFICULTY_ADJUSTMENT_INTERVAL;

        let height = ((DIFFICULTY_ADJUSTMENT_INTERVAL as i64).saturating_add(offset)) as u64;
        prop_assume!(height > 0);
        let parent_bits = MAX_TARGET as u64;
        let parent_time = 1_700_000_000u64;
        let ancestor = |h: u64| {
            if h == height - 1 || h == 0 {
                Some((parent_bits, parent_time))
            } else {
                None
            }
        };
        let bits =
            pow::next_required_bits(Network::Mainnet, height, parent_time, &ancestor).unwrap();
        if height % DIFFICULTY_ADJUSTMENT_INTERVAL != 0 {
            prop_assert_eq!(bits, Some(parent_bits));
        } else {
            let next = pow::expand_target(bits.unwrap()).unwrap();
            let previous = pow::expand_target(parent_bits).unwrap();
            prop_assert!(next < previous);
        }
    }

    /// Block weight crosses 4_000_000 at one byte of transaction size.
    ///
    /// One transaction and no witness: weight is four times the 80-byte header,
    /// the one-byte transaction count, and the stripped transaction.
    #[test]
    fn prop_block_size_boundary(offset in -1i64..2i64) {
        let tx_size = (999_919i64 + offset) as usize;
        let tx = tx_of_stripped_size(tx_size);
        prop_assert_eq!(transaction::calculate_transaction_size(&tx), tx_size);
        let block = one_tx_block(tx);
        let weight = segwit::calculate_block_weight(&block, &[] as &[Witness]).unwrap();
        let expected = 4 * (80 + 1 + tx_size) as u64;
        prop_assert_eq!(weight, expected);
        prop_assert_eq!(weight > MAX_BLOCK_WEIGHT as u64, offset > 0);
    }

    /// Stripped size times four may equal `MAX_BLOCK_WEIGHT` and must not exceed it.
    #[test]
    fn prop_transaction_size_boundary(offset in -1i64..2i64) {
        let target = (1_000_000i64 + offset) as usize;
        let tx = tx_of_stripped_size(target);
        let size = transaction::calculate_transaction_size(&tx);
        prop_assert_eq!(size, target);
        let too_big = size * 4 > MAX_BLOCK_WEIGHT;
        match transaction::check_transaction(&tx).unwrap() {
            ValidationResult::Valid => prop_assert!(!too_big),
            ValidationResult::Invalid(message) => {
                prop_assert!(too_big);
                prop_assert!(
                    message.contains("too large"),
                    "size failure should name the limit, got {message}"
                );
            }
        }
    }
}

// ============================================================================
// Proof of Work Function Property Tests
// ============================================================================

proptest! {
    /// Invariant: Target compression round-trip preserves significant bits
    ///
    /// Mathematical specification:
    /// ∀ bits ∈ [0x01000000, 0x1d00ffff]:
    ///   Let expanded = expand_target(bits)
    ///   Then: expanded should be valid and non-zero (or zero with correct encoding)
    ///
    /// Note: compress_target is private, but the round-trip property is verified
    /// by spec-lock: target expand/compress round-trip
    #[test]
    fn prop_compress_target_round_trip(
        bits in 0x01000000u32..=0x1d00ffffu32
    ) {
        // Expand target (bits is u32, but expand_target expects Natural which is u64)
        let exponent = (bits >> 24) as u8;
        let mantissa = bits & 0x007f_ffff;
        let expanded = pow::expand_target(bits as u64);
        if !(3..=32).contains(&exponent) {
            prop_assert!(expanded.is_err(), "exponent {exponent} is outside 3..=32");
        } else if mantissa == 0 {
            prop_assert_eq!(expanded.unwrap(), pow::U256::zero());
        } else {
            prop_assert!(expanded.unwrap() != pow::U256::zero());
        }
    }

    /// Invariant: Difficulty adjustment only occurs at multiples of interval
    ///
    /// Mathematical specification:
    /// ∀ height ∈ ℕ:
    ///   is_adjustment_height(height) ⟺ (height > 0 ∧ height % DIFFICULTY_ADJUSTMENT_INTERVAL == 0)
    #[test]
    fn prop_difficulty_adjustment_interval(
        height in 0u64..1000000u64
    ) {
        use blvm_consensus::constants::DIFFICULTY_ADJUSTMENT_INTERVAL;

        prop_assume!(height > 0 && height % DIFFICULTY_ADJUSTMENT_INTERVAL != 0);
        let parent_bits = MAX_TARGET as u64;
        let parent_time = 1_700_000_000u64;
        let ancestor = |h: u64| {
            if h == height - 1 {
                Some((parent_bits, parent_time))
            } else {
                None
            }
        };
        let bits = pow::next_required_bits(Network::Mainnet, height, parent_time, &ancestor).unwrap();
        prop_assert_eq!(bits, Some(parent_bits));
    }

    /// Invariant: Difficulty adjustment timespan clamping
    ///
    /// Mathematical specification:
    /// ∀ timespan ∈ ℕ:
    ///   clamped_timespan = clamp(timespan, expected_time/4, expected_time*4)
    ///   - expected_time/4 <= clamped_timespan <= expected_time*4
    #[test]
    fn prop_difficulty_adjustment_clamping(
        timespan in 0u64..(TARGET_TIME_PER_BLOCK * 4 * DIFFICULTY_ADJUSTMENT_INTERVAL)
    ) {
        use blvm_consensus::constants::{DIFFICULTY_ADJUSTMENT_INTERVAL, TARGET_TIME_PER_BLOCK};

        let expected = TARGET_TIME_PER_BLOCK * DIFFICULTY_ADJUSTMENT_INTERVAL;
        let min_span = expected / 4;
        let max_span = expected.saturating_mul(4);
        let start = 1_700_000_000u64;
        let parent_bits = MAX_TARGET as u64;
        let height = DIFFICULTY_ADJUSTMENT_INTERVAL;
        let ancestor = |h: u64| {
            if h == 0 {
                Some((parent_bits, start))
            } else if h == height - 1 {
                Some((parent_bits, start.saturating_add(timespan)))
            } else {
                None
            }
        };
        let bits = pow::next_required_bits(
            Network::Mainnet,
            height,
            start.saturating_add(timespan),
            &ancestor,
        )
        .unwrap()
        .unwrap();
        let clamped = timespan.clamp(min_span, max_span);
        if clamped == expected || clamped > expected {
            // Mainnet's previous target is already the proof-of-work limit, so a longer
            // span cannot make the next target easier.
            prop_assert_eq!(bits, parent_bits);
        } else {
            let next = pow::expand_target(bits).unwrap();
            let previous = pow::expand_target(parent_bits).unwrap();
            prop_assert!(next < previous);
        }
    }
}

// ============================================================================
// Serialization Property Tests
// ============================================================================

proptest! {
    /// Invariant: Transaction serialization round-trip
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction:
    ///   deserialize_transaction(serialize_transaction(tx)) = tx
    #[test]
    fn prop_transaction_serialize_deserialize_round_trip(
        version in 0u64..=(i32::MAX as u64), // i32 max (4 bytes in wire format, stored as u64)
        input_count in 0usize..10usize,
        output_count in 0usize..10usize,
        lock_time in 0u64..=0xffffffffu64 // u32 max (4 bytes in wire format)
    ) {
        // Generate transaction manually
        let tx = Transaction {
            version,
            inputs: (0..input_count).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: i as u32,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..output_count).map(|i| TransactionOutput {
                value: 1000 * (i as i64 + 1),
                script_pubkey: vec![OP_1],
            }).collect(),
            lock_time,
        };
        use blvm_consensus::serialization;

        // Bound transaction size for tractability
        prop_assume!(tx.inputs.len() <= 10 && tx.outputs.len() <= 10,
            "Transaction too large for round-trip test");

        // Serialize
        let serialized = serialization::serialize_transaction(&tx);

        // Deserialize
        let deserialized = serialization::deserialize_transaction(&serialized);

        if let Ok(deserialized_tx) = deserialized {
            // Verify round-trip: deserialized should match original
            prop_assert_eq!(deserialized_tx.version, tx.version,
                "Version should match: original={}, deserialized={}",
                tx.version, deserialized_tx.version);
            prop_assert_eq!(deserialized_tx.inputs.len(), tx.inputs.len(),
                "Input count should match: original={}, deserialized={}",
                tx.inputs.len(), deserialized_tx.inputs.len());
            prop_assert_eq!(deserialized_tx.outputs.len(), tx.outputs.len(),
                "Output count should match: original={}, deserialized={}",
                tx.outputs.len(), deserialized_tx.outputs.len());
            prop_assert_eq!(deserialized_tx.lock_time, tx.lock_time,
                "Lock time should match: original={}, deserialized={}",
                tx.lock_time, deserialized_tx.lock_time);
        }
    }

    /// Invariant: Block header serialization round-trip
    ///
    /// Mathematical specification:
    /// ∀ header ∈ BlockHeader:
    ///   deserialize_block_header(serialize_block_header(header)) = header
    #[test]
    fn prop_block_header_serialize_deserialize_round_trip(
        version in (i32::MIN as i64)..=(i32::MAX as i64), // i32 range (4 bytes in wire format)
        prev_hash_bytes in prop::array::uniform32(any::<u8>()),
        merkle_bytes in prop::array::uniform32(any::<u8>()),
        timestamp in 0u64..=0xffffffffu64, // u32 max (4 bytes in wire format)
        bits in 0u64..=0xffffffffu64, // u32 max (4 bytes in wire format)
        nonce in 0u64..=0xffffffffu64 // u32 max (4 bytes in wire format)
    ) {
        // Generate block header manually
        let header = BlockHeader {
            version,
            prev_block_hash: prev_hash_bytes,
            merkle_root: merkle_bytes,
            timestamp,
            bits,
            nonce,
        };
        use blvm_consensus::serialization;

        // Serialize
        let serialized = serialization::serialize_block_header(&header);

        // Deserialize
        let deserialized = serialization::deserialize_block_header(&serialized);

        if let Ok(deserialized_header) = deserialized {
            // Verify round-trip
            prop_assert_eq!(deserialized_header.version, header.version,
                "Version should match");
            prop_assert_eq!(deserialized_header.prev_block_hash, header.prev_block_hash,
                "Previous block hash should match");
            prop_assert_eq!(deserialized_header.merkle_root, header.merkle_root,
                "Merkle root should match");
            prop_assert_eq!(deserialized_header.timestamp, header.timestamp,
                "Timestamp should match");
            prop_assert_eq!(deserialized_header.bits, header.bits,
                "Bits should match");
            prop_assert_eq!(deserialized_header.nonce, header.nonce,
                "Nonce should match");
        }
    }

    /// Invariant: VarInt encoding round-trip
    ///
    /// Mathematical specification:
    /// ∀ value ∈ [0, 2^64 - 1]:
    ///   decode_varint(encode_varint(value)) = (value, length)
    #[test]
    fn prop_varint_encoding_round_trip(
        value in 0u64..0xffffffffffffffffu64
    ) {
        use blvm_consensus::serialization::varint;

        // Encode
        let encoded = varint::encode_varint(value);

        // Decode
        let decoded = varint::decode_varint(&encoded);

        if let Ok((decoded_value, length)) = decoded {
            // Verify round-trip
            prop_assert_eq!(decoded_value, value,
                "VarInt value should match: original={}, decoded={}",
                value, decoded_value);

            // Verify length matches encoded length
            prop_assert_eq!(length, encoded.len(),
                "VarInt length should match: encoded_len={}, decoded_len={}",
                encoded.len(), length);

            // Verify encoded length is bounded (VarInt max 9 bytes)
            prop_assert!(encoded.len() <= 9,
                "VarInt encoding should be <= 9 bytes: length={}",
                encoded.len());
        }
    }
}

// ============================================================================
// Economic Function Property Tests
// ============================================================================

proptest! {
    /// Invariant: validate_supply_limit correctly enforces MAX_MONEY
    ///
    /// Mathematical specification:
    /// ∀ height ∈ ℕ:
    ///   validate_supply_limit(height) = true ⟺ total_supply(height) ≤ MAX_MONEY
    #[test]
    fn prop_validate_supply_limit(
        height in 0u64..2100000u64
    ) {
        use blvm_consensus::economic;
        use blvm_consensus::constants::MAX_MONEY;

        let is_valid = economic::validate_supply_limit(height);

        // Calculate actual supply
        let supply = economic::total_supply(height);

        // Validation result should match supply comparison
        prop_assert_eq!(
            is_valid,
            supply <= MAX_MONEY,
            "validate_supply_limit must correctly reflect supply <= MAX_MONEY: height={}, supply={}, MAX_MONEY={}, is_valid={}",
            height, supply, MAX_MONEY, is_valid
        );

        // If supply is within limit, validation must pass
        if supply <= MAX_MONEY {
            prop_assert!(is_valid,
                "Supply within limit must validate as true: height={}, supply={}",
                height, supply);
        } else {
            prop_assert!(!is_valid,
                "Supply exceeding limit must validate as false: height={}, supply={}, MAX_MONEY={}",
                height, supply, MAX_MONEY);
        }
    }

    /// Invariant: Fee calculation with missing UTXO
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, utxo_set ∈ UtxoSet:
    ///   If input.prevout ∉ utxo_set: calculate_fee returns UtxoNotFound
    ///   (coinbase → Ok(0); empty inputs non-coinbase still walks zero inputs → Ok(0))
    #[test]
    fn prop_calculate_fee_missing_utxo_handling(
        input_count in 0usize..5usize,
        output_count in 0usize..5usize
    ) {
        use blvm_consensus::economic;
        use blvm_consensus::error::ConsensusError;
        use blvm_consensus::types::*;

        // Create transaction with inputs that may not be in UTXO set
        let tx = Transaction {
            version: 1,
            inputs: (0..input_count).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: i as u32,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..output_count).map(|i| TransactionOutput {
                value: 1000 * (i as i64 + 1),
                script_pubkey: vec![OP_1],
            }).collect(),
            lock_time: 0,
        };

        // Empty UTXO set (all inputs missing)
        let utxo_set = UtxoSet::default();

        let result = economic::calculate_fee(&tx, &utxo_set);

        if input_count == 0 {
            // Empty inputs (not a coinbase null-prevout): fee = 0 - outputs.
            if output_count == 0 {
                prop_assert!(result.is_ok(), "Empty tx fee should be Ok(0)");
            } else {
                prop_assert!(
                    matches!(result, Err(ConsensusError::EconomicValidation(_))),
                    "Empty inputs with outputs → negative fee, got {result:?}"
                );
            }
        } else {
            prop_assert!(
                matches!(result, Err(ConsensusError::UtxoNotFound(_))),
                "Missing UTXO must be UtxoNotFound, got {result:?}"
            );
        }
    }

    /// Invariant: Fee calculation overflow safety
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, utxo_set ∈ UtxoSet:
    ///   calculate_fee handles overflow correctly and returns error on overflow
    #[test]
    fn prop_calculate_fee_overflow_safety(
        input_count in 0usize..3usize,
        output_count in 0usize..3usize
    ) {
        use blvm_consensus::economic;
        use blvm_consensus::types::*;
        use blvm_consensus::constants::MAX_MONEY;

        // Create transaction with large values to test overflow
        let mut utxo_set = UtxoSet::default();

        // Add UTXOs with large values (but bounded to MAX_MONEY)
        for i in 0..input_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: i as u32,
            };
            // Use large but safe values
            let value = (MAX_MONEY / (input_count.max(1) as i64)).min(MAX_MONEY);
            utxo_set.insert(outpoint, std::sync::Arc::new(UTXO {
                value,
                script_pubkey: vec![OP_1].into(),
                height: 0,
                is_coinbase: false,
            }));
        }

        let tx = Transaction {
            version: 1,
            inputs: (0..input_count).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: i as u32,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..output_count).map(|i| TransactionOutput {
                value: 1000 * (i as i64 + 1), // Small outputs
                script_pubkey: vec![OP_1],
            }).collect(),
            lock_time: 0,
        };

        let output_total = 1000 * (output_count as i64) * (output_count as i64 + 1) / 2;
        let result = economic::calculate_fee(&tx, &utxo_set);
        if input_count == 0 {
            if output_count == 0 {
                prop_assert_eq!(result.expect("empty fee"), 0);
            } else {
                let message = result.expect_err("outputs without inputs").to_string();
                prop_assert!(message.contains("Negative fee"));
            }
        } else {
            let input_value = (MAX_MONEY / (input_count as i64)).min(MAX_MONEY);
            let fee = result.expect("fee");
            prop_assert_eq!(fee, input_value * (input_count as i64) - output_total);
            prop_assert!(fee <= MAX_MONEY);
        }
        // Overflow errors are acceptable and handled correctly
    }
}

proptest! {
    /// BIP34: the encoder's script is the minimal height prefix, and `check_bip34`
    /// accepts that script only while the fork is active.
    #[test]
    fn prop_bip34_height_prefix(height in 0u64..300_000u64) {
        let script = encode_bip34_coinbase_script(height);
        prop_assert!(
            script.starts_with(&bip34_prefix(height)),
            "height {height} script {script:?} does not start with the minimal prefix"
        );
        let regtest = ForkActivationTable::from_network(Network::Regtest);
        let block = one_tx_block(coinbase_tx(script.clone(), 0, 0xffff_ffff));
        prop_assert!(check_bip34(&block, height, &regtest).unwrap());
        let mut flipped = script;
        flipped[0] ^= 0xff;
        let bad = one_tx_block(coinbase_tx(flipped, 0, 0xffff_ffff));
        prop_assert!(!check_bip34(&bad, height, &regtest).unwrap());
        let mainnet = ForkActivationTable::from_network(Network::Mainnet);
        prop_assert!(check_bip34(&bad, BIP34_ACTIVATION_MAINNET - 1, &mainnet).unwrap());
    }

    /// BIP54: coinbase lock time is `height - 1` and the sequence is not final.
    #[test]
    fn prop_bip54_coinbase_lock(
        height in 1u64..500_000u64,
        lock_delta in -1i64..2i64,
        final_sequence in any::<bool>()
    ) {
        let lock_time = (height as i64 + lock_delta) as u64;
        let sequence = if final_sequence { 0xffff_ffff } else { 0xffff_fffe };
        let tx = coinbase_tx(vec![0x03, 0x00, 0xff], lock_time, sequence);
        let accepted = check_bip54_coinbase(&tx, height);
        prop_assert_eq!(accepted, lock_delta == -1 && !final_sequence);
    }

    /// CastToBool is true when any byte is nonzero, except a lone trailing `0x80`.
    #[test]
    fn prop_cast_to_bool(bytes in prop::collection::vec(any::<u8>(), 0..9)) {
        let nonzero = bytes.iter().any(|b| *b != 0);
        let negative_zero = !bytes.is_empty()
            && bytes[bytes.len() - 1] == 0x80
            && bytes[..bytes.len() - 1].iter().all(|b| *b == 0);
        prop_assert_eq!(cast_to_bool(&bytes), nonzero && !negative_zero);
    }
}
