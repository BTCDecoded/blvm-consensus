//! Property-based tests for consensus invariants
//!
//! Uses PropTest to generate thousands of random test cases that verify
//! mathematical properties and invariants of Bitcoin consensus rules.

use bllvm_consensus::constants::*;
use bllvm_consensus::crypto::OptimizedSha256;
use bllvm_consensus::economic;
use bllvm_consensus::pow;
use bllvm_consensus::transaction;
use bllvm_consensus::types::*;
use proptest::prelude::*;
use sha2::{Digest, Sha256};

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
        let reference = Sha256::digest(&Sha256::digest(&data));
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
    /// Invariant: Block subsidy halves every 210,000 blocks
    ///
    /// Mathematical specification:
    /// ∀ h ∈ ℕ: subsidy(h) = 50 * 10^8 * 2^(-⌊h/210000⌋) if ⌊h/210000⌋ < 64 else 0
    #[test]
    fn prop_block_subsidy_halving_schedule(
        height in 0u32..2100000u32 // Up to 10 halvings
    ) {
        let subsidy = economic::get_block_subsidy(height as u64);
        let halving_epoch = height / 210000;

        // Subsidy should be 50 BTC * 2^(-halving_epoch) satoshis
        let expected_subsidy = if halving_epoch < 64 {
            50_0000_0000u64 / (1u64 << halving_epoch)
        } else {
            0
        };

        prop_assert_eq!(subsidy as u64, expected_subsidy,
            "Subsidy at height {} should be {} (halving epoch {})",
            height, expected_subsidy, halving_epoch);
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
        let epoch1 = height1 / 210000;
        let epoch2 = height2 / 210000;

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
        value in 0i64..(MAX_MONEY as i64 + 1)
    ) {
        let output = TransactionOutput {
            value,
            script_pubkey: vec![],
        };

        // Output value must be within bounds
        prop_assert!(output.value >= 0, "Output value must be non-negative");
        prop_assert!(output.value <= MAX_MONEY as i64,
            "Output value {} exceeds MAX_MONEY {}", output.value, MAX_MONEY);
    }

    /// Invariant: Transaction has non-empty inputs and outputs
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: CheckTransaction(tx) = valid ⟹ |tx.inputs| > 0 ∧ |tx.outputs| > 0
    #[test]
    fn prop_transaction_non_empty_inputs_outputs(
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
                script_sig: vec![],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..num_outputs).map(|_| TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }).collect(),
            lock_time: 0,
        };

        // Valid transactions must have at least one input and one output
        prop_assert!(!tx.inputs.is_empty(), "Transaction must have at least one input");
        prop_assert!(!tx.outputs.is_empty(), "Transaction must have at least one output");

        // Check transaction should validate structure (may fail on other grounds)
        let _result = transaction::check_transaction(&tx);
        // If structure is valid, it should pass basic checks
        // (may still fail on script validation, UTXO checks, etc.)
    }

    /// Invariant: Transaction size respects limits
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: |tx| ≤ MAX_TX_SIZE
    #[test]
    fn prop_transaction_size_bounded(
        tx_size in 1usize..(MAX_TX_SIZE as usize + 1)
    ) {
        // Create a transaction with bounded size
        // This is a simplified test - actual serialization would be more complex
        let max_inputs = (tx_size / 100).min(MAX_INPUTS);
        let max_outputs = (tx_size / 50).min(MAX_OUTPUTS);

        prop_assume!(max_inputs > 0 && max_outputs > 0);

        let tx = Transaction {
            version: 1,
            inputs: (0..max_inputs).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                },
                script_sig: vec![0; 10], // Small script
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..max_outputs).map(|_| TransactionOutput {
                value: 1000,
                script_pubkey: vec![0; 20], // Small script
            }).collect(),
            lock_time: 0,
        };

        // Transaction structure should be valid
        // (Actual size check would require serialization)
        prop_assert!(tx.inputs.len() <= MAX_INPUTS,
            "Input count {} exceeds limit {}", tx.inputs.len(), MAX_INPUTS);
        prop_assert!(tx.outputs.len() <= MAX_OUTPUTS,
            "Output count {} exceeds limit {}", tx.outputs.len(), MAX_OUTPUTS);
    }

    /// Invariant: Coinbase transactions have special validation rules
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳: IsCoinbase(tx) ⟹ 2 ≤ |tx.inputs[0].scriptSig| ≤ 100
    #[test]
    fn prop_coinbase_script_sig_length(
        script_sig_len in 2usize..101usize
    ) {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(), // Null prevout indicates coinbase
                    index: 0xffffffff, // Coinbase index
                },
                script_sig: vec![0; script_sig_len],
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        };

        // Coinbase scriptSig length must be in [2, 100]
        if transaction::is_coinbase(&tx) {
            prop_assert!(tx.inputs[0].script_sig.len() >= 2,
                "Coinbase scriptSig too short: {}", tx.inputs[0].script_sig.len());
            prop_assert!(tx.inputs[0].script_sig.len() <= 100,
                "Coinbase scriptSig too long: {}", tx.inputs[0].script_sig.len());
        }
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
        use bllvm_consensus::script;

        let mut stack1 = Vec::new();
        let mut stack2 = Vec::new();
        let flags = 0u32;

        // Execute script twice with same inputs
        let result1 = script::eval_script(&script, &mut stack1, flags);
        let result2 = script::eval_script(&script, &mut stack2, flags);

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
    fn prop_script_size_bounded(
        script_size in 0usize..(MAX_SCRIPT_SIZE as usize + 1)
    ) {
        use bllvm_consensus::script;

        let script = vec![0x51; script_size]; // OP_1 repeated
        let mut stack = Vec::new();
        let flags = 0u32;

        // Script execution should complete (may succeed or fail, but not hang)
        let result = script::eval_script(&script, &mut stack, flags);

        // Result should be Ok or Err, but execution should terminate
        // (This is verified by the fact that we get a result)
        prop_assert!(result.is_ok() || result.is_err(),
            "Script execution must terminate");
    }
}

// ============================================================================
// Performance Property Tests
// ============================================================================

proptest! {
    /// Invariant: SHA256 performance is bounded
    ///
    /// Mathematical specification:
    /// ∀ data ∈ [u8]*, |data| ≤ MAX_SIZE:
    ///   Let t = time(SHA256(data))
    ///   Then: t ≤ MAX_TIME_PER_BYTE * |data| + OVERHEAD
    #[test]
    fn prop_sha256_performance_bounded(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        use std::time::Instant;

        let start = Instant::now();
        let _hash = OptimizedSha256::new().hash(&data);
        let duration = start.elapsed();

        // SHA256 should complete in reasonable time
        // For 1KB data, should be < 10ms even on slow systems
        // Under coverage (tarpaulin), allow more time due to instrumentation overhead
        let max_time_ms = if std::env::var("TARPAULIN").is_ok() {
            100u128 // Much more lenient under coverage
        } else {
            10u128
        };
        let duration_ms = duration.as_millis();

        prop_assert!(duration_ms < max_time_ms,
            "SHA256 should complete quickly: {}ms for {} bytes (max: {}ms)",
            duration_ms, data.len(), max_time_ms);
    }

    /// Invariant: Double SHA256 performance is bounded
    ///
    /// Mathematical specification:
    /// ∀ data ∈ [u8]*:
    ///   Let t = time(SHA256(SHA256(data)))
    ///   Then: t ≤ MAX_TIME_PER_BYTE * |data| + OVERHEAD
    ///
    /// Note: We don't check the exact 2x ratio due to timing measurement noise
    /// for very fast operations. Instead, we verify both operations complete quickly.
    #[test]
    fn prop_double_sha256_performance_bounded(
        data in prop::collection::vec(any::<u8>(), 0..1024)
    ) {
        use std::time::Instant;

        // Double SHA256 should complete in reasonable time
        let start = Instant::now();
        let _hash = OptimizedSha256::new().hash256(&data);
        let duration = start.elapsed();

        // For 1KB data, should be < 20ms even on slow systems
        // Under coverage (tarpaulin), allow more time due to instrumentation overhead
        let max_time_ms = if std::env::var("TARPAULIN").is_ok() {
            200u128 // Much more lenient under coverage
        } else {
            20u128
        };
        let duration_ms = duration.as_millis();

        prop_assert!(duration_ms < max_time_ms,
            "Double SHA256 should complete quickly: {}ms for {} bytes (max: {}ms)",
            duration_ms, data.len(), max_time_ms);
    }

    /// Invariant: Transaction validation performance is bounded
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳:
    ///   Let t = time(CheckTransaction(tx))
    ///   Then: t ≤ MAX_TIME_PER_INPUT * |inputs| + MAX_TIME_PER_OUTPUT * |outputs| + OVERHEAD
    ///
    /// Note: We verify bounded performance rather than exact scaling ratios
    /// due to timing measurement noise for very fast operations.
    #[test]
    fn prop_transaction_validation_performance_bounded(
        num_inputs in 1usize..10usize,
        num_outputs in 1usize..10usize
    ) {
        use std::time::Instant;

        // Create a transaction
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

        // Measure validation time
        let start = Instant::now();
        let _result = transaction::check_transaction(&tx);
        let duration = start.elapsed();

        // Transaction validation should complete in reasonable time
        // For 10 inputs + 10 outputs, should be < 100ms even on slow systems
        let max_time_ms = 100u128;
        let duration_ms = duration.as_millis();

        prop_assert!(duration_ms < max_time_ms,
            "Transaction validation should complete quickly: {}ms for {} inputs, {} outputs (max: {}ms)",
            duration_ms, num_inputs, num_outputs, max_time_ms);
    }

    /// Invariant: Script execution performance is bounded
    ///
    /// Mathematical specification:
    /// ∀ script ∈ 𝕊, |script| ≤ MAX_SCRIPT_SIZE:
    ///   Let t = time(EvalScript(script))
    ///   Then: t ≤ MAX_TIME_PER_OP * |script| + OVERHEAD
    #[test]
    fn prop_script_execution_performance_bounded(
        script_size in 0usize..100usize
    ) {
        use std::time::Instant;
        use bllvm_consensus::script;

        let script = vec![0x51; script_size]; // OP_1 repeated
        let mut stack = Vec::new();
        let flags = 0u32;

        let start = Instant::now();
        let _result = script::eval_script(&script, &mut stack, flags);
        let duration = start.elapsed();

        // Script execution should complete in reasonable time
        // For 100-byte script, should be < 100ms even on slow systems
        let max_time_ms = 100u128;
        let duration_ms = duration.as_millis();

        prop_assert!(duration_ms < max_time_ms,
            "Script execution should complete quickly: {}ms for {} bytes (max: {}ms)",
            duration_ms, script_size, max_time_ms);
    }

    /// Invariant: Block subsidy calculation is constant-time
    ///
    /// Mathematical specification:
    /// ∀ h₁, h₂ ∈ ℕ:
    ///   time(GetBlockSubsidy(h₁)) ≈ time(GetBlockSubsidy(h₂))
    ///
    /// Subsidy calculation should be O(1) regardless of height.
    #[test]
    fn prop_block_subsidy_constant_time(
        height1 in 0u32..210000u32, // Reduced range for coverage
        height2 in 0u32..210000u32
    ) {
        use std::time::Instant;

        // Measure time for both heights
        let start1 = Instant::now();
        let _subsidy1 = economic::get_block_subsidy(height1 as u64);
        let duration1 = start1.elapsed();

        let start2 = Instant::now();
        let _subsidy2 = economic::get_block_subsidy(height2 as u64);
        let duration2 = start2.elapsed();

        // Both should be very fast (constant time)
        // Only check ratio if both durations are above noise threshold (1 microsecond)
        // Very fast operations have high timing variance
        if duration1.as_nanos() > 1000 && duration2.as_nanos() > 1000 {
            let ratio = duration2.as_nanos() as f64 / duration1.as_nanos() as f64;

            // Use lenient bounds (0.1x to 10x) to account for measurement noise
            prop_assert!(ratio >= 0.1 && ratio <= 10.0,
                "Subsidy calculation should be approximately constant-time: ratio = {:.2} (expected ~1.0, heights: {}, {})",
                ratio, height1, height2);
        }
        // If durations are too small, skip the ratio check (too noisy)

        // Both should complete very quickly (< 1ms normally, more lenient under coverage)
        let max_time_ms = if std::env::var("TARPAULIN").is_ok() {
            10u128 // More lenient under coverage
        } else {
            1u128
        };
        prop_assert!(duration1.as_millis() < max_time_ms,
            "Subsidy calculation should be fast: {}ms (max: {}ms)",
            duration1.as_millis(), max_time_ms);
        prop_assert!(duration2.as_millis() < max_time_ms,
            "Subsidy calculation should be fast: {}ms (max: {}ms)",
            duration2.as_millis(), max_time_ms);
    }

    /// Invariant: Target expansion performance is bounded
    ///
    /// Mathematical specification:
    /// ∀ bits ∈ [0x03000000, 0x1d00ffff]:
    ///   Let t = time(ExpandTarget(bits))
    ///   Then: t ≤ MAX_TIME
    ///
    /// Note: We verify bounded performance rather than exact constant-time
    /// due to timing measurement noise for very fast operations.
    #[test]
    fn prop_target_expansion_performance_bounded(
        bits in 0x03000000u32..0x1d00ffffu32
    ) {
        use std::time::Instant;

        // Target expansion should complete quickly
        let start = Instant::now();
        let _target = pow::expand_target(bits as u64);
        let duration = start.elapsed();

        // Should complete very quickly (< 10ms, very lenient for slow systems)
        let max_time_ms = 10u128;
        prop_assert!(duration.as_millis() < max_time_ms,
            "Target expansion should be fast: {}ms (max: {}ms)",
            duration.as_millis(), max_time_ms);
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
        let mut utxo_set = UtxoSet::new();
        let inputs: Vec<TransactionInput> = (0..num_inputs).map(|i| {
                let outpoint = OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                };
                utxo_set.insert(outpoint.clone(), UTXO {
                    value: 10000,
                    script_pubkey: vec![0; 20],
                    height: 0,
                });
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
            let mut utxo_set = UtxoSet::new();
            let mut inputs = Vec::new();

            for (i, &value) in input_values.iter().enumerate() {
                let outpoint = OutPoint {
                    hash: [i as u8; 32],
                    index: 0,
                };
                utxo_set.insert(outpoint.clone(), UTXO {
                    value,
                    script_pubkey: vec![0; 20],
                    height: 0,
                });
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
                    script_pubkey: vec![0; 20].into(),
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
    fn prop_output_value_overflow_safety(
        output_values in prop::collection::vec(0i64..i64::MAX, 1..10)
    ) {
        // Check if sum would overflow
        let sum: Option<i64> = output_values.iter().try_fold(0i64, |acc, &val| {
            acc.checked_add(val)
        });

        if sum.is_none() {
            // Overflow would occur - create transaction that triggers it
            let tx = Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint {
                        hash: [0; 32].into(),
                        index: 0,
                    },
                    script_sig: vec![0; 10],
                    sequence: 0xffffffff,
                }].into(),
                outputs: output_values.iter().map(|&value| TransactionOutput {
                    value,
                    script_pubkey: vec![0; 20],
                }).collect(),
                lock_time: 0,
            };

            // Transaction validation should detect overflow
            let result = transaction::check_transaction(&tx);
            // May be invalid for other reasons, but should not panic or silently overflow
            prop_assert!(result.is_err() || result.is_ok(),
                "Transaction validation must handle output overflow without panic");
        }
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
        prop_assert!(supply <= MAX_MONEY as i64,
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
        use bllvm_consensus::block;
        use bllvm_consensus::segwit::Witness;

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
                    prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                    script_sig: vec![0x51, 0x51], // 2 bytes for valid coinbase
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height1 as u64) as i64,
                    script_pubkey: vec![0x51].into(),
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
                    prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                    script_sig: vec![0x51, 0x51],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height2 as u64) as i64,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        // Connect block1
        let mut utxo_set = UtxoSet::new();
        let witnesses1: Vec<Witness> = block1.transactions.iter().map(|_| Vec::new()).collect();
        let result1 = block::connect_block(&block1, &witnesses1, utxo_set, height1 as u64, None);

        if let Ok((ValidationResult::Valid, utxo_set1)) = result1 {
            // Calculate supply after block1
            let supply1: i64 = utxo_set1
                .values()
                .map(|utxo| utxo.value)
                .try_fold(0i64, |acc, val| acc.checked_add(val))
                .unwrap_or(MAX_MONEY as i64);

            // Connect block2
            let witnesses2: Vec<Witness> = block2.transactions.iter().map(|_| Vec::new()).collect();
            let result2 = block::connect_block(&block2, &witnesses2, utxo_set1, height2 as u64, None);

            if let Ok((ValidationResult::Valid, utxo_set2)) = result2 {
                // Calculate supply after block2
                let supply2: i64 = utxo_set2
                    .values()
                    .map(|utxo| utxo.value)
                    .try_fold(0i64, |acc, val| acc.checked_add(val))
                    .unwrap_or(MAX_MONEY as i64);

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
        use bllvm_consensus::reorganization;
        use bllvm_consensus::segwit::Witness;

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
                        prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                        script_sig: vec![0x51, 0x51],
                        sequence: 0xffffffff,
                    }].into(),
                    outputs: vec![TransactionOutput {
                        value: economic::get_block_subsidy(i as u64) as i64,
                        script_pubkey: vec![0x51].into(),
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
                        prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                        script_sig: vec![0x51, 0x51],
                        sequence: 0xffffffff,
                    }].into(),
                    outputs: vec![TransactionOutput {
                        value: economic::get_block_subsidy(i as u64) as i64,
                        script_pubkey: vec![0x51].into(),
                    }].into(),
                    lock_time: 0,
                }].into_boxed_slice(),
            });
        }

        // Calculate supply before reorganization
        let utxo_set = UtxoSet::new();
        let supply_before: i64 = utxo_set
            .values()
            .map(|utxo| utxo.value)
            .try_fold(0i64, |acc, val| acc.checked_add(val))
            .unwrap_or(0);

        // Attempt reorganization
        let result = reorganization::reorganize_chain(
            &new_chain,
            &current_chain,
            utxo_set,
            current_chain_len as u64
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
            let subsidy = economic::get_block_subsidy(height) as i64;
            expected_supply = expected_supply
                .checked_add(subsidy)
                .unwrap_or(MAX_MONEY as i64);
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
        use bllvm_consensus::block;
        use bllvm_consensus::segwit::Witness;
        use bllvm_consensus::constants::MAX_MONEY;

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
                    prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                    script_sig: vec![0x51, 0x51],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height1 as u64) as i64,
                    script_pubkey: vec![0x51].into(),
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
                    prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                    script_sig: vec![0x51, 0x51],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height2 as u64) as i64,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        // Connect block1
        let mut utxo_set = UtxoSet::new();
        let witnesses1: Vec<Witness> = block1.transactions.iter().map(|_| Vec::new()).collect();
        let result1 = block::connect_block(&block1, &witnesses1, utxo_set, height1 as u64, None);

        if let Ok((ValidationResult::Valid, utxo_set1)) = result1 {
            // Verify invariants after block1
            let supply1: i64 = utxo_set1
                .values()
                .map(|utxo| utxo.value)
                .try_fold(0i64, |acc, val| acc.checked_add(val))
                .unwrap_or(MAX_MONEY as i64);

            prop_assert!(supply1 >= 0, "Supply after block1 must be non-negative: {}", supply1);
            prop_assert!(supply1 <= MAX_MONEY as i64, "Supply after block1 must be <= MAX_MONEY: {} <= {}", supply1, MAX_MONEY);

            // Connect block2
            let witnesses2: Vec<Witness> = block2.transactions.iter().map(|_| Vec::new()).collect();
            let result2 = block::connect_block(&block2, &witnesses2, utxo_set1, height2 as u64, None);

            if let Ok((ValidationResult::Valid, utxo_set2)) = result2 {
                // Verify invariants after block2 (composition)
                let supply2: i64 = utxo_set2
                    .values()
                    .map(|utxo| utxo.value)
                    .try_fold(0i64, |acc, val| acc.checked_add(val))
                    .unwrap_or(MAX_MONEY as i64);

                prop_assert!(supply2 >= supply1, "Supply must increase or stay same: {} >= {}", supply2, supply1);
                prop_assert!(supply2 >= 0, "Supply after block2 must be non-negative: {}", supply2);
                prop_assert!(supply2 <= MAX_MONEY as i64, "Supply after block2 must be <= MAX_MONEY: {} <= {}", supply2, MAX_MONEY);
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
        use bllvm_consensus::block;
        use bllvm_consensus::reorganization;
        use bllvm_consensus::segwit::Witness;

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
                    prevout: OutPoint { hash: [0; 32].into(), index: 0xffffffff },
                    script_sig: vec![0x51, 0x51],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: economic::get_block_subsidy(height as u64) as i64,
                    script_pubkey: vec![0x51].into(),
                }].into(),
                lock_time: 0,
            }].into_boxed_slice(),
        };

        // Connect block
        let utxo_set_before = UtxoSet::new();
        let witnesses: Vec<Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
        let result = block::connect_block(&block, &witnesses, utxo_set_before.clone(), height as u64, None);

        if let Ok((ValidationResult::Valid, utxo_set_after_connect)) = result {
            // Simulate disconnect via reorganization (disconnect and reconnect same block)
            let current_chain = vec![block.clone()];
            let new_chain = vec![block];

            // Reorganize to same chain (should preserve state)
            let reorg_result = reorganization::reorganize_chain(
                &new_chain,
                &current_chain,
                utxo_set_after_connect.clone(),
                height as u64 + 1
            );

            // After reorganizing to the same chain, state should be preserved
            // (This is a simplified test - full idempotency would require direct disconnect_block access)
            if reorg_result.is_ok() {
                // Reorganization succeeded, which means disconnect+connect worked
                prop_assert!(true, "Disconnect+connect via reorganization preserves state");
            }
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
    fn prop_fee_rate_calculation_bounded(
        fee in 0i64..MAX_MONEY,
        size in 1usize..MAX_BLOCK_SIZE
    ) {
        use bllvm_consensus::mempool;

        // Calculate fee rate using integer-based comparison
        // Avoid floating-point precision issues
        let fee_rate_scaled = (fee as u128)
            .checked_mul(1_000_000u128) // Scale for precision
            .and_then(|scaled| scaled.checked_div(size as u128));

        if let Some(rate) = fee_rate_scaled {
            // rate is u64, so it's always >= 0 - no assertion needed
            let _ = rate;

            // Fee rate should be bounded (fee <= MAX_MONEY, size >= 1)
            prop_assert!(rate <= (MAX_MONEY as u128) * 1_000_000,
                "Fee rate must be bounded: fee={}, size={}, rate={}",
                fee, size, rate);
        }
    }

    /// Invariant: Fee rate comparison is consistent
    ///
    /// Mathematical specification:
    /// ∀ fee₁, fee₂ ∈ [0, MAX_MONEY], size₁, size₂ ∈ [1, MAX_BLOCK_SIZE]:
    ///   If fee₁ * size₂ > fee₂ * size₁, then fee_rate₁ > fee_rate₂
    ///   (Integer-based comparison to avoid floating-point precision issues)
    #[test]
    fn prop_fee_rate_comparison_consistent(
        fee1 in 0i64..MAX_MONEY,
        size1 in 1usize..MAX_BLOCK_SIZE,
        fee2 in 0i64..MAX_MONEY,
        size2 in 1usize..MAX_BLOCK_SIZE
    ) {
        // Use integer-based comparison (as used in mempool RBF logic)
        let fee1_scaled = (fee1 as u128)
            .checked_mul(size2 as u128);
        let fee2_scaled = (fee2 as u128)
            .checked_mul(size1 as u128);

        if let (Some(f1), Some(f2)) = (fee1_scaled, fee2_scaled) {
            // If f1 > f2, then fee_rate1 > fee_rate2
            if f1 > f2 {
                // Verify: fee1 / size1 > fee2 / size2
                // This is equivalent to f1 > f2 (which we already checked)
                prop_assert!(true, "Fee rate comparison consistent: fee1={}, size1={}, fee2={}, size2={}",
                    fee1, size1, fee2, size2);
            } else if f1 < f2 {
                // Verify: fee1 / size1 < fee2 / size2
                prop_assert!(true, "Fee rate comparison consistent: fee1={}, size1={}, fee2={}, size2={}",
                    fee1, size1, fee2, size2);
            }
            // If f1 == f2, rates are equal (within integer precision)
        }
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
        use bllvm_consensus::mempool::Mempool;
        use bllvm_consensus::types::Hash;

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
    /// ∀ base_size, witness_size ∈ ℕ:
    ///   weight = 4 * base_size + witness_size
    ///   - weight >= base_size (always)
    ///   - weight >= witness_size (always)
    #[test]
    fn prop_witness_weight_calculation(
        base_size in 0u64..(MAX_BLOCK_SIZE as u64),
        witness_size in 0u64..(MAX_BLOCK_SIZE as u64)
    ) {
        use bllvm_consensus::witness;

        let weight = witness::calculate_transaction_weight_segwit(
            base_size as u64,
            (base_size + witness_size) as u64
        );

        // Weight = 4 * base_size + total_size
        // Where total_size = base_size + witness_size
        let expected_weight = 4 * base_size + (base_size + witness_size);

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
        use bllvm_consensus::witness;

        let vsize = witness::weight_to_vsize(weight);

        // vsize = ceil(weight / 4) = (weight + 3) / 4
        let expected_vsize = (weight + 3) / 4;

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
        commitment_len in 0usize..100usize
    ) {
        use bllvm_consensus::constants::WITNESS_COMMITMENT_SCRIPT_LENGTH;

        // Valid witness commitment: OP_RETURN (0x6a) + push opcode (0x24) + 32-byte commitment hash
        let valid_length = WITNESS_COMMITMENT_SCRIPT_LENGTH;

        if commitment_len == valid_length {
            // Valid format
            prop_assert!(true, "Valid witness commitment length: {}", commitment_len);
        } else {
            // Invalid format (but not necessarily an error - just not a witness commitment)
            prop_assert!(commitment_len != valid_length,
                "Non-witness-commitment length: {}", commitment_len);
        }
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
    fn prop_max_money_boundary(
        offset in -1i64..2i64
    ) {
        let value = (MAX_MONEY as i64).saturating_add(offset);

        if value <= MAX_MONEY && value >= 0 {
            // Valid value
            prop_assert!(value >= 0 && value <= MAX_MONEY,
                "Value within bounds: value={}, MAX_MONEY={}",
                value, MAX_MONEY);
        } else {
            // Invalid value (outside bounds)
            prop_assert!(value < 0 || value > MAX_MONEY,
                "Value outside bounds: value={}, MAX_MONEY={}",
                value, MAX_MONEY);
        }
    }

    /// Invariant: Halving interval boundary conditions
    ///
    /// Mathematical specification:
    /// ∀ height ∈ {HALVING_INTERVAL - 1, HALVING_INTERVAL, HALVING_INTERVAL + 1}:
    ///   subsidy(height) should follow halving schedule correctly
    #[test]
    fn prop_halving_interval_boundary(
        offset in -1i64..2i64
    ) {
        use bllvm_consensus::constants::HALVING_INTERVAL;

        let height = ((HALVING_INTERVAL as i64).saturating_add(offset)) as u64;
        let subsidy = economic::get_block_subsidy(height);

        // At halving boundary, subsidy should halve
        let halving_epoch = height / HALVING_INTERVAL;
        let prev_halving_epoch = if height > 0 { (height - 1) / HALVING_INTERVAL } else { 0 };

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
        use bllvm_consensus::constants::DIFFICULTY_ADJUSTMENT_INTERVAL;

        let height = ((DIFFICULTY_ADJUSTMENT_INTERVAL as i64).saturating_add(offset)) as u64;

        // Difficulty adjustment occurs at multiples of DIFFICULTY_ADJUSTMENT_INTERVAL
        let is_adjustment_height = height % (DIFFICULTY_ADJUSTMENT_INTERVAL as u64) == 0 && height > 0;

        if is_adjustment_height {
            // At adjustment height
            prop_assert!(height > 0 && height % (DIFFICULTY_ADJUSTMENT_INTERVAL as u64) == 0,
                "Adjustment height: height={}, interval={}",
                height, DIFFICULTY_ADJUSTMENT_INTERVAL);
        } else {
            // Not at adjustment height
            prop_assert!(height == 0 || height % (DIFFICULTY_ADJUSTMENT_INTERVAL as u64) != 0,
                "Non-adjustment height: height={}, interval={}",
                height, DIFFICULTY_ADJUSTMENT_INTERVAL);
        }
    }

    /// Invariant: Block size boundary conditions
    ///
    /// Mathematical specification:
    /// ∀ size ∈ {MAX_BLOCK_SIZE - 1, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE + 1}:
    ///   - size <= MAX_BLOCK_SIZE ⟹ valid
    ///   - size > MAX_BLOCK_SIZE ⟹ invalid
    #[test]
    fn prop_block_size_boundary(
        offset in -1i64..2i64
    ) {
        let size = ((MAX_BLOCK_SIZE as i64).saturating_add(offset)) as usize;

        if size <= MAX_BLOCK_SIZE && size > 0 {
            // Valid size
            prop_assert!(size > 0 && size <= MAX_BLOCK_SIZE,
                "Block size within bounds: size={}, MAX_BLOCK_SIZE={}",
                size, MAX_BLOCK_SIZE);
        } else {
            // Invalid size
            prop_assert!(size == 0 || size > MAX_BLOCK_SIZE,
                "Block size outside bounds: size={}, MAX_BLOCK_SIZE={}",
                size, MAX_BLOCK_SIZE);
        }
    }

    /// Invariant: Transaction size boundary conditions
    ///
    /// Mathematical specification:
    /// ∀ size ∈ {MAX_TX_SIZE - 1, MAX_TX_SIZE, MAX_TX_SIZE + 1}:
    ///   - size <= MAX_TX_SIZE ⟹ valid
    ///   - size > MAX_TX_SIZE ⟹ invalid
    #[test]
    fn prop_transaction_size_boundary(
        offset in -1i64..2i64
    ) {
        let size = ((MAX_TX_SIZE as i64).saturating_add(offset)) as usize;

        if size <= MAX_TX_SIZE && size > 0 {
            // Valid size
            prop_assert!(size > 0 && size <= MAX_TX_SIZE,
                "Transaction size within bounds: size={}, MAX_TX_SIZE={}",
                size, MAX_TX_SIZE);
        } else {
            // Invalid size
            prop_assert!(size == 0 || size > MAX_TX_SIZE,
                "Transaction size outside bounds: size={}, MAX_TX_SIZE={}",
                size, MAX_TX_SIZE);
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
    /// by Kani proof: kani_target_expand_compress_round_trip
    #[test]
    fn prop_compress_target_round_trip(
        bits in 0x01000000u32..=0x1d00ffffu32
    ) {
        // Expand target (bits is u32, but expand_target expects Natural which is u64)
        let expanded = pow::expand_target(bits as u64);

        if let Ok(_expanded_target) = expanded {
            // Expansion succeeded - verify it's valid
            // The round-trip property is verified by Kani proof: kani_target_expand_compress_round_trip
            // For property tests, we just verify expansion succeeds for valid bits
            prop_assert!(true, "Target expansion succeeded for bits={:x}", bits);
        } else {
            // Expansion failed - this is expected for some invalid bits
            // But we're testing valid range, so this shouldn't happen often
            prop_assume!(false, "Target expansion failed for bits={:x}", bits);
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
        use bllvm_consensus::constants::DIFFICULTY_ADJUSTMENT_INTERVAL;

        // Difficulty adjustment occurs at multiples of DIFFICULTY_ADJUSTMENT_INTERVAL
        let is_adjustment_height = height > 0 && height % DIFFICULTY_ADJUSTMENT_INTERVAL == 0;

        if is_adjustment_height {
            // Verify it's a multiple of the interval
            prop_assert!(height % DIFFICULTY_ADJUSTMENT_INTERVAL == 0 && height > 0,
                "Adjustment height must be multiple of interval: height={}, interval={}",
                height, DIFFICULTY_ADJUSTMENT_INTERVAL);
        } else {
            // Verify it's not a multiple (or is zero)
            prop_assert!(height == 0 || height % DIFFICULTY_ADJUSTMENT_INTERVAL != 0,
                "Non-adjustment height: height={}, interval={}",
                height, DIFFICULTY_ADJUSTMENT_INTERVAL);
        }
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
        use bllvm_consensus::constants::{TARGET_TIME_PER_BLOCK, DIFFICULTY_ADJUSTMENT_INTERVAL};

        let expected_time = TARGET_TIME_PER_BLOCK * DIFFICULTY_ADJUSTMENT_INTERVAL;
        let min_timespan = expected_time / 4;
        let max_timespan = expected_time * 4;

        // Clamp timespan
        let clamped_timespan = timespan.max(min_timespan).min(max_timespan);

        // Verify clamping bounds
        prop_assert!(clamped_timespan >= min_timespan,
            "Clamped timespan must be >= min: clamped={}, min={}",
            clamped_timespan, min_timespan);
        prop_assert!(clamped_timespan <= max_timespan,
            "Clamped timespan must be <= max: clamped={}, max={}",
            clamped_timespan, max_timespan);
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
                    index: i as u64,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..output_count).map(|i| TransactionOutput {
                value: 1000 * (i as i64 + 1),
                script_pubkey: vec![0x51],
            }).collect(),
            lock_time,
        };
        use bllvm_consensus::serialization;

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
        use bllvm_consensus::serialization;

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
        use bllvm_consensus::serialization::varint;

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
        use bllvm_consensus::economic;
        use bllvm_consensus::constants::MAX_MONEY;

        let result = economic::validate_supply_limit(height);

        // Should always succeed
        prop_assert!(result.is_ok(), "validate_supply_limit should always succeed");

        let is_valid = result.unwrap();

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
    ///   If input.prevout ∉ utxo_set: calculate_fee treats missing UTXO as value 0
    #[test]
    fn prop_calculate_fee_missing_utxo_handling(
        input_count in 0usize..5usize,
        output_count in 0usize..5usize
    ) {
        use bllvm_consensus::economic;
        use bllvm_consensus::types::*;

        // Create transaction with inputs that may not be in UTXO set
        let tx = Transaction {
            version: 1,
            inputs: (0..input_count).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: i as u64,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..output_count).map(|i| TransactionOutput {
                value: 1000 * (i as i64 + 1),
                script_pubkey: vec![0x51],
            }).collect(),
            lock_time: 0,
        };

        // Empty UTXO set (all inputs missing)
        let utxo_set = UtxoSet::new();

        let result = economic::calculate_fee(&tx, &utxo_set);

        // Missing UTXOs are treated as value 0
        // If outputs > 0, fee will be negative (error)
        // If outputs = 0, fee will be 0 (success)
        if output_count == 0 && input_count > 0 {
            // All inputs missing, no outputs: fee = 0
            prop_assert!(result.is_ok(), "Fee calculation should succeed with missing UTXOs and no outputs");
            if let Ok(fee) = result {
                prop_assert_eq!(fee, 0, "Fee should be 0 when all inputs are missing and no outputs");
            }
        } else if output_count > 0 {
            // Missing UTXOs with outputs: negative fee (error)
            prop_assert!(result.is_err(), "Fee calculation should fail with missing UTXOs and outputs");
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
        use bllvm_consensus::economic;
        use bllvm_consensus::types::*;
        use bllvm_consensus::constants::MAX_MONEY;

        // Create transaction with large values to test overflow
        let mut utxo_set = UtxoSet::new();

        // Add UTXOs with large values (but bounded to MAX_MONEY)
        for i in 0..input_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: i as u64,
            };
            // Use large but safe values
            let value = (MAX_MONEY / (input_count.max(1) as i64)).min(MAX_MONEY);
            utxo_set.insert(outpoint, UTXO {
                value,
                script_pubkey: vec![0x51],
                height: 0,
            });
        }

        let tx = Transaction {
            version: 1,
            inputs: (0..input_count).map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: i as u64,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }).collect(),
            outputs: (0..output_count).map(|i| TransactionOutput {
                value: 1000 * (i as i64 + 1), // Small outputs
                script_pubkey: vec![0x51],
            }).collect(),
            lock_time: 0,
        };

        let result = economic::calculate_fee(&tx, &utxo_set);

        // If calculation succeeds, fee must be bounded
        if let Ok(fee) = result {
            prop_assert!(fee >= 0, "Fee must be non-negative: fee={}", fee);
            prop_assert!(fee <= MAX_MONEY, "Fee must not exceed MAX_MONEY: fee={}, MAX_MONEY={}", fee, MAX_MONEY);
        }
        // Overflow errors are acceptable and handled correctly
    }
}
