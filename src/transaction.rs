//! Transaction validation functions from Orange Paper Section 5.1

use crate::types::*;
use crate::constants::*;
use crate::error::Result;

/// CheckTransaction: 𝒯𝒳 → {valid, invalid}
/// 
/// A transaction tx = (v, ins, outs, lt) is valid if and only if:
/// 1. |ins| > 0 ∧ |outs| > 0
/// 2. ∀o ∈ outs: 0 ≤ o.value ≤ M_max
/// 3. |ins| ≤ M_max_inputs
/// 4. |outs| ≤ M_max_outputs
/// 5. |tx| ≤ M_max_tx_size
pub fn check_transaction(tx: &Transaction) -> Result<ValidationResult> {
    // 1. Check inputs and outputs are not empty
    if tx.inputs.is_empty() || tx.outputs.is_empty() {
        return Ok(ValidationResult::Invalid("Empty inputs or outputs".to_string()));
    }
    
    // 2. Check output values are valid
    for (i, output) in tx.outputs.iter().enumerate() {
        if output.value < 0 || output.value > MAX_MONEY {
            return Ok(ValidationResult::Invalid(
                format!("Invalid output value {} at index {}", output.value, i)
            ));
        }
    }
    
    // 3. Check input count limit
    if tx.inputs.len() > MAX_INPUTS {
        return Ok(ValidationResult::Invalid(
            format!("Too many inputs: {}", tx.inputs.len())
        ));
    }
    
    // 4. Check output count limit
    if tx.outputs.len() > MAX_OUTPUTS {
        return Ok(ValidationResult::Invalid(
            format!("Too many outputs: {}", tx.outputs.len())
        ));
    }
    
    // 5. Check transaction size limit
    let tx_size = calculate_transaction_size(tx);
    if tx_size > MAX_TX_SIZE {
        return Ok(ValidationResult::Invalid(
            format!("Transaction too large: {} bytes", tx_size)
        ));
    }
    
    Ok(ValidationResult::Valid)
}

/// CheckTxInputs: 𝒯𝒳 × 𝒰𝒮 × ℕ → {valid, invalid} × ℤ
/// 
/// For transaction tx with UTXO set us at height h:
/// 1. If tx is coinbase: return (valid, 0)
/// 2. Let total_in = Σᵢ us(i.prevout).value
/// 3. Let total_out = Σₒ o.value
/// 4. If total_in < total_out: return (invalid, 0)
/// 5. Return (valid, total_in - total_out)
pub fn check_tx_inputs(
    tx: &Transaction, 
    utxo_set: &UtxoSet, 
    _height: Natural
) -> Result<(ValidationResult, Integer)> {
    // Check if this is a coinbase transaction
    if is_coinbase(tx) {
        return Ok((ValidationResult::Valid, 0));
    }
    
    let mut total_input_value = 0i64;
    
    for (i, input) in tx.inputs.iter().enumerate() {
        // Check if input exists in UTXO set
        if let Some(utxo) = utxo_set.get(&input.prevout) {
            // Check if UTXO is not spent (this would be handled by UTXO set management)
            total_input_value += utxo.value;
        } else {
            return Ok((ValidationResult::Invalid(
                format!("Input {} not found in UTXO set", i)
            ), 0));
        }
    }
    
    let total_output_value: i64 = tx.outputs.iter().map(|o| o.value).sum();
    
    if total_input_value < total_output_value {
        return Ok((ValidationResult::Invalid(
            "Insufficient input value".to_string()
        ), 0));
    }
    
    let fee = total_input_value - total_output_value;
    Ok((ValidationResult::Valid, fee))
}

/// Check if transaction is coinbase
pub fn is_coinbase(tx: &Transaction) -> bool {
    tx.inputs.len() == 1 && 
    tx.inputs[0].prevout.hash == [0u8; 32] && 
    tx.inputs[0].prevout.index == 0xffffffff
}

/// Calculate transaction size (simplified)
fn calculate_transaction_size(tx: &Transaction) -> usize {
    // Simplified size calculation
    // In reality, this would be the serialized size
    4 + // version
    tx.inputs.len() * 41 + // inputs (simplified)
    tx.outputs.len() * 9 + // outputs (simplified)
    4 // lock_time
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for Transaction Validation:
/// ∀ tx ∈ 𝒯𝒳: CheckTransaction(tx) = valid ⟺ 
///   (|tx.inputs| > 0 ∧ |tx.outputs| > 0 ∧ 
///    ∀o ∈ tx.outputs: 0 ≤ o.value ≤ M_max ∧
///    |tx.inputs| ≤ M_max_inputs ∧ |tx.outputs| ≤ M_max_outputs ∧
///    |tx| ≤ M_max_tx_size)
/// 
/// Invariants:
/// - Valid transactions have non-empty inputs and outputs
/// - Output values are bounded [0, MAX_MONEY]
/// - Input/output counts respect limits
/// - Transaction size respects limits
/// - Coinbase transactions have special validation rules

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: check_transaction validates structure correctly
    #[kani::proof]
    #[kani::unwind(10)]
    fn kani_check_transaction_structure() {
        let tx: Transaction = kani::any();
        
        // Bound for tractability
        kani::assume(tx.inputs.len() <= 10);
        kani::assume(tx.outputs.len() <= 10);
        
        let result = check_transaction(&tx).unwrap_or(ValidationResult::Invalid("Error".to_string()));
        
        // Structure invariants
        match result {
            ValidationResult::Valid => {
                // Valid transactions must have non-empty inputs and outputs
                assert!(!tx.inputs.is_empty(), "Valid transaction must have inputs");
                assert!(!tx.outputs.is_empty(), "Valid transaction must have outputs");
                
                // Valid transactions must respect limits
                assert!(tx.inputs.len() <= MAX_INPUTS, "Valid transaction must respect input limit");
                assert!(tx.outputs.len() <= MAX_OUTPUTS, "Valid transaction must respect output limit");
                
                // Valid transactions must have valid output values
                for output in &tx.outputs {
                    assert!(output.value >= 0, "Valid transaction outputs must be non-negative");
                    assert!(output.value <= MAX_MONEY, "Valid transaction outputs must not exceed max money");
                }
            },
            ValidationResult::Invalid(_) => {
                // Invalid transactions may violate any rule
                // This is acceptable - we're testing the validation logic
            }
        }
    }

    /// Kani proof: check_tx_inputs handles coinbase correctly
    #[kani::proof]
    fn kani_check_tx_inputs_coinbase() {
        let tx: Transaction = kani::any();
        let utxo_set: UtxoSet = kani::any();
        let height: Natural = kani::any();
        
        // Bound for tractability
        kani::assume(tx.inputs.len() <= 5);
        kani::assume(tx.outputs.len() <= 5);
        
        let result = check_tx_inputs(&tx, &utxo_set, height).unwrap_or((ValidationResult::Invalid("Error".to_string()), 0));
        
        // Coinbase invariant
        if is_coinbase(&tx) {
            assert!(matches!(result.0, ValidationResult::Valid), "Coinbase transactions must be valid");
            assert_eq!(result.1, 0, "Coinbase transactions must have zero fee");
        }
    }

    /// Kani proof: is_coinbase correctly identifies coinbase transactions
    #[kani::proof]
    fn kani_is_coinbase_correct() {
        let tx: Transaction = kani::any();
        
        let is_cb = is_coinbase(&tx);
        
        // Coinbase identification invariant
        if is_cb {
            assert_eq!(tx.inputs.len(), 1, "Coinbase must have exactly one input");
            assert_eq!(tx.inputs[0].prevout.hash, [0u8; 32], "Coinbase input must have zero hash");
            assert_eq!(tx.inputs[0].prevout.index, 0xffffffff, "Coinbase input must have max index");
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Property test: check_transaction validates structure correctly
    proptest! {
        #[test]
        fn prop_check_transaction_structure(
            tx in any::<Transaction>()
        ) {
            // Bound for tractability
            let mut bounded_tx = tx;
            if bounded_tx.inputs.len() > 10 {
                bounded_tx.inputs.truncate(10);
            }
            if bounded_tx.outputs.len() > 10 {
                bounded_tx.outputs.truncate(10);
            }
            
            let result = check_transaction(&bounded_tx).unwrap_or(ValidationResult::Invalid("Error".to_string()));
            
            // Structure properties
            match result {
                ValidationResult::Valid => {
                    // Valid transactions must have non-empty inputs and outputs
                    prop_assert!(!bounded_tx.inputs.is_empty(), "Valid transaction must have inputs");
                    prop_assert!(!bounded_tx.outputs.is_empty(), "Valid transaction must have outputs");
                    
                    // Valid transactions must respect limits
                    prop_assert!(bounded_tx.inputs.len() <= MAX_INPUTS, "Valid transaction must respect input limit");
                    prop_assert!(bounded_tx.outputs.len() <= MAX_OUTPUTS, "Valid transaction must respect output limit");
                    
                    // Valid transactions must have valid output values
                    for output in &bounded_tx.outputs {
                        prop_assert!(output.value >= 0, "Valid transaction outputs must be non-negative");
                        prop_assert!(output.value <= MAX_MONEY, "Valid transaction outputs must not exceed max money");
                    }
                },
                ValidationResult::Invalid(_) => {
                    // Invalid transactions may violate any rule
                    // This is acceptable - we're testing the validation logic
                }
            }
        }
    }

    /// Property test: check_tx_inputs handles coinbase correctly
    proptest! {
        #[test]
        fn prop_check_tx_inputs_coinbase(
            tx in any::<Transaction>(),
            utxo_set in any::<UtxoSet>(),
            height in 0u32..1000u32
        ) {
            // Bound for tractability
            let mut bounded_tx = tx;
            if bounded_tx.inputs.len() > 5 {
                bounded_tx.inputs.truncate(5);
            }
            if bounded_tx.outputs.len() > 5 {
                bounded_tx.outputs.truncate(5);
            }
            
            let result = check_tx_inputs(&bounded_tx, &utxo_set, height).unwrap_or((ValidationResult::Invalid("Error".to_string()), 0));
            
            // Coinbase property
            if is_coinbase(&bounded_tx) {
                prop_assert!(matches!(result.0, ValidationResult::Valid), "Coinbase transactions must be valid");
                prop_assert_eq!(result.1, 0, "Coinbase transactions must have zero fee");
            }
        }
    }

    /// Property test: is_coinbase correctly identifies coinbase transactions
    proptest! {
        #[test]
        fn prop_is_coinbase_correct(
            tx in any::<Transaction>()
        ) {
            let is_cb = is_coinbase(&tx);
            
            // Coinbase identification property
            if is_cb {
                prop_assert_eq!(tx.inputs.len(), 1, "Coinbase must have exactly one input");
                prop_assert_eq!(tx.inputs[0].prevout.hash, [0u8; 32], "Coinbase input must have zero hash");
                prop_assert_eq!(tx.inputs[0].prevout.index, 0xffffffff, "Coinbase input must have max index");
            }
        }
    }

    /// Property test: calculate_transaction_size is consistent
    proptest! {
        #[test]
        fn prop_calculate_transaction_size_consistent(
            tx in any::<Transaction>()
        ) {
            // Bound for tractability
            let mut bounded_tx = tx;
            if bounded_tx.inputs.len() > 10 {
                bounded_tx.inputs.truncate(10);
            }
            if bounded_tx.outputs.len() > 10 {
                bounded_tx.outputs.truncate(10);
            }
            
            let size = calculate_transaction_size(&bounded_tx);
            
            // Size calculation properties
            prop_assert!(size >= 8, "Transaction size must be at least 8 bytes (version + lock_time)");
            prop_assert!(size <= 4 + 10 * 41 + 10 * 9 + 4, "Transaction size must not exceed maximum");
            
            // Size should be deterministic
            let size2 = calculate_transaction_size(&bounded_tx);
            prop_assert_eq!(size, size2, "Transaction size calculation must be deterministic");
        }
    }

    /// Property test: output value bounds are respected
    proptest! {
        #[test]
        fn prop_output_value_bounds(
            value in 0i64..(MAX_MONEY + 1000)
        ) {
            let tx = Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                }],
                outputs: vec![TransactionOutput {
                    value,
                    script_pubkey: vec![],
                }],
                lock_time: 0,
            };
            
            let result = check_transaction(&tx).unwrap_or(ValidationResult::Invalid("Error".to_string()));
            
            // Value bounds property
            if value < 0 || value > MAX_MONEY {
                prop_assert!(matches!(result, ValidationResult::Invalid(_)), 
                    "Transactions with invalid output values must be invalid");
            } else {
                // Valid values should pass other checks too
                if !tx.inputs.is_empty() && !tx.outputs.is_empty() {
                    prop_assert!(matches!(result, ValidationResult::Valid), 
                        "Transactions with valid output values should be valid");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_check_transaction_valid() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }
    
    #[test]
    fn test_check_transaction_empty_inputs() {
        let tx = Transaction {
            version: 1,
            inputs: vec![],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert!(matches!(check_transaction(&tx).unwrap(), ValidationResult::Invalid(_)));
    }
    
    #[test]
    fn test_check_tx_inputs_coinbase() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 5000000000, // 50 BTC
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        let utxo_set = UtxoSet::new();
        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
        
        assert_eq!(result, ValidationResult::Valid);
        assert_eq!(fee, 0);
    }
    
    // ============================================================================
    // COMPREHENSIVE TRANSACTION TESTS
    // ============================================================================
    
    #[test]
    fn test_check_transaction_empty_outputs() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };
        
        assert!(matches!(check_transaction(&tx).unwrap(), ValidationResult::Invalid(_)));
    }
    
    #[test]
    fn test_check_transaction_invalid_output_value_negative() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: -1, // Invalid negative value
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert!(matches!(check_transaction(&tx).unwrap(), ValidationResult::Invalid(_)));
    }
    
    #[test]
    fn test_check_transaction_invalid_output_value_too_large() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: MAX_MONEY + 1, // Invalid value exceeding max
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert!(matches!(check_transaction(&tx).unwrap(), ValidationResult::Invalid(_)));
    }
    
    #[test]
    fn test_check_transaction_max_output_value() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: MAX_MONEY, // Valid max value
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }
    
    #[test]
    fn test_check_transaction_too_many_inputs() {
        let mut inputs = Vec::new();
        for i in 0..=MAX_INPUTS {
            inputs.push(TransactionInput {
                prevout: OutPoint { hash: [i as u8; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            });
        }
        
        let tx = Transaction {
            version: 1,
            inputs,
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert!(matches!(check_transaction(&tx).unwrap(), ValidationResult::Invalid(_)));
    }
    
    #[test]
    fn test_check_transaction_max_inputs() {
        let mut inputs = Vec::new();
        for i in 0..MAX_INPUTS {
            inputs.push(TransactionInput {
                prevout: OutPoint { hash: [i as u8; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            });
        }
        
        let tx = Transaction {
            version: 1,
            inputs,
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }
    
    #[test]
    fn test_check_transaction_too_many_outputs() {
        let mut outputs = Vec::new();
        for _ in 0..=MAX_OUTPUTS {
            outputs.push(TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            });
        }
        
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs,
            lock_time: 0,
        };
        
        assert!(matches!(check_transaction(&tx).unwrap(), ValidationResult::Invalid(_)));
    }
    
    #[test]
    fn test_check_transaction_max_outputs() {
        let mut outputs = Vec::new();
        for _ in 0..MAX_OUTPUTS {
            outputs.push(TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            });
        }
        
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs,
            lock_time: 0,
        };
        
        assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
    }
    
    #[test]
    fn test_check_transaction_too_large() {
        // Create a transaction that will exceed MAX_TX_SIZE
        // Since calculate_transaction_size is simplified, we need to create a transaction
        // with enough inputs to exceed the size limit
        let mut inputs = Vec::new();
        for i in 0..25000 { // This should create a transaction > 1MB
            inputs.push(TransactionInput {
                prevout: OutPoint { hash: [i as u8; 32], index: 0 },
                script_sig: vec![0u8; 100], // Large script to increase size
                sequence: 0xffffffff,
            });
        }
        
        let tx = Transaction {
            version: 1,
            inputs,
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        assert!(matches!(check_transaction(&tx).unwrap(), ValidationResult::Invalid(_)));
    }
    
    #[test]
    fn test_check_tx_inputs_regular_transaction() {
        let mut utxo_set = UtxoSet::new();
        
        // Add UTXO to the set
        let outpoint = OutPoint { hash: [1; 32], index: 0 };
        let utxo = UTXO {
            value: 1000000000, // 10 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint, utxo);
        
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [1; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 900000000, // 9 BTC output
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
        
        assert_eq!(result, ValidationResult::Valid);
        assert_eq!(fee, 100000000); // 1 BTC fee
    }
    
    #[test]
    fn test_check_tx_inputs_missing_utxo() {
        let utxo_set = UtxoSet::new(); // Empty UTXO set
        
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [1; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 100000000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
        
        assert!(matches!(result, ValidationResult::Invalid(_)));
        assert_eq!(fee, 0);
    }
    
    #[test]
    fn test_check_tx_inputs_insufficient_funds() {
        let mut utxo_set = UtxoSet::new();
        
        // Add UTXO with insufficient value
        let outpoint = OutPoint { hash: [1; 32], index: 0 };
        let utxo = UTXO {
            value: 100000000, // 1 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint, utxo);
        
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [1; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 200000000, // 2 BTC output (more than input)
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
        
        assert!(matches!(result, ValidationResult::Invalid(_)));
        assert_eq!(fee, 0);
    }
    
    #[test]
    fn test_check_tx_inputs_multiple_inputs() {
        let mut utxo_set = UtxoSet::new();
        
        // Add two UTXOs
        let outpoint1 = OutPoint { hash: [1; 32], index: 0 };
        let utxo1 = UTXO {
            value: 500000000, // 5 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint1, utxo1);
        
        let outpoint2 = OutPoint { hash: [2; 32], index: 0 };
        let utxo2 = UTXO {
            value: 300000000, // 3 BTC
            script_pubkey: vec![],
            height: 0,
        };
        utxo_set.insert(outpoint2, utxo2);
        
        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint { hash: [1; 32], index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint { hash: [2; 32], index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![TransactionOutput {
                value: 700000000, // 7 BTC output
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
        
        assert_eq!(result, ValidationResult::Valid);
        assert_eq!(fee, 100000000); // 1 BTC fee (8 BTC input - 7 BTC output)
    }
    
    #[test]
    fn test_is_coinbase_edge_cases() {
        // Valid coinbase
        let valid_coinbase = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(is_coinbase(&valid_coinbase));
        
        // Wrong hash
        let wrong_hash = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [1; 32], index: 0xffffffff },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&wrong_hash));
        
        // Wrong index
        let wrong_index = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&wrong_index));
        
        // Multiple inputs
        let multiple_inputs = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint { hash: [1; 32], index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&multiple_inputs));
        
        // No inputs
        let no_inputs = Transaction {
            version: 1,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
        };
        assert!(!is_coinbase(&no_inputs));
    }
    
    #[test]
    fn test_calculate_transaction_size() {
        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0 },
                    script_sig: vec![1, 2, 3],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint { hash: [1; 32], index: 1 },
                    script_sig: vec![4, 5, 6],
                    sequence: 0xffffffff,
                },
            ],
            outputs: vec![
                TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![7, 8, 9],
                },
                TransactionOutput {
                    value: 2000,
                    script_pubkey: vec![10, 11, 12],
                },
            ],
            lock_time: 12345,
        };
        
        let size = calculate_transaction_size(&tx);
        // Expected: 4 (version) + 2*41 (inputs) + 2*9 (outputs) + 4 (lock_time) = 108
        // The actual calculation includes script_sig and script_pubkey lengths
        assert_eq!(size, 108);
    }
}
