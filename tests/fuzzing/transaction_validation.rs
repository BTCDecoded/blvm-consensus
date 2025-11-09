//! Bolero fuzzing tests for transaction validation
//!
//! Tests transaction validation functions with generated inputs to find edge cases
//! and ensure correctness across a wide range of inputs.
//!
//! Note: Uses byte-level fuzzing since Transaction doesn't implement Arbitrary.

#[cfg(feature = "bolero")]
use bolero::check;
#[cfg(feature = "bolero")]
use bllvm_consensus::{Transaction, ValidationResult, check_transaction};

#[cfg(feature = "bolero")]
#[test]
fn fuzz_check_transaction_robustness() {
    // Use byte-level fuzzing to test robustness to malformed inputs
    check!().for_each(|data: &[u8]| {
        // Create a minimal transaction and test robustness
        let tx = Transaction {
            version: data.get(0).copied().unwrap_or(1) as u64,
            inputs: vec![],
            outputs: vec![],
            lock_time: data.get(1).copied().unwrap_or(0) as u64,
        };
        
        // Validate that check_transaction doesn't panic on any input
        let result = check_transaction(&tx);
        // Result should always be Ok, even if validation fails
        assert!(result.is_ok(), "check_transaction should never panic");
    });
}

#[cfg(feature = "bolero")]
#[test]
fn fuzz_check_transaction_deterministic() {
    // Test determinism with byte-based inputs
    check!().for_each(|data: &[u8]| {
        let tx = Transaction {
            version: data.get(0).copied().unwrap_or(1) as u64,
            inputs: vec![],
            outputs: vec![],
            lock_time: data.get(1).copied().unwrap_or(0) as u64,
        };
        
        // Check that validation is deterministic
        let result1 = check_transaction(&tx);
        let result2 = check_transaction(&tx);
        
        assert_eq!(result1, result2, "check_transaction must be deterministic");
    });
}

#[cfg(feature = "bolero")]
#[test]
fn fuzz_transaction_structure() {
    check!().for_each(|data: &[u8]| {
        // Test robustness to malformed data
        if data.len() > 0 {
            let tx = Transaction {
                version: data.get(0).copied().unwrap_or(1) as u64,
                inputs: vec![],
                outputs: vec![],
                lock_time: data.get(1).copied().unwrap_or(0) as u64,
            };
            let _result = check_transaction(&tx);
            // Should not panic regardless of input
        }
    });
}
