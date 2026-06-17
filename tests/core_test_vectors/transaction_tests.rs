//! consensus transaction test vector integration

#[path = "tx_loader.rs"]
mod tx_loader;

pub use tx_loader::{TransactionTestVector, load_transaction_test_vectors};

use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::transaction::check_transaction;
use hex;

/// Run reference transaction test vectors
pub fn run_core_transaction_tests(
    vectors: &[TransactionTestVector],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed = 0;
    let mut failed = 0;

    for (i, vector) in vectors.iter().enumerate() {
        let result = check_transaction(&vector.transaction);

        match result {
            Ok(validation_result) => {
                let is_valid = matches!(validation_result, blvm_consensus::ValidationResult::Valid);
                if is_valid == vector.expected_result {
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!(
                        "Test {} failed: expected {}, got {}. Description: {}",
                        i,
                        if vector.expected_result {
                            "valid"
                        } else {
                            "invalid"
                        },
                        if is_valid { "valid" } else { "invalid" },
                        vector.description
                    );
                }
            }
            Err(e) => {
                if !vector.expected_result {
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!(
                        "Test {} failed with error: {}. Description: {}",
                        i, e, vector.description
                    );
                }
            }
        }
    }

    println!(
        "Reference transaction test vectors: {} passed, {} failed",
        passed, failed
    );

    if failed > 0 {
        Err(format!("{} test vectors failed", failed).into())
    } else {
        Ok(())
    }
}

/// Run vectors and return `(passed, failed)` without failing early.
pub fn score_core_transaction_tests(vectors: &[TransactionTestVector]) -> (usize, usize) {
    let mut passed = 0usize;
    let mut failed = 0usize;

    for vector in vectors {
        let result = check_transaction(&vector.transaction);
        let matches = match result {
            Ok(validation_result) => {
                let is_valid = matches!(validation_result, blvm_consensus::ValidationResult::Valid);
                is_valid == vector.expected_result
            }
            Err(_) => !vector.expected_result,
        };
        if matches {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    (passed, failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_test_vector_loading() {
        let vectors =
            load_transaction_test_vectors("tests/test_data/core_vectors/transactions").unwrap();
        assert!(
            !vectors.is_empty(),
            "expected Core tx_valid.json vectors to load"
        );
    }

    #[test]
    fn test_parse_simple_transaction_vector() {
        let tx_hex = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff08044c86041b020602ffffffff0100f2052a010000004341041b0e8c2567c12536aa13357b79a073dc4444acb83c4ec7a0e2f99dd7457516c5817242da796924ca4e99947d087fedf9ce467cb9f7c6287078f801df276fdf84ac00000000";
        let tx_bytes = hex::decode(tx_hex).unwrap();
        assert!(deserialize_transaction(&tx_bytes).is_ok());
    }
}
