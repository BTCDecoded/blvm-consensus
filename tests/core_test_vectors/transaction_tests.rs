//! consensus transaction test vector integration

#[path = "tx_loader.rs"]
mod tx_loader;

pub use tx_loader::{TransactionTestVector, load_transaction_test_vectors};

use blvm_consensus::ValidationResult;
use blvm_consensus::serialization::transaction::deserialize_transaction;
use hex;
use tx_loader::structural_validation;

/// `BADTX` must be `Invalid`. Every other row must be structurally `Valid`.
/// An `Err` matches neither.
fn structure_matches(vector: &TransactionTestVector) -> bool {
    match structural_validation(&vector.transaction) {
        Ok(ValidationResult::Valid) => !vector.structural_failure,
        Ok(ValidationResult::Invalid(_)) => vector.structural_failure,
        Err(_) => false,
    }
}

/// Run reference transaction test vectors
pub fn run_core_transaction_tests(
    vectors: &[TransactionTestVector],
) -> Result<(), Box<dyn std::error::Error>> {
    let (passed, failed) = score_core_transaction_tests(vectors);
    println!("Reference transaction test vectors: {passed} passed, {failed} failed");
    if failed > 0 {
        Err(format!("{failed} transaction structure vectors failed").into())
    } else {
        Ok(())
    }
}

/// Run vectors and return `(passed, failed)` without failing early.
pub fn score_core_transaction_tests(vectors: &[TransactionTestVector]) -> (usize, usize) {
    let mut passed = 0usize;
    let mut failed = 0usize;

    for vector in vectors {
        if structure_matches(vector) {
            passed += 1;
        } else {
            failed += 1;
            if failed <= 8 {
                eprintln!(
                    "tx structure mismatch badtx={} desc={}",
                    vector.structural_failure, vector.description
                );
            }
        }
    }

    (passed, failed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_transaction_vector() {
        let tx_hex = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff08044c86041b020602ffffffff0100f2052a010000004341041b0e8c2567c12536aa13357b79a073dc4444acb83c4ec7a0e2f99dd7457516c5817242da796924ca4e99947d087fedf9ce467cb9f7c6287078f801df276fdf84ac00000000";
        let tx_bytes = hex::decode(tx_hex).unwrap();
        let tx = deserialize_transaction(&tx_bytes).expect("coinbase transaction");
        assert_eq!(tx.inputs.len(), 1);
        assert_eq!(tx.inputs[0].prevout.index, 0xffffffff);
        assert_eq!(tx.outputs.len(), 1);
        assert_eq!(tx.outputs[0].value, 5_000_000_000);
    }
}
