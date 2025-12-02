//! Differential Testing Integration
//!
//! This module provides integration points for differential testing with Bitcoin Core.
//! The full implementation is in `bllvm-bench`, but this provides basic functionality
//! for testing within the consensus crate.
//!
//! For comprehensive differential testing, use `bllvm-bench` which includes:
//! - Bitcoin Core binary detection and management
//! - Regtest node management
//! - Full RPC client wrapper
//! - BIP-specific differential tests

use bllvm_consensus::*;
use bllvm_consensus::serialization::transaction::serialize_transaction;
use bllvm_consensus::serialization::block::serialize_block;

/// Compare transaction validation results
///
/// This is a basic comparison function. For full differential testing with
/// Bitcoin Core RPC, use `bllvm-bench`.
pub fn compare_transaction_validation_local(
    tx: &Transaction,
) -> Result<ValidationResult, Box<dyn std::error::Error>> {
    check_transaction(tx)
}

/// Compare block validation results
///
/// This is a basic comparison function. For full differential testing with
/// Bitcoin Core RPC, use `bllvm-bench`.
pub fn compare_block_validation_local(
    block: &Block,
    utxo_set: &UtxoSet,
    height: u64,
    network: crate::types::Network,
) -> Result<(ValidationResult, UtxoSet), Box<dyn std::error::Error>> {
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    connect_block(block, &witnesses, utxo_set.clone(), height, None, network)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[path = "../test_helpers.rs"]
    mod test_helpers;
    use test_helpers::*;

    #[test]
    fn test_transaction_validation_comparison() {
        // Test valid transaction
        let tx = create_test_tx(1000, None, None, None);
        let result = compare_transaction_validation_local(&tx);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), ValidationResult::Valid));

        // Test invalid transaction (empty inputs)
        let invalid_tx = create_invalid_transaction();
        let result = compare_transaction_validation_local(&invalid_tx);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_block_validation_comparison() {
        // Create a simple block
        let coinbase = create_coinbase_tx(50_000_000_000);
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1234567890,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![coinbase].into(),
        };

        let utxo_set = UtxoSet::new();
        let result = compare_block_validation_local(&block, &utxo_set, 0, crate::types::Network::Mainnet);
        
        // Block should validate (basic structure is valid)
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_round_trip() {
        // Test that serialization preserves validation results
        let tx = create_test_tx(1000, None, None, None);
        
        // Validate original
        let original_result = compare_transaction_validation_local(&tx).unwrap();
        
        // Serialize and deserialize
        let serialized = serialize_transaction(&tx);
        let deserialized = bincode::deserialize::<Transaction>(&serialized).unwrap();
        
        // Validate after round-trip
        let round_trip_result = compare_transaction_validation_local(&deserialized).unwrap();
        
        // Results should match
        match (original_result, round_trip_result) {
            (ValidationResult::Valid, ValidationResult::Valid) => {
                // Both valid - good
            }
            (ValidationResult::Invalid(_), ValidationResult::Invalid(_)) => {
                // Both invalid - acceptable (errors may differ)
            }
            _ => {
                panic!("Validation results should match after round-trip");
            }
        }
    }
}

