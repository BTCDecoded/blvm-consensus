//! Differential Testing Integration
//!
//! Basic local validation helpers. For full consensus RPC differential testing, use `blvm-bench`.

#[path = "../test_helpers.rs"]
mod test_helpers;

use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::serialization::transaction::{deserialize_transaction, serialize_transaction};
use blvm_consensus::*;

use super::helpers;

/// Compare transaction validation results (local consensus only).
pub fn compare_transaction_validation_local(
    tx: &Transaction,
) -> Result<ValidationResult, Box<dyn std::error::Error>> {
    blvm_consensus::transaction::check_transaction(tx)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Compare block validation results (local consensus only).
pub fn compare_block_validation_local(
    block: &Block,
    utxo_set: &UtxoSet,
    height: u64,
    network: blvm_consensus::types::Network,
) -> Result<(ValidationResult, UtxoSet), Box<dyn std::error::Error>> {
    let witnesses = helpers::per_tx_witnesses(block);
    let ctx = blvm_consensus::block::BlockValidationContext::for_network(network);
    blvm_consensus::block::connect_block(block, &witnesses, utxo_set.clone(), height, &ctx)
        .map(|(result, new_utxo_set, _undo_log)| (result, new_utxo_set))
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

fn test_coinbase_block(value: i64, timestamp: u64) -> Block {
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x01, 0x00].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let merkle_root = calculate_merkle_root(std::slice::from_ref(&coinbase)).expect("merkle root");
    Block {
        header: BlockHeader {
            version: 2,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![coinbase].into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_helpers::*;

    #[test]
    fn test_transaction_validation_comparison() {
        let tx = create_test_tx(1000, None, None, None);
        let result = compare_transaction_validation_local(&tx);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), ValidationResult::Valid));

        let invalid_tx = create_invalid_transaction();
        let result = compare_transaction_validation_local(&invalid_tx);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_block_validation_comparison() {
        let block = test_coinbase_block(50_000_000_000, 1_231_006_505);
        let utxo_set = UtxoSet::default();
        let result = compare_block_validation_local(
            &block,
            &utxo_set,
            1,
            blvm_consensus::types::Network::Regtest,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_serialization_round_trip() {
        let tx = create_test_tx(1000, None, None, None);
        let original_result = compare_transaction_validation_local(&tx).unwrap();

        let serialized = serialize_transaction(&tx);
        let deserialized = deserialize_transaction(&serialized).unwrap();
        let round_trip_result = compare_transaction_validation_local(&deserialized).unwrap();

        match (original_result, round_trip_result) {
            (ValidationResult::Valid, ValidationResult::Valid) => {}
            (ValidationResult::Invalid(_), ValidationResult::Invalid(_)) => {}
            _ => panic!("Validation results should match after round-trip"),
        }
    }
}
