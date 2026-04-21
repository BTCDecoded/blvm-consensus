//! Property test strategies and shared test fixtures.
//!
//! Enables integration tests to use strategies that satisfy Orange Paper constraints
//! (e.g. |w| = |tx.inputs| for SegWit round-trip) and shared UTXO-set helpers.

#[cfg(any(test, feature = "property-tests"))]
use proptest::prelude::*;

#[cfg(any(test, feature = "property-tests"))]
use crate::segwit::Witness;
use crate::types::{
    BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet, UTXO,
};

#[cfg(any(test, feature = "property-tests"))]
/// Strategy yielding (Transaction, Vec<Witness>) with |w| = |tx.inputs|.
/// Use for SegWit round-trip property tests per Orange Paper 8.2.2.
pub fn transaction_with_witness_strategy() -> impl Strategy<Value = (Transaction, Vec<Witness>)> {
    (1..10usize).prop_flat_map(|input_count| {
        let tx_strategy = transaction_with_input_count_strategy(input_count);
        let witness_strategy = prop::collection::vec(
            prop::collection::vec(prop::collection::vec(any::<u8>(), 0..64), 0..5),
            input_count,
        );
        (tx_strategy, witness_strategy).prop_map(|(tx, witnesses)| (tx, witnesses))
    })
}

#[cfg(any(test, feature = "property-tests"))]
fn transaction_with_input_count_strategy(input_count: usize) -> impl Strategy<Value = Transaction> {
    prop::collection::vec(any::<u8>(), 0..10).prop_map(move |output_data| {
        let inputs: Vec<TransactionInput> = (0..input_count)
            .map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: i as u32,
                },
                script_sig: vec![0x51], // OP_1
                sequence: 0xffffffff,
            })
            .collect();
        let outputs: Vec<TransactionOutput> = output_data
            .iter()
            .map(|_| TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            })
            .collect();
        Transaction {
            version: 1,
            inputs: inputs.into(),
            outputs: outputs.into(),
            lock_time: 0,
        }
    })
}

#[cfg(any(test, feature = "property-tests"))]
/// Strategy yielding Transaction for legacy (non-SegWit) round-trip tests.
pub fn transaction_strategy() -> impl Strategy<Value = Transaction> {
    // Wire round-trip matches Bitcoin serialization: at least one vin (vout may be empty).
    (1..10usize, 0..10usize).prop_map(|(input_count, output_count)| {
        let inputs: Vec<TransactionInput> = (0..input_count)
            .map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: i as u32,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            })
            .collect();
        let outputs: Vec<TransactionOutput> = (0..output_count)
            .map(|_| TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            })
            .collect();
        Transaction {
            version: 1,
            inputs: inputs.into(),
            outputs: outputs.into(),
            lock_time: 0,
        }
    })
}

/// Create a block header with the given timestamp and previous block hash.
/// Shared by integration tests so fixture behavior stays consistent.
pub fn create_test_header(timestamp: u64, prev_hash: [u8; 32]) -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: prev_hash,
        merkle_root: [0; 32],
        timestamp,
        bits: 0x1d00ffff,
        nonce: 0,
    }
}

/// Create a valid coinbase transaction (scriptSig 2–100 bytes per consensus).
/// Shared by integration tests so fixture behavior stays consistent.
pub fn create_coinbase_tx(value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x03, 0x01, 0x00, 0x00], // 4 bytes (valid: 2–100)
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    }
}

/// UTXO set with two outputs for fee-calculation style tests (1 BTC and 0.5 BTC).
/// Shared by integration tests so fixture behavior stays consistent.
pub fn create_test_utxo_set_two_outputs() -> UtxoSet {
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 100_000_000, // 1 BTC
            script_pubkey: vec![0x51].into(),
            height: 100,
            is_coinbase: false,
        }),
    );
    utxo_set.insert(
        OutPoint {
            hash: [2; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 50_000_000, // 0.5 BTC
            script_pubkey: vec![0x52].into(),
            height: 101,
            is_coinbase: false,
        }),
    );
    utxo_set
}

#[cfg(all(test, feature = "property-tests"))]
#[test]
fn test_transaction_with_witness_strategy_satisfies_constraint() {
    use proptest::test_runner::Config;
    proptest!(Config::with_cases(50), |((tx, w) in transaction_with_witness_strategy())| {
        prop_assert_eq!(w.len(), tx.inputs.len());
    });
}
