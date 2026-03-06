//! Property test strategies for spec-lock generated tests.
//!
//! Enables integration tests to use strategies that satisfy Orange Paper constraints
//! (e.g. |w| = |tx.inputs| for SegWit round-trip).

#[cfg(any(test, feature = "property-tests"))]
use proptest::prelude::*;

use crate::segwit::Witness;
use crate::types::Transaction;
use crate::types::{OutPoint, TransactionInput, TransactionOutput};

#[cfg(any(test, feature = "property-tests"))]
/// Strategy yielding (Transaction, Vec<Witness>) with |w| = |tx.inputs|.
/// Use for SegWit round-trip property tests per Orange Paper 8.2.2.
pub fn transaction_with_witness_strategy(
) -> impl Strategy<Value = (Transaction, Vec<Witness>)> {
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
fn transaction_with_input_count_strategy(
    input_count: usize,
) -> impl Strategy<Value = Transaction> {
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
    (0..10usize, 0..10usize).prop_map(|(input_count, output_count)| {
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

#[cfg(all(test, feature = "property-tests"))]
#[test]
fn test_transaction_with_witness_strategy_satisfies_constraint() {
    use proptest::test_runner::Config;
    proptest!(Config::with_cases(50), |((tx, w) in transaction_with_witness_strategy())| {
        prop_assert_eq!(w.len(), tx.inputs.len());
    });
}
