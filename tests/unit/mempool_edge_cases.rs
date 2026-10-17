//! Property tests for mempool validation edge cases.

use blvm_consensus::constants::{MAX_SCRIPT_SIZE, MIN_RELAY_FEE, SEQUENCE_RBF};
use blvm_consensus::economic::calculate_fee;
use blvm_consensus::mempool::{
    self, MempoolResult, has_conflict_with_tx, is_standard_tx, replacement_checks,
};
use blvm_consensus::types::*;
use proptest::prelude::*;
use std::sync::Arc;

fn spend_tx(prevout: OutPoint, value: i64, sequence: u64, script: Vec<u8>) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            script_sig: script,
            sequence,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: p2pkh(),
        }]
        .into(),
        lock_time: 0,
    }
}

fn p2pkh() -> Vec<u8> {
    let mut script = vec![0x76, 0xa9, 0x14];
    script.extend([0x11; 20]);
    script.extend([0x88, 0xac]);
    script
}

fn rejected_message(result: MempoolResult) -> String {
    match result {
        MempoolResult::Rejected(message) => message,
        MempoolResult::Accepted => String::new(),
    }
}

proptest! {
    #[test]
    fn prop_mempool_no_duplicates(tag in any::<u8>()) {
        let tx = spend_tx(OutPoint { hash: [1; 32], index: 0 }, 1000, 0xffff_ffff, vec![tag]);
        let mut pool = mempool::Mempool::new();
        let utxo_set = UtxoSet::default();
        let first = mempool::accept_to_memory_pool(
            &tx, None, &utxo_set, &pool, 0, None, Network::Mainnet, None,
        )
        .unwrap();
        let first_message = rejected_message(first);
        prop_assert!(!first_message.contains("already in mempool"));

        pool.insert(mempool::calculate_tx_id(&tx));
        let second = mempool::accept_to_memory_pool(
            &tx, None, &utxo_set, &pool, 0, None, Network::Mainnet, None,
        )
        .unwrap();
        prop_assert!(rejected_message(second).contains("already in mempool"));
    }
}

proptest! {
    #[test]
    fn prop_mempool_fee_rate_non_negative(
        input_value in 0i64..=50_000,
        output_value in 0i64..=50_000
    ) {
        let prevout = OutPoint { hash: [1; 32], index: 0 };
        let mut utxo_set = UtxoSet::default();
        utxo_set.insert(
            prevout,
            Arc::new(UTXO {
                value: input_value,
                script_pubkey: vec![].into(),
                height: 0,
                is_coinbase: false,
            }),
        );
        let tx = spend_tx(prevout, output_value, 0xffff_ffff, vec![0x51]);
        let fee = calculate_fee(&tx, &utxo_set);
        if output_value <= input_value {
            prop_assert_eq!(fee.unwrap(), input_value - output_value);
        } else {
            prop_assert!(fee.is_err());
        }
    }
}

proptest! {
    #[test]
    fn prop_rbf_fee_rate_requirement(
        signals in any::<bool>(),
        share in any::<bool>(),
        existing_fee in 0i64..20_000,
        new_fee in 0i64..40_000
    ) {
        let input_value = 100_000i64;
        let prev_a = OutPoint { hash: [1; 32], index: 0 };
        let prev_b = if share {
            prev_a
        } else {
            OutPoint { hash: [2; 32], index: 0 }
        };
        let mut utxo_set = UtxoSet::default();
        for prevout in [prev_a, prev_b] {
            utxo_set.insert(
                prevout,
                Arc::new(UTXO {
                    value: input_value,
                    script_pubkey: vec![].into(),
                    height: 0,
                    is_coinbase: false,
                }),
            );
        }
        let sequence = if signals {
            (SEQUENCE_RBF as u64).saturating_sub(1)
        } else {
            0xffff_ffff
        };
        let existing = spend_tx(prev_a, input_value - existing_fee, sequence, vec![0x51]);
        let new_tx = spend_tx(prev_b, input_value - new_fee, 0xffff_ffff, vec![0x52]);
        let accepted = replacement_checks(&new_tx, &existing, &utxo_set, &mempool::Mempool::new()).unwrap();
        let expected = signals && share && new_fee > existing_fee + MIN_RELAY_FEE;
        prop_assert_eq!(accepted, expected);
    }
}

proptest! {
    #[test]
    fn prop_mempool_transaction_size_limits(
        script_len in MAX_SCRIPT_SIZE.saturating_sub(4)..=MAX_SCRIPT_SIZE.saturating_add(4)
    ) {
        let tx = spend_tx(
            OutPoint { hash: [1; 32], index: 0 },
            1000,
            0xffff_ffff,
            vec![0x51; script_len],
        );
        let standard = is_standard_tx(&tx).unwrap();
        prop_assert_eq!(standard, script_len <= MAX_SCRIPT_SIZE);
    }
}

proptest! {
    #[test]
    fn prop_mempool_conflict_detection(
        hash in any::<[u8; 32]>(),
        index in any::<u32>(),
        share in any::<bool>()
    ) {
        let mut other = hash;
        if !share {
            other[0] ^= 0xff;
        }
        let tx1 = spend_tx(OutPoint { hash, index }, 1000, 0xffff_ffff, vec![0x51]);
        let tx2 = spend_tx(OutPoint { hash: other, index }, 1000, 0xffff_ffff, vec![0x52]);
        prop_assert_eq!(has_conflict_with_tx(&tx1, &tx2), share);
    }
}

proptest! {
    #[test]
    fn prop_mempool_fee_calculation(value in 0i64..=20_000) {
        let coinbase = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0xffff_ffff },
                script_sig: vec![0x51, 0x51],
                sequence: 0xffff_ffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value,
                script_pubkey: p2pkh(),
            }]
            .into(),
            lock_time: 0,
        };
        prop_assert_eq!(calculate_fee(&coinbase, &UtxoSet::default()).unwrap(), 0);
    }
}

proptest! {
    #[test]
    fn prop_mempool_removal_consistency(tx_count in 1usize..10) {
        let mut pool = mempool::Mempool::new();
        let mut ids = Vec::new();
        for i in 0..tx_count {
            let tx = spend_tx(
                OutPoint { hash: [i as u8; 32], index: 0 },
                1000,
                0xffff_ffff,
                vec![0x51],
            );
            let id = mempool::calculate_tx_id(&tx);
            prop_assert!(pool.insert(id));
            prop_assert!(pool.contains(&id));
            ids.push(id);
        }
        prop_assert_eq!(pool.len(), tx_count);
        for id in ids {
            prop_assert!(pool.remove(&id));
            prop_assert!(!pool.contains(&id));
            prop_assert!(!pool.remove(&id));
        }
        prop_assert_eq!(pool.len(), 0);
    }
}

proptest! {
    #[test]
    fn prop_mempool_size_limits(tx_count in 1usize..20) {
        let mut pool = mempool::Mempool::new();
        for i in 0..tx_count {
            let tx = spend_tx(
                OutPoint { hash: [i as u8; 32], index: i as u32 },
                1000,
                0xffff_ffff,
                vec![0x51],
            );
            prop_assert!(pool.insert(mempool::calculate_tx_id(&tx)));
        }
        prop_assert_eq!(pool.len(), tx_count);
        let again = spend_tx(
            OutPoint { hash: [0; 32], index: 0 },
            1000,
            0xffff_ffff,
            vec![0x51],
        );
        prop_assert!(!pool.insert(mempool::calculate_tx_id(&again)));
        prop_assert_eq!(pool.len(), tx_count);
    }
}
