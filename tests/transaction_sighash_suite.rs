//! COV-C-03a: Sighash golden-path coverage (legacy, ANYONECANPAY, BIP143).

use blvm_consensus::opcodes::{OP_1, OP_2, OP_3, OP_DUP, OP_HASH160, PUSH_20_BYTES};
use blvm_consensus::transaction_hash::{
    SighashType, batch_compute_bip143_sighashes, batch_compute_sighashes, calculate_bip143_sighash,
    calculate_transaction_sighash, calculate_transaction_sighash_single_input,
    calculate_transaction_sighash_with_script_code,
};
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

fn sample_tx() -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x11; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x22; 32],
                    index: 1,
                },
                script_sig: vec![].into(),
                sequence: 0xfffffffe,
            },
        ]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 50_000,
                script_pubkey: vec![OP_1].into(),
            },
            TransactionOutput {
                value: 40_000,
                script_pubkey: vec![OP_2].into(),
            },
        ]
        .into(),
        lock_time: 100,
    }
}

fn sample_prevouts() -> Vec<TransactionOutput> {
    vec![
        TransactionOutput {
            value: 100_000,
            script_pubkey: vec![OP_DUP, OP_HASH160, PUSH_20_BYTES].into(), // partial P2PKH prefix
        },
        TransactionOutput {
            value: 200_000,
            script_pubkey: vec![OP_1].into(),
        },
    ]
}

#[test]
fn test_legacy_sighash_all_differs_from_anyonecanpay() {
    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let all = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL).unwrap();
    let acp =
        calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL_ANYONECANPAY).unwrap();
    assert_ne!(all, acp, "ANYONECANPAY must change legacy sighash");
}

#[test]
fn test_legacy_sighash_none_differs_from_all() {
    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let all = calculate_transaction_sighash(&tx, 1, &prevouts, SighashType::ALL).unwrap();
    let none = calculate_transaction_sighash(&tx, 1, &prevouts, SighashType::NONE).unwrap();
    assert_ne!(all, none, "SIGHASH_NONE must differ from ALL");
}

#[test]
fn test_legacy_sighash_single_uses_output_index() {
    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let single_on_input0 =
        calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::SINGLE).unwrap();
    let single_on_input1 =
        calculate_transaction_sighash(&tx, 1, &prevouts, SighashType::SINGLE).unwrap();
    assert_ne!(
        single_on_input0, single_on_input1,
        "SINGLE sighash depends on input index"
    );
}

#[test]
fn test_bip143_differs_from_legacy_for_witness_input() {
    let tx = sample_tx();
    let script_code = vec![OP_1, OP_2, OP_3];
    let prevouts = sample_prevouts();
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_scripts: Vec<&[u8]> = prevouts.iter().map(|p| p.script_pubkey.as_ref()).collect();
    let legacy = calculate_transaction_sighash_with_script_code(
        &tx,
        0,
        &prevout_values,
        &prevout_scripts,
        SighashType::ALL,
        Some(&script_code),
        #[cfg(feature = "production")]
        None,
    )
    .unwrap();
    let witness = calculate_bip143_sighash(
        &tx,
        0,
        &script_code,
        prevout_values[0],
        SighashType::ALL.0,
        None,
    )
    .unwrap();
    assert_ne!(
        legacy, witness,
        "BIP143 witness sighash must differ from legacy"
    );
}

#[test]
fn test_bip143_anyonecanpay_differs_from_all() {
    let tx = sample_tx();
    let script_code = vec![OP_1];
    let amount = 100_000i64;
    let all = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x01, None).unwrap();
    let acp = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x81, None).unwrap();
    assert_ne!(all, acp, "BIP143 ANYONECANPAY must change sighash");
}

#[test]
fn test_bip143_none_differs_from_all() {
    let tx = sample_tx();
    let script_code = vec![OP_1];
    let amount = 100_000i64;
    let all = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x01, None).unwrap();
    let none = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x02, None).unwrap();
    assert_ne!(all, none, "BIP143 SIGHASH_NONE must differ from ALL");
}

#[test]
fn test_legacy_sighash_single_out_of_range_returns_one() {
    let tx = Transaction {
        version: 2,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x11; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x22; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![
        TransactionOutput {
            value: 100_000,
            script_pubkey: vec![OP_1].into(),
        },
        TransactionOutput {
            value: 200_000,
            script_pubkey: vec![OP_1].into(),
        },
    ];
    let hash = calculate_transaction_sighash(&tx, 1, &prevouts, SighashType::SINGLE).unwrap();
    let mut expected = [0u8; 32];
    expected[0] = 1;
    assert_eq!(
        hash, expected,
        "SIGHASH_SINGLE with input_index >= outputs.len() must return 0x0000...0001"
    );
}

#[test]
fn test_legacy_all_legacy_differs_from_all_in_preimage() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x33; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: 100_000,
        script_pubkey: vec![OP_1].into(),
    }];
    let all = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL).unwrap();
    let legacy = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL_LEGACY).unwrap();
    assert_ne!(
        all, legacy,
        "raw sighash byte (0x00 vs 0x01) is included in the legacy preimage"
    );
}

#[test]
fn test_bip143_sighash_single_out_of_range_differs_from_in_range() {
    let tx = Transaction {
        version: 2,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x44; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x55; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let script_code = vec![OP_1];
    let amount = 100_000i64;
    let in_range = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x03, None).unwrap();
    let out_of_range = calculate_bip143_sighash(&tx, 1, &script_code, amount, 0x03, None).unwrap();
    assert_ne!(in_range, out_of_range);
}

#[test]
fn test_single_input_api_matches_full_prevouts_for_one_in_one_out() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x66; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = vec![TransactionOutput {
        value: 100_000,
        script_pubkey: vec![OP_1].into(),
    }];
    let full = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL).unwrap();
    let single = calculate_transaction_sighash_single_input(
        &tx,
        0,
        prevouts[0].script_pubkey.as_ref(),
        prevouts[0].value,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .unwrap();
    assert_eq!(full, single);
}

#[test]
fn test_legacy_none_anyonecanpay_differs_from_none() {
    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let none = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::NONE).unwrap();
    let none_acp =
        calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::NONE_ANYONECANPAY).unwrap();
    assert_ne!(none, none_acp);
}

#[test]
fn test_bip143_single_anyonecanpay_differs_from_single() {
    let tx = sample_tx();
    let script_code = vec![OP_1];
    let amount = 100_000i64;
    let single = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x03, None).unwrap();
    let single_acp = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x83, None).unwrap();
    assert_ne!(single, single_acp);
}

#[test]
fn test_batch_compute_sighashes_matches_per_input() {
    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let batch = batch_compute_sighashes(&tx, &prevouts, SighashType::ALL).unwrap();
    assert_eq!(batch.len(), tx.inputs.len());
    for i in 0..tx.inputs.len() {
        let single = calculate_transaction_sighash(&tx, i, &prevouts, SighashType::ALL).unwrap();
        assert_eq!(
            batch[i], single,
            "batch entry {i} must match single-input sighash"
        );
    }
}

#[test]
fn test_batch_compute_sighashes_rejects_prevout_mismatch() {
    let tx = sample_tx();
    let prevouts = vec![sample_prevouts()[0].clone()];
    assert!(batch_compute_sighashes(&tx, &prevouts, SighashType::ALL).is_err());
}

#[test]
fn test_batch_compute_bip143_sighashes_matches_single() {
    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_scripts: Vec<&[u8]> = prevouts.iter().map(|p| p.script_pubkey.as_ref()).collect();
    let script_codes: Vec<&[u8]> = prevout_scripts.clone();
    let batch = batch_compute_bip143_sighashes(
        &tx,
        &prevout_values,
        &prevout_scripts,
        &script_codes,
        SighashType::ALL.0,
    )
    .unwrap();
    assert_eq!(batch.len(), tx.inputs.len());
    for i in 0..tx.inputs.len() {
        let single = calculate_bip143_sighash(
            &tx,
            i,
            script_codes[i],
            prevout_values[i],
            SighashType::ALL.0,
            None,
        )
        .unwrap();
        assert_eq!(batch[i], single);
    }
}

#[cfg(feature = "production")]
#[test]
fn test_production_legacy_sighash_nocache_and_buffered_match() {
    use blvm_consensus::transaction_hash::{
        compute_legacy_sighash_buffered, compute_legacy_sighash_nocache, compute_sighashes_batch,
    };

    let tx = sample_tx();
    let script_code = vec![OP_1, OP_2, OP_3];
    let sighash_byte = SighashType::ALL.0;

    let nocache = compute_legacy_sighash_nocache(&tx, 0, &script_code, sighash_byte);
    let buffered = compute_legacy_sighash_buffered(&tx, 0, &script_code, sighash_byte);
    assert_eq!(nocache, buffered, "buffered and nocache paths must agree");

    let script_codes: Vec<&[u8]> = vec![&script_code; tx.inputs.len()];
    let sighash_bytes = vec![sighash_byte; tx.inputs.len()];
    let batch = compute_sighashes_batch(&tx, &script_codes, &sighash_bytes);
    assert_eq!(batch.len(), tx.inputs.len());
    assert_eq!(batch[0], nocache);
}

#[cfg(feature = "production")]
#[test]
fn test_batch_compute_legacy_sighashes_matches_single() {
    use blvm_consensus::transaction_hash::batch_compute_legacy_sighashes;

    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_scripts: Vec<&[u8]> = prevouts.iter().map(|p| p.script_pubkey.as_ref()).collect();
    let script_code = vec![OP_1, OP_2, OP_3];
    let specs = vec![(0usize, SighashType::ALL.0, script_code.as_slice())];

    let batch =
        batch_compute_legacy_sighashes(&tx, &prevout_values, &prevout_scripts, &specs).unwrap();
    let single = calculate_transaction_sighash_with_script_code(
        &tx,
        0,
        &prevout_values,
        &prevout_scripts,
        SighashType::ALL,
        Some(&script_code),
        None,
    )
    .unwrap();
    assert_eq!(batch[0], single);
}

#[cfg(feature = "production")]
#[test]
fn test_batch_compute_legacy_sighashes_rejects_prevout_length_mismatch() {
    use blvm_consensus::transaction_hash::batch_compute_legacy_sighashes;

    let tx = sample_tx();
    let script_code = vec![OP_1];
    let specs = vec![(0usize, SighashType::ALL.0, script_code.as_slice())];
    assert!(batch_compute_legacy_sighashes(&tx, &[100_000], &[&script_code[..]], &specs,).is_err());
}

#[cfg(feature = "production")]
#[test]
fn test_sighash_midstate_cache_returns_same_hash() {
    use blvm_consensus::transaction_hash::SighashMidstateCache;
    use std::sync::{Arc, Mutex};

    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_scripts: Vec<&[u8]> = prevouts.iter().map(|p| p.script_pubkey.as_ref()).collect();
    let script_code = vec![OP_1, OP_2, OP_3];
    let cache: SighashMidstateCache = Arc::new(Mutex::new(rustc_hash::FxHashMap::default()));

    let first = calculate_transaction_sighash_with_script_code(
        &tx,
        0,
        &prevout_values,
        &prevout_scripts,
        SighashType::ALL,
        Some(&script_code),
        Some(&cache),
    )
    .unwrap();
    let second = calculate_transaction_sighash_with_script_code(
        &tx,
        0,
        &prevout_values,
        &prevout_scripts,
        SighashType::ALL,
        Some(&script_code),
        Some(&cache),
    )
    .unwrap();
    assert_eq!(first, second);
    assert!(cache.lock().unwrap().len() >= 1);
}

#[test]
fn test_bip143_precomputed_hashes_match_fresh_compute() {
    use blvm_consensus::transaction_hash::Bip143PrecomputedHashes;

    let tx = sample_tx();
    let prevouts = sample_prevouts();
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_scripts: Vec<&[u8]> = prevouts.iter().map(|p| p.script_pubkey.as_ref()).collect();
    let script_code = vec![OP_1, OP_2, OP_3];
    let precomputed = Bip143PrecomputedHashes::compute(&tx, &prevout_values, &prevout_scripts);

    let fresh = calculate_bip143_sighash(
        &tx,
        0,
        &script_code,
        prevout_values[0],
        SighashType::ALL.0,
        None,
    )
    .unwrap();
    let cached = calculate_bip143_sighash(
        &tx,
        0,
        &script_code,
        prevout_values[0],
        SighashType::ALL.0,
        Some(&precomputed),
    )
    .unwrap();
    assert_eq!(fresh, cached);
}

#[test]
fn test_bip143_none_and_all_differ() {
    let tx = sample_tx();
    let script_code = vec![OP_1];
    let amount = 100_000i64;
    let all =
        calculate_bip143_sighash(&tx, 0, &script_code, amount, SighashType::ALL.0, None).unwrap();
    let none =
        calculate_bip143_sighash(&tx, 0, &script_code, amount, SighashType::NONE.0, None).unwrap();
    assert_ne!(all, none);
}

#[test]
fn test_sighash_apis_reject_out_of_range_input_index() {
    let tx = sample_tx();
    let prevouts = sample_prevouts();
    assert!(calculate_transaction_sighash(&tx, 99, &prevouts, SighashType::ALL).is_err());
    assert!(calculate_bip143_sighash(&tx, 99, &[OP_1], 1, SighashType::ALL.0, None).is_err());
    assert!(
        calculate_transaction_sighash_single_input(
            &tx,
            99,
            &[OP_1],
            1,
            SighashType::ALL,
            #[cfg(feature = "production")]
            None,
        )
        .is_err()
    );
}
