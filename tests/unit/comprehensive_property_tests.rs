//! Property tests that call consensus checks on generated transactions and headers.

use blvm_consensus::bip_validation::check_bip54_timewarp;
use blvm_consensus::constants::{MAX_BLOCK_WEIGHT, MAX_MONEY};
use blvm_consensus::mining;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::script::{self, SigVersion};
use blvm_consensus::segwit::{self, Witness};
use blvm_consensus::serialization::{
    deserialize_block_header, deserialize_transaction, serialize_block_header,
    serialize_transaction,
};
use blvm_consensus::transaction::{self, check_transaction};
use blvm_consensus::types::*;
use proptest::prelude::*;
use sha2::{Digest, Sha256};

fn make_header(version: i64, timestamp: u64, nonce: u64) -> BlockHeader {
    BlockHeader {
        version,
        prev_block_hash: [0; 32],
        merkle_root: [1; 32],
        timestamp,
        bits: 0x1d00ffff,
        nonce,
    }
}

fn make_tx(version: u64, lock_time: u64) -> Transaction {
    Transaction {
        version,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![OP_1],
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time,
    }
}

fn round_trip_tx(tx: &Transaction) -> Transaction {
    deserialize_transaction(&serialize_transaction(tx)).unwrap()
}

fn round_trip_header(header: &BlockHeader) -> BlockHeader {
    deserialize_block_header(&serialize_block_header(header)).unwrap()
}

fn sha256d(data: &[u8]) -> [u8; 32] {
    let second = Sha256::digest(Sha256::digest(data));
    let mut out = [0u8; 32];
    out.copy_from_slice(&second);
    out
}

fn tx_with_inputs(inputs: Vec<TransactionInput>, value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    }
}

fn one_input(hash: [u8; 32], index: u32, script_sig: Vec<u8>) -> TransactionInput {
    TransactionInput {
        prevout: OutPoint { hash, index },
        script_sig,
        sequence: 0xffff_ffff,
    }
}

fn is_valid(tx: &Transaction) -> bool {
    matches!(check_transaction(tx), Ok(ValidationResult::Valid))
}

/// The wire version is 32 bits. A wider value comes back as its low half.
proptest! {
    #[test]
    fn prop_transaction_version_valid(low in any::<u32>()) {
        let narrow = make_tx(u64::from(low), 0);
        prop_assert_eq!(round_trip_tx(&narrow).version, u64::from(low));
        let wide = make_tx(u64::from(low) + (1u64 << 32), 0);
        prop_assert_eq!(round_trip_tx(&wide).version, u64::from(low));
    }
}

/// The wire lock time is 32 bits.
proptest! {
    #[test]
    fn prop_transaction_lock_time(low in any::<u32>()) {
        let narrow = make_tx(1, u64::from(low));
        prop_assert_eq!(round_trip_tx(&narrow).lock_time, u64::from(low));
        let wide = make_tx(1, u64::from(low) + (1u64 << 32));
        prop_assert_eq!(round_trip_tx(&wide).lock_time, u64::from(low));
    }
}

/// The wire sequence is 32 bits.
proptest! {
    #[test]
    fn prop_transaction_sequence(low in any::<u32>()) {
        let mut tx = make_tx(1, 0);
        tx.inputs[0].sequence = u64::from(low);
        prop_assert_eq!(round_trip_tx(&tx).inputs[0].sequence, u64::from(low));
        tx.inputs[0].sequence = u64::from(low) + (1u64 << 32);
        prop_assert_eq!(round_trip_tx(&tx).inputs[0].sequence, u64::from(low));
    }
}

/// Header nonce survives the 80-byte header encoding.
proptest! {
    #[test]
    fn prop_block_header_nonce(nonce in 0u64..=u32::MAX as u64) {
        let header = make_header(1, 1_234_567_890, nonce);
        prop_assert_eq!(round_trip_header(&header).nonce, nonce);
    }
}

/// Two inputs are valid exactly when their prevouts differ.
proptest! {
    #[test]
    fn prop_outpoint_hash_uniqueness(
        hash1 in prop::array::uniform32(any::<u8>()),
        hash2 in prop::array::uniform32(any::<u8>()),
        index in 0u32..1000u32
    ) {
        let tx = tx_with_inputs(
            vec![
                one_input(hash1, index, vec![OP_1]),
                one_input(hash2, index, vec![OP_1]),
            ],
            1000,
        );
        prop_assert_eq!(is_valid(&tx), hash1 != hash2);
    }
}

/// A coinbase prevout is the zero hash at index `0xffffffff`, and that pair round-trips.
proptest! {
    #[test]
    fn prop_outpoint_index_range(index in 0u32..=u32::MAX) {
        let hash = [0u8; 32];
        let tx = tx_with_inputs(vec![one_input(hash, index, vec![0x03, 0x00])], 1000);
        prop_assert_eq!(transaction::is_coinbase(&tx), index == 0xffff_ffff);
        prop_assert_eq!(round_trip_tx(&tx).inputs[0].prevout.index, index);
    }
}

/// One output is valid exactly when its value is inside MoneyRange.
proptest! {
    #[test]
    fn prop_transaction_output_value_bounds(which in 0usize..6usize) {
        let value = [-1, 0, 1, MAX_MONEY - 1, MAX_MONEY, MAX_MONEY + 1][which];
        let tx = tx_with_inputs(vec![one_input([1; 32], 0, vec![OP_1])], value);
        prop_assert_eq!(is_valid(&tx), (0..=MAX_MONEY).contains(&value));
    }
}

/// A push larger than 520 bytes is a push-size error. A smaller push leaves those bytes.
proptest! {
    #[test]
    fn prop_script_pubkey_size_bounds(script_size in 0usize..=530usize) {
        let script_bytes = if script_size <= 75 {
            let mut script = vec![script_size as u8];
            script.extend(std::iter::repeat(0x11).take(script_size));
            script
        } else {
            let mut script = vec![0x4e];
            script.extend_from_slice(&(script_size as u32).to_le_bytes());
            script.extend(std::iter::repeat(0x11).take(script_size));
            script
        };
        let mut stack = Vec::new();
        let result = script::eval_script(&script_bytes, &mut stack, 0, SigVersion::Base);
        if script_size > 520 {
            let code = match result {
                Err(blvm_consensus::ConsensusError::ScriptErrorWithCode { code, .. }) => Some(code),
                _ => None,
            };
            prop_assert_eq!(code, Some(blvm_consensus::error::ScriptErrorCode::PushSize));
        } else {
            prop_assert_eq!(result.unwrap(), true);
            prop_assert_eq!(stack.len(), 1);
            prop_assert_eq!(stack[0].len(), script_size);
        }
    }
}

/// Prevout hash and index survive transaction serialization.
proptest! {
    #[test]
    fn prop_transaction_input_prevout(
        hash_bytes in prop::array::uniform32(any::<u8>()),
        index in any::<u32>()
    ) {
        let tx = tx_with_inputs(vec![one_input(hash_bytes, index, vec![OP_1])], 1000);
        let back = round_trip_tx(&tx);
        prop_assert_eq!(back.inputs[0].prevout.hash, hash_bytes);
        prop_assert_eq!(back.inputs[0].prevout.index, index);
    }
}

/// BIP54 timewarp compares the header timestamp with the period-boundary times.
proptest! {
    #[test]
    fn prop_block_timestamp_progression(
        parent_time in 1_600_000_000u64..1_800_000_000u64,
        delta in -10_000i64..10_000i64,
        kind in 0u8..3u8
    ) {
        let timestamp = (parent_time as i64 + delta) as u64;
        let header = make_header(1, timestamp, 0);
        let boundary = Bip54BoundaryTimestamps {
            timestamp_n_minus_1: parent_time,
            timestamp_n_minus_2015: parent_time,
        };
        let height = match kind {
            0 => 2016,
            1 => 2015,
            _ => 100,
        };
        let active = check_bip54_timewarp(&header, height, Some(&boundary), true);
        let expect = match kind {
            0 => timestamp >= parent_time.saturating_sub(7200),
            1 => timestamp >= parent_time,
            _ => true,
        };
        prop_assert_eq!(active, expect);
        prop_assert!(check_bip54_timewarp(&header, height, Some(&boundary), false));
    }
}

/// Header version survives the 80-byte encoding, including versions below 1.
proptest! {
    #[test]
    fn prop_block_header_version_consistency(version in -2i64..10i64) {
        let header = make_header(version, 1_234_567_890, 0);
        prop_assert_eq!(round_trip_header(&header).version, version);
    }
}

/// A one-transaction merkle root is the transaction id. Two transactions hash the pair.
proptest! {
    #[test]
    fn prop_merkle_root_format(tag in 1u8..20u8) {
        let tx1 = make_tx(1, u64::from(tag));
        let mut tx2 = make_tx(1, u64::from(tag) + 1);
        tx2.inputs[0].prevout.hash = [tag; 32];
        let id1 = sha256d(&serialize_transaction(&tx1));
        let id2 = sha256d(&serialize_transaction(&tx2));
        prop_assert_eq!(mining::calculate_merkle_root(&[tx1]).unwrap(), id1);
        let mut pair = [0u8; 64];
        pair[..32].copy_from_slice(&id1);
        pair[32..].copy_from_slice(&id2);
        prop_assert_eq!(mining::calculate_merkle_root(&[make_tx(1, u64::from(tag)), tx2]).unwrap(), sha256d(&pair));
    }
}

/// Previous-block hash survives the header encoding.
proptest! {
    #[test]
    fn prop_prev_block_hash_format(prev_hash_bytes in prop::array::uniform32(any::<u8>())) {
        let mut header = make_header(1, 1_234_567_890, 0);
        header.prev_block_hash = prev_hash_bytes;
        prop_assert_eq!(round_trip_header(&header).prev_block_hash, prev_hash_bytes);
    }
}

/// A coinbase scriptSig is valid exactly at lengths 2..=100.
proptest! {
    #[test]
    fn prop_coinbase_structure(script_len in 0usize..120usize) {
        let tx = tx_with_inputs(
            vec![one_input([0; 32], 0xffff_ffff, vec![0x01; script_len])],
            1000,
        );
        prop_assert!(transaction::is_coinbase(&tx));
        prop_assert_eq!(is_valid(&tx), (2..=100).contains(&script_len));
    }
}

/// A transaction needs at least one output.
proptest! {
    #[test]
    fn prop_transaction_output_count(output_count in 0usize..8usize) {
        let tx = Transaction {
            version: 1,
            inputs: vec![one_input([2; 32], 0, vec![OP_1])].into(),
            outputs: (0..output_count)
                .map(|_| TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![OP_1],
                })
                .collect(),
            lock_time: 0,
        };
        prop_assert_eq!(is_valid(&tx), output_count > 0);
    }
}

/// A transaction needs at least one input, and distinct prevouts stay valid.
proptest! {
    #[test]
    fn prop_transaction_input_count(input_count in 0usize..8usize) {
        let tx = Transaction {
            version: 1,
            inputs: (0..input_count)
                .map(|i| one_input([i as u8; 32], i as u32, vec![OP_1]))
                .collect(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![OP_1],
            }]
            .into(),
            lock_time: 0,
        };
        prop_assert_eq!(is_valid(&tx), input_count > 0);
    }
}

/// `is_coinbase` follows the transaction's prevout, whichever position it occupies.
proptest! {
    #[test]
    fn prop_block_coinbase_first(coinbase_first in any::<bool>()) {
        let coinbase = tx_with_inputs(vec![one_input([0; 32], 0xffff_ffff, vec![0x03, 0x00])], 1000);
        let regular = tx_with_inputs(vec![one_input([9; 32], 1, vec![OP_1])], 1000);
        let txs = if coinbase_first {
            vec![coinbase, regular]
        } else {
            vec![regular, coinbase]
        };
        prop_assert_eq!(transaction::is_coinbase(&txs[0]), coinbase_first);
        prop_assert_eq!(transaction::is_coinbase(&txs[1]), !coinbase_first);
    }
}

/// The 2..=100 scriptSig rule is coinbase-only. An ordinary input may use the same lengths.
proptest! {
    #[test]
    fn prop_script_sig_size_bounds(script_sig_size in 0usize..150usize) {
        let script_sig = vec![0x51; script_sig_size];
        let ordinary = tx_with_inputs(vec![one_input([3; 32], 0, script_sig.clone())], 1000);
        prop_assert!(is_valid(&ordinary));
        let coinbase = tx_with_inputs(vec![one_input([0; 32], 0xffff_ffff, script_sig)], 1000);
        prop_assert_eq!(is_valid(&coinbase), (2..=100).contains(&script_sig_size));
    }
}

/// Block weight counts the header and the transaction count at four bytes each.
proptest! {
    #[test]
    fn prop_block_size_bounds(tx_count in 1usize..30usize) {
        let txs: Vec<Transaction> = (0..tx_count)
            .map(|i| tx_with_inputs(vec![one_input([i as u8; 32], 0, vec![OP_1])], 1000))
            .collect();
        let stripped: usize = txs.iter().map(transaction::calculate_transaction_size).sum();
        let block = Block {
            header: make_header(1, 1_234_567_890, 0),
            transactions: txs.into_boxed_slice(),
        };
        let weight = segwit::calculate_block_weight(&block, &[] as &[Witness]).unwrap();
        prop_assert_eq!(weight, 4 * (80 + 1 + stripped) as u64);
        prop_assert!(weight <= MAX_BLOCK_WEIGHT as u64);
    }
}
