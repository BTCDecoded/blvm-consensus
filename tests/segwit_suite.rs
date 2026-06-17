//! COV-C-06d: SegWit weight, witness merkle, and commitment validation.

use bitcoin_hashes::{Hash as BitcoinHash, sha256, sha256d};
use blvm_consensus::constants::MAX_BLOCK_WEIGHT;
use blvm_consensus::opcodes::{OP_0, OP_1, OP_CHECKSIG, OP_RETURN, PUSH_32_BYTES, PUSH_36_BYTES};
use blvm_consensus::segwit::{
    Witness, calculate_block_weight_from_nested, calculate_transaction_weight,
    compute_witness_merkle_root, compute_witness_merkle_root_from_nested, is_segwit_transaction,
    validate_witness_commitment,
};
use blvm_consensus::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput,
};

const WITNESS_COMMITMENT_MAGIC: [u8; 4] = [0xaa, 0x21, 0xa9, 0xed];

fn coinbase(value: i64) -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1, OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

fn p2wpkh_scriptpubkey() -> Vec<u8> {
    let mut spk = vec![OP_0, 0x14];
    spk.extend_from_slice(&[0xab; 20]);
    spk
}

fn witness_commitment_script(witness_root: &[u8; 32], nonce: &[u8; 32]) -> Vec<u8> {
    let mut preimage = [0u8; 64];
    preimage[..32].copy_from_slice(witness_root);
    preimage[32..].copy_from_slice(nonce);
    let h = sha256d::Hash::hash(&preimage);
    let mut script = vec![OP_RETURN, PUSH_36_BYTES];
    script.extend_from_slice(&WITNESS_COMMITMENT_MAGIC);
    script.extend_from_slice(&h[..]);
    script
}

#[test]
fn test_is_segwit_transaction_detects_v0_witness_program() {
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x01; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: p2wpkh_scriptpubkey().into(),
        }]
        .into(),
        lock_time: 0,
    };
    assert!(is_segwit_transaction(&tx));
    assert!(!is_segwit_transaction(&coinbase(50_000_000_000)));
}

#[test]
fn test_calculate_transaction_weight_includes_witness_bytes() {
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x02; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: p2wpkh_scriptpubkey().into(),
        }]
        .into(),
        lock_time: 0,
    };
    let base = calculate_transaction_weight(&tx, None).unwrap();
    let witness: Witness = vec![vec![0x30; 72], vec![0x21; 33]];
    let with_witness = calculate_transaction_weight(&tx, Some(&witness)).unwrap();
    assert!(with_witness > base);
}

#[test]
fn test_compute_witness_merkle_root_single_coinbase() {
    let block = Block {
        header: BlockHeader {
            version: 2,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1_500_000_000,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![coinbase(50_000_000_000)].into(),
    };
    let witnesses = vec![Witness::default()];
    let root = compute_witness_merkle_root(&block, &witnesses).unwrap();
    assert_eq!(
        root, [0u8; 32],
        "single coinbase witness merkle root is zero per BIP141"
    );
}

#[test]
fn test_validate_witness_commitment_accepts_matching_op_return() {
    let nonce = [0x11u8; 32];
    let block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1_500_000_000,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![coinbase(50_000_000_000)].into(),
    };
    let witnesses: Vec<Vec<Witness>> = vec![vec![vec![nonce.to_vec()]]];
    let root = compute_witness_merkle_root_from_nested(&block, &witnesses).unwrap();

    let mut cb = coinbase(50_000_000_000);
    cb.outputs = vec![
        TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        },
        TransactionOutput {
            value: 0,
            script_pubkey: witness_commitment_script(&root, &nonce).into(),
        },
    ]
    .into();

    assert!(validate_witness_commitment(&cb, &root, &witnesses[0]).unwrap());
}

#[test]
fn test_validate_witness_commitment_rejects_wrong_hash() {
    let nonce = [0x22u8; 32];
    let root = [0x33u8; 32];
    let cb = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: witness_commitment_script(&[0xff; 32], &nonce).into(),
        }]
        .into(),
        lock_time: 0,
    };
    let witnesses = vec![vec![nonce.to_vec()]];
    assert!(!validate_witness_commitment(&cb, &root, &witnesses).unwrap());
}

#[test]
fn test_validate_witness_commitment_accepts_missing_commitment_output() {
    let cb = coinbase(50_000_000_000);
    let root = [0x44u8; 32];
    let witnesses = vec![Witness::default()];
    assert!(validate_witness_commitment(&cb, &root, &witnesses).unwrap());
}

#[test]
fn test_validate_witness_commitment_uses_last_matching_output() {
    let nonce = [0x55u8; 32];
    let block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1_500_000_000,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![coinbase(50_000_000_000)].into(),
    };
    let witnesses: Vec<Vec<Witness>> = vec![vec![vec![nonce.to_vec()]]];
    let root = compute_witness_merkle_root_from_nested(&block, &witnesses).unwrap();

    let mut cb = coinbase(50_000_000_000);
    cb.outputs = vec![
        TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        },
        TransactionOutput {
            value: 0,
            script_pubkey: witness_commitment_script(&[0xff; 32], &nonce).into(),
        },
        TransactionOutput {
            value: 0,
            script_pubkey: witness_commitment_script(&root, &nonce).into(),
        },
    ]
    .into();

    assert!(validate_witness_commitment(&cb, &root, &witnesses[0]).unwrap());
}

#[test]
fn test_calculate_block_weight_from_nested_under_limit() {
    let witness_script = vec![OP_CHECKSIG];
    let hash = sha256::Hash::hash(&witness_script);
    let mut spk = vec![OP_0, PUSH_32_BYTES];
    spk.extend_from_slice(hash.as_ref());

    let spend = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x44; 32],
                index: 0,
            },
            script_sig: vec![].into(),
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

    let block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1_500_000_000,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![coinbase(50_000_000_000), spend].into(),
    };
    let witnesses: Vec<Vec<Witness>> =
        vec![vec![Witness::default()], vec![vec![witness_script.clone()]]];
    let weight = calculate_block_weight_from_nested(&block, &witnesses).unwrap();
    assert!(weight <= MAX_BLOCK_WEIGHT as u64);
}
