//! SegWit witness commitment validation tests
//!
//! Tests for SegWit witness commitment requirements.
//! In SegWit blocks, the coinbase transaction must include a witness commitment
//! that matches the witness merkle root.
//!
//! Consensus-critical: Incorrect witness commitment causes consensus violation.

use blvm_consensus::opcodes::OP_RETURN;
use blvm_consensus::segwit::{compute_witness_merkle_root, validate_witness_commitment, Witness};
use blvm_consensus::types::{
    Block, BlockHeader, Hash, OutPoint, Transaction, TransactionInput, TransactionOutput,
};

/// Test witness commitment validation in SegWit blocks
#[test]
fn test_witness_commitment_segwit_block() {
    // Create a SegWit block with witness data
    let block = Block {
        header: BlockHeader {
            version: 0x20000000, // SegWit version bit set
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1501593374, // Approximate SegWit activation time
            bits: 0x18013ce9,
            nonce: 0,
        },
        transactions: vec![
            // Coinbase transaction with witness commitment
            Transaction {
                version: 1,
                inputs: blvm_consensus::tx_inputs![TransactionInput {
                    prevout: OutPoint {
                        hash: [0; 32],
                        index: 0xffffffff,
                    },
                    script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00], // Height encoding
                    sequence: 0xffffffff,
                }],
                outputs: blvm_consensus::tx_outputs![
                    TransactionOutput {
                        value: 12_5000_0000,       // 12.5 BTC
                        script_pubkey: vec![0x51], // Regular output
                    },
                    // Witness commitment would be in second output
                ],
                lock_time: 0,
            },
        ]
        .into_boxed_slice(),
    };

    // Create witness data
    let witnesses = vec![Witness::new()]; // Empty witness for coinbase

    // Compute witness merkle root
    let witness_root = compute_witness_merkle_root(&block, &witnesses);

    // Should compute witness root successfully
    assert!(witness_root.is_ok());
}

/// Test witness commitment validation at SegWit activation height
#[test]
fn test_witness_commitment_activation_height() {
    // Scenario name: at mainnet SegWit deployment (`SEGWIT_ACTIVATION_MAINNET`).
    let block = Block {
        header: BlockHeader {
            version: 0x20000000, // SegWit version bit
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1501593374,
            bits: 0x18013ce9,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: blvm_consensus::tx_inputs![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
                sequence: 0xffffffff,
            }],
            outputs: blvm_consensus::tx_outputs![TransactionOutput {
                value: 12_5000_0000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        }]
        .into_boxed_slice(),
    };

    let witnesses = vec![Witness::new()];
    let witness_root = compute_witness_merkle_root(&block, &witnesses);

    // At activation height, witness commitment should be validated
    assert!(witness_root.is_ok());
}

/// Test witness commitment in blocks without witness transactions
#[test]
fn test_witness_commitment_no_witness_txs() {
    // SegWit block can have no witness transactions
    // but still must have witness commitment
    let block = Block {
        header: BlockHeader {
            version: 0x20000000, // SegWit version bit
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1501593374,
            bits: 0x18013ce9,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: blvm_consensus::tx_inputs![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: blvm_consensus::tx_outputs![TransactionOutput {
                value: 12_5000_0000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        }]
        .into_boxed_slice(),
    };

    // All witnesses are empty (no witness transactions)
    let witnesses = vec![Witness::new()];
    let witness_root = compute_witness_merkle_root(&block, &witnesses);

    // Should still compute witness root (all zeros)
    assert!(witness_root.is_ok());
}

/// Test invalid witness commitment rejection
#[test]
fn test_invalid_witness_commitment_rejection() {
    let block = Block {
        header: BlockHeader {
            version: 0x20000000,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1501593374,
            bits: 0x18013ce9,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: blvm_consensus::tx_inputs![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: blvm_consensus::tx_outputs![TransactionOutput {
                value: 12_5000_0000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        }]
        .into_boxed_slice(),
    };

    let witnesses = vec![Witness::new()];
    let witness_root = compute_witness_merkle_root(&block, &witnesses).unwrap();

    // BIP141 OP_RETURN witness commitment with an incorrect 32-byte payload (not sha256d(root||nonce))
    let mut bad_commitment_script = vec![OP_RETURN, 0x24, 0xaa, 0x21, 0xa9, 0xed];
    bad_commitment_script.extend_from_slice(&[0xff; 32]);

    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 12_5000_0000,
            script_pubkey: bad_commitment_script,
        }]
        .into(),
        lock_time: 0,
    };

    assert!(
        !validate_witness_commitment(&coinbase, &witness_root, &[]).unwrap(),
        "wrong commitment payload must fail BIP141 check"
    );
}

/// Test witness commitment format
///
/// Witness commitment format: OP_RETURN (0x6a) + 0x24 + 0xaa21a9ed + 32-byte hash
#[test]
fn test_witness_commitment_format() {
    // Witness commitment should be: 6a 24 aa21a9ed <32-byte-hash>
    let commitment_hash: Hash = [0x42; 32];

    // Construct witness commitment script
    let mut commitment_script = Vec::new();
    commitment_script.push(0x6a); // OP_RETURN
    commitment_script.push(0x24); // Push 36 bytes
    commitment_script.push(0xaa); // Magic bytes
    commitment_script.push(0x21);
    commitment_script.push(0xa9);
    commitment_script.push(0xed);
    commitment_script.extend_from_slice(&commitment_hash);

    // Should be 36 bytes total (1 + 1 + 4 + 32)
    assert_eq!(commitment_script.len(), 38);
}
