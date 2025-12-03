//! Transaction size calculation edge cases
//!
//! Tests for consensus-critical transaction size calculation that must match
//! Bitcoin Core's GetSerializeSize(TX_NO_WITNESS(tx)) exactly.
//!
//! Consensus-critical: Size calculation differences can cause different validation results.

use blvm_consensus::serialization::transaction::serialize_transaction;
use blvm_consensus::transaction::calculate_transaction_size;
use blvm_consensus::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};

/// Test that calculate_transaction_size matches actual serialization size
///
/// This is the critical test: the size calculation must match the actual
/// serialized size (without witness data) to match Bitcoin Core's behavior.
#[test]
fn test_transaction_size_matches_serialization() {
    // Simple transaction
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51], // OP_1
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(), // OP_1
        }]
        .into(),
        lock_time: 0,
    };

    let calculated_size = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    let actual_size = serialized.len();

    // CRITICAL: Calculated size must match actual serialized size
    assert_eq!(
        calculated_size, actual_size,
        "Transaction size calculation ({}) must match actual serialization size ({})",
        calculated_size, actual_size
    );
}

/// Test transaction size with variable-length scripts (varint encoding)
///
/// Varint encoding means script lengths affect the total size:
/// - Script length < 0xfd: 1 byte varint
/// - Script length >= 0xfd: 3 bytes varint (0xfd + 2 bytes)
#[test]
fn test_transaction_size_varint_script_lengths() {
    // Test with script length = 0xfc (1 byte varint)
    let tx_small_script = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51; 0xfc], // 252 bytes (0xfc)
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx_small_script);
    let serialized = serialize_transaction(&tx_small_script);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must account for varint encoding (script length 0xfc): calculated={}, actual={}",
        calculated, serialized.len()
    );

    // Test with script length = 0xfd (3 byte varint)
    let tx_medium_script = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51; 0xfd], // 253 bytes (0xfd)
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx_medium_script);
    let serialized = serialize_transaction(&tx_medium_script);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must account for varint encoding (script length 0xfd): calculated={}, actual={}",
        calculated, serialized.len()
    );
}

/// Test transaction size with multiple inputs/outputs (varint encoding for counts)
#[test]
fn test_transaction_size_varint_counts() {
    // Test with input count = 0xfc (1 byte varint)
    let mut inputs = Vec::new();
    for i in 0..0xfc {
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        });
    }

    let tx = Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must account for varint encoding of input count: calculated={}, actual={}",
        calculated, serialized.len()
    );
}

/// Test transaction size edge case: empty scripts
#[test]
fn test_transaction_size_empty_scripts() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![], // Empty script
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(), // Empty script
        }]
        .into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    assert_eq!(
        calculated,
        serialized.len(),
        "Size calculation must handle empty scripts correctly: calculated={}, actual={}",
        calculated,
        serialized.len()
    );
}

/// Test that size calculation matches Core's TX_NO_WITNESS behavior
#[test]
fn test_transaction_size_no_witness() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51, 0x52, 0x53],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32].into(),
                    index: 1,
                },
                script_sig: vec![0x54, 0x55],
                sequence: 0xfffffffe,
            },
        ]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51, 0x52].into(),
            },
            TransactionOutput {
                value: 2000,
                script_pubkey: vec![0x53, 0x54, 0x55].into(),
            },
        ]
        .into(),
        lock_time: 12345,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);

    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must match non-witness serialization (TX_NO_WITNESS): calculated={}, actual={}",
        calculated, serialized.len()
    );
}

/// Test transaction size with coinbase transaction
#[test]
fn test_transaction_size_coinbase() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00], // Height encoding (5 bytes)
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000, // 50 BTC
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    assert_eq!(
        calculated,
        serialized.len(),
        "Size calculation must handle coinbase transactions correctly: calculated={}, actual={}",
        calculated,
        serialized.len()
    );
}
