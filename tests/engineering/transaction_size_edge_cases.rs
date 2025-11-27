//! Transaction size calculation edge cases
//!
//! Tests for consensus-critical transaction size calculation that must match
//! Bitcoin Core's GetSerializeSize(TX_NO_WITNESS(tx)) exactly.
//!
//! Consensus-critical: Size calculation differences can cause different validation results.

use bllvm_consensus::transaction::calculate_transaction_size;
use bllvm_consensus::serialization::transaction::serialize_transaction;
use bllvm_consensus::types::{Transaction, TransactionInput, TransactionOutput, OutPoint};

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
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(), // OP_1
        }].into(),
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
/// - Script length >= 0x10000: 5 bytes varint (0xfe + 4 bytes)
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
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx_small_script);
    let serialized = serialize_transaction(&tx_small_script);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must account for varint encoding (script length 0xfc)"
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
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx_medium_script);
    let serialized = serialize_transaction(&tx_medium_script);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must account for varint encoding (script length 0xfd)"
    );
    // The size should be 2 bytes larger (3-byte varint vs 1-byte varint)
    assert_eq!(
        calculated, tx_small_script.inputs[0].script_sig.len() + 2,
        "Size should increase by 2 bytes for 3-byte varint"
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
        }].into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must account for varint encoding of input count"
    );

    // Test with input count = 0xfd (3 byte varint)
    let mut inputs_large = Vec::new();
    for i in 0..0xfd {
        inputs_large.push(TransactionInput {
            prevout: OutPoint {
                hash: [i as u8; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        });
    }

    let tx_large = Transaction {
        version: 1,
        inputs: inputs_large.into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx_large);
    let serialized = serialize_transaction(&tx_large);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must account for 3-byte varint encoding of input count"
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
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(), // Empty script
        }].into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must handle empty scripts correctly"
    );
}

/// Test transaction size edge case: maximum script sizes
#[test]
fn test_transaction_size_max_scripts() {
    use bllvm_consensus::constants::MAX_SCRIPT_SIZE;

    // Transaction with maximum-size scriptSig
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51; MAX_SCRIPT_SIZE],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must handle maximum script sizes correctly"
    );
}

/// Test that size calculation matches Core's TX_NO_WITNESS behavior
///
/// Core uses GetSerializeSize(TX_NO_WITNESS(tx)), which serializes
/// the transaction without witness data. Our serialize_transaction()
/// should match this exactly.
#[test]
fn test_transaction_size_no_witness() {
    // Create a transaction that could have witness data
    // (but we're testing non-witness serialization)
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
        ].into(),
        outputs: vec![
            TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51, 0x52].into(),
            },
            TransactionOutput {
                value: 2000,
                script_pubkey: vec![0x53, 0x54, 0x55].into(),
            },
        ].into(),
        lock_time: 12345,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    
    // The serialized transaction should NOT include witness data
    // (our serialize_transaction doesn't include witness)
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must match non-witness serialization (TX_NO_WITNESS)"
    );
}

/// Test transaction size with coinbase transaction
#[test]
fn test_transaction_size_coinbase() {
    // Coinbase transaction has special input format
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00], // Height encoding (5 bytes)
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000, // 50 BTC
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 0,
    };

    let calculated = calculate_transaction_size(&tx);
    let serialized = serialize_transaction(&tx);
    assert_eq!(
        calculated, serialized.len(),
        "Size calculation must handle coinbase transactions correctly"
    );
}

/// Test that size calculation is deterministic
#[test]
fn test_transaction_size_deterministic() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51, 0x52, 0x53],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51, 0x52].into(),
        }].into(),
        lock_time: 0,
    };

    let size1 = calculate_transaction_size(&tx);
    let size2 = calculate_transaction_size(&tx);
    let size3 = calculate_transaction_size(&tx);

    assert_eq!(size1, size2, "Size calculation must be deterministic");
    assert_eq!(size2, size3, "Size calculation must be deterministic");
}

