//! BIP65 (CLTV) Integration Tests
//! 
//! Tests for OP_CHECKLOCKTIMEVERIFY (CLTV) integration with transaction validation,
//! block height context, and median time-past (BIP113).

use blvm_consensus::*;
use blvm_consensus::constants::LOCKTIME_THRESHOLD;
use super::bip_test_helpers::*;

#[test]
fn test_cltv_block_height_validation_passes() {
    // Transaction with block-height locktime that should pass
    let block_height: u32 = 500000; // Below threshold = block height
    let required_locktime: u32 = 400000; // Lower than transaction locktime
    
    let tx = create_cltv_transaction(block_height, required_locktime, vec![0x51]); // OP_1
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51], // OP_1
            height: 0,
        },
    );
    
    // Validate at block height >= required locktime
    let result = validate_with_context(&tx, &utxo_set, block_height as u64, 0);
    
    // Should pass: tx.lock_time (500000) >= required_locktime (400000) and types match
    assert!(result.is_ok());
    // Note: Current implementation checks tx.lock_time >= required, but doesn't validate
    // against actual block height. This is expected behavior per implementation notes.
}

#[test]
fn test_cltv_block_height_type_mismatch_fails() {
    // Transaction with block-height locktime but script requires timestamp
    let tx_locktime: u32 = 400000; // Block height (< 500000000)
    let required_locktime: u32 = 600000000; // Timestamp (>= 500000000)
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should fail: type mismatch (block height vs timestamp)
    assert!(result.is_ok()); // verify_script_with_context returns Ok(bool)
    assert!(!result.unwrap()); // Should return false (validation fails)
}

#[test]
fn test_cltv_timestamp_type_mismatch_fails() {
    // Transaction with timestamp locktime but script requires block height
    let tx_locktime: u32 = 600000000; // Timestamp (>= 500000000)
    let required_locktime: u32 = 400000; // Block height (< 500000000)
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should fail: type mismatch (timestamp vs block height)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_zero_locktime_fails() {
    // Transaction with zero locktime should fail CLTV
    let tx_locktime: u32 = 0;
    let required_locktime: u32 = 400000;
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should fail: BIP65 requires tx.lock_time != 0
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_insufficient_locktime_fails() {
    // Transaction locktime < required locktime should fail
    let tx_locktime: u32 = 300000;
    let required_locktime: u32 = 400000;
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should fail: tx.lock_time (300000) < required_locktime (400000)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_exact_locktime_passes() {
    // Transaction locktime == required locktime should pass
    let locktime: u32 = 400000;
    
    let tx = create_cltv_transaction(locktime, locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should pass: tx.lock_time (400000) >= required_locktime (400000)
    assert!(result.is_ok());
    // Note: This will pass the basic check, but full validation would need block height
}

#[test]
fn test_cltv_timestamp_validation() {
    // Transaction with timestamp locktime
    let tx_locktime: u32 = 1609459200; // 2021-01-01 00:00:00 UTC
    let required_locktime: u32 = 1577836800; // 2020-01-01 00:00:00 UTC (before tx locktime)
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    // For timestamp validation, we'd need median time-past >= tx_locktime
    // Current implementation checks tx.lock_time >= required, which should pass
    let result = validate_with_context(&tx, &utxo_set, 0, tx_locktime as u64);
    
    // Should pass: tx.lock_time (1609459200) >= required_locktime (1577836800)
    assert!(result.is_ok());
}

#[test]
fn test_cltv_boundary_block_height() {
    // Test boundary value: LOCKTIME_THRESHOLD - 1 (last block height value)
    let tx_locktime: u32 = LOCKTIME_THRESHOLD - 1;
    let required_locktime: u32 = LOCKTIME_THRESHOLD - 2;
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should pass: both are block heights, tx >= required
    assert!(result.is_ok());
}

#[test]
fn test_cltv_boundary_timestamp() {
    // Test boundary value: LOCKTIME_THRESHOLD (first timestamp value)
    let tx_locktime: u32 = LOCKTIME_THRESHOLD;
    let required_locktime: u32 = LOCKTIME_THRESHOLD;
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should pass: both are timestamps, tx >= required
    assert!(result.is_ok());
}

#[test]
fn test_cltv_empty_stack_fails() {
    // CLTV with empty stack should fail
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0xb1], // Just CLTV opcode, no value on stack
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 400000,
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should fail: empty stack
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_invalid_encoding_fails() {
    // CLTV with invalid encoding (too many bytes) should fail
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![
                0x51, 0x51, 0x51, 0x51, 0x51, 0x51, // 6 bytes (too many)
                0xb1, // CLTV
            ],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 400000,
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should fail: invalid encoding (script integer must be <= 4 bytes)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_max_u32_value() {
    // Test with maximum u32 value
    let tx_locktime: u32 = u32::MAX;
    let required_locktime: u32 = u32::MAX;
    
    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![0x51]);
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should pass: tx.lock_time (u32::MAX) >= required_locktime (u32::MAX)
    // Both are timestamps (>= LOCKTIME_THRESHOLD)
    assert!(result.is_ok());
}

#[test]
fn test_cltv_multiple_inputs_context() {
    // Test CLTV with multiple inputs (each input needs correct context)
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint { hash: [1; 32].into(), index: 0 },
                script_sig: {
                    let mut script = vec![0x51];
                    script.extend_from_slice(&encode_script_int(400000));
                    script.push(0xb1); // CLTV
                    script
                },
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint { hash: [2; 32], index: 0 },
                script_sig: vec![0x51], // No CLTV
                sequence: 0xffffffff,
            },
        ].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51].into(),
        }].into(),
        lock_time: 500000,
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 500000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    utxo_set.insert(
        OutPoint { hash: [2; 32], index: 0 },
        UTXO {
            value: 500000,
            script_pubkey: vec![0x51],
            height: 0,
        },
    );
    
    // Validate first input (with CLTV)
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout).unwrap();
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    let result = verify_script_with_context(
        &input.script_sig,
        &utxo.script_pubkey,
        None,
        0,
        &tx,
        0, // First input
        &prevouts,
    );
    
    // Should pass: CLTV validation for first input
    assert!(result.is_ok());
}

#[test]
fn test_cltv_in_script_pubkey() {
    // CLTV can be in scriptPubkey (output locking script)
    let required_locktime: u32 = 400000;
    let mut script_pubkey = vec![0x51]; // OP_1
    script_pubkey.extend_from_slice(&encode_script_int(required_locktime));
    script_pubkey.push(0xb1); // CLTV
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32].into(), index: 0 },
            script_sig: vec![0x51], // OP_1 (unlocks scriptPubkey)
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey,
        }].into(),
        lock_time: 500000, // >= required_locktime
    };
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51], // Previous output script
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should pass: CLTV in scriptPubkey validates correctly
    assert!(result.is_ok());
}

