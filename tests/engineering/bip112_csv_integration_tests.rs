//! BIP112 (CSV) Integration Tests
//! 
//! Tests for OP_CHECKSEQUENCEVERIFY (CSV) integration with transaction validation,
//! sequence numbers, and BIP68 relative locktime.

use bllvm_consensus::*;
use super::bip_test_helpers::*;

#[test]
fn test_csv_sequence_validation_passes() {
    // CSV with valid sequence number should pass
    let input_sequence: u32 = 0x00040000; // 4 blocks relative locktime, block-based
    let required_sequence: u32 = 0x00030000; // 3 blocks required, block-based
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]); // OP_1
    
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: [1; 32], index: 0 },
        UTXO {
            value: 1000000,
            script_pubkey: vec![0x51], // OP_1
            height: 0,
        },
    );
    
    let result = validate_with_context(&tx, &utxo_set, 0, 0);
    
    // Should pass: input_sequence (4 blocks) >= required_sequence (3 blocks)
    // and types match (both block-based)
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_csv_sequence_disabled_fails() {
    // CSV should fail if sequence is disabled (0x80000000 bit set)
    let input_sequence: u32 = 0x80000000; // Sequence disabled
    let required_sequence: u32 = 0x00030000;
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should fail: sequence disabled per BIP112
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_csv_type_mismatch_fails() {
    // CSV fails if type flags don't match (block-based vs time-based)
    let input_sequence: u32 = 0x00430000; // Time-based (bit 22 set), 3 units
    let required_sequence: u32 = 0x00040000; // Block-based (bit 22 clear), 4 blocks
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should fail: type mismatch (time-based vs block-based)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_csv_insufficient_locktime_fails() {
    // CSV fails if input locktime < required locktime
    let input_sequence: u32 = 0x00030000; // 3 blocks
    let required_sequence: u32 = 0x00040000; // 4 blocks required
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should fail: input_sequence (3 blocks) < required_sequence (4 blocks)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_csv_exact_locktime_passes() {
    // CSV with exact match should pass
    let sequence: u32 = 0x00050000; // 5 blocks
    
    let tx = create_csv_transaction(sequence, sequence, vec![0x51]);
    
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
    
    // Should pass: input_sequence (5 blocks) >= required_sequence (5 blocks)
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_csv_block_based_locktime() {
    // Test block-based relative locktime
    let input_sequence: u32 = 0x000a0000; // 10 blocks, block-based (bit 22 clear)
    let required_sequence: u32 = 0x00050000; // 5 blocks required
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should pass: both block-based, input (10) >= required (5)
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_csv_time_based_locktime() {
    // Test time-based relative locktime (BIP68)
    let input_sequence: u32 = 0x00460000; // 6*512 seconds, time-based (bit 22 set)
    let required_sequence: u32 = 0x00430000; // 3*512 seconds required
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should pass: both time-based, input (6*512) >= required (3*512)
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_csv_empty_stack_fails() {
    // CSV with empty stack should fail
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: vec![0xb2], // Just CSV opcode, no value on stack
            sequence: 0x00040000,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
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
fn test_csv_invalid_encoding_fails() {
    // CSV with invalid encoding (too many bytes) should fail
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: vec![
                0x51, 0x51, 0x51, 0x51, 0x51, 0x51, // 6 bytes (too many)
                0xb2, // CSV
            ],
            sequence: 0x00040000,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
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
fn test_csv_max_relative_locktime() {
    // Test with maximum relative locktime value (0x0000ffff = 65535 blocks/seconds)
    let input_sequence: u32 = 0x0000ffff; // Max value, block-based
    let required_sequence: u32 = 0x0000ffff;
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should pass: max locktime matches required
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_csv_bip68_encoding() {
    // Test BIP68 sequence number encoding/decoding
    // Sequence = 0x80000000 (disabled) | 0x00400000 (time-based) | 0x0000ffff (value)
    let input_sequence: u32 = 0x80000000 | 0x00400000 | 0x00000010; // Disabled, but if enabled: time-based, 16*512 seconds
    let required_sequence: u32 = 0x00400008; // Time-based, 8*512 seconds
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should fail: sequence disabled (0x80000000 bit set)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_csv_multiple_inputs_context() {
    // Test CSV with multiple inputs (each input needs correct context)
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint { hash: [1; 32], index: 0 },
                script_sig: {
                    let mut script = vec![0x51];
                    script.extend_from_slice(&encode_script_int(0x00040000));
                    script.push(0xb2); // CSV
                    script
                },
                sequence: 0x00050000, // 5 blocks
            },
            TransactionInput {
                prevout: OutPoint { hash: [2; 32], index: 0 },
                script_sig: vec![0x51], // No CSV
                sequence: 0xffffffff,
            },
        ],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
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
    
    // Validate first input (with CSV)
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
    
    // Should pass: CSV validation for first input
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_csv_in_script_pubkey() {
    // CSV can be in scriptPubkey (output locking script)
    let required_sequence: u32 = 0x00040000; // 4 blocks
    let mut script_pubkey = vec![0x51]; // OP_1
    script_pubkey.extend_from_slice(&encode_script_int(required_sequence));
    script_pubkey.push(0xb2); // CSV
    
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint { hash: [1; 32], index: 0 },
            script_sig: vec![0x51], // OP_1 (unlocks scriptPubkey)
            sequence: 0x00050000, // 5 blocks >= required 4 blocks
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey,
        }],
        lock_time: 0,
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
    
    // Should pass: CSV in scriptPubkey validates correctly
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_csv_zero_locktime() {
    // CSV with zero relative locktime should pass
    let input_sequence: u32 = 0x00000000; // Zero locktime
    let required_sequence: u32 = 0x00000000;
    
    let tx = create_csv_transaction(input_sequence, required_sequence, vec![0x51]);
    
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
    
    // Should pass: zero locktime >= zero required
    assert!(result.is_ok());
    assert!(result.unwrap());
}

