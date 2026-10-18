//! BIP65 (CLTV) Integration Tests
//!
//! Tests for OP_CHECKLOCKTIMEVERIFY (CLTV) integration with transaction validation,
//! block height context, and median time-past (BIP113).

use super::bip_test_helpers::*;
use blvm_consensus::constants::LOCKTIME_THRESHOLD;
use blvm_consensus::opcodes::*;
use blvm_consensus::script::verify_script_with_context;
use blvm_consensus::*;

#[test]
fn test_cltv_block_height_validation_passes() {
    // Transaction with block-height locktime that should pass
    let block_height: u32 = 500000; // Below threshold = block height
    let required_locktime: u32 = 400000; // Lower than transaction locktime

    let tx = create_cltv_transaction(block_height, required_locktime, vec![OP_1]); // OP_1

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(), // OP_1
            is_coinbase: false,
            height: 0,
        }),
    );

    // Validate at block height >= required locktime
    let result = validate_with_context(&tx, &utxo_set, block_height as u64, 0, 0x200);

    assert!(
        result.expect("cltv"),
        "tx locktime 500000 satisfies the 400000 push"
    );
}

#[test]
fn test_cltv_block_height_type_mismatch_fails() {
    // Transaction with block-height locktime but script requires timestamp
    let tx_locktime: u32 = 400000; // Block height (< 500000000)
    let required_locktime: u32 = 600000000; // Timestamp (>= 500000000)

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    // Should fail: type mismatch (block height vs timestamp)
    assert!(result.is_ok()); // verify_script_with_context returns Ok(bool)
    assert!(!result.unwrap()); // Should return false (validation fails)
}

#[test]
fn test_cltv_timestamp_type_mismatch_fails() {
    // Transaction with timestamp locktime but script requires block height
    let tx_locktime: u32 = 600000000; // Timestamp (>= 500000000)
    let required_locktime: u32 = 400000; // Block height (< 500000000)

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    // Should fail: type mismatch (timestamp vs block height)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_zero_locktime_fails() {
    // Transaction with zero locktime should fail CLTV
    let tx_locktime: u32 = 0;
    let required_locktime: u32 = 400000;

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    // Should fail: BIP65 requires tx.lock_time != 0
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_insufficient_locktime_fails() {
    // Transaction locktime < required locktime should fail
    let tx_locktime: u32 = 300000;
    let required_locktime: u32 = 400000;

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    // Should fail: tx.lock_time (300000) < required_locktime (400000)
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_cltv_exact_locktime_passes() {
    // Transaction locktime == required locktime should pass
    let locktime: u32 = 400000;

    let tx = create_cltv_transaction(locktime, locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    assert!(result.expect("cltv"), "equal locktimes satisfy CLTV");
}

#[test]
fn test_cltv_timestamp_validation() {
    // Transaction with timestamp locktime
    let tx_locktime: u32 = 1609459200; // 2021-01-01 00:00:00 UTC
    let required_locktime: u32 = 1577836800; // 2020-01-01 00:00:00 UTC (before tx locktime)

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    // For timestamp validation, we'd need median time-past >= tx_locktime
    // Current implementation checks tx.lock_time >= required, which should pass
    let result = validate_with_context(&tx, &utxo_set, 0, tx_locktime as u64, 0x200);

    assert!(
        result.expect("cltv"),
        "a later timestamp satisfies the earlier push"
    );
}

#[test]
fn test_cltv_boundary_block_height() {
    // Test boundary value: LOCKTIME_THRESHOLD - 1 (last block height value)
    let tx_locktime: u32 = LOCKTIME_THRESHOLD - 1;
    let required_locktime: u32 = LOCKTIME_THRESHOLD - 2;

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    assert!(
        result.expect("cltv"),
        "the last block-height locktime satisfies the push below it"
    );
}

#[test]
fn test_cltv_boundary_timestamp() {
    // Test boundary value: LOCKTIME_THRESHOLD (first timestamp value)
    let tx_locktime: u32 = LOCKTIME_THRESHOLD;
    let required_locktime: u32 = LOCKTIME_THRESHOLD;

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    assert!(
        result.expect("cltv"),
        "equal timestamp locktimes satisfy CLTV"
    );
}

#[test]
fn test_cltv_empty_stack_fails() {
    // CLTV with empty stack should fail
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![OP_CHECKLOCKTIMEVERIFY], // Just CLTV opcode, no value on stack
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 400000,
    };

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    match result {
        Err(blvm_consensus::error::ConsensusError::ScriptErrorWithCode { code, .. })
            if code == blvm_consensus::error::ScriptErrorCode::InvalidStackOperation => {}
        other => panic!("CLTV on an empty stack must be InvalidStackOperation, got {other:?}"),
    }
}

#[test]
fn test_cltv_invalid_encoding_fails() {
    // CLTV with invalid encoding (too many bytes) should fail
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![
                0x06,
                0x01,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00, // six-byte number
                OP_CHECKLOCKTIMEVERIFY,
            ],
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 400000,
    };

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    match result {
        Err(blvm_consensus::error::ConsensusError::ScriptErrorWithCode { code, .. })
            if code == blvm_consensus::error::ScriptErrorCode::MinimalData => {}
        other => panic!("a six-byte CLTV operand must be MinimalData, got {other:?}"),
    }
}

#[test]
fn test_cltv_max_u32_value() {
    // Test with maximum u32 value
    let tx_locktime: u32 = u32::MAX;
    let required_locktime: u32 = u32::MAX;

    let tx = create_cltv_transaction(tx_locktime, required_locktime, vec![OP_1]);

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);

    assert!(
        result.expect("cltv"),
        "equal u32::MAX timestamp locktimes satisfy CLTV"
    );
}

#[test]
fn test_cltv_multiple_inputs_context() {
    // Test CLTV with multiple inputs (each input needs correct context)
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: {
                    let mut script = vec![OP_1];
                    script.extend_from_slice(&encode_script_int(400000));
                    script.push(OP_CHECKLOCKTIMEVERIFY); // CLTV
                    script
                },
                sequence: 0,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32],
                    index: 0,
                },
                script_sig: vec![OP_1], // No CLTV
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 500000,
    };

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 500000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );
    utxo_set.insert(
        OutPoint {
            hash: [2; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 500000,
            script_pubkey: vec![OP_1].into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    // Validate first input (with CLTV)
    let utxo1 = utxo_set.get(&tx.inputs[0].prevout).unwrap();
    let utxo2 = utxo_set.get(&tx.inputs[1].prevout).unwrap();
    let prevouts = vec![
        TransactionOutput {
            value: utxo1.value,
            script_pubkey: utxo1.script_pubkey.to_vec(),
        },
        TransactionOutput {
            value: utxo2.value,
            script_pubkey: utxo2.script_pubkey.to_vec(),
        },
    ];

    let result = verify_script_with_context(
        &tx.inputs[0].script_sig,
        &utxo1.script_pubkey,
        None,
        0x200,
        &tx,
        0,
        &prevouts,
        Some(500_000u64),
        Network::Regtest,
    );

    assert!(
        result.expect("cltv input 0"),
        "input 0 locktime 500000 satisfies the 400000 push"
    );
    let second = verify_script_with_context(
        &tx.inputs[1].script_sig,
        &utxo2.script_pubkey,
        None,
        0x200,
        &tx,
        1,
        &prevouts,
        Some(500_000u64),
        Network::Regtest,
    );
    assert!(second.expect("input 1"), "input 1 is OP_1 spending OP_1");
}

#[test]
fn test_cltv_in_script_pubkey() {
    let required_locktime: u32 = 400000;
    let mut script_pubkey = encode_script_int(required_locktime);
    script_pubkey.push(OP_CHECKLOCKTIMEVERIFY);
    script_pubkey.push(OP_1);

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![],
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 500000,
    };

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: [1; 32],
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 1000000,
            script_pubkey: script_pubkey.into(),
            is_coinbase: false,
            height: 0,
        }),
    );

    let result = validate_with_context(&tx, &utxo_set, 0, 0, 0x200);
    assert!(
        result.expect("cltv scriptPubKey"),
        "the spent scriptPubKey accepts locktime 500000"
    );
}
