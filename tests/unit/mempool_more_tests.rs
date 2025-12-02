use bllvm_consensus::mempool;
use bllvm_consensus::{Transaction, TransactionInput, TransactionOutput};
use bllvm_consensus::config::MempoolConfig;

#[path = "../test_helpers.rs"]
mod test_helpers;
use test_helpers::create_test_utxo;

#[test]
fn test_negative_fee_rejected() {
    let (set, prev) = create_test_utxo(1000);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput { prevout: prev, script_sig: vec![0x51].into(), sequence: 0xffffffff }].into(),
        outputs: vec![TransactionOutput { value: 2000, script_pubkey: vec![0x51].into() }].into(),
        lock_time: 0,
    };
    let pool = mempool::Mempool::new();
    let res = mempool::accept_to_memory_pool(&tx, None, &set, &pool, 1);
    assert!(res.is_err(), "Outputs exceed inputs should be rejected");
}

#[test]
fn test_non_standard_script_flagged() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput { value: 0, script_pubkey: vec![0x6a].into() }].into(), // OP_RETURN only
        lock_time: 0,
    };
    // Whether standard depends on policy; just exercise the path
    let _ = mempool::is_standard_tx(&tx);
}

// OP_RETURN Size Limit Tests

#[test]
fn test_op_return_within_size_limit() {
    // OP_RETURN (0x6a) + push opcode (0x4c = OP_PUSHDATA1) + 80 bytes data
    let mut script = vec![0x6a, 0x4c, 80];
    script.extend(vec![0x00; 80]);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(mempool::is_standard_tx(&tx).unwrap(), "OP_RETURN with 80 bytes should be valid");
}

#[test]
fn test_op_return_exceeds_size_limit() {
    // OP_RETURN (0x6a) + push opcode (0x4c) + 81 bytes data (exceeds 80 byte limit)
    let mut script = vec![0x6a, 0x4c, 81];
    script.extend(vec![0x00; 81]);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(!mempool::is_standard_tx(&tx).unwrap(), "OP_RETURN exceeding 80 bytes should be rejected");
}

#[test]
fn test_multiple_op_return_rejected() {
    // Two OP_RETURN outputs
    let op_return_script = vec![0x6a, 0x01, 0x00]; // OP_RETURN + push 1 byte + 1 byte data
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![
            TransactionOutput {
                value: 0,
                script_pubkey: op_return_script.clone().into(),
            },
            TransactionOutput {
                value: 0,
                script_pubkey: op_return_script.into(),
            },
        ].into(),
        lock_time: 0,
    };
    
    assert!(!mempool::is_standard_tx(&tx).unwrap(), "Multiple OP_RETURN outputs should be rejected");
}

#[test]
fn test_op_return_config_custom_limit() {
    let mut config = MempoolConfig::default();
    config.max_op_return_size = 40; // Stricter limit
    
    // OP_RETURN with 41 bytes (should be rejected with custom limit)
    let mut script = vec![0x6a, 0x4c, 41];
    script.extend(vec![0x00; 41]);
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(!mempool::is_standard_tx_with_config(&tx, Some(&config)).unwrap(),
        "OP_RETURN exceeding custom limit should be rejected");
}

#[test]
fn test_op_return_config_allow_multiple() {
    let mut config = MempoolConfig::default();
    config.reject_multiple_op_return = false; // Allow multiple
    
    let op_return_script = vec![0x6a, 0x01, 0x00];
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![
            TransactionOutput {
                value: 0,
                script_pubkey: op_return_script.clone().into(),
            },
            TransactionOutput {
                value: 0,
                script_pubkey: op_return_script.into(),
            },
        ].into(),
        lock_time: 0,
    };
    
    assert!(mempool::is_standard_tx_with_config(&tx, Some(&config)).unwrap(),
        "Multiple OP_RETURN should be allowed when config permits");
}

// Envelope Protocol Tests

#[test]
fn test_envelope_protocol_rejected() {
    // OP_FALSE (0x00) OP_IF (0x63) - envelope protocol used by Ordinals
    let script = vec![0x00, 0x63, 0x01, 0x00, 0x68]; // OP_FALSE OP_IF ... OP_ENDIF
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(!mempool::is_standard_tx(&tx).unwrap(), "Envelope protocol should be rejected");
}

#[test]
fn test_envelope_protocol_config_allow() {
    let mut config = MempoolConfig::default();
    config.reject_envelope_protocol = false; // Allow envelope protocol
    
    let script = vec![0x00, 0x63, 0x01, 0x00, 0x68];
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(mempool::is_standard_tx_with_config(&tx, Some(&config)).unwrap(),
        "Envelope protocol should be allowed when config permits");
}

// Large Script Size Tests

#[test]
fn test_large_script_rejected() {
    let mut config = MempoolConfig::default();
    config.max_standard_script_size = 200;
    
    // Script larger than 200 bytes
    let script = vec![0x51; 201]; // 201 bytes of OP_1
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(!mempool::is_standard_tx_with_config(&tx, Some(&config)).unwrap(),
        "Scripts larger than max_standard_script_size should be rejected");
}

#[test]
fn test_script_at_limit_accepted() {
    let mut config = MempoolConfig::default();
    config.max_standard_script_size = 200;
    
    // Script exactly at limit
    let script = vec![0x51; 200]; // 200 bytes of OP_1
    
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }].into(),
        lock_time: 0,
    };
    
    assert!(mempool::is_standard_tx_with_config(&tx, Some(&config)).unwrap(),
        "Scripts at max_standard_script_size limit should be accepted");
}



























