use super::*;

// ============================================================================
// TEST BUILDERS
// ============================================================================

fn make_input(hash: [u8; 32], index: u32) -> TransactionInput {
    TransactionInput {
        prevout: OutPoint { hash, index },
        script_sig: vec![],
        sequence: 0xffffffff,
    }
}

fn make_output(value: i64) -> TransactionOutput {
    TransactionOutput {
        value,
        script_pubkey: vec![],
    }
}

/// Minimal valid non-coinbase transaction with one input (hash=0, index=0).
fn make_simple_tx(value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![make_input([0; 32], 0)].into(),
        outputs: vec![make_output(value)].into(),
        lock_time: 0,
    }
}

/// N unique inputs with distinct hashes (hash bytes 0-3 encode `i` as little-endian u32).
fn make_n_inputs(n: usize) -> Vec<TransactionInput> {
    (0..n)
        .map(|i| {
            let mut hash = [0u8; 32];
            hash[..4].copy_from_slice(&(i as u32).to_le_bytes());
            make_input(hash, i as u32)
        })
        .collect()
}

/// N outputs each with the given value.
fn make_n_outputs(n: usize, value: i64) -> Vec<TransactionOutput> {
    (0..n).map(|_| make_output(value)).collect()
}

/// N inputs with large `script_sig` bytes (for size-limit tests).
fn make_n_large_inputs(n: usize, script_size: usize) -> Vec<TransactionInput> {
    (0..n)
        .map(|i| TransactionInput {
            prevout: OutPoint {
                hash: {
                    let mut h = [0u8; 32];
                    h[..4].copy_from_slice(&(i as u32).to_le_bytes());
                    h
                },
                index: 0,
            },
            script_sig: vec![0u8; script_size],
            sequence: 0xffffffff,
        })
        .collect()
}

/// Coinbase transaction (null prevout, index=0xffffffff).
fn make_coinbase_tx(value: i64) -> Transaction {
    Transaction {
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
        outputs: vec![make_output(value)].into(),
        lock_time: 0,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[test]
fn test_check_transaction_valid() {
    assert_eq!(
        check_transaction(&make_simple_tx(1000)).unwrap(),
        ValidationResult::Valid
    );
}

#[test]
fn test_check_transaction_empty_inputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![make_output(1000)].into(),
        lock_time: 0,
    };
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

#[test]
fn test_check_tx_inputs_coinbase() {
    let utxo_set = UtxoSet::default();
    let (result, fee) = check_tx_inputs(&make_coinbase_tx(5_000_000_000), &utxo_set, 0).unwrap();
    assert_eq!(result, ValidationResult::Valid);
    assert_eq!(fee, 0);
}

// ============================================================================
// COMPREHENSIVE TRANSACTION TESTS
// ============================================================================

#[test]
fn test_check_transaction_empty_outputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([0; 32], 0)].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

#[test]
fn negative_money_is_invalid_without_aborting() {
    let prevout = OutPoint {
        hash: [1u8; 32],
        index: 0,
    };
    let tx_with = |output_value: i64| Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            script_sig: vec![],
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: output_value,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };
    let utxo = |value: i64| {
        let mut set = UtxoSet::default();
        set.insert(
            prevout,
            std::sync::Arc::new(UTXO {
                value,
                script_pubkey: vec![].into(),
                height: 0,
                is_coinbase: false,
            }),
        );
        set
    };

    let negative_output = tx_with(-1);
    let (owned, _) =
        check_tx_inputs_with_owned_data(&negative_output, 0, &[Some((1_000, false, 0))]).unwrap();
    assert!(matches!(owned, ValidationResult::Invalid(_)));
    let (overlay, _) = check_tx_inputs(&negative_output, &utxo(1_000), 0).unwrap();
    assert!(matches!(overlay, ValidationResult::Invalid(_)));

    let spend = tx_with(0);
    let (negative_utxo, _) = check_tx_inputs(&spend, &utxo(-1), 0).unwrap();
    assert!(matches!(negative_utxo, ValidationResult::Invalid(_)));
    let (owned_utxo, _) =
        check_tx_inputs_with_owned_data(&spend, 0, &[Some((-1, false, 0))]).unwrap();
    assert!(matches!(owned_utxo, ValidationResult::Invalid(_)));
    let (huge_utxo, _) = check_tx_inputs(&spend, &utxo(MAX_MONEY + 1), 0).unwrap();
    assert!(matches!(huge_utxo, ValidationResult::Invalid(_)));

    let (ok, fee) =
        check_tx_inputs_with_owned_data(&tx_with(400), 0, &[Some((1_000, false, 0))]).unwrap();
    assert_eq!(ok, ValidationResult::Valid);
    assert_eq!(fee, 600);

    let over = Transaction {
        version: 1,
        inputs: spend.inputs.clone(),
        outputs: vec![
            TransactionOutput {
                value: MAX_MONEY,
                script_pubkey: vec![],
            },
            TransactionOutput {
                value: MAX_MONEY,
                script_pubkey: vec![],
            },
        ]
        .into(),
        lock_time: 0,
    };
    let (sum, _) = check_tx_inputs(&over, &utxo(MAX_MONEY), 0).unwrap();
    assert!(matches!(sum, ValidationResult::Invalid(_)));
    let (sum_owned, _) =
        check_tx_inputs_with_owned_data(&over, 0, &[Some((MAX_MONEY, false, 0))]).unwrap();
    assert!(matches!(sum_owned, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_invalid_output_value_negative() {
    assert!(matches!(
        check_transaction(&make_simple_tx(-1)).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

#[test]
fn test_check_transaction_invalid_output_value_too_large() {
    assert!(matches!(
        check_transaction(&make_simple_tx(MAX_MONEY + 1)).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

#[test]
fn test_check_transaction_max_output_value() {
    assert_eq!(
        check_transaction(&make_simple_tx(MAX_MONEY)).unwrap(),
        ValidationResult::Valid
    );
}

#[test]
fn test_check_transaction_too_many_inputs() {
    // MAX_INPUTS + 1 inputs → must be rejected
    let tx = Transaction {
        version: 1,
        inputs: make_n_inputs(MAX_INPUTS + 1).into(),
        outputs: vec![make_output(1000)].into(),
        lock_time: 0,
    };
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

#[test]
fn test_check_transaction_max_inputs() {
    // 20_000 unique inputs — fits within the 1 MB stripped-size limit
    // (each ≈ 41 bytes; 20_000 × 41 ≈ 820 kB < 1 MB)
    let tx = Transaction {
        version: 1,
        inputs: make_n_inputs(20_000).into(),
        outputs: vec![make_output(1000)].into(),
        lock_time: 0,
    };
    assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
}

#[test]
fn test_check_transaction_too_many_outputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([0; 32], 0)].into(),
        outputs: make_n_outputs(MAX_OUTPUTS + 1, 1000).into(),
        lock_time: 0,
    };
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

#[test]
fn test_check_transaction_max_outputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([0; 32], 0)].into(),
        outputs: make_n_outputs(MAX_OUTPUTS, 1000).into(),
        lock_time: 0,
    };
    assert_eq!(check_transaction(&tx).unwrap(), ValidationResult::Valid);
}

#[test]
fn test_check_transaction_too_large() {
    // MAX_INPUTS inputs each with a 1 kB script_sig pushes the stripped size
    // past 1 MB (MAX_BLOCK_WEIGHT / WITNESS_SCALE_FACTOR).
    let tx = Transaction {
        version: 1,
        inputs: make_n_large_inputs(MAX_INPUTS, 1000).into(),
        outputs: vec![make_output(1000)].into(),
        lock_time: 0,
    };
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

fn make_utxo_set(entries: &[([u8; 32], u32, i64)]) -> UtxoSet {
    let mut utxo_set = UtxoSet::default();
    for &(hash, index, value) in entries {
        utxo_set.insert(
            OutPoint { hash, index },
            std::sync::Arc::new(UTXO {
                value,
                script_pubkey: vec![].into(),
                height: 0,
                is_coinbase: false,
            }),
        );
    }
    utxo_set
}

#[test]
fn test_check_tx_inputs_regular_transaction() {
    let utxo_set = make_utxo_set(&[([1; 32], 0, 1_000_000_000)]);
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([1; 32], 0)].into(),
        outputs: vec![make_output(900_000_000)].into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
    assert_eq!(result, ValidationResult::Valid);
    assert_eq!(fee, 100_000_000);
}

#[test]
fn test_check_tx_inputs_missing_utxo() {
    let utxo_set = UtxoSet::default();
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([1; 32], 0)].into(),
        outputs: vec![make_output(100_000_000)].into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_insufficient_funds() {
    let utxo_set = make_utxo_set(&[([1; 32], 0, 100_000_000)]);
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([1; 32], 0)].into(),
        outputs: vec![make_output(200_000_000)].into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_multiple_inputs() {
    let utxo_set = make_utxo_set(&[([1; 32], 0, 500_000_000), ([2; 32], 0, 300_000_000)]);
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([1; 32], 0), make_input([2; 32], 0)].into(),
        outputs: vec![make_output(700_000_000)].into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
    assert_eq!(result, ValidationResult::Valid);
    assert_eq!(fee, 100_000_000); // 8 BTC in - 7 BTC out
}

#[test]
fn test_check_tx_inputs_input_sum_exceeds_max_money() {
    let utxo_set = make_utxo_set(&[([1; 32], 0, MAX_MONEY), ([2; 32], 0, MAX_MONEY)]);
    let tx = Transaction {
        version: 1,
        inputs: vec![make_input([1; 32], 0), make_input([2; 32], 0)].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx, &utxo_set, 0).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
    assert_eq!(fee, 0);

    let owned = [
        Some((MAX_MONEY, false, 0u64)),
        Some((MAX_MONEY, false, 0u64)),
    ];
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, 0, &owned).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
    assert_eq!(fee, 0);

    // The sum may sit on the cap. One input of MAX_MONEY spends cleanly.
    let at_cap = make_utxo_set(&[([1; 32], 0, MAX_MONEY)]);
    let tx_at_cap = Transaction {
        version: 1,
        inputs: vec![make_input([1; 32], 0)].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx_at_cap, &at_cap, 0).unwrap();
    assert_eq!(result, ValidationResult::Valid);
    assert_eq!(fee, MAX_MONEY);
}

fn bare_tx(inputs: Vec<TransactionInput>) -> Transaction {
    Transaction {
        version: 1,
        inputs: inputs.into(),
        outputs: vec![].into(),
        lock_time: 0,
    }
}

#[test]
fn test_is_coinbase_edge_cases() {
    assert!(is_coinbase(&bare_tx(vec![make_input([0; 32], 0xffffffff)])));
    assert!(!is_coinbase(&bare_tx(vec![make_input(
        [1; 32], 0xffffffff
    )]))); // wrong hash
    assert!(!is_coinbase(&bare_tx(vec![make_input([0; 32], 0)]))); // wrong index
    assert!(!is_coinbase(&bare_tx(vec![
        make_input([0; 32], 0xffffffff),
        make_input([1; 32], 0),
    ]))); // multiple inputs
    assert!(!is_coinbase(&bare_tx(vec![]))); // no inputs
}

#[test]
fn test_calculate_transaction_size() {
    // 4 (version) + 1 (input_count) +
    // 2 × (32 + 4 + 1 + 3 + 4)  [hash + index + script_len varint + script + sequence] +
    // 1 (output_count) +
    // 2 × (8 + 1 + 3)            [value + script_len varint + script] +
    // 4 (lock_time) = 122
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![1, 2, 3],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 1,
                },
                script_sig: vec![4, 5, 6],
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 1000,
                script_pubkey: vec![7, 8, 9],
            },
            TransactionOutput {
                value: 2000,
                script_pubkey: vec![10, 11, 12],
            },
        ]
        .into(),
        lock_time: 12345,
    };
    assert_eq!(calculate_transaction_size(&tx), 122);
}

fn coinbase_with_script(script_sig: Vec<u8>, outputs: Vec<TransactionOutput>) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig,
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: outputs.into(),
        lock_time: 0,
    }
}

#[test]
fn test_output_sum_just_below_overflow_is_invalid() {
    let tx = coinbase_with_script(vec![0x51, 0xff], make_n_outputs(4_392, MAX_MONEY));
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

#[test]
fn test_output_sum_overflow_is_invalid() {
    let tx = coinbase_with_script(vec![0x51, 0xff], make_n_outputs(4_393, MAX_MONEY));
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(ref reason) if reason.contains("overflow")
    ));
}

#[test]
fn test_output_sum_overflow_input_path_is_invalid() {
    let prevout = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            script_sig: vec![0x51, 0x51],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: make_n_outputs(4_393, MAX_MONEY).into(),
        lock_time: 0,
    };
    let mut set = UtxoSet::default();
    set.insert(
        prevout,
        std::sync::Arc::new(UTXO {
            value: MAX_MONEY,
            script_pubkey: vec![0x51].into(),
            height: 0,
            is_coinbase: false,
        }),
    );
    let (result, fee) = check_tx_inputs(&tx, &set, 200).unwrap();
    assert_eq!(fee, 0);
    assert!(matches!(result, ValidationResult::Invalid(ref reason) if reason.contains("overflow")));
}

#[test]
fn test_output_sum_overflow_connects_as_invalid() {
    let tx = coinbase_with_script(vec![0x51, 0xff], make_n_outputs(4_393, MAX_MONEY));
    let block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root: [1; 32],
            timestamp: 1_000,
            bits: 0x207fffff,
            nonce: 0,
        },
        transactions: vec![tx].into(),
    };
    let result = crate::ConsensusProof::new().validate_block_with_time_context(
        &block,
        &[],
        UtxoSet::default(),
        1,
        Some(TimeContext {
            network_time: 1_700_000_000,
            median_time_past: 0,
        }),
        Network::Regtest,
    );
    match result {
        Ok((ValidationResult::Invalid(_), _)) => {}
        other => panic!("overflow must be an invalid block, got {other:?}"),
    }
}

#[test]
fn test_non_coinbase_null_prevout_is_invalid() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![0x51, 0x51, 0x51, 0x51],
                sequence: 0xffffffff,
            },
            make_input([1; 32], 0),
        ]
        .into(),
        outputs: vec![make_output(0)].into(),
        lock_time: 0,
    };
    assert!(!is_coinbase(&tx));
    assert!(matches!(
        check_transaction(&tx).unwrap(),
        ValidationResult::Invalid(ref reason) if reason.contains("null prevout")
    ));
    let not_null = make_simple_tx(0);
    assert_eq!(
        check_transaction(&not_null).unwrap(),
        ValidationResult::Valid
    );
}

#[test]
fn test_coinbase_scriptsig_length_bounds() {
    for (len, valid) in [(0, false), (1, false), (2, true), (100, true), (101, false)] {
        let tx = coinbase_with_script(vec![0x51; len], vec![make_output(0)]);
        let result = check_transaction(&tx).unwrap();
        assert_eq!(
            matches!(result, ValidationResult::Valid),
            valid,
            "len {len}"
        );
    }
}
