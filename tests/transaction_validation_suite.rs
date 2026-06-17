//! COV-C-03b: Transaction validation API (`check_transaction`, `check_tx_inputs`, sizing).

use blvm_consensus::constants::{COINBASE_MATURITY, MAX_MONEY};
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::transaction::{
    calculate_transaction_size, check_transaction, check_tx_inputs,
    check_tx_inputs_with_owned_data, check_tx_inputs_with_utxos, is_coinbase,
};
use blvm_consensus::utxo_overlay::UtxoOverlay;
use blvm_consensus::{
    OutPoint, Transaction, TransactionInput, TransactionOutput, UTXO, UtxoSet, ValidationResult,
};
use std::sync::Arc;

fn fund_tx(value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x10; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
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

fn seed_utxo(set: &mut UtxoSet, hash_byte: u8, value: i64, height: u64, is_cb: bool) {
    set.insert(
        OutPoint {
            hash: [hash_byte; 32],
            index: 0,
        },
        Arc::new(UTXO {
            value,
            script_pubkey: vec![OP_1].into(),
            height,
            is_coinbase: is_cb,
        }),
    );
}

#[test]
fn test_is_coinbase_detects_null_prevout() {
    let cb = Transaction {
        version: 1,
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
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    assert!(is_coinbase(&cb));
    assert!(!is_coinbase(&fund_tx(1_000)));
}

#[test]
fn test_calculate_transaction_size_nonzero() {
    let tx = fund_tx(1_000);
    assert!(calculate_transaction_size(&tx) > 0);
}

#[test]
fn test_check_tx_inputs_valid_spend_computes_fee() {
    let mut set = UtxoSet::default();
    seed_utxo(&mut set, 0x10, 10_000, 200, false);
    let tx = fund_tx(9_000);
    let (result, fee) = check_tx_inputs(&tx, &set, 300).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
    assert_eq!(fee, 1_000);
}

#[test]
fn test_check_tx_inputs_rejects_missing_prevout() {
    let tx = fund_tx(1_000);
    let (result, fee) = check_tx_inputs(&tx, &UtxoSet::default(), 100).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_rejects_premature_coinbase_spend() {
    let mut set = UtxoSet::default();
    seed_utxo(&mut set, 0x11, 10_000, 100, true);
    let mut tx = fund_tx(9_000);
    tx.inputs[0].prevout.hash = [0x11; 32];
    let height = 100 + COINBASE_MATURITY - 1;
    let (result, _) = check_tx_inputs(&tx, &set, height).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("Premature")));
}

#[test]
fn test_check_tx_inputs_rejects_insufficient_input_value() {
    let mut set = UtxoSet::default();
    seed_utxo(&mut set, 0x12, 1_000, 200, false);
    let mut tx = fund_tx(2_000);
    tx.inputs[0].prevout.hash = [0x12; 32];
    let (result, _) = check_tx_inputs(&tx, &set, 300).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("Insufficient")));
}

#[test]
fn test_check_tx_inputs_with_owned_data_matches_lookup() {
    let mut set = UtxoSet::default();
    seed_utxo(&mut set, 0x13, 5_000, 200, false);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x13; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 4_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let owned = vec![Some((5_000i64, false, 200u64))];
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, 300, &owned).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
    assert_eq!(fee, 1_000);

    let (lookup_result, lookup_fee) = check_tx_inputs(&tx, &set, 300).unwrap();
    assert_eq!(lookup_result, result);
    assert_eq!(lookup_fee, fee);
}

#[test]
fn test_check_tx_inputs_with_owned_data_rejects_length_mismatch() {
    let tx = fund_tx(1_000);
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, 100, &[]).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
    assert_eq!(fee, 0);
}

#[test]
fn test_check_transaction_rejects_output_sum_overflow() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: MAX_MONEY,
                script_pubkey: vec![OP_1].into(),
            },
            TransactionOutput {
                value: 1,
                script_pubkey: vec![OP_1].into(),
            },
        ]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_rejects_duplicate_inputs() {
    let prevout = OutPoint {
        hash: [0x14; 32],
        index: 0,
    };
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: prevout.clone(),
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout,
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("Duplicate")));
}

#[test]
fn test_check_transaction_rejects_empty_outputs() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x15; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("output")));
}

#[test]
fn test_check_transaction_rejects_total_output_above_max_money() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![OP_1, OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: MAX_MONEY,
                script_pubkey: vec![OP_1].into(),
            },
            TransactionOutput {
                value: 1,
                script_pubkey: vec![OP_1].into(),
            },
        ]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_rejects_coinbase_scriptsig_too_short() {
    let tx = Transaction {
        version: 1,
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
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("scriptSig")));
}

#[test]
fn test_check_transaction_rejects_coinbase_scriptsig_too_long() {
    let mut script_sig = vec![0x4c, 100];
    script_sig.extend(vec![0xab; 100]);
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: script_sig.into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("scriptSig")));
}

#[test]
fn test_check_tx_inputs_with_pre_collected_utxos() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x1c, 10_000, 100, false);
    let op = OutPoint {
        hash: [0x1c; 32],
        index: 0,
    };
    let utxo_ref = base.get(&op).unwrap().as_ref();
    let mut tx = fund_tx(9_000);
    tx.inputs[0].prevout = op;
    let pre = vec![Some(utxo_ref)];
    let (result, fee) = check_tx_inputs_with_utxos(&tx, &base, 200, Some(&pre)).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
    assert_eq!(fee, 1_000);
}

#[test]
fn test_check_transaction_rejects_empty_inputs_non_coinbase() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_check_transaction_rejects_negative_output_value() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x16; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: -1,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("Invalid output value"))
    );
}

#[test]
fn test_check_transaction_rejects_oversized_stripped_tx() {
    use blvm_consensus::constants::MAX_BLOCK_WEIGHT;
    const WITNESS_SCALE: usize = 4;
    let oversize = MAX_BLOCK_WEIGHT / WITNESS_SCALE + 1;
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x17; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![OP_1; oversize.saturating_sub(60)].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("too large")));
}

#[test]
fn test_check_tx_inputs_coinbase_returns_zero_fee() {
    let cb = Transaction {
        version: 1,
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
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&cb, &UtxoSet::default(), 1).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_with_owned_data_rejects_missing_utxo() {
    let tx = fund_tx(1_000);
    let owned = vec![None];
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, 100, &owned).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(ref r) if r.contains("not found")));
    assert_eq!(fee, 0);
}

#[test]
fn test_check_transaction_rejects_output_above_max_money_fast_path() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x19; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: MAX_MONEY + 1,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let result = check_transaction(&tx).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("Invalid output value")),
        "unexpected result: {result:?}"
    );
}

#[test]
fn test_check_tx_inputs_rejects_empty_inputs_non_coinbase() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx, &UtxoSet::default(), 100).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("inputs")),
        "unexpected result: {result:?}"
    );
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_rejects_null_prevout_on_spend() {
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x18; 32],
                    index: 0,
                },
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let (result, fee) = check_tx_inputs(&tx, &UtxoSet::default(), 100).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("null prevout")),
        "unexpected result: {result:?}"
    );
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_with_owned_data_rejects_premature_coinbase() {
    let tx = fund_tx(9_000);
    let owned = vec![Some((10_000i64, true, 100u64))];
    let height = 100 + COINBASE_MATURITY - 1;
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, height, &owned).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("Premature")),
        "unexpected result: {result:?}"
    );
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_with_owned_data_rejects_utxo_value_out_of_bounds() {
    let tx = fund_tx(1_000);
    let owned = vec![Some((-1i64, false, 100u64))];
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, 200, &owned).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("out of bounds")),
        "unexpected result: {result:?}"
    );
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_with_owned_data_rejects_insufficient_input_value() {
    let tx = fund_tx(5_000);
    let owned = vec![Some((1_000i64, false, 100u64))];
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, 200, &owned).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("Insufficient")),
        "unexpected result: {result:?}"
    );
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_with_owned_data_rejects_total_output_above_max_money() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x1a; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: MAX_MONEY / 2,
                script_pubkey: vec![OP_1].into(),
            },
            TransactionOutput {
                value: MAX_MONEY / 2 + 1,
                script_pubkey: vec![OP_1].into(),
            },
        ]
        .into(),
        lock_time: 0,
    };
    let owned = vec![Some((MAX_MONEY, false, 100u64))];
    let (result, fee) = check_tx_inputs_with_owned_data(&tx, 200, &owned).unwrap();
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("maximum")),
        "unexpected result: {result:?}"
    );
    assert_eq!(fee, 0);
}

#[test]
fn test_check_tx_inputs_with_utxo_overlay() {
    let mut base = UtxoSet::default();
    seed_utxo(&mut base, 0x1b, 10_000, 100, false);
    let overlay = UtxoOverlay::new(&base);
    let mut tx = fund_tx(9_000);
    tx.inputs[0].prevout = OutPoint {
        hash: [0x1b; 32],
        index: 0,
    };
    let (result, fee) = check_tx_inputs_with_utxos(&tx, &overlay, 200, None).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
    assert_eq!(fee, 1_000);
}
