//! COV-C-05b: Sigop counting and cost coverage (legacy, P2SH, witness, tapscript).

use bitcoin_hashes::{Hash as BitcoinHash, hash160, sha256};
use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_3, OP_CHECKMULTISIG, OP_CHECKSIG, OP_CHECKSIGADD, OP_EQUAL, OP_HASH160,
    PUSH_20_BYTES, PUSH_32_BYTES,
};
use blvm_consensus::script::flags::SCRIPT_VERIFY_WITNESS;
use blvm_consensus::segwit::Witness;
use blvm_consensus::sigop::{
    count_sigops_in_script, count_tapscript_sigops, get_legacy_sigop_count,
    get_legacy_sigop_count_accurate, get_p2sh_sigop_count, get_transaction_sigop_cost,
    get_transaction_sigop_cost_with_utxos, get_transaction_sigop_count,
    get_transaction_sigop_count_for_bip54, is_pay_to_script_hash,
};
use blvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput, UTXO, UtxoSet};
use std::sync::Arc;

fn p2sh_scriptpubkey(redeem_script: &[u8]) -> Vec<u8> {
    let h = hash160::Hash::hash(redeem_script);
    let mut spk = vec![OP_HASH160, PUSH_20_BYTES];
    spk.extend_from_slice(h.as_ref());
    spk.push(OP_EQUAL);
    spk
}

#[test]
fn test_count_sigops_in_script_accurate_vs_fast() {
    let script = vec![OP_CHECKSIG, OP_CHECKSIG];
    assert_eq!(count_sigops_in_script(&script, false), 2);
    assert_eq!(count_sigops_in_script(&script, true), 2);
}

#[test]
fn test_count_tapscript_sigops_includes_checksigadd() {
    let script = vec![OP_CHECKSIG, OP_CHECKSIGADD];
    assert_eq!(count_tapscript_sigops(&script), 2);
}

#[test]
fn test_is_pay_to_script_hash() {
    let redeem = vec![OP_1];
    let spk = p2sh_scriptpubkey(&redeem);
    assert!(is_pay_to_script_hash(&spk));
    assert!(!is_pay_to_script_hash(&vec![OP_1]));
}

#[test]
fn test_get_p2sh_sigop_count_from_redeem_script() {
    let redeem = vec![OP_CHECKSIG, OP_CHECKSIG];
    let spk = p2sh_scriptpubkey(&redeem);
    let mut script_sig = Vec::new();
    script_sig.push(redeem.len() as u8);
    script_sig.extend_from_slice(&redeem);

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x01; 32],
                index: 0,
            },
            script_sig: script_sig.into(),
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

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        tx.inputs[0].prevout.clone(),
        Arc::new(UTXO {
            value: 10_000,
            script_pubkey: spk.into(),
            height: 0,
            is_coinbase: false,
        }),
    );

    let count = get_p2sh_sigop_count(&tx, &utxo_set).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_get_transaction_sigop_cost_scales_legacy() {
    let script = vec![OP_CHECKSIG; 3];
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }]
        .into(),
        lock_time: 0,
    };
    let cost = get_transaction_sigop_cost(&tx, &UtxoSet::default(), None, 0).unwrap();
    assert_eq!(cost, 12, "3 legacy sigops × witness scale factor 4 = 12");
    assert_eq!(get_legacy_sigop_count(&tx), 3);
}

#[test]
fn test_get_transaction_sigop_count_with_p2wsh_witness() {
    let witness_script = vec![OP_CHECKSIG];
    let hash = sha256::Hash::hash(&witness_script);
    let mut spk = vec![OP_0, PUSH_32_BYTES];
    spk.extend_from_slice(hash.as_ref());

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
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        tx.inputs[0].prevout.clone(),
        Arc::new(UTXO {
            value: 10_000,
            script_pubkey: spk.into(),
            height: 0,
            is_coinbase: false,
        }),
    );

    let witnesses: Witness = vec![witness_script];
    let total =
        get_transaction_sigop_count(&tx, &utxo_set, Some(&[witnesses]), SCRIPT_VERIFY_WITNESS)
            .unwrap();
    assert_eq!(total, 1, "P2WSH witness script should contribute 1 sigop");
}

#[test]
fn test_get_legacy_sigop_count_accurate_matches_fast_for_checksigs() {
    let script = vec![OP_CHECKSIG, OP_CHECKSIG];
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }]
        .into(),
        lock_time: 0,
    };
    assert_eq!(
        get_legacy_sigop_count(&tx),
        get_legacy_sigop_count_accurate(&tx)
    );
}

#[test]
fn test_get_transaction_sigop_cost_with_utxos_p2wpkh_witness() {
    let mut spk = vec![OP_0, 0x14];
    spk.extend_from_slice(&[0xcd; 20]);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x03; 32],
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
    let utxo = UTXO {
        value: 10_000,
        script_pubkey: spk.into(),
        height: 0,
        is_coinbase: false,
    };
    let witnesses: Vec<Witness> = vec![vec![vec![0x30; 72], vec![0x21; 33]]];
    let cost = get_transaction_sigop_cost_with_utxos(
        &tx,
        &[Some(&utxo)],
        Some(&witnesses),
        SCRIPT_VERIFY_WITNESS,
    )
    .unwrap();
    assert_eq!(cost, 1, "P2WPKH witness stack should add 1 sigop cost unit");
}

#[test]
fn test_get_transaction_sigop_count_for_bip54_uses_accurate_counting() {
    let script = vec![OP_CHECKSIG; 2];
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script.into(),
        }]
        .into(),
        lock_time: 0,
    };
    let count = get_transaction_sigop_count_for_bip54(&tx, &UtxoSet::default(), None, 0).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_count_sigops_accurate_multisig_uses_op_n() {
    let script = vec![OP_3, OP_1, OP_1, OP_1, OP_3, OP_CHECKMULTISIG];
    assert_eq!(count_sigops_in_script(&script, true), 3);
    assert_eq!(count_sigops_in_script(&script, false), 20);
}

#[test]
fn test_get_p2sh_sigop_count_pushdata2_redeem_script() {
    let redeem = vec![OP_CHECKSIG, OP_CHECKSIG];
    let spk = p2sh_scriptpubkey(&redeem);
    let mut script_sig = vec![
        0x4d,
        (redeem.len() & 0xff) as u8,
        ((redeem.len() >> 8) & 0xff) as u8,
    ];
    script_sig.extend_from_slice(&redeem);

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x04; 32],
                index: 0,
            },
            script_sig: script_sig.into(),
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

    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        tx.inputs[0].prevout.clone(),
        Arc::new(UTXO {
            value: 10_000,
            script_pubkey: spk.into(),
            height: 0,
            is_coinbase: false,
        }),
    );

    assert_eq!(get_p2sh_sigop_count(&tx, &utxo_set).unwrap(), 2);
}
