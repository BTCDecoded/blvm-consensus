//! A non-witness spend with a witness stack of empty items.
//!
//! The stack is absent only when it has no items. One empty item is still
//! witness data, and both verifiers must reject it. A stack with no items
//! must be accepted.

#![cfg(feature = "production")]

use bitcoinconsensus::{Error, Utxo, VERIFY_P2SH, VERIFY_WITNESS, verify_with_flags};
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::script::flags::{SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_WITNESS};
use blvm_consensus::script::{SigVersion, verify_script_with_context_full};
use blvm_consensus::serialization::transaction::{
    serialize_transaction, serialize_transaction_with_witness,
};
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};

const FLAGS: u32 = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS;
const VALUE: i64 = 1_000;

fn spend() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: VALUE,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    }
}

fn library_accepts(raw: &[u8], script_pubkey: &[u8]) -> bool {
    let spent = [Utxo {
        script_pubkey: script_pubkey.as_ptr(),
        script_pubkey_len: script_pubkey.len() as u32,
        value: VALUE as i64,
    }];
    match verify_with_flags(script_pubkey, VALUE as u64, raw, Some(&spent), 0, FLAGS) {
        Ok(()) => true,
        Err(Error::ERR_SCRIPT) => false,
        Err(err) => panic!("script library api error: {err}"),
    }
}

fn blvm_accepts(tx: &Transaction, script_pubkey: &[u8], witness: Option<&Vec<Vec<u8>>>) -> bool {
    let prevout_values = [VALUE];
    let prevouts: [&[u8]; 1] = [script_pubkey];
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        witness,
        FLAGS,
        tx,
        0,
        &prevout_values,
        &prevouts,
        Some(500_000),
        None,
        Network::Mainnet,
        SigVersion::Base,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    matches!(result, Ok(true))
}

#[test]
fn unexpected_witness_matches_script_library() {
    assert_eq!(SCRIPT_VERIFY_P2SH, VERIFY_P2SH);
    assert_eq!(SCRIPT_VERIFY_WITNESS, VERIFY_WITNESS);

    let tx = spend();
    let script_pubkey = vec![OP_1];

    let bare = serialize_transaction(&tx);
    assert_eq!(
        blvm_accepts(&tx, &script_pubkey, None),
        library_accepts(&bare, &script_pubkey),
        "no witness items"
    );
    assert!(library_accepts(&bare, &script_pubkey));

    for witness in [vec![vec![]], vec![vec![], vec![]]] {
        let raw = serialize_transaction_with_witness(&tx, std::slice::from_ref(&witness));
        let ours = blvm_accepts(&tx, &script_pubkey, Some(&witness));
        let theirs = library_accepts(&raw, &script_pubkey);
        assert_eq!(
            ours,
            theirs,
            "witness stack of {} empty items",
            witness.len()
        );
        assert!(!theirs, "empty items are still a witness stack");
    }
}
