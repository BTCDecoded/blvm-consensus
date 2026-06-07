//! COV-C-02g: OP_CHECKMULTISIG / CHECKMULTISIGVERIFY interpreter paths.

use blvm_consensus::opcodes::{OP_0, OP_1, OP_2, OP_3, OP_CHECKMULTISIG, OP_CHECKMULTISIGVERIFY};
use blvm_consensus::script::{eval_script, verify_script, SigVersion};

#[test]
fn test_checkmultisig_zero_of_zero_succeeds() {
    let script = vec![OP_0, OP_0, OP_0, OP_CHECKMULTISIG];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_checkmultisigverify_zero_of_zero_leaves_true() {
    let script = vec![OP_0, OP_0, OP_0, OP_CHECKMULTISIGVERIFY, OP_1];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_verify_script_multisig_accepts_valid_null_dummy() {
    let script_sig = vec![OP_0, OP_0];
    let script_pubkey = vec![OP_0, OP_1, OP_1, OP_2, OP_CHECKMULTISIG];
    assert!(verify_script(&script_sig, &script_pubkey, None, 0).unwrap());
}

#[test]
fn test_checkmultisig_three_of_three_script_evaluates() {
    let script = vec![
        OP_0,
        OP_0,
        OP_0,
        OP_0,
        OP_0,
        OP_3,
        OP_1,
        OP_1,
        OP_1,
        OP_3,
        OP_CHECKMULTISIG,
    ];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}
