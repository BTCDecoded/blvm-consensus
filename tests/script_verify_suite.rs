//! COV-C-02g: verify_script coverage for common success/failure combinations.

use blvm_consensus::opcodes::{OP_0, OP_1, OP_2, OP_3, OP_ADD, OP_EQUALVERIFY, OP_NIP, OP_VERIFY};
use blvm_consensus::script::{eval_script, verify_script, SigVersion};

#[test]
fn test_verify_script_op_true_succeeds() {
    let script_sig: blvm_consensus::types::ByteString = vec![OP_1].into();
    let script_pubkey: blvm_consensus::types::ByteString = vec![OP_1].into();
    assert!(verify_script(&script_sig, &script_pubkey, None, 0).unwrap());
}

#[test]
fn test_verify_script_empty_stack_fails() {
    let script_sig: blvm_consensus::types::ByteString = vec![].into();
    let script_pubkey: blvm_consensus::types::ByteString = vec![OP_ADD].into();
    assert!(!verify_script(&script_sig, &script_pubkey, None, 0).unwrap());
}

#[test]
fn test_verify_script_false_top_fails() {
    let script_sig: blvm_consensus::types::ByteString = vec![OP_1].into();
    let script_pubkey: blvm_consensus::types::ByteString = vec![OP_0].into();
    assert!(!verify_script(&script_sig, &script_pubkey, None, 0).unwrap());
}

#[test]
fn test_op_verify_fails_on_false() {
    let script = vec![OP_0, OP_VERIFY, OP_1];
    let mut stack = Vec::new();
    assert!(!eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_equalverify_mismatch_fails() {
    let script = vec![OP_1, OP_2, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_op_equalverify_match_succeeds() {
    let script = vec![OP_1, OP_1, OP_EQUALVERIFY, OP_2];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.last().unwrap().as_slice(), &[2]);
}

#[test]
fn test_op_nip_success_path() {
    let script = vec![OP_1, OP_2, OP_NIP, OP_3];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[3]);
}
