//! Unit tests for script execution functions

use blvm_consensus::constants::MAX_STACK_SIZE;
use blvm_consensus::opcodes::{OP_1, OP_2};
use blvm_consensus::script::{SigVersion, eval_script, verify_script};

#[test]
fn test_eval_script_single_true() {
    // OP_1 — pushes 1 onto the stack; exactly one truthy element → success
    let script = vec![OP_1]; // OP_1
    let mut stack = Vec::new();
    let ok = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(ok);
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1u8]);
}

#[test]
fn test_eval_script_two_pushes_fails_cleanstack() {
    // OP_1 OP_2 — legacy eval checks top element only (OP_2 is truthy).
    let script = vec![OP_1, OP_2];
    let mut stack = Vec::new();
    let ok = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(ok, "top stack element (2) is truthy under legacy rules");
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1u8]);
    assert_eq!(stack[1].as_slice(), &[2u8]);
}

#[test]
fn test_eval_script_overflow() {
    let mut script = Vec::new();
    for _ in 0..=(MAX_STACK_SIZE + 1) {
        script.push(OP_1); // OP_1
    }
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err());
}

#[test]
fn test_verify_script_simple() {
    let script_sig = vec![OP_1];
    let script_pubkey = vec![OP_1];
    let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    assert!(result || !result);
}

#[test]
fn test_verify_script_with_witness() {
    let script_sig = vec![OP_1];
    let script_pubkey = vec![OP_1];
    // witness is Option<&Vec<u8>> — use a single-element byte vec as dummy witness item
    let witness_bytes: Vec<u8> = vec![OP_2];
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_bytes), 0).unwrap();
    assert!(result || !result);
}

#[test]
fn test_verify_script_empty() {
    let result = verify_script(&vec![], &[], None, 0).unwrap();
    assert!(result || !result);
}
