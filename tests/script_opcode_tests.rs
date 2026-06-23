//! Tests for script opcode execution

use blvm_consensus::constants::{MAX_SCRIPT_OPS, MAX_STACK_SIZE};
use blvm_consensus::opcodes::*;
use blvm_consensus::script::cast_to_bool;
use blvm_consensus::script::*;

// Operation limit counts only opcodes **above** OP_16 (0x60); OP_1–OP_16 are push ops and do not
// increment the counter (matches Bitcoin Core). Use OP_NOP (0x61) to exercise the limit.

#[test]
fn test_eval_script_op_1() {
    let script = vec![OP_1];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result, "OP_1 must leave a truthy stack");
}

#[test]
fn test_eval_script_op_dup() {
    let script = vec![OP_1, OP_DUP];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
    assert_eq!(stack.len(), 2);
}

#[test]
fn test_eval_script_op_hash160() {
    let script = vec![OP_1, OP_HASH160];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_eval_script_op_equal() {
    let script = vec![OP_1, OP_1, OP_EQUAL];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
}

#[test]
fn test_eval_script_op_equal_false() {
    let script = vec![OP_1, OP_2, OP_EQUAL];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result, "eval_script completes without opcode failure");
    assert_eq!(stack.len(), 1);
    assert!(
        !cast_to_bool(&stack[0]),
        "OP_EQUAL must push false for unequal operands"
    );
}

#[test]
fn test_eval_script_op_verify() {
    let script = vec![OP_1, OP_VERIFY];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
    assert!(stack.is_empty(), "OP_VERIFY consumes the verified element");
}

#[test]
fn test_eval_script_op_verify_false() {
    let script = vec![OP_0, OP_VERIFY];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result, "OP_VERIFY on false must fail script");
}

#[test]
fn test_eval_script_op_equalverify() {
    let script = vec![OP_1, OP_1, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
}

#[test]
fn test_eval_script_op_checksig() {
    let script = vec![OP_1, OP_1, OP_CHECKSIG];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result, "eval_script completes without opcode failure");
    assert_eq!(stack.len(), 1);
    assert!(
        !cast_to_bool(&stack[0]),
        "OP_CHECKSIG with dummy pushes must leave false on stack"
    );
}

#[test]
fn test_eval_script_unknown_opcode() {
    let script = vec![0xff];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result, "unknown opcode must fail script");
}

#[test]
fn test_eval_script_stack_underflow() {
    let script = vec![OP_DUP];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result, "stack underflow must fail script");
}

#[test]
fn test_eval_script_operation_limit() {
    let script = vec![OP_NOP; MAX_SCRIPT_OPS + 1];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err(), "non-push opcodes exceed MAX_SCRIPT_OPS");
}

#[test]
fn test_eval_script_stack_overflow() {
    let script = vec![OP_1; MAX_STACK_SIZE + 1];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err(), "stack overflow must error");
}

#[test]
fn test_verify_script_basic() {
    let script_sig = vec![OP_1];
    let script_pubkey = vec![OP_1];
    let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    assert!(result, "OP_1/OP_1 must verify");
}

#[test]
fn test_verify_script_with_witness() {
    let script_sig = vec![OP_1];
    let script_pubkey = vec![OP_1];
    let witness = Some(vec![OP_2]);
    let result = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
    assert!(result, "witness must not break OP_1/OP_1 verify");
}

#[test]
fn test_verify_script_empty() {
    let script_sig = vec![];
    let script_pubkey = vec![];
    let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    assert!(
        !result,
        "empty scriptSig/scriptPubKey leaves empty stack → verify false"
    );
}

#[test]
fn test_verify_script_large_scripts() {
    let script_sig = vec![OP_NOP; MAX_SCRIPT_OPS + 1];
    let script_pubkey = vec![OP_1];
    let result = verify_script(&script_sig, &script_pubkey, None, 0);
    assert!(result.is_err());
}
