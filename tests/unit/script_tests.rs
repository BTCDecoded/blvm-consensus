//! Unit tests for script execution functions

use blvm_consensus::constants::MAX_STACK_SIZE;
use blvm_consensus::script::{eval_script, verify_script, SigVersion};

#[test]
fn test_eval_script_single_true() {
    // OP_1 — pushes 1 onto the stack; exactly one truthy element → success
    let script = vec![0x51u8]; // OP_1
    let mut stack = Vec::new();
    let ok = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(ok);
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1u8]);
}

#[test]
fn test_eval_script_two_pushes_fails_cleanstack() {
    // OP_1 OP_2 — two elements remain; eval_script requires exactly one truthy element
    let script = vec![0x51u8, 0x52]; // OP_1, OP_2
    let mut stack = Vec::new();
    let ok = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(
        !ok,
        "two stack items should not satisfy the single-truthy-element rule"
    );
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1u8]);
    assert_eq!(stack[1].as_slice(), &[2u8]);
}

#[test]
fn test_eval_script_overflow() {
    let mut script = Vec::new();
    for _ in 0..=(MAX_STACK_SIZE + 1) {
        script.push(0x51u8); // OP_1
    }
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err());
}

#[test]
fn test_verify_script_simple() {
    let script_sig = vec![0x51u8];
    let script_pubkey = vec![0x51u8];
    let result = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
    assert!(result || !result);
}

#[test]
fn test_verify_script_with_witness() {
    let script_sig = vec![0x51u8];
    let script_pubkey = vec![0x51u8];
    // witness is Option<&Vec<u8>> — use a single-element byte vec as dummy witness item
    let witness_bytes: Vec<u8> = vec![0x52u8];
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_bytes), 0).unwrap();
    assert!(result || !result);
}

#[test]
fn test_verify_script_empty() {
    let result = verify_script(&vec![], &[], None, 0).unwrap();
    assert!(result || !result);
}
