//! COV-C-02b: Script control-flow and VERIFY opcode coverage (IF/NOTIF/ELSE/ENDIF, MINIMALIF).

use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_2, OP_3, OP_ELSE, OP_ENDIF, OP_IF, OP_NOTIF, OP_VERIFY,
};
use blvm_consensus::script::flags::SCRIPT_VERIFY_MINIMALIF;
use blvm_consensus::script::{eval_script, verify_script, SigVersion};

#[test]
fn test_op_if_else_true_branch() {
    // OP_1 OP_IF OP_2 OP_ELSE OP_3 OP_ENDIF → stack [2]
    let script = vec![OP_1, OP_IF, OP_2, OP_ELSE, OP_3, OP_ENDIF];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[2]);
}

#[test]
fn test_op_if_else_false_branch() {
    // OP_0 OP_IF OP_2 OP_ELSE OP_3 OP_ENDIF → stack [3]
    let script = vec![OP_0, OP_IF, OP_2, OP_ELSE, OP_3, OP_ENDIF];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[3]);
}

#[test]
fn test_op_notif_executes_when_condition_false() {
    // OP_0 OP_NOTIF OP_1 OP_ENDIF → stack [1]
    let script = vec![OP_0, OP_NOTIF, OP_1, OP_ENDIF];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_op_notif_skips_when_condition_true() {
    // OP_1 OP_NOTIF OP_2 OP_ENDIF — NOTIF pops the condition; true skips the branch.
    let script = vec![OP_1, OP_NOTIF, OP_2, OP_ENDIF];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert!(stack.is_empty());
}

#[test]
fn test_op_if_non_minimal_rejected_with_minimalif() {
    // MINIMALIF applies under witness v0/tapscript, not bare Base scripts.
    let script = vec![0x02, 0x01, 0x00, OP_IF, OP_1, OP_ENDIF];
    let mut stack = Vec::new();
    let result = eval_script(
        &script,
        &mut stack,
        SCRIPT_VERIFY_MINIMALIF,
        SigVersion::WitnessV0,
    );
    assert!(result.is_err());
}

#[test]
fn test_op_verify_success_consumes_top() {
    let script = vec![OP_1, OP_VERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert!(stack.is_empty());
}

#[test]
fn test_op_verify_failure_on_false() {
    let script = vec![OP_0, OP_VERIFY];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err() || result == Ok(false));
}

#[test]
fn test_op_else_without_if_fails() {
    let script = vec![OP_ELSE, OP_ENDIF];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err() || result == Ok(false));
}

#[test]
fn test_op_if_nested_branches() {
    let script = vec![OP_1, OP_IF, OP_1, OP_IF, OP_2, OP_ENDIF, OP_3, OP_ENDIF];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert!(stack.iter().any(|item| item.as_slice() == &[2]));
}

#[test]
fn test_verify_script_if_else_true() {
    let script_sig = vec![OP_1, OP_IF, OP_2, OP_ELSE, OP_3, OP_ENDIF];
    let script_pubkey = vec![OP_1];
    assert!(verify_script(&script_sig, &script_pubkey, None, 0).unwrap());
}

#[test]
fn test_op_if_empty_stack_fails() {
    let script = vec![OP_IF, OP_1, OP_ENDIF];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err() || result == Ok(false));
}

#[test]
fn test_op_endif_without_if_fails() {
    let script = vec![OP_1, OP_ENDIF];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err() || result == Ok(false));
}
