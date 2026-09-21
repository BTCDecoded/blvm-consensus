use blvm_consensus::constants::{MAX_SCRIPT_OPS, MAX_STACK_SIZE};
use blvm_consensus::opcodes::{OP_1, OP_NOP};
use blvm_consensus::script::{SigVersion, eval_script, verify_script};

#[test]
fn test_eval_script_op_limit_exceeded() {
    // Non-push opcodes count toward MAX_SCRIPT_OPS (OP_1 is a push and does not).
    let script = vec![OP_NOP; (MAX_SCRIPT_OPS as usize) + 1];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(
        result.is_err(),
        "Script should fail when operation count exceeds limit"
    );
}

#[test]
fn test_eval_script_stack_overflow() {
    // OP_1 pushes a value; exceeding MAX_STACK_SIZE should fail
    let script = vec![OP_1; (MAX_STACK_SIZE as usize) + 1];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(
        result.is_err(),
        "Script should fail when stack size exceeds limit"
    );
}

#[test]
fn test_verify_script_large_scripts_fail() {
    // Large scriptSig + scriptPubKey that together push op/stack constraints
    let large = vec![OP_1; 2048];
    let result = verify_script(&large, &large, None, 0);
    assert!(
        result.is_err() || matches!(result, Ok(false)),
        "Large scripts should fail under current constraints"
    );
}
