use blvm_consensus::opcodes::OP_1;
use blvm_consensus::script::{eval_script, verify_script};
use blvm_consensus::constants::{MAX_SCRIPT_OPS, MAX_STACK_SIZE};

#[test]
fn test_eval_script_op_limit_exceeded() {
    // Create a script with more than MAX_SCRIPT_OPS of OP_1
    let mut script = Vec::new();
    script.resize((MAX_SCRIPT_OPS as usize) + 1, OP_1);
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0).unwrap();
    assert!(!result, "Script should fail when operation count exceeds limit");
}

#[test]
fn test_eval_script_stack_overflow() {
    // OP_1 pushes a value; exceeding MAX_STACK_SIZE should fail
    let mut script = Vec::new();
    script.resize((MAX_STACK_SIZE as usize) + 1, OP_1);
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0).unwrap();
    assert!(!result, "Script should fail when stack size exceeds limit");
}

#[test]
fn test_verify_script_large_scripts_fail() {
    // Large scriptSig + scriptPubKey that together push op/stack constraints
    let mut large = vec![OP_1; 2048];
    let ok = verify_script(&large, &large, None, 0).unwrap();
    assert!(!ok, "Large scripts should fail under current constraints");
}




































