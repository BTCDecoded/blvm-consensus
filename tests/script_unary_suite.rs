//! COV-C-02k: Unary numeric opcode coverage (1ADD, 1SUB, NEGATE, ABS, NOT, 0NOTEQUAL).

use blvm_consensus::opcodes::{
    OP_0, OP_0NOTEQUAL, OP_1, OP_1ADD, OP_1NEGATE, OP_1SUB, OP_2, OP_ABS, OP_EQUALVERIFY,
    OP_NEGATE, OP_NOT,
};
use blvm_consensus::script::{eval_script, SigVersion};

#[test]
fn test_op_1add_and_1sub() {
    let add = vec![OP_1, OP_1ADD, OP_2, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&add, &mut stack, 0, SigVersion::Base).unwrap());

    stack.clear();
    let sub = vec![OP_1, OP_1SUB, OP_0, OP_EQUALVERIFY];
    assert!(eval_script(&sub, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_negate_and_abs() {
    let mut stack = Vec::new();
    assert!(eval_script(
        &[OP_1NEGATE, OP_NEGATE, OP_1, OP_EQUALVERIFY],
        &mut stack,
        0,
        SigVersion::Base
    )
    .unwrap());

    stack.clear();
    assert!(eval_script(
        &[OP_1NEGATE, OP_ABS, OP_1, OP_EQUALVERIFY],
        &mut stack,
        0,
        SigVersion::Base
    )
    .unwrap());
}

#[test]
fn test_op_not_and_0notequal() {
    let mut stack = Vec::new();
    assert!(eval_script(
        &[OP_0, OP_NOT, OP_1, OP_EQUALVERIFY],
        &mut stack,
        0,
        SigVersion::Base
    )
    .unwrap());

    stack.clear();
    assert!(eval_script(
        &[OP_1, OP_NOT, OP_0, OP_EQUALVERIFY],
        &mut stack,
        0,
        SigVersion::Base
    )
    .unwrap());

    stack.clear();
    assert!(eval_script(
        &[OP_0, OP_0NOTEQUAL, OP_0, OP_EQUALVERIFY],
        &mut stack,
        0,
        SigVersion::Base
    )
    .unwrap());

    stack.clear();
    assert!(eval_script(
        &[OP_1, OP_0NOTEQUAL, OP_1, OP_EQUALVERIFY],
        &mut stack,
        0,
        SigVersion::Base
    )
    .unwrap());
}
