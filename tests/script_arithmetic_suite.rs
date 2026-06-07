//! COV-C-02h: Script numeric and comparison opcode coverage.

use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_2, OP_3, OP_5, OP_6, OP_ADD, OP_BOOLAND, OP_BOOLOR, OP_DIV, OP_EQUALVERIFY,
    OP_GREATERTHAN, OP_GREATERTHANOREQUAL, OP_LESSTHAN, OP_LESSTHANOREQUAL, OP_MAX, OP_MIN, OP_MOD,
    OP_MUL, OP_NUMEQUAL, OP_NUMEQUALVERIFY, OP_NUMNOTEQUAL, OP_SUB, OP_WITHIN,
};
use blvm_consensus::script::{eval_script, SigVersion};

#[test]
fn test_op_sub_produces_difference() {
    let script = vec![OP_3, OP_1, OP_SUB, OP_2, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_booland_and_boolor() {
    for script in [
        vec![OP_1, OP_2, OP_BOOLAND, OP_1, OP_EQUALVERIFY],
        vec![OP_0, OP_2, OP_BOOLAND, OP_0, OP_EQUALVERIFY],
        vec![OP_0, OP_2, OP_BOOLOR, OP_1, OP_EQUALVERIFY],
        vec![OP_0, OP_0, OP_BOOLOR, OP_0, OP_EQUALVERIFY],
    ] {
        let mut stack = Vec::new();
        assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    }
}

#[test]
fn test_op_numequal_and_notequal() {
    let equal = vec![OP_2, OP_2, OP_NUMEQUAL, OP_1, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&equal, &mut stack, 0, SigVersion::Base).unwrap());

    let notequal = vec![OP_1, OP_2, OP_NUMNOTEQUAL, OP_1, OP_EQUALVERIFY];
    stack.clear();
    assert!(eval_script(&notequal, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_numequalverify_match_and_mismatch() {
    let ok = vec![OP_1, OP_1, OP_NUMEQUALVERIFY, OP_2];
    let mut stack = Vec::new();
    assert!(eval_script(&ok, &mut stack, 0, SigVersion::Base).unwrap());

    let bad = vec![OP_1, OP_2, OP_NUMEQUALVERIFY];
    stack.clear();
    assert!(!eval_script(&bad, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_comparison_ops() {
    let less = vec![OP_3, OP_5, OP_LESSTHAN, OP_1, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&less, &mut stack, 0, SigVersion::Base).unwrap());

    let greater = vec![OP_5, OP_3, OP_GREATERTHAN, OP_1, OP_EQUALVERIFY];
    stack.clear();
    assert!(eval_script(&greater, &mut stack, 0, SigVersion::Base).unwrap());

    let le = vec![OP_3, OP_3, OP_LESSTHANOREQUAL, OP_1, OP_EQUALVERIFY];
    stack.clear();
    assert!(eval_script(&le, &mut stack, 0, SigVersion::Base).unwrap());

    let ge = vec![OP_3, OP_3, OP_GREATERTHANOREQUAL, OP_1, OP_EQUALVERIFY];
    stack.clear();
    assert!(eval_script(&ge, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_min_max_within() {
    let min_script = vec![OP_2, OP_5, OP_MIN, OP_2, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&min_script, &mut stack, 0, SigVersion::Base).unwrap());

    let max_script = vec![OP_2, OP_5, OP_MAX, OP_5, OP_EQUALVERIFY];
    stack.clear();
    assert!(eval_script(&max_script, &mut stack, 0, SigVersion::Base).unwrap());

    // x=3, min=2, max=5 → within [2,5)
    let within = vec![OP_3, OP_2, OP_5, OP_WITHIN, OP_1, OP_EQUALVERIFY];
    stack.clear();
    assert!(eval_script(&within, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_add_underflow_stack_fails() {
    let script = vec![OP_ADD];
    let mut stack = Vec::new();
    assert!(!eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_within_rejects_out_of_range() {
    let script = vec![OP_6, OP_2, OP_5, OP_WITHIN];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert!(
        stack[0].is_empty(),
        "false WITHIN pushes empty CScriptNum for zero"
    );
}

#[test]
fn test_op_booland_false_when_one_zero() {
    let script = vec![OP_1, OP_0, OP_BOOLAND, OP_0, OP_EQUALVERIFY];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_disabled_arithmetic_opcodes_error() {
    for opcode in [OP_MUL, OP_DIV, OP_MOD] {
        let script = vec![OP_1, OP_1, opcode];
        let mut stack = Vec::new();
        assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
    }
}
