//! COV-C-02j: Disabled and reserved script opcodes must fail evaluation.

use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_2DIV, OP_2MUL, OP_AND, OP_CAT, OP_INVERT, OP_LEFT, OP_MUL, OP_OR, OP_RIGHT,
    OP_VER, OP_XOR,
};
use blvm_consensus::script::{eval_script, SigVersion};

#[test]
fn test_disabled_string_opcodes_error() {
    let script = vec![OP_1, OP_1, OP_CAT];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_disabled_bitwise_opcodes_error() {
    let script = vec![OP_1, OP_INVERT];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_op_ver_fails_evaluation() {
    let script = vec![OP_VER];
    let mut stack = Vec::new();
    assert!(!eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_disabled_numeric_opcodes_error() {
    let script = vec![OP_1, OP_2MUL];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_disabled_mul_still_errors_with_stack_setup() {
    let script = vec![OP_0, OP_0, OP_MUL];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_disabled_2div_opcode_error() {
    let script = vec![OP_1, OP_2DIV];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_disabled_string_slice_opcodes_error() {
    for opcode in [OP_LEFT, OP_RIGHT] {
        let script = vec![OP_1, OP_1, opcode];
        let mut stack = Vec::new();
        assert!(
            eval_script(&script, &mut stack, 0, SigVersion::Base).is_err(),
            "opcode 0x{opcode:02x} should be disabled"
        );
    }
}

#[test]
fn test_disabled_and_or_xor_opcodes_error() {
    for opcode in [OP_AND, OP_OR, OP_XOR] {
        let script = vec![OP_1, OP_1, opcode];
        let mut stack = Vec::new();
        assert!(
            eval_script(&script, &mut stack, 0, SigVersion::Base).is_err(),
            "opcode 0x{opcode:02x} should be disabled"
        );
    }
}
