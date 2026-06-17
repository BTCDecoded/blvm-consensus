//! COV-C-02l: OP_NOP and OP_NOP1..OP_NOP10 coverage.

use blvm_consensus::opcodes::{OP_1, OP_NOP, OP_NOP1, OP_NOP5, OP_NOP10};
use blvm_consensus::script::{SigVersion, eval_script};

#[test]
fn test_op_nop_leaves_stack_unchanged() {
    let script = vec![OP_1, OP_NOP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_op_nop_range_executes_as_noops() {
    for opcode in [OP_NOP1, OP_NOP5, OP_NOP10] {
        let script = vec![OP_1, opcode];
        let mut stack = Vec::new();
        assert!(
            eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap(),
            "opcode 0x{opcode:02x} should succeed as NOP"
        );
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].as_slice(), &[1]);
    }
}
