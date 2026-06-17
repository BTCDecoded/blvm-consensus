//! COV-C-02i: Stack manipulation opcode coverage (DEPTH, PICK, ROLL, SIZE, dup/swap family).

use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_2, OP_2DROP, OP_2DUP, OP_2OVER, OP_2ROT, OP_2SWAP, OP_3, OP_3DUP, OP_4, OP_5,
    OP_6, OP_DEPTH, OP_FROMALTSTACK, OP_IFDUP, OP_NIP, OP_OVER, OP_PICK, OP_ROLL, OP_ROT, OP_SIZE,
    OP_SWAP, OP_TOALTSTACK, OP_TUCK,
};
use blvm_consensus::script::{SigVersion, eval_script};

#[test]
fn test_op_depth_pushes_stack_height() {
    let script = vec![OP_1, OP_2, OP_DEPTH];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[2].as_slice(), &[2]);
}

#[test]
fn test_op_pick_copies_nth_item() {
    let script = vec![OP_1, OP_2, OP_3, OP_1, OP_PICK];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 4);
    assert_eq!(stack[3].as_slice(), &[2]);
}

#[test]
fn test_op_pick_zero_duplicates_top() {
    let script = vec![OP_1, OP_0, OP_PICK];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), stack[1].as_slice());
}

#[test]
fn test_op_pick_out_of_range_fails() {
    let script = vec![OP_1, OP_2, OP_PICK];
    let mut stack = Vec::new();
    assert!(!eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_roll_moves_nth_item_to_top() {
    let script = vec![OP_1, OP_2, OP_3, OP_1, OP_ROLL];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[3]);
    assert_eq!(stack[2].as_slice(), &[2]);
}

#[test]
fn test_op_roll_zero_is_noop() {
    let script = vec![OP_1, OP_0, OP_ROLL];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_op_size_pushes_byte_length_without_popping() {
    let script = vec![OP_1, OP_SIZE];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[1]);

    stack.clear();
    let empty_top = vec![OP_0, OP_SIZE];
    assert!(eval_script(&empty_top, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 2);
    assert!(stack[0].is_empty());
    assert!(stack[1].is_empty());
}

#[test]
fn test_op_ifdup_duplicates_truthy_top() {
    let script = vec![OP_1, OP_IFDUP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[1]);
}

#[test]
fn test_op_ifdup_skips_falsy_top() {
    let script = vec![OP_0, OP_IFDUP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_nip_removes_second_item() {
    let script = vec![OP_1, OP_2, OP_NIP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[2]);
}

#[test]
fn test_op_over_copies_second_to_top() {
    let script = vec![OP_1, OP_2, OP_OVER];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[2].as_slice(), &[1]);
}

#[test]
fn test_op_swap_exchanges_top_two() {
    let script = vec![OP_1, OP_2, OP_SWAP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[1]);
}

#[test]
fn test_op_tuck_inserts_copy_below_second() {
    let script = vec![OP_1, OP_2, OP_TUCK];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[1]);
    assert_eq!(stack[2].as_slice(), &[2]);
}

#[test]
fn test_op_2drop_removes_top_two() {
    let script = vec![OP_1, OP_2, OP_3, OP_2DROP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_op_2dup_duplicates_top_pair() {
    let script = vec![OP_1, OP_2, OP_2DUP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 4);
    assert_eq!(stack[2].as_slice(), &[1]);
    assert_eq!(stack[3].as_slice(), &[2]);
}

#[test]
fn test_op_3dup_duplicates_top_triple() {
    let script = vec![OP_1, OP_2, OP_3, OP_3DUP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 6);
    assert_eq!(stack[3].as_slice(), &[1]);
    assert_eq!(stack[4].as_slice(), &[2]);
    assert_eq!(stack[5].as_slice(), &[3]);
}

#[test]
fn test_op_2over_copies_second_pair() {
    let script = vec![OP_1, OP_2, OP_3, OP_4, OP_2OVER];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 6);
    assert_eq!(stack[4].as_slice(), &[1]);
    assert_eq!(stack[5].as_slice(), &[2]);
}

#[test]
fn test_op_2rot_rotates_third_pair_to_top() {
    let script = vec![OP_1, OP_2, OP_3, OP_4, OP_5, OP_6, OP_2ROT];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 6);
    assert_eq!(stack[4].as_slice(), &[1]);
    assert_eq!(stack[5].as_slice(), &[2]);
}

#[test]
fn test_op_rot_rotates_third_item_to_top() {
    let script = vec![OP_1, OP_2, OP_3, OP_ROT];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[3]);
    assert_eq!(stack[2].as_slice(), &[1]);
}

#[test]
fn test_op_2swap_swaps_second_pair() {
    let script = vec![OP_1, OP_2, OP_3, OP_4, OP_2SWAP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 4);
    assert_eq!(stack[0].as_slice(), &[3]);
    assert_eq!(stack[1].as_slice(), &[4]);
    assert_eq!(stack[2].as_slice(), &[1]);
    assert_eq!(stack[3].as_slice(), &[2]);
}

#[test]
fn test_op_codeseparator_is_noop() {
    use blvm_consensus::opcodes::OP_CODESEPARATOR;
    let script = vec![OP_1, OP_CODESEPARATOR, OP_2];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[2]);
}

#[test]
fn test_op_return_fails_script() {
    use blvm_consensus::opcodes::OP_RETURN;
    let script = vec![OP_1, OP_RETURN];
    let mut stack = Vec::new();
    assert!(!eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn test_op_toaltstack_fromaltstack_roundtrip() {
    let script = vec![OP_1, OP_TOALTSTACK, OP_2, OP_FROMALTSTACK];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[1]);
}

#[test]
fn test_op_fromaltstack_empty_altstack_fails() {
    let script = vec![OP_FROMALTSTACK];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err() || result == Ok(false));
}

#[test]
fn test_unknown_opcode_fails() {
    let script = vec![OP_1, 0xff];
    let mut stack = Vec::new();
    assert!(!eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
}
