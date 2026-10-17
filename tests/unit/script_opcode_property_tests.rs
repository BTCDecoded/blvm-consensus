//! Property tests for script opcode combinations and edge cases.
//!
//! Each test calls `eval_script` and compares the stack to the opcode rule.

use blvm_consensus::constants::MAX_STACK_SIZE;
use blvm_consensus::error::{ConsensusError, ScriptErrorCode};
use blvm_consensus::script::{self, SigVersion};
use proptest::prelude::*;
use ripemd::Ripemd160;
use sha2::{Digest, Sha256};

use blvm_consensus::opcodes::{
    OP_0, OP_1, OP_1_RANGE_END, OP_1_RANGE_START, OP_2DROP, OP_ADD, OP_CHECKSIG, OP_DUP, OP_ELSE,
    OP_ENDIF, OP_EQUAL, OP_EQUALVERIFY, OP_HASH160, OP_IF, OP_MUL, OP_N_BASE, OP_RETURN,
    OP_RIPEMD160, OP_SHA256, OP_SUB, OP_SWAP, OP_VERIFY,
};

fn eval(script: &[u8], stack: &mut Vec<script::StackElement>) -> Result<bool, ConsensusError> {
    script::eval_script(script, stack, 0, SigVersion::Base)
}

fn script_code(result: Result<bool, ConsensusError>) -> Option<ScriptErrorCode> {
    match result {
        Err(ConsensusError::ScriptErrorWithCode { code, .. }) => Some(code),
        _ => None,
    }
}

fn numeric_opcode(n: u8) -> u8 {
    if n == 0 { OP_0 } else { OP_N_BASE + n }
}

/// Minimal script number for `|n| < 128`. Zero is the empty element. Negatives set the high bit.
fn script_num(n: i64) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let magnitude = n.unsigned_abs() as u8;
    if n < 0 {
        vec![magnitude | 0x80]
    } else {
        vec![magnitude]
    }
}

fn push_data(data: &[u8]) -> Vec<u8> {
    let mut script = Vec::with_capacity(1 + data.len());
    script.push(data.len() as u8);
    script.extend_from_slice(data);
    script
}

/// Property test: OP_DUP duplicates stack top correctly
proptest! {
    #[test]
    fn prop_op_dup_duplicates(
        item in prop::collection::vec(any::<u8>(), 0..50)
    ) {
        let mut script = push_data(&item);
        script.push(OP_DUP);
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        prop_assert_eq!(stack.len(), 2);
        prop_assert_eq!(stack[0].as_slice(), item.as_slice());
        prop_assert_eq!(stack[1].as_slice(), item.as_slice());
    }
}

/// OP_EQUALVERIFY succeeds exactly when the two pushes are equal, and pops both.
proptest! {
    #[test]
    fn prop_script_with_op_equalverify(
        item1 in prop::collection::vec(any::<u8>(), 0..10),
        item2 in prop::collection::vec(any::<u8>(), 0..10)
    ) {
        let mut script = push_data(&item1);
        script.extend(push_data(&item2));
        script.push(OP_EQUALVERIFY);
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        if item1 == item2 {
            prop_assert_eq!(result.unwrap(), true);
            prop_assert!(stack.is_empty());
        } else {
            prop_assert_eq!(script_code(result), Some(ScriptErrorCode::EqualVerify));
        }
    }
}

/// OP_HASH160 pushes RIPEMD160(SHA256(input)).
proptest! {
    #[test]
    fn prop_op_hash160_fixed_length(
        input in prop::collection::vec(any::<u8>(), 1..=75)
    ) {
        let mut script = push_data(&input);
        script.push(OP_HASH160);
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        let expected = Ripemd160::digest(Sha256::digest(&input));
        prop_assert_eq!(stack.len(), 1);
        prop_assert_eq!(stack[0].as_slice(), expected.as_slice());
    }
}

/// An empty signature makes OP_CHECKSIG push false. A bare OP_CHECKSIG is a stack error.
proptest! {
    #[test]
    fn prop_op_checksig_stack_requirements(
        pubkey in prop::collection::vec(any::<u8>(), 0..75),
        bare in any::<bool>()
    ) {
        let mut stack = Vec::new();
        if bare {
            let result = eval(&[OP_CHECKSIG], &mut stack);
            prop_assert_eq!(script_code(result), Some(ScriptErrorCode::InvalidStackOperation));
        } else {
            let mut script = vec![OP_0];
            script.extend(push_data(&pubkey));
            script.push(OP_CHECKSIG);
            let result = eval(&script, &mut stack);
            prop_assert_eq!(result.unwrap(), true);
            prop_assert_eq!(stack.len(), 1);
            prop_assert!(stack[0].is_empty());
        }
    }
}

/// OP_1 through OP_16 push the byte `opcode - 0x50`.
proptest! {
    #[test]
    fn prop_op_1_to_16_values(
        opcode in OP_1_RANGE_START..=OP_1_RANGE_END
    ) {
        let mut stack = Vec::new();
        let result = eval(&[opcode], &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        prop_assert_eq!(stack.len(), 1);
        prop_assert_eq!(stack[0].as_slice(), &[opcode - 0x50]);
    }
}

/// OP_0 pushes an empty array.
#[test]
fn prop_op_0_pushes_empty() {
    let mut stack = Vec::new();
    let result = eval(&[OP_0], &mut stack);
    assert_eq!(result.unwrap(), true);
    assert_eq!(stack.len(), 1);
    assert!(stack[0].is_empty());
}

/// OP_IF / OP_ELSE selects the branch with CastToBool, then OP_ENDIF closes it.
proptest! {
    #[test]
    fn prop_script_with_op_if(
        condition in prop::collection::vec(any::<u8>(), 0..10)
    ) {
        let mut script = push_data(&condition);
        script.extend_from_slice(&[OP_IF, OP_1, OP_ELSE, OP_1 + 1, OP_ENDIF]);
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        prop_assert_eq!(stack.len(), 1);
        let taken = if script::cast_to_bool(&condition) { 1 } else { 2 };
        prop_assert_eq!(stack[0].as_slice(), &[taken]);
    }
}

/// A direct push of 1..=75 bytes leaves those bytes on the stack.
proptest! {
    #[test]
    fn prop_pushdata_preserves_data(
        data in prop::collection::vec(any::<u8>(), 1..=75)
    ) {
        let script = push_data(&data);
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        prop_assert_eq!(stack.len(), 1);
        prop_assert_eq!(stack[0].as_slice(), data.as_slice());
    }
}

/// OP_ADD and OP_SUB push the minimal script number. OP_MUL is disabled.
proptest! {
    #[test]
    fn prop_script_arithmetic_opcodes(
        opcode in OP_ADD..=OP_MUL,
        a in 0u8..=16u8,
        b in 0u8..=16u8
    ) {
        let script = vec![numeric_opcode(a), numeric_opcode(b), opcode];
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        if opcode == OP_MUL {
            prop_assert_eq!(script_code(result), Some(ScriptErrorCode::DisabledOpcode));
        } else {
            let value = if opcode == OP_ADD {
                i64::from(a) + i64::from(b)
            } else {
                i64::from(a) - i64::from(b)
            };
            let expected = script_num(value);
            prop_assert_eq!(result.unwrap(), true);
            prop_assert_eq!(stack.len(), 1);
            prop_assert_eq!(stack[0].as_slice(), expected.as_slice());
        }
    }
}

/// OP_EQUAL pushes `[1]` when the numbers match and an empty element when they do not.
proptest! {
    #[test]
    fn prop_script_op_equal(
        a in 0u8..=16u8,
        b in 0u8..=16u8
    ) {
        let script = vec![numeric_opcode(a), numeric_opcode(b), OP_EQUAL];
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        let expect: &[u8] = if a == b { &[1] } else { &[] };
        prop_assert_eq!(stack[0].as_slice(), expect);
    }
}

/// OP_VERIFY succeeds exactly when the pushed number is nonzero, and then the stack is empty.
proptest! {
    #[test]
    fn prop_script_op_verify(
        value in 0u8..=16u8
    ) {
        let script = vec![numeric_opcode(value), OP_VERIFY];
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), value != 0);
        prop_assert!(stack.is_empty());
    }
}

/// OP_RETURN fails the script.
#[test]
fn prop_script_op_return_always_fails() {
    let mut stack = Vec::new();
    let result = eval(&[OP_1, OP_RETURN], &mut stack);
    assert_eq!(result.unwrap(), false);
}

/// OP_SHA256 pushes the 32-byte SHA-256 digest.
proptest! {
    #[test]
    fn prop_op_sha256_fixed_length(
        input in prop::collection::vec(any::<u8>(), 1..=75)
    ) {
        let mut script = push_data(&input);
        script.push(OP_SHA256);
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        let expected = Sha256::digest(&input);
        prop_assert_eq!(stack[0].as_slice(), expected.as_slice());
    }
}

/// OP_RIPEMD160 pushes the 20-byte RIPEMD-160 digest.
proptest! {
    #[test]
    fn prop_op_ripemd160_fixed_length(
        input in prop::collection::vec(any::<u8>(), 1..=75)
    ) {
        let mut script = push_data(&input);
        script.push(OP_RIPEMD160);
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        let expected = Ripemd160::digest(&input);
        prop_assert_eq!(stack[0].as_slice(), expected.as_slice());
    }
}

/// Pushing past `MAX_STACK_SIZE` is a stack-size error. The boundary itself is accepted.
proptest! {
    #[test]
    fn prop_stack_size_limit_enforced(
        push_count in (MAX_STACK_SIZE - 2)..=(MAX_STACK_SIZE + 2)
    ) {
        let script = vec![OP_1; push_count];
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        if push_count > MAX_STACK_SIZE {
            prop_assert_eq!(script_code(result), Some(ScriptErrorCode::StackSize));
        } else {
            prop_assert_eq!(result.unwrap(), true);
            prop_assert_eq!(stack.len(), push_count);
        }
    }
}

/// OP_2DROP removes two items. One item makes the opcode return false.
proptest! {
    #[test]
    fn prop_script_op_2drop(
        a in 1u8..=16u8,
        b in 1u8..=16u8,
        both in any::<bool>()
    ) {
        let script = if both {
            vec![numeric_opcode(a), numeric_opcode(b), OP_2DROP]
        } else {
            vec![numeric_opcode(a), OP_2DROP]
        };
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), both);
        if both {
            prop_assert!(stack.is_empty());
        } else {
            prop_assert_eq!(stack.len(), 1);
            prop_assert_eq!(stack[0].as_slice(), &[a]);
        }
    }
}

/// OP_SWAP exchanges the two numeric pushes.
proptest! {
    #[test]
    fn prop_script_op_swap(
        a in 1u8..=16u8,
        b in 1u8..=16u8
    ) {
        let script = vec![numeric_opcode(a), numeric_opcode(b), OP_SWAP];
        let mut stack = Vec::new();
        let result = eval(&script, &mut stack);
        prop_assert_eq!(result.unwrap(), true);
        prop_assert_eq!(stack[0].as_slice(), &[b]);
        prop_assert_eq!(stack[1].as_slice(), &[a]);
    }
}
