//! Control flow structures for script execution (OP_IF, OP_NOTIF, OP_ENDIF).

use blvm_spec_lock::spec_locked;

use crate::opcodes::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlBlock {
    If { executing: bool },
    NotIf { executing: bool },
}

/// True if we're in a non-executing branch (IF/NOTIF with executing=false). Used by both paths.
#[inline(always)]
pub(crate) fn in_false_branch(control_stack: &[ControlBlock]) -> bool {
    control_stack.iter().any(|b| {
        !matches!(
            b,
            ControlBlock::If { executing: true } | ControlBlock::NotIf { executing: true }
        )
    })
}

/// Minimal IF/NOTIF condition encoding (MINIMALIF).
/// Valid encodings: empty (false), or single byte 0, 1..16, or OP_1..OP_16.
#[spec_locked("5.2.4", "IsMinimalIfCondition")]
#[blvm_spec_lock::ensures(result == false || bytes.len() <= 1)]
pub(crate) fn is_minimal_if_condition(bytes: &[u8]) -> bool {
    match bytes.len() {
        0 => true, // empty = minimal false
        1 => {
            let b = bytes[0];
            b == 0 || (1..=16).contains(&b) || (OP_1..=OP_16).contains(&b)
        }
        _ => false,
    }
}
