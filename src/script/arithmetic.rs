//! Numeric and comparison opcodes for script execution.
//!
//! OP_ADD, OP_SUB, OP_BOOLAND, OP_BOOLOR, OP_NUMEQUAL, OP_LESSTHAN, etc.

use crate::error::{ConsensusError, Result, ScriptErrorCode};

use super::stack::{StackElement, to_stack_element};

pub(crate) fn op_add(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(b + a)));
    Ok(true)
}

pub(crate) fn op_sub(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(b - a)));
    Ok(true)
}

pub(crate) fn op_mul_disabled() -> Result<bool> {
    Err(ConsensusError::ScriptErrorWithCode {
        code: ScriptErrorCode::DisabledOpcode,
        message: "OP_MUL is disabled".into(),
    })
}

pub(crate) fn op_div_disabled() -> Result<bool> {
    Err(ConsensusError::ScriptErrorWithCode {
        code: ScriptErrorCode::DisabledOpcode,
        message: "OP_DIV is disabled".into(),
    })
}

pub(crate) fn op_mod_disabled() -> Result<bool> {
    Err(ConsensusError::ScriptErrorWithCode {
        code: ScriptErrorCode::DisabledOpcode,
        message: "OP_MOD is disabled".into(),
    })
}

pub(crate) fn op_lshift_disabled() -> Result<bool> {
    Err(ConsensusError::ScriptErrorWithCode {
        code: ScriptErrorCode::DisabledOpcode,
        message: "OP_LSHIFT is disabled".into(),
    })
}

pub(crate) fn op_rshift_disabled() -> Result<bool> {
    Err(ConsensusError::ScriptErrorWithCode {
        code: ScriptErrorCode::DisabledOpcode,
        message: "OP_RSHIFT is disabled".into(),
    })
}

pub(crate) fn op_booland(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(
        if a != 0 && b != 0 { 1 } else { 0 },
    )));
    Ok(true)
}

pub(crate) fn op_boolor(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(
        if a != 0 || b != 0 { 1 } else { 0 },
    )));
    Ok(true)
}

pub(crate) fn op_numequal(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(if a == b {
        1
    } else {
        0
    })));
    Ok(true)
}

pub(crate) fn op_numequalverify(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    Ok(a == b)
}

pub(crate) fn op_numnotequal(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(if a != b {
        1
    } else {
        0
    })));
    Ok(true)
}

pub(crate) fn op_lessthan(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(if b < a {
        1
    } else {
        0
    })));
    Ok(true)
}

pub(crate) fn op_greaterthan(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(if b > a {
        1
    } else {
        0
    })));
    Ok(true)
}

pub(crate) fn op_lessthanorequal(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(if b <= a {
        1
    } else {
        0
    })));
    Ok(true)
}

pub(crate) fn op_greaterthanorequal(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(if b >= a {
        1
    } else {
        0
    })));
    Ok(true)
}

pub(crate) fn op_min(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(std::cmp::min(
        b, a,
    ))));
    Ok(true)
}

pub(crate) fn op_max(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let a = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let b = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(std::cmp::max(
        b, a,
    ))));
    Ok(true)
}

pub(crate) fn op_within(stack: &mut Vec<StackElement>) -> Result<bool> {
    if stack.len() < 3 {
        return Ok(false);
    }
    let max_val = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let min_val = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    let x = super::script_num_decode(&stack.pop().unwrap(), 4)?;
    stack.push(to_stack_element(&super::script_num_encode(
        if x >= min_val && x < max_val { 1 } else { 0 },
    )));
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_add_decodes_script_numbers() {
        let mut stack = vec![to_stack_element(&[2]), to_stack_element(&[3])];
        assert!(op_add(&mut stack).unwrap());
        assert_eq!(&stack[0][..], &[5]);

        // 0x82 is script-number -2. Pop order is a = 3, b = -2, so b + a = 1.
        let mut stack = vec![to_stack_element(&[0x82]), to_stack_element(&[3])];
        assert!(op_add(&mut stack).unwrap());
        assert_eq!(&stack[0][..], &[1]);

        let mut empty = Vec::new();
        assert!(!op_add(&mut empty).unwrap());
    }
}
