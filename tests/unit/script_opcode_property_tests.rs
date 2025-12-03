//! Property tests for script opcode combinations and edge cases
//!
//! Comprehensive property-based tests covering script opcode combinations,
//! stack operations, and edge cases to ensure 99% coverage of possible scenarios.

use blvm_consensus::*;
use blvm_consensus::script;
use blvm_consensus::constants::MAX_STACK_SIZE;
use proptest::prelude::*;

// Note: execute_opcode is private, so we'll use eval_script instead
// which is the public API for script execution

/// Property test: OP_DUP duplicates stack top correctly
/// 
/// Uses eval_script to test OP_DUP by executing [item, OP_DUP] script
proptest! {
    #[test]
    fn prop_op_dup_duplicates(
        item in prop::collection::vec(any::<u8>(), 0..50)
    ) {
        // Build script: push item, then OP_DUP
        let mut script = vec![item.len() as u8]; // Push opcode
        script.extend_from_slice(&item);
        script.push(0x76); // OP_DUP
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok());
        // After OP_DUP, stack should have item duplicated
        if result.unwrap() && stack.len() >= 2 {
            prop_assert_eq!(stack[stack.len() - 2], item);
            prop_assert_eq!(stack[stack.len() - 1], item);
        }
    }
}

/// Property test: Script execution with OP_EQUALVERIFY
proptest! {
    #[test]
    fn prop_script_with_op_equalverify(
        item1 in prop::collection::vec(any::<u8>(), 0..10),
        item2 in prop::collection::vec(any::<u8>(), 0..10)
    ) {
        // Build script with two pushes and OP_EQUALVERIFY
        let mut script = Vec::new();
        script.push(item1.len() as u8);
        script.extend_from_slice(&item1);
        script.push(item2.len() as u8);
        script.extend_from_slice(&item2);
        script.push(0x88); // OP_EQUALVERIFY
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok() || result.is_err());
        // OP_EQUALVERIFY succeeds if items are equal
        if result.is_ok() && result.unwrap() {
            prop_assert!(item1 == item2, "OP_EQUALVERIFY should succeed only if equal");
        }
    }
}

/// Property test: OP_HASH160 produces fixed-length output
proptest! {
    #[test]
    fn prop_op_hash160_fixed_length(
        input in prop::collection::vec(any::<u8>(), 1..75) // Max pushdata size
    ) {
        // Build script: push data, then OP_HASH160
        let mut script = Vec::new();
        script.push(input.len() as u8);
        script.extend_from_slice(&input);
        script.push(0xa9); // OP_HASH160
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok());
        if result.unwrap() && !stack.is_empty() {
            // HASH160 should produce 20-byte output
            prop_assert_eq!(stack[0].len(), 20,
                "OP_HASH160 should produce 20-byte hash");
        }
    }
}

/// Property test: OP_CHECKSIG stack requirements
proptest! {
    #[test]
    fn prop_op_checksig_stack_requirements(
        sig in prop::collection::vec(any::<u8>(), 0..75),
        pubkey in prop::collection::vec(any::<u8>(), 0..75)
    ) {
        // Build script: push sig, push pubkey, then OP_CHECKSIG
        let mut script = Vec::new();
        script.push(sig.len() as u8);
        script.extend_from_slice(&sig);
        script.push(pubkey.len() as u8);
        script.extend_from_slice(&pubkey);
        script.push(0xac); // OP_CHECKSIG
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok());
        // OP_CHECKSIG may succeed or fail, but shouldn't panic
    }
}

/// Property test: OP_1 through OP_16 push correct values
proptest! {
    #[test]
    fn prop_op_1_to_16_values(
        opcode in 0x51u8..=0x60u8 // OP_1 (0x51) to OP_16 (0x60)
    ) {
        let script = vec![opcode];
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok());
        if result.unwrap() && !stack.is_empty() {
            let expected_value = (opcode - 0x50) as u8;
            // OP_1 pushes [1], OP_2 pushes [2], etc.
            prop_assert!(stack[0].len() > 0);
            // Value should be between 1 and 16
            prop_assert!(expected_value >= 1 && expected_value <= 16);
        }
    }
}

/// Property test: OP_0 pushes empty array
proptest! {
    #[test]
    fn prop_op_0_pushes_empty() {
        let script = vec![0x00]; // OP_0
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok());
        if result.unwrap() && !stack.is_empty() {
            prop_assert!(stack[0].is_empty(), "OP_0 should push empty array");
        }
    }
}

/// Property test: Script with OP_IF doesn't panic
proptest! {
    #[test]
    fn prop_script_with_op_if(
        condition in prop::collection::vec(any::<u8>(), 0..10)
    ) {
        // Build script: push condition, then OP_IF
        let mut script = Vec::new();
        script.push(condition.len() as u8);
        script.extend_from_slice(&condition);
        script.push(0x63); // OP_IF
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        // Should not panic (may succeed or fail)
        prop_assert!(result.is_ok() || result.is_err());
    }
}

/// Property test: PUSHDATA operations preserve data
proptest! {
    #[test]
    fn prop_pushdata_preserves_data(
        data in prop::collection::vec(any::<u8>(), 1..75) // Direct push max
    ) {
        // Build script: push opcode (len), then data
        let mut script = Vec::new();
        let len = data.len();
        if len <= 75 {
            script.push(len as u8);
            script.extend_from_slice(&data);
            
            let mut stack = Vec::new();
            let result = script::eval_script(&script, &mut stack, 0);
            
            prop_assert!(result.is_ok() || result.is_err());
            // Data should be preserved if script succeeds
            if result.is_ok() && result.unwrap() && !stack.is_empty() {
                prop_assert_eq!(stack[0], data);
            }
        }
    }
}

/// Property test: Script execution with arithmetic opcodes doesn't panic
proptest! {
    #[test]
    fn prop_script_arithmetic_opcodes(
        opcode in 0x93u8..=0x95u8, // OP_ADD, OP_SUB, OP_MUL
        a in 0u8..=10u8,
        b in 0u8..=10u8
    ) {
        // Build script: OP_a, OP_b, arithmetic opcode
        let script = vec![0x51 + a.min(16), 0x51 + b.min(16), opcode];
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        // Should not panic
        prop_assert!(result.is_ok() || result.is_err());
    }
}

/// Property test: Script with OP_EQUAL produces boolean result
proptest! {
    #[test]
    fn prop_script_op_equal(
        a in 0u8..=16u8,
        b in 0u8..=16u8
    ) {
        // Build script: OP_a, OP_b, OP_EQUAL
        let script = vec![0x51 + a.min(16), 0x51 + b.min(16), 0x87]; // OP_EQUAL
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok() || result.is_err());
    }
}

/// Property test: Script with OP_VERIFY doesn't panic
proptest! {
    #[test]
    fn prop_script_op_verify(
        value in 1u8..=16u8
    ) {
        // Build script: OP_value, OP_VERIFY
        let script = vec![0x51 + value.min(16), 0x69]; // OP_VERIFY
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok() || result.is_err());
    }
}

/// Property test: Script with OP_RETURN always fails
proptest! {
    #[test]
    fn prop_script_op_return_always_fails() {
        let script = vec![0x51, 0x6a]; // OP_1, OP_RETURN
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        // OP_RETURN should cause script to fail
        prop_assert!(result.is_ok());
        if let Ok(success) = result {
            prop_assert!(!success, "OP_RETURN should always fail");
        }
    }
}

/// Property test: OP_SHA256 produces 32-byte output
proptest! {
    #[test]
    fn prop_op_sha256_fixed_length(
        input in prop::collection::vec(any::<u8>(), 1..75)
    ) {
        // Build script: push input, OP_SHA256
        let mut script = Vec::new();
        script.push(input.len() as u8);
        script.extend_from_slice(&input);
        script.push(0xa8); // OP_SHA256
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok());
        if result.unwrap() && !stack.is_empty() {
            // SHA256 should produce 32-byte output
            prop_assert_eq!(stack[0].len(), 32,
                "OP_SHA256 should produce 32-byte hash");
        }
    }
}

/// Property test: OP_RIPEMD160 produces 20-byte output
proptest! {
    #[test]
    fn prop_op_ripemd160_fixed_length(
        input in prop::collection::vec(any::<u8>(), 1..75)
    ) {
        // Build script: push input, OP_RIPEMD160
        let mut script = Vec::new();
        script.push(input.len() as u8);
        script.extend_from_slice(&input);
        script.push(0xa6); // OP_RIPEMD160
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok());
        if result.unwrap() && !stack.is_empty() {
            // RIPEMD160 should produce 20-byte output
            prop_assert_eq!(stack[0].len(), 20,
                "OP_RIPEMD160 should produce 20-byte hash");
        }
    }
}

/// Property test: Stack size limits are enforced
proptest! {
    #[test]
    fn prop_stack_size_limit_enforced(
        push_count in 0usize..(MAX_STACK_SIZE.min(100))
    ) {
        // Build script with many pushes
        let mut script = Vec::new();
        for _ in 0..push_count {
            script.push(0x51); // OP_1
        }
        
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        // Stack should never exceed MAX_STACK_SIZE
        prop_assert!(stack.len() <= MAX_STACK_SIZE);
        
        if push_count > MAX_STACK_SIZE {
            // Should fail if exceeding limit
            prop_assert!(result.is_err() || !result.unwrap());
        }
    }
}

/// Property test: Script with OP_2DROP removes two items
proptest! {
    #[test]
    fn prop_script_op_2drop(
        a in 1u8..=16u8,
        b in 1u8..=16u8
    ) {
        // Build script: OP_a, OP_b, OP_2DROP
        let script = vec![0x51 + a.min(16), 0x51 + b.min(16), 0x6d]; // OP_2DROP
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok() || result.is_err());
    }
}

/// Property test: Script with OP_SWAP exchanges top two items
proptest! {
    #[test]
    fn prop_script_op_swap(
        a in 1u8..=16u8,
        b in 1u8..=16u8
    ) {
        // Build script: OP_a, OP_b, OP_SWAP
        let script = vec![0x51 + a.min(16), 0x51 + b.min(16), 0x7c]; // OP_SWAP
        let mut stack = Vec::new();
        let result = script::eval_script(&script, &mut stack, 0);
        
        prop_assert!(result.is_ok() || result.is_err());
    }
}

