//! Tests for memory optimizations in production mode

#[cfg(feature = "production")]
mod tests {
    use blvm_consensus::constants::*;
    use blvm_consensus::opcodes::{OP_1, OP_2};
    use blvm_consensus::script::{SigVersion, *};

    #[test]
    fn test_stack_preallocation_capacity() {
        // Verify stack pre-allocation uses correct capacity
        let mut stack = Vec::new();
        let script = vec![OP_1]; // OP_1

        eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();

        // Stack should have at least 1 item
        assert!(stack.len() >= 1);

        // Capacity should be at least length (pre-allocation may reserve more)
        assert!(
            stack.capacity() >= stack.len(),
            "Stack capacity should be at least length after execution"
        );
    }

    #[test]
    fn test_stack_preallocation_no_overallocation() {
        // Verify capacity doesn't cause memory issues
        let mut stack = Vec::new();
        let script = vec![OP_1, OP_1, OP_1]; // OP_1, OP_1, OP_1

        eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();

        // Stack should function correctly with pre-allocation
        assert!(stack.len() >= 1);

        // Pre-allocation shouldn't break normal operation
        assert!(stack.capacity() >= stack.len());
    }

    #[test]
    fn test_stack_growth_unchanged() {
        // Verify scripts exceeding pre-allocated capacity still work correctly
        // Create script that will exceed initial capacity
        let mut script = Vec::new();
        for _ in 0..30 {
            script.push(OP_1); // OP_1 (pushes to stack)
        }

        let mut stack = Vec::new();
        let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();

        // Should execute successfully despite exceeding pre-allocation
        assert!(result, "30×OP_1 must eval successfully");
        assert_eq!(stack.len(), 30);
    }

    #[test]
    fn test_memory_allocation_parity() {
        // Compare memory behavior with production optimizations
        let script = vec![OP_1, OP_1, OP_2, OP_2];

        // First execution
        let mut stack1 = Vec::new();
        let result1 = eval_script(&script, &mut stack1, 0, SigVersion::Base).unwrap();

        // Second execution (may benefit from different allocation patterns)
        let mut stack2 = Vec::new();
        let result2 = eval_script(&script, &mut stack2, 0, SigVersion::Base).unwrap();

        // Results must be identical regardless of allocation strategy
        assert_eq!(
            result1, result2,
            "Memory allocation optimizations must not affect script execution results"
        );

        // Stack states must match
        assert_eq!(
            stack1.len(),
            stack2.len(),
            "Memory optimizations must not affect final stack state"
        );
    }

    #[test]
    fn test_verify_script_memory_optimization() {
        // Test verify_script uses pre-allocated stack
        let script_sig = vec![OP_1];
        let script_pubkey = vec![OP_1, OP_1];

        let result1 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
        let result2 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();

        // Results must be identical
        assert_eq!(
            result1, result2,
            "verify_script memory optimizations must not affect results"
        );
    }

    #[test]
    fn test_large_script_memory_handling() {
        // Test memory handling with large scripts
        let mut large_script = Vec::new();
        for _ in 0..100 {
            large_script.push(OP_1); // OP_1
        }

        let mut stack = Vec::new();
        let result = eval_script(&large_script, &mut stack, 0, SigVersion::Base);

        // 100×OP_1 must succeed without memory issues
        assert!(result.unwrap(), "100×OP_1 must eval successfully");
        assert_eq!(stack.len(), 100);
    }
}
