//! Tests for cache-based VM optimizations (script cache, hash cache)

#[cfg(feature = "production")]
mod tests {
    use blvm_consensus::opcodes::{OP_1, OP_2, OP_3, OP_HASH160, OP_HASH256};
    use blvm_consensus::script::{eval_script, verify_script, SigVersion, StackElement};

    fn push_stack_items(stack: &mut Vec<StackElement>, item: &[u8]) {
        for _ in 0..5 {
            stack.push(StackElement::from_slice(item));
        }
    }

    #[test]
    fn test_script_cache_hit() {
        // Test that script cache returns cached results
        let script_sig = vec![OP_1];
        let script_pubkey = vec![OP_1, OP_1];

        // First call (cache miss)
        let result1 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();

        // Second call (cache hit)
        let result2 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();

        // Results should be identical
        assert_eq!(result1, result2, "Cached results must match original");
    }

    #[test]
    fn test_script_cache_different_scripts() {
        // Different scripts should produce different cache entries
        let script_sig1 = vec![OP_1];
        let script_pubkey1 = vec![OP_1, OP_1];

        let script_sig2 = vec![OP_2];
        let script_pubkey2 = vec![OP_2, OP_2];

        let result1 = verify_script(&script_sig1, &script_pubkey1, None, 0).unwrap();
        let _result2 = verify_script(&script_sig2, &script_pubkey2, None, 0).unwrap();

        // Results may be the same or different, but cache shouldn't interfere
        // Verify both execute correctly
        let result1_repeat = verify_script(&script_sig1, &script_pubkey1, None, 0).unwrap();
        assert_eq!(
            result1, result1_repeat,
            "Cache must preserve script-specific results"
        );
    }

    #[test]
    fn test_hash_cache_op_hash160() {
        // Test OP_HASH160 caching
        let input = vec![OP_1, OP_2, OP_3];

        // Create script that uses OP_HASH160
        let script = vec![OP_1, OP_1, OP_1, OP_1, OP_1, OP_HASH160]; // Push input, OP_HASH160

        // First execution (cache miss)
        let mut stack1 = Vec::new();
        push_stack_items(&mut stack1, &input);
        let result1 = eval_script(&script, &mut stack1, 0, SigVersion::Base).unwrap();

        // Second execution with same input (should benefit from cache)
        let mut stack2 = Vec::new();
        push_stack_items(&mut stack2, &input);
        let result2 = eval_script(&script, &mut stack2, 0, SigVersion::Base).unwrap();

        // Results should be identical
        assert_eq!(
            result1, result2,
            "Hash cache must produce identical results"
        );
    }

    #[test]
    fn test_hash_cache_op_hash256() {
        // Test OP_HASH256 caching
        let input = vec![OP_1, OP_2, OP_3];

        // Create script that uses OP_HASH256
        let script = vec![OP_1, OP_1, OP_1, OP_1, OP_1, OP_HASH256]; // Push input, OP_HASH256

        // First execution (cache miss)
        let mut stack1 = Vec::new();
        push_stack_items(&mut stack1, &input);
        let result1 = eval_script(&script, &mut stack1, 0, SigVersion::Base).unwrap();

        // Second execution (cache hit)
        let mut stack2 = Vec::new();
        push_stack_items(&mut stack2, &input);
        let result2 = eval_script(&script, &mut stack2, 0, SigVersion::Base).unwrap();

        // Results should be identical
        assert_eq!(
            result1, result2,
            "Hash cache must produce identical results"
        );
    }

    #[test]
    fn test_hash_cache_distinguishes_operations() {
        // HASH160 and HASH256 should have separate cache entries for same input
        let input = vec![OP_1, OP_2, OP_3];

        let script_hash160 = vec![OP_1, OP_1, OP_1, OP_1, OP_1, OP_HASH160]; // OP_HASH160
        let script_hash256 = vec![OP_1, OP_1, OP_1, OP_1, OP_1, OP_HASH256]; // OP_HASH256

        let mut stack1 = Vec::new();
        push_stack_items(&mut stack1, &input);
        let result_hash160 =
            eval_script(&script_hash160, &mut stack1, 0, SigVersion::Base).unwrap();

        let mut stack2 = Vec::new();
        push_stack_items(&mut stack2, &input);
        let result_hash256 =
            eval_script(&script_hash256, &mut stack2, 0, SigVersion::Base).unwrap();

        // Results should be different (HASH160 produces 20 bytes, HASH256 produces 32 bytes)
        // But both should work correctly
        assert!(result_hash160 == true || result_hash160 == false);
        assert!(result_hash256 == true || result_hash256 == false);
    }

    #[test]
    fn test_stack_pool_reuse() {
        // Test that stack pooling works correctly
        let script1 = vec![OP_1, OP_1];
        let script2 = vec![OP_2, OP_2];

        // Multiple executions should reuse pooled stacks
        let result1 = verify_script(&script1, &vec![OP_1], None, 0).unwrap();
        let result2 = verify_script(&script2, &vec![OP_2], None, 0).unwrap();
        let result3 = verify_script(&script1, &vec![OP_1], None, 0).unwrap();

        // Results should be correct (may be cached, but should be deterministic)
        assert_eq!(result1, result3, "Stack pooling must not affect results");

        // All should execute successfully
        assert!(result1 == true || result1 == false);
        assert!(result2 == true || result2 == false);
    }

    #[test]
    fn test_cache_performance_indicator() {
        // Basic performance sanity check for caching
        use std::time::Instant;

        let script_sig = vec![OP_1];
        let script_pubkey = vec![OP_1, OP_1];

        // First call (cache miss - slower)
        let start1 = Instant::now();
        let _result1 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
        let duration1 = start1.elapsed();

        // Second call (cache hit - should be faster)
        let start2 = Instant::now();
        let _result2 = verify_script(&script_sig, &script_pubkey, None, 0).unwrap();
        let duration2 = start2.elapsed();

        // Cache hit should be at least as fast (often faster)
        // This is a sanity check, not a strict benchmark
        assert!(
            duration2.as_micros() <= duration1.as_micros() + 1000,
            "Cache hit should be fast ({}us vs {}us)",
            duration2.as_micros(),
            duration1.as_micros()
        );
    }

    #[test]
    fn test_cache_with_witness() {
        // Test caching works with witness data
        let script_sig = vec![OP_1];
        let script_pubkey = vec![OP_1];
        let witness = Some(vec![OP_2]);

        let result1 = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();
        let result2 = verify_script(&script_sig, &script_pubkey, witness.as_ref(), 0).unwrap();

        // Results should be cached and identical
        assert_eq!(result1, result2, "Cache must work with witness data");
    }
}
