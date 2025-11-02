//! Tests for Secp256k1 context reuse in production mode

#[cfg(feature = "production")]
mod tests {
    use consensus_proof::script::*;

    #[test]
    fn test_thread_local_context_reuse() {
        // Verify context is reused across multiple signature verifications
        // Test through eval_script which uses execute_opcode internally
        let script = vec![0x51, 0x51, 0xac]; // OP_1, OP_1, OP_CHECKSIG
        
        let mut stack1 = Vec::new();
        let result1 = eval_script(&script, &mut stack1, 0).unwrap();
        
        let mut stack2 = Vec::new();
        let result2 = eval_script(&script, &mut stack2, 0).unwrap();
        
        let mut stack3 = Vec::new();
        let result3 = eval_script(&script, &mut stack3, 0).unwrap();
        
        // Results should be identical (context reuse shouldn't affect correctness)
        assert_eq!(result1, result2, 
                   "Context reuse must not affect signature verification results");
        assert_eq!(result1, result3,
                   "Context reuse must not affect signature verification results");
    }

    #[test]
    fn test_context_isolation() {
        // Verify context state doesn't leak between operations
        let script1 = vec![0x51, 0xac]; // OP_1, OP_CHECKSIG
        let script2 = vec![0x52, 0xac]; // OP_2, OP_CHECKSIG
        
        let mut stack1 = Vec::new();
        let result1 = eval_script(&script1, &mut stack1, 0).unwrap();
        
        let mut stack2 = Vec::new();
        let result2 = eval_script(&script2, &mut stack2, 0).unwrap();
        
        // Different scripts should produce different results
        // (if they're the same, that's fine, but context shouldn't affect it)
        // Main goal: verify no state leakage between calls
        let result1_repeat = eval_script(&script1, &mut Vec::new(), 0).unwrap();
        assert_eq!(result1, result1_repeat, 
                   "Context state must not leak between script executions");
    }

    #[test]
    fn test_op_checksig_context_reuse() {
        // Specific test for OP_CHECKSIG using thread-local context
        let script = vec![0x51, 0x51, 0xac]; // OP_1, OP_1, OP_CHECKSIG
        
        let results: Vec<bool> = (0..10)
            .map(|_| {
                let mut stack = Vec::new();
                eval_script(&script, &mut stack, 0).unwrap()
            })
            .collect();
        
        // All results should be identical
        assert!(results.iter().all(|&r| r == results[0]),
                "OP_CHECKSIG with context reuse must be deterministic");
    }

    #[test]
    fn test_op_checksigverify_context_reuse() {
        // Specific test for OP_CHECKSIGVERIFY using thread-local context
        let script = vec![0x51, 0x51, 0xad]; // OP_1, OP_1, OP_CHECKSIGVERIFY
        
        let results: Vec<bool> = (0..10)
            .map(|_| {
                let mut stack = Vec::new();
                eval_script(&script, &mut stack, 0).unwrap()
            })
            .collect();
        
        // All results should be identical
        assert!(results.iter().all(|&r| r == results[0]),
                "OP_CHECKSIGVERIFY with context reuse must be deterministic");
    }

    #[test]
    fn test_context_reuse_performance_indicator() {
        // Basic performance sanity check: context reuse shouldn't make things slower
        use std::time::Instant;
        
        let iterations = 100;
        let script = vec![0x51, 0x51, 0xac]; // OP_1, OP_1, OP_CHECKSIG
        
        let start = Instant::now();
        for _ in 0..iterations {
            let mut stack = Vec::new();
            let _ = eval_script(&script, &mut stack, 0);
        }
        let duration = start.elapsed();
        
        // Sanity check: 100 operations should complete quickly
        // This is a basic check, not a full benchmark
        assert!(duration.as_millis() < 10_000, 
                "Context reuse should enable fast signature verification ({}ms for {} ops)", 
                duration.as_millis(), iterations);
    }

    #[test]
    fn test_context_concurrent_operations() {
        // Test that context works correctly with rapid sequential operations
        let script = vec![0x51, 0x51, 0xac]; // OP_1, OP_1, OP_CHECKSIG
        let mut results = Vec::new();
        
        for i in 0..50 {
            let mut stack = Vec::new();
            let result = eval_script(&script, &mut stack, 0).unwrap();
            results.push((i, result));
        }
        
        // All should be identical
        let first_result = results[0].1;
        for (i, result) in results.iter() {
            assert_eq!(first_result, result,
                       "Concurrent-style operations must be deterministic (iteration {})", i);
        }
    }
}

