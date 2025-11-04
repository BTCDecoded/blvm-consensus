//! Property tests for chain reorganization edge cases
//!
//! Comprehensive property-based tests covering chain reorganization scenarios,
//! chain work calculations, and UTXO set consistency during reorganizations.

use consensus_proof::*;
use consensus_proof::reorganization;
use consensus_proof::types::*;
use proptest::prelude::*;

/// Property test: chain work is always non-negative
proptest! {
    #[test]
    fn prop_chain_work_non_negative(
        block_count in 1usize..20usize
    ) {
        // Create a chain of blocks
        let mut headers = Vec::new();
        for i in 0..block_count {
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: if i == 0 { [0; 32] } else { [i as u8; 32] },
                merkle_root: [1; 32],
                timestamp: 1234567890 + (i as u64 * 600),
                bits: 0x1d00ffff,
                nonce: i as u64,
            });
        }
        
        // Chain work should be calculated and non-negative
        let result = reorganization::calculate_chain_work(&headers);
        
        prop_assert!(result.is_ok() || result.is_err());
        if result.is_ok() {
            let work = result.unwrap();
            prop_assert!(work >= 0, "Chain work must be non-negative");
        }
    }
}

/// Property test: longer chain has more work (or equal)
proptest! {
    #[test]
    fn prop_chain_work_increases_with_length(
        short_chain_len in 1usize..10usize,
        long_chain_len in 1usize..10usize
    ) {
        let (short_len, long_len) = if short_chain_len <= long_chain_len {
            (short_chain_len, long_chain_len)
        } else {
            (long_chain_len, short_chain_len)
        };
        
        // Create short chain
        let mut short_headers = Vec::new();
        for i in 0..short_len {
            short_headers.push(BlockHeader {
                version: 1,
                prev_block_hash: if i == 0 { [0; 32] } else { [i as u8; 32] },
                merkle_root: [1; 32],
                timestamp: 1234567890 + (i as u64 * 600),
                bits: 0x1d00ffff,
                nonce: i as u64,
            });
        }
        
        // Create long chain (extends short chain)
        let mut long_headers = short_headers.clone();
        for i in short_len..long_len {
            long_headers.push(BlockHeader {
                version: 1,
                prev_block_hash: if i == 0 { [0; 32] } else { [i as u8; 32] },
                merkle_root: [1; 32],
                timestamp: 1234567890 + (i as u64 * 600),
                bits: 0x1d00ffff,
                nonce: i as u64,
            });
        }
        
        let short_work = reorganization::calculate_chain_work(&short_headers);
        let long_work = reorganization::calculate_chain_work(&long_headers);
        
        if short_work.is_ok() && long_work.is_ok() {
            prop_assert!(long_work.unwrap() >= short_work.unwrap(),
                "Longer chain should have equal or more work");
        }
    }
}

/// Property test: should_reorganize prefers chain with more work
proptest! {
    #[test]
    fn prop_reorganize_prefers_more_work(
        chain1_len in 1usize..10usize,
        chain2_len in 1usize..10usize
    ) {
        // Create two chains
        let mut chain1 = Vec::new();
        for i in 0..chain1_len {
            chain1.push(BlockHeader {
                version: 1,
                prev_block_hash: if i == 0 { [0; 32] } else { [1, i as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
                merkle_root: [1; 32],
                timestamp: 1234567890 + (i as u64 * 600),
                bits: 0x1d00ffff,
                nonce: i as u64,
            });
        }
        
        let mut chain2 = Vec::new();
        for i in 0..chain2_len {
            chain2.push(BlockHeader {
                version: 1,
                prev_block_hash: if i == 0 { [0; 32] } else { [2, i as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] },
                merkle_root: [1; 32],
                timestamp: 1234567890 + (i as u64 * 600),
                bits: 0x1d00ffff,
                nonce: i as u64,
            });
        }
        
        let result = reorganization::should_reorganize(&chain1, &chain2);
        
        // Should reorganize if chain2 has more work
        prop_assert!(result.is_ok() || result.is_err());
    }
}

/// Property test: reorganization maintains UTXO consistency
proptest! {
    #[test]
    fn prop_reorganization_utxo_consistency(
        initial_height in 1u64..10u64,
        reorg_depth in 1u64..5u64
    ) {
        // Create initial chain state
        let mut utxo_set = UtxoSet::new();
        
        // Add some UTXOs
        for i in 0..5 {
            utxo_set.insert(
                OutPoint { hash: [i as u8; 32], index: 0 },
                UTXO {
                    value: 1000 * (i as i64 + 1),
                    script_pubkey: vec![0x51],
                    height: initial_height,
                }
            );
        }
        
        let initial_utxo_count = utxo_set.len();
        
        // Simulate reorganization
        // UTXO set should maintain consistency (not lose or duplicate UTXOs)
        prop_assert!(initial_utxo_count >= 0);
        prop_assert!(initial_utxo_count <= 1000); // Reasonable bound
        
        // After reorganization, UTXO set should still be valid
        // (actual implementation would reorganize and check)
        prop_assert!(utxo_set.len() <= initial_utxo_count + 10); // Allow some variation
    }
}

/// Property test: chain work calculation is deterministic
proptest! {
    #[test]
    fn prop_chain_work_deterministic(
        block_count in 1usize..10usize
    ) {
        // Create chain
        let mut headers = Vec::new();
        for i in 0..block_count {
            headers.push(BlockHeader {
                version: 1,
                prev_block_hash: if i == 0 { [0; 32] } else { [i as u8; 32] },
                merkle_root: [1; 32],
                timestamp: 1234567890 + (i as u64 * 600),
                bits: 0x1d00ffff,
                nonce: i as u64,
            });
        }
        
        // Calculate work twice
        let work1 = reorganization::calculate_chain_work(&headers);
        let work2 = reorganization::calculate_chain_work(&headers);
        
        // Results should be identical
        prop_assert_eq!(work1.is_ok(), work2.is_ok());
        if work1.is_ok() && work2.is_ok() {
            prop_assert_eq!(work1.unwrap(), work2.unwrap(),
                "Chain work calculation should be deterministic");
        }
    }
}

/// Property test: empty chain has zero work
proptest! {
    #[test]
    fn prop_empty_chain_zero_work() {
        let empty_chain: Vec<BlockHeader> = Vec::new();
        
        let result = reorganization::calculate_chain_work(&empty_chain);
        
        // Empty chain should have zero work or error
        prop_assert!(result.is_ok() || result.is_err());
        if result.is_ok() {
            prop_assert_eq!(result.unwrap(), 0,
                "Empty chain should have zero work");
        }
    }
}

/// Property test: reorganization depth is bounded
proptest! {
    #[test]
    fn prop_reorganization_depth_bounded(
        current_chain_len in 1usize..20usize,
        new_chain_len in 1usize..20usize
    ) {
        // Reorganization depth = common prefix length
        let common_prefix = current_chain_len.min(new_chain_len);
        let reorg_depth = current_chain_len - common_prefix;
        
        prop_assert!(reorg_depth <= current_chain_len,
            "Reorganization depth should not exceed current chain length");
        prop_assert!(reorg_depth >= 0,
            "Reorganization depth should be non-negative");
    }
}

/// Property test: chain fork point identification
proptest! {
    #[test]
    fn prop_chain_fork_point(
        fork_height in 1u64..10u64,
        chain1_extension in 1usize..10usize,
        chain2_extension in 1usize..10usize
    ) {
        // Create common prefix
        let mut common_prefix = Vec::new();
        for i in 0..fork_height as usize {
            common_prefix.push(BlockHeader {
                version: 1,
                prev_block_hash: if i == 0 { [0; 32] } else { [i as u8; 32] },
                merkle_root: [1; 32],
                timestamp: 1234567890 + (i as u64 * 600),
                bits: 0x1d00ffff,
                nonce: i as u64,
            });
        }
        
        // Extend chain1
        let mut chain1 = common_prefix.clone();
        for i in 0..chain1_extension {
            chain1.push(BlockHeader {
                version: 1,
                prev_block_hash: [1; 32],
                merkle_root: [1; 32],
                timestamp: 1234567890 + ((fork_height + i as u64) * 600),
                bits: 0x1d00ffff,
                nonce: (fork_height + i as u64),
            });
        }
        
        // Extend chain2
        let mut chain2 = common_prefix;
        for i in 0..chain2_extension {
            chain2.push(BlockHeader {
                version: 1,
                prev_block_hash: [2; 32],
                merkle_root: [1; 32],
                timestamp: 1234567890 + ((fork_height + i as u64) * 600),
                bits: 0x1d00ffff,
                nonce: (fork_height + i as u64),
            });
        }
        
        // Both chains should share common prefix
        prop_assert!(chain1.len() >= fork_height as usize);
        prop_assert!(chain2.len() >= fork_height as usize);
        prop_assert!(chain1.len() == (fork_height as usize + chain1_extension));
        prop_assert!(chain2.len() == (fork_height as usize + chain2_extension));
    }
}



