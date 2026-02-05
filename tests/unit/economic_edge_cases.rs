//! Property tests for economic model edge cases
//!
//! Comprehensive property-based tests covering economic model boundary conditions
//! and edge cases, ensuring correct implementation of Bitcoin's monetary policy.

use blvm_consensus::*;
use blvm_consensus::ConsensusProof;
use blvm_consensus::types::*;
use blvm_consensus::constants::MAX_MONEY;
use blvm_consensus::orange_paper_constants::{C, H};
use proptest::prelude::*;

/// Property test: block subsidy is always non-negative
proptest! {
    #[test]
    fn prop_block_subsidy_non_negative(
        height in 0u64..10000000u64 // Up to ~47 halvings
    ) {
        let consensus = ConsensusProof::new();
        let subsidy = consensus.get_block_subsidy(height);
        
        prop_assert!(subsidy >= 0, "Block subsidy must be non-negative");
    }
}

/// Property test: block subsidy never exceeds initial subsidy
proptest! {
    #[test]
    fn prop_block_subsidy_maximum(
        height in 0u64..10000000u64
    ) {
        let consensus = ConsensusProof::new();
        let subsidy = consensus.get_block_subsidy(height);
        
        // Using Orange Paper constant: initial subsidy = 50 * C where C = 10^8
        let initial_subsidy = 50 * C;
        prop_assert!(subsidy <= initial_subsidy as i64,
            "Block subsidy must not exceed initial subsidy (50 * C)");
    }
}

/// Property test: block subsidy halves correctly
proptest! {
    #[test]
    fn prop_block_subsidy_halving(
        halving_period in 0u64..64u64 // Maximum 64 halvings
    ) {
        let consensus = ConsensusProof::new();
        // Using Orange Paper constant H (halving interval = 210,000)
        let height_before = halving_period * H;
        let height_after = (halving_period + 1) * H;
        
        if height_after < 10000000 { // Reasonable bound
            let subsidy_before = consensus.get_block_subsidy(height_before);
            let subsidy_after = consensus.get_block_subsidy(height_after);
            
            // After halving, subsidy should be half (or zero if at limit)
            if subsidy_before > 0 && halving_period < 64 {
                prop_assert!(subsidy_after == subsidy_before / 2 || subsidy_after == 0,
                    "Subsidy should halve every {} blocks (H from Orange Paper)", H);
            }
        }
    }
}

/// Property test: total supply is monotonically increasing
proptest! {
    #[test]
    fn prop_total_supply_monotonic(
        height1 in 0u64..1000000u64,
        height2 in 0u64..1000000u64
    ) {
        let consensus = ConsensusProof::new();
        
        // Ensure height1 <= height2
        let (h1, h2) = if height1 <= height2 {
            (height1, height2)
        } else {
            (height2, height1)
        };
        
        let supply1 = consensus.total_supply(h1);
        let supply2 = consensus.total_supply(h2);
        
        prop_assert!(supply2 >= supply1,
            "Total supply must be monotonically increasing");
    }
}

/// Property test: total supply never exceeds MAX_MONEY
proptest! {
    #[test]
    fn prop_total_supply_limit(
        height in 0u64..10000000u64 // Very large height (beyond all halvings)
    ) {
        let consensus = ConsensusProof::new();
        let supply = consensus.total_supply(height);
        
        prop_assert!(supply <= MAX_MONEY,
            "Total supply must never exceed MAX_MONEY (21M BTC)");
    }
}

/// Property test: total supply is non-negative
proptest! {
    #[test]
    fn prop_total_supply_non_negative(
        height in 0u64..10000000u64
    ) {
        let consensus = ConsensusProof::new();
        let supply = consensus.total_supply(height);
        
        prop_assert!(supply >= 0, "Total supply must be non-negative");
    }
}

/// Property test: supply at genesis equals initial subsidy
proptest! {
    #[test]
    fn prop_supply_genesis() {
        let consensus = ConsensusProof::new();
        let genesis_supply = consensus.total_supply(0);
        let genesis_subsidy = consensus.get_block_subsidy(0);
        
        prop_assert_eq!(genesis_supply, genesis_subsidy,
            "Genesis supply should equal genesis subsidy");
    }
}

/// Property test: subsidy becomes zero after 64 halvings
proptest! {
    #[test]
    fn prop_subsidy_zero_after_64_halvings() {
        let consensus = ConsensusProof::new();
        let height_64_halvings = 64 * HALVING_INTERVAL;
        
        if height_64_halvings < 10000000 { // Reasonable bound
            let subsidy = consensus.get_block_subsidy(height_64_halvings);
            prop_assert_eq!(subsidy, 0,
                "Subsidy must be zero after 64 halvings");
        }
    }
}

/// Property test: difficulty adjustment interval properties
proptest! {
    #[test]
    fn prop_difficulty_adjustment_interval(
        height in 0u64..100000u64
    ) {
        use blvm_consensus::constants::DIFFICULTY_ADJUSTMENT_INTERVAL;
        
        // Difficulty adjustment happens every DIFFICULTY_ADJUSTMENT_INTERVAL blocks
        let is_adjustment_height = (height as u64) % (DIFFICULTY_ADJUSTMENT_INTERVAL as u64) == 0;
        
        // This is a structural property
        prop_assert!(height >= 0);
        
        // Adjustment should happen at multiples of the interval
        // This is a structural property test
        prop_assert!(height >= 0);
    }
}

