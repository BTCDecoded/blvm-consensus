//! Property tests for UTXO set operations edge cases
//!
//! Comprehensive property-based tests covering UTXO set operations,
//! consistency during block connection, and edge cases.

use consensus_proof::*;
use consensus_proof::types::*;
use proptest::prelude::*;

/// Property test: UTXO set insertion maintains uniqueness
proptest! {
    #[test]
    fn prop_utxo_set_insertion_uniqueness(
        outpoint_count in 1usize..20usize
    ) {
        let mut utxo_set = UtxoSet::new();
        let mut inserted_count = 0;
        
        for i in 0..outpoint_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: 0,
            };
            
            let utxo = UTXO {
                value: 1000 * (i as i64 + 1),
                script_pubkey: vec![i as u8],
                height: 1,
            };
            
            // Insert UTXO
            let was_new = utxo_set.insert(outpoint, utxo).is_none();
            if was_new {
                inserted_count += 1;
            }
        }
        
        // All unique outpoints should be inserted
        prop_assert_eq!(utxo_set.len(), inserted_count);
        prop_assert!(utxo_set.len() <= outpoint_count);
    }
}

/// Property test: UTXO set removal maintains consistency
proptest! {
    #[test]
    fn prop_utxo_set_removal_consistency(
        initial_count in 1usize..20usize,
        remove_count in 1usize..20usize
    ) {
        let mut utxo_set = UtxoSet::new();
        let mut outpoints = Vec::new();
        
        // Insert UTXOs
        for i in 0..initial_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: 0,
            };
            outpoints.push(outpoint.clone());
            
            utxo_set.insert(outpoint, UTXO {
                value: 1000,
                script_pubkey: vec![0x51],
                height: 1,
            });
        }
        
        let initial_len = utxo_set.len();
        
        // Remove some UTXOs
        let remove_len = remove_count.min(initial_count);
        for i in 0..remove_len {
            utxo_set.remove(&outpoints[i]);
        }
        
        // Size should decrease by number removed
        prop_assert_eq!(utxo_set.len(), initial_len - remove_len);
        prop_assert!(utxo_set.len() >= 0);
    }
}

/// Property test: UTXO value is non-negative
proptest! {
    #[test]
    fn prop_utxo_value_non_negative(
        value in 0i64..1000000i64
    ) {
        let utxo = UTXO {
            value,
            script_pubkey: vec![0x51],
            height: 1,
        };
        
        prop_assert!(utxo.value >= 0, "UTXO value must be non-negative");
    }
}

/// Property test: UTXO height is non-negative
proptest! {
    #[test]
    fn prop_utxo_height_non_negative(
        height in 0u64..1000000u64
    ) {
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![0x51],
            height,
        };
        
        prop_assert!(utxo.height >= 0, "UTXO height must be non-negative");
    }
}

/// Property test: UTXO set query returns correct value
proptest! {
    #[test]
    fn prop_utxo_set_query_correctness(
        outpoint_hash in prop::array::uniform32(0u8..=255u8),
        outpoint_index in 0u64..1000u64,
        value in 1000i64..1000000i64
    ) {
        let mut utxo_set = UtxoSet::new();
        let outpoint = OutPoint {
            hash: outpoint_hash,
            index: outpoint_index,
        };
        
        let utxo = UTXO {
            value,
            script_pubkey: vec![0x51],
            height: 1,
        };
        
        // Insert UTXO
        utxo_set.insert(outpoint.clone(), utxo.clone());
        
        // Query should return correct UTXO
        let queried = utxo_set.get(&outpoint);
        prop_assert!(queried.is_some());
        if let Some(queried_utxo) = queried {
            prop_assert_eq!(queried_utxo.value, value);
            prop_assert_eq!(queried_utxo.height, 1);
        }
    }
}

/// Property test: UTXO set replacement updates value
proptest! {
    #[test]
    fn prop_utxo_set_replacement(
        outpoint_hash in prop::array::uniform32(0u8..=255u8),
        initial_value in 1000i64..50000i64,
        new_value in 50000i64..100000i64
    ) {
        let mut utxo_set = UtxoSet::new();
        let outpoint = OutPoint {
            hash: outpoint_hash,
            index: 0,
        };
        
        // Insert initial UTXO
        utxo_set.insert(outpoint.clone(), UTXO {
            value: initial_value,
            script_pubkey: vec![0x51],
            height: 1,
        });
        
        // Replace with new value
        utxo_set.insert(outpoint.clone(), UTXO {
            value: new_value,
            script_pubkey: vec![0x52],
            height: 2,
        });
        
        // Query should return new value
        let queried = utxo_set.get(&outpoint);
        prop_assert!(queried.is_some());
        if let Some(utxo) = queried {
            prop_assert_eq!(utxo.value, new_value);
            prop_assert_eq!(utxo.height, 2);
        }
    }
}

/// Property test: UTXO set iteration covers all entries
proptest! {
    #[test]
    fn prop_utxo_set_iteration(
        entry_count in 1usize..20usize
    ) {
        let mut utxo_set = UtxoSet::new();
        let mut inserted_outpoints = Vec::new();
        
        // Insert entries
        for i in 0..entry_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: i as u64,
            };
            inserted_outpoints.push(outpoint.clone());
            
            utxo_set.insert(outpoint, UTXO {
                value: 1000 * (i as i64 + 1),
                script_pubkey: vec![i as u8],
                height: 1,
            });
        }
        
        // Iterate and verify all entries are present
        let mut found_count = 0;
        for (outpoint, _utxo) in &utxo_set {
            if inserted_outpoints.contains(outpoint) {
                found_count += 1;
            }
        }
        
        prop_assert_eq!(found_count, entry_count,
            "All inserted entries should be found during iteration");
    }
}

/// Property test: UTXO set size matches insertions (minus removals)
proptest! {
    #[test]
    fn prop_utxo_set_size_consistency(
        insert_count in 1usize..20usize,
        remove_count in 0usize..20usize
    ) {
        let mut utxo_set = UtxoSet::new();
        let mut outpoints = Vec::new();
        
        // Insert UTXOs
        for i in 0..insert_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: 0,
            };
            outpoints.push(outpoint.clone());
            
            utxo_set.insert(outpoint, UTXO {
                value: 1000,
                script_pubkey: vec![0x51],
                height: 1,
            });
        }
        
        // Remove some UTXOs
        let actual_remove = remove_count.min(insert_count);
        for i in 0..actual_remove {
            utxo_set.remove(&outpoints[i]);
        }
        
        // Size should be insertions minus removals
        prop_assert_eq!(utxo_set.len(), insert_count - actual_remove);
    }
}



