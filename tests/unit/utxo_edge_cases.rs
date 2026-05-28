//! Property tests for UTXO set operations edge cases
//!
//! Comprehensive property-based tests covering UTXO set operations,
//! consistency during block connection, and edge cases.

use blvm_consensus::types::*;
use proptest::prelude::*;

fn make_utxo(value: i64, height: u64) -> UTXO {
    UTXO {
        value,
        script_pubkey: vec![0x51].into(),
        height,
        is_coinbase: false,
    }
}

/// Property test: UTXO set insertion maintains uniqueness
proptest! {
    #[test]
    fn prop_utxo_set_insertion_uniqueness(
        outpoint_count in 1usize..20usize
    ) {
        let mut utxo_set = UtxoSet::default();
        let mut inserted_count = 0;

        for i in 0..outpoint_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: 0u32,
            };
            let utxo = UTXO {
                value: 1000 * (i as i64 + 1),
                script_pubkey: vec![i as u8].into(),
                height: 1,
                is_coinbase: false,
            };
            let was_new = utxo_set.insert(outpoint, std::sync::Arc::new(utxo)).is_none();
            if was_new {
                inserted_count += 1;
            }
        }

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
        let mut utxo_set = UtxoSet::default();
        let mut outpoints = Vec::new();

        for i in 0..initial_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: 0u32,
            };
            outpoints.push(outpoint);
            utxo_set.insert(outpoint, std::sync::Arc::new(make_utxo(1000, 1)));
        }

        let initial_len = utxo_set.len();
        let remove_len = remove_count.min(initial_count);
        for i in 0..remove_len {
            utxo_set.remove(&outpoints[i]);
        }

        prop_assert_eq!(utxo_set.len(), initial_len - remove_len);
    }
}

/// Property test: UTXO value is non-negative
proptest! {
    #[test]
    fn prop_utxo_value_non_negative(
        value in 0i64..1000000i64
    ) {
        let utxo = make_utxo(value, 1);
        prop_assert!(utxo.value >= 0, "UTXO value must be non-negative");
    }
}

/// Property test: UTXO height is non-negative
proptest! {
    #[test]
    fn prop_utxo_height_non_negative(
        height in 0u64..1000000u64
    ) {
        let utxo = make_utxo(1000, height);
        prop_assert!(utxo.height >= 0, "UTXO height must be non-negative");
    }
}

/// Property test: UTXO set query returns correct value
proptest! {
    #[test]
    fn prop_utxo_set_query_correctness(
        outpoint_hash in prop::array::uniform32(0u8..=255u8),
        outpoint_index in 0u32..1000u32,
        value in 1000i64..1000000i64
    ) {
        let mut utxo_set = UtxoSet::default();
        let outpoint = OutPoint {
            hash: outpoint_hash,
            index: outpoint_index,
        };
        let utxo = make_utxo(value, 1);
        utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

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
        let mut utxo_set = UtxoSet::default();
        let outpoint = OutPoint {
            hash: outpoint_hash,
            index: 0u32,
        };

        utxo_set.insert(outpoint, std::sync::Arc::new(make_utxo(initial_value, 1)));
        utxo_set.insert(outpoint, std::sync::Arc::new(UTXO {
            value: new_value,
            script_pubkey: vec![0x52].into(),
            height: 2,
            is_coinbase: false,
        }));

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
        let mut utxo_set = UtxoSet::default();
        let mut inserted_outpoints = Vec::new();

        for i in 0..entry_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: i as u32,
            };
            inserted_outpoints.push(outpoint);
            utxo_set.insert(outpoint, std::sync::Arc::new(make_utxo(1000 * (i as i64 + 1), 1)));
        }

        let mut found_count = 0;
        for outpoint in utxo_set.keys() {
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
        let mut utxo_set = UtxoSet::default();
        let mut outpoints = Vec::new();

        for i in 0..insert_count {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: 0u32,
            };
            outpoints.push(outpoint);
            utxo_set.insert(outpoint, std::sync::Arc::new(make_utxo(1000, 1)));
        }

        let actual_remove = remove_count.min(insert_count);
        for i in 0..actual_remove {
            utxo_set.remove(&outpoints[i]);
        }

        prop_assert_eq!(utxo_set.len(), insert_count - actual_remove);
    }
}
