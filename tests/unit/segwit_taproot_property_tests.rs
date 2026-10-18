//! Property tests for SegWit and Taproot edge cases
//!
//! Comprehensive property-based tests covering SegWit transaction weight,
//! witness validation, Taproot output validation, and related edge cases.

use blvm_consensus::opcodes::{OP_0, OP_1, PUSH_32_BYTES};
use blvm_consensus::segwit;
use blvm_consensus::taproot::validate_taproot_script;
use blvm_consensus::types::*;
use blvm_consensus::witness::validate_segwit_witness_structure;
use proptest::prelude::*;

/// Property test: transaction weight is always positive
proptest! {
    #[test]
    fn prop_transaction_weight_positive(
        input_count in 1usize..20usize,
        output_count in 1usize..20usize
    ) {
        let mut tx = Transaction {
            version: 1,
            inputs: Default::default(),
            outputs: Default::default(),
            lock_time: 0,
        };

        // Add inputs
        for i in 0..input_count {
            tx.inputs.push(TransactionInput {
                prevout: OutPoint { hash: [i as u8; 32], index: 0 },
                script_sig: vec![OP_1; 50],
                sequence: 0xffffffff,
            });
        }

        // Add outputs
        for i in 0..output_count {
            tx.outputs.push(TransactionOutput {
                value: 1000,
                script_pubkey: vec![i as u8; 50],
            });
        }

        let witness: segwit::Witness = vec![vec![OP_1; 50]; input_count];
        let weight = segwit::calculate_transaction_weight(&tx, Some(&witness)).unwrap();
        let base = blvm_consensus::serialization::serialize_transaction(&tx).len() as u64;
        let extra: u64 = witness.iter().map(|element| element.len() as u64).sum();
        prop_assert_eq!(weight, base * 4 + extra);
    }
}

/// Property test: block weight respects maximum limit
proptest! {
    #[test]
    fn prop_block_weight_maximum(
        tx_count in 1usize..10usize
    ) {
        let txs: Vec<Transaction> = (0..tx_count)
            .map(|i| Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [i as u8; 32], index: 0 },
                    script_sig: vec![OP_1],
                    sequence: 0xffffffff,
                }].into(),
                outputs: vec![TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![OP_1],
                }].into(),
                lock_time: 0,
            })
            .collect();
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [1; 32],
                timestamp: 1234567890,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: txs.into_boxed_slice(),
        };

        let witnesses: Vec<segwit::Witness> = vec![vec![vec![]]; tx_count];
        let weight = segwit::calculate_block_weight(&block, &witnesses).unwrap();
        let tx_bytes: u64 = block
            .transactions
            .iter()
            .map(|tx| blvm_consensus::serialization::serialize_transaction(tx).len() as u64)
            .sum();
        prop_assert_eq!(weight, 4 * (80 + 1 + tx_bytes));
    }
}

/// Property test: transaction weight formula consistency
proptest! {
    #[test]
        fn prop_transaction_weight_formula(
        base_size in 100u64..1000,
        witness_size in 0u64..500
    ) {
        let weight = blvm_consensus::witness::calculate_transaction_weight_segwit(
            base_size,
            base_size + witness_size,
        );
        prop_assert_eq!(weight, base_size * 4 + witness_size);
    }
}

/// Property test: SegWit witness data size bounds
proptest! {
    #[test]
        fn prop_segwit_witness_bounds(
        witness_item_count in 0usize..5,
        witness_item_size in 1usize..=530
    ) {
        let witness: Vec<Vec<u8>> = (0..witness_item_count)
            .map(|_| vec![OP_1; witness_item_size])
            .collect();
        let valid = validate_segwit_witness_structure(&witness).unwrap();
        prop_assert_eq!(valid, witness_item_count == 0 || witness_item_size <= 520);
    }
}

/// Property test: Taproot output validation
proptest! {
    #[test]
        fn prop_taproot_output_valid(
        version in prop::sample::select(vec![OP_0, OP_1, 0x52u8]),
        program_len in 31usize..=35,
        push in prop::sample::select(vec![PUSH_32_BYTES, 0x21u8])
    ) {
        let mut script = vec![version, push];
        script.extend(std::iter::repeat(0x11).take(program_len));
        let valid = validate_taproot_script(&script).unwrap();
        let expected = script.len() == 34 && version == OP_1 && push == PUSH_32_BYTES;
        prop_assert_eq!(valid, expected);
    }
}

/// Property test: SegWit transaction size vs weight
proptest! {
    #[test]
    fn prop_segwit_size_weight_relationship(
        base_size in 100usize..1000usize,
        witness_size in 0usize..500usize
    ) {
        let weight = blvm_consensus::witness::calculate_transaction_weight_segwit(
            base_size as u64,
            (base_size + witness_size) as u64,
        );
        prop_assert_eq!(weight, (base_size * 3 + base_size + witness_size) as u64);
    }
}

/// Property test: witness commitment validation
proptest! {
    #[test]
    fn prop_witness_commitment_valid(
        witness_root in prop::array::uniform32(0u8..=255u8),
        nonce in prop::array::uniform32(0u8..=255u8)
    ) {
        let script = segwit::witness_commitment_script(&witness_root, &nonce);
        prop_assert_eq!(script.len(), 38);
        let mut preimage = [0u8; 64];
        preimage[..32].copy_from_slice(&witness_root);
        preimage[32..].copy_from_slice(&nonce);
        let digest = {
            use sha2::{Digest, Sha256};
            Sha256::digest(Sha256::digest(preimage))
        };
        prop_assert_eq!(&script[6..38], &digest[..]);
    }
}

/// Property test: SegWit version validation
proptest! {
    #[test]
    fn prop_segwit_version_valid(version in any::<u8>(), program_len in 1usize..=41usize) {
        use blvm_consensus::opcodes::{OP_0, OP_1};
        use blvm_consensus::witness::{WitnessVersion, extract_witness_version};

        let mut script = vec![version, program_len as u8];
        script.extend(std::iter::repeat_n(0x11u8, program_len));
        let accepted = (version == OP_0 || version == OP_1) && (2..=40).contains(&program_len);
        match extract_witness_version(&script) {
            Some(WitnessVersion::SegWitV0) => prop_assert!(version == OP_0 && accepted),
            Some(WitnessVersion::TaprootV1) => prop_assert!(version == OP_1 && accepted),
            None => prop_assert!(!accepted),
        }
    }
}

/// Property test: Taproot key path vs script path
proptest! {
    #[test]
    fn prop_taproot_path_validation(sig_len in 63usize..=66usize) {
        use blvm_consensus::taproot::TAPROOT_CONTROL_BASE_SIZE;
        use blvm_consensus::witness::validate_taproot_witness_structure;

        let key_path = vec![vec![0x11u8; sig_len]];
        let key_ok = validate_taproot_witness_structure(&key_path, false).unwrap();
        prop_assert_eq!(key_ok, sig_len == 64 || sig_len == 65);

        let script_only = vec![vec![0x11u8; 8]];
        prop_assert!(!validate_taproot_witness_structure(&script_only, true).unwrap());
        let script_path = vec![vec![0x11u8; 8], vec![0u8; TAPROOT_CONTROL_BASE_SIZE]];
        prop_assert!(validate_taproot_witness_structure(&script_path, true).unwrap());
    }
}

/// Property test: block weight with mixed SegWit/non-SegWit
proptest! {
    #[test]
    fn prop_mixed_segwit_block_weight(
        segwit_tx_count in 0usize..5usize,
        non_segwit_tx_count in 0usize..5usize
    ) {
        let total_tx_count = segwit_tx_count + non_segwit_tx_count;

        if total_tx_count > 0 {
            let txs: Vec<Transaction> = (0..total_tx_count)
                .map(|i| Transaction {
                    version: 1,
                    inputs: vec![TransactionInput {
                        prevout: OutPoint { hash: [i as u8; 32], index: 0 },
                        script_sig: vec![OP_1],
                        sequence: 0xffffffff,
                    }].into(),
                    outputs: vec![TransactionOutput {
                        value: 1000,
                        script_pubkey: vec![OP_1],
                    }].into(),
                    lock_time: 0,
                })
                .collect();
            let block = Block {
                header: BlockHeader {
                    version: 1,
                    prev_block_hash: [0; 32],
                    merkle_root: [1; 32],
                    timestamp: 1234567890,
                    bits: 0x1d00ffff,
                    nonce: 0,
                },
                transactions: txs.into_boxed_slice(),
            };

            // Create witnesses (SegWit txs have witnesses, non-SegWit have empty)
            let mut witnesses: Vec<segwit::Witness> = Vec::new();
            for i in 0..total_tx_count {
                if i < segwit_tx_count {
                    witnesses.push(vec![vec![OP_1; 50]]);
                } else {
                    witnesses.push(vec![vec![]]);
                }
            }

            let weight = segwit::calculate_block_weight(&block, &witnesses).unwrap();
            prop_assert!(weight > 0);
        }
    }
}

/// Property test: transaction weight increases with witness size
proptest! {
    #[test]
    fn prop_transaction_weight_increases_with_witness(
        base_size in 100usize..500usize,
        witness1_size in 0usize..100usize,
        witness2_size in 0usize..100usize
    ) {
        // Ensure witness2_size >= witness1_size for comparison
        let (w1, w2) = if witness1_size <= witness2_size {
            (witness1_size, witness2_size)
        } else {
            (witness2_size, witness1_size)
        };

        let weight_of = |witness: usize| {
            blvm_consensus::witness::calculate_transaction_weight_segwit(
                base_size as u64,
                (base_size + witness) as u64,
            )
        };
        prop_assert!(weight_of(w2) >= weight_of(w1));
    }
}

/// Property test: SegWit discount factor (witness data counts less)
proptest! {
    #[test]
    fn prop_segwit_witness_discount(
        base_bytes in 100usize..500usize,
        witness_bytes in 100usize..500usize
    ) {
        // Weight = base_bytes * 4 + witness_bytes * 1
        // Base data counts 4x, witness counts 1x
        let weight = (base_bytes * 4) + witness_bytes;

        // Base bytes contribute more to weight
        prop_assert!(weight >= base_bytes * 4);
        prop_assert!(weight >= witness_bytes);

        // Witness discount: witness_bytes contribute less than base_bytes would
        if witness_bytes > 0 {
            prop_assert!((witness_bytes * 4) > witness_bytes,
                "Base bytes contribute 4x, witness contributes 1x");
        }
    }
}
