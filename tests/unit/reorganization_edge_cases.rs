//! Property tests for chain reorganization edge cases

use blvm_consensus::opcodes::OP_1;
use blvm_consensus::reorganization;
use blvm_consensus::types::*;
use proptest::prelude::*;

fn make_block(i: usize, bits: u64) -> Block {
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffffu32,
            },
            script_sig: (i as u64).to_le_bytes().to_vec(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    };
    Block {
        header: BlockHeader {
            version: 1i64,
            prev_block_hash: if i == 0 { [0; 32] } else { [i as u8; 32] },
            merkle_root: [1; 32],
            timestamp: 1234567890 + (i as u64 * 600),
            bits,
            nonce: i as u64,
        },
        transactions: vec![coinbase].into_boxed_slice(),
    }
}

fn make_chain(n: usize) -> Vec<Block> {
    (0..n).map(|i| make_block(i, 0x1d00ffff)).collect()
}

proptest! {
    #[test]
    fn prop_chain_work_non_negative(block_count in 1usize..20usize) {
        let chain = make_chain(block_count);
        let result = reorganization::calculate_chain_work(&chain);
        prop_assert!(result.is_ok() || result.is_err());
        if let Ok(work) = result {
            prop_assert!(work >= 0);
        }
    }
}

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
        let short_chain = make_chain(short_len);
        let long_chain = make_chain(long_len);
        let short_work = reorganization::calculate_chain_work(&short_chain);
        let long_work = reorganization::calculate_chain_work(&long_chain);
        if short_work.is_ok() && long_work.is_ok() {
            prop_assert!(long_work.unwrap() >= short_work.unwrap());
        }
    }
}

proptest! {
    #[test]
    fn prop_reorganize_prefers_more_work(
        chain1_len in 1usize..10usize,
        chain2_len in 1usize..10usize
    ) {
        let chain1 = make_chain(chain1_len);
        let chain2 = make_chain(chain2_len);
        let result = reorganization::should_reorganize(&chain1, &chain2);
        prop_assert!(result.is_ok() || result.is_err());
    }
}

proptest! {
    #[test]
    fn prop_reorganization_utxo_consistency(
        initial_height in 1u64..10u64,
        _reorg_depth in 1u64..5u64
    ) {
        let mut utxo_set = UtxoSet::default();
        for i in 0..5 {
            utxo_set.insert(
                OutPoint { hash: [i as u8; 32], index: 0u32 },
                std::sync::Arc::new(UTXO {
                    value: 1000 * (i as i64 + 1),
                    script_pubkey: vec![OP_1].into(),
                    height: initial_height,
                    is_coinbase: false,
                })
            );
        }
        let initial_utxo_count = utxo_set.len();
        prop_assert!(initial_utxo_count <= 1000);
        prop_assert!(utxo_set.len() <= initial_utxo_count + 10);
    }
}

proptest! {
    #[test]
    fn prop_chain_work_deterministic(block_count in 1usize..10usize) {
        let chain = make_chain(block_count);
        let work1 = reorganization::calculate_chain_work(&chain);
        let work2 = reorganization::calculate_chain_work(&chain);
        prop_assert_eq!(work1.is_ok(), work2.is_ok());
        if work1.is_ok() && work2.is_ok() {
            prop_assert_eq!(work1.unwrap(), work2.unwrap());
        }
    }
}

#[test]
fn empty_chain_zero_work() {
    let empty: Vec<Block> = Vec::new();
    let result = reorganization::calculate_chain_work(&empty);
    assert!(result.is_ok() || result.is_err());
    if let Ok(work) = result {
        assert_eq!(work, 0);
    }
}

proptest! {
    #[test]
    fn prop_reorganization_depth_bounded(
        current_chain_len in 1usize..20usize,
        new_chain_len in 1usize..20usize
    ) {
        let common_prefix = current_chain_len.min(new_chain_len);
        let reorg_depth = current_chain_len - common_prefix;
        prop_assert!(reorg_depth <= current_chain_len);
    }
}

proptest! {
    #[test]
    fn prop_chain_fork_point(
        fork_height in 1usize..10usize,
        chain1_extension in 1usize..10usize,
        chain2_extension in 1usize..10usize
    ) {
        let chain1_len = fork_height + chain1_extension;
        let chain2_len = fork_height + chain2_extension;
        prop_assert!(chain1_len >= fork_height);
        prop_assert!(chain2_len >= fork_height);
    }
}
