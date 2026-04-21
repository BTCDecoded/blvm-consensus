#![no_main]
use blvm_consensus::block::{connect_block, BlockValidationContext};
use blvm_consensus::pow::check_proof_of_work;
use blvm_consensus::types::{Block, BlockHeader, Network, TimeContext, Transaction, TransactionInput, TransactionOutput, UtxoSet};
use blvm_consensus::witness::Witness;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Block header validation fuzzing
    // Tests header field validation, timestamp validation, difficulty target validation, and PoW validation

    if data.len() < 88 {
        return; // Need at least block header (88 bytes)
    }

    // Test 1: Basic header field validation
    // Parse header from fuzzed data
    let header = BlockHeader {
        version: if data.len() >= 4 {
            i64::from_le_bytes([
                data[0] as u8,
                data.get(1).copied().unwrap_or(0),
                data.get(2).copied().unwrap_or(0),
                data.get(3).copied().unwrap_or(0),
                data.get(4).copied().unwrap_or(0),
                data.get(5).copied().unwrap_or(0),
                data.get(6).copied().unwrap_or(0),
                data.get(7).copied().unwrap_or(0),
            ])
        } else {
            1
        },
        prev_block_hash: data
            .get(8..40)
            .unwrap_or(&[0; 32])
            .try_into()
            .unwrap_or([0; 32]),
        merkle_root: data
            .get(40..72)
            .unwrap_or(&[0; 32])
            .try_into()
            .unwrap_or([0; 32]),
        timestamp: if data.len() >= 80 {
            u64::from_le_bytes([
                data.get(72).copied().unwrap_or(0),
                data.get(73).copied().unwrap_or(0),
                data.get(74).copied().unwrap_or(0),
                data.get(75).copied().unwrap_or(0),
                data.get(76).copied().unwrap_or(0),
                data.get(77).copied().unwrap_or(0),
                data.get(78).copied().unwrap_or(0),
                data.get(79).copied().unwrap_or(0),
            ])
        } else {
            0
        },
        bits: if data.len() >= 84 {
            u32::from_le_bytes([
                data.get(80).copied().unwrap_or(0),
                data.get(81).copied().unwrap_or(0),
                data.get(82).copied().unwrap_or(0),
                data.get(83).copied().unwrap_or(0),
            ]) as u64
        } else {
            0
        },
        nonce: if data.len() >= 88 {
            u32::from_le_bytes([
                data.get(84).copied().unwrap_or(0),
                data.get(85).copied().unwrap_or(0),
                data.get(86).copied().unwrap_or(0),
                data.get(87).copied().unwrap_or(0),
            ]) as u64
        } else {
            0
        },
    };

    // Test 2: Proof of Work validation
    let _pow_result = check_proof_of_work(&header);
    let _pow_result_testnet = check_proof_of_work(&header);
    let _pow_result_regtest = check_proof_of_work(&header);

    // Test 3: Header validation through block connection
    // Build minimal block with header
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: blvm_consensus::types::OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x00, 0x00].into(), // Minimal coinbase scriptSig
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000, // 50 BTC
            script_pubkey: vec![].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let block = Block {
        header: header.clone(),
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };

    let empty_utxo_set = UtxoSet::default();
    let witnesses: Vec<Vec<Witness>> = vec![vec![]; block.transactions.len()];

    // Test with different networks
    let networks = [Network::Mainnet, Network::Testnet, Network::Regtest];
    let heights = [0u64, 100, 1000, 100000, 500000];

    for &network in &networks {
        for &height in &heights {
            // Create time context
            let time_context = TimeContext {
                network_time: header.timestamp.saturating_add(3600), // 1 hour in future
                median_time_past: header.timestamp.saturating_sub(3600), // 1 hour in past
            };

            // Test block connection (which validates header internally)
            let ctx_with_time = BlockValidationContext::from_time_context_and_network(
                Some(time_context.clone()),
                network,
                None,
            );
            let _result = connect_block(
                &block,
                &witnesses,
                empty_utxo_set.clone(),
                height,
                &ctx_with_time,
            );

            // Test with None time context
            let ctx_no_time = BlockValidationContext::for_network(network);
            let _result_no_time = connect_block(
                &block,
                &witnesses,
                empty_utxo_set.clone(),
                height,
                &ctx_no_time,
            );
        }
    }

    // Test 4: Edge cases
    // Zero timestamp
    let mut header_zero_ts = header.clone();
    header_zero_ts.timestamp = 0;
    let block_zero_ts = Block {
        header: header_zero_ts,
        transactions: block.transactions.clone(),
    };
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let _result_zero_ts = connect_block(
        &block_zero_ts,
        &witnesses,
        empty_utxo_set.clone(),
        0,
        &ctx,
    );

    // Zero bits
    let mut header_zero_bits = header.clone();
    header_zero_bits.bits = 0;
    let block_zero_bits = Block {
        header: header_zero_bits,
        transactions: block.transactions.clone(),
    };
    let _result_zero_bits = connect_block(
        &block_zero_bits,
        &witnesses,
        empty_utxo_set.clone(),
        0,
        &ctx,
    );

    // Zero merkle root
    let mut header_zero_merkle = header.clone();
    header_zero_merkle.merkle_root = [0; 32];
    let block_zero_merkle = Block {
        header: header_zero_merkle,
        transactions: block.transactions.clone(),
    };
    let _result_zero_merkle = connect_block(
        &block_zero_merkle,
        &witnesses,
        empty_utxo_set.clone(),
        0,
        &ctx,
    );

    // Invalid version
    let mut header_invalid_version = header.clone();
    header_invalid_version.version = -1;
    let block_invalid_version = Block {
        header: header_invalid_version,
        transactions: block.transactions.clone(),
    };
    let _result_invalid_version = connect_block(
        &block_invalid_version,
        &witnesses,
        empty_utxo_set.clone(),
        0,
        &ctx,
    );
});

