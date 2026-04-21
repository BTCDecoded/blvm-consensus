#![no_main]
//! Fuzz block validation and UTXO operations used by compact block reconstruction.
//!
//! Compact block (BIP152) reconstruction logic lives in blvm-node; this target
//! exercises the consensus-layer block connection and UTXO handling that compact
//! blocks depend on. For full BIP152 fuzzing (shortids, prefilled txs, etc.),
//! see blvm-node/fuzz/ when available.

use blvm_consensus::block::connect_block;
use blvm_consensus::{Block, BlockHeader, Transaction, TransactionOutput, UtxoSet};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz block operations that compact blocks depend on
    // Tests block validation, UTXO operations, and transaction handling
    // Note: Full compact block fuzzing requires blvm-node, see blvm-node/fuzz/

    if data.len() < 88 {
        return; // Need at least block header (88 bytes)
    }

    // Create a minimal block header from fuzzed data
    let header = BlockHeader {
        version: i64::from_le_bytes([
            data.get(0).copied().unwrap_or(1) as u8,
            data.get(1).copied().unwrap_or(0),
            data.get(2).copied().unwrap_or(0),
            data.get(3).copied().unwrap_or(0),
            data.get(4).copied().unwrap_or(0),
            data.get(5).copied().unwrap_or(0),
            data.get(6).copied().unwrap_or(0),
            data.get(7).copied().unwrap_or(0),
        ]),
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
        timestamp: u64::from_le_bytes([
            data.get(72).copied().unwrap_or(0),
            data.get(73).copied().unwrap_or(0),
            data.get(74).copied().unwrap_or(0),
            data.get(75).copied().unwrap_or(0),
            data.get(76).copied().unwrap_or(0),
            data.get(77).copied().unwrap_or(0),
            data.get(78).copied().unwrap_or(0),
            data.get(79).copied().unwrap_or(0),
        ]),
        bits: u32::from_le_bytes([
            data.get(80).copied().unwrap_or(0),
            data.get(81).copied().unwrap_or(0),
            data.get(82).copied().unwrap_or(0),
            data.get(83).copied().unwrap_or(0),
        ]) as u64,
        nonce: u32::from_le_bytes([
            data.get(84).copied().unwrap_or(0),
            data.get(85).copied().unwrap_or(0),
            data.get(86).copied().unwrap_or(0),
            data.get(87).copied().unwrap_or(0),
        ]) as u64,
    };

    // Create minimal block with coinbase transaction
    let block = Block {
        header,
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![].into(), // Coinbase
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![0x51].into(), // OP_1
            }]
            .into(),
            lock_time: 0,
        }]
        .into_boxed_slice(),
    };

    let utxo_set = UtxoSet::default();
    let witnesses: Vec<Vec<blvm_consensus::Witness>> = block
        .transactions
        .iter()
        .map(|tx| (0..tx.inputs.len()).map(|_| vec![]).collect())
        .collect();

    // Test block connection - should never panic
    // This exercises the same code paths that compact blocks use
    let ctx = blvm_consensus::block::BlockValidationContext::for_network(
        blvm_consensus::types::Network::Mainnet,
    );
    let _result = connect_block(&block, &witnesses, utxo_set, 0, &ctx);
});
