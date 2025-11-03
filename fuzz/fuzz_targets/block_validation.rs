#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::{Block, BlockHeader, UtxoSet};
use consensus_proof::block::connect_block;

fuzz_target!(|data: &[u8]| {
    // Test block validation robustness
    if data.len() < 88 {
        return; // Need at least block header (88 bytes)
    }
    
    // Create minimal block from fuzzed data
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
        prev_block_hash: data.get(8..40)
            .unwrap_or(&[0; 32])
            .try_into()
            .unwrap_or([0; 32]),
        merkle_root: data.get(40..72)
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
    
    // Create transactions from remaining data if available
    let mut transactions = Vec::new();
    if data.len() > 88 {
        // Try to parse at least one transaction for more realistic testing
        let tx_data = &data[88..];
        if tx_data.len() >= 100 {
            // Create a minimal coinbase transaction
            transactions.push(consensus_proof::Transaction {
                version: 1,
                inputs: vec![consensus_proof::TransactionInput {
                    prevout: consensus_proof::OutPoint {
                        hash: [0u8; 32],
                        index: 0xffffffff,
                    },
                    script_sig: tx_data[..tx_data.len().min(100)].to_vec(),
                    sequence: 0xffffffff,
                }],
                outputs: vec![consensus_proof::TransactionOutput {
                    value: 5000000000,
                    script_pubkey: vec![0x51], // OP_1
                }],
                lock_time: 0,
            });
        }
    }
    
    let block = Block {
        header,
        transactions,
    };
    
    let utxo_set = UtxoSet::new();
    
    // Should never panic - test robustness
    let _result = connect_block(&block, utxo_set, 0);
});

