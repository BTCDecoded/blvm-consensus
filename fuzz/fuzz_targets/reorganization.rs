#![no_main]
use libfuzzer_sys::fuzz_target;
use bllvm_consensus::{Block, BlockHeader, UtxoSet};
use bllvm_consensus::reorganization::reorganize_chain;

fuzz_target!(|data: &[u8]| {
    // Test chain reorganization robustness
    // Fuzz reorganization logic with random chains
    
    if data.len() < 80 {
        return; // Need at least one block header (80 bytes)
    }
    
    // Create minimal chains from fuzzed data
    // Split data into two parts for current_chain and new_chain
    let split_point = data.len() / 2;
    let current_chain_data = &data[..split_point];
    let new_chain_data = &data[split_point..];
    
    // Create simple blocks from data
    let mut current_chain = Vec::new();
    let mut new_chain = Vec::new();
    
    // Parse current chain (simplified - just create minimal valid blocks)
    let mut offset = 0;
    while offset + 80 <= current_chain_data.len() && current_chain.len() < 5 {
        let header = BlockHeader {
            version: 1,
            prev_block_hash: current_chain_data[offset..offset + 32]
                .try_into()
                .unwrap_or([0; 32]),
            merkle_root: current_chain_data[offset + 32..offset + 64]
                .try_into()
                .unwrap_or([0; 32]),
            timestamp: u64::from_le_bytes([
                current_chain_data.get(offset + 64).copied().unwrap_or(0),
                current_chain_data.get(offset + 65).copied().unwrap_or(0),
                current_chain_data.get(offset + 66).copied().unwrap_or(0),
                current_chain_data.get(offset + 67).copied().unwrap_or(0),
                current_chain_data.get(offset + 68).copied().unwrap_or(0),
                current_chain_data.get(offset + 69).copied().unwrap_or(0),
                current_chain_data.get(offset + 70).copied().unwrap_or(0),
                current_chain_data.get(offset + 71).copied().unwrap_or(0),
            ]),
            bits: 0x1d00ffff,
            nonce: u64::from_le_bytes([
                current_chain_data.get(offset + 72).copied().unwrap_or(0),
                current_chain_data.get(offset + 73).copied().unwrap_or(0),
                current_chain_data.get(offset + 74).copied().unwrap_or(0),
                current_chain_data.get(offset + 75).copied().unwrap_or(0),
                current_chain_data.get(offset + 76).copied().unwrap_or(0),
                current_chain_data.get(offset + 77).copied().unwrap_or(0),
                current_chain_data.get(offset + 78).copied().unwrap_or(0),
                current_chain_data.get(offset + 79).copied().unwrap_or(0),
            ]),
        };
        
        // Create minimal block with coinbase transaction
        let block = Block {
            header,
            transactions: vec![bllvm_consensus::Transaction {
                version: 1,
                inputs: vec![bllvm_consensus::TransactionInput {
                    prevout: bllvm_consensus::OutPoint {
                        hash: [0; 32],
                        index: 0xffffffff,
                    },
                    script_sig: vec![0x51, 0x51], // 2 bytes for valid coinbase
                    sequence: 0xffffffff,
                }],
                outputs: vec![bllvm_consensus::TransactionOutput {
                    value: 5000000000, // 50 BTC
                    script_pubkey: vec![0x51],
                }],
                lock_time: 0,
            }].into_boxed_slice(),
        };
        
        current_chain.push(block);
        offset += 80;
    }
    
    // Parse new chain (same logic)
    let mut offset = 0;
    while offset + 80 <= new_chain_data.len() && new_chain.len() < 5 {
        let header = BlockHeader {
            version: 1,
            prev_block_hash: new_chain_data[offset..offset + 32]
                .try_into()
                .unwrap_or([0; 32]),
            merkle_root: new_chain_data[offset + 32..offset + 64]
                .try_into()
                .unwrap_or([0; 32]),
            timestamp: u64::from_le_bytes([
                new_chain_data.get(offset + 64).copied().unwrap_or(0),
                new_chain_data.get(offset + 65).copied().unwrap_or(0),
                new_chain_data.get(offset + 66).copied().unwrap_or(0),
                new_chain_data.get(offset + 67).copied().unwrap_or(0),
                new_chain_data.get(offset + 68).copied().unwrap_or(0),
                new_chain_data.get(offset + 69).copied().unwrap_or(0),
                new_chain_data.get(offset + 70).copied().unwrap_or(0),
                new_chain_data.get(offset + 71).copied().unwrap_or(0),
            ]),
            bits: 0x1d00ffff,
            nonce: u64::from_le_bytes([
                new_chain_data.get(offset + 72).copied().unwrap_or(0),
                new_chain_data.get(offset + 73).copied().unwrap_or(0),
                new_chain_data.get(offset + 74).copied().unwrap_or(0),
                new_chain_data.get(offset + 75).copied().unwrap_or(0),
                new_chain_data.get(offset + 76).copied().unwrap_or(0),
                new_chain_data.get(offset + 77).copied().unwrap_or(0),
                new_chain_data.get(offset + 78).copied().unwrap_or(0),
                new_chain_data.get(offset + 79).copied().unwrap_or(0),
            ]),
        };
        
        let block = Block {
            header,
            transactions: vec![bllvm_consensus::Transaction {
                version: 1,
                inputs: vec![bllvm_consensus::TransactionInput {
                    prevout: bllvm_consensus::OutPoint {
                        hash: [0; 32],
                        index: 0xffffffff,
                    },
                    script_sig: vec![0x51, 0x51],
                    sequence: 0xffffffff,
                }],
                outputs: vec![bllvm_consensus::TransactionOutput {
                    value: 5000000000,
                    script_pubkey: vec![0x51],
                }],
                lock_time: 0,
            }].into_boxed_slice(),
        };
        
        new_chain.push(block);
        offset += 80;
    }
    
    // Test reorganization
    let utxo_set = UtxoSet::new();
    let current_height = current_chain.len() as u64;
    
    let _result = reorganize_chain(
        &new_chain,
        &current_chain,
        utxo_set,
        current_height,
    );
    
    // Don't assert on result - just exercise the code path
    // Fuzzing goal is to find crashes, not verify correctness
});

