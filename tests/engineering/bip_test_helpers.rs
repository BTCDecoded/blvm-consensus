//! Helper functions for BIP integration tests
//! 
//! Provides utilities for creating test transactions, block contexts, and
//! validation scenarios for testing BIP65, BIP112, and related BIPs.

use bllvm_consensus::*;
use bllvm_consensus::script::verify_script_with_context_full;
use bllvm_consensus::bip113::get_median_time_past;

/// Create a block header with specified timestamp
pub fn create_test_header(timestamp: u64, prev_hash: [u8; 32]) -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: prev_hash,
        merkle_root: [0u8; 32],
        timestamp,
        bits: 0x1d00ffff,
        nonce: 0,
    }
}

/// Create a chain of block headers with timestamps
pub fn create_header_chain(timestamps: Vec<u64>) -> Vec<BlockHeader> {
    let mut headers = Vec::new();
    let mut prev_hash = [0u8; 32];
    
    for timestamp in timestamps {
        let header = create_test_header(timestamp, prev_hash);
        // Use a simple hash derivation for testing
        prev_hash = {
            let mut hash = [0u8; 32];
            hash[0..8].copy_from_slice(&timestamp.to_le_bytes());
            hash[8..16].copy_from_slice(&prev_hash[0..8]);
            hash
        };
        headers.push(header);
    }
    
    headers
}

/// Calculate median time-past for testing
pub fn get_test_median_time_past(timestamps: Vec<u64>) -> u64 {
    let headers = create_header_chain(timestamps);
    get_median_time_past(&headers)
}

/// Encode a value as script integer (little-endian, minimal encoding)
pub fn encode_script_int(value: u32) -> Vec<u8> {
    if value == 0 {
        return vec![0];
    }
    
    let mut bytes = Vec::new();
    let mut v = value;
    while v > 0 {
        bytes.push((v & 0xff) as u8);
        v >>= 8;
    }
    bytes
}

/// Create a transaction with CLTV opcode
pub fn create_cltv_transaction(
    locktime: u32,
    required_locktime: u32,
    script_sig: Vec<u8>,
) -> Transaction {
    let mut script = script_sig.clone();
    
    // Add required locktime to script
    script.extend_from_slice(&encode_script_int(required_locktime));
    
    // Add CLTV opcode
    script.push(0xb1); // OP_CHECKLOCKTIMEVERIFY
    
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: script,
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
        }],
        lock_time: locktime as u64,
    }
}

/// Create a transaction with CSV opcode
pub fn create_csv_transaction(
    input_sequence: u32,
    required_sequence: u32,
    script_sig: Vec<u8>,
) -> Transaction {
    let mut script = script_sig.clone();
    
    // Add required sequence to script
    script.extend_from_slice(&encode_script_int(required_sequence));
    
    // Add CSV opcode
    script.push(0xb2); // OP_CHECKSEQUENCEVERIFY
    
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: script,
            sequence: input_sequence as u64,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
        }],
        lock_time: 0,
    }
}

/// Validate a transaction with CLTV/CSV using full context
pub fn validate_with_context(
    tx: &Transaction,
    utxo_set: &UtxoSet,
    block_height: u64,
    median_time_past: u64,
) -> Result<bool> {
    // Get the scriptPubkey from UTXO
    let input = &tx.inputs[0];
    let utxo = utxo_set.get(&input.prevout)
        .ok_or_else(|| ConsensusError::UtxoNotFound("UTXO not found".to_string()))?;
    
    // Create prevouts for context
    let prevouts = vec![TransactionOutput {
        value: utxo.value,
        script_pubkey: utxo.script_pubkey.clone(),
    }];
    
    // Verify script with context (now supports block height and median time-past)
    verify_script_with_context_full(
        &input.script_sig,
        &utxo.script_pubkey,
        None, // No witness for basic tests
        0,    // Flags
        tx,
        0,    // Input index
        &prevouts,
        Some(block_height), // Optional block height
        Some(median_time_past), // Optional median time-past
    )
}

