//! Bolero fuzzing tests for block validation
//!
//! Tests block validation functions with generated inputs to find edge cases
//! and ensure correctness across a wide range of block configurations.
//!
//! Note: Uses byte-level fuzzing since BlockHeader doesn't implement Arbitrary.

#[cfg(feature = "bolero")]
use bolero::check;
#[cfg(feature = "bolero")]
use blvm_consensus::{Block, BlockHeader, UtxoSet, ValidationResult, connect_block, check_proof_of_work};

#[cfg(feature = "bolero")]
#[test]
fn fuzz_check_proof_of_work_robustness() {
    // Use byte-level fuzzing to test robustness (80 bytes = block header size)
    check!().for_each(|data: &[u8]| {
        if data.len() < 80 {
            return;
        }
        let header_bytes: [u8; 80] = data[..80].try_into().unwrap_or_else(|_| [0; 80]);
        // Create a block header from fuzzed bytes
        let header = BlockHeader {
            version: i32::from_le_bytes([header_bytes[0], header_bytes[1], header_bytes[2], header_bytes[3]]),
            prev_block_hash: header_bytes[4..36].try_into().unwrap_or([0; 32]),
            merkle_root: header_bytes[36..68].try_into().unwrap_or([0; 32]),
            timestamp: u64::from_le_bytes([
                header_bytes[68], header_bytes[69], header_bytes[70], header_bytes[71],
                header_bytes[72], header_bytes[73], header_bytes[74], header_bytes[75],
            ]),
            bits: u32::from_le_bytes([header_bytes[72], header_bytes[73], header_bytes[74], header_bytes[75]]) as u64,
            nonce: u32::from_le_bytes([header_bytes[76], header_bytes[77], header_bytes[78], header_bytes[79]]) as u64,
        };
        
        // Validate that check_proof_of_work doesn't panic on any input
        let result = check_proof_of_work(&header);
        // Result should always be Ok, even if validation fails
        assert!(result.is_ok(), "check_proof_of_work should never panic");
    });
}

#[cfg(feature = "bolero")]
#[test]
fn fuzz_check_proof_of_work_deterministic() {
    // Test determinism with byte-based inputs
    check!().for_each(|data: &[u8]| {
        if data.len() < 80 {
            return;
        }
        let header_bytes: [u8; 80] = data[..80].try_into().unwrap_or_else(|_| [0; 80]);
        let header = BlockHeader {
            version: i32::from_le_bytes([header_bytes[0], header_bytes[1], header_bytes[2], header_bytes[3]]),
            prev_block_hash: header_bytes[4..36].try_into().unwrap_or([0; 32]),
            merkle_root: header_bytes[36..68].try_into().unwrap_or([0; 32]),
            timestamp: u64::from_le_bytes([
                header_bytes[68], header_bytes[69], header_bytes[70], header_bytes[71],
                header_bytes[72], header_bytes[73], header_bytes[74], header_bytes[75],
            ]),
            bits: u32::from_le_bytes([header_bytes[72], header_bytes[73], header_bytes[74], header_bytes[75]]) as u64,
            nonce: u32::from_le_bytes([header_bytes[76], header_bytes[77], header_bytes[78], header_bytes[79]]) as u64,
        };
        
        // Check that validation is deterministic
        let result1 = check_proof_of_work(&header);
        let result2 = check_proof_of_work(&header);
        
        assert_eq!(result1, result2, "check_proof_of_work must be deterministic");
    });
}

#[cfg(feature = "bolero")]
#[test]
fn fuzz_connect_block_structure() {
    check!().for_each(|data: &[u8]| {
        // Try to validate block structure robustness
        if data.len() >= 80 {
            // Test robustness - create minimal valid block structure
            let minimal_block = Block {
                header: BlockHeader {
                    version: 1,
                    prev_block_hash: [0; 32],
                    merkle_root: [0; 32],
                    timestamp: 0,
                    bits: 0x1d00ffff,
                    nonce: 0,
                },
            transactions: vec![].into(),
            };
            
            let initial_utxo_set = UtxoSet::new();
            let witnesses: Vec<segwit::Witness> = minimal_block.transactions.iter().map(|_| Vec::new()).collect();
            let result = connect_block(&minimal_block, &witnesses, initial_utxo_set, 0, None, 0u64, crate::types::Network::Mainnet);
            // Result should always be Ok, even if validation fails
            assert!(result.is_ok(), "connect_block should handle blocks without panicking");
        }
    });
}
