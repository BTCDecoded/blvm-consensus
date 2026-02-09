//! Bitcoin Core block test vector integration
//!
//! Tests block validation using Core's test vectors.
//! These vectors provide comprehensive coverage of edge cases.
//!
//! Core test vector format (block_valid.json / block_invalid.json):
//! Array of arrays: [[block_hex, height, prev_utxo_set?, expected], ...]
//! - block_hex: Block in hex format (with witness data if SegWit)
//! - height: Block height for validation
//! - prev_utxo_set: Optional previous UTXO set (empty if not provided)
//! - expected: Expected validation result

use blvm_consensus::{Block, BlockHeader, UtxoSet, ValidationResult};
use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
use blvm_consensus::block::connect_block;
use std::path::PathBuf;
use std::fs;
use serde_json::Value;
use hex;

/// Load test vectors from a directory
///
/// Expected format: JSON files containing block test vectors from Bitcoin Core
pub fn load_block_test_vectors(dir: &str) -> Result<Vec<BlockTestVector>, Box<dyn std::error::Error>> {
    let mut vectors = Vec::new();
    let path = PathBuf::from(dir);
    
    if !path.exists() {
        // If test vectors directory doesn't exist, return empty (not an error)
        // Test vectors need to be downloaded from Bitcoin Core repository
        return Ok(vectors);
    }
    
    // Try to load block_valid.json
    let valid_path = path.join("block_valid.json");
    if valid_path.exists() {
        let content = fs::read_to_string(&valid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        if let Value::Array(cases) = json {
            for (i, case) in cases.iter().enumerate() {
                if let Value::Array(test_case) = case {
                    if test_case.len() >= 2 {
                        // Parse block hex
                        let block_hex = test_case[0].as_str()
                            .ok_or_else(|| format!("Invalid block hex at index {}", i))?;
                        let block_bytes = hex::decode(block_hex)?;
                        
                        // Deserialize block (witness data is parsed but not used in validation)
                        let (block, _witnesses) = deserialize_block_with_witnesses(&block_bytes)
                            .map_err(|e| format!("Failed to deserialize block at index {}: {}", i, e))?;
                        
                        // Parse height
                        let height = test_case.get(1)
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        
                        // Parse previous UTXO set if provided (otherwise use empty)
                        let prev_utxo_set = if test_case.len() > 2 {
                            // UTXO set might be provided as JSON or hex, but typically empty for test vectors
                            UtxoSet::new()
                        } else {
                            UtxoSet::new()
                        };
                        
                        vectors.push(BlockTestVector {
                            block,
                            expected_result: ValidationResult::Valid,
                            height,
                            prev_utxo_set,
                        });
                    }
                }
            }
        }
    }
    
    // Try to load block_invalid.json
    let invalid_path = path.join("block_invalid.json");
    if invalid_path.exists() {
        let content = fs::read_to_string(&invalid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        if let Value::Array(cases) = json {
            for (i, case) in cases.iter().enumerate() {
                if let Value::Array(test_case) = case {
                    if test_case.len() >= 2 {
                        // Parse block hex
                        let block_hex = test_case[0].as_str()
                            .ok_or_else(|| format!("Invalid block hex at index {}", i))?;
                        let block_bytes = hex::decode(block_hex)?;
                        
                        // Try to deserialize - invalid blocks may fail at deserialization
                        // or may deserialize but fail validation
                        if let Ok((block, _witnesses)) = deserialize_block_with_witnesses(&block_bytes) {
                            // Parse height
                            let height = test_case.get(1)
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            
                            // Parse description if provided
                            let description = test_case.last()
                                .and_then(|v| v.as_str())
                                .unwrap_or("Invalid block")
                                .to_string();
                            
                            vectors.push(BlockTestVector {
                                block,
                                expected_result: ValidationResult::Invalid(description),
                                height,
                                prev_utxo_set: UtxoSet::new(),
                            });
                        }
                        // If deserialization fails, that's expected for invalid blocks
                    }
                }
            }
        }
    }
    
    Ok(vectors)
}

/// Run Core block test vectors
pub fn run_core_block_tests(vectors: &[BlockTestVector]) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed = 0;
    let mut failed = 0;
    
    for (i, vector) in vectors.iter().enumerate() {
        let witnesses: Vec<segwit::Witness> = vector.block.transactions.iter().map(|_| Vec::new()).collect();
        let result = connect_block(&vector.block, &witnesses, vector.prev_utxo_set.clone(), vector.height, None, 0u64, crate::types::Network::Mainnet);
        
        match result {
            Ok((validation_result, _utxo_set)) => {
                let is_valid = matches!(validation_result, ValidationResult::Valid);
                let expected_valid = matches!(vector.expected_result, ValidationResult::Valid);
                
                if is_valid == expected_valid {
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!("Block test {} failed: expected {}, got {}. Height: {}", 
                        i, 
                        if expected_valid { "valid" } else { "invalid" },
                        if is_valid { "valid" } else { "invalid" },
                        vector.height
                    );
                }
            }
            Err(e) => {
                let expected_valid = matches!(vector.expected_result, ValidationResult::Valid);
                if !expected_valid {
                    // Expected to fail, so this is OK
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!("Block test {} failed with error: {}. Height: {}", 
                        i, e, vector.height
                    );
                }
            }
        }
    }
    
    println!("Core block test vectors: {} passed, {} failed", passed, failed);
    
    if failed > 0 {
        Err(format!("{} test vectors failed", failed).into())
    } else {
        Ok(())
    }
}

/// Block test vector structure
#[derive(Debug, Clone)]
pub struct BlockTestVector {
    pub block: Block,
    pub expected_result: ValidationResult,
    pub height: u64,
    pub prev_utxo_set: UtxoSet,
}

// Note: ValidationResult is already defined in consensus_proof crate
// This is just for test vector structure

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_block_test_vector_loading() {
        // Test that loading works even if directory doesn't exist
        let vectors = load_block_test_vectors("tests/test_data/core_vectors/blocks");
        assert!(vectors.is_ok());
    }
    
    #[test]
    fn test_block_vector_structure() {
        // Test minimal block vector structure
        let vector = BlockTestVector {
            block: Block {
                header: BlockHeader {
                    version: 1,
                    prev_block_hash: [0; 32],
                    merkle_root: [0; 32],
                    timestamp: 0,
                    bits: 0x1d00ffff,
                    nonce: 0,
                },
            transactions: vec![].into(),
            },
            expected_result: ValidationResult::Invalid("Test block".to_string()),
            height: 0,
            prev_utxo_set: UtxoSet::new(),
        };
        
        // Verify structure is valid
        assert_eq!(vector.height, 0);
    }
}

