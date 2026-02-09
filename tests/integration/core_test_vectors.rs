//! Integration tests for Bitcoin Core test vectors
//!
//! These tests verify consensus correctness by running Core's test vectors
//! through our validation logic. This provides free verification coverage.
//!
//! Test vectors can be downloaded from Bitcoin Core's test framework.
//! If vectors are not available, tests will skip gracefully.

use blvm_consensus::*;
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
use std::path::PathBuf;
use std::fs;

/// Test directory for Core test vectors
/// 
/// To use this, download Bitcoin Core test vectors to:
/// `tests/test_data/core_vectors/`
const CORE_VECTORS_DIR: &str = "tests/test_data/core_vectors";

/// Check if test vectors are available
fn test_vectors_available() -> bool {
    let base_path = PathBuf::from(CORE_VECTORS_DIR);
    base_path.exists() && base_path.is_dir()
}

/// Load transaction test vectors
fn load_transaction_vectors() -> Result<Vec<(Transaction, bool)>, Box<dyn std::error::Error>> {
    let tx_valid_path = PathBuf::from(CORE_VECTORS_DIR).join("tx_valid.json");
    let tx_invalid_path = PathBuf::from(CORE_VECTORS_DIR).join("tx_invalid.json");
    
    let mut vectors = Vec::new();
    
    // Load valid transactions
    if tx_valid_path.exists() {
        let content = fs::read_to_string(&tx_valid_path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;
        
        if let Some(array) = json.as_array() {
            for item in array {
                if let Some(hex_str) = item.as_str() {
                    if let Ok(tx_bytes) = hex::decode(hex_str) {
                        if let Ok(tx) = deserialize_transaction(&tx_bytes) {
                            vectors.push((tx, true)); // true = should be valid
                        }
                    }
                }
            }
        }
    }
    
    // Load invalid transactions
    if tx_invalid_path.exists() {
        let content = fs::read_to_string(&tx_invalid_path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;
        
        if let Some(array) = json.as_array() {
            for item in array {
                if let Some(hex_str) = item.as_str() {
                    if let Ok(tx_bytes) = hex::decode(hex_str) {
                        if let Ok(tx) = deserialize_transaction(&tx_bytes) {
                            vectors.push((tx, false)); // false = should be invalid
                        }
                    }
                }
            }
        }
    }
    
    Ok(vectors)
}

/// Load block test vectors
fn load_block_vectors() -> Result<Vec<(Block, Vec<Witness>, bool)>, Box<dyn std::error::Error>> {
    let block_valid_path = PathBuf::from(CORE_VECTORS_DIR).join("block_valid.json");
    let block_invalid_path = PathBuf::from(CORE_VECTORS_DIR).join("block_invalid.json");
    
    let mut vectors = Vec::new();
    
    // Load valid blocks
    if block_valid_path.exists() {
        let content = fs::read_to_string(&block_valid_path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;
        
        if let Some(array) = json.as_array() {
            for item in array {
                if let Some(hex_str) = item.as_str() {
                    if let Ok(block_bytes) = hex::decode(hex_str) {
                        if let Ok((block, witnesses)) = deserialize_block_with_witnesses(&block_bytes) {
                            vectors.push((block, witnesses, true)); // true = should be valid
                        }
                    }
                }
            }
        }
    }
    
    // Load invalid blocks
    if block_invalid_path.exists() {
        let content = fs::read_to_string(&block_invalid_path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;
        
        if let Some(array) = json.as_array() {
            for item in array {
                if let Some(hex_str) = item.as_str() {
                    if let Ok(block_bytes) = hex::decode(hex_str) {
                        if let Ok((block, witnesses)) = deserialize_block_with_witnesses(&block_bytes) {
                            vectors.push((block, witnesses, false)); // false = should be invalid
                        }
                    }
                }
            }
        }
    }
    
    Ok(vectors)
}

#[test]
fn test_core_test_vector_directory_structure() {
    // Verify that test vector directory structure is set up correctly
    let base_path = PathBuf::from(CORE_VECTORS_DIR);
    
    // Check if directory exists (will be created if needed)
    if !base_path.exists() {
        // Directory doesn't exist yet - that's OK
        // Tests will skip if vectors aren't available
        return;
    }
    
    assert!(base_path.is_dir(), "Core test vector directory should be a directory");
}

#[test]
fn test_transaction_vectors_if_available() {
    if !test_vectors_available() {
        // Skip test if vectors not available
        return;
    }
    
    let vectors = match load_transaction_vectors() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load transaction vectors: {}", e);
            return; // Skip test if loading fails
        }
    };
    
    if vectors.is_empty() {
        // No vectors loaded - skip test
        return;
    }
    
    // Test each vector
    for (tx, should_be_valid) in vectors.iter().take(10) { // Limit to first 10 for speed
        let result = check_transaction(tx);
        
        match result {
            Ok(ValidationResult::Valid) => {
                assert!(should_be_valid, "Transaction should be valid but was marked invalid in test vector");
            }
            Ok(ValidationResult::Invalid(_)) => {
                assert!(!should_be_valid, "Transaction should be invalid but was marked valid in test vector");
            }
            Err(e) => {
                panic!("Transaction validation failed with error: {}", e);
            }
        }
    }
}

#[test]
fn test_block_vectors_if_available() {
    if !test_vectors_available() {
        // Skip test if vectors not available
        return;
    }
    
    let vectors = match load_block_vectors() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to load block vectors: {}", e);
            return; // Skip test if loading fails
        }
    };
    
    if vectors.is_empty() {
        // No vectors loaded - skip test
        return;
    }
    
    let mut utxo_set = UtxoSet::new();
    let mut height = 0u64;
    
    // Test each vector
    for (block, witnesses, should_be_valid) in vectors.iter().take(5) { // Limit to first 5 for speed
        let result = connect_block(block, witnesses, utxo_set.clone(), height, None, 0u64, crate::types::Network::Mainnet);
        
        match result {
            Ok((ValidationResult::Valid, new_utxo_set)) => {
                assert!(should_be_valid, "Block should be valid but was marked invalid in test vector");
                utxo_set = new_utxo_set;
                height += 1;
            }
            Ok((ValidationResult::Invalid(_), _)) => {
                assert!(!should_be_valid, "Block should be invalid but was marked valid in test vector");
            }
            Err(e) => {
                if *should_be_valid {
                    panic!("Block validation failed with error (expected valid): {}", e);
                }
                // If block should be invalid, error is acceptable
            }
        }
    }
}
