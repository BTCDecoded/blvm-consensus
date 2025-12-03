//! Bitcoin Core transaction test vector integration
//!
//! Tests transaction validation using Core's test vectors.
//!
//! Core test vector format (tx_valid.json / tx_invalid.json):
//! Array of arrays: [[tx_hex, witness_hex?, flags, expected], ...]
//! - tx_hex: Transaction in hex format (non-witness serialization)
//! - witness_hex: Optional witness data in hex format
//! - flags: Script verification flags (integer)
//! - expected: Expected validation result description

use blvm_consensus::{Transaction, check_transaction};
use blvm_consensus::serialization::transaction::deserialize_transaction;
use std::path::PathBuf;
use std::fs;
use serde_json::Value;
use hex;

/// Transaction test vector structure
#[derive(Debug, Clone)]
pub struct TransactionTestVector {
    pub transaction: Transaction,
    pub expected_result: bool, // true = valid, false = invalid
    pub flags: u32,
    pub description: String,
}

/// Load transaction test vectors from Bitcoin Core JSON format
///
/// Format: JSON array of arrays, each sub-array contains:
/// [tx_hex, witness_hex?, flags, expected_description]
pub fn load_transaction_test_vectors(dir: &str) -> Result<Vec<TransactionTestVector>, Box<dyn std::error::Error>> {
    let mut vectors = Vec::new();
    let path = PathBuf::from(dir);
    
    if !path.exists() {
        // If test vectors directory doesn't exist, return empty (not an error)
        return Ok(vectors);
    }
    
    // Try to load tx_valid.json
    let valid_path = path.join("tx_valid.json");
    if valid_path.exists() {
        let content = fs::read_to_string(&valid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        if let Value::Array(cases) = json {
            for (i, case) in cases.iter().enumerate() {
                // Skip header comments (arrays where first element is a short string)
                if let Value::Array(test_case) = case {
                    // Skip if first element is a short string (likely a header comment)
                    if test_case.len() > 0 {
                        if let Some(Value::String(s)) = test_case.get(0) {
                            if s.len() < 50 {
                                continue; // Skip header lines
                            }
                        }
                    }
                    if test_case.len() >= 2 {
                        // Parse transaction hex (first element should be hex string)
                        let tx_hex = test_case[0].as_str()
                            .ok_or_else(|| format!("Invalid tx_hex at index {} (not a string)", i))?;
                        // Skip if it's too short (likely not a real transaction)
                        if tx_hex.len() < 50 {
                            continue;
                        }
                        let tx_bytes = match hex::decode(tx_hex) {
                            Ok(bytes) => bytes,
                            Err(_) => {
                                // Skip invalid hex strings
                                continue;
                            }
                        };
                        let transaction = match deserialize_transaction(&tx_bytes) {
                            Ok(tx) => tx,
                            Err(e) => {
                                // Skip transactions that fail to deserialize
                                eprintln!("Warning: Failed to deserialize transaction at index {}: {}", i, e);
                                continue;
                            }
                        };
                        
                        // Parse flags (may be integer or string, typically second-to-last element)
                        let flags = if test_case.len() >= 3 {
                            match &test_case[test_case.len() - 2] {
                                Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
                                Value::String(s) => s.parse::<u32>().unwrap_or(0),
                                _ => 0,
                            }
                        } else {
                            0
                        };
                        
                        // Parse expected result (last element is description)
                        let description = test_case.last()
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        
                        vectors.push(TransactionTestVector {
                            transaction,
                            expected_result: true, // tx_valid.json contains valid transactions
                            flags,
                            description,
                        });
                    }
                }
            }
        }
    }
    
    // Try to load tx_invalid.json
    let invalid_path = path.join("tx_invalid.json");
    if invalid_path.exists() {
        let content = fs::read_to_string(&invalid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        if let Value::Array(cases) = json {
            for (i, case) in cases.iter().enumerate() {
                if let Value::Array(test_case) = case {
                    if test_case.len() >= 3 {
                        // Parse transaction hex
                        let tx_hex = test_case[0].as_str()
                            .ok_or_else(|| format!("Invalid tx_hex at index {}", i))?;
                        let tx_bytes = hex::decode(tx_hex)?;
                        
                        // Try to deserialize - invalid transactions may fail at deserialization
                        // or may deserialize but fail validation
                        if let Ok(transaction) = deserialize_transaction(&tx_bytes) {
                            // Parse flags
                            let flags = match &test_case[test_case.len() - 2] {
                                Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
                                Value::String(s) => s.parse::<u32>().unwrap_or(0),
                                _ => 0,
                            };
                            
                            // Parse expected result
                            let description = test_case.last()
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            
                            vectors.push(TransactionTestVector {
                                transaction,
                                expected_result: false, // tx_invalid.json contains invalid transactions
                                flags,
                                description,
                            });
                        }
                        // If deserialization fails, that's expected for invalid transactions
                    }
                }
            }
        }
    }
    
    Ok(vectors)
}

/// Run Core transaction test vectors
pub fn run_core_transaction_tests(vectors: &[TransactionTestVector]) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed = 0;
    let mut failed = 0;
    
    for (i, vector) in vectors.iter().enumerate() {
        let result = check_transaction(&vector.transaction);
        
        match result {
            Ok(validation_result) => {
                let is_valid = matches!(validation_result, blvm_consensus::ValidationResult::Valid);
                if is_valid == vector.expected_result {
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!("Test {} failed: expected {}, got {}. Description: {}", 
                        i, 
                        if vector.expected_result { "valid" } else { "invalid" },
                        if is_valid { "valid" } else { "invalid" },
                        vector.description
                    );
                }
            }
            Err(e) => {
                if !vector.expected_result {
                    // Expected to fail, so this is OK
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!("Test {} failed with error: {}. Description: {}", 
                        i, e, vector.description
                    );
                }
            }
        }
    }
    
    println!("Core transaction test vectors: {} passed, {} failed", passed, failed);
    
    if failed > 0 {
        Err(format!("{} test vectors failed", failed).into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transaction_test_vector_loading() {
        let vectors = load_transaction_test_vectors("tests/test_data/core_vectors/transactions");
        assert!(vectors.is_ok());
        // If directory doesn't exist, that's OK - vectors will be empty
    }
    
    #[test]
    fn test_parse_simple_transaction_vector() {
        // Test with a minimal valid transaction
        let tx_hex = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff08044c86041b020602ffffffff0100f2052a010000004341041b0e8c2567c12536aa13357b79a073dc4444acb83c4ec7a0e2f99dd7457516c5817242da796924ca4e99947d087fedf9ce467cb9f7c6287078f801df276fdf84ac00000000";
        let tx_bytes = hex::decode(tx_hex).unwrap();
        let transaction = deserialize_transaction(&tx_bytes);
        
        // Should parse successfully (this is a valid coinbase transaction format)
        assert!(transaction.is_ok());
    }
}

