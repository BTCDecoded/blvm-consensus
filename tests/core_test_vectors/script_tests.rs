//! Bitcoin Core script test vector integration
//!
//! Tests script execution using Core's script test vectors.
//! These cover opcode behavior, edge cases, and validation rules.
//!
//! Core test vector format (script_valid.json / script_invalid.json):
//! Array of arrays: [[scriptSig_hex, scriptPubKey_hex, flags, expected, description], ...]
//! - scriptSig_hex: ScriptSig in hex format
//! - scriptPubKey_hex: ScriptPubKey in hex format
//! - flags: Script verification flags (integer)
//! - expected: Expected validation result (true/false)
//! - description: Human-readable description

use std::path::PathBuf;
use std::fs;
use serde_json::Value;
use hex;
use bllvm_consensus::script::verify_script;

/// Script test vector structure
#[derive(Debug, Clone)]
pub struct ScriptTestVector {
    pub script_sig: Vec<u8>,
    pub script_pubkey: Vec<u8>,
    pub expected_result: bool,
    pub flags: u32,
    pub description: String,
}

/// Load script test vectors from Bitcoin Core JSON format
///
/// Core uses script_tests.json with format:
/// [scriptSig_string, scriptPubKey_string, flags_string, expected_result, description]
/// 
/// Script strings can be in human-readable format (e.g., "1 2 EQUAL") or hex format.
/// Flags are comma-separated strings (e.g., "P2SH,STRICTENC").
pub fn load_script_test_vectors(dir: &str) -> Result<Vec<ScriptTestVector>, Box<dyn std::error::Error>> {
    let mut vectors = Vec::new();
    let path = PathBuf::from(dir);
    
    if !path.exists() {
        return Ok(vectors);
    }
    
    // Try to load script_valid.json
    let valid_path = path.join("script_valid.json");
    if valid_path.exists() {
        let content = fs::read_to_string(&valid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        if let Value::Array(cases) = json {
            for (i, case) in cases.iter().enumerate() {
                if let Value::Array(test_case) = case {
                    if test_case.len() >= 4 {
                        // Parse scriptSig hex
                        let script_sig_hex = test_case[0].as_str()
                            .ok_or_else(|| format!("Invalid scriptSig hex at index {}", i))?;
                        let script_sig = hex::decode(script_sig_hex)?;
                        
                        // Parse scriptPubKey hex
                        let script_pubkey_hex = test_case[1].as_str()
                            .ok_or_else(|| format!("Invalid scriptPubKey hex at index {}", i))?;
                        let script_pubkey = hex::decode(script_pubkey_hex)?;
                        
                        // Parse flags (may be integer or string)
                        let flags = match &test_case[2] {
                            Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
                            Value::String(s) => {
                                // Try to parse as hex first, then decimal
                                if s.starts_with("0x") {
                                    u32::from_str_radix(&s[2..], 16).unwrap_or(0)
                                } else {
                                    s.parse::<u32>().unwrap_or(0)
                                }
                            },
                            _ => 0,
                        };
                        
                        // Parse expected result (true for script_valid.json)
                        let expected_result = test_case.get(3)
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        
                        // Parse description (last element)
                        let description = test_case.last()
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        
                        vectors.push(ScriptTestVector {
                            script_sig,
                            script_pubkey,
                            expected_result,
                            flags,
                            description,
                        });
                    }
                }
            }
        }
    }
    
    // Try to load script_invalid.json
    let invalid_path = path.join("script_invalid.json");
    if invalid_path.exists() {
        let content = fs::read_to_string(&invalid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        if let Value::Array(cases) = json {
            for (i, case) in cases.iter().enumerate() {
                if let Value::Array(test_case) = case {
                    if test_case.len() >= 4 {
                        // Parse scriptSig hex
                        let script_sig_hex = test_case[0].as_str()
                            .ok_or_else(|| format!("Invalid scriptSig hex at index {}", i))?;
                        let script_sig = hex::decode(script_sig_hex)?;
                        
                        // Parse scriptPubKey hex
                        let script_pubkey_hex = test_case[1].as_str()
                            .ok_or_else(|| format!("Invalid scriptPubKey hex at index {}", i))?;
                        let script_pubkey = hex::decode(script_pubkey_hex)?;
                        
                        // Parse flags
                        let flags = match &test_case[2] {
                            Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
                            Value::String(s) => {
                                if s.starts_with("0x") {
                                    u32::from_str_radix(&s[2..], 16).unwrap_or(0)
                                } else {
                                    s.parse::<u32>().unwrap_or(0)
                                }
                            },
                            _ => 0,
                        };
                        
                        // Parse expected result (false for script_invalid.json)
                        let expected_result = test_case.get(3)
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        
                        // Parse description
                        let description = test_case.last()
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        
                        vectors.push(ScriptTestVector {
                            script_sig,
                            script_pubkey,
                            expected_result,
                            flags,
                            description,
                        });
                    }
                }
            }
        }
    }
    
    Ok(vectors)
}

/// Run Core script test vectors
pub fn run_core_script_tests(vectors: &[ScriptTestVector]) -> Result<(), Box<dyn std::error::Error>> {
    let mut passed = 0;
    let mut failed = 0;
    
    for (i, vector) in vectors.iter().enumerate() {
        let result = verify_script(&vector.script_sig, &vector.script_pubkey, None, vector.flags);
        
        match result {
            Ok(is_valid) => {
                if is_valid == vector.expected_result {
                    passed += 1;
                } else {
                    failed += 1;
                    eprintln!("Script test {} failed: expected {}, got {}. Flags: 0x{:x}. Description: {}", 
                        i, 
                        if vector.expected_result { "valid" } else { "invalid" },
                        if is_valid { "valid" } else { "invalid" },
                        vector.flags,
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
                    eprintln!("Script test {} failed with error: {}. Flags: 0x{:x}. Description: {}", 
                        i, e, vector.flags, vector.description
                    );
                }
            }
        }
    }
    
    println!("Core script test vectors: {} passed, {} failed", passed, failed);
    
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
    fn test_script_test_vector_loading() {
        let vectors = load_script_test_vectors("tests/test_data/core_vectors/scripts");
        assert!(vectors.is_ok());
        // If directory doesn't exist, that's OK - vectors will be empty
    }
    
    #[test]
    fn test_parse_simple_script_vector() {
        // Test with a simple script: OP_1 OP_1 OP_EQUAL
        let script_sig = vec![0x51]; // OP_1
        let script_pubkey = vec![0x51, 0x87]; // OP_1 OP_EQUAL
        let flags = 0u32;
        
        let result = verify_script(&script_sig, &script_pubkey, None, flags);
        assert!(result.is_ok());
        // Should evaluate to true (1 == 1)
        assert_eq!(result.unwrap(), true);
    }
}

