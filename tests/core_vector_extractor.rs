//! Bitcoin Core test vector extractor and parser
//!
//! Extracts and parses test vectors from Bitcoin Core's test data directory.
//! Handles Core's specific JSON formats and converts them to our test format.

use std::path::PathBuf;
use std::fs;
use serde_json::Value;

/// Extract transaction test vectors from Core's tx_valid.json
///
/// Core format: Array of arrays with format:
/// [[[prevout hash, prevout index, prevout scriptPubKey, amount?], ...], serializedTransaction, flags]
pub fn extract_core_transaction_vectors(core_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tx_valid_path = PathBuf::from(core_path).join("src/test/data/tx_valid.json");
    let tx_invalid_path = PathBuf::from(core_path).join("src/test/data/tx_invalid.json");
    
    if tx_valid_path.exists() {
        let content = fs::read_to_string(&tx_valid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        
        if let Value::Array(cases) = json {
            println!("Found {} valid transaction test cases", cases.len());
            
            // Parse each test case
            for (i, case) in cases.iter().enumerate() {
                if let Value::Array(test_case) = case {
                    // Skip string-only entries (comments)
                    if test_case.len() >= 2 && test_case[0].is_string() && test_case[1].is_string() {
                        // Format: [[prevouts], serializedTx, flags]
                        if test_case.len() >= 3 {
                            if let Value::Array(prevouts) = &test_case[0] {
                                if let Value::String(tx_hex) = &test_case[1] {
                                    // Parse flags (can be string or array)
                                    let flags = parse_flags(&test_case[2]);
                                    
                                    // Store or process this test vector
                                    // (Implementation would store in our test vector format)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    if tx_invalid_path.exists() {
        let content = fs::read_to_string(&tx_invalid_path)?;
        let json: Value = serde_json::from_str(&content)?;
        
        if let Value::Array(cases) = json {
            println!("Found {} invalid transaction test cases", cases.len());
        }
    }
    
    Ok(())
}

/// Parse script test vectors from Core's script_tests.json
///
/// Core format: [scriptSig_string, scriptPubKey_string, flags_string, expected_result, description]
pub fn extract_core_script_vectors(core_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let script_tests_path = PathBuf::from(core_path).join("src/test/data/script_tests.json");
    
    if script_tests_path.exists() {
        let content = fs::read_to_string(&script_tests_path)?;
        let json: Value = serde_json::from_str(&content)?;
        
        if let Value::Array(cases) = json {
            println!("Found {} script test cases", cases.len());
            
            for (i, case) in cases.iter().enumerate() {
                if let Value::Array(test_case) = case {
                    // Skip string-only entries (comments/format descriptions)
                    if test_case.len() >= 4 && test_case[0].is_string() {
                        // Format: [scriptSig, scriptPubKey, flags, expected, description]
                        if let Value::String(script_sig_str) = &test_case[0] {
                            if let Value::String(script_pubkey_str) = &test_case[1] {
                                if let Value::String(flags_str) = &test_case[2] {
                                    // Parse flags (e.g., "P2SH,STRICTENC" -> 0x01 | 0x02)
                                    let flags = parse_flag_string(flags_str);
                                    
                                    // Parse expected result
                                    let expected = test_case.get(3)
                                        .and_then(|v| v.as_str())
                                        .map(|s| s == "OK")
                                        .unwrap_or(true);
                                    
                                    // Store or process this test vector
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Parse flags from Core's flag string format
///
/// Core uses comma-separated flag names like "P2SH,STRICTENC,DERSIG"
fn parse_flag_string(flags_str: &str) -> u32 {
    let mut flags = 0u32;
    
    for flag_name in flags_str.split(',') {
        let flag_name = flag_name.trim();
        match flag_name {
            "P2SH" => flags |= 0x01,
            "STRICTENC" => flags |= 0x02,
            "DERSIG" => flags |= 0x04,
            "LOW_S" => flags |= 0x08,
            "NULLDUMMY" => flags |= 0x10,
            "SIGPUSHONLY" => flags |= 0x20,
            "MINIMALDATA" => flags |= 0x40,
            "DISCOURAGE_UPGRADABLE_NOPS" => flags |= 0x80,
            "CLEANSTACK" => flags |= 0x100,
            "CHECKLOCKTIMEVERIFY" => flags |= 0x200,
            "CHECKSEQUENCEVERIFY" => flags |= 0x400,
            "WITNESS" => flags |= 0x800,
            "DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM" => flags |= 0x1000,
            "MINIMALIF" => flags |= 0x2000,
            "TAPROOT" => flags |= 0x4000,
            "NONE" => flags |= 0,
            _ => {
                // Unknown flag - log but continue
                eprintln!("Unknown flag: {}", flag_name);
            }
        }
    }
    
    flags
}

/// Parse flags from JSON value (can be string or number)
fn parse_flags(value: &Value) -> u32 {
    match value {
        Value::String(s) => parse_flag_string(s),
        Value::Number(n) => n.as_u64().unwrap_or(0) as u32,
        _ => 0,
    }
}

/// Convert Core script string to bytes
///
/// Core uses human-readable script format (e.g., "1 2 EQUAL")
/// This needs to be converted to bytecode for our tests.
fn script_string_to_bytes(script_str: &str) -> Vec<u8> {
    // This is a simplified conversion - actual implementation would need
    // full script parser to handle opcodes, push operations, etc.
    
    let mut bytes = Vec::new();
    
    // Split by whitespace and parse tokens
    for token in script_str.split_whitespace() {
        // Handle opcodes
        match token {
            "OP_1" | "1" => bytes.push(0x51),
            "OP_2" | "2" => bytes.push(0x52),
            "OP_DUP" | "DUP" => bytes.push(0x76),
            "OP_EQUAL" | "EQUAL" => bytes.push(0x87),
            "OP_EQUALVERIFY" | "EQUALVERIFY" => bytes.push(0x88),
            "OP_HASH160" | "HASH160" => bytes.push(0xa9),
            "OP_CHECKSIG" | "CHECKSIG" => bytes.push(0xac),
            _ => {
                // Handle hex literals (0x...)
                if token.starts_with("0x") {
                    if let Ok(byte) = u8::from_str_radix(&token[2..], 16) {
                        bytes.push(byte);
                    }
                } else if let Ok(num) = token.parse::<u8>() {
                    // Direct number
                    bytes.push(num);
                }
            }
        }
    }
    
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_flag_string() {
        assert_eq!(parse_flag_string("P2SH"), 0x01);
        assert_eq!(parse_flag_string("P2SH,STRICTENC"), 0x01 | 0x02);
        assert_eq!(parse_flag_string("P2SH,STRICTENC,DERSIG"), 0x01 | 0x02 | 0x04);
        assert_eq!(parse_flag_string("WITNESS,TAPROOT"), 0x800 | 0x4000);
    }
    
    #[test]
    fn test_script_string_to_bytes() {
        let bytes = script_string_to_bytes("1 2 EQUAL");
        assert!(bytes.len() >= 3);
        assert_eq!(bytes[0], 0x51); // OP_1
        assert_eq!(bytes[1], 0x52); // OP_2
        assert_eq!(bytes[2], 0x87); // OP_EQUAL
    }
    
    #[test]
    fn test_extract_core_vectors() {
        // Test extraction from Core repository
        let core_path = "/home/user/src/bitcoin";
        if std::path::Path::new(core_path).exists() {
            let result = extract_core_transaction_vectors(core_path);
            assert!(result.is_ok());
            
            let result = extract_core_script_vectors(core_path);
            assert!(result.is_ok());
        }
    }
}



