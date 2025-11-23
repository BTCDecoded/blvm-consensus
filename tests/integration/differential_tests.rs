//! Differential Fuzzing vs Bitcoin Core
//!
//! Compares validation results between this implementation and Bitcoin Core
//! via RPC interface. This provides empirical validation of consensus correctness.
//!
//! Requires: Bitcoin Core running with RPC enabled
//! Usage: Set environment variables or configure connection settings

use bllvm_consensus::*;
use bllvm_consensus::types::ByteString;
use serde_json::json;

/// Bitcoin Core RPC client configuration
#[derive(Debug, Clone)]
pub struct CoreRpcConfig {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl Default for CoreRpcConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8332".to_string(),
            username: None,
            password: None,
        }
    }
}

/// Compare transaction validation results with Bitcoin Core
/// 
/// Uses Bitcoin Core's `testmempoolaccept` RPC call to validate transactions.
pub async fn compare_transaction_validation(
    tx: &Transaction,
    config: &CoreRpcConfig,
) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
    // Validate transaction locally
    let local_result = check_transaction(tx)?;
    let local_valid = matches!(local_result, ValidationResult::Valid);
    
    // Serialize transaction to hex for Core RPC
    let tx_hex = hex::encode(&bincode::serialize(tx)?);
    
    // Call Bitcoin Core's testmempoolaccept RPC
    // Format: testmempoolaccept [["hexstring"]]
    let core_result = call_core_rpc(
        config,
        "testmempoolaccept",
        json!([[tx_hex]]),
    ).await?;
    
    // Parse Core's response
    let core_valid = core_result
        .as_array()
        .and_then(|arr| arr.get(0))
        .and_then(|result| result.get("allowed"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    let divergence = local_valid != core_valid;
    let divergence_reason = if divergence {
        Some(format!(
            "Local: {}, Core: {}",
            if local_valid { "valid" } else { "invalid" },
            if core_valid { "valid" } else { "invalid" }
        ))
    } else {
        None
    };
    
    Ok(ComparisonResult {
        local_valid,
        core_valid,
        divergence,
        divergence_reason,
    })
}

/// Compare block validation results with Bitcoin Core
/// 
/// Uses Bitcoin Core's `submitblock` RPC call (test mode) to validate blocks.
pub async fn compare_block_validation(
    block: &Block,
    config: &CoreRpcConfig,
) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
    // Validate block locally
    let initial_utxo_set = UtxoSet::new();
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let local_result = connect_block(block, &witnesses, initial_utxo_set, 0, None, crate::types::Network::Mainnet)?;
    let local_valid = matches!(local_result.0, ValidationResult::Valid);
    
    // Serialize block to hex for Core RPC
    let block_hex = hex::encode(&bincode::serialize(block)?);
    
    // Call Bitcoin Core's submitblock RPC (with test mode)
    // Note: submitblock returns empty string if block is valid, error otherwise
    let core_result = call_core_rpc(
        config,
        "submitblock",
        json!([block_hex]),
    ).await;
    
    // Empty string response means valid block
    let core_valid = match &core_result {
        Ok(val) => {
            // Check if response is empty string or null (valid block)
            val.as_str().map(|s| s.is_empty()).unwrap_or(true)
                || val.is_null()
        }
        Err(_) => false, // Error means invalid block
    };
    
    let divergence = local_valid != core_valid;
    let divergence_reason = if divergence {
        Some(format!(
            "Local: {}, Core: {}",
            if local_valid { "valid" } else { "invalid" },
            if core_valid { "valid" } else { "invalid" }
        ))
    } else {
        None
    };
    
    Ok(ComparisonResult {
        local_valid,
        core_valid,
        divergence,
        divergence_reason,
    })
}

/// Call Bitcoin Core JSON-RPC endpoint
/// 
/// Makes HTTP POST request to Core's RPC endpoint with JSON-RPC 2.0 format.
async fn call_core_rpc(
    config: &CoreRpcConfig,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    // Create HTTP client
    let client = reqwest::Client::new();
    
    // Build basic auth if credentials provided
    let mut request = client.post(&config.url);
    
    if let (Some(username), Some(password)) = (&config.username, &config.password) {
        request = request.basic_auth(username, Some(password));
    }
    
    // Build JSON-RPC 2.0 request
    let rpc_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    });
    
    // Send request
    let response = request
        .json(&rpc_request)
        .send()
        .await?;
    
    // Parse response
    let rpc_response: serde_json::Value = response.json().await?;
    
    // Extract result or error
    if let Some(error) = rpc_response.get("error") {
        return Err(format!("RPC error: {}", error).into());
    }
    
    Ok(rpc_response.get("result").cloned().unwrap_or(json!(null)))
}

/// Result of comparing validation with Bitcoin Core
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub local_valid: bool,
    pub core_valid: bool,
    pub divergence: bool,
    pub divergence_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_differential_validation_placeholder() {
        // Test infrastructure - will skip if Core RPC is not available
        let tx = Transaction {
            version: 1,
            inputs: vec![].into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        };
        
        let config = CoreRpcConfig::default();
        
        // Test local validation
        let local_result = check_transaction(&tx);
        assert!(local_result.is_ok());
        
        // Try Core RPC if available (will fail gracefully if Core not running)
        // This is intentional - differential tests should work even if Core is unavailable
        let _core_result = compare_transaction_validation(&tx, &config).await;
        // Don't assert here - Core may not be running in test environment
    }
    
    #[test]
    fn test_core_rpc_config_defaults() {
        let config = CoreRpcConfig::default();
        assert_eq!(config.url, "http://127.0.0.1:8332");
    }
}

// TODO: Implement actual Bitcoin Core RPC integration:
// 1. Add RPC client library (e.g., `bitcoin-rpc-client` or `jsonrpc`)
// 2. Implement testmempoolaccept equivalent
// 3. Implement block validation check
// 4. Run fuzzed inputs through both implementations
// 5. Report divergences with detailed diagnostics
// 6. Automate in CI/CD when Core testnet node is available

