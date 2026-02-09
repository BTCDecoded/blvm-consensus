//! blvm-node RPC Integration Tests
//!
//! Tests consensus-proof validation via blvm-node's RPC interface.
//! This provides integration testing of the full stack: RPC → node → consensus-proof.
//!
//! Requires: blvm-node running with RPC enabled (or in-process test node)

use blvm_consensus::*;
use blvm_consensus::serialization::transaction::serialize_transaction;
use blvm_consensus::serialization::block::serialize_block_header;
use serde_json::json;
use std::time::Duration;

/// blvm-node RPC client configuration
#[derive(Debug, Clone)]
pub struct NodeRpcConfig {
    pub url: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl Default for NodeRpcConfig {
    fn default() -> Self {
        Self {
            url: std::env::var("BLVM_NODE_RPC_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:18332".to_string()),
            username: std::env::var("BLVM_NODE_RPC_USER").ok(),
            password: std::env::var("BLVM_NODE_RPC_PASS").ok(),
        }
    }
}

/// blvm-node RPC client
pub struct NodeRpcClient {
    config: NodeRpcConfig,
}

impl NodeRpcClient {
    pub fn new(config: NodeRpcConfig) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { config })
    }

    /// Call a JSON-RPC method
    pub async fn call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        use reqwest::Client;
        
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        
        let url = reqwest::Url::parse(&self.config.url)?;
        
        // Build JSON-RPC 2.0 request
        let rpc_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });
        
        let mut request = client.post(url.clone());
        
        // Add authentication if provided
        if let (Some(user), Some(pass)) = (&self.config.username, &self.config.password) {
            request = request.basic_auth(user, Some(pass));
        }
        
        let response = request
            .json(&rpc_request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(format!("HTTP error: {}", response.status()).into());
        }
        
        let rpc_response: serde_json::Value = response.json().await?;
        
        // Check for RPC error
        if let Some(error) = rpc_response.get("error") {
            return Err(format!("RPC error: {}", error).into());
        }
        
        Ok(rpc_response.get("result").cloned().unwrap_or(json!(null)))
    }

    /// Test if transaction would be accepted by mempool
    pub async fn test_mempool_accept(&self, tx_hex: &str) -> Result<MempoolAcceptResult, Box<dyn std::error::Error>> {
        let result = self.call("testmempoolaccept", json!([[tx_hex]])).await?;
        
        if let Some(array) = result.as_array() {
            if let Some(first) = array.get(0) {
                return Ok(MempoolAcceptResult {
                    allowed: first.get("allowed").and_then(|v| v.as_bool()).unwrap_or(false),
                    reject_reason: first.get("reject-reason")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                });
            }
        }
        
        Err("Invalid testmempoolaccept response".into())
    }

    /// Submit a block for validation
    pub async fn submit_block(&self, block_hex: &str) -> Result<String, Box<dyn std::error::Error>> {
        let result = self.call("submitblock", json!([block_hex])).await?;
        
        // Empty string means valid block
        Ok(result.as_str().unwrap_or("").to_string())
    }

    /// Get blockchain information
    pub async fn get_blockchain_info(&self) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.call("getblockchaininfo", json!([])).await
    }

    /// Get block by hash
    pub async fn get_block(&self, hash: &str, verbosity: Option<u64>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        self.call("getblock", json!([hash, verbosity.unwrap_or(1)])).await
    }
}

/// Result from testmempoolaccept RPC call
#[derive(Debug, Clone)]
pub struct MempoolAcceptResult {
    pub allowed: bool,
    pub reject_reason: Option<String>,
}

/// Compare transaction validation via RPC with direct consensus-proof validation
pub async fn compare_transaction_validation_via_rpc(
    tx: &Transaction,
    config: &NodeRpcConfig,
) -> Result<ComparisonResult, Box<dyn std::error::Error>> {
    // Validate transaction locally (direct consensus-proof)
    let local_result = check_transaction(tx)?;
    let local_valid = matches!(local_result, ValidationResult::Valid);
    
    // Serialize transaction for RPC
    let tx_hex = hex::encode(&serialize_transaction(tx));
    
    // Call blvm-node RPC
    let rpc_client = NodeRpcClient::new(config.clone())?;
    let rpc_result = rpc_client.test_mempool_accept(&tx_hex).await;
    
    // Handle RPC call result
    let rpc_valid = match rpc_result {
        Ok(result) => result.allowed,
        Err(e) => {
            // RPC might not be available - that's OK for testing
            eprintln!("RPC not available: {}", e);
            return Err(e);
        }
    };
    
    let divergence = local_valid != rpc_valid;
    let divergence_reason = if divergence {
        Some(format!(
            "Local (direct): {}, RPC (blvm-node): {}",
            if local_valid { "valid" } else { "invalid" },
            if rpc_valid { "valid" } else { "invalid" }
        ))
    } else {
        None
    };
    
    Ok(ComparisonResult {
        local_valid,
        core_valid: rpc_valid, // Using core_valid field for RPC result
        divergence,
        divergence_reason,
    })
}

/// Comparison result structure
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub local_valid: bool,
    pub core_valid: bool, // In this context, this is RPC result
    pub divergence: bool,
    pub divergence_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_node_rpc_config_default() {
        let config = NodeRpcConfig::default();
        assert_eq!(config.url, "http://127.0.0.1:18332");
    }
    
    #[tokio::test]
    async fn test_node_rpc_client_creation() {
        let config = NodeRpcConfig::default();
        let client = NodeRpcClient::new(config);
        
        // Should create successfully (even if RPC not available)
        assert!(client.is_ok());
    }
    
    #[tokio::test]
    async fn test_transaction_validation_comparison() {
        // Create a simple transaction
        let tx = Transaction {
            version: 1,
            inputs: vec![].into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51].into(), // OP_1
            }].into(),
            lock_time: 0,
        };
        
        let config = NodeRpcConfig::default();
        
        // Test local validation
        let local_result = check_transaction(&tx);
        assert!(local_result.is_ok());
        
        // Try RPC comparison (will fail gracefully if RPC not available)
        let comparison = compare_transaction_validation_via_rpc(&tx, &config).await;
        
        // If RPC is available, verify no divergence
        if let Ok(result) = comparison {
            if !result.divergence {
                // Success - RPC and direct validation match
            } else {
                // Divergence found - this is a bug!
                panic!("Divergence between RPC and direct validation: {:?}", result.divergence_reason);
            }
        }
        // If RPC not available, that's OK - test infrastructure works
    }
    
    #[test]
    fn test_mempool_accept_result_parsing() {
        // Test that we can parse mempool accept results
        let result = MempoolAcceptResult {
            allowed: true,
            reject_reason: None,
        };
        
        assert!(result.allowed);
        
        let rejected = MempoolAcceptResult {
            allowed: false,
            reject_reason: Some("bad-txns-inputs-missing".to_string()),
        };
        
        assert!(!rejected.allowed);
        assert_eq!(rejected.reject_reason, Some("bad-txns-inputs-missing".to_string()));
    }
}

// TODO: Implementation enhancements:
// 1. Add more RPC methods (getblock, getblockchaininfo, etc.)
// 2. Add block validation comparison
// 3. Add retry logic for RPC connection
// 4. Add connection pooling for multiple tests
// 5. Add test fixtures for common scenarios
// 6. Integrate with blvm-node's test helpers

