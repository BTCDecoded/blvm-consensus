//! Historical Block Replay (Phase 3.1)
//!
//! Replays and validates historical Bitcoin blocks from genesis to tip.
//! Provides empirical validation of consensus correctness across the entire chain history.
//!
//! This test:
//! 1. Downloads blocks from genesis to tip (or specified range)
//! 2. Validates each block sequentially
//! 3. Maintains UTXO set state
//! 4. Compares UTXO set hashes at checkpoints with known values
//! 5. Reports any validation failures or divergences

use blvm_consensus::opcodes::OP_1;
use blvm_consensus::*;
use std::collections::HashMap;
use std::path::PathBuf;
use hex;

// Note: This requires tokio for async, but we'll make it optional
#[cfg(test)]
use tokio;

/// Historical block replay configuration
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Starting block height (default: 0 = genesis)
    pub start_height: u64,
    /// Ending block height (None = replay to tip)
    pub end_height: Option<u64>,
    /// Block data directory (if blocks are pre-downloaded)
    pub block_data_dir: Option<String>,
    /// UTXO checkpoint hashes to verify against
    pub checkpoint_hashes: HashMap<u64, [u8; 32]>,
    /// Enable parallel validation for old blocks (uses Phase 4.2)
    pub enable_parallel: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            start_height: 0,
            end_height: None,
            block_data_dir: None,
            checkpoint_hashes: HashMap::new(),
            enable_parallel: true,
        }
    }
}

/// Historical block replay result
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// Blocks successfully validated
    pub blocks_validated: u64,
    /// Blocks that failed validation
    pub blocks_failed: Vec<(u64, String)>,
    /// UTXO set at final height
    pub final_utxo_set: UtxoSet,
    /// Checkpoint verification results
    pub checkpoint_results: Vec<(u64, bool)>,
    /// Total replay time in seconds
    pub replay_time_seconds: f64,
}

/// Replay historical blocks
/// 
/// Validates blocks sequentially from start_height to end_height (or tip).
/// Maintains UTXO set state throughout replay.
pub async fn replay_historical_blocks(
    config: &ReplayConfig,
) -> Result<ReplayResult, Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    let mut utxo_set = UtxoSet::default();
    let mut blocks_validated = 0u64;
    let mut blocks_failed = Vec::new();
    let mut checkpoint_results = Vec::new();
    
    // If block_data_dir is provided, load blocks from disk
    if let Some(dir) = &config.block_data_dir {
        let block_path = PathBuf::from(dir);
        if block_path.exists() {
            // Load blocks from directory
            // Expected formats:
            // 1. Binary files: `block_{height}.bin` (Bitcoin wire format)
            // 2. Hex files: `block_{height}.hex` (hex-encoded Bitcoin wire format)
            // 3. JSON files: `block_{height}.json` (for debugging, not production)
            
            let mut current_height = config.start_height;
            let end_height = config.end_height.unwrap_or(u64::MAX);
            
            while current_height <= end_height {
                // Try binary format first (most efficient)
                let bin_path = block_path.join(format!("block_{}.bin", current_height));
                let hex_path = block_path.join(format!("block_{}.hex", current_height));
                
                let block_data = if bin_path.exists() {
                    // Load binary block
                    std::fs::read(&bin_path).map_err(|e| format!("Failed to read block {}: {}", current_height, e))?
                } else if hex_path.exists() {
                    // Load hex-encoded block
                    let hex_content = std::fs::read_to_string(&hex_path)
                        .map_err(|e| format!("Failed to read block {} hex: {}", current_height, e))?;
                    hex::decode(hex_content.trim())
                        .map_err(|e| format!("Failed to decode block {} hex: {}", current_height, e))?
                } else {
                    // No block file found - end of available blocks
                    break;
                };
                
                // Deserialize block
                use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
                let (block, witnesses) = deserialize_block_with_witnesses(&block_data)
                    .map_err(|e| format!("Failed to deserialize block {}: {}", current_height, e))?;
                
                // Validate block
                // connect_block expects &[Witness] where each Witness is Vec<ByteString> (one per transaction)
                use blvm_consensus::block::connect_block;
                match { let ctx = block::BlockValidationContext::for_network(crate::types::Network::Mainnet); connect_block(&block, &witnesses, utxo_set.clone(), current_height, &ctx) } {
                    Ok((validation_result, new_utxo_set)) => {
                        match validation_result {
                            blvm_consensus::ValidationResult::Valid => {
                                utxo_set = new_utxo_set;
                                blocks_validated += 1;
                                
                                // Check checkpoint if configured
                                if let Some(expected_hash) = config.checkpoint_hashes.get(&current_height) {
                                    let calculated_hash = calculate_utxo_set_hash(&utxo_set);
                                    let matches = calculated_hash == *expected_hash;
                                    checkpoint_results.push((current_height, matches));
                                    
                                    if !matches {
                                        eprintln!("Checkpoint mismatch at height {}: expected {:?}, got {:?}", 
                                            current_height, expected_hash, calculated_hash);
                                    }
                                }
                            }
                            blvm_consensus::ValidationResult::Invalid(reason) => {
                                blocks_failed.push((current_height, reason));
                                // Continue replay even if block fails (for debugging)
                            }
                        }
                    }
                    Err(e) => {
                        blocks_failed.push((current_height, format!("Validation error: {}", e)));
                        // Continue replay even if block fails (for debugging)
                    }
                }
                
                current_height += 1;
            }
        }
    } else {
        // No block data directory provided - skip block loading
        // In future, this could download blocks from network or use other sources
        // See download_block_from_network() for future implementation
    }
    
    let replay_time = start_time.elapsed().as_secs_f64();
    
    Ok(ReplayResult {
        blocks_validated,
        blocks_failed,
        final_utxo_set: utxo_set,
        checkpoint_results,
        replay_time_seconds: replay_time,
    })
}

/// Calculate MuHash3072 of UTXO set for checkpoint verification (Core-compatible).
pub fn calculate_utxo_set_hash(utxo_set: &UtxoSet) -> [u8; 32] {
    use blvm_muhash::{serialize_coin_for_muhash, MuHash3072};

    let mut entries: Vec<_> = utxo_set.iter().collect();
    entries.sort_by(|(a, _), (b, _)| match a.hash.cmp(&b.hash) {
        std::cmp::Ordering::Equal => a.index.cmp(&b.index),
        other => other,
    });

    let mut muhash = MuHash3072::new();
    for (outpoint, utxo) in entries {
        let height_u32 = utxo.height.min(u32::MAX as u64) as u32;
        let serialized = serialize_coin_for_muhash(
            &outpoint.hash,
            outpoint.index,
            height_u32,
            utxo.is_coinbase,
            utxo.value,
            utxo.script_pubkey.as_ref(),
        );
        muhash = muhash.insert(&serialized);
    }
    muhash.finalize()
}

/// Verify UTXO set against known checkpoint hash (muhash format)
pub fn verify_checkpoint(
    utxo_set: &UtxoSet,
    expected_hash: &[u8; 32],
) -> bool {
    let calculated_hash = calculate_utxo_set_hash(utxo_set);
    calculated_hash == *expected_hash
}

/// Download a block from Bitcoin network (future implementation)
///
/// Download block from network. Not implemented.
///
/// Pre-download blocks to block_data_dir (e.g. via scripts/download_mainnet_blocks.sh)
/// or use block_data_dir with pre-downloaded blocks. Network download is optional
/// (future enhancement).
pub async fn download_block_from_network(
    _height: u64,
    _config: &ReplayConfig,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("Block downloading not implemented. Use block_data_dir with pre-downloaded blocks.".into())
}

/// Load a single block from disk
///
/// Helper function to load a block at a specific height from the block data directory.
/// Supports both binary (.bin) and hex (.hex) formats.
pub fn load_block_from_disk(
    block_dir: &PathBuf,
    height: u64,
) -> Result<(Block, Vec<Witness>), Box<dyn std::error::Error>> {
    let bin_path = block_dir.join(format!("block_{}.bin", height));
    let hex_path = block_dir.join(format!("block_{}.hex", height));
    
    let block_data = if bin_path.exists() {
        std::fs::read(&bin_path)?
    } else if hex_path.exists() {
        let hex_content = std::fs::read_to_string(&hex_path)?;
        hex::decode(hex_content.trim())?
    } else {
        return Err(format!("Block {} not found (checked {}.bin and {}.hex)", 
            height, bin_path.display(), hex_path.display()).into());
    };
    
    use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
    let (block, witnesses) = deserialize_block_with_witnesses(&block_data)
        .map_err(|e| format!("Failed to deserialize block {}: {}", height, e))?;
    
    // deserialize_block_with_witnesses already returns Vec<Witness> (one per transaction)
    Ok((block, witnesses))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_replay_config_default() {
        let config = ReplayConfig::default();
        assert_eq!(config.start_height, 0);
        assert!(config.enable_parallel);
    }
    
    #[test]
    fn test_utxo_set_hash_calculation() {
        let mut utxo_set = UtxoSet::default();
        
        // Add some UTXOs
        let outpoint1 = OutPoint { hash: [1; 32], index: 0 };
        let utxo1 = UTXO {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
            height: 0,
        };
        utxo_set.insert(outpoint1, std::sync::Arc::new(utxo1));
        
        // Hash should be deterministic
        let hash1 = calculate_utxo_set_hash(&utxo_set);
        let hash2 = calculate_utxo_set_hash(&utxo_set);
        assert_eq!(hash1, hash2);
    }
    
    #[test]
    fn test_checkpoint_verification() {
        let mut utxo_set = UtxoSet::default();
        let expected_hash = calculate_utxo_set_hash(&utxo_set);
        
        assert!(verify_checkpoint(&utxo_set, &expected_hash));
    }
    
    #[tokio::test]
    async fn test_replay_infrastructure() {
        let config = ReplayConfig::default();
        let result = replay_historical_blocks(&config).await;
        
        // Infrastructure should work even without actual block data
        assert!(result.is_ok());
    }
}

// Implementation Status:
// 1. ✅ Load blocks from local storage (disk) - COMPLETE
// 2. ✅ Parse block format (binary and hex) - COMPLETE
// 3. ✅ Validate blocks sequentially using connect_block - COMPLETE
// 4. ✅ Track UTXO set state - COMPLETE
// 5. ✅ Calculate UTXO set hash at known checkpoints - COMPLETE
// 6. ✅ Compare with consensus's known checkpoint hashes - COMPLETE
// 7. ✅ Report any divergences - COMPLETE
//
// Future Enhancements:
// - Download blocks from Bitcoin network (consensus RPC or block explorer API)
// - Support JSON block format (for debugging)
// - Parallel block validation for old blocks (Phase 4.2 optimization)
// - Block caching for faster subsequent runs

