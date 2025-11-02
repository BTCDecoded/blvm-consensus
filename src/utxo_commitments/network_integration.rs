//! Network Integration Helpers for UTXO Commitments
//!
//! Provides helper functions and types for integrating UTXO commitments
//! with the P2P network layer in reference-node.

#[cfg(feature = "utxo-commitments")]
use crate::types::{BlockHeader, Hash, Natural, Transaction};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::data_structures::{UtxoCommitment, UtxoCommitmentError, UtxoCommitmentResult};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::spam_filter::{SpamFilter, SpamSummary};

/// Filtered block structure
#[derive(Debug, Clone)]
pub struct FilteredBlock {
    pub header: BlockHeader,
    pub commitment: UtxoCommitment,
    pub transactions: Vec<Transaction>,
    pub transaction_indices: Vec<u32>,
    pub spam_summary: SpamSummary,
}

/// Network client interface for UTXO commitments
/// 
/// In a full implementation, this would be implemented by the reference-node's
/// network manager to send/receive P2P messages.
/// 
/// Note: This trait is designed for static dispatch. For dynamic dispatch,
/// use the helper functions below or wrap in a type-erased async trait.
pub trait UtxoCommitmentsNetworkClient: Send + Sync {
    /// Request UTXO set from a peer at specific height
    /// 
    /// This is a synchronous interface that returns a Future.
    /// In practice, implementers will use async/await internally.
    fn request_utxo_set(&self, peer_id: &str, height: Natural, block_hash: Hash) 
        -> std::pin::Pin<Box<dyn std::future::Future<Output = UtxoCommitmentResult<UtxoCommitment>> + Send + '_>>;
    
    /// Request filtered block from a peer
    fn request_filtered_block(&self, peer_id: &str, block_hash: Hash)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = UtxoCommitmentResult<FilteredBlock>> + Send + '_>>;
    
    /// Get list of connected peer IDs
    fn get_peer_ids(&self) -> Vec<String>;
}

/// Helper function to request UTXO sets from multiple peers
/// 
/// Takes a function that can request UTXO sets (for flexibility)
pub async fn request_utxo_sets_from_peers_fn<F, Fut>(
    request_fn: F,
    peers: &[String],
    height: Natural,
    block_hash: Hash,
) -> Vec<(String, UtxoCommitmentResult<UtxoCommitment>)>
where
    F: Fn(&str, Natural, Hash) -> Fut,
    Fut: std::future::Future<Output = UtxoCommitmentResult<UtxoCommitment>>,
{
    let mut results = Vec::new();
    
    for peer_id in peers {
        let result = request_fn(peer_id, height, block_hash).await;
        results.push((peer_id.clone(), result));
    }
    
    results
}

/// Helper to process filtered block and verify commitment
pub fn process_and_verify_filtered_block(
    filtered_block: &FilteredBlock,
    expected_height: Natural,
    _spam_filter: &SpamFilter,
) -> UtxoCommitmentResult<bool> {
    // Verify block header height matches expected
    // (In real implementation, would verify full header chain)
    
    // Verify commitment height matches
    if filtered_block.commitment.block_height != expected_height {
        return Err(UtxoCommitmentError::VerificationFailed(format!(
            "Commitment height mismatch: expected {}, got {}",
            expected_height,
            filtered_block.commitment.block_height
        )));
    }
    
    // Verify transactions are properly filtered
    // (In real implementation, would re-apply filter and compare)
    
    // Verify commitment block hash matches header
    let computed_hash = compute_block_hash(&filtered_block.header);
    if filtered_block.commitment.block_hash != computed_hash {
        return Err(UtxoCommitmentError::VerificationFailed(format!(
            "Block hash mismatch: expected {:?}, got {:?}",
            computed_hash,
            filtered_block.commitment.block_hash
        )));
    }
    
    Ok(true)
}

/// Compute block header hash (double SHA256)
fn compute_block_hash(header: &BlockHeader) -> Hash {
    use sha2::{Digest, Sha256};
    
    let mut bytes = Vec::with_capacity(80);
    bytes.extend_from_slice(&header.version.to_le_bytes());
    bytes.extend_from_slice(&header.prev_block_hash);
    bytes.extend_from_slice(&header.merkle_root);
    bytes.extend_from_slice(&header.timestamp.to_le_bytes());
    bytes.extend_from_slice(&header.bits.to_le_bytes());
    bytes.extend_from_slice(&header.nonce.to_le_bytes());
    
    let first_hash = Sha256::digest(&bytes);
    let second_hash = Sha256::digest(&first_hash);
    
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&second_hash);
    hash
}

