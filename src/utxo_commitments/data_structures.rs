//! UTXO Commitment Data Structures
//!
//! Defines the core data structures for UTXO commitments:
//! - UTXO: Unspent Transaction Output with metadata
//! - UTXOCommitment: Cryptographic commitment to UTXO set state
//! - UtxoCommitmentSet: UTXO set with Merkle tree commitment support

use crate::types::{Hash, Natural};
use serde::{Deserialize, Serialize};
use blvm_spec_lock::spec_locked;

/// UTXO Commitment
///
/// Cryptographic commitment to the UTXO set state at a specific block height.
/// Contains the Merkle root hash of the UTXO set and metadata for verification.
///
/// Size: 84 bytes (constant)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UtxoCommitment {
    /// Merkle root of the UTXO set
    pub merkle_root: Hash,

    /// Total Bitcoin supply in the UTXO set (satoshis)
    pub total_supply: u64,

    /// Number of UTXOs in the set
    pub utxo_count: u64,

    /// Block height at which this commitment was generated
    pub block_height: Natural,

    /// Block hash at which this commitment was generated
    pub block_hash: Hash,
}

impl UtxoCommitment {
    /// Create a new UTXO commitment
    pub fn new(
        merkle_root: Hash,
        total_supply: u64,
        utxo_count: u64,
        block_height: Natural,
        block_hash: Hash,
    ) -> Self {
        Self {
            merkle_root,
            total_supply,
            utxo_count,
            block_height,
            block_hash,
        }
    }

    /// Serialize commitment to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(84);
        bytes.extend_from_slice(&self.merkle_root);
        bytes.extend_from_slice(&self.total_supply.to_be_bytes());
        bytes.extend_from_slice(&self.utxo_count.to_be_bytes());
        bytes.extend_from_slice(&self.block_height.to_be_bytes());
        bytes.extend_from_slice(&self.block_hash);
        bytes
    }

    /// Deserialize commitment from bytes
    #[spec_locked("13.1")]
    pub fn from_bytes(data: &[u8]) -> Result<Self, UtxoCommitmentError> {
        if data.len() != 84 {
            return Err(UtxoCommitmentError::InvalidSize(data.len()));
        }

        let mut offset = 0;
        let merkle_root: Hash = {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&data[offset..offset + 32]);
            offset += 32;
            hash
        };

        let total_supply = u64::from_be_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| UtxoCommitmentError::InvalidSize(data.len()))?,
        );
        offset += 8;

        let utxo_count = u64::from_be_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| UtxoCommitmentError::InvalidSize(data.len()))?,
        );
        offset += 8;

        let block_height = u64::from_be_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| UtxoCommitmentError::InvalidSize(data.len()))?,
        );
        offset += 8;

        let block_hash: Hash = {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&data[offset..offset + 32]);
            hash
        };

        Ok(Self {
            merkle_root,
            total_supply,
            utxo_count,
            block_height,
            block_hash,
        })
    }

    /// Verify commitment matches UTXO set parameters
    #[spec_locked("13.2")]
    pub fn verify_supply(&self, expected_supply: u64) -> bool {
        self.total_supply == expected_supply
    }

    /// Verify commitment matches expected UTXO count
    #[spec_locked("13.2")]
    pub fn verify_count(&self, expected_count: u64) -> bool {
        self.utxo_count == expected_count
    }
}

/// Error type for UTXO commitment operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UtxoCommitmentError {
    /// Invalid commitment size
    InvalidSize(usize),
    /// Merkle tree operation failed
    MerkleTreeError(String),
    /// Invalid UTXO data
    InvalidUtxo(String),
    /// Commitment verification failed
    VerificationFailed(String),
    /// Serialization/deserialization error
    SerializationError(String),
    /// Transaction application error (during UTXO set updates)
    TransactionApplication(String),
}

impl std::fmt::Display for UtxoCommitmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UtxoCommitmentError::InvalidSize(size) => {
                write!(f, "Invalid commitment size: {} (expected 84)", size)
            }
            UtxoCommitmentError::MerkleTreeError(msg) => {
                write!(f, "Merkle tree error: {}", msg)
            }
            UtxoCommitmentError::InvalidUtxo(msg) => {
                write!(f, "Invalid UTXO: {}", msg)
            }
            UtxoCommitmentError::VerificationFailed(msg) => {
                write!(f, "Verification failed: {}", msg)
            }
            UtxoCommitmentError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            UtxoCommitmentError::TransactionApplication(msg) => {
                write!(f, "Transaction application error: {}", msg)
            }
        }
    }
}

impl std::error::Error for UtxoCommitmentError {}

/// Result type for UTXO commitment operations
pub type UtxoCommitmentResult<T> = Result<T, UtxoCommitmentError>;

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for UTXO Commitment:
/// ∀ commitment ∈ UtxoCommitment:
/// - to_bytes() → Vec<u8] where |bytes| = 84
/// - from_bytes(to_bytes(commitment)) = commitment (round-trip)
/// - verify_supply(expected) = true ⟺ commitment.total_supply = expected
///
// Invariants:
// - Serialization is deterministic and reversible

// End of module
