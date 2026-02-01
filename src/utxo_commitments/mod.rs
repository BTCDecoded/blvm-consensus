//! UTXO Commitments Module
//!
//! Implements cryptographic commitments to the UTXO set using Merkle trees.
//! This module enables efficient UTXO set synchronization and verification
//! without requiring full blockchain download.
//!
//! ## Architecture
//!
//! - **Data Structures**: UTXO, UTXO Set, UTXO Commitment
//! - **Merkle Tree**: Sparse Merkle Tree for incremental updates
//! - **Peer Consensus**: N of M peer verification model
//! - **Verification**: PoW-based commitment verification
//!
//! ## Usage
//!
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use blvm_consensus::utxo_commitments::{UtxoMerkleTree, UtxoCommitment};
//! use blvm_consensus::types::{OutPoint, UTXO, Hash};
//!
//! // Create UTXO Merkle tree
//! let mut utxo_tree = UtxoMerkleTree::new()?;
//!
//! // Add UTXO
//! let outpoint = OutPoint { hash: [1; 32].into(), index: 0 };
//! let utxo = UTXO { value: 1000, script_pubkey: vec![].into(), height: 0, is_coinbase: false };
//! utxo_tree.insert(outpoint, utxo)?;
//!
//! // Generate commitment
//! # let block_hash: Hash = [0; 32].into();
//! # let height = 0;
//! let root = utxo_tree.root();
//! let commitment = UtxoCommitment::new(root, 1000, 1, height, block_hash);
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "utxo-commitments")]
pub mod config;
#[cfg(feature = "utxo-commitments")]
pub mod data_structures;
#[cfg(feature = "utxo-commitments")]
pub mod initial_sync;
#[cfg(feature = "utxo-commitments")]
pub mod merkle_tree;
#[cfg(feature = "utxo-commitments")]
pub mod network_integration;
#[cfg(feature = "utxo-commitments")]
pub mod peer_consensus;
#[cfg(feature = "utxo-commitments")]
pub mod verification;

// Re-export main types
#[cfg(feature = "utxo-commitments")]
pub use config::*;
#[cfg(feature = "utxo-commitments")]
pub use data_structures::*;
#[cfg(feature = "utxo-commitments")]
pub use initial_sync::{update_commitments_after_block, InitialSync};
#[cfg(feature = "utxo-commitments")]
pub use merkle_tree::UtxoMerkleTree;
#[cfg(feature = "utxo-commitments")]
pub use network_integration::*;
#[cfg(feature = "utxo-commitments")]
pub use peer_consensus::*;
#[cfg(feature = "utxo-commitments")]
pub use verification::*;
