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
//! use blvm_consensus::utxo_commitments::{UtxoCommitmentSet, UtxoCommitment};
//!
//! // Create UTXO commitment set
//! let mut commitment_set = UtxoCommitmentSet::new();
//!
//! // Add UTXO
//! let outpoint = OutPoint { hash: [1; 32], index: 0 };
//! let utxo = UTXO { value: 1000, script_pubkey: vec![], height: 0 };
//! commitment_set.insert(outpoint, utxo)?;
//!
//! // Generate commitment
//! let commitment = commitment_set.generate_commitment(block_hash, height)?;
//! ```

pub mod config;
pub mod data_structures;
pub mod initial_sync;
pub mod merkle_tree;
pub mod network_integration;
pub mod peer_consensus;
pub mod verification;

// Re-export main types
pub use config::*;
pub use data_structures::*;
pub use initial_sync::{update_commitments_after_block, InitialSync};
pub use merkle_tree::UtxoMerkleTree;
pub use network_integration::*;
pub use peer_consensus::*;
pub use verification::*;
