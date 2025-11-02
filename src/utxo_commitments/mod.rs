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
//! use consensus_proof::utxo_commitments::{UtxoCommitmentSet, UtxoCommitment};
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

pub mod data_structures;
pub mod merkle_tree;
pub mod verification;
pub mod peer_consensus;
pub mod initial_sync;
pub mod spam_filter;
pub mod config;
pub mod network_integration;

// Re-export main types
pub use data_structures::*;
pub use merkle_tree::UtxoMerkleTree;
pub use verification::*;
pub use peer_consensus::*;
pub use initial_sync::InitialSync;
pub use spam_filter::*;
pub use config::*;
pub use network_integration::*;

