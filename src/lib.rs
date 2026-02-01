//! # Consensus-Proof
//!
//! Direct mathematical implementation of Bitcoin consensus rules from the Orange Paper.
//!
//! This crate provides pure, side-effect-free functions that implement the mathematical
//! specifications defined in the Orange Paper. It serves as the mathematical foundation
//! for Bitcoin consensus validation.
//!
//! ## Architecture
//!
//! The system follows a layered architecture:
//! - Orange Paper (mathematical specifications)
//! - Consensus Proof (this crate - direct implementation)
//! - Reference Node (minimal Bitcoin implementation)
//! - Developer SDK (developer-friendly interface)
//!
//! ## Design Principles
//!
//! 1. **Pure Functions**: All functions are deterministic and side-effect-free
//! 2. **Mathematical Accuracy**: Direct implementation of Orange Paper specifications
//! 3. **Exact Version Pinning**: All consensus-critical dependencies pinned to exact versions
//! 4. **No Consensus Rule Interpretation**: Only mathematical implementation
//!
//! ## Usage
//!
//! ```rust
//! use blvm_consensus::transaction::check_transaction;
//! use blvm_consensus::types::*;
//!
//! let transaction = Transaction {
//!     version: 1,
//!     inputs: vec![].into(),
//!     outputs: vec![TransactionOutput {
//!         value: 1000,
//!         script_pubkey: vec![0x51].into(),
//!     }].into(),
//!     lock_time: 0,
//! };
//! let result = check_transaction(&transaction).unwrap();
//! ```

#![allow(unused_doc_comments)] // Allow doc comments before macros (proptest, etc.)

pub mod config;
pub mod constants;
pub mod script;
pub mod transaction;
pub mod transaction_hash;
pub mod types;

use blvm_spec_lock::spec_locked;
#[cfg(all(feature = "production", feature = "benchmarking"))]
pub use block::{reset_assume_valid_height, set_assume_valid_height};
#[cfg(feature = "production")]
pub use script::batch_verify_signatures;
#[cfg(all(feature = "production", feature = "benchmarking"))]
pub use script::{
    clear_all_caches, clear_hash_cache, clear_script_cache, clear_stack_pool, disable_caching,
    reset_benchmarking_state,
};
#[cfg(all(feature = "production", feature = "benchmarking"))]
pub use transaction_hash::clear_sighash_templates;
pub mod bip113;
pub mod bip_validation;
pub mod block;
pub mod crypto;
pub mod economic;
pub mod locktime;
pub mod mempool;
pub mod mining;
pub mod network;
pub mod optimizations;
pub mod pow;
pub mod reorganization;
#[cfg(feature = "k256")]
pub mod script_k256;
pub mod segwit;
pub mod sequence_locks;
pub mod serialization;
pub mod sigop;
pub mod spam_filter;
pub mod taproot;
#[cfg(feature = "utxo-commitments")]
pub mod utxo_commitments;
pub mod witness;

pub mod error;

/// Consensus Proof - wrapper struct for consensus validation functions
///
/// This struct provides a convenient API for accessing all consensus validation
/// functions. All methods delegate to the corresponding module functions.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConsensusProof;

impl ConsensusProof {
    /// Create a new ConsensusProof instance
    pub fn new() -> Self {
        Self
    }

    /// Validate a transaction according to consensus rules
    #[spec_locked("5.1")]
    pub fn validate_transaction(&self, tx: &types::Transaction) -> error::Result<types::ValidationResult> {
        transaction::check_transaction(tx)
    }

    /// Validate transaction inputs against UTXO set
    #[spec_locked("5.1")]
    pub fn validate_tx_inputs(
        &self,
        tx: &types::Transaction,
        utxo_set: &types::UtxoSet,
        height: types::Natural,
    ) -> error::Result<(types::ValidationResult, types::Integer)> {
        transaction::check_tx_inputs(tx, utxo_set, height)
    }

    /// Validate a complete block
    #[spec_locked("5.3")]
    pub fn validate_block(
        &self,
        block: &types::Block,
        utxo_set: types::UtxoSet,
        height: types::Natural,
    ) -> error::Result<(types::ValidationResult, types::UtxoSet)> {
        // Create empty witnesses for backward compatibility
        let witnesses: Vec<Vec<segwit::Witness>> = block.transactions.iter().map(|_| Vec::new()).collect();
        let network_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let (result, new_utxo_set, _undo_log) = block::connect_block(
            block,
            &witnesses,
            utxo_set,
            height,
            None,
            network_time,
            types::Network::Mainnet,
        )?;
        Ok((result, new_utxo_set))
    }

    /// Validate a complete block with witness data and time context
    #[spec_locked("5.3")]
    pub fn validate_block_with_time_context(
        &self,
        block: &types::Block,
        witnesses: &[Vec<segwit::Witness>],
        utxo_set: types::UtxoSet,
        height: types::Natural,
        time_context: Option<types::TimeContext>,
        network: types::Network,
    ) -> error::Result<(types::ValidationResult, types::UtxoSet)> {
        let (result, new_utxo_set, _undo_log) = block::connect_block_with_context(
            block,
            witnesses,
            utxo_set,
            height,
            time_context,
            network,
        )?;
        Ok((result, new_utxo_set))
    }

    /// Verify script execution
    #[spec_locked("5.2")]
    pub fn verify_script(
        &self,
        script_sig: &types::ByteString,
        script_pubkey: &types::ByteString,
        witness: Option<&types::ByteString>,
        flags: u32,
    ) -> error::Result<bool> {
        script::verify_script(script_sig, script_pubkey, witness, flags)
    }

    /// Check proof of work
    #[spec_locked("7.2")]
    pub fn check_proof_of_work(&self, header: &types::BlockHeader) -> error::Result<bool> {
        pow::check_proof_of_work(header)
    }

    /// Get block subsidy for height
    #[spec_locked("6.1")]
    pub fn get_block_subsidy(&self, height: types::Natural) -> types::Integer {
        economic::get_block_subsidy(height)
    }

    /// Calculate total supply at height
    #[spec_locked("6.2")]
    pub fn total_supply(&self, height: types::Natural) -> types::Integer {
        economic::total_supply(height)
    }

    /// Get next work required for difficulty adjustment
    #[spec_locked("7.1")]
    pub fn get_next_work_required(
        &self,
        current_header: &types::BlockHeader,
        prev_headers: &[types::BlockHeader],
    ) -> error::Result<types::Natural> {
        pow::get_next_work_required(current_header, prev_headers)
    }

    /// Accept transaction to memory pool
    #[spec_locked("9.1")]
    pub fn accept_to_memory_pool(
        &self,
        tx: &types::Transaction,
        utxo_set: &types::UtxoSet,
        mempool: &mempool::Mempool,
        height: types::Natural,
        time_context: Option<types::TimeContext>,
    ) -> error::Result<mempool::MempoolResult> {
        mempool::accept_to_memory_pool(tx, None, utxo_set, mempool, height, time_context)
    }

    /// Check if transaction is standard
    #[spec_locked("9.2")]
    pub fn is_standard_tx(&self, tx: &types::Transaction) -> error::Result<bool> {
        mempool::is_standard_tx(tx)
    }

    /// Check if transaction can replace existing one (RBF)
    #[spec_locked("9.3")]
    pub fn replacement_checks(
        &self,
        new_tx: &types::Transaction,
        existing_tx: &types::Transaction,
        utxo_set: &types::UtxoSet,
        mempool: &mempool::Mempool,
    ) -> error::Result<bool> {
        mempool::replacement_checks(new_tx, existing_tx, utxo_set, mempool)
    }

    /// Create new block from mempool transactions
    #[allow(clippy::too_many_arguments)]
    #[spec_locked("12.1")]
    pub fn create_new_block(
        &self,
        utxo_set: &types::UtxoSet,
        mempool_txs: &[types::Transaction],
        height: types::Natural,
        prev_header: &types::BlockHeader,
        prev_headers: &[types::BlockHeader],
        coinbase_script: &types::ByteString,
        coinbase_address: &types::ByteString,
    ) -> error::Result<types::Block> {
        mining::create_new_block(
            utxo_set,
            mempool_txs,
            height,
            prev_header,
            prev_headers,
            coinbase_script,
            coinbase_address,
        )
    }

    /// Mine a block by finding valid nonce
    #[spec_locked("12.3")]
    pub fn mine_block(
        &self,
        block: types::Block,
        max_attempts: types::Natural,
    ) -> error::Result<(types::Block, mining::MiningResult)> {
        mining::mine_block(block, max_attempts)
    }

    /// Create block template for mining
    #[allow(clippy::too_many_arguments)]
    #[spec_locked("12.1")]
    pub fn create_block_template(
        &self,
        utxo_set: &types::UtxoSet,
        mempool_txs: &[types::Transaction],
        height: types::Natural,
        prev_header: &types::BlockHeader,
        prev_headers: &[types::BlockHeader],
        coinbase_script: &types::ByteString,
        coinbase_address: &types::ByteString,
    ) -> error::Result<mining::BlockTemplate> {
        mining::create_block_template(
            utxo_set,
            mempool_txs,
            height,
            prev_header,
            prev_headers,
            coinbase_script,
            coinbase_address,
        )
    }

    /// Reorganize chain when longer chain is found
    #[spec_locked("11.3")]
    pub fn reorganize_chain(
        &self,
        new_chain: &[types::Block],
        current_chain: &[types::Block],
        current_utxo_set: types::UtxoSet,
        current_height: types::Natural,
    ) -> error::Result<reorganization::ReorganizationResult> {
        reorganization::reorganize_chain(new_chain, current_chain, current_utxo_set, current_height)
    }

    /// Check if reorganization is beneficial
    #[spec_locked("11.3")]
    pub fn should_reorganize(
        &self,
        new_chain: &[types::Block],
        current_chain: &[types::Block],
    ) -> error::Result<bool> {
        reorganization::should_reorganize(new_chain, current_chain)
    }

    /// Process incoming network message
    #[spec_locked("9.2")]
    pub fn process_network_message(
        &self,
        message: &network::NetworkMessage,
        peer_state: &mut network::PeerState,
        chain_state: &network::ChainState,
    ) -> error::Result<network::NetworkResponse> {
        network::process_network_message(message, peer_state, chain_state)
    }

    /// Calculate transaction weight for SegWit
    #[spec_locked("11.1")]
    pub fn calculate_transaction_weight(
        &self,
        tx: &types::Transaction,
        witness: Option<&segwit::Witness>,
    ) -> error::Result<types::Natural> {
        segwit::calculate_transaction_weight(tx, witness)
    }

    /// Validate SegWit block
    #[spec_locked("11.1")]
    pub fn validate_segwit_block(
        &self,
        block: &types::Block,
        witnesses: &[segwit::Witness],
        max_block_weight: types::Natural,
    ) -> error::Result<bool> {
        segwit::validate_segwit_block(block, witnesses, max_block_weight)
    }

    /// Validate Taproot transaction
    #[spec_locked("11.2")]
    pub fn validate_taproot_transaction(
        &self,
        tx: &types::Transaction,
        witness: Option<&segwit::Witness>,
    ) -> error::Result<bool> {
        taproot::validate_taproot_transaction(tx, witness)
    }

    /// Check if transaction output is Taproot
    #[spec_locked("11.2")]
    pub fn is_taproot_output(&self, output: &types::TransactionOutput) -> bool {
        taproot::is_taproot_output(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::check_transaction;
    use crate::network::{ChainState, NetworkAddress, NetworkMessage, PeerState, VersionMessage};

    #[test]
    fn test_validate_transaction() {
        let tx = Transaction {
            version: 1,
            inputs: vec![].into(),
            outputs: vec![].into(),
            lock_time: 0,
        };
        let result = check_transaction(&tx);
        assert!(result.is_ok());
    }
}
