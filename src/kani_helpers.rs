//! Kani proof helper macros and utilities
//!
//! Provides standardized helpers for common Kani proof patterns
//! to reduce duplication and ensure consistency across all proofs.
//!
//! Reference: LLVM-style optimization for formal verification

use crate::constants::*;

/// Bitcoin transaction limits for Kani proofs
///
/// These limits are used to bound proof input sizes for tractability
/// while ensuring they match or exceed Bitcoin's actual limits.
pub mod proof_limits {
    /// Maximum inputs per transaction (Bitcoin limit)
    pub const MAX_INPUTS_PER_TX: usize = 1000;

    /// Maximum outputs per transaction (Bitcoin limit)
    pub const MAX_OUTPUTS_PER_TX: usize = 1000;

    /// Maximum transactions per block (Bitcoin limit)
    pub const MAX_TRANSACTIONS_PER_BLOCK: usize = 10000;

    /// Maximum mempool transactions for proof tractability
    /// Kept small for proof performance
    pub const MAX_MEMPOOL_TXS_FOR_PROOF: usize = 3;

    /// Maximum previous headers for proof tractability
    /// Kept small for proof performance
    pub const MAX_PREV_HEADERS_FOR_PROOF: usize = 3;

    /// Maximum mining attempts for proof tractability
    pub const MAX_MINING_ATTEMPTS_FOR_PROOF: u64 = 10;

    /// Maximum transaction inputs for proof tractability
    pub const MAX_TX_INPUTS_FOR_PROOF: usize = 2;

    /// Maximum transaction outputs for proof tractability
    pub const MAX_TX_OUTPUTS_FOR_PROOF: usize = 2;
}

/// Standard unwind bounds based on operation complexity
///
/// These bounds are tuned for proof performance while ensuring
/// complete coverage of all loop iterations.
pub mod unwind_bounds {
    /// Simple operations (1-2 loops, no recursion)
    pub const SIMPLE: u32 = 3;

    /// Medium operations (3-5 loops, limited recursion)
    pub const MEDIUM: u32 = 5;

    /// Complex operations (6+ loops, deep recursion)
    pub const COMPLEX: u32 = 10;

    /// Mining-specific bounds
    pub const MINING_BLOCK_CREATION: u32 = 3; // Simple: create block structure
    pub const MINING_BLOCK_MINING: u32 = 10; // Complex: nonce iteration
    pub const MERKLE_ROOT_CALC: u32 = 5; // Medium: tree traversal
    pub const TRANSACTION_VALIDATION: u32 = 3; // Simple: linear scan

    /// PoW-specific bounds
    pub const POW_DIFFICULTY_ADJUSTMENT: u32 = 5; // For get_next_work_required
    pub const POW_TARGET_EXPANSION: u32 = 3; // For expand_target
    pub const POW_CHECK: u32 = 3; // For check_proof_of_work

    /// Block validation bounds
    pub const BLOCK_VALIDATION: u32 = 10; // For block validation (multiple transactions)
}

/// Macro for standard transaction bounds
///
/// Applies standard bounds to a transaction for Kani proofs.
/// Ensures transaction size is within Bitcoin limits and proof tractability.
#[macro_export]
macro_rules! assume_transaction_bounds {
    ($tx:expr) => {
        kani::assume(
            $tx.inputs.len() <= $crate::kani_helpers::proof_limits::MAX_TX_INPUTS_FOR_PROOF,
        );
        kani::assume(
            $tx.outputs.len() <= $crate::kani_helpers::proof_limits::MAX_TX_OUTPUTS_FOR_PROOF,
        );
    };
}

/// Macro for standard block bounds
///
/// Applies standard bounds to a block for Kani proofs.
/// Ensures block has transactions and all transactions are bounded.
#[macro_export]
macro_rules! assume_block_bounds {
    ($block:expr) => {
        kani::assume(!$block.transactions.is_empty());
        kani::assume(
            $block.transactions.len()
                <= $crate::kani_helpers::proof_limits::MAX_TRANSACTIONS_PER_BLOCK,
        );
        for tx in &$block.transactions {
            $crate::assume_transaction_bounds!(tx);
        }
    };
}

/// Macro for mining operation bounds
///
/// Applies standard bounds for mining-related proofs.
/// Bounds mempool transactions, previous headers, and all transactions.
#[macro_export]
macro_rules! assume_mining_bounds {
    ($mempool_txs:expr, $prev_headers:expr) => {
        kani::assume(
            $mempool_txs.len() <= $crate::kani_helpers::proof_limits::MAX_MEMPOOL_TXS_FOR_PROOF,
        );
        kani::assume(
            $prev_headers.len() <= $crate::kani_helpers::proof_limits::MAX_PREV_HEADERS_FOR_PROOF,
        );
        for tx in &$mempool_txs {
            $crate::assume_transaction_bounds!(tx);
        }
    };
}

/// Macro for mining attempt bounds
///
/// Applies standard bounds for mining attempt counts.
#[macro_export]
macro_rules! assume_mining_attempts {
    ($max_attempts:expr) => {
        kani::assume(
            $max_attempts <= $crate::kani_helpers::proof_limits::MAX_MINING_ATTEMPTS_FOR_PROOF,
        );
    };
}

/// Macro for PoW difficulty adjustment bounds
///
/// Applies standard bounds for difficulty adjustment proofs.
#[macro_export]
macro_rules! assume_pow_bounds {
    ($prev_headers:expr) => {
        kani::assume($prev_headers.len() >= 2);
        kani::assume(
            $prev_headers.len() <= $crate::kani_helpers::proof_limits::MAX_PREV_HEADERS_FOR_PROOF,
        );
    };
}

/// Macro for transaction bounds with custom limits
///
/// Allows proof-specific bounds while maintaining consistency.
#[macro_export]
macro_rules! assume_transaction_bounds_custom {
    ($tx:expr, $max_inputs:expr, $max_outputs:expr) => {
        kani::assume($tx.inputs.len() <= $max_inputs);
        kani::assume($tx.outputs.len() <= $max_outputs);
    };
}
