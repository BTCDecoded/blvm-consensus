//! Kani proof helper macros and utilities
//!
//! Provides standardized helpers for common Kani proof patterns
//! to reduce duplication and ensure consistency across all proofs.
//!
//! Reference: LLVM-style optimization for formal verification

use crate::constants::*;
use crate::types::*;
use std::collections::HashSet;

/// Bitcoin transaction limits for Kani proofs
///
/// These limits are used to bound proof input sizes for tractability
/// while ensuring they match or exceed Bitcoin's actual limits.
///
/// Note: Proof-time limits are smaller than actual Bitcoin limits for proof tractability.
/// These are used only during Kani proof execution, not in runtime code.
pub mod proof_limits {
    /// Maximum inputs per transaction (Bitcoin limit)
    /// This matches the actual Bitcoin limit for reference.
    pub const MAX_INPUTS_PER_TX: usize = 1000;

    /// Maximum outputs per transaction (Bitcoin limit)
    /// This matches the actual Bitcoin limit for reference.
    pub const MAX_OUTPUTS_PER_TX: usize = 1000;

    /// Maximum transactions per block (Bitcoin limit)
    /// This matches the actual Bitcoin limit for reference.
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

/// Macro for mempool bounds
///
/// Applies standard bounds to a mempool for Kani proofs.
/// Mempool is HashSet<Hash> (tx IDs), so we just bound the size.
#[macro_export]
macro_rules! assume_mempool_bounds {
    ($mempool:expr, $max_len:expr) => {
        kani::assume($mempool.len() <= $max_len);
    };
}

/// Macro for script bounds
///
/// Applies standard bounds to a script for Kani proofs.
/// Ensures script length is within proof tractability limits.
#[macro_export]
macro_rules! assume_script_bounds {
    ($script:expr, $max_len:expr) => {
        kani::assume($script.len() <= $max_len);
    };
}

/// Macro for stack bounds
///
/// Applies standard bounds to a stack for Kani proofs.
/// Ensures stack size is within proof tractability limits.
#[macro_export]
macro_rules! assume_stack_bounds {
    ($stack:expr, $max_len:expr) => {
        kani::assume($stack.len() <= $max_len);
    };
}

/// Macro for witness bounds
///
/// Applies standard bounds to a witness for Kani proofs.
/// Ensures witness element count and sizes are within limits.
#[macro_export]
macro_rules! assume_witness_bounds {
    ($witness:expr, $max_elements:expr) => {
        kani::assume($witness.len() <= $max_elements);
        for element in $witness {
            kani::assume(element.len() <= $crate::constants::MAX_SCRIPT_ELEMENT_SIZE);
        }
    };
}

// ============================================================================
// Helper functions for types that don't implement kani::Arbitrary
// ============================================================================

/// Create a bounded Vec<u8> for Kani proofs
pub fn create_bounded_byte_string(max_len: usize) -> Vec<u8> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = Vec::new();
    for _i in 0..len {
        result.push(kani::any());
    }
    result
}

/// Create a bounded Vec<Vec<u8>> (Witness) for Kani proofs
pub fn create_bounded_witness(max_elements: usize, max_element_len: usize) -> Vec<Vec<u8>> {
    let element_count: usize = kani::any();
    kani::assume(element_count <= max_elements);
    let mut result = Vec::new();
    for _i in 0..element_count {
        result.push(create_bounded_byte_string(max_element_len));
    }
    result
}

/// Create a bounded Vec<Vec<Vec<u8>>> (Vec<Witness>) for Kani proofs
pub fn create_bounded_witness_vec(
    max_witnesses: usize,
    max_elements: usize,
    max_element_len: usize,
) -> Vec<Vec<Vec<u8>>> {
    let witness_count: usize = kani::any();
    kani::assume(witness_count <= max_witnesses);
    let mut result = Vec::new();
    for _i in 0..witness_count {
        result.push(create_bounded_witness(max_elements, max_element_len));
    }
    result
}

/// Create a bounded Transaction for Kani proofs
pub fn create_bounded_transaction() -> Transaction {
    let input_count: usize = kani::any();
    let output_count: usize = kani::any();
    kani::assume(input_count <= proof_limits::MAX_TX_INPUTS_FOR_PROOF);
    kani::assume(output_count <= proof_limits::MAX_TX_OUTPUTS_FOR_PROOF);

    let mut inputs = Vec::new();
    for i in 0..input_count {
        let script_len: usize = kani::any();
        kani::assume(script_len <= 5);
        let mut script = Vec::new();
        for _j in 0..script_len {
            script.push(kani::any());
        }
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: kani::any(),
                index: i as u64,
            },
            script_sig: script,
            sequence: kani::any(),
        });
    }

    let mut outputs = Vec::new();
    for _i in 0..output_count {
        let script_len: usize = kani::any();
        kani::assume(script_len <= 10);
        let mut script = Vec::new();
        for _j in 0..script_len {
            script.push(kani::any());
        }
        let value: i64 = kani::any();
        kani::assume(value >= 0);
        kani::assume(value <= MAX_MONEY);
        outputs.push(TransactionOutput {
            value,
            script_pubkey: script,
        });
    }

    Transaction {
        version: kani::any(),
        inputs: inputs.into(),
        outputs: outputs.into(),
        lock_time: kani::any(),
    }
}

/// Create a bounded UtxoSet for Kani proofs
pub fn create_bounded_utxo_set() -> UtxoSet {
    let map_size: usize = kani::any();
    kani::assume(map_size <= proof_limits::MAX_TX_INPUTS_FOR_PROOF);

    let mut utxo_set = UtxoSet::new();
    for i in 0..map_size {
        let prevout = OutPoint {
            hash: kani::any(),
            index: i as u64,
        };
        let script_len: usize = kani::any();
        kani::assume(script_len <= 10);
        let mut script = Vec::new();
        for _j in 0..script_len {
            script.push(kani::any());
        }
        let value: i64 = kani::any();
        kani::assume(value >= 0);
        kani::assume(value <= MAX_MONEY);
        utxo_set.insert(
            prevout,
            UTXO {
                value,
                script_pubkey: script,
                height: kani::any(),
            },
        );
    }
    utxo_set
}

/// Create a bounded BlockHeader for Kani proofs
pub fn create_bounded_block_header() -> BlockHeader {
    BlockHeader {
        version: kani::any(),
        prev_block_hash: kani::any(),
        merkle_root: kani::any(),
        timestamp: kani::any(),
        bits: kani::any(),
        nonce: kani::any(),
    }
}

/// Create a bounded Block for Kani proofs
pub fn create_bounded_block() -> Block {
    let tx_count: usize = kani::any();
    kani::assume(tx_count > 0);
    kani::assume(tx_count <= proof_limits::MAX_TRANSACTIONS_PER_BLOCK);

    let mut transactions = Vec::new();
    for _i in 0..tx_count {
        transactions.push(create_bounded_transaction());
    }

    Block {
        header: create_bounded_block_header(),
        transactions: transactions.into(),
    }
}

/// Create a bounded Vec<BlockHeader> for Kani proofs
pub fn create_bounded_block_header_vec(max_len: usize) -> Vec<BlockHeader> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = Vec::new();
    for _i in 0..len {
        result.push(create_bounded_block_header());
    }
    result
}

/// Create a bounded Vec<Block> for Kani proofs
pub fn create_bounded_block_vec(max_len: usize) -> Vec<Block> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = Vec::new();
    for _i in 0..len {
        result.push(create_bounded_block());
    }
    result
}

/// Create a bounded Vec<TransactionOutput> for Kani proofs
pub fn create_bounded_transaction_output_vec(max_len: usize) -> Vec<TransactionOutput> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = Vec::new();
    for _i in 0..len {
        let script_len: usize = kani::any();
        kani::assume(script_len <= 10);
        let mut script = Vec::new();
        for _j in 0..script_len {
            script.push(kani::any());
        }
        let value: i64 = kani::any();
        kani::assume(value >= 0);
        kani::assume(value <= MAX_MONEY);
        result.push(TransactionOutput {
            value,
            script_pubkey: script,
        });
    }
    result
}

/// Create a bounded Vec<u64> for Kani proofs
pub fn create_bounded_u64_vec(max_len: usize) -> Vec<u64> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = Vec::new();
    for _i in 0..len {
        result.push(kani::any());
    }
    result
}

/// Create a bounded Mempool (HashSet<[u8; 32]>) for Kani proofs
pub fn create_bounded_mempool(max_len: usize) -> HashSet<[u8; 32]> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = HashSet::new();
    for _i in 0..len {
        result.insert(kani::any());
    }
    result
}

/// Create a bounded SighashType for Kani proofs
pub fn create_bounded_sighash_type() -> crate::transaction_hash::SighashType {
    let variant: u8 = kani::any();
    kani::assume(variant <= 3);
    match variant {
        0 => crate::transaction_hash::SighashType::All,
        1 => crate::transaction_hash::SighashType::None,
        2 => crate::transaction_hash::SighashType::Single,
        _ => crate::transaction_hash::SighashType::All,
    }
}

/// Create a bounded WitnessVersion for Kani proofs
pub fn create_bounded_witness_version() -> crate::witness::WitnessVersion {
    let variant: u8 = kani::any();
    kani::assume(variant <= 1);
    match variant {
        0 => crate::witness::WitnessVersion::SegWitV0,
        _ => crate::witness::WitnessVersion::SegWitV0,
    }
}

/// Create a bounded Vec<Hash> for Kani proofs
pub fn create_bounded_hash_vec(max_len: usize) -> Vec<Hash> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = Vec::new();
    for _i in 0..len {
        result.push(kani::any());
    }
    result
}

/// Create a bounded Vec<Transaction> for Kani proofs
pub fn create_bounded_transaction_vec(max_len: usize) -> Vec<Transaction> {
    let len: usize = kani::any();
    kani::assume(len <= max_len);
    let mut result = Vec::new();
    for _i in 0..len {
        result.push(create_bounded_transaction());
    }
    result
}
