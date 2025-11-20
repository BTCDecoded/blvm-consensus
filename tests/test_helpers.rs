//! Test helper utilities for CI-aware testing
//!
//! Provides utilities to detect CI environment and adjust test behavior
//! without disabling tests.

/// Check if running in CI environment
///
/// Detects common CI environment variables:
/// - GITHUB_ACTIONS (GitHub Actions)
/// - CI (generic CI environment)
/// - CONTINUOUS_INTEGRATION (generic CI)
pub fn is_ci() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("CONTINUOUS_INTEGRATION").is_ok()
}

/// Get performance threshold multiplier for CI
///
/// Returns a multiplier to adjust timing thresholds in CI environments.
/// CI environments may have slower resources or higher load, so we
/// allow more time for operations while still maintaining test validity.
pub fn performance_threshold_multiplier() -> u64 {
    if is_ci() {
        5 // Allow 5x more time in CI
    } else {
        1 // Normal threshold locally
    }
}

/// Get adjusted timeout for CI environments
///
/// Adjusts timeout values based on environment.
/// CI environments get more lenient timeouts.
pub fn adjusted_timeout(base_timeout_ms: u64) -> u64 {
    base_timeout_ms * performance_threshold_multiplier()
}

// ============================================================================
// Transaction Creation Helpers
// ============================================================================

use bllvm_consensus::{Transaction, TransactionInput, TransactionOutput, OutPoint, UtxoSet, UTXO};

/// Create a test transaction with configurable parameters
///
/// # Arguments
/// * `value` - Output value in satoshis
/// * `sequence` - Sequence number (default: 0xffffffff for non-RBF)
/// * `prevout_hash` - Previous transaction hash (default: [1; 32])
/// * `prevout_index` - Previous transaction output index (default: 0)
pub fn create_test_tx(
    value: i64,
    sequence: Option<u64>,
    prevout_hash: Option<[u8; 32]>,
    prevout_index: Option<u64>,
) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: prevout_hash.unwrap_or([1; 32]),
                index: prevout_index.unwrap_or(0),
            },
            script_sig: vec![0x51], // OP_1
            sequence: sequence.unwrap_or(0xffffffff),
        }],
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![0x51], // OP_1
        }],
        lock_time: 0,
    }
}

/// Create an RBF transaction (sequence < 0xffffffff)
pub fn create_rbf_tx(sequence: u64) -> Transaction {
    create_test_tx(1000, Some(sequence), None, None)
}

/// Create a transaction with a specific output value
pub fn create_tx_with_value(value: i64) -> Transaction {
    create_test_tx(value, None, None, None)
}

/// Create a coinbase transaction
pub fn create_coinbase_tx(value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
    }
}

/// Create a UTXO set with a single UTXO
///
/// Returns (UtxoSet, OutPoint) for the created UTXO
pub fn create_test_utxo(value: i64) -> (UtxoSet, OutPoint) {
    let mut set = UtxoSet::new();
    let txid = [1u8; 32];
    let op = OutPoint { hash: txid, index: 0 };
    set.insert(
        op.clone(),
        UTXO {
            value,
            script_pubkey: vec![0x51],
            height: 1,
        },
    );
    (set, op)
}

/// Create a UTXO set with a single UTXO (alternative name for compatibility)
pub fn create_test_utxo_set() -> UtxoSet {
    let (set, _) = create_test_utxo(10000);
    set
}

/// Create an invalid transaction (empty inputs)
pub fn create_invalid_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![], // Empty inputs - invalid
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
    }
}
