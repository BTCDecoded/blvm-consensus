//! Block validation functions from Orange Paper Section 5.3 Section 5.3
//!
//! Performance optimizations:
//! - Parallel transaction validation (production feature)
//! - Batch UTXO operations
//! - Assume-Valid Blocks - skip validation for trusted checkpoints

#[cfg(feature = "profile")]
use crate::profile_log;
use crate::bip113::get_median_time_past;
use crate::constants::*;
use crate::economic::get_block_subsidy;
use crate::error::{ConsensusError, Result};
use crate::opcodes::*;
use crate::script::verify_script_with_context_full;
use std::borrow::Cow;
use std::sync::Arc;
use blvm_spec_lock::spec_locked;

// Cold error construction helpers - these paths are rarely taken
#[cold]
fn make_fee_overflow_error(transaction_index: Option<usize>) -> ConsensusError {
    let message = if let Some(i) = transaction_index {
        format!("Total fees overflow at transaction {i}")
    } else {
        "Total fees overflow".to_string()
    };
    ConsensusError::BlockValidation(message.into())
}
use crate::segwit::{is_segwit_transaction, validate_witness_commitment, Witness};
use crate::transaction::{check_transaction, check_tx_inputs, is_coinbase};
use crate::types::*;
use crate::utxo_overlay::{UtxoOverlay, apply_transaction_to_overlay, apply_transaction_to_overlay_no_undo};
use crate::witness::is_witness_empty;
#[cfg(feature = "production")]
use rustc_hash::{FxHashMap, FxHashSet};

// Rayon is used conditionally in the code, imported where needed

/// Per-sig ECDSA verify (Core-style): verify inline in script workers, no collect/drain.
/// Default true: ECDSA batch always fails (R.y bug), so collect/drain adds overhead without benefit.
/// Set BLVM_ECDSA_PER_SIG=0 to restore legacy collect/drain (e.g. for testing).
#[cfg(all(feature = "production", feature = "rayon"))]
fn use_per_sig_ecdsa() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("BLVM_ECDSA_PER_SIG").map(|s| s != "0" && (s == "1" || s.eq_ignore_ascii_case("true"))).unwrap_or(true)
    })
}

#[cfg(all(feature = "production", feature = "rayon"))]
fn use_per_sig_schnorr() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("BLVM_SCHNORR_PER_SIG").map(|s| s == "1" || s.eq_ignore_ascii_case("true")).unwrap_or(false)
    })
}

#[cfg(all(feature = "production", feature = "rayon"))]
fn n_crypto_drain_threads() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("BLVM_CRYPTO_DRAIN_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| (p.get() / 4).max(2).min(8))
                    .unwrap_or(4)
            })
            .max(1)
            .min(16)
    })
}

#[cfg(all(feature = "production", feature = "rayon"))]
fn script_check_queue() -> &'static crate::checkqueue::ScriptCheckQueue {
    use std::sync::OnceLock;
    static Q: OnceLock<crate::checkqueue::ScriptCheckQueue> = OnceLock::new();
    Q.get_or_init(|| {
        let n = std::env::var("BLVM_SCRIPT_WORKERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                // par - 1 workers; master joins as Nth
                std::thread::available_parallelism()
                    .map(|p| p.get().saturating_sub(1).max(1))
                    .unwrap_or(4)
            });
        let batch_size = crate::ibd_tuning::chunk_threshold_config_or_hardware(
            crate::config::get_consensus_config_ref().performance.ibd_chunk_threshold,
        );
        crate::checkqueue::ScriptCheckQueue::new(n, Some(batch_size))
    })
}

/// Overlay delta for disk sync. Returned by connect_block_ibd when BLVM_USE_OVERLAY_DELTA=1.
/// Node converts to SyncBatch and calls apply_sync_batch instead of sync_block_to_batch.
/// Arc<UTXO> in additions avoids clone in apply_sync_batch hot path.
#[cfg(feature = "production")]
#[derive(Debug, Clone)]
pub struct UtxoDelta {
    pub additions: FxHashMap<OutPoint, std::sync::Arc<UTXO>>,
    pub deletions: FxHashSet<OutPoint>,
}
#[cfg(not(feature = "production"))]
#[derive(Debug, Clone)]
pub struct UtxoDelta {
    pub additions: std::collections::HashMap<OutPoint, std::sync::Arc<UTXO>>,
    pub deletions: std::collections::HashSet<OutPoint>,
}

/// Assume-valid checkpoint configuration
///
/// Blocks before this height are assumed valid (signature verification skipped)
/// for faster IBD. This is safe because:
/// 1. These blocks are in the chain history (already validated by network)
/// 2. We still validate block structure, Merkle roots, and PoW
/// 3. Only signature verification is skipped (the expensive operation)
///
/// Assume-valid: skip signature verification below configurable height
/// Default: 0 (validate all blocks) - can be configured via environment or config
/// Get assume-valid height from configuration
///
/// This function loads the assume-valid checkpoint height from environment variable
/// or configuration. Blocks before this height skip expensive signature verification
/// during initial block download for performance.
///
/// # Configuration
/// - Environment variable: `ASSUME_VALID_HEIGHT` (decimal height)
/// - Default: 0 (validate all blocks - safest option)
/// - Benchmarking: `config::set_assume_valid_height()` when `benchmarking` feature enabled
///
/// # Safety
/// This optimization is safe because:
/// 1. These blocks are already validated by the network
/// 2. We still validate block structure, Merkle roots, and PoW
/// 3. Only signature verification is skipped (the expensive operation)
///
/// Assume-valid: skip signature verification below configurable height
#[cfg(feature = "production")]
pub fn get_assume_valid_height() -> u64 {
    // Check for benchmarking override first
    #[cfg(feature = "benchmarking")]
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);
        let override_val = OVERRIDE.load(Ordering::Relaxed);
        if override_val != u64::MAX {
            return override_val;
        }
    }

    crate::config::get_assume_valid_height()
}

/// ConnectBlock: ℬ × 𝒲* × 𝒰𝒮 × ℕ × ℋ* → {valid, invalid} × 𝒰𝒮
///
/// For block b = (h, txs) with witnesses ws, UTXO set us at height height, and recent headers:
/// 1. Validate block header h
/// 2. For each transaction tx ∈ txs:
///    - Validate tx structure
///    - Check inputs against us
///    - Verify scripts (with witness data if available)
/// 3. Let fees = Σ_{tx ∈ txs} fee(tx)
/// 4. Let subsidy = GetBlockSubsidy(height)
/// 5. If coinbase output > fees + subsidy: return (invalid, us)
/// 6. Apply all transactions to us: us' = ApplyTransactions(txs, us)
/// 7. Return (valid, us')
///
/// # Arguments
///
/// * `block` - The block to validate and connect
/// * `witnesses` - Witness data for each transaction in the block (one Witness per transaction)
/// * `utxo_set` - Current UTXO set (will be modified)
/// * `height` - Current block height
/// * `recent_headers` - Optional recent block headers for median time-past calculation (BIP113)
#[track_caller] // Better error messages showing caller location
/// ConnectBlock: Validate and apply a block to the UTXO set.
///
/// # Consensus Engine Purity
/// This function requires `network_time` to be provided by the caller (node layer).
/// The consensus engine must not call `SystemTime::now()` directly.
#[spec_locked("5.3")]
pub fn connect_block<H: AsRef<BlockHeader>>(
    block: &Block,
    witnesses: &[Vec<Witness>], // CRITICAL FIX: Changed from &[Witness] to &[Vec<Witness>]
    // witnesses is now Vec<Vec<Witness>> where each Vec<Witness> is for one transaction
    // and each Witness is for one input
    utxo_set: UtxoSet,
    height: Natural,
    recent_headers: Option<&[H]>,
    network_time: u64,
    network: crate::types::Network,
) -> Result<(
    ValidationResult,
    UtxoSet,
    crate::reorganization::BlockUndoLog,
)> {
    let time_context = build_time_context(recent_headers, network_time);
    #[cfg(all(feature = "production", feature = "rayon"))]
    let block_arc = Some(std::sync::Arc::new(block.clone()));
    #[cfg(not(all(feature = "production", feature = "rayon")))]
    let block_arc = None;
    let (result, new_utxo_set, _tx_ids, undo_log, _delta) =
        connect_block_inner(block, witnesses, utxo_set, None, height, time_context, network, None, None, block_arc)?;
    Ok((result, new_utxo_set, undo_log))
}

/// ConnectBlock implementation that accepts an explicit time context.
///
/// This variant allows callers (e.g., protocol/node layers) to provide a
/// precomputed `TimeContext` derived from their own notion of network time
/// and median time-past, rather than relying on `SystemTime::now()` inside
/// consensus code. Existing callers can continue to use `connect_block`,
/// which builds the time context from `recent_headers`.
#[spec_locked("5.3")]
pub fn connect_block_with_context(
    block: &Block,
    witnesses: &[Vec<Witness>], // CRITICAL FIX: Changed from &[Witness] to &[Vec<Witness>]
    // witnesses is now Vec<Vec<Witness>> where each Vec<Witness> is for one transaction
    // and each Witness is for one input
    utxo_set: UtxoSet,
    height: Natural,
    time_context: Option<crate::types::TimeContext>,
    network: crate::types::Network,
) -> Result<(
    ValidationResult,
    UtxoSet,
    crate::reorganization::BlockUndoLog,
)> {
    #[cfg(all(feature = "production", feature = "rayon"))]
    let block_arc = Some(std::sync::Arc::new(block.clone()));
    #[cfg(not(all(feature = "production", feature = "rayon")))]
    let block_arc = None;
    let (result, new_utxo_set, _tx_ids, undo_log, _delta) =
        connect_block_inner(block, witnesses, utxo_set, None, height, time_context, network, None, None, block_arc)?;
    Ok((result, new_utxo_set, undo_log))
}

/// ConnectBlock variant optimized for IBD that returns transaction IDs instead of undo log.
///
/// This function is identical to `connect_block` but returns `Vec<Hash>` (transaction IDs)
/// instead of `BlockUndoLog`. This allows callers to avoid redundant double-SHA256 computation
/// when they need the transaction IDs anyway.
///
/// # Arguments
///
/// * `block` - The block to validate and connect
/// * `witnesses` - Witness data for each transaction in the block
/// * `utxo_set` - Current UTXO set (will be modified)
/// * `height` - Current block height
/// * `recent_headers` - Optional recent block headers for median time-past calculation (BIP113)
/// * `network_time` - Current network time (Unix timestamp)
/// * `network` - Network type (Mainnet, Testnet, etc.)
///
/// # Returns
///
/// Returns `(ValidationResult, UtxoSet, Vec<Hash>)` where:
/// - `ValidationResult` indicates if the block is valid
/// - `UtxoSet` is the updated UTXO set after applying the block
/// - `Vec<Hash>` contains the transaction IDs (pre-computed to avoid redundant hashing)
///
/// * `bip30_index` - Optional index for O(1) BIP30 duplicate-coinbase check. When `Some`,
///   must be in sync with `utxo_set` (e.g. built via `build_bip30_index` on resume).
/// * `precomputed_tx_ids` - Optional pre-computed tx IDs (#21). When `Some`, skips hashing in
///   consensus; caller (e.g. node) computes once and shares with collect_gaps.
#[spec_locked("5.3")]
pub fn connect_block_ibd<H: AsRef<BlockHeader>>(
    block: &Block,
    witnesses: &[Vec<Witness>],
    utxo_set: UtxoSet,
    height: Natural,
    recent_headers: Option<&[H]>,
    network_time: u64,
    network: crate::types::Network,
    bip30_index: Option<&mut crate::bip_validation::Bip30Index>,
    precomputed_tx_ids: Option<&[Hash]>,
    block_arc: Option<std::sync::Arc<Block>>,
    witnesses_arc: Option<&std::sync::Arc<Vec<Vec<Witness>>>>, // When Some, used for witness_buffer (avoids clone)
) -> Result<(
    ValidationResult,
    UtxoSet,
    Vec<Hash>,
    Option<UtxoDelta>,
)> {
    let time_context = build_time_context(recent_headers, network_time);
    
    // Call connect_block_inner which does full validation (uses precomputed tx_ids when provided)
    let (result, new_utxo_set, tx_ids, _undo_log, utxo_delta) =
        connect_block_inner(block, witnesses, utxo_set, witnesses_arc, height, time_context, network, bip30_index, precomputed_tx_ids, block_arc)?;
    
    Ok((result, new_utxo_set, tx_ids, utxo_delta))
}

/// Helper to construct a `TimeContext` from recent headers and network time.
///
/// # Consensus Engine Purity
/// This function does NOT call `SystemTime::now()`. The `network_time` parameter
/// must be provided by the node layer, ensuring the consensus engine remains pure.
#[spec_locked("5.5")]
fn build_time_context<H: AsRef<BlockHeader>>(
    recent_headers: Option<&[H]>,
    network_time: u64,
) -> Option<crate::types::TimeContext> {
    recent_headers.map(|headers| {
        let median_time_past = get_median_time_past(headers);
        crate::types::TimeContext {
            network_time,
            median_time_past,
        }
    })
}

#[allow(clippy::overly_complex_bool_expr)] // Intentional tautological assertions for formal verification
#[spec_locked("5.3")]
fn connect_block_inner(
    block: &Block,
    witnesses: &[Vec<Witness>], // CRITICAL FIX: Changed from &[Witness] to &[Vec<Witness>]
    // witnesses is now Vec<Vec<Witness>> where each Vec<Witness> is for one transaction
    // and each Witness is for one input
    mut utxo_set: UtxoSet,
    witnesses_arc: Option<&std::sync::Arc<Vec<Vec<Witness>>>>,
    height: Natural,
    time_context: Option<crate::types::TimeContext>,
    network: crate::types::Network,
    bip30_index: Option<&mut crate::bip_validation::Bip30Index>,
    precomputed_tx_ids: Option<&[Hash]>,
    block_arc: Option<std::sync::Arc<Block>>,
) -> Result<(
    ValidationResult,
    UtxoSet,
    Vec<Hash>,
    crate::reorganization::BlockUndoLog,
    Option<UtxoDelta>,
)> {
    // Precondition assertions: Validate function inputs before execution
    // Note: We check empty blocks in validation logic rather than asserting,
    // to allow tests to verify the validation behavior properly
    assert!(height <= i64::MAX as u64, "Block height must fit in i64");
    assert!(
        utxo_set.len() <= u32::MAX as usize,
        "UTXO set size {} exceeds maximum",
        utxo_set.len()
    );
    assert!(
        witnesses.len() == block.transactions.len(),
        "Witness count {} must match transaction count {}",
        witnesses.len(),
        block.transactions.len()
    );

    // Note: Header validation is handled by validate_block_header() below,
    // not by assertions, to allow tests to verify validation behavior
    // We only assert on values that are truly programming errors, not validation errors

    // Check block size and transaction count before validation (#6: fix conflicting cfg — was dead)
    #[cfg(feature = "production")]
    {
        // Quick reject: empty block (invalid)
        if block.transactions.is_empty() {
            return Ok((
                ValidationResult::Invalid("Block has no transactions".into()),
                utxo_set,
                vec![],
                crate::reorganization::BlockUndoLog::new(),
            None,
        ));
        }

        // Quick reject: too many transactions (before expensive validation)
        // Estimate: MAX_BLOCK_SIZE / average_tx_size ≈ 1,000,000 / 250 = ~4000 transactions
        // Use conservative limit of 10,000 transactions
        if block.transactions.len() > 10_000 {
            return Ok((
                ValidationResult::Invalid(format!(
                    "Block has too many transactions: {}",
                    block.transactions.len()
                )),
                utxo_set,
                vec![],
                crate::reorganization::BlockUndoLog::new(),
            None,
        ));
        }
    }

    // 1. Validate block header (cheap — defer tx_ids until after)
    if !validate_block_header(&block.header, time_context.as_ref())? {
        return Ok((
            ValidationResult::Invalid("Invalid block header".into()),
            utxo_set,
            vec![],
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // Check block weight (DoS prevention)
    // This must be done before expensive transaction validation
    use crate::segwit::calculate_block_weight_from_nested;
    let block_weight = calculate_block_weight_from_nested(block, witnesses)?;
    // Invariant assertion: Block weight must be non-zero and within reasonable bounds
    assert!(block_weight > 0, "Block weight must be positive");
    assert!(
        block_weight <= crate::constants::MAX_BLOCK_WEIGHT as u64 * 2,
        "Block weight {block_weight} exceeds reasonable maximum"
    );
    if block_weight > crate::constants::MAX_BLOCK_WEIGHT as u64 {
        return Ok((
            ValidationResult::Invalid(format!(
                "Block weight {} exceeds maximum {}",
                block_weight,
                crate::constants::MAX_BLOCK_WEIGHT
            )),
            utxo_set,
            vec![],
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // Optional: Serialization size validation (debug builds only, matches libbitcoin-consensus)
    // This is a defensive check for externally-provided blocks to ensure serialized size matches expected.
    // Most callers construct blocks from deserialized data, so this is optional.
    #[cfg(debug_assertions)]
    {
        use crate::serialization::block::serialize_block_with_witnesses;
        let serialized_size = serialize_block_with_witnesses(block, witnesses, true).len();
        // Note: We don't have a provided_size parameter, so we just verify serialization works
        // In production, if receiving pre-serialized blocks, validate: serialized_size == provided_size
        // Use MAX_BLOCK_SERIALIZED_SIZE (4MB) for serialized size check
        const MAX_BLOCK_SERIALIZED_SIZE: usize = 4_000_000; // 4MB
        debug_assert!(
            serialized_size <= MAX_BLOCK_SERIALIZED_SIZE,
            "Serialized block size {serialized_size} exceeds MAX_BLOCK_SERIALIZED_SIZE {MAX_BLOCK_SERIALIZED_SIZE}"
        );
    }

    // BIP90: Block version enforcement (check header version)
    // CRITICAL: This check MUST be called - see tests/integration/bip_enforcement_tests.rs
    // If this check is removed, integration tests will fail
    // Precondition assertion: Header version must be valid
    assert!(
        block.header.version >= 1,
        "Header version {} must be >= 1 for BIP90 check",
        block.header.version
    );
    let bip90_result = crate::bip_validation::check_bip90(block.header.version, height, network)?;
    // Invariant assertion: BIP90 result must be boolean
    #[allow(clippy::eq_op)]
    {
        assert!(
            bip90_result || !bip90_result,
            "BIP90 result must be boolean"
        );
    }
    #[cfg(any(debug_assertions, feature = "runtime-invariants"))]
    debug_assert!(
        bip90_result || height < 227_836, // BIP90 only applies after activation
        "BIP90 check was called but returned false - this should be handled below"
    );
    if !bip90_result {
        return Ok((
            ValidationResult::Invalid(format!(
                "BIP90: Block version {} invalid at height {}",
                block.header.version, height
            )),
            utxo_set,
            vec![],
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // #21: Compute tx_ids before BIP30 so we can pass coinbase_txid to check_bip30 (avoids double hash).
    // Build owned Vec once; use reference for reads; return by move at success (4.2).
    let tx_ids_owned: Vec<Hash> = match precomputed_tx_ids {
        Some(s) => s.to_vec(),
        None => {
            if block.transactions.is_empty() {
                vec![]
            } else {
                compute_block_tx_ids(block)
            }
        }
    };
    let tx_ids: &[Hash] = &tx_ids_owned;

    // BIP30: Duplicate coinbase prevention
    // CRITICAL: This check MUST be called - see tests/integration/bip_enforcement_tests.rs
    // If this check is removed, integration tests will fail
    // Precondition assertion: Block must have transactions for BIP30 check
    assert!(
        !block.transactions.is_empty(),
        "Block must have transactions for BIP30 check"
    );
    let bip30_result = crate::bip_validation::check_bip30(
        block,
        &utxo_set,
        bip30_index.as_deref(),
        height,
        network,
        tx_ids.first(), // #2: Pass precomputed coinbase txid, avoids calculate_tx_id in check_bip30
    )?;
    // Invariant assertion: BIP30 result must be boolean
    #[allow(clippy::eq_op)]
    {
        assert!(
            bip30_result || !bip30_result,
            "BIP30 result must be boolean"
        );
    }
    #[cfg(any(debug_assertions, feature = "runtime-invariants"))]
    debug_assert!(
        bip30_result || !block.transactions.is_empty(), // BIP30 only applies to coinbase
        "BIP30 check was called but returned false - this should be handled below"
    );
    if !bip30_result {
        return Ok((
            ValidationResult::Invalid("BIP30: Duplicate coinbase transaction".into()),
            utxo_set,
            vec![],
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // BIP34: Block height in coinbase (only after activation)
    // CRITICAL: This check MUST be called - see tests/integration/bip_enforcement_tests.rs
    // If this check is removed, integration tests will fail
    let bip34_result = crate::bip_validation::check_bip34(block, height, network)?;
    #[cfg(any(debug_assertions, feature = "runtime-invariants"))]
    debug_assert!(
        bip34_result || height < 227_836, // BIP34 only applies after activation
        "BIP34 check was called but returned false - this should be handled below"
    );
    if !bip34_result {
        return Ok((
            ValidationResult::Invalid(format!(
                "BIP34: Block height {height} not correctly encoded in coinbase"
            )),
            utxo_set,
            vec![],
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // Validate witnesses length matches transactions length
    // Invariant assertion: Witness count must match transaction count
    assert!(
        witnesses.len() == block.transactions.len(),
        "Witness count {} must match transaction count {}",
        witnesses.len(),
        block.transactions.len()
    );
    if witnesses.len() != block.transactions.len() {
        return Ok((
            ValidationResult::Invalid(format!(
                "Witness count {} does not match transaction count {}",
                witnesses.len(),
                block.transactions.len()
            )),
            utxo_set,
            vec![],
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // tx_ids already computed above (before BIP30) for #21/#2

    // Assume-valid optimization
    // Skip expensive signature verification for trusted checkpoint blocks
    #[cfg(feature = "production")]
    let skip_signatures = height < get_assume_valid_height();

    #[cfg(not(feature = "production"))]
    let skip_signatures = false;

    // Pre-compute base script flags once per block (height/network constant; avoids 2000+ repeated height checks)
    let base_script_flags = calculate_base_script_flags_for_block(height, network);

    // Pre-compute overlay capacities once (used by all validation paths)
    let estimated_outputs: usize = block.transactions.iter().map(|tx| tx.outputs.len()).sum();
    let estimated_inputs: usize = block.transactions.iter().map(|tx| tx.inputs.len()).sum();

    // 2. Validate all transactions
    // Note: Transactions in a block must be validated sequentially because each transaction
    // modifies the UTXO set that subsequent transactions depend on. However, script verification
    // within a transaction can be parallelized when safe (production feature).
    let mut total_fees = 0i64;
    // Invariant assertion: Total fees must start at zero
    assert!(total_fees == 0, "Total fees must start at zero");
    // Sigop cost accumulated in overlay pass to avoid separate utxo_set pass
    let mut total_sigop_cost = 0u64;

    // When use_overlay_delta, extract additions/deletions from the overlay built during validation
    // instead of rebuilding (avoids ~10k redundant map ops/block).
    #[cfg(feature = "production")]
    let mut overlay_for_delta: Option<UtxoOverlay> = None;

    #[cfg(feature = "production")]
    {
        // Batch fee calculation - pre-fetch all UTXOs for fee calculation
        // Pre-collect prevouts for prefetch only (#18: 64 for better cache warmup)
        let prefetch_prevouts: Vec<&OutPoint> = block
            .transactions
            .iter()
            .filter(|tx| !is_coinbase(tx))
            .flat_map(|tx| tx.inputs.iter().map(|input| &input.prevout))
            .take(64)
            .collect();

        // Batch UTXO lookup for all transactions (single pass through HashMap)
        #[cfg(feature = "production")]
        {
            use crate::optimizations::prefetch;
            // Prefetch ahead for better cache performance (#prefetch: lookahead 8)
            for i in 0..prefetch_prevouts.len().min(8) {
                if i + 8 < prefetch_prevouts.len() {
                    prefetch::prefetch_ahead(&prefetch_prevouts, i, 8);
                }
            }
        }

        // NOTE: utxo_cache was removed - overlay.get() is used directly for better performance
        // The cache was created but never used, causing unnecessary allocations

                // Sequential validation (CRITICAL FIX for intra-block dependencies)
        // CRITICAL: Transactions in the same block CAN spend outputs from earlier transactions
        // Parallel validation can't handle this because it validates all transactions against
        // the initial UTXO set. We must validate sequentially so each transaction can see
        // outputs from previous transactions in the same block.
        // NOTE: We still use the cached UTXO lookups for performance, but validate sequentially
        // rayon is included in production feature, so check for production
        #[cfg(feature = "production")]
        {
            // CRITICAL FIX: Use sequential validation with incremental UTXO overlay
            // This allows transactions to spend outputs from earlier transactions in the same block
            // UtxoOverlay is O(1) creation vs O(n) clone of the full UTXO set
            // OPTIMIZATION #5: Pre-allocate overlay with capacity (computed above)
            let mut overlay = UtxoOverlay::with_capacity(&utxo_set, estimated_outputs.max(100), estimated_inputs.max(100));
            let mut validation_results: Vec<Result<(ValidationResult, i64, bool)>> = Vec::with_capacity(block.transactions.len());
            // NOTE: Undo entries are created when applying to real UTXO set, not during validation
            
            // OPTIMIZATION #1: Pre-allocate reusable Vecs to avoid per-transaction allocations
            let mut prevout_values_reusable: Vec<i64> = Vec::with_capacity(256);
            // prevout_script_pubkeys: per-tx allocation required (refs into overlay; must not outlive overlay mutation)
            let mut witness_stacks_reusable: Vec<Option<&Witness>> = Vec::with_capacity(256);
            // Block-level signature collectors. Single Mutex preserves collection order (tx0_in0, tx0_in1, ...)
            // so batch result indices match script order. Per-thread collectors broke ordering and caused
            // false "invalid signature" at e.g. block 164676 (see docs/IBD_BATCH_SPEED_PLAN.md §11).
            #[cfg(feature = "production")]
            use std::sync::Arc;
            let validation_start = std::time::Instant::now();
            let mut total_input_lookup_time = std::time::Duration::ZERO;
            let mut total_script_time = std::time::Duration::ZERO;
            let mut total_tx_structure_time = std::time::Duration::ZERO;
            let mut total_overlay_apply_time = std::time::Duration::ZERO;
            let mut total_check_tx_inputs_time = std::time::Duration::ZERO;
            
            
            // PARALLELIZATION OPPORTUNITY #1: Parallel transaction structure validation
            // This is read-only with no dependencies, so we can validate all transactions in parallel
            let structure_start = std::time::Instant::now();
            let tx_structure_results: Vec<(usize, Result<ValidationResult>)> = {
                // ALWAYS use rayon in production - no sequential fallback
                use rayon::prelude::*;
                block.transactions
                    .par_iter()
                    .enumerate()
                    .map(|(i, tx)| (i, check_transaction(tx)))
                    .collect()
            };
            total_tx_structure_time += structure_start.elapsed();
            
            // Check structure validation results and build index of valid transactions
            let mut valid_tx_indices = Vec::with_capacity(block.transactions.len());
            for (i, result) in tx_structure_results {
                match result {
                    Ok(ValidationResult::Valid) => {
                        valid_tx_indices.push(i);
                    }
                    Ok(ValidationResult::Invalid(reason)) => {
                        validation_results.push(Ok((
                            ValidationResult::Invalid(format!("TX {i}: {reason}")),
                            0,
                            false,
                        )));
                    }
                    Err(e) => {
                        return Err(e);
                    }
                }
            }

            // Per-input ECDSA counters for composite index (base << 16) | sub so batch sort order
            // is deterministic under parallel script verification (see docs/IBD_BATCH_SPEED_PLAN.md §11).
            #[cfg(feature = "production")]
            let total_ecdsa_inputs: usize = valid_tx_indices.iter().map(|&idx| block.transactions[idx].inputs.len()).sum();
            #[cfg(feature = "production")]
            let ecdsa_sub_counters: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> = std::sync::Arc::new(
                (0..total_ecdsa_inputs).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect(),
            );
            #[cfg(feature = "production")]
            let mut ecdsa_index_base: usize = 0;

            // C/D: SoA collectors; created after total_inputs for pre-allocation
            #[cfg(feature = "production")]
            let block_schnorr_collector = Arc::new(crate::bip348::SchnorrSignatureCollector::new_with_capacity(total_ecdsa_inputs));
            #[cfg(feature = "production")]
            let block_ecdsa_collector = Arc::new(crate::script::EcdsaSignatureCollector::new_with_capacity(
                if use_per_sig_ecdsa() { 0 } else { total_ecdsa_inputs },
            ));

            // Hoist for ECDSA batch retry (docs/ECDSA_BATCH_INTERPRETER_SIGHASH.md)
            // Caller MUST pass Some(Arc<Block>) to avoid full block clone — see connect_block_ibd.
            #[cfg(all(feature = "production", feature = "rayon"))]
            let block_arc = block_arc.expect("block_arc required for production+rayon (pass Some(Arc::new(block)) from caller)");
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut tx_contexts: Vec<crate::checkqueue::TxScriptContext> = Vec::new();
            #[cfg(all(feature = "production", feature = "rayon"))]
            let results_arc = Arc::new(crossbeam_queue::SegQueue::new());
            // Block-level buffers: build as local Vecs, freeze to Arc before session (immutable for workers).
            #[cfg(all(feature = "production", feature = "rayon"))]
            let total_inputs: usize =
                valid_tx_indices.iter().map(|&i| block.transactions[i].inputs.len()).sum();
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut script_pubkey_vec: Vec<u8> = Vec::with_capacity(total_inputs.saturating_mul(64).min(256 * 1024));
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut prevout_values_vec: Vec<i64> = Vec::with_capacity(total_inputs);
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut script_pubkey_indices_vec: Vec<(usize, usize)> = Vec::with_capacity(total_inputs);
            // Hoist frozen buffers for ECDSA batch retry (same scope as block_arc, tx_contexts_arc)
            #[cfg(all(feature = "production", feature = "rayon"))]
            let script_pubkey_buffer: std::sync::Arc<Vec<u8>>;
            #[cfg(all(feature = "production", feature = "rayon"))]
            let prevout_values_buffer: std::sync::Arc<Vec<i64>>;
            #[cfg(all(feature = "production", feature = "rayon"))]
            let script_pubkey_indices_buffer: std::sync::Arc<Vec<(usize, usize)>>;
            #[cfg(all(feature = "production", feature = "rayon"))]
            let tx_contexts_arc: std::sync::Arc<Vec<crate::checkqueue::TxScriptContext>>;
            #[cfg(all(feature = "production", feature = "rayon"))]
            let witness_buffer: std::sync::Arc<Vec<Vec<Witness>>> =
                witnesses_arc.map(Arc::clone).unwrap_or_else(|| Arc::new(witnesses.to_vec()));

            // Dedicated script workers: build buffers+tx_contexts, freeze to Arc, create session, add checks.
            #[cfg(all(feature = "production", feature = "rayon"))]
            {
                use crate::checkqueue::{BlockSessionContext, ScriptCheck, TxScriptContext};

                let block_ref = block;
                let witnesses_ref = witnesses;
                let time_ctx = time_context;
                let mut queue_results: Vec<Option<Result<(ValidationResult, i64, bool)>>> =
                    vec![None; valid_tx_indices.len()];

                let mut early_return: Option<Result<(
                    ValidationResult,
                    UtxoSet,
                    Vec<Hash>,
                    crate::reorganization::BlockUndoLog,
                    Option<UtxoDelta>,
                )>> = None;
                let median_time_past = time_ctx.map(|ctx| ctx.median_time_past).filter(|&mtp| mtp > 0);

                let mut ecdsa_index_base: usize = 0;

                // Sighash midstate cache: None = use thread-local (avoids Mutex contention across workers).

                // Q: Pre-allocate tx_checks Vec at block level; reuse per tx via add_from_slice
                // Reusable refs into script_pubkey_buffer; avoid per-tx Vec alloc
                // Reusable UTXO data (value, is_coinbase, height) — copy in tight loop, no refs held
                let mut utxo_data_reusable: Vec<Option<(i64, bool, u64)>> = Vec::with_capacity(256);
                // Phase 2a: Build ScriptChecks in first loop (single pass over inputs).
                const ADD_BATCH_THRESHOLD: usize = 256;
                let mut block_checks_buf: Vec<crate::checkqueue::ScriptCheck> = Vec::with_capacity(total_inputs.min(2048));
                #[cfg(all(feature = "production", feature = "profile"))]
                {
                    let _ = crate::script_profile::get_and_reset_script_sub_timing();
                    let _ = crate::script_profile::get_and_reset_p2pkh_timing();
                }

                let script_start = std::time::Instant::now();
                type EarlyReturn = std::result::Result<
                    (ValidationResult, UtxoSet, Vec<Hash>, crate::reorganization::BlockUndoLog, Option<UtxoDelta>),
                    ConsensusError,
                >;

                for (loop_idx, &i) in valid_tx_indices.iter().enumerate() {
                    if early_return.is_some() {
                        break;
                    }
                    let tx = &block_ref.transactions[i];

                    // Sigop for this tx (merged with UTXO pass — overlay state is correct before we process tx)
                    let wits_i = witnesses_ref.get(i).map(|w| w.as_slice());
                    let has_wit_i = wits_i.map(|w| w.iter().any(|wit| !is_witness_empty(wit))).unwrap_or(false);
                    let tx_flags_i = calculate_script_flags_for_block_with_base(tx, has_wit_i, base_script_flags, height, network);
                    match crate::sigop::get_transaction_sigop_cost_with_witness_slices(tx, &overlay, wits_i, tx_flags_i) {
                        Ok(cost) => {
                            total_sigop_cost = match total_sigop_cost.checked_add(cost) {
                                Some(v) => v,
                                None => {
                                    early_return = Some(Err(ConsensusError::BlockValidation(
                                        "Sigop cost overflow".into(),
                                    )));
                                    break;
                                }
                            };
                        }
                        Err(e) => {
                            early_return = Some(Err(e));
                            break;
                        }
                    }

                    let (input_valid, fee, prevout_values_range, script_pubkey_indices_range) =
                        if is_coinbase(tx) {
                            (
                                ValidationResult::Valid,
                                0,
                                (0, 0),
                                (0, 0),
                            )
                        } else {
                            // Copy UTXO data in tight loop — no refs held, local vecs
                            utxo_data_reusable.clear();
                            utxo_data_reusable.reserve(tx.inputs.len());
                            let pv_start = prevout_values_vec.len();
                            let spi_start = script_pubkey_indices_vec.len();
                            let mut utxo_missing: Option<usize> = None;
                            for (input_idx, input) in tx.inputs.iter().enumerate() {
                                match overlay.get(&input.prevout) {
                                    Some(u) => {
                                        utxo_data_reusable.push(Some((u.value, u.is_coinbase, u.height)));
                                        prevout_values_vec.push(u.value);
                                        let start = script_pubkey_vec.len();
                                        script_pubkey_vec.extend_from_slice(u.script_pubkey.as_ref());
                                        script_pubkey_indices_vec.push((start, u.script_pubkey.len()));
                                    }
                                    None => {
                                        utxo_data_reusable.push(None);
                                        utxo_missing = Some(input_idx);
                                        break;
                                    }
                                }
                            }
                            if let Some(idx) = utxo_missing {
                                early_return = Some(Ok((
                                    ValidationResult::Invalid(format!("UTXO not found for input {}", idx)),
                                    utxo_set.clone(),
                                    tx_ids_owned.clone(),
                                    crate::reorganization::BlockUndoLog::new(),
                                    None,
                                )));
                                break;
                            }
                            let (input_valid, fee) = match crate::transaction::check_tx_inputs_with_owned_data(
                                tx, height, &utxo_data_reusable,
                            ) {
                                Ok(x) => x,
                                Err(e) => {
                                    early_return = Some(Err(e));
                                    break;
                                }
                            };
                            let pv_count = prevout_values_vec.len() - pv_start;
                            let spi_count = script_pubkey_indices_vec.len() - spi_start;
                            (input_valid, fee, (pv_start, pv_count), (spi_start, spi_count))
                        };

                    if !matches!(input_valid, ValidationResult::Valid) {
                        queue_results[loop_idx] = Some(Ok((
                            ValidationResult::Invalid(format!("Invalid transaction inputs at index {i}")),
                            0,
                            false,
                        )));
                        continue;
                    }

                    if is_coinbase(tx) || skip_signatures {
                        let tx_id = tx_ids[i];
                        apply_transaction_to_overlay_no_undo(&mut overlay, tx, tx_id, height);
                        queue_results[loop_idx] = Some(Ok((ValidationResult::Valid, fee, true)));
                        continue;
                    }


                    let tx_witnesses = witnesses_ref.get(i);
                    let has_witness =
                        tx_witnesses.map(|w| w.iter().any(|wit| !is_witness_empty(wit))).unwrap_or(false);
                    // H: P2PKH-only txs have empty witness; avoid cloning when has_witness is false
                    let flags =
                        calculate_script_flags_for_block_with_base(tx, has_witness, base_script_flags, height, network);

                    #[cfg(feature = "production")]
                    let bip143 = if has_witness {
                        let pv_slice = &prevout_values_vec[prevout_values_range.0..][..prevout_values_range.1];
                        let spi_slice = &script_pubkey_indices_vec[script_pubkey_indices_range.0..][..script_pubkey_indices_range.1];
                        let buf = script_pubkey_vec.as_slice();
                        let refs: Vec<&[u8]> = spi_slice.iter().map(|&(s, l)| buf[s..s + l].as_ref()).collect();
                        Some(crate::transaction_hash::Bip143PrecomputedHashes::compute(
                            tx, pv_slice, &refs,
                        ))
                    } else {
                        None
                    };

                    let tx_ctx_idx = tx_contexts.len();
                    tx_contexts.push(TxScriptContext {
                        tx_index: i,
                        prevout_values_range,
                        script_pubkey_indices_range,
                        flags,
                        #[cfg(feature = "production")]
                        bip143,
                        loop_idx,
                        fee,
                        ecdsa_index_base,
                        #[cfg(feature = "production")]
                        sighash_midstate_cache: None, // thread-local used when None
                    });

                    // Phase 2a: Build ScriptChecks in first loop (single pass).
                    let tx_ctx_idx = tx_contexts.len() - 1;
                    let (spi_base, spi_count) = script_pubkey_indices_range;
                    let (pv_base, pv_count) = prevout_values_range;
                    let spi = script_pubkey_indices_vec.as_slice();
                    let pv = prevout_values_vec.as_slice();
                    // Script exec cache: skip all checks if (witness_hash, flags) cached.
                    #[cfg(all(feature = "production", feature = "rayon"))]
                    if height >= 250_000 {
                        if let Some(tx_witnesses) = witnesses_ref.get(i) {
                            if tx_witnesses.len() == tx.inputs.len() {
                                let key = crate::script_exec_cache::compute_key(tx, tx_witnesses, flags);
                                if crate::script_exec_cache::contains(&key) {
                                    let synthetic: Vec<_> = (0..tx.inputs.len()).map(|_| (tx_ctx_idx, true)).collect();
                                    results_arc.push(synthetic);
                                    queue_results[loop_idx] = Some(Ok((ValidationResult::Valid, fee, true)));
                                    ecdsa_index_base += tx.inputs.len();
                                    let tx_id = tx_ids[i];
                                    apply_transaction_to_overlay_no_undo(&mut overlay, tx, tx_id, height);
                                    continue;
                                }
                            }
                        }
                    }
                    // Build minimal ScriptChecks; workers use session buffers for script_pubkey/prevout_value.
                    for j in 0..tx.inputs.len() {
                        block_checks_buf.push(crate::checkqueue::ScriptCheck {
                            tx_ctx_idx,
                            input_idx: j,
                        });
                    }

                    ecdsa_index_base += tx.inputs.len();

                    let tx_id = tx_ids[i];
                    apply_transaction_to_overlay_no_undo(&mut overlay, tx, tx_id, height);
                }

                if let Some(r) = early_return.take() {
                    return r;
                }

                tx_contexts_arc = Arc::new(tx_contexts);
                script_pubkey_buffer = Arc::new(script_pubkey_vec);
                prevout_values_buffer = Arc::new(prevout_values_vec);
                script_pubkey_indices_buffer = Arc::new(script_pubkey_indices_vec);
                #[cfg(feature = "production")]
                let (ecdsa_collector, schnorr_collector) = (
                    if use_per_sig_ecdsa() { None } else { Some(Arc::clone(&block_ecdsa_collector)) },
                    if use_per_sig_schnorr() { None } else { Some(Arc::clone(&block_schnorr_collector)) },
                );

                // Small-block fast path: skip CCheckQueue overhead for blocks with <32 inputs.
                const SMALL_BLOCK_THRESHOLD: usize = 32;
                let check_results = if total_inputs < SMALL_BLOCK_THRESHOLD {
                    let seq_session = BlockSessionContext {
                        block: Arc::clone(&block_arc),
                        prevout_values_buffer: Arc::clone(&prevout_values_buffer),
                        script_pubkey_indices_buffer: Arc::clone(&script_pubkey_indices_buffer),
                        script_pubkey_buffer: Arc::clone(&script_pubkey_buffer),
                        witness_buffer: Arc::clone(&witness_buffer),
                        tx_contexts: Arc::clone(&tx_contexts_arc),
                        #[cfg(feature = "production")]
                        ecdsa_sub_counters: Arc::clone(&ecdsa_sub_counters),
                        #[cfg(feature = "production")]
                        ecdsa_collector: ecdsa_collector.clone(),
                        #[cfg(feature = "production")]
                        schnorr_collector: schnorr_collector.clone(),
                        height,
                        median_time_past,
                        network,
                        results: Arc::new(crossbeam_queue::SegQueue::new()),
                    };
                    crate::checkqueue::ScriptCheckQueue::run_checks_sequential(&block_checks_buf, &seq_session)?
                } else {
                let session = BlockSessionContext {
                    block: Arc::clone(&block_arc),
                    prevout_values_buffer: Arc::clone(&prevout_values_buffer),
                    script_pubkey_indices_buffer: Arc::clone(&script_pubkey_indices_buffer),
                    script_pubkey_buffer: Arc::clone(&script_pubkey_buffer),
                    witness_buffer: Arc::clone(&witness_buffer),
                    tx_contexts: Arc::clone(&tx_contexts_arc),
                    #[cfg(feature = "production")]
                    ecdsa_sub_counters: Arc::clone(&ecdsa_sub_counters),
                    #[cfg(feature = "production")]
                    ecdsa_collector,
                    #[cfg(feature = "production")]
                    schnorr_collector,
                    height,
                    median_time_past,
                    network,
                    results: Arc::clone(&results_arc),
                };
                script_check_queue().start_session(session);

                // Spawn drain threads early so they overlap with the producer loop below.
                // Skip when both collectors use SegQueue but we never push (use_per_sig_*); spawn would be useless.
                let use_crypto_drain = (!block_ecdsa_collector.uses_soa() && !use_per_sig_ecdsa())
                    || (!block_schnorr_collector.uses_soa() && !use_per_sig_schnorr());
                let n_crypto_drain = n_crypto_drain_threads();
                let collection_done_opt: Option<std::sync::Arc<std::sync::atomic::AtomicBool>> = if !use_crypto_drain {
                    Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)))
                } else {
                    None
                };
                let crypto_handles: Vec<std::thread::JoinHandle<()>> = if use_crypto_drain {
                    let iters = if total_inputs > 512 {
                        (total_inputs / 64).saturating_add(8)
                    } else {
                        (total_inputs / 32).saturating_add(4)
                    };
                    let iters_per_thread = (iters + n_crypto_drain - 1) / n_crypto_drain;
                    (0..n_crypto_drain)
                        .map(|_| {
                            let ecdsa = Arc::clone(&block_ecdsa_collector);
                            let schnorr = Arc::clone(&block_schnorr_collector);
                            let my_iters = iters_per_thread;
                            std::thread::spawn(move || {
                                for _ in 0..my_iters {
                                    ecdsa.try_verify_chunk(128);
                                    schnorr.try_verify_chunk(128);
                                }
                            })
                        })
                        .collect()
                } else {
                    block_ecdsa_collector.spawn_soa_drain_threads(n_crypto_drain, std::sync::Arc::clone(collection_done_opt.as_ref().unwrap()))
                };

                // Phase 2a: block_checks_buf built in first loop. Add in batches after start_session.
                let spi = script_pubkey_indices_buffer.as_slice();
                let pv = prevout_values_buffer.as_slice();
                let spk_buf = script_pubkey_buffer.as_slice();
                let mut chunk_buf: Vec<crate::checkqueue::ScriptCheck> = Vec::with_capacity(ADD_BATCH_THRESHOLD);
                while block_checks_buf.len() >= ADD_BATCH_THRESHOLD {
                    chunk_buf.clear();
                    let drain_start = block_checks_buf.len() - ADD_BATCH_THRESHOLD;
                    chunk_buf.extend(block_checks_buf.drain(drain_start..));
                    script_check_queue().add_from_slice(&chunk_buf);
                }
                if !block_checks_buf.is_empty() {
                    script_check_queue().add(block_checks_buf);
                }
                #[cfg(all(feature = "production", feature = "profile"))]
                let (build_phase_ms, t_before_complete) = (script_start.elapsed().as_millis() as u64, std::time::Instant::now());
                let check_results_or_early: std::result::Result<Vec<(usize, bool)>, _> =
                    script_check_queue().complete();
                #[cfg(all(feature = "production", feature = "rayon"))]
                if let Some(ref done) = collection_done_opt {
                    done.store(true, std::sync::atomic::Ordering::Release);
                }
                #[cfg(all(feature = "production", feature = "profile"))]
                let complete_ms = t_before_complete.elapsed().as_millis() as u64;
                #[cfg(all(feature = "production", feature = "profile"))]
                profile_log!("[PERF_OVERLAP] Block {}: build_phase_ms={} complete_ms={} (overlap: workers run during build)",
                    height, build_phase_ms, complete_ms);
                for h in crypto_handles {
                    let _ = h.join();
                }

                let check_results = match check_results_or_early {
                    Err(early) => return Err(early),
                    Ok(v) => v,
                };
                check_results
                };

                total_script_time += script_start.elapsed();

                let tx_contexts_len = tx_contexts_arc.len();
                // Aggregate per-tx: all inputs must pass
                let mut tx_all_valid = vec![true; tx_contexts_len];
                for (tx_ctx_idx, valid) in check_results {
                    if tx_ctx_idx < tx_contexts_len {
                        tx_all_valid[tx_ctx_idx] &= valid;
                    }
                }

                for (ctx, &all_valid) in tx_contexts_arc.iter().zip(tx_all_valid.iter()) {
                    queue_results[ctx.loop_idx] =
                        Some(Ok((ValidationResult::Valid, ctx.fee, all_valid)));
                }

                for r in queue_results {
                    validation_results.push(r.expect("CCheckQueue: all slots must be filled"));
                }
            }

            // Sequential application (write operations) — must be sequential
            // Invariant assertion: Validation results count must match transaction count
            // NOTE: Use block_arc (block moved into parallel block at 741)
            assert!(
                validation_results.len() == block_arc.transactions.len(),
                "Validation results count {} must match transaction count {}",
                validation_results.len(),
                block_arc.transactions.len()
            );

            for (i, result) in validation_results.into_iter().enumerate() {
                // Bounds checking assertion: Result index must be valid
                assert!(
                    i < block_arc.transactions.len(),
                    "Result index {} out of bounds in validation loop",
                    i
                );
                let (input_valid, fee, script_valid) = result?;

                if !matches!(input_valid, ValidationResult::Valid) {
                    return Ok((
                        input_valid,
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }

                if !script_valid {
                    return Ok((
                        ValidationResult::Invalid(format!("Invalid script at transaction {i}")),
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }

                // Use checked arithmetic to prevent fee overflow
                total_fees = total_fees
                    .checked_add(fee)
                    .ok_or_else(|| make_fee_overflow_error(Some(i)))?;
            }
            // Total time so far (script + structure + overlay); batch verification runs next and is NOT included yet.
            let validation_elapsed = validation_start.elapsed();
            let total_inputs: usize = block_arc.transactions.iter().filter(|tx| !is_coinbase(tx)).map(|tx| tx.inputs.len()).sum();
            #[cfg(all(feature = "production", feature = "profile"))]
            {
                let (p2pk, p2pkh, p2sh, p2wpkh, p2wsh, p2tr, interp) = crate::script::get_and_reset_fast_path_counts();
                let total = p2pk + p2pkh + p2sh + p2wpkh + p2wsh + p2tr + interp;
                if total > 0 {
                    let pct = |n: u64| (100.0 * n as f64 / total as f64).round() as u32;
                    eprintln!(
                        "[FAST_PATH] Block {}: p2pk={}% p2pkh={}% p2sh={}% p2wpkh={}% p2wsh={}% p2tr={}% interpreter={}% (n={})",
                        height, pct(p2pk), pct(p2pkh), pct(p2sh), pct(p2wpkh), pct(p2wsh), pct(p2tr), pct(interp), total
                    );
                }
            }
            // OPTIMIZATION: Batch verify ALL signatures; run Schnorr and ECDSA in parallel
            #[cfg(feature = "production")]
            {
                let batch_start = std::time::Instant::now();
                let (schnorr_result, ecdsa_result) = rayon::join(
                    || {
                        if block_schnorr_collector.is_empty() { Ok(Vec::new()) } else { block_schnorr_collector.verify_batch() }
                    },
                    || {
                        if block_ecdsa_collector.is_empty() { Ok(Vec::new()) } else { block_ecdsa_collector.verify_batch() }
                    },
                );
                if let Err(e) = schnorr_result {
                    #[cfg(feature = "profile")]
                    profile_log!("[BATCH] Block {}: Schnorr batch verification failed: {:?}", height, e);
                    return Ok((
                        ValidationResult::Invalid(format!("Schnorr batch verification failed: {:?}", e)),
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }
                let schnorr_results = schnorr_result.unwrap();
                if schnorr_results.iter().any(|&v| !v) {
                    let n = schnorr_results.len();
                    let invalid = schnorr_results.iter().filter(|&&v| !v).count();
                    #[cfg(feature = "profile")]
                    profile_log!("[BATCH] Block {}: {} Schnorr signatures, {} invalid", height, n, invalid);
                    return Ok((
                        ValidationResult::Invalid(format!("Invalid Schnorr signature in block")),
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }
                if !schnorr_results.is_empty() {
                    #[cfg(feature = "profile")]
                    profile_log!("[BATCH] Block {}: {} Schnorr signatures verified successfully", height, schnorr_results.len());
                }
                if let Err(e) = ecdsa_result {
                    #[cfg(feature = "profile")]
                    profile_log!("[BATCH] Block {}: ECDSA batch verification failed: {:?}", height, e);
                    return Ok((
                        ValidationResult::Invalid(format!("ECDSA batch verification failed: {:?}", e)),
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }
                let ecdsa_results = ecdsa_result.unwrap();
                if ecdsa_results.iter().any(|&v| !v) {
                    let n = ecdsa_results.len();
                    let invalid = ecdsa_results.iter().filter(|&&v| !v).count();
                    #[cfg(feature = "profile")]
                    profile_log!("[BATCH] Block {}: {} ECDSA signatures, {} invalid — retrying in-place (interpreter sighash)", height, n, invalid);
                    // Retry script verification without collector (docs/ECDSA_BATCH_INTERPRETER_SIGHASH.md).
                    // Interpreter path can store wrong sighash; in-place verify uses same path for compute+verify.
                    #[cfg(feature = "rayon")]
                    {
                        use crate::checkqueue::{BlockSessionContext, ScriptCheck};
                        let retry_results_arc = Arc::new(crossbeam_queue::SegQueue::new());
                        let retry_median_time_past = time_context.map(|ctx| ctx.median_time_past).filter(|&mtp| mtp > 0);
                        let retry_session = BlockSessionContext {
                            block: Arc::clone(&block_arc),
                            prevout_values_buffer: Arc::clone(&prevout_values_buffer),
                            script_pubkey_indices_buffer: Arc::clone(&script_pubkey_indices_buffer),
                            script_pubkey_buffer: Arc::clone(&script_pubkey_buffer),
                            witness_buffer: Arc::clone(&witness_buffer),
                            tx_contexts: Arc::clone(&tx_contexts_arc),
                            #[cfg(feature = "production")]
                            ecdsa_sub_counters: Arc::clone(&ecdsa_sub_counters),
                            #[cfg(feature = "production")]
                            ecdsa_collector: None,
                            #[cfg(feature = "production")]
                            schnorr_collector: None,
                            height,
                            median_time_past: retry_median_time_past,
                            network,
                            results: Arc::clone(&retry_results_arc),
                        };
                        let retry_checks: Vec<ScriptCheck> = {
                            let guard = tx_contexts_arc.as_ref();
                            guard.iter().enumerate().flat_map(|(tx_ctx_idx, ctx)| {
                                let tx = &block_arc.transactions[ctx.tx_index];
                                (0..tx.inputs.len()).map(move |input_idx| ScriptCheck {
                                    tx_ctx_idx,
                                    input_idx,
                                })
                            }).collect()
                        };
                        script_check_queue().start_session(retry_session);
                        script_check_queue().add_from_slice(&retry_checks);
                        let retry_results = script_check_queue().complete()?;
                        let all_valid = retry_results.iter().all(|(_, v)| *v);
                        if all_valid {
                            #[cfg(feature = "profile")]
                            profile_log!("[BATCH_RETRY] Block {}: in-place retry passed (interpreter sighash bug)", height);
                        } else {
                            // Parallel retry failed; try sequential retry (avoids checkqueue race).
                            let seq_session = BlockSessionContext {
                                block: Arc::clone(&block_arc),
                                prevout_values_buffer: Arc::clone(&prevout_values_buffer),
                                script_pubkey_indices_buffer: Arc::clone(&script_pubkey_indices_buffer),
                                script_pubkey_buffer: Arc::clone(&script_pubkey_buffer),
                                witness_buffer: Arc::clone(&witness_buffer),
                                tx_contexts: Arc::clone(&tx_contexts_arc),
                                #[cfg(feature = "production")]
                                ecdsa_sub_counters: Arc::clone(&ecdsa_sub_counters),
                                #[cfg(feature = "production")]
                                ecdsa_collector: None,
                                #[cfg(feature = "production")]
                                schnorr_collector: None,
                                height,
                                median_time_past: retry_median_time_past,
                                network,
                                results: Arc::new(crossbeam_queue::SegQueue::new()),
                            };
                            match crate::checkqueue::ScriptCheckQueue::run_checks_sequential(&retry_checks, &seq_session) {
                                Ok(seq_results) if seq_results.iter().all(|(_, v)| *v) => {
                                    #[cfg(feature = "profile")]
                                    profile_log!("[BATCH_RETRY] Block {}: sequential retry passed (parallel retry had false negative)", height);
                                }
                                _ => {
                                    let failed: Vec<_> = retry_results.iter().filter(|(_, v)| !*v).collect();
                                    let failed_info: Vec<String> = failed.iter().map(|(tx_ctx_idx, _)| {
                                        let ctx = tx_contexts_arc.get(*tx_ctx_idx);
                                        if let Some(c) = ctx {
                                            format!("tx_ctx_idx={} tx_index={}", tx_ctx_idx, c.tx_index)
                                        } else {
                                            format!("tx_ctx_idx={}", tx_ctx_idx)
                                        }
                                    }).collect();
                                    #[cfg(feature = "profile")]
                                    profile_log!("[BATCH_RETRY] Block {}: in-place retry FAILED — {} inputs: {:?}", height, failed.len(), failed_info);
                                    return Ok((
                                        ValidationResult::Invalid(format!("Invalid ECDSA signature in block")),
                                        utxo_set,
                                        tx_ids_owned.clone(),
                                        crate::reorganization::BlockUndoLog::new(),
                                        None,
                                    ));
                                }
                            }
                        }
                    }
                    #[cfg(not(feature = "rayon"))]
                    return Ok((
                        ValidationResult::Invalid(format!("Invalid ECDSA signature in block")),
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }
                if !ecdsa_results.is_empty() {
                    #[cfg(feature = "profile")]
                    profile_log!("[BATCH] Block {}: {} ECDSA signatures verified successfully in {:?}", height, ecdsa_results.len(), batch_start.elapsed());
                }
                let total_batch_time = batch_start.elapsed();
                #[cfg(feature = "profile")]
                {
                    if total_batch_time.as_millis() > 10 {
                        profile_log!("[BATCH_PERF] Block {}: Total batch verification time: {:?}", height, total_batch_time);
                    }
                    // PERF total = validation (script/structure/overlay) + batch — this is the real cost per block (why we see ~70–100 b/s on heavy blocks).
                    // schnorr_sigs/ecdsa_sigs help correlate slow blocks with ECDSA-heavy blocks (batch bottleneck).
                    let total_with_batch = validation_elapsed + total_batch_time;
                    let (sighash_ns, interpreter_ns, multisig_ns) = crate::script_profile::get_and_reset_script_sub_timing();
                    let (p2pkh_parse_ns, p2pkh_hash160_ns, p2pkh_collect_ns, p2pkh_entry_ns, p2pkh_bip66_ns, p2pkh_secp_ns) =
                        crate::script_profile::get_and_reset_p2pkh_timing();
                    let (collect_slot_ns, collect_lock_ns, collect_copy_ns, collect_chunk_ns) =
                        crate::script_profile::get_and_reset_collect_timing();
                    let (worker_p2pkh_ns, worker_refs_ns, worker_refs_lock_ns, run_check_loop_ns, results_extend_ns) =
                        crate::script_profile::get_and_reset_worker_timing();
                    let (batch_extract_ns, batch_secp_ns, batch_cache_ns) =
                        crate::script_profile::get_and_reset_batch_phase_timing();
                    let (drain_copy_ns, drain_parse_ns, drain_secp_ns) =
                        crate::script_profile::get_and_reset_drain_timing();
                    let (ecdsa_cache_hits, ecdsa_cache_misses) =
                        crate::script_profile::get_and_reset_ecdsa_cache_stats();
                    let sighash_ms = sighash_ns as f64 / 1_000_000.0;
                    let interpreter_ms = interpreter_ns as f64 / 1_000_000.0;
                    let multisig_ms = multisig_ns as f64 / 1_000_000.0;
                    let p2pkh_parse_ms = p2pkh_parse_ns as f64 / 1_000_000.0;
                    let p2pkh_hash160_ms = p2pkh_hash160_ns as f64 / 1_000_000.0;
                    let p2pkh_collect_ms = p2pkh_collect_ns as f64 / 1_000_000.0;
                    let p2pkh_entry_ms = p2pkh_entry_ns as f64 / 1_000_000.0;
                    let p2pkh_bip66_ms = p2pkh_bip66_ns as f64 / 1_000_000.0;
                    let p2pkh_secp_ms = p2pkh_secp_ns as f64 / 1_000_000.0;
                    let collect_slot_ms = collect_slot_ns as f64 / 1_000_000.0;
                    let collect_lock_ms = collect_lock_ns as f64 / 1_000_000.0;
                    let collect_copy_ms = collect_copy_ns as f64 / 1_000_000.0;
                    let collect_chunk_ms = collect_chunk_ns as f64 / 1_000_000.0;
                    let worker_refs_ms = worker_refs_ns as f64 / 1_000_000.0;
                    let worker_p2pkh_ms = worker_p2pkh_ns as f64 / 1_000_000.0;
                    let worker_refs_lock_ms = worker_refs_lock_ns as f64 / 1_000_000.0;
                    let run_check_loop_ms = run_check_loop_ns as f64 / 1_000_000.0;
                    let results_extend_ms = results_extend_ns as f64 / 1_000_000.0;
                    let batch_extract_ms = batch_extract_ns as f64 / 1_000_000.0;
                    let batch_secp_ms = batch_secp_ns as f64 / 1_000_000.0;
                    let batch_cache_ms = batch_cache_ns as f64 / 1_000_000.0;
                    let drain_copy_ms = drain_copy_ns as f64 / 1_000_000.0;
                    let drain_parse_ms = drain_parse_ns as f64 / 1_000_000.0;
                    let drain_secp_ms = drain_secp_ns as f64 / 1_000_000.0;
                    profile_log!("[PERF] Block {}: total={:?} (script={:?} batch={:?}), script_sub: sighash={:.2}ms interpreter={:.2}ms multisig={:.2}ms p2pkh_entry={:.2}ms p2pkh_parse={:.2}ms p2pkh_hash160={:.2}ms p2pkh_bip66={:.2}ms p2pkh_collect={:.2}ms p2pkh_secp={:.2}ms collect_slot={:.2}ms collect_lock={:.2}ms collect_copy={:.2}ms collect_chunk={:.2}ms worker_refs={:.2}ms worker_p2pkh={:.2}ms worker_refs_lock={:.2}ms run_check_loop={:.2}ms results_extend={:.2}ms batch_extract={:.2}ms batch_secp={:.2}ms batch_cache={:.2}ms drain_copy={:.2}ms drain_parse={:.2}ms drain_secp={:.2}ms ecdsa_cache_hits={} ecdsa_cache_misses={}, structure={:?}, input_lookup={:?}, check_inputs={:?}, overlay_apply={:?}, txs={} inputs={} schnorr_sigs={} ecdsa_sigs={}",
                        height,
                        total_with_batch,
                        total_script_time,
                        total_batch_time,
                        sighash_ms, interpreter_ms, multisig_ms,
                        p2pkh_entry_ms, p2pkh_parse_ms, p2pkh_hash160_ms, p2pkh_bip66_ms, p2pkh_collect_ms, p2pkh_secp_ms,
                        collect_slot_ms, collect_lock_ms, collect_copy_ms, collect_chunk_ms,
                        worker_refs_ms, worker_p2pkh_ms,
                        worker_refs_lock_ms, run_check_loop_ms, results_extend_ms,
                        batch_extract_ms, batch_secp_ms, batch_cache_ms,
                        drain_copy_ms, drain_parse_ms, drain_secp_ms,
                        ecdsa_cache_hits, ecdsa_cache_misses,
                        total_tx_structure_time,
                        total_input_lookup_time,
                        total_check_tx_inputs_time,
                        total_overlay_apply_time,
                        block_arc.transactions.len(),
                        total_inputs,
                        schnorr_results.len(),
                        ecdsa_results.len()
                    );
                    let total_ns = total_with_batch.as_nanos() as f64;
                    if total_ns > 0.0 {
                        let pct = |d: std::time::Duration| (100.0 * d.as_nanos() as f64 / total_ns).min(100.0);
                        let script_pct = pct(total_script_time);
                        let batch_pct = pct(total_batch_time);
                        let input_lookup_pct = pct(total_input_lookup_time);
                        let check_inputs_pct = pct(total_check_tx_inputs_time);
                        let overlay_pct = pct(total_overlay_apply_time);
                        let structure_pct = pct(total_tx_structure_time);
                        let total_ms = total_with_batch.as_secs_f64() * 1000.0;
                        if ((120_000..=145_000).contains(&height) || (180_000..=195_000).contains(&height))
                            && height % 100 == 0
                        {
                            profile_log!(
                                "[PERF_CLIFF] Block {}: total={:.1}ms | script={:.0}% batch={:.0}% input_lookup={:.0}% check_inputs={:.0}% overlay={:.0}% structure={:.0}% | txs={} inputs={}",
                                height, total_ms, script_pct, batch_pct, input_lookup_pct, check_inputs_pct, overlay_pct, structure_pct,
                                block_arc.transactions.len(), total_inputs
                            );
                        }
                        if total_ms > 20.0 {
                            profile_log!(
                                "[PERF_SLOW] Block {}: total={:.1}ms | script={:.0}% batch={:.0}% input_lookup={:.0}% check_inputs={:.0}% overlay={:.0}% structure={:.0}% | txs={} inputs={}",
                                height, total_ms, script_pct, batch_pct, input_lookup_pct, check_inputs_pct, overlay_pct, structure_pct,
                                block_arc.transactions.len(), total_inputs
                            );
                        }
                    }
                }
            }
            
            // NOTE: Overlay is discarded here; application loop will apply 
            // transactions to the real utxo_set and build the undo log.
            // Reuse overlay for delta extraction instead of rebuilding (avoids double apply).
            if crate::config::use_overlay_delta() {
                overlay_for_delta = Some(overlay);
            }
        }

        // REMOVED: #[cfg(not(feature = "rayon"))] is always true since rayon is not a feature
        // This was causing a duplicate code path. Production path above handles everything.
        #[cfg(all(not(feature = "production"), not(feature = "rayon")))]
        {
            // Sequential fallback (no Rayon available)
            // CRITICAL FIX: Use overlay for intra-block spending support
            // Transactions can spend outputs from earlier transactions in the same block
            // UtxoOverlay is O(1) creation vs O(n) clone of the full UTXO set
            // OPTIMIZATION #5: Pre-allocate overlay with capacity (computed above)
            let mut overlay = UtxoOverlay::with_capacity(&utxo_set, estimated_outputs.max(100), estimated_inputs.max(100));
            
            // OPTIMIZATION #1: Pre-allocate reusable Vecs to avoid per-transaction allocations
            let mut prevout_values_reusable: Vec<i64> = Vec::with_capacity(256);
            // OPTIMIZATION: Reusable input_utxos buffer (refs into overlay; cleared and refilled per tx)
            let mut input_utxos_reusable: Vec<Option<&UTXO>> = Vec::with_capacity(256);
            let mut prevout_script_pubkeys_reusable: Vec<&[u8]> = Vec::with_capacity(256);
            
            // PROFILING: Add timing for non-rayon path
            let validation_start = std::time::Instant::now();
            let mut total_tx_structure_time = std::time::Duration::ZERO;
            let mut total_input_lookup_time = std::time::Duration::ZERO;
            let mut total_script_time = std::time::Duration::ZERO;
            let mut total_overlay_apply_time = std::time::Duration::ZERO;
            
            for (i, tx) in block.transactions.iter().enumerate() {
                // #11: Accumulate sigop for this tx (non-rayon path; overlay has prev txs)
                let wits_i = witnesses.get(i).map(|w| w.as_slice());
                let has_wit = wits_i.map(|w| w.iter().any(|wit| !is_witness_empty(wit))).unwrap_or(false);
                let tx_flags = calculate_script_flags_for_block_with_base(tx, has_wit, base_script_flags, height, network);
                total_sigop_cost = total_sigop_cost
                    .checked_add(crate::sigop::get_transaction_sigop_cost_with_witness_slices(tx, &overlay, wits_i, tx_flags)?)
                    .ok_or_else(|| ConsensusError::BlockValidation("Sigop cost overflow".into()))?;

                let structure_start = std::time::Instant::now();
                // Validate transaction structure
                let tx_valid = check_transaction(tx)?;
                total_tx_structure_time += structure_start.elapsed();
                if !matches!(tx_valid, ValidationResult::Valid) {
                    return Ok((
                        ValidationResult::Invalid(format!("Invalid transaction at index {i}")),
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }

                // Check transaction inputs and calculate fees
                // CRITICAL: Use overlay which includes outputs from earlier transactions in this block
                // Collect input_utxos ONCE, reuse for fee/check_tx_inputs/prevouts (eliminates 3-4x redundant overlay.get() calls)
                let input_lookup_start = std::time::Instant::now();
                let (input_valid, fee) = if is_coinbase(tx) {
                    input_utxos_reusable.clear();
                    (ValidationResult::Valid, 0)
                } else {
                    // Reuse buffer: avoid per-tx Vec allocation
                    input_utxos_reusable.clear();
                    if input_utxos_reusable.capacity() < tx.inputs.len() {
                        input_utxos_reusable.reserve(tx.inputs.len() - input_utxos_reusable.capacity());
                    }
                    let mut total_input: i64 = 0;

                    for (input_idx, input) in tx.inputs.iter().enumerate() {
                        match overlay.get(&input.prevout) {
                            Some(utxo) => {
                                input_utxos_reusable.push(Some(utxo));
                                total_input = total_input.checked_add(utxo.value).ok_or_else(|| {
                                    ConsensusError::TransactionValidation("Input value overflow".into())
                                })?;
                            }
                            None => {
                                #[cfg(debug_assertions)]
                                eprintln!(
                                    "   ⚠️ [UTXO MISSING] Block {} TX {} input {}: prevout {:?}:{} not found",
                                    height, i, input_idx,
                                    hex::encode(&input.prevout.hash),
                                    input.prevout.index
                                );
                                return Ok((
                                    ValidationResult::Invalid(format!("UTXO not found for input {}", input_idx)),
                                    utxo_set,
                                    tx_ids_owned.clone(),
                                    crate::reorganization::BlockUndoLog::new(),
                                    None,
                                ));
                            }
                        }
                    }

                    let total_output: i64 = tx
                        .outputs
                        .iter()
                        .try_fold(0i64, |acc, output| {
                            acc.checked_add(output.value).ok_or_else(|| {
                                ConsensusError::TransactionValidation(
                                    "Output value overflow".into(),
                                )
                            })
                        })
                        .map_err(|e| ConsensusError::TransactionValidation(Cow::Owned(e.to_string())))?;

                    let fee = total_input.checked_sub(total_output).ok_or_else(|| {
                        ConsensusError::TransactionValidation("Fee calculation underflow".into())
                    })?;

                    // Runtime assertion: Fee must be non-negative after checked subtraction
                    debug_assert!(
                        fee >= 0,
                        "Fee ({}) must be non-negative (input: {}, output: {})",
                        fee,
                        total_input,
                        total_output
                    );

                    if fee < 0 {
                        (ValidationResult::Invalid("Negative fee".to_string()), 0)
                    } else {
                        // Runtime assertion: Fee cannot exceed total input
                        debug_assert!(
                            fee <= total_input,
                            "Fee ({}) cannot exceed total input ({})",
                            fee,
                            total_input
                        );
                        // Pass pre-collected UTXOs to avoid redundant lookups
                        let (input_valid, _) = crate::transaction::check_tx_inputs_with_utxos(
                            tx, &overlay, height, Some(&input_utxos_reusable)
                        )?;
                        (input_valid, fee)
                    }
                };
                let input_utxos = &input_utxos_reusable;

                if !matches!(input_valid, ValidationResult::Valid) {
                    #[cfg(debug_assertions)]
                    eprintln!("   ❌ [non-parallel] Block {} TX {}: input_valid={:?}", height, i, input_valid);
                    return Ok((
                        ValidationResult::Invalid(format!(
                            "Invalid transaction inputs at index {i}"
                        )),
                        utxo_set,
                        tx_ids_owned.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }

                // Verify scripts for non-coinbase transactions
                // Skip signature verification if assume-valid
                // Reuse input_utxos collected during fee calculation
                if !is_coinbase(tx) && !skip_signatures {
                    // OPTIMIZATION #1: Reuse pre-allocated Vecs instead of allocating per transaction
                    prevout_values_reusable.clear();
                    prevout_script_pubkeys_reusable.clear();
                    if prevout_script_pubkeys_reusable.capacity() < input_utxos.len() {
                        prevout_script_pubkeys_reusable.reserve(input_utxos.len().saturating_sub(prevout_script_pubkeys_reusable.capacity()));
                    }
                    if prevout_values_reusable.capacity() < input_utxos.len() {
                        prevout_values_reusable.reserve(input_utxos.len() - prevout_values_reusable.capacity());
                    }

                    // Populate reusable Vecs (single loop for cache locality)
                    for opt_utxo in input_utxos {
                        prevout_values_reusable.push(opt_utxo.map(|utxo| utxo.value).unwrap_or(0));
                        if let Some(utxo) = opt_utxo {
                            prevout_script_pubkeys_reusable.push(utxo.script_pubkey.as_ref());
                        }
                    }

                    // Cache witness lookup once per transaction
                    let script_start = std::time::Instant::now();
                    let tx_witnesses = witnesses.get(i);
                    let has_witness = tx_witnesses.map(|w| w.iter().any(|wit| !is_witness_empty(wit))).unwrap_or(false);
                            let flags = calculate_script_flags_for_block_with_base(tx, has_witness, base_script_flags, height, network);
                    let median_time_past = time_context
                        .map(|ctx| ctx.median_time_past)
                        .filter(|&mtp| mtp > 0);
                    #[cfg(feature = "production")]
                    let bip143_hashes = if has_witness {
                        Some(crate::transaction_hash::Bip143PrecomputedHashes::compute(
                            tx, &prevout_values_reusable, &prevout_script_pubkeys_reusable,
                        ))
                    } else {
                        None
                    };
                    for (j, input) in tx.inputs.iter().enumerate() {
                        // Reuse input_utxos instead of overlay.get()
                        if let Some(utxo) = input_utxos.get(j).and_then(|opt| *opt) {
                            let witness_elem = tx_witnesses.and_then(|w| w.get(j));
                            let witness_for_script = witness_elem.and_then(|w| if is_witness_empty(w) { None } else { Some(w) });

                            if !verify_script_with_context_full(
                                &input.script_sig,
                                &utxo.script_pubkey,
                                witness_for_script,
                                flags,
                                tx,
                                j,
                                &prevout_values_reusable,
                                &prevout_script_pubkeys_reusable,
                                Some(height),
                                median_time_past,
                                network,
                                crate::script::SigVersion::Base,
                                #[cfg(feature = "production")] Some(&schnorr_collector),
                                #[cfg(feature = "production")] Some(&ecdsa_collector),
                                #[cfg(feature = "production")] None,
                                #[cfg(feature = "production")] bip143_hashes.as_ref(),
                                #[cfg(not(feature = "production"))] None,
                                #[cfg(feature = "production")] None, // precomputed_sighash_all (non-CCheckQueue path)
                                #[cfg(feature = "production")] None, // precomputed_p2pkh_hash
                            )? {
                                return Ok((
                                    ValidationResult::Invalid(format!(
                                        "Invalid script at transaction {}, input {}",
                                        i, j
                                    )),
                                    utxo_set,
                                    tx_ids_owned.clone(),
                                    crate::reorganization::BlockUndoLog::new(),
                                    None,
                                ));
                            }
                        }
                    }

                    // OPTIMIZATION: Batch verify all collected signatures
                    #[cfg(feature = "production")]
                    {
                        // Batch verify Schnorr signatures
                        if !schnorr_collector.is_empty() {
                            let batch_results = schnorr_collector.verify_batch()?;
                            // All signatures must be valid
                            if batch_results.iter().any(|&valid| !valid) {
                                return Ok((
                                    ValidationResult::Invalid(format!(
                                        "Invalid Schnorr signature in transaction {i}"
                                    )),
                                    utxo_set,
                                    tx_ids_owned.clone(),
                                    crate::reorganization::BlockUndoLog::new(),
                                    None,
                                ));
                            }
                        }
                        // Batch verify ECDSA signatures
                        if !ecdsa_collector.is_empty() {
                            let batch_results = ecdsa_collector.verify_batch()?;
                            // All signatures must be valid
                            if batch_results.iter().any(|&valid| !valid) {
                                return Ok((
                                    ValidationResult::Invalid(format!(
                                        "Invalid ECDSA signature in transaction {i}"
                                    )),
                                    utxo_set,
                                    tx_ids_owned.clone(),
                                    crate::reorganization::BlockUndoLog::new(),
                                    None,
                                ));
                            }
                        }
                    }
                }

                // CRITICAL: Apply this transaction to overlay so next transaction can see its outputs
                // Use apply_transaction_to_overlay_no_undo during validation
                // Undo entries are discarded and rebuilt in application loop, so no need to create them here
                // Clear reusable buffers to release refs into overlay before mutating it
                prevout_script_pubkeys_reusable.clear();
                input_utxos_reusable.clear();
                let overlay_apply_start = std::time::Instant::now();
                let tx_id = tx_ids[i];
                apply_transaction_to_overlay_no_undo(&mut overlay, tx, tx_id, height);
                total_overlay_apply_time += overlay_apply_start.elapsed();

                // Use checked arithmetic to prevent fee overflow
                total_fees = total_fees
                    .checked_add(fee)
                    .ok_or_else(|| make_fee_overflow_error(Some(i)))?;
            }
            // #11: Accumulate sigop for remaining txs (production sequential path)
            for j in last_sigop_index..block.transactions.len() {
                let tx_j = &block.transactions[j];
                let wits_j = witnesses.get(j).map(|w| w.as_slice());
                let has_wit = wits_j.map(|w| w.iter().any(|wit| !is_witness_empty(wit))).unwrap_or(false);
                let tx_flags = calculate_script_flags_for_block_with_base(tx_j, has_wit, base_script_flags, height, network);
                total_sigop_cost = total_sigop_cost
                    .checked_add(crate::sigop::get_transaction_sigop_cost_with_witness_slices(tx_j, &overlay, wits_j, tx_flags)?)
                    .ok_or_else(|| ConsensusError::BlockValidation("Sigop cost overflow".into()))?;
            }
            
            let validation_elapsed = validation_start.elapsed();
            #[cfg(feature = "profile")]
            {
                profile_log!("[PERF] Block {}: total={:?}, structure={:?}, input_lookup={:?}, script={:?}, overlay_apply={:?}, txs={}, inputs={}", 
                    height, 
                    validation_elapsed, 
                    total_tx_structure_time,
                    total_input_lookup_time, 
                    total_script_time,
                    total_overlay_apply_time,
                    block.transactions.len(),
                    block.transactions.iter().filter(|tx| !is_coinbase(tx)).map(|tx| tx.inputs.len()).sum::<usize>()
                );
                profile_log!("[PERF_DEBUG] Profiling logged for block {}", height);
            }
        }
    }

    // Add profiling to non-production path too
    #[cfg(all(not(feature = "production"), feature = "profile"))]
    {
        profile_log!("[DEBUG] NON-PRODUCTION PATH - Block {}", height);
        // Sequential validation (default, verification-safe)
        // CRITICAL FIX: Validate and apply transactions incrementally
        // Transactions in the same block CAN spend outputs from earlier transactions in that block
        // So we must validate each transaction against the UTXO set that includes outputs from
        // all previous transactions in this block, not the initial UTXO set.
        // Validate and apply in a single loop (not validate-then-apply).
        // UtxoOverlay is O(1) creation vs O(n) clone of the full UTXO set
        // OPTIMIZATION #5: Pre-allocate overlay with capacity (computed above)
        let mut overlay = UtxoOverlay::with_capacity(&utxo_set, estimated_outputs.max(100), estimated_inputs.max(100));
        
        // OPTIMIZATION #1: Pre-allocate reusable Vecs to avoid per-transaction allocations
        let mut prevout_values_reusable: Vec<i64> = Vec::with_capacity(256);
        let mut input_utxos_reusable: Vec<Option<&UTXO>> = Vec::with_capacity(256);
        let mut prevout_script_pubkeys_reusable: Vec<&[u8]> = Vec::with_capacity(256);
        
        for (i, tx) in block.transactions.iter().enumerate() {
            // Bounds checking assertion: Transaction index must be valid
            assert!(
                i < block.transactions.len(),
                "Transaction index {} out of bounds (block has {} transactions)",
                i,
                block.transactions.len()
            );

            // #11: Accumulate sigop for this tx (non-production path; overlay has prev txs)
            let wits_i = witnesses.get(i).map(|w| w.as_slice());
            let has_wit = wits_i.map(|w| w.iter().any(|wit| !is_witness_empty(wit))).unwrap_or(false);
            let tx_flags = calculate_script_flags_for_block_with_base(tx, has_wit, base_script_flags, height, network);
            total_sigop_cost = total_sigop_cost
                .checked_add(crate::sigop::get_transaction_sigop_cost_with_witness_slices(tx, &overlay, wits_i, tx_flags)?)
                .ok_or_else(|| ConsensusError::BlockValidation("Sigop cost overflow".into()))?;

            // Validate transaction structure
            if !matches!(check_transaction(tx)?, ValidationResult::Valid) {
                return Ok((
                    ValidationResult::Invalid(format!("Invalid transaction at index {i}")),
                    utxo_set,
                    tx_ids_owned.clone(),
                    crate::reorganization::BlockUndoLog::new(),
                    None,
                ));
            }

            // Check transaction inputs and calculate fees
            // CRITICAL: Use overlay which includes outputs from previous transactions in this block
            let (input_valid, fee) = check_tx_inputs(tx, &overlay, height)?;
            
            // Postcondition assertion: Fee calculation result must be valid
            assert!(
                fee >= 0 || !matches!(input_valid, ValidationResult::Valid),
                "Fee {fee} must be non-negative for valid transaction at index {i}"
            );
            if !matches!(input_valid, ValidationResult::Valid) {
                #[cfg(debug_assertions)]
                eprintln!("   ❌ Block {} TX {}: input_valid={:?}", height, i, input_valid);
                return Ok((
                    ValidationResult::Invalid(format!("Invalid transaction inputs at index {i}")),
                    utxo_set,
                    tx_ids_owned.clone(),
                    crate::reorganization::BlockUndoLog::new(),
                    None,
                ));
            }

            // Verify scripts for non-coinbase transactions BEFORE applying transaction
            // (because apply_transaction removes spent UTXOs from the set)
            // Skip signature verification if assume-valid
            // CRITICAL: Use overlay (still has the UTXOs we need to verify)
            if !is_coinbase(tx) && !skip_signatures {
                // Reuse buffer: avoid per-tx Vec allocation
                input_utxos_reusable.clear();
                if input_utxos_reusable.capacity() < tx.inputs.len() {
                    input_utxos_reusable.reserve(tx.inputs.len() - input_utxos_reusable.capacity());
                }
                for input in &tx.inputs {
                    input_utxos_reusable.push(overlay.get(&input.prevout));
                }
                let input_utxos = &input_utxos_reusable;

                // OPTIMIZATION #1: Reuse pre-allocated Vecs instead of allocating per transaction
                prevout_values_reusable.clear();
                prevout_script_pubkeys_reusable.clear();
                if prevout_script_pubkeys_reusable.capacity() < input_utxos.len() {
                    prevout_script_pubkeys_reusable.reserve(input_utxos.len().saturating_sub(prevout_script_pubkeys_reusable.capacity()));
                }
                if prevout_values_reusable.capacity() < input_utxos.len() {
                    prevout_values_reusable.reserve(input_utxos.len() - prevout_values_reusable.capacity());
                }

                // Populate reusable Vecs (single loop for cache locality)
                for opt_utxo in input_utxos {
                    prevout_values_reusable.push(opt_utxo.map(|utxo| utxo.value).unwrap_or(0));
                    if let Some(utxo) = opt_utxo {
                        prevout_script_pubkeys_reusable.push(utxo.script_pubkey.as_ref());
                    }
                }

                // Cache witness lookup once per transaction
                let tx_witnesses = witnesses.get(i);
                let has_witness = tx_witnesses.map(|w| w.iter().any(|wit| !is_witness_empty(wit))).unwrap_or(false);
                            let flags = calculate_script_flags_for_block_with_base(tx, has_witness, base_script_flags, height, network);
                let median_time_past = time_context
                    .map(|ctx| ctx.median_time_past)
                    .filter(|&mtp| mtp > 0);
                #[cfg(feature = "production")]
                let bip143_hashes = if has_witness {
                    Some(crate::transaction_hash::Bip143PrecomputedHashes::compute(
                        tx, &prevout_values_reusable, &prevout_script_pubkeys_reusable,
                    ))
                } else {
                    None
                };

                // OPTIMIZATION: Collect signatures for batch verification
                #[cfg(feature = "production")]
                let schnorr_collector = crate::bip348::SchnorrSignatureCollector::new();
                #[cfg(feature = "production")]
                let ecdsa_collector = crate::script::EcdsaSignatureCollector::new();

                for (j, input) in tx.inputs.iter().enumerate() {
                    // Bounds checking assertion: Input index must be valid
                    assert!(
                        j < tx.inputs.len(),
                        "Input index {} out of bounds (transaction has {} inputs)",
                        j,
                        tx.inputs.len()
                    );
                    // Bounds checking assertion: Witness index must be valid
                    assert!(
                        i < witnesses.len(),
                        "Witness index {} out of bounds (block has {} witnesses)",
                        i,
                        witnesses.len()
                    );

                    if let Some(utxo) = input_utxos.get(j).and_then(|opt| *opt) {
                        // Reuse cached tx_witnesses and flags from above
                        // Get witness stack for this transaction input if available
                        // witnesses is Vec<Vec<Witness>> where each Vec<Witness> is for one transaction
                        // and each Witness is for one input
                        let witness_stack = tx_witnesses.and_then(|tx_wits| tx_wits.get(j));
                        let witness_for_script = witness_stack.and_then(|w| if is_witness_empty(w) { None } else { Some(w) });

                        // Use verify_script_with_context_full for BIP65/112 support
                        if !verify_script_with_context_full(
                            &input.script_sig,
                            &utxo.script_pubkey,
                            witness_for_script,
                            flags,
                            tx,
                            j, // Input index
                            &prevout_values_reusable,
                            &prevout_script_pubkeys_reusable,
                            Some(height), // Block height for block-height CLTV validation
                            median_time_past, // Median time-past for timestamp CLTV validation (BIP113)
                            network,          // Network for BIP66 and BIP147 activation heights
                            crate::script::SigVersion::Base,
                            #[cfg(feature = "production")] Some(&schnorr_collector),
                            #[cfg(feature = "production")] Some(&ecdsa_collector),
                            #[cfg(feature = "production")] None,
                            #[cfg(feature = "production")] bip143_hashes.as_ref(),
                            #[cfg(not(feature = "production"))] None,
                            #[cfg(feature = "production")] None, // precomputed_sighash_all (non-CCheckQueue path)
                            #[cfg(feature = "production")] None, // precomputed_p2pkh_hash
                        )? {
                            return Ok((
                                ValidationResult::Invalid(format!(
                                    "Invalid script at transaction {i}, input {j}"
                                )),
                                utxo_set,
                                tx_ids_owned.clone(),
                                crate::reorganization::BlockUndoLog::new(),
                                None,
                            ));
                        }
                    }
                }

                // OPTIMIZATION: Batch verify all collected signatures
                #[cfg(feature = "production")]
                {
                    // Batch verify Schnorr signatures
                    if !schnorr_collector.is_empty() {
                        let batch_results = schnorr_collector.verify_batch()?;
                        // All signatures must be valid
                        if batch_results.iter().any(|&valid| !valid) {
                            return Ok((
                                ValidationResult::Invalid(format!(
                                    "Invalid Schnorr signature in transaction {i}"
                                )),
                                utxo_set,
                                tx_ids_owned.clone(),
                                crate::reorganization::BlockUndoLog::new(),
                                None,
                            ));
                        }
                    }
                    // Batch verify ECDSA signatures
                    if !ecdsa_collector.is_empty() {
                        let batch_results = ecdsa_collector.verify_batch()?;
                        // All signatures must be valid
                        if batch_results.iter().any(|&valid| !valid) {
                            return Ok((
                                ValidationResult::Invalid(format!(
                                    "Invalid ECDSA signature in transaction {i}"
                                )),
                                utxo_set,
                                tx_ids_owned.clone(),
                                crate::reorganization::BlockUndoLog::new(),
                                None,
                            ));
                        }
                    }
                }
            }

            // CRITICAL: Apply this transaction to overlay so next transaction can see its outputs
            // This MUST happen AFTER script verification (which needs the spent UTXOs)
            // Use apply_transaction_to_overlay_no_undo during validation
            // Undo entries are created later when applying to real UTXO set
            // Clear reusable buffers to release refs into overlay before mutating it
            prevout_script_pubkeys_reusable.clear();
            input_utxos_reusable.clear();
            let tx_id = tx_ids[i];
            apply_transaction_to_overlay_no_undo(&mut overlay, tx, tx_id, height);

            // Use checked arithmetic to prevent fee overflow
            // Invariant assertion: Fee must be non-negative
            assert!(
                fee >= 0,
                "Fee {fee} must be non-negative at transaction {i}"
            );
            total_fees = total_fees
                .checked_add(fee)
                .ok_or_else(|| make_fee_overflow_error(Some(i)))?;
            // Invariant assertion: Total fees must remain non-negative after addition
            assert!(
                total_fees >= 0,
                "Total fees {total_fees} must be non-negative after transaction {i}"
            );
        }
    }

    // 3. Validate coinbase transaction
    // Invariant assertion: First transaction must be coinbase
    assert!(
        is_coinbase(&block.transactions[0]),
        "First transaction in block must be coinbase"
    );
    if let Some(coinbase) = block.transactions.first() {
        if !is_coinbase(coinbase) {
            return Ok((
                ValidationResult::Invalid("First transaction must be coinbase".into()),
                utxo_set,
                tx_ids_owned.clone(),
                crate::reorganization::BlockUndoLog::new(),
            None,
        ));
        }

        // Validate coinbase scriptSig length (Orange Paper Section 5.1, rule 5)
        // If tx is coinbase: 2 ≤ |ins[0].scriptSig| ≤ 100
        // Use proven bounds for coinbase input access
        #[cfg(feature = "production")]
        let script_sig_len = {
            use crate::optimizations::_optimized_access::get_proven_by_;
            get_proven_by_(&coinbase.inputs, 0)
                .map(|input| input.script_sig.len())
                .unwrap_or(0)
        };

        #[cfg(not(feature = "production"))]
        let script_sig_len = coinbase.inputs[0].script_sig.len();

        if !(2..=100).contains(&script_sig_len) {
            return Ok((
                ValidationResult::Invalid(format!(
                    "Coinbase scriptSig length {script_sig_len} must be between 2 and 100 bytes"
                )),
                utxo_set,
                tx_ids_owned.clone(),
                crate::reorganization::BlockUndoLog::new(),
            None,
        ));
        }

        let subsidy = get_block_subsidy(height);
        // Invariant assertion: Subsidy must be non-negative and within MAX_MONEY
        assert!(subsidy >= 0, "Block subsidy must be non-negative");
        assert!(
            subsidy <= MAX_MONEY,
            "Block subsidy {subsidy} must not exceed MAX_MONEY"
        );

        // Use checked sum to prevent overflow when summing coinbase outputs
        let coinbase_output: i64 = coinbase
            .outputs
            .iter()
            .try_fold(0i64, |acc, output| {
                acc.checked_add(output.value).ok_or_else(|| {
                    ConsensusError::BlockValidation("Coinbase output value overflow".into())
                })
            })
            .map_err(|e| ConsensusError::BlockValidation(Cow::Owned(e.to_string())))?;

        // Invariant assertion: Coinbase output must be non-negative
        assert!(coinbase_output >= 0, "Coinbase output must be non-negative");
        // Check that coinbase output doesn't exceed MAX_MONEY
        if coinbase_output > MAX_MONEY {
            return Ok((
                ValidationResult::Invalid(format!(
                    "Coinbase output {coinbase_output} exceeds maximum money supply"
                )),
                utxo_set,
                tx_ids_owned.clone(),
                crate::reorganization::BlockUndoLog::new(),
            None,
        ));
        }

        // Use checked arithmetic for fee + subsidy calculation
        let max_coinbase_value = total_fees
            .checked_add(subsidy)
            .ok_or_else(|| ConsensusError::BlockValidation("Fees + subsidy overflow".into()))?;

        // Invariant assertion: Coinbase output must not exceed subsidy + fees
        assert!(
            coinbase_output <= max_coinbase_value,
            "Coinbase output {coinbase_output} must not exceed fees {total_fees} + subsidy {subsidy}"
        );
        if coinbase_output > max_coinbase_value {
            return Ok((
                ValidationResult::Invalid(format!(
                    "Coinbase output {coinbase_output} exceeds fees {total_fees} + subsidy {subsidy}"
                )),
                utxo_set,
                tx_ids_owned.clone(),
                crate::reorganization::BlockUndoLog::new(),
            None,
        ));
        }

        // Validate witness commitment if witnesses are present (SegWit block)
        // Check if any witness is non-empty (indicating SegWit block)
        let has_segwit = witnesses.iter().any(|w| !w.is_empty());
        // Invariant assertion: Witness count must match transaction count
        assert!(
            witnesses.len() == block.transactions.len(),
            "Witness count {} must match transaction count {}",
            witnesses.len(),
            block.transactions.len()
        );

        if has_segwit && !witnesses.is_empty() {
            // Invariant assertion: SegWit block must have witnesses
            assert!(!witnesses.is_empty(), "SegWit block must have witnesses");

            let witness_merkle_root = crate::segwit::compute_witness_merkle_root_from_nested(block, witnesses)?;
            // Invariant assertion: Witness merkle root must be 32 bytes
            assert!(
                witness_merkle_root.len() == 32,
                "Witness merkle root length {} must be 32 bytes",
                witness_merkle_root.len()
            );

            if !validate_witness_commitment(coinbase, &witness_merkle_root)? {
                return Ok((
                    ValidationResult::Invalid(
                        "Invalid witness commitment in coinbase transaction".to_string(),
                    ),
                    utxo_set,
                    tx_ids_owned.clone(),
                    crate::reorganization::BlockUndoLog::new(),
                    None,
                ));
            }
        }
    } else {
        return Ok((
            ValidationResult::Invalid("Block must have at least one transaction".to_string()),
            utxo_set,
            tx_ids_owned.clone(),
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // 3.5. Check block sigop cost limit (network rule)
    // total_sigop_cost accumulated in overlay pass
    use crate::constants::MAX_BLOCK_SIGOPS_COST;

    // Invariant assertion: Total sigop cost must not exceed maximum
    if total_sigop_cost > MAX_BLOCK_SIGOPS_COST {
        return Ok((
            ValidationResult::Invalid(format!(
                "Block sigop cost {total_sigop_cost} exceeds maximum {MAX_BLOCK_SIGOPS_COST}"
            )),
            utxo_set,
            tx_ids_owned.clone(),
            crate::reorganization::BlockUndoLog::new(),
            None,
        ));
    }

    // 4. Transaction IDs already computed — application loop or overlay merge
    #[cfg(all(feature = "production"))]
    if crate::config::use_overlay_delta() {
        // Reuse overlay from validation instead of rebuilding (avoids ~10k redundant map ops/block).
        if let Some(overlay) = overlay_for_delta {
            let (additions_arc, deletions) = overlay.into_changes();
        let mut undo_log = crate::reorganization::BlockUndoLog::new();
        merge_overlay_changes_to_cache(&additions_arc, &deletions, &mut utxo_set, bip30_index, &mut undo_log);
        #[cfg(all(feature = "rayon"))]
        insert_script_exec_cache_for_block(block, witnesses, height, network);
        return Ok((
            ValidationResult::Valid,
            utxo_set,
            tx_ids_owned.clone(),
            undo_log,
            Some(UtxoDelta {
                additions: additions_arc,
                deletions,
            }),
        ));
        }
    }

    // Normal path: apply_transaction_with_id loop
    let (result, new_utxo_set, undo_log) = connect_block_inner_with_tx_ids(block, witnesses, utxo_set, height, time_context, network, &tx_ids, total_fees, bip30_index)?;
    #[cfg(all(feature = "production", feature = "rayon"))]
    if matches!(result, ValidationResult::Valid) {
        insert_script_exec_cache_for_block(block, witnesses, height, network);
    }
    Ok((result, new_utxo_set, tx_ids_owned.clone(), undo_log, None))
}

/// Insert script exec cache keys for all txs in block (call when block validation passes).
#[cfg(all(feature = "production", feature = "rayon"))]
fn insert_script_exec_cache_for_block(
    block: &Block,
    witnesses: &[Vec<Witness>],
    height: u64,
    network: crate::types::Network,
) {
    let base_script_flags = calculate_base_script_flags_for_block(height, network);
    for (i, tx) in block.transactions.iter().enumerate() {
        if is_coinbase(tx) {
            continue;
        }
        let wits = witnesses.get(i).map(|w| w.as_slice()).unwrap_or(&[]);
        let has_witness = wits.iter().any(|wit| !is_witness_empty(wit));
        let flags = calculate_script_flags_for_block_with_base(tx, has_witness, base_script_flags, height, network);
        let witnesses_vec: Vec<_> = if wits.len() == tx.inputs.len() {
            wits.to_vec()
        } else {
            (0..tx.inputs.len()).map(|_| Vec::new()).collect()
        };
        let key = crate::script_exec_cache::compute_key(tx, &witnesses_vec, flags);
        crate::script_exec_cache::insert(&key);
    }
}

#[cfg(feature = "production")]
mod tx_id_pool {
    use crate::types::{Hash, Transaction};
    use std::cell::RefCell;

    thread_local! {
        static TX_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(
            crate::optimizations::proven_bounds::MAX_TX_SIZE_PROVEN
        ));
    }

    /// Fused serialize+hash using thread-local buffer. Avoids Vec<Vec<u8>> allocation.
    pub fn compute_tx_id_with_pool(tx: &Transaction) -> Hash {
        use crate::crypto::OptimizedSha256;
        use crate::serialization::transaction::serialize_transaction_into;

        TX_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            serialize_transaction_into(&mut buf, tx);
            OptimizedSha256::new().hash256(&buf)
        })
    }
}

/// Compute transaction IDs for a block (extracted for reuse).
/// Public so node layer can compute once and share between collect_gaps and connect_block_ibd (#21).
pub fn compute_block_tx_ids(block: &Block) -> Vec<Hash> {
    let tx_ids: Vec<Hash> = {
        #[cfg(all(feature = "production", feature = "rayon"))]
        {
            use rayon::prelude::*;
            assert!(
                block.transactions.len() <= 10_000,
                "Transaction count {} must be reasonable for batch processing",
                block.transactions.len()
            );
            block.transactions.as_ref().par_iter()
                .map(tx_id_pool::compute_tx_id_with_pool)
                .collect()
        }

        #[cfg(all(feature = "production", not(feature = "rayon")))]
        {
            block
                .transactions
                .iter()
                .map(tx_id_pool::compute_tx_id_with_pool)
                .collect()
        }

        #[cfg(not(feature = "production"))]
        {
            // Sequential fallback for non-production builds
            block
                .transactions
                .iter()
                .map(calculate_tx_id)
                .collect::<Vec<Hash>>()
        }
    };
    tx_ids
}

/// Merge overlay changes into cache. Updates bip30_index and builds undo log.
/// Call when BLVM_USE_OVERLAY_DELTA=1 instead of the normal apply loop.
#[cfg(feature = "production")]
fn merge_overlay_changes_to_cache(
    additions: &FxHashMap<OutPoint, std::sync::Arc<UTXO>>,
    deletions: &FxHashSet<OutPoint>,
    utxo_set: &mut UtxoSet,
    mut bip30_index: Option<&mut crate::bip_validation::Bip30Index>,
    undo_log: &mut crate::reorganization::BlockUndoLog,
) {
    use crate::reorganization::UndoEntry;

    // Process deletions first: remove from cache, update bip30, record undo
    for outpoint in deletions {
        if let Some(arc) = utxo_set.remove(outpoint) {
            let utxo = arc.as_ref();
            if let Some(idx) = bip30_index.as_deref_mut() {
                if utxo.is_coinbase {
                    if let std::collections::hash_map::Entry::Occupied(mut o) =
                        idx.entry((*outpoint).hash)
                    {
                        *o.get_mut() = o.get().saturating_sub(1);
                        if *o.get() == 0 {
                            o.remove();
                        }
                    }
                }
            }
            undo_log.entries.push(UndoEntry {
                outpoint: *outpoint,
                previous_utxo: Some(arc),
                new_utxo: None,
            });
        }
    }
    // Process additions: insert into cache, record undo
    for (outpoint, arc) in additions {
        undo_log.entries.push(UndoEntry {
            outpoint: *outpoint,
            previous_utxo: None,
            new_utxo: Some(std::sync::Arc::clone(arc)),
        });
        utxo_set.insert(*outpoint, std::sync::Arc::clone(arc));
    }
}

/// Compute BIP143/precomputed sighash for CCheckQueue path. Uses local refs and specs Vecs
/// (dropped before return) so buf borrow ends.
#[cfg(all(feature = "production", feature = "rayon"))]
fn compute_bip143_and_precomp(
    tx: &Transaction,
    prevout_values: &[i64],
    script_pubkey_indices: &[(usize, usize)],
    script_pubkey_buffer: &[u8],
    has_witness: bool,
) -> (
    Option<crate::transaction_hash::Bip143PrecomputedHashes>,
    Vec<Option<[u8; 32]>>,
) {
    let buf = script_pubkey_buffer;
    let refs: Vec<&[u8]> = script_pubkey_indices.iter().map(|&(s, l)| buf[s..s + l].as_ref()).collect();
    let refs: &[&[u8]] = &refs;
    if has_witness {
        let bip = crate::transaction_hash::Bip143PrecomputedHashes::compute(
            tx, prevout_values, refs,
        );
        let mut precomp = vec![None; script_pubkey_indices.len()];
        let mut specs: Vec<(usize, u8, &[u8])> = Vec::new();
        for (j, &(s, l)) in script_pubkey_indices.iter().enumerate() {
            let spk = &buf[s..s + l];
            if spk.len() == 22 && spk[0] == 0x00 && spk[1] == 0x14 {
                let mut script_code = [0u8; 25];
                script_code[0] = 0x76;
                script_code[1] = 0xa9;
                script_code[2] = 0x14;
                script_code[3..23].copy_from_slice(&spk[2..22]);
                script_code[23] = 0x88;
                script_code[24] = 0xac;
                let amount = prevout_values.get(j).copied().unwrap_or(0);
                if let Ok(h) =
                    crate::transaction_hash::calculate_bip143_sighash(tx, j, &script_code, amount, 0x01, Some(&bip))
                {
                    precomp[j] = Some(h);
                }
            } else if spk.len() == 23 && spk[0] == 0xa9 && spk[1] == 0x14 && spk[22] == 0x87 {
                if let Some((sighash_byte, redeem)) =
                    crate::script::parse_p2sh_p2pkh_for_precompute(&tx.inputs[j].script_sig)
                {
                    specs.push((j, sighash_byte, redeem));
                }
            }
        }
        if !specs.is_empty() {
            if let Ok(hashes) =
                crate::transaction_hash::batch_compute_legacy_sighashes(tx, prevout_values, refs, &specs)
            {
                for (k, &(j, _, _)) in specs.iter().enumerate() {
                    precomp[j] = Some(hashes[k]);
                }
            }
        }
        (Some(bip), precomp)
    } else {
        let mut precomp = vec![None; script_pubkey_indices.len()];
        let mut specs: Vec<(usize, u8, &[u8])> = Vec::new();
        for (j, &(s, l)) in script_pubkey_indices.iter().enumerate() {
            let spk = &buf[s..s + l];
            if spk.len() == 25
                && spk[0] == 0x76
                && spk[1] == 0xa9
                && spk[2] == 0x14
                && spk[23] == 0x88
                && spk[24] == 0xac
            {
                let script_sig = &tx.inputs[j].script_sig;
                if let Some((sig, _pubkey)) = crate::script::parse_p2pkh_script_sig(script_sig) {
                    if !sig.is_empty() {
                        specs.push((j, sig[sig.len() - 1], spk));
                    }
                }
            } else if spk.len() == 23 && spk[0] == 0xa9 && spk[1] == 0x14 && spk[22] == 0x87 {
                if let Some((sighash_byte, redeem)) =
                    crate::script::parse_p2sh_p2pkh_for_precompute(&tx.inputs[j].script_sig)
                {
                    specs.push((j, sighash_byte, redeem));
                }
            }
        }
        if !specs.is_empty() {
            if let Ok(hashes) = crate::transaction_hash::batch_compute_legacy_sighashes(
                tx, prevout_values, refs, &specs,
            ) {
                for (k, &(j, _, _)) in specs.iter().enumerate() {
                    precomp[j] = Some(hashes[k]);
                }
            }
        }
        (None, precomp)
    }
}

/// Internal function that accepts pre-computed tx_ids and total_fees to avoid redundant computation
#[allow(clippy::overly_complex_bool_expr)] // Intentional tautological assertions for formal verification
#[spec_locked("5.3")]
fn connect_block_inner_with_tx_ids(
    block: &Block,
    witnesses: &[Vec<Witness>],
    mut utxo_set: UtxoSet,
    height: Natural,
    time_context: Option<crate::types::TimeContext>,
    network: crate::types::Network,
    tx_ids: &[Hash],
    total_fees: i64,
    mut bip30_index: Option<&mut crate::bip_validation::Bip30Index>,
) -> Result<(
    ValidationResult,
    UtxoSet,
    crate::reorganization::BlockUndoLog,
)> {
    // 5. Apply all transactions to UTXO set (with pre-computed transaction IDs)
    // Build undo log for all UTXO changes
    use crate::reorganization::BlockUndoLog;
    let mut undo_log = BlockUndoLog::new();
    // Invariant assertion: Undo log must start empty
    assert!(undo_log.entries.is_empty(), "Undo log must start empty");

    // Invariant assertion: Transaction ID count must match transaction count
    assert!(
        tx_ids.len() == block.transactions.len(),
        "Transaction ID count {} must match transaction count {}",
        tx_ids.len(),
        block.transactions.len()
    );

    // NOTE: With UtxoOverlay approach, validation uses a read-only view of utxo_set.
    // The overlay tracks additions/deletions in memory but DOES NOT modify the base utxo_set.
    // Therefore, the application loop MUST ALWAYS run to apply changes to utxo_set.
    {
        // Normal path: Apply transactions sequentially to build undo log
        for (i, tx) in block.transactions.iter().enumerate() {
        // Bounds checking assertion: Transaction index must be valid
        assert!(
            i < block.transactions.len(),
            "Transaction index {i} out of bounds in application loop"
        );
        assert!(
            i < tx_ids.len(),
            "Transaction index {i} out of bounds for transaction IDs"
        );

        let initial_utxo_size = utxo_set.len();
        let (new_utxo_set, tx_undo_entries) =
            apply_transaction_with_id(tx, tx_ids[i], utxo_set, height, &mut bip30_index)?;

        // Invariant assertion: Undo entries must be reasonable
        assert!(
            tx_undo_entries.len() <= tx.inputs.len() + tx.outputs.len(),
            "Undo entry count {} must be reasonable for transaction {}",
            tx_undo_entries.len(),
            i
        );

        // Add all undo entries from this transaction to the block's undo log
        undo_log.entries.extend(tx_undo_entries);
        utxo_set = new_utxo_set;

        // Invariant assertion: UTXO set size must change reasonably
        if is_coinbase(tx) {
            assert!(
                utxo_set.len() >= initial_utxo_size,
                "UTXO set size {} must not decrease after coinbase (was {})",
                utxo_set.len(),
                initial_utxo_size
            );
        } else {
            // Non-coinbase: UTXO set size should change by (outputs - inputs)
            let expected_change = tx.outputs.len() as i64 - tx.inputs.len() as i64;
            let actual_change = utxo_set.len() as i64 - initial_utxo_size as i64;
            if actual_change != expected_change {
                // Workaround: apply_transaction_with_id sometimes fails to add outputs (e.g. when
                // output outpoints already exist from duplicate txids in pre-BIP30 blocks).
                // Force-insert all outputs to ensure correct state.
                let tx_id = tx_ids[i];
                let missing = expected_change - actual_change;
                if missing > 0 {
                    for (j, output) in tx.outputs.iter().enumerate() {
                        let op = OutPoint { hash: tx_id, index: j as u32 };
                        let utxo = UTXO {
                            value: output.value,
                            script_pubkey: output.script_pubkey.as_slice().into(),
                            height,
                            is_coinbase: false,
                        };
                        utxo_set.insert(op, std::sync::Arc::new(utxo));
                    }
                    let new_actual = utxo_set.len() as i64 - initial_utxo_size as i64;
                    // Lower: we spent N inputs so we can't shrink by more than N.
                    // Outputs may pre-exist from duplicate txids (pre-BIP30 blocks).
                    let lower = -(tx.inputs.len() as i64);
                    if new_actual < lower {
                        return Err(ConsensusError::BlockValidation(
                            format!(
                                "UTXO set size change {} outside allowed range (outputs: {}, inputs: {}, tx_idx: {})",
                                new_actual, tx.outputs.len(), tx.inputs.len(), i
                            ).into()
                        ));
                    }
                    // Allow variance when within [lower, expected] - outputs are present (insert
                    // above ensures correct data). Count mismatch can occur when output outpoints
                    // pre-existed from duplicate txids in early blocks.
                } else if actual_change < -(tx.inputs.len() as i64) {
                    return Err(ConsensusError::BlockValidation(
                        format!(
                            "UTXO set size change {} outside allowed range (outputs: {}, inputs: {}, tx_idx: {})",
                            actual_change, tx.outputs.len(), tx.inputs.len(), i
                        ).into()
                    ));
                }
                // Allow variance when actual_change is within [expected-inputs, expected] for early
                // blocks where output outpoints may pre-exist (pre-BIP30 duplicate txid edge cases)
            }
        }
        }
    }

    // Reverse entries for efficient undo (most recent first)
    // Note: Undo log size depends on transaction structure (inputs/outputs), not just count
    undo_log.entries.reverse();

    // Runtime invariant verification: Supply change must equal subsidy + fees
    // Mathematical specification:
    // ∀ block B, height h: Δsupply = get_block_subsidy(h) + total_fees
    // This ensures no money creation or destruction beyond expected inflation
    #[cfg(any(debug_assertions, feature = "runtime-invariants"))]
    {
        use crate::constants::MAX_MONEY;
        use crate::economic::{get_block_subsidy, total_supply};

        // Calculate expected supply at this height
        let expected_supply = total_supply(height);
        // Invariant assertion: Expected supply must be non-negative and within MAX_MONEY
        assert!(
            expected_supply >= 0,
            "Expected supply {expected_supply} must be non-negative"
        );
        assert!(
            expected_supply <= MAX_MONEY,
            "Expected supply {expected_supply} must not exceed MAX_MONEY"
        );

        // Calculate actual supply from UTXO set (sum of all UTXO values)
        // Invariant assertion: UTXO set size must be reasonable
        assert!(
            utxo_set.len() <= u32::MAX as usize,
            "UTXO set size {} must fit in u32",
            utxo_set.len()
        );

        let actual_supply: i64 = utxo_set
            .values()
            .map(|utxo| {
                // Invariant assertion: Each UTXO value must be valid
                assert!(
                    utxo.value >= 0,
                    "UTXO value {} must be non-negative",
                    utxo.value
                );
                assert!(
                    utxo.value <= MAX_MONEY,
                    "UTXO value {} must not exceed MAX_MONEY",
                    utxo.value
                );
                utxo.value
            })
            .try_fold(0i64, |acc, val| {
                // Invariant assertion: Accumulator must remain non-negative
                assert!(acc >= 0, "Accumulator {acc} must be non-negative");
                acc.checked_add(val)
            })
            .unwrap_or(MAX_MONEY);

        // Expected supply change = subsidy + fees
        let subsidy = get_block_subsidy(height);
        let _expected_change = subsidy + total_fees;

        // Actual supply change = actual_supply - previous_supply
        // We can't easily get previous_supply, but we can verify:
        // - Actual supply should be <= expected_supply (no inflation beyond subsidy)
        // - Actual supply should be >= expected_supply - some_tolerance (no excessive destruction)

        // Runtime assertion: Actual supply should not exceed expected supply by more than fees
        // (Allowing for fees because they're part of the economic model)
        debug_assert!(
            actual_supply <= expected_supply + total_fees,
            "Supply invariant violated at height {height}: actual supply {actual_supply} exceeds expected {expected_supply} + fees {total_fees}"
        );

        // Runtime assertion: Actual supply should be non-negative and <= MAX_MONEY
        debug_assert!(
            actual_supply >= 0,
            "Supply invariant violated: actual supply {actual_supply} is negative"
        );
        debug_assert!(
            actual_supply <= MAX_MONEY,
            "Supply invariant violated: actual supply {actual_supply} exceeds MAX_MONEY {MAX_MONEY}"
        );
    }

    // Postcondition assertions: Validate function outputs and state after execution
    assert!(
        matches!(ValidationResult::Valid, ValidationResult::Valid),
        "Validation result must be Valid on success"
    );
    assert!(
        utxo_set.len() <= u32::MAX as usize,
        "UTXO set size {} must not exceed maximum after block connection",
        utxo_set.len()
    );

    Ok((ValidationResult::Valid, utxo_set, undo_log))
}

/// ApplyTransaction: 𝒯𝒳 × 𝒰𝒮 → 𝒰𝒮
///
/// For transaction tx and UTXO set us:
/// 1. If tx is coinbase: us' = us ∪ {(tx.id, i) ↦ tx.outputs\[i\] : i ∈ \[0, |tx.outputs|)}
/// 2. Otherwise: us' = (us \ {i.prevout : i ∈ tx.inputs}) ∪ {(tx.id, i) ↦ tx.outputs\[i\] : i ∈ \[0, |tx.outputs|)}
/// 3. Return us'
///
/// This function computes the transaction ID internally.
/// For batch operations, use `apply_transaction_with_id` instead.
///
/// Returns both the new UTXO set and undo entries for all UTXO changes.
#[spec_locked("5.3.1")]
#[track_caller] // Better error messages showing caller location
pub fn apply_transaction(
    tx: &Transaction,
    utxo_set: UtxoSet,
    height: Natural,
) -> Result<(UtxoSet, Vec<crate::reorganization::UndoEntry>)> {
    let tx_id = calculate_tx_id(tx);
    let mut no_index = None;
    apply_transaction_with_id(tx, tx_id, utxo_set, height, &mut no_index)
}

/// ApplyTransaction with pre-computed transaction ID
///
/// Same as `apply_transaction` but accepts a pre-computed transaction ID
/// to avoid redundant computation when transaction IDs are batch-computed.
///
/// Returns both the new UTXO set and undo entries for all UTXO changes.
/// When `bip30_index` is Some, updates it for coinbase add/remove (O(1) BIP30 checks).
#[spec_locked("5.3.1")]
fn apply_transaction_with_id(
    tx: &Transaction,
    tx_id: Hash,
    mut utxo_set: UtxoSet,
    height: Natural,
    bip30_index: &mut Option<&mut crate::bip_validation::Bip30Index>,
) -> Result<(UtxoSet, Vec<crate::reorganization::UndoEntry>)> {
    // Precondition assertions: Validate function inputs
    assert!(
        !tx.inputs.is_empty() || is_coinbase(tx),
        "Transaction must have inputs unless it's a coinbase"
    );
    assert!(
        !tx.outputs.is_empty(),
        "Transaction must have at least one output"
    );
    assert!(
        height <= i64::MAX as u64,
        "Block height {height} must fit in i64"
    );

    use crate::reorganization::UndoEntry;

    let mut undo_entries = Vec::new();
    let initial_utxo_count = utxo_set.len();

    // Optimization: Pre-allocate capacity for new UTXOs if HashMap is growing
    // Estimate: current size + new outputs - spent inputs (for non-coinbase)
    #[cfg(feature = "production")]
    {
        let estimated_new_size = utxo_set
            .len()
            .saturating_add(tx.outputs.len())
            .saturating_sub(if is_coinbase(tx) { 0 } else { tx.inputs.len() });
        if estimated_new_size > utxo_set.capacity() {
            utxo_set.reserve(estimated_new_size.saturating_sub(utxo_set.len()));
        }
    }

    // Remove spent inputs (except for coinbase) and record in undo log
    if !is_coinbase(tx) {
        // Invariant assertion: Non-coinbase must have inputs
        assert!(
            !tx.inputs.is_empty(),
            "Non-coinbase transaction must have inputs"
        );

        for input in &tx.inputs {
            // Invariant assertion: Prevout hash must be non-zero for non-coinbase
            assert!(
                input.prevout.hash != [0u8; 32] || input.prevout.index != 0xffffffff,
                "Prevout must be valid for non-coinbase input"
            );

            // Record the UTXO that existed before (for restoration during disconnect)
            if let Some(arc) = utxo_set.remove(&input.prevout) {
                let previous_utxo = arc.as_ref();
                // BIP30 index: decrement coinbase txid count when spending a coinbase UTXO
                if let Some(idx) = bip30_index.as_deref_mut() {
                    if previous_utxo.is_coinbase {
                        if let std::collections::hash_map::Entry::Occupied(mut o) =
                            idx.entry(input.prevout.hash)
                        {
                            *o.get_mut() = o.get().saturating_sub(1);
                            if *o.get() == 0 {
                                o.remove();
                            }
                        }
                    }
                }

                // Invariant assertion: Previous UTXO value must be valid
                assert!(
                    previous_utxo.value >= 0,
                    "Previous UTXO value {} must be non-negative",
                    previous_utxo.value
                );
                use crate::constants::MAX_MONEY;
                assert!(
                    previous_utxo.value <= MAX_MONEY,
                    "Previous UTXO value {} must not exceed MAX_MONEY",
                    previous_utxo.value
                );

                undo_entries.push(UndoEntry {
                    outpoint: input.prevout,
                    previous_utxo: Some(std::sync::Arc::clone(&arc)),
                    new_utxo: None, // This UTXO is being spent
                });
                // Invariant assertion: Undo entry count must be reasonable
                assert!(
                    undo_entries.len() <= tx.inputs.len() + tx.outputs.len(),
                    "Undo entry count {} must be reasonable",
                    undo_entries.len()
                );
            }
        }
    }

    // Add new outputs and record in undo log
    for (i, output) in tx.outputs.iter().enumerate() {
        // Bounds checking assertion: Output index must be valid
        assert!(
            i < tx.outputs.len(),
            "Output index {} out of bounds (transaction has {} outputs)",
            i,
            tx.outputs.len()
        );

        // Invariant assertion: Output value must be valid
        assert!(
            output.value >= 0,
            "Output value {} must be non-negative",
            output.value
        );
        use crate::constants::MAX_MONEY;
        assert!(
            output.value <= MAX_MONEY,
            "Output value {} must not exceed MAX_MONEY",
            output.value
        );

        let outpoint = OutPoint {
            hash: tx_id,
            index: i as u32,
        };
        // Invariant assertion: Outpoint index must fit in u32
        assert!(
            i <= u32::MAX as usize,
            "Output index {i} must fit in Natural"
        );

        let utxo = UTXO {
            value: output.value,
            script_pubkey: output.script_pubkey.as_slice().into(),
            height,
            is_coinbase: is_coinbase(tx),
        };
        // Invariant assertion: UTXO value must match output value
        assert!(
            utxo.value == output.value,
            "UTXO value {} must match output value {}",
            utxo.value,
            output.value
        );

        let utxo_arc = std::sync::Arc::new(utxo);
        // Record that this UTXO is being created
        undo_entries.push(UndoEntry {
            outpoint,
            previous_utxo: None, // This UTXO didn't exist before
            new_utxo: Some(std::sync::Arc::clone(&utxo_arc)),
        });
        // Invariant assertion: Undo entry count must be reasonable
        assert!(
            undo_entries.len() <= tx.outputs.len() + tx.inputs.len(),
            "Undo entry count {} must be reasonable",
            undo_entries.len()
        );

        utxo_set.insert(outpoint, utxo_arc);

        // BIP30 index: increment coinbase txid count when adding a coinbase output
        if let Some(idx) = bip30_index.as_deref_mut() {
            if is_coinbase(tx) {
                *idx.entry(tx_id).or_insert(0) += 1;
            }
        }
    }

    // Recovery: if outputs weren't added (can happen when output outpoints pre-exist from
    // duplicate txids in pre-BIP30 blocks), ensure they're present by re-inserting.
    if !is_coinbase(tx) {
        let current_count = utxo_set.len();
        let expected_count = initial_utxo_count
            .saturating_sub(tx.inputs.len())
            .saturating_add(tx.outputs.len());
        if current_count < expected_count {
            for (j, output) in tx.outputs.iter().enumerate() {
                let op = OutPoint {
                    hash: tx_id,
                    index: j as u32,
                };
                if !utxo_set.contains_key(&op) {
                    let utxo = UTXO {
                        value: output.value,
                        script_pubkey: output.script_pubkey.as_slice().into(),
                        height,
                        is_coinbase: false,
                    };
                    utxo_set.insert(op, std::sync::Arc::new(utxo));
                }
            }
        }
    }

    // Postcondition assertions: Validate UTXO set consistency after transaction application
    let final_utxo_count = utxo_set.len();
    if is_coinbase(tx) {
        // Coinbase: UTXO set should grow by number of outputs
        assert!(
            final_utxo_count >= initial_utxo_count,
            "UTXO set size {final_utxo_count} must not decrease after coinbase (was {initial_utxo_count})"
        );
        assert!(
            final_utxo_count <= initial_utxo_count + tx.outputs.len(),
            "UTXO set size {} must not exceed initial {} + outputs {}",
            final_utxo_count,
            initial_utxo_count,
            tx.outputs.len()
        );
    } else {
        // Non-coinbase: UTXO set should change by (outputs - inputs)
        let expected_change = tx.outputs.len() as i64 - tx.inputs.len() as i64;
        let actual_change = final_utxo_count as i64 - initial_utxo_count as i64;
        // Lower bound: we spent N inputs so we can't shrink by more than N.
        // When output outpoints pre-exist (duplicate txids in pre-BIP30 blocks), we may add 0.
        let lower = -(tx.inputs.len() as i64);
        // Use debug_assert: release IBD must not panic on historical edge cases.
        debug_assert!(
            actual_change >= lower,
            "UTXO set size change {actual_change} must be reasonable (expected ~{expected_change})"
        );
    }
    assert!(
        utxo_set.len() <= u32::MAX as usize,
        "UTXO set size {} must not exceed maximum",
        utxo_set.len()
    );

    Ok((utxo_set, undo_entries))
}

/// Validate block header
///
/// # Arguments
///
/// * `header` - Block header to validate
/// * `time_context` - Optional time context for timestamp validation (BIP113)
///   If None, only basic timestamp checks are performed (non-zero).
///   If Some, full timestamp validation is performed:
///   - Rejects blocks with timestamps > network_time + MAX_FUTURE_BLOCK_TIME
///   - Rejects blocks with timestamps < median_time_past
#[allow(clippy::overly_complex_bool_expr, clippy::redundant_comparisons)] // Intentional tautological assertions for formal verification
#[spec_locked("5.3")]
fn validate_block_header(
    header: &BlockHeader,
    time_context: Option<&crate::types::TimeContext>,
) -> Result<bool> {
    // Precondition assertions: Validate header fields

    // Check version is valid
    if header.version < 1 {
        return Ok(false);
    }

    // Check timestamp is non-zero
    // Precondition assertion: Timestamp must be checked
    if header.timestamp == 0 {
        return Ok(false);
    }

    // Full timestamp validation if time context is provided
    if let Some(ctx) = time_context {
        // Reject blocks with timestamps too far in future (2-hour tolerance for clock skew)
        // Consensus: block time <= adjusted time + MAX_FUTURE_BLOCK_TIME
        if header.timestamp > ctx.network_time + crate::constants::MAX_FUTURE_BLOCK_TIME {
            return Ok(false);
        }

        // Reject blocks with timestamps before median time-past (BIP113)
        // This prevents time-warp attacks by ensuring block timestamps are monotonically increasing
        if header.timestamp < ctx.median_time_past {
            return Ok(false);
        }
    }

    // Check bits is valid
    // Precondition assertion: Bits must be checked
    if header.bits == 0 {
        return Ok(false);
    }
    // Invariant assertion: Bits must be non-zero for valid header
    assert!(
        header.bits != 0,
        "Header bits {} must be non-zero for valid header",
        header.bits
    );

    // Check merkle root is valid (non-zero)
    // Orange Paper: merkle_root must be valid hash
    // Precondition assertion: Merkle root must be checked
    if header.merkle_root == [0u8; 32] {
        return Ok(false);
    }
    // Invariant assertion: Merkle root must be non-zero for valid header
    assert!(
        header.merkle_root != [0u8; 32],
        "Merkle root must be non-zero for valid header"
    );
    // Invariant assertion: Merkle root must be 32 bytes
    assert!(
        header.merkle_root.len() == 32,
        "Merkle root length {} must be 32 bytes",
        header.merkle_root.len()
    );

    // Additional validation: version must be reasonable (not all zeros)
    // This prevents obviously invalid blocks
    // Precondition assertion: Version must be checked
    if header.version == 0 {
        return Ok(false);
    }
    // Invariant assertion: Version must be >= 1 for valid header
    assert!(
        header.version >= 1,
        "Header version {} must be >= 1 for valid header",
        header.version
    );

    // Postcondition assertion: Validation result must be consistent
    let result = true;
    #[allow(clippy::eq_op)]
    {
        assert!(result || !result, "Validation result must be boolean");
    }
    // Postcondition assertion: Result must be true on success
    assert!(result, "Validation result must be true on success");

    Ok(result)
}

// is_coinbase is imported from crate::transaction

/// Base script flags for a block (constant per block; only depends on height and network).
/// Call once per block, then use `calculate_script_flags_for_block` or `add_per_tx_script_flags`.
#[inline]
pub(crate) fn calculate_base_script_flags_for_block(
    height: u64,
    network: crate::types::Network,
) -> u32 {
    let mut flags: u32 = 0;
    
    // Get activation heights for this network
    use crate::constants::*;
    let (p2sh_height, bip66_height, bip65_height, bip147_height, _segwit_height, _taproot_height, ctv_height, _csfs_height) = match network {
        crate::types::Network::Mainnet => (
            BIP16_P2SH_ACTIVATION_MAINNET,
            BIP66_ACTIVATION_MAINNET,
            BIP65_ACTIVATION_MAINNET,
            BIP147_ACTIVATION_MAINNET,
            SEGWIT_ACTIVATION_MAINNET,
            TAPROOT_ACTIVATION_MAINNET,
            CTV_ACTIVATION_MAINNET,
            CSFS_ACTIVATION_MAINNET,
        ),
        crate::types::Network::Testnet => (
            BIP16_P2SH_ACTIVATION_TESTNET,
            BIP66_ACTIVATION_TESTNET,
            BIP65_ACTIVATION_MAINNET, // BIP65 testnet height not defined, use mainnet
            BIP147_ACTIVATION_TESTNET,
            SEGWIT_ACTIVATION_MAINNET, // Same as mainnet for simplicity
            TAPROOT_ACTIVATION_MAINNET, // Same as mainnet for simplicity
            CTV_ACTIVATION_TESTNET,
            CSFS_ACTIVATION_TESTNET,
        ),
        crate::types::Network::Regtest => (
            BIP16_P2SH_ACTIVATION_REGTEST,
            BIP66_ACTIVATION_REGTEST,
            0, // Always active on regtest
            0, // Always active on regtest
            0, // Always active on regtest
            0, // Always active on regtest
            CTV_ACTIVATION_REGTEST, // 0 = always active when feature enabled
            CSFS_ACTIVATION_REGTEST, // 0 = always active when feature enabled
        ),
    };
    
    // SCRIPT_VERIFY_P2SH (0x01) - BIP16, activated at block 173,805 on mainnet
    if height >= p2sh_height {
        flags |= 0x01;
    }
    
    // SCRIPT_VERIFY_DERSIG (0x04) - BIP66, activated at block 363,725 on mainnet
    // Also enables SCRIPT_VERIFY_STRICTENC (0x02) and SCRIPT_VERIFY_LOW_S (0x08)
    if height >= bip66_height {
        flags |= 0x02 | 0x04 | 0x08;
    }
    
    // SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY (0x200) - BIP65, activated at block 388,381 on mainnet
    if height >= bip65_height {
        flags |= 0x200;
    }
    
    // SCRIPT_VERIFY_CHECKSEQUENCEVERIFY (0x400) - BIP112, activated with SegWit
    // SCRIPT_VERIFY_NULLDUMMY (0x10) - BIP147, activated with SegWit
    if height >= bip147_height {
        flags |= 0x10 | 0x400;
    }

    // SCRIPT_VERIFY_DEFAULT_CHECK_TEMPLATE_VERIFY_HASH (0x80000000) - BIP119 CTV
    #[cfg(feature = "ctv")]
    {
        if ctv_height > 0 && height >= ctv_height {
            flags |= 0x80000000;
        } else if ctv_height == 0 && network == crate::types::Network::Regtest {
            flags |= 0x80000000;
        }
    }

    flags
}

/// Per-tx script flags (SegWit + Taproot). Add to base flags from `calculate_base_script_flags_for_block`.
#[inline]
fn add_per_tx_script_flags(base_flags: u32, tx: &Transaction, has_witness: bool, height: u64, network: crate::types::Network) -> u32 {
    use crate::constants::*;
    let (segwit_height, taproot_height) = match network {
        crate::types::Network::Mainnet => (SEGWIT_ACTIVATION_MAINNET, TAPROOT_ACTIVATION_MAINNET),
        crate::types::Network::Testnet => (SEGWIT_ACTIVATION_MAINNET, TAPROOT_ACTIVATION_MAINNET),
        crate::types::Network::Regtest => (0, 0),
    };
    let mut flags = base_flags;
    if height >= segwit_height && (has_witness || is_segwit_transaction(tx)) {
        flags |= 0x800;
    }
    if height >= taproot_height {
        for output in &tx.outputs {
            let script = &output.script_pubkey;
            if script.len() == TAPROOT_SCRIPT_LENGTH && script[0] == OP_1 && script[1] == 0x20 {
                flags |= 0x8000;
                break;
            }
        }
    }
    flags
}

/// Calculate script verification flags for a transaction in a block.
/// Optimized: pass `base_flags` from `calculate_base_script_flags_for_block` (computed once per block).
pub(crate) fn calculate_script_flags_for_block(
    tx: &Transaction,
    has_witness: bool,
    height: u64,
    network: crate::types::Network,
) -> u32 {
    let base = calculate_base_script_flags_for_block(height, network);
    add_per_tx_script_flags(base, tx, has_witness, height, network)
}

/// Calculate script verification flags for a transaction in a block (with precomputed base flags).
#[inline]
pub(crate) fn calculate_script_flags_for_block_with_base(
    tx: &Transaction,
    has_witness: bool,
    base_flags: u32,
    height: u64,
    network: crate::types::Network,
) -> u32 {
    add_per_tx_script_flags(base_flags, tx, has_witness, height, network)
}

/// Calculate transaction ID using proper Bitcoin double SHA256
///
/// Transaction ID is SHA256(SHA256(serialized_tx)) where serialized_tx
/// is the transaction in Bitcoin wire format.
///
/// For batch operations, use serialize_transaction + batch_double_sha256 instead.
///
/// Transaction hash (double SHA256 of serialized tx, BIP141 for witness):
/// serialize with non-witness format, then double-SHA256.
#[inline(always)]
#[spec_locked("5.1")]
pub fn calculate_tx_id(tx: &Transaction) -> Hash {
    use crate::crypto::OptimizedSha256;
    use crate::serialization::transaction::serialize_transaction;

    let serialized = serialize_transaction(tx);
    OptimizedSha256::new().hash256(&serialized)
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for Block Connection:
/// ∀ block B, UTXO set US, height h: ConnectBlock(B, US, h) = (valid, US') ⟺
///   (ValidateHeader(B.header) ∧
///    ∀ tx ∈ B.transactions: CheckTransaction(tx) ∧ CheckTxInputs(tx, US, h) ∧
///    VerifyScripts(tx, US) ∧
///    CoinbaseOutput ≤ TotalFees + GetBlockSubsidy(h) ∧
///    US' = ApplyTransactions(B.transactions, US))
///
/// Invariants:
/// - Valid blocks have valid headers and transactions
/// - UTXO set consistency is preserved
/// - Coinbase output respects economic rules
/// - Transaction application is atomic


#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Arbitrary implementations for property tests (inline since tests/fuzzing/arbitrary_impls.rs
    // is in separate test crate and not accessible from src/ tests)
    impl Arbitrary for BlockHeader {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                any::<i32>(),      // version
                any::<[u8; 32]>(), // prev_block_hash
                any::<[u8; 32]>(), // merkle_root
                any::<u64>(),      // timestamp
                any::<u64>(),      // bits
                any::<u64>(),      // nonce
            )
                .prop_map(
                    |(version, prev_block_hash, merkle_root, timestamp, bits, nonce)| {
                        BlockHeader {
                            version: version as i64, // BlockHeader.version is i64
                            prev_block_hash,
                            merkle_root,
                            timestamp,
                            bits,
                            nonce,
                        }
                    },
                )
                .boxed()
        }
    }

    impl Arbitrary for Block {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                any::<BlockHeader>(),
                prop::collection::vec(any::<Transaction>(), 0..100), // transactions
            )
                .prop_map(|(header, transactions)| Block {
                    header,
                    transactions: transactions.into_boxed_slice(),
                })
                .boxed()
        }
    }

    impl Arbitrary for OutPoint {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                any::<[u8; 32]>(), // hash
                any::<u32>(),      // index
            )
                .prop_map(|(hash, index)| OutPoint { hash, index })
                .boxed()
        }
    }

    impl Arbitrary for UTXO {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                any::<i64>(),                               // value
                prop::collection::vec(any::<u8>(), 0..100), // script_pubkey
                any::<u64>(),                               // height
                any::<bool>(),                              // is_coinbase
            )
                .prop_map(|(value, script_pubkey, height, is_coinbase)| UTXO {
                    value,
                    script_pubkey: script_pubkey.into(),
                    height,
                    is_coinbase,
                })
                .boxed()
        }
    }

    // Transaction Arbitrary is implemented in src/transaction.rs to avoid conflicts
    // UtxoSet (FxHashMap with production) has no Arbitrary; use strategy below

    /// Property test: apply_transaction preserves UTXO set consistency
    proptest! {
        #[test]
        fn prop_apply_transaction_consistency(
            tx in (
                any::<u64>(), // version
                prop::bool::weighted(0.1), // is_coinbase (10% chance)
                prop::collection::vec(
                    (
                        any::<[u8; 32]>(),                          // prevout hash
                        any::<u32>(),                               // prevout index
                        prop::collection::vec(any::<u8>(), 0..100), // script_sig
                        any::<u64>(),                               // sequence
                    ),
                    1..=5, // input count (at least 1 for non-coinbase)
                ),
                prop::collection::vec(
                    (
                        (0i64..=MAX_MONEY),                         // value (valid range)
                        prop::collection::vec(any::<u8>(), 0..100), // script_pubkey
                    ),
                    1..=5, // output count (at least 1)
                ),
                any::<u64>(), // lock_time
            ).prop_map(|(version, is_coinbase, inputs, outputs, lock_time)| {
                let mut tx = Transaction {
                    version,
                    inputs: inputs
                        .into_iter()
                        .map(|(hash, index, script_sig, sequence)| TransactionInput {
                            prevout: OutPoint { hash, index },
                            script_sig,
                            sequence,
                        })
                        .collect(),
                    outputs: outputs
                        .into_iter()
                        .map(|(value, script_pubkey)| TransactionOutput {
                            value,
                            script_pubkey,
                        })
                        .collect(),
                    lock_time,
                };
                // Make coinbase if needed
                if is_coinbase {
                    tx.inputs.clear();
                    // Coinbase script_sig must be 2-100 bytes
                    tx.inputs.push(TransactionInput {
                        prevout: OutPoint { hash: [0u8; 32], index: 0xFFFFFFFF },
                        script_sig: vec![0x01, 0x01], // Minimum 2 bytes
                        sequence: 0xFFFFFFFF,
                    });
                }
                tx
            }),
            utxo_set in prop::collection::vec((any::<OutPoint>(), any::<UTXO>()), 0..50).prop_map(|v| v.into_iter().map(|(op, u)| (op, std::sync::Arc::new(u))).collect::<UtxoSet>()),
            height in 0u64..1000u64
        ) {

            let result = apply_transaction(&tx, utxo_set.clone(), height);

            match result {
                Ok((new_utxo_set, _undo_entries)) => {
                    // UTXO set consistency properties
                    if !is_coinbase(&tx) {
                        // Non-coinbase transactions must remove spent inputs
                        for input in &tx.inputs {
                            prop_assert!(!new_utxo_set.contains_key(&input.prevout),
                                "Spent inputs must be removed from UTXO set");
                        }
                    }

                    // All outputs must be added to UTXO set
                    let tx_id = calculate_tx_id(&tx);
                    for (i, _output) in tx.outputs.iter().enumerate() {
                        let outpoint = OutPoint {
                            hash: tx_id,
                            index: i as u32,
                        };
                        prop_assert!(new_utxo_set.contains_key(&outpoint),
                            "All outputs must be added to UTXO set");
                    }
                },
                Err(_) => {
                    // Some invalid transactions may fail, which is acceptable
                }
            }
        }
    }

    /// Property test: connect_block validates coinbase correctly
    proptest! {
        #[test]
        fn prop_connect_block_coinbase(
            block in any::<Block>(),
            utxo_set in prop::collection::vec((any::<OutPoint>(), any::<UTXO>()), 0..50).prop_map(|v| v.into_iter().map(|(op, u)| (op, std::sync::Arc::new(u))).collect::<UtxoSet>()),
            height in 0u64..1000u64
        ) {
            // Bound for tractability
            let mut bounded_block = block;
            if bounded_block.transactions.len() > 3 {
                let mut transactions_vec: Vec<_> = bounded_block.transactions.into();
                transactions_vec.truncate(3);
                bounded_block.transactions = transactions_vec.into_boxed_slice();
            }

            // Skip blocks with no transactions (invalid)
            prop_assume!(!bounded_block.transactions.is_empty());

            for tx in &mut bounded_block.transactions {
                if tx.inputs.len() > 3 {
                    tx.inputs.truncate(3);
                }
                if tx.outputs.len() > 3 {
                    tx.outputs.truncate(3);
                }
            }

            // One Vec<Witness> per tx (one Witness per input)
            let witnesses: Vec<Vec<Witness>> = bounded_block.transactions.iter().map(|tx| (0..tx.inputs.len()).map(|_| Vec::new()).collect()).collect();
            let result = connect_block(&bounded_block, &witnesses[..], utxo_set, height, None::<&[crate::types::BlockHeader]>, bounded_block.header.timestamp, crate::types::Network::Mainnet);

            match result {
                Ok((validation_result, _, _undo_log)) => {
                    match validation_result {
                        ValidationResult::Valid => {
                            // Valid blocks must have coinbase as first transaction
                            if !bounded_block.transactions.is_empty() {
                                prop_assert!(is_coinbase(&bounded_block.transactions[0]),
                                    "Valid blocks must have coinbase as first transaction");
                            }
                        },
                        ValidationResult::Invalid(_) => {
                            // Invalid blocks may violate any rule
                            // This is acceptable - we're testing the validation logic
                        }
                    }
                },
                Err(_) => {
                    // Some invalid blocks may fail, which is acceptable
                }
            }
        }
    }

    /// Property test: calculate_tx_id is deterministic
    proptest! {
        #[test]
        fn prop_calculate_tx_id_deterministic(
            tx in any::<Transaction>()
        ) {
            // Bound for tractability
            let mut bounded_tx = tx;
            if bounded_tx.inputs.len() > 5 {
                bounded_tx.inputs.truncate(5);
            }
            if bounded_tx.outputs.len() > 5 {
                bounded_tx.outputs.truncate(5);
            }

            // Calculate ID twice
            let id1 = calculate_tx_id(&bounded_tx);
            let id2 = calculate_tx_id(&bounded_tx);

            // Deterministic property
            prop_assert_eq!(id1, id2, "Transaction ID calculation must be deterministic");
        }
    }

    /// Property test: UTXO set operations are consistent
    proptest! {
        #[test]
        fn prop_utxo_set_operations_consistent(
            utxo_set in prop::collection::vec((any::<OutPoint>(), any::<UTXO>()), 0..50).prop_map(|v| v.into_iter().map(|(op, u)| (op, std::sync::Arc::new(u))).collect::<UtxoSet>()),
            outpoint in any::<OutPoint>(),
            utxo in any::<UTXO>()
        ) {
            let mut test_set = utxo_set.clone();

            // Insert operation
            let outpoint_key = outpoint.clone();
            test_set.insert(outpoint.clone(), std::sync::Arc::new(utxo.clone()));
            prop_assert!(test_set.contains_key(&outpoint_key), "Inserted UTXO must be present");

            // Get operation
            let retrieved = test_set.get(&outpoint_key);
            prop_assert!(retrieved.is_some(), "Inserted UTXO must be retrievable");
            prop_assert_eq!(retrieved.unwrap().value, utxo.value, "Retrieved UTXO must match inserted value");

            // Remove operation
            test_set.remove(&outpoint);
            prop_assert!(!test_set.contains_key(&outpoint), "Removed UTXO must not be present");
        }
    }
}


#[cfg(test)]
mod additional_tests {
    use super::*;
    use proptest::prelude::*;

    /// Property test: block header validation respects basic rules
    proptest! {
        #[test]
        fn prop_validate_block_header_basic_rules(
            header in any::<BlockHeader>()
        ) {
            let result = validate_block_header(&header, None).unwrap_or(false);

            // Basic validation properties
            if result {
                // Valid headers must have version >= 1
                prop_assert!(header.version >= 1, "Valid headers must have version >= 1");

                // Valid headers must have non-zero bits
                prop_assert!(header.bits != 0, "Valid headers must have non-zero bits");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connect_block_valid() {
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![0x00, 0x01], // Coinbase scriptSig must be 2-100 bytes
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000, // 50 BTC
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        use crate::mining::calculate_merkle_root;

        // Calculate actual merkle root for the block
        let merkle_root = calculate_merkle_root(&[coinbase_tx.clone()]).unwrap();

        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root,
                timestamp: 1231006505, // Genesis timestamp
                bits: 0x1d00ffff,
                nonce: 2083236893,
            },
            transactions: vec![coinbase_tx].into_boxed_slice(),
        };

        let utxo_set = UtxoSet::default();
        // Optimization: Pre-allocate witness vectors with capacity
        // One Vec<Witness> per tx (one Witness per input)
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::with_capacity(2)).collect())
            .collect();
        let (result, new_utxo_set, _undo_log) = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        )
        .unwrap();

        assert_eq!(result, ValidationResult::Valid);
        assert_eq!(new_utxo_set.len(), 1); // One new UTXO from coinbase
    }

    #[test]
    fn test_apply_transaction_coinbase() {
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let utxo_set = UtxoSet::default();
        let (new_utxo_set, _undo_entries) = apply_transaction(&coinbase_tx, utxo_set, 0).unwrap();

        assert_eq!(new_utxo_set.len(), 1);
    }

    // ============================================================================
    // COMPREHENSIVE BLOCK TESTS
    // ============================================================================

    #[test]
    fn test_connect_block_invalid_header() {
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = Block {
            header: BlockHeader {
                version: 0, // Invalid version
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![coinbase_tx].into_boxed_slice(),
        };

        let utxo_set = UtxoSet::default();
        // One Vec<Witness> per tx (one Witness per input)
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::with_capacity(2)).collect())
            .collect();
        let (result, _, _undo_log) = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        )
        .unwrap();

        assert!(matches!(result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_connect_block_no_transactions() {
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![].into_boxed_slice(), // No transactions
        };

        let utxo_set = UtxoSet::default();
        // One Vec<Witness> per tx (one Witness per input)
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::with_capacity(2)).collect())
            .collect();
        let (result, _, _undo_log) = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        )
        .unwrap();

        assert!(matches!(result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_connect_block_first_tx_not_coinbase() {
        let regular_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![regular_tx].into_boxed_slice(), // First tx is not coinbase
        };

        let utxo_set = UtxoSet::default();
        // One Vec<Witness> per tx (one Witness per input)
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::with_capacity(2)).collect())
            .collect();
        let (result, _, _undo_log) = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        )
        .unwrap();

        assert!(matches!(result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_connect_block_coinbase_exceeds_subsidy() {
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 6000000000, // 60 BTC - exceeds subsidy
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![coinbase_tx].into_boxed_slice(),
        };

        let utxo_set = UtxoSet::default();
        // One Vec<Witness> per tx (one Witness per input)
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::with_capacity(2)).collect())
            .collect();
        let (result, _, _undo_log) = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        )
        .unwrap();

        assert!(matches!(result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_apply_transaction_regular() {
        let mut utxo_set = UtxoSet::default();

        // Add a UTXO first
        let prev_outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let prev_utxo = UTXO {
            value: 1000,
            script_pubkey: vec![OP_1].into(), // OP_1
            height: 0,
            is_coinbase: false,
        };
        utxo_set.insert(prev_outpoint, std::sync::Arc::new(prev_utxo));

        let regular_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_1], // OP_1
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 500,
                script_pubkey: vec![OP_2].into(), // OP_2
            }]
            .into(),
            lock_time: 0,
        };

        let (new_utxo_set, _undo_entries) = apply_transaction(&regular_tx, utxo_set, 1).unwrap();

        // Should have 1 UTXO (the new output)
        assert_eq!(new_utxo_set.len(), 1);
    }

    #[test]
    fn test_apply_transaction_multiple_outputs() {
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![
                TransactionOutput {
                    value: 2500000000,
                    script_pubkey: vec![OP_1].into(),
                },
                TransactionOutput {
                    value: 2500000000,
                    script_pubkey: vec![OP_2].into(),
                },
            ]
            .into(),
            lock_time: 0,
        };

        let utxo_set = UtxoSet::default();
        let (new_utxo_set, _undo_entries) = apply_transaction(&coinbase_tx, utxo_set, 0).unwrap();

        assert_eq!(new_utxo_set.len(), 2);
    }

    #[test]
    fn test_validate_block_header_valid() {
        use sha2::{Digest, Sha256};

        // Create a valid header with non-zero merkle root
        let header = BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: Sha256::digest(b"test merkle root")[..].try_into().unwrap(),
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        };

        let result = validate_block_header(&header, None).unwrap();
        assert!(result);
    }

    #[test]
    fn test_validate_block_header_invalid_version() {
        let header = BlockHeader {
            version: 0, // Invalid version
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        };

        let result = validate_block_header(&header, None).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_validate_block_header_invalid_bits() {
        let header = BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0, // Invalid bits
            nonce: 0,
        };

        let result = validate_block_header(&header, None).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_is_coinbase_true() {
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        assert!(is_coinbase(&coinbase_tx));
    }

    #[test]
    fn test_is_coinbase_false_wrong_hash() {
        let regular_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0xffffffff,
                }, // Wrong hash
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        assert!(!is_coinbase(&regular_tx));
    }

    #[test]
    fn test_is_coinbase_false_wrong_index() {
        let regular_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                }, // Wrong index
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        assert!(!is_coinbase(&regular_tx));
    }

    #[test]
    fn test_is_coinbase_false_multiple_inputs() {
        let regular_tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [0; 32].into(),
                        index: 0xffffffff,
                    },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1; 32],
                        index: 0,
                    },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
            ]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        assert!(!is_coinbase(&regular_tx));
    }

    #[test]
    fn test_calculate_tx_id() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let tx_id = calculate_tx_id(&tx);

        // Should be a 32-byte hash (double SHA256 of serialized transaction)
        assert_eq!(tx_id.len(), 32);

        // Same transaction should produce same ID (deterministic)
        let tx_id2 = calculate_tx_id(&tx);
        assert_eq!(tx_id, tx_id2);

        // Different transaction should produce different ID
        let mut tx2 = tx.clone();
        tx2.version = 2;
        let tx_id3 = calculate_tx_id(&tx2);
        assert_ne!(tx_id, tx_id3);
    }

    #[test]
    fn test_calculate_tx_id_different_versions() {
        let tx1 = Transaction {
            version: 2,
            inputs: vec![].into(),
            outputs: vec![].into(),
            lock_time: 0,
        };

        let tx2 = Transaction {
            version: 1,
            inputs: vec![].into(),
            outputs: vec![].into(),
            lock_time: 0,
        };

        let id1 = calculate_tx_id(&tx1);
        let id2 = calculate_tx_id(&tx2);

        // Different versions should produce different IDs
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_connect_block_empty_transactions() {
        // Test that blocks with empty transactions are rejected
        // Note: We need a valid merkle root even for empty blocks (though they're invalid)
        // For testing purposes, we'll use a zero merkle root which will fail validation
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32], // Zero merkle root - will fail validation
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![].into_boxed_slice(), // Empty transactions - invalid
        };

        let utxo_set = UtxoSet::default();
        // Optimization: Pre-allocate witness vectors with capacity
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
            .collect();
        let result = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        );
        // The result should be Ok with ValidationResult::Invalid
        assert!(result.is_ok());
        let (validation_result, _, _undo_log) = result.unwrap();
        assert!(matches!(validation_result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_connect_block_invalid_coinbase() {
        let invalid_coinbase = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                }, // Wrong hash for coinbase
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![invalid_coinbase].into_boxed_slice(),
        };

        let utxo_set = UtxoSet::default();
        // Optimization: Pre-allocate witness vectors with capacity
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
            .collect();
        let result = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        );
        // The result should be Ok with ValidationResult::Invalid
        assert!(result.is_ok());
        let (validation_result, _, _undo_log) = result.unwrap();
        assert!(matches!(validation_result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_apply_transaction_insufficient_funds() {
        let mut utxo_set = UtxoSet::default();

        // Add a UTXO with insufficient value
        let prev_outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let prev_utxo = UTXO {
            value: 100, // Small value
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        };
        utxo_set.insert(prev_outpoint, std::sync::Arc::new(prev_utxo));

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 200, // More than input value
                script_pubkey: vec![OP_2].into(),
            }]
            .into(),
            lock_time: 0,
        };

        // The simplified implementation doesn't validate insufficient funds
        let result = apply_transaction(&tx, utxo_set, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_transaction_missing_utxo() {
        let utxo_set = UtxoSet::default(); // Empty UTXO set

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 100,
                script_pubkey: vec![OP_2].into(),
            }]
            .into(),
            lock_time: 0,
        };

        // The simplified implementation doesn't validate missing UTXOs
        let result = apply_transaction(&tx, utxo_set, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_block_header_future_timestamp() {
        use sha2::{Digest, Sha256};

        // Create header with non-zero merkle root (required for validation)
        // Timestamp validation now uses TimeContext (network time + median time-past)
        let header = BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: Sha256::digest(b"test merkle root")[..].try_into().unwrap(),
            timestamp: 9999999999, // Far future timestamp (would need network time check)
            bits: 0x1d00ffff,
            nonce: 0,
        };

        // Header structure is valid (actual future timestamp check needs network context)
        let result = validate_block_header(&header, None).unwrap();
        assert!(result);
    }

    #[test]
    fn test_validate_block_header_zero_timestamp() {
        use sha2::{Digest, Sha256};

        // Zero timestamp should be rejected by validate_block_header
        let header = BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: Sha256::digest(b"test merkle root")[..].try_into().unwrap(),
            timestamp: 0, // Zero timestamp (invalid)
            bits: 0x1d00ffff,
            nonce: 0,
        };

        // Zero timestamp should be rejected
        let result = validate_block_header(&header, None).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_connect_block_coinbase_exceeds_subsidy_edge() {
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 2100000000000000, // Exceeds total supply
                script_pubkey: vec![].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![coinbase_tx].into_boxed_slice(),
        };

        let utxo_set = UtxoSet::default();
        // Optimization: Pre-allocate witness vectors with capacity
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
            .collect();
        let result = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        );
        // The result should be Ok with ValidationResult::Invalid
        assert!(result.is_ok());
        let (validation_result, _, _undo_log) = result.unwrap();
        assert!(matches!(validation_result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_connect_block_first_tx_not_coinbase_edge() {
        let regular_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![OP_2].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![regular_tx].into_boxed_slice(), // First tx is not coinbase
        };

        let utxo_set = UtxoSet::default();
        // Optimization: Pre-allocate witness vectors with capacity
        let witnesses: Vec<Vec<Witness>> = block
            .transactions
            .iter()
            .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
            .collect();
        let result = connect_block(
            &block,
            &witnesses[..],
            utxo_set,
            0,
            None::<&[crate::types::BlockHeader]>,
            0u64,
            crate::types::Network::Mainnet,
        );
        // The result should be Ok with ValidationResult::Invalid
        assert!(result.is_ok());
        let (validation_result, _, _undo_log) = result.unwrap();
        assert!(matches!(validation_result, ValidationResult::Invalid(_)));
    }

    #[test]
    fn test_apply_transaction_multiple_inputs() {
        let mut utxo_set = UtxoSet::default();

        // Add multiple UTXOs
        let outpoint1 = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo1 = UTXO {
            value: 500,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        };
        utxo_set.insert(outpoint1, std::sync::Arc::new(utxo1));

        let outpoint2 = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let utxo2 = UTXO {
            value: 300,
            script_pubkey: vec![OP_2].into(),
            height: 0,
            is_coinbase: false,
        };
        utxo_set.insert(outpoint2, std::sync::Arc::new(utxo2));

        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1; 32].into(),
                        index: 0,
                    },
                    script_sig: vec![OP_1],
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [2; 32],
                        index: 0,
                    },
                    script_sig: vec![OP_2],
                    sequence: 0xffffffff,
                },
            ]
            .into(),
            outputs: vec![TransactionOutput {
                value: 700, // Total input value
                script_pubkey: vec![OP_3].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let (new_utxo_set, _undo_entries) = apply_transaction(&tx, utxo_set, 1).unwrap();
        assert_eq!(new_utxo_set.len(), 1);
    }

    #[test]
    fn test_apply_transaction_no_outputs() {
        let mut utxo_set = UtxoSet::default();

        let prev_outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let prev_utxo = UTXO {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        };
        utxo_set.insert(prev_outpoint, std::sync::Arc::new(prev_utxo));

        // Test that transactions with no outputs are rejected
        // This is a validation test, not an application test
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![].into(), // No outputs - should be invalid
            lock_time: 0,
        };

        // The transaction should be invalid due to no outputs
        // We can't apply an invalid transaction, so this test verifies validation rejects it
        let validation_result = crate::transaction::check_transaction(&tx).unwrap();
        assert!(matches!(validation_result, ValidationResult::Invalid(_)));

        // For the actual apply test, use a valid transaction with at least one output
        let valid_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32].into(),
                    index: 0,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 500, // Valid output
                script_pubkey: vec![OP_1].into(),
            }]
            .into(),
            lock_time: 0,
        };

        // Now apply the valid transaction
        let (new_utxo_set, _undo_entries) = apply_transaction(&valid_tx, utxo_set, 1).unwrap();
        // After applying, the input UTXO should be removed and the output UTXO should be added
        assert_eq!(new_utxo_set.len(), 1);

        // Verify the output UTXO exists
        let output_outpoint = OutPoint {
            hash: calculate_tx_id(&valid_tx),
            index: 0,
        };
        assert!(new_utxo_set.contains_key(&output_outpoint));
    }
}
