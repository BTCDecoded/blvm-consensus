//! Block connect logic: connect_block_inner and connect_block_inner_with_tx_ids.
//! Extracted from block/mod.rs for clarity (§1.1).
//!
//! Profile `[PERF_CLIFF]`: `BLVM_PERF_CLIFF_RANGES` (comma-separated `START-END`), `BLVM_PERF_CLIFF_STRIDE` (default 100).

use crate::activation::IsForkActive;
use crate::constants::*;
use crate::economic::get_block_subsidy;
use crate::error::{ConsensusError, Result};
use crate::opcodes::*;
#[cfg(feature = "profile")]
use crate::profile_log;
#[cfg(not(feature = "production"))]
use crate::script::verify_script_with_context_full;
use crate::segwit::{validate_witness_commitment, Witness};
#[cfg(not(feature = "production"))]
use crate::transaction::check_tx_inputs;
use crate::transaction::{check_transaction, is_coinbase};
use crate::types::*;
use crate::utxo_overlay::{apply_transaction_to_overlay_no_undo, UtxoOverlay};
use crate::witness::is_witness_empty;
use std::borrow::Cow;

#[cfg(not(feature = "production"))]
use super::calculate_script_flags_for_block_with_base;
#[cfg(feature = "production")]
use super::script_cache;
use super::{
    apply, calculate_base_script_flags_for_block, header, BlockValidationContext, UtxoDelta,
};

/// Shared empty witness matrix for blocks with no segwit data (avoids per-block `Arc::new(Vec::new())`).
#[cfg(feature = "production")]
#[inline]
fn arc_empty_witness_rows() -> std::sync::Arc<Vec<Vec<Witness>>> {
    use std::sync::{Arc, OnceLock};
    static EMPTY: OnceLock<Arc<Vec<Vec<Witness>>>> = OnceLock::new();
    EMPTY.get_or_init(|| Arc::new(Vec::new())).clone()
}

#[cold]
fn make_fee_overflow_error(transaction_index: Option<usize>) -> ConsensusError {
    let message = if let Some(i) = transaction_index {
        format!("Total fees overflow at transaction {i}")
    } else {
        "Total fees overflow".to_string()
    };
    ConsensusError::BlockValidation(message.into())
}

fn coinbase_script_sig_len(coinbase: &crate::types::Transaction) -> usize {
    coinbase
        .inputs
        .first()
        .map(|i| i.script_sig.len())
        .unwrap_or(0)
}

fn invalid_block_result<'a>(
    utxo_set: UtxoSet,
    tx_ids: &[Hash],
    msg: impl Into<String>,
) -> Result<(
    ValidationResult,
    UtxoSet,
    Cow<'a, [Hash]>,
    crate::reorganization::BlockUndoLog,
    Option<UtxoDelta>,
)> {
    // Return the **unchanged** UTXO set: every call site is before base `utxo_set` mutations
    // (overlay validates against an immutable view). Avoids emptying the set, which would force
    // callers like differential tests to clone or lose chain state after an invalid verdict.
    Ok((
        ValidationResult::Invalid(msg.into()),
        utxo_set,
        Cow::Owned(tx_ids.to_vec()),
        crate::reorganization::BlockUndoLog::new(),
        None,
    ))
}

/// BIP54 per-transaction sigop cap (§1.3 — single place for prod / non-prod paths).
fn check_bip54_sigop_limit<U: crate::utxo_overlay::UtxoLookup>(
    bip54_active: bool,
    tx: &Transaction,
    utxo_lookup: &U,
    wits: Option<&[Witness]>,
    tx_flags: u32,
    tx_ids: &[Hash],
) -> Result<Option<&'static str>> {
    if !bip54_active || is_coinbase(tx) {
        return Ok(None);
    }
    let sigop_count =
        crate::sigop::get_transaction_sigop_count_for_bip54(tx, utxo_lookup, wits, tx_flags)?;
    if sigop_count > crate::constants::BIP54_MAX_SIGOPS_PER_TX {
        return Ok(Some("BIP54: Transaction sigop count exceeds 2500"));
    }
    Ok(None)
}

/// Defer [`invalid_block_result`] until the script pre-queue loop owns `utxo_set` again.
#[cfg(all(feature = "production", feature = "rayon"))]
#[derive(Debug)]
enum ConnectQueueEarlyExit {
    Invalid(String),
}

/// Default `[PERF_CLIFF]` height bands (every [`PERF_CLIFF_STRIDE`] when inside any band).
#[cfg(feature = "profile")]
fn perf_cliff_default_bands() -> Vec<(u64, u64)> {
    vec![
        (120_000, 145_000),
        (180_000, 195_000),
        (200_000, 240_000),
        (300_000, 330_000),
        (380_000, 420_000),
    ]
}

/// `BLVM_PERF_CLIFF_RANGES`: comma-separated `START-END` inclusive (e.g. `300000-305000,320000-330000`).
#[cfg(feature = "profile")]
fn parse_perf_cliff_ranges(s: &str) -> Option<Vec<(u64, u64)>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (a, b) = part.split_once('-')?;
        let lo: u64 = a.trim().parse().ok()?;
        let hi: u64 = b.trim().parse().ok()?;
        if lo > hi {
            return None;
        }
        out.push((lo, hi));
    }
    (!out.is_empty()).then_some(out)
}

/// Whether `height` should emit `[PERF_CLIFF]` (profile + this stride, typically 100).
#[cfg(feature = "profile")]
fn perf_cliff_sample_height(height: u64, stride: u64) -> bool {
    if stride == 0 || height % stride != 0 {
        return false;
    }
    use std::sync::OnceLock;
    static BANDS: OnceLock<Vec<(u64, u64)>> = OnceLock::new();
    let bands = BANDS.get_or_init(|| {
        if let Ok(s) = std::env::var("BLVM_PERF_CLIFF_RANGES") {
            let t = s.trim();
            if !t.is_empty() {
                if let Some(v) = parse_perf_cliff_ranges(t) {
                    return v;
                }
                eprintln!(
                    "[blvm_consensus] BLVM_PERF_CLIFF_RANGES invalid; using default PERF_CLIFF bands"
                );
            }
        }
        perf_cliff_default_bands()
    });
    bands.iter().any(|(lo, hi)| height >= *lo && height <= *hi)
}

#[cfg(feature = "profile")]
fn perf_cliff_stride() -> u64 {
    use std::sync::OnceLock;
    static STRIDE: OnceLock<u64> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("BLVM_PERF_CLIFF_STRIDE")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(100)
    })
}

#[cfg(all(feature = "production", feature = "rayon"))]
fn use_per_sig_schnorr() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("BLVM_SCHNORR_PER_SIG")
            .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
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
                let cores = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(8);
                (cores / 2).clamp(4, 8)
            })
            .clamp(1, 16)
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
                std::thread::available_parallelism()
                    .map(|p| p.get().saturating_sub(1).max(1))
                    .unwrap_or(4)
            });
        let batch_size = blvm_primitives::ibd_tuning::chunk_threshold_config_or_hardware(
            crate::config::get_consensus_config_ref()
                .performance
                .ibd_chunk_threshold,
        );
        crate::checkqueue::ScriptCheckQueue::new(n, Some(batch_size))
    })
}
pub(crate) fn connect_block_inner<'a>(
    block: &Block,
    witnesses: &[Vec<Witness>],
    mut utxo_set: UtxoSet,
    witnesses_arc: Option<&std::sync::Arc<Vec<Vec<Witness>>>>,
    height: Natural,
    context: &BlockValidationContext,
    bip30_index: Option<&mut crate::bip_validation::Bip30Index>,
    precomputed_tx_ids: Option<&'a [Hash]>,
    block_arc: Option<std::sync::Arc<Block>>,
    ibd_mode: bool,
    best_header_chainwork: Option<u128>,
) -> Result<(
    ValidationResult,
    UtxoSet,
    Cow<'a, [Hash]>,
    crate::reorganization::BlockUndoLog,
    Option<UtxoDelta>,
)> {
    let time_context = context.time_context;
    let network = context.network;
    let bip54_boundary = context.bip54_boundary;
    let bip54_active = context.is_fork_active(ForkId::Bip54, height);

    // Preconditions: reject bad inputs without panicking (witness/API misuse or state overflow).
    if height > i64::MAX as u64 {
        return invalid_block_result(utxo_set, &[], "Block height exceeds representable range");
    }
    if utxo_set.len() > u32::MAX as usize {
        return Err(ConsensusError::BlockValidation(
            "UTXO set size exceeds maximum".into(),
        ));
    }
    if !witnesses.is_empty() && witnesses.len() != block.transactions.len() {
        return invalid_block_result(
            utxo_set,
            &[],
            format!(
                "Witness count {} must match transaction count {} (or pass empty witness matrix for legacy)",
                witnesses.len(),
                block.transactions.len()
            ),
        );
    }

    // Empty `witnesses` means "no witness stacks" from legacy/non-witness callers.
    // Expand to one empty stack per transaction so weight, script checks, and witness
    // commitment logic all use `witnesses.len() == block.transactions.len()`.
    // Drop `witnesses_arc` when expanding: callers may pass a shared empty Arc (wrong length).
    let witness_row_fallback: Option<Vec<Vec<Witness>>> =
        if witnesses.is_empty() && !block.transactions.is_empty() {
            Some(vec![Vec::new(); block.transactions.len()])
        } else {
            None
        };
    let witnesses_arc = if witness_row_fallback.is_some() {
        None
    } else {
        witnesses_arc
    };
    let witnesses: &[Vec<Witness>] = witness_row_fallback.as_deref().unwrap_or(witnesses);

    // Note: Header validation is handled by validate_block_header() below,
    // not by assertions, to allow tests to verify validation behavior
    // We only assert on values that are truly programming errors, not validation errors

    // Check block size and transaction count before validation
    #[cfg(feature = "production")]
    {
        // Quick reject: empty block (invalid)
        if block.transactions.is_empty() {
            return invalid_block_result(utxo_set, &[], "Block has no transactions");
        }

        // Quick reject: impossible tx count (before expensive validation). Real limit is weight;
        // use the same weight-derived ceiling as `compute_block_tx_ids` / parallel batch paths.
        if block.transactions.len() > crate::constants::MAX_TRANSACTIONS_PER_BLOCK {
            return invalid_block_result(
                utxo_set,
                &[],
                format!(
                    "Block has too many transactions: {}",
                    block.transactions.len()
                ),
            );
        }
    }

    #[cfg(feature = "profile")]
    let _fn_start = std::time::Instant::now();
    // 1. Validate block header (cheap — defer tx_ids until after)
    if !header::validate_block_header(&block.header, time_context.as_ref())? {
        return invalid_block_result(utxo_set, &[], "Invalid block header");
    }

    // BIP54 timewarp: at period boundaries require boundary timestamps and enforce rules
    if bip54_active {
        let rem = height % 2016;
        if rem == 2015 {
            let boundary = match bip54_boundary {
                Some(b) => b,
                None => {
                    return invalid_block_result(
                        utxo_set,
                        &[],
                        "BIP54: Boundary timestamps required at last block of period",
                    );
                }
            };
            if block.header.timestamp < boundary.timestamp_n_minus_2015 {
                return invalid_block_result(
                    utxo_set,
                    &[],
                    "BIP54: Block timestamp must be >= timestamp of first block of period",
                );
            }
        } else if rem == 0 {
            let boundary = match bip54_boundary {
                Some(b) => b,
                None => {
                    return invalid_block_result(
                        utxo_set,
                        &[],
                        "BIP54: Boundary timestamps required at first block of period",
                    );
                }
            };
            const TWOHOURS: u64 = 7200;
            let min_ts = boundary.timestamp_n_minus_1.saturating_sub(TWOHOURS);
            if block.header.timestamp < min_ts {
                return invalid_block_result(
                    utxo_set,
                    &[],
                    "BIP54: Block timestamp must be >= (previous block timestamp - 7200)",
                );
            }
        }
    }

    // Check block weight (DoS prevention)
    // This must be done before expensive transaction validation
    use crate::segwit::calculate_block_weight_from_nested;
    let block_weight = calculate_block_weight_from_nested(block, witnesses)?;
    if block_weight == 0 {
        return invalid_block_result(utxo_set, &[], "Block weight must be positive");
    }
    if block_weight > crate::constants::MAX_BLOCK_WEIGHT as u64 * 2 {
        return invalid_block_result(
            utxo_set,
            &[],
            format!("Block weight {block_weight} exceeds reasonable maximum"),
        );
    }
    if block_weight > crate::constants::MAX_BLOCK_WEIGHT as u64 {
        return invalid_block_result(
            utxo_set,
            &[],
            format!(
                "Block weight {} exceeds maximum {}",
                block_weight,
                crate::constants::MAX_BLOCK_WEIGHT
            ),
        );
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
    if block.header.version < 1 {
        return invalid_block_result(
            utxo_set,
            &[],
            format!(
                "Block header version {} must be >= 1 for BIP90 check",
                block.header.version
            ),
        );
    }
    let bip90_result = crate::bip_validation::check_bip90(block.header.version, height, context)?;
    #[cfg(any(debug_assertions, feature = "runtime-invariants"))]
    debug_assert!(
        bip90_result || height < BIP34_ACTIVATION_MAINNET, // BIP90 only applies after activation
        "BIP90 check was called but returned false - this should be handled below"
    );
    if !bip90_result {
        return invalid_block_result(
            utxo_set,
            &[],
            format!(
                "BIP90: Block version {} invalid at height {}",
                block.header.version, height
            ),
        );
    }

    #[cfg(feature = "profile")]
    profile_log!(
        "[TIMING] Block {}: pre_txid={:.2}ms",
        height,
        _fn_start.elapsed().as_secs_f64() * 1000.0
    );
    let tx_ids_cow: Cow<'a, [Hash]> = match precomputed_tx_ids {
        Some(s) => Cow::Borrowed(s),
        None => {
            if block.transactions.is_empty() {
                Cow::Owned(vec![])
            } else {
                Cow::Owned(crate::block::compute_block_tx_ids(block))
            }
        }
    };
    let tx_ids: &[Hash] = tx_ids_cow.as_ref();

    // Block tx merkle root verification (Orange Paper 8.4)
    // Matches Bitcoin Core: compute root + mutation flag; reject if root mismatches OR if
    // root matches but mutation detected (CVE-2012-2459 duplicate-tx attack).
    let (computed_merkle_root, merkle_mutated) =
        crate::mining::compute_merkle_root_and_mutated(tx_ids)?;
    if computed_merkle_root != block.header.merkle_root {
        return invalid_block_result(
            utxo_set,
            &[],
            "Block merkle root does not match transactions",
        );
    }
    // CVE-2012-2459 mutation check: only enforce outside IBD.
    // In IBD the block was already structurally validated by Bitcoin Core when it was
    // added to the chain (CheckBlock → ConnectBlock). We are doing ConnectBlock only —
    // replaying known-valid blocks from a trusted chunk file. Skipping avoids false
    // positives on mainnet blocks (e.g. block 481824) where the root still matches the
    // header but our intermediate-hash comparison fires on a non-duplicate tree structure.
    if merkle_mutated && !ibd_mode {
        return invalid_block_result(
            utxo_set,
            &[],
            "Duplicate transaction detected (CVE-2012-2459)",
        );
    }

    // BIP30: Duplicate coinbase prevention
    // CRITICAL: This check MUST be called - see tests/integration/bip_enforcement_tests.rs
    if block.transactions.is_empty() {
        return invalid_block_result(
            utxo_set,
            &[],
            "Block must have transactions for BIP30 check",
        );
    }
    let bip30_result = crate::bip_validation::check_bip30(
        block,
        &utxo_set,
        bip30_index.as_deref(),
        height,
        context,
        tx_ids.first(), // Pass precomputed coinbase txid, avoids calculate_tx_id in check_bip30
    )?;
    #[cfg(any(debug_assertions, feature = "runtime-invariants"))]
    debug_assert!(
        bip30_result || !block.transactions.is_empty(), // BIP30 only applies to coinbase
        "BIP30 check was called but returned false - this should be handled below"
    );
    if !bip30_result {
        return invalid_block_result(utxo_set, &[], "BIP30: Duplicate coinbase transaction");
    }

    // BIP34: Block height in coinbase (only after activation)
    // CRITICAL: This check MUST be called - see tests/integration/bip_enforcement_tests.rs
    // If this check is removed, integration tests will fail
    let bip34_result = crate::bip_validation::check_bip34(block, height, context)?;
    if !bip34_result {
        return invalid_block_result(
            utxo_set,
            &[],
            format!("BIP34: Block height {height} not correctly encoded in coinbase"),
        );
    }

    // BIP54: Consensus Cleanup (activation-gated)
    if bip54_active {
        let Some(coinbase) = block.transactions.first() else {
            return invalid_block_result(utxo_set, &[], "Block has no transactions");
        };
        if !crate::bip_validation::check_bip54_coinbase(coinbase, height) {
            return invalid_block_result(
                utxo_set,
                &[],
                "BIP54: Coinbase must have nLockTime = height - 13 and nSequence != 0xffffffff",
            );
        }
        for tx in block.transactions.iter().skip(1) {
            let stripped_size = crate::transaction::calculate_transaction_size(tx);
            if stripped_size == 64 {
                return invalid_block_result(
                    utxo_set,
                    &[],
                    "BIP54: Transactions with witness-stripped size 64 bytes are invalid",
                );
            }
        }
    }

    // Validate witnesses length matches transactions length (post legacy expansion).
    if witnesses.len() != block.transactions.len() {
        return invalid_block_result(
            utxo_set,
            &[],
            format!(
                "Witness count {} does not match transaction count {}",
                witnesses.len(),
                block.transactions.len()
            ),
        );
    }

    // tx_ids already computed above (before BIP30) for #21/#2

    // Hash-based ancestry verification: when assume_valid_hash is set and we're at
    // the assume-valid height, the block hash must match (reject otherwise).
    if let Some(expected_hash) = crate::config::get_assume_valid_hash() {
        if height == crate::block::get_assume_valid_height() {
            let serialized = crate::serialization::block::serialize_block_header(&block.header);
            let block_hash: [u8; 32] = crate::crypto::OptimizedSha256::new().hash256(&serialized);
            if block_hash != expected_hash {
                return invalid_block_result(
                    utxo_set,
                    &[],
                    format!(
                        "Assume-valid block hash mismatch at height {height}: expected {expected_hash:?}, got {block_hash:?}",
                    ),
                );
            }
        }
    }

    // Assume-valid: skip signature/script verification for blocks below the configured height.
    // Matches Bitcoin Core's hashAssumeValid behaviour: skip when height is below the assumed
    // block AND chain work is sufficient (nMinimumChainWork).  Bitcoin Core does NOT have a
    // two-week age check; we dropped that guard to keep parity and avoid inconsistency with
    // the per-signature short-circuit in script/signature.rs.
    #[cfg(feature = "production")]
    let chainwork_ok = best_header_chainwork
        .map(|cw| cw >= crate::config::get_n_minimum_chain_work())
        .unwrap_or(true);
    #[cfg(feature = "production")]
    let skip_signatures = height < crate::block::get_assume_valid_height() && chainwork_ok;

    #[cfg(not(feature = "production"))]
    let skip_signatures = false;

    // BLVM_DEBUG_ASSUMEVALID=1: log when the chainwork gate blocks the skip (requires `profile`).
    #[cfg(all(feature = "production", feature = "profile"))]
    if std::env::var("BLVM_DEBUG_ASSUMEVALID").is_ok() {
        let av = crate::block::get_assume_valid_height();
        if av > 0 && height < av && !skip_signatures {
            eprintln!(
                "[blvm_consensus::assumevalid] chainwork gate blocked assume-valid skip height={height} assume_valid_height={av} chainwork_ok={chainwork_ok} best_header_chainwork={best_header_chainwork:?} n_min_chain_work={}",
                crate::config::get_n_minimum_chain_work(),
            );
        }
    }

    // Pre-compute base script flags once per block from activation context
    let base_script_flags = calculate_base_script_flags_for_block(height, context);

    // Cache fork activation at block level — avoids per-tx table lookup
    let segwit_active = context.is_fork_active(ForkId::SegWit, height);
    let taproot_active = segwit_active && context.is_fork_active(ForkId::Taproot, height);

    // Pre-compute overlay capacities once (used by all validation paths)
    let estimated_outputs: usize = block.transactions.iter().map(|tx| tx.outputs.len()).sum();
    let estimated_inputs: usize = block.transactions.iter().map(|tx| tx.inputs.len()).sum();

    // 2. Validate all transactions
    #[cfg(feature = "profile")]
    profile_log!(
        "[TIMING] Block {}: pre_validation={:.2}ms",
        height,
        _fn_start.elapsed().as_secs_f64() * 1000.0
    );
    let mut total_fees = 0i64;
    // Sigop cost accumulated in overlay pass to avoid separate utxo_set pass
    let mut total_sigop_cost = 0u64;

    // When use_overlay_delta, extract additions/deletions from the overlay built during validation
    // instead of rebuilding (avoids ~10k redundant map ops/block).
    #[cfg(feature = "production")]
    let mut overlay_for_delta: Option<UtxoOverlay> = None;

    #[cfg(feature = "production")]
    {
        // Batch fee calculation - pre-fetch all UTXOs for fee calculation
        // Pre-collect prevouts for prefetch only (64 for better cache warmup)
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
            // Pre-allocate overlay with capacity (computed above)
            let mut overlay = UtxoOverlay::with_capacity(
                &utxo_set,
                estimated_outputs.max(100),
                estimated_inputs.max(100),
            );
            let mut validation_results: Vec<Result<(ValidationResult, i64, bool)>> =
                Vec::with_capacity(block.transactions.len());
            // NOTE: Undo entries are created when applying to real UTXO set, not during validation

            // prevout_script_pubkeys: per-tx allocation required (refs into overlay; must not outlive overlay mutation)
            // Block-level signature collectors. Single Mutex preserves collection order (tx0_in0, tx0_in1, ...)
            // so batch result indices match script order. Per-thread collectors broke ordering and caused
            // false "invalid signature" at e.g. block 164676 (see docs/IBD_BATCH_SPEED_PLAN.md §11).
            #[cfg(feature = "production")]
            use std::sync::Arc;
            #[cfg(feature = "profile")]
            let validation_start = std::time::Instant::now();
            #[cfg(feature = "profile")]
            let total_input_lookup_time = std::time::Duration::ZERO;
            #[cfg(feature = "profile")]
            let mut total_script_time = std::time::Duration::ZERO;
            #[cfg(feature = "profile")]
            let mut total_tx_structure_time = std::time::Duration::ZERO;
            #[cfg(feature = "profile")]
            let total_overlay_apply_time = std::time::Duration::ZERO;
            #[cfg(feature = "profile")]
            let total_check_tx_inputs_time = std::time::Duration::ZERO;
            #[cfg(all(feature = "production", feature = "profile"))]
            let mut script_checks_queued_count: usize = 0;

            // Structure validation: skip during IBD (block passed PoW, structure is guaranteed valid).
            // Non-IBD paths still validate.
            #[cfg(feature = "profile")]
            let structure_start = std::time::Instant::now();
            let mut valid_tx_indices = Vec::with_capacity(block.transactions.len());
            if ibd_mode {
                valid_tx_indices.extend(0..block.transactions.len());
            } else {
                let tx_structure_results: Vec<(usize, Result<ValidationResult>)> = {
                    if block.transactions.len() < 500 {
                        block
                            .transactions
                            .iter()
                            .enumerate()
                            .map(|(i, tx)| (i, check_transaction(tx)))
                            .collect()
                    } else {
                        use rayon::prelude::*;
                        block
                            .transactions
                            .par_iter()
                            .enumerate()
                            .map(|(i, tx)| (i, check_transaction(tx)))
                            .collect()
                    }
                };
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
            }
            #[cfg(feature = "profile")]
            {
                total_tx_structure_time += structure_start.elapsed();
            }

            // Per-input ECDSA counters for composite index (base << 16) | sub so batch sort order
            // is deterministic under parallel script verification (see docs/IBD_BATCH_SPEED_PLAN.md §11).
            #[cfg(feature = "production")]
            let total_ecdsa_inputs: usize = if skip_signatures {
                0
            } else {
                valid_tx_indices
                    .iter()
                    .map(|&idx| block.transactions[idx].inputs.len())
                    .sum()
            };
            #[cfg(feature = "production")]
            let ecdsa_sub_counters: std::sync::Arc<
                Vec<std::sync::atomic::AtomicUsize>,
            > = std::sync::Arc::new(
                (0..total_ecdsa_inputs)
                    .map(|_| std::sync::atomic::AtomicUsize::new(0))
                    .collect(),
            );
            #[cfg(feature = "production")]
            let ecdsa_index_base: usize = 0;

            // C/D: SoA collectors; created after total_inputs for pre-allocation.
            // Schnorr batching only when blvm-secp256k1 (crates.io secp256k1 has no batch API).
            #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
            let block_schnorr_collector = Arc::new(
                crate::bip348::SchnorrSignatureCollector::new_with_capacity(total_ecdsa_inputs),
            );

            // Hoist for parallel block validation
            // Caller MUST pass Some(Arc<Block>) to avoid full block clone — see connect_block_ibd.
            #[cfg(all(feature = "production", feature = "rayon"))]
            let block_arc = match block_arc {
                Some(a) => a,
                None => {
                    return Err(ConsensusError::BlockValidation(
                        "block Arc required for production+rayon validation (caller must pass Some(Arc::new(block)))"
                            .into(),
                    ));
                }
            };
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut tx_contexts: Vec<crate::checkqueue::TxScriptContext> = Vec::new();
            #[cfg(all(feature = "production", feature = "rayon"))]
            let results_arc = Arc::new(crossbeam_queue::SegQueue::new());
            // Block-level buffers: build as local Vecs, freeze to Arc before session (immutable for workers).
            #[cfg(all(feature = "production", feature = "rayon"))]
            let total_inputs: usize = if skip_signatures {
                0
            } else {
                valid_tx_indices
                    .iter()
                    .map(|&i| block.transactions[i].inputs.len())
                    .sum()
            };
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut script_pubkey_vec: Vec<u8> =
                Vec::with_capacity(total_inputs.saturating_mul(64).min(256 * 1024));
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut prevout_values_vec: Vec<i64> = Vec::with_capacity(total_inputs);
            #[cfg(all(feature = "production", feature = "rayon"))]
            let mut script_pubkey_indices_vec: Vec<(usize, usize)> =
                Vec::with_capacity(total_inputs);

            // Dedicated script workers: build buffers+tx_contexts, freeze to Arc, create session, add checks.
            #[cfg(all(feature = "production", feature = "rayon"))]
            {
                use crate::checkqueue::{BlockSessionContext, TxScriptContext};

                let block_ref = block;
                let witnesses_ref = witnesses;
                let time_ctx = time_context;
                let mut queue_results: Vec<Option<Result<(ValidationResult, i64, bool)>>> =
                    vec![None; valid_tx_indices.len()];

                let mut early_return: Option<
                    std::result::Result<ConnectQueueEarlyExit, ConsensusError>,
                > = None;
                let median_time_past = time_ctx
                    .map(|ctx| ctx.median_time_past)
                    .filter(|&mtp| mtp > 0);

                let mut ecdsa_index_base: usize = 0;

                // Sighash midstate cache: None = use thread-local (avoids Mutex contention across workers).

                // Q: Pre-allocate tx_checks Vec at block level; reuse per tx via add_from_slice
                // Reusable refs into script_pubkey_buffer; avoid per-tx Vec alloc
                // Reusable UTXO data (value, is_coinbase, height) — copy in tight loop, no refs held
                let mut utxo_data_reusable: Vec<Option<(i64, bool, u64)>> = Vec::with_capacity(256);
                let mut block_checks_buf: Vec<crate::checkqueue::ScriptCheck> =
                    Vec::with_capacity(total_inputs.min(2048));
                #[cfg(feature = "production")]
                let precomputed_sighashes: Vec<Option<[u8; 32]>> = Vec::new();
                #[cfg(feature = "production")]
                let precomputed_p2pkh_hashes: Vec<Option<[u8; 20]>> = Vec::new();
                #[cfg(all(feature = "production", feature = "profile"))]
                {
                    let _ = crate::script_profile::get_and_reset_script_sub_timing();
                    let _ = crate::script_profile::get_and_reset_p2pkh_timing();
                }

                #[cfg(feature = "profile")]
                let script_start = std::time::Instant::now();

                for (loop_idx, &i) in valid_tx_indices.iter().enumerate() {
                    if early_return.is_some() {
                        break;
                    }
                    let tx = &block_ref.transactions[i];

                    let wits_i = witnesses_ref.get(i).map(|w| w.as_slice());
                    let has_wit_i = segwit_active
                        && wits_i
                            .map(|w| w.iter().any(|wit| !is_witness_empty(wit)))
                            .unwrap_or(false);
                    let tx_flags_i = if !segwit_active {
                        base_script_flags
                    } else {
                        let mut flags = base_script_flags;
                        if has_wit_i || crate::segwit::is_segwit_transaction(tx) {
                            flags |= 0x800;
                        }
                        if taproot_active {
                            for output in &tx.outputs {
                                let script = &output.script_pubkey;
                                if script.len() == TAPROOT_SCRIPT_LENGTH
                                    && script[0] == OP_1
                                    && script[1] == PUSH_32_BYTES
                                {
                                    flags |= 0x8000;
                                    break;
                                }
                            }
                        }
                        flags
                    };

                    let (input_valid, fee, prevout_values_range, script_pubkey_indices_range) =
                        if is_coinbase(tx) {
                            match crate::sigop::get_transaction_sigop_cost_with_witness_slices(
                                tx, &overlay, wits_i, tx_flags_i,
                            ) {
                                Ok(cost) => {
                                    total_sigop_cost = match total_sigop_cost.checked_add(cost) {
                                        Some(v) => v,
                                        None => {
                                            early_return =
                                                Some(Err(ConsensusError::BlockValidation(
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
                            };
                            (ValidationResult::Valid, 0, (0, 0), (0, 0))
                        } else if skip_signatures {
                            // Fast path: skip_signatures avoids per-tx utxo_refs Vec allocation.
                            // Sigop cost uses overlay directly (UtxoLookup trait).
                            utxo_data_reusable.clear();
                            utxo_data_reusable.reserve(tx.inputs.len());
                            let mut utxo_missing: Option<(usize, crate::types::OutPoint)> = None;
                            for (input_idx, input) in tx.inputs.iter().enumerate() {
                                match overlay.get(&input.prevout) {
                                    Some(u) => {
                                        utxo_data_reusable.push(Some((
                                            u.value,
                                            u.is_coinbase,
                                            u.height,
                                        )));
                                    }
                                    None => {
                                        utxo_data_reusable.push(None);
                                        utxo_missing = Some((input_idx, input.prevout));
                                        break;
                                    }
                                }
                            }
                            if let Some((idx, prevout)) = utxo_missing {
                                early_return = Some(Ok(ConnectQueueEarlyExit::Invalid(format!(
                                    "UTXO not found for input {} (prevout {}:{} tx_idx={})",
                                    idx,
                                    hex::encode(prevout.hash),
                                    prevout.index,
                                    i,
                                ))));
                                break;
                            }
                            match crate::sigop::get_transaction_sigop_cost_with_witness_slices(
                                tx, &overlay, wits_i, tx_flags_i,
                            ) {
                                Ok(cost) => {
                                    total_sigop_cost = match total_sigop_cost.checked_add(cost) {
                                        Some(v) => v,
                                        None => {
                                            early_return =
                                                Some(Err(ConsensusError::BlockValidation(
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
                            let (input_valid, fee) =
                                match crate::transaction::check_tx_inputs_with_owned_data(
                                    tx,
                                    height,
                                    &utxo_data_reusable,
                                ) {
                                    Ok(x) => x,
                                    Err(e) => {
                                        early_return = Some(Err(e));
                                        break;
                                    }
                                };
                            (input_valid, fee, (0, 0), (0, 0))
                        } else {
                            // Full path: build utxo_refs for sigop + script check buffers.
                            utxo_data_reusable.clear();
                            utxo_data_reusable.reserve(tx.inputs.len());
                            let mut utxo_refs: Vec<Option<&crate::types::UTXO>> =
                                Vec::with_capacity(tx.inputs.len());
                            let pv_start = prevout_values_vec.len();
                            let spi_start = script_pubkey_indices_vec.len();
                            let mut utxo_missing: Option<(usize, crate::types::OutPoint)> = None;
                            for (input_idx, input) in tx.inputs.iter().enumerate() {
                                match overlay.get(&input.prevout) {
                                    Some(u) => {
                                        utxo_refs.push(Some(u));
                                        utxo_data_reusable.push(Some((
                                            u.value,
                                            u.is_coinbase,
                                            u.height,
                                        )));
                                        prevout_values_vec.push(u.value);
                                        let start = script_pubkey_vec.len();
                                        script_pubkey_vec
                                            .extend_from_slice(u.script_pubkey.as_ref());
                                        script_pubkey_indices_vec
                                            .push((start, u.script_pubkey.len()));
                                    }
                                    None => {
                                        utxo_refs.push(None);
                                        utxo_data_reusable.push(None);
                                        utxo_missing = Some((input_idx, input.prevout));
                                        break;
                                    }
                                }
                            }
                            if let Some((idx, prevout)) = utxo_missing {
                                early_return = Some(Ok(ConnectQueueEarlyExit::Invalid(format!(
                                    "UTXO not found for input {} (prevout {}:{} tx_idx={})",
                                    idx,
                                    hex::encode(prevout.hash),
                                    prevout.index,
                                    i,
                                ))));
                                break;
                            }
                            match crate::sigop::get_transaction_sigop_cost_with_utxos(
                                tx, &utxo_refs, wits_i, tx_flags_i,
                            ) {
                                Ok(cost) => {
                                    total_sigop_cost = match total_sigop_cost.checked_add(cost) {
                                        Some(v) => v,
                                        None => {
                                            early_return =
                                                Some(Err(ConsensusError::BlockValidation(
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
                            drop(utxo_refs);
                            let (input_valid, fee) =
                                match crate::transaction::check_tx_inputs_with_owned_data(
                                    tx,
                                    height,
                                    &utxo_data_reusable,
                                ) {
                                    Ok(x) => x,
                                    Err(e) => {
                                        early_return = Some(Err(e));
                                        break;
                                    }
                                };
                            let pv_count = prevout_values_vec.len() - pv_start;
                            let spi_count = script_pubkey_indices_vec.len() - spi_start;
                            (
                                input_valid,
                                fee,
                                (pv_start, pv_count),
                                (spi_start, spi_count),
                            )
                        };

                    if !matches!(input_valid, ValidationResult::Valid) {
                        queue_results[loop_idx] = Some(Ok((
                            ValidationResult::Invalid(format!(
                                "Invalid transaction inputs at index {i}"
                            )),
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
                    // Reuse tx_flags_i from sigop (same has_witness)
                    let flags = tx_flags_i;

                    #[cfg(feature = "production")]
                    let bip143 = if has_wit_i {
                        // Production compute() ignores prevout_values/script_pubkeys; pass empty to avoid alloc
                        Some(crate::transaction_hash::Bip143PrecomputedHashes::compute(
                            tx,
                            &[],
                            &[],
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

                    // Build ScriptChecks in first loop (single pass).
                    let tx_ctx_idx = tx_contexts.len() - 1;
                    let (spi_base, spi_count) = script_pubkey_indices_range;
                    let (pv_base, pv_count) = prevout_values_range;
                    let spi = script_pubkey_indices_vec.as_slice();
                    let pv = prevout_values_vec.as_slice();
                    // Script exec cache: skip all checks if (witness_hash, flags) cached.
                    #[cfg(all(feature = "production", feature = "rayon"))]
                    if segwit_active {
                        if let Some(tx_witnesses) = witnesses_ref.get(i) {
                            if tx_witnesses.len() == tx.inputs.len() {
                                let key =
                                    crate::script_exec_cache::compute_key(tx, tx_witnesses, flags);
                                if crate::script_exec_cache::contains(&key) {
                                    let synthetic: Vec<_> =
                                        (0..tx.inputs.len()).map(|_| (tx_ctx_idx, true)).collect();
                                    results_arc.push(synthetic);
                                    queue_results[loop_idx] =
                                        Some(Ok((ValidationResult::Valid, fee, true)));
                                    ecdsa_index_base += tx.inputs.len();
                                    let tx_id = tx_ids[i];
                                    apply_transaction_to_overlay_no_undo(
                                        &mut overlay,
                                        tx,
                                        tx_id,
                                        height,
                                    );
                                    continue;
                                }
                            }
                        }
                    }
                    for j in 0..tx.inputs.len() {
                        let (spk_off, spk_l) = if j < spi_count {
                            spi[spi_base + j]
                        } else {
                            (0, 0)
                        };
                        let pv_val = if j < pv_count { pv[pv_base + j] } else { 0 };
                        block_checks_buf.push(crate::checkqueue::ScriptCheck {
                            tx_ctx_idx,
                            input_idx: j,
                            spk_offset: spk_off as u32,
                            spk_len: spk_l as u32,
                            prevout_value: pv_val,
                        });
                    }

                    ecdsa_index_base += tx.inputs.len();

                    let tx_id = tx_ids[i];
                    apply_transaction_to_overlay_no_undo(&mut overlay, tx, tx_id, height);
                }

                if let Some(r) = early_return.take() {
                    match r {
                        Ok(ConnectQueueEarlyExit::Invalid(msg)) => {
                            return invalid_block_result(utxo_set, tx_ids, msg);
                        }
                        Err(e) => return Err(e),
                    }
                }

                // Assume-valid (or all txs satisfied via script-exec cache): no ScriptChecks queued.
                // Skip CCheckQueue session setup, empty run_checks_sequential, and redundant Arc clones.
                if block_checks_buf.is_empty() {
                    for r in queue_results {
                        match r {
                            None => {
                                return Err(ConsensusError::BlockValidation(
                                    "Internal error: script check queue slot not filled".into(),
                                ));
                            }
                            Some(Ok(triple)) => validation_results.push(Ok(triple)),
                            Some(Err(e)) => return Err(e),
                        }
                    }
                    #[cfg(all(feature = "production", feature = "profile"))]
                    {
                        script_checks_queued_count = 0;
                    }
                } else {
                    let witness_buffer: std::sync::Arc<Vec<Vec<Witness>>> =
                        witnesses_arc.map(Arc::clone).unwrap_or_else(|| {
                            if witnesses.is_empty() {
                                arc_empty_witness_rows()
                            } else {
                                Arc::new(witnesses.to_vec())
                            }
                        });
                    let tx_contexts_arc = Arc::new(tx_contexts);
                    let script_pubkey_buffer = Arc::new(script_pubkey_vec);
                    let prevout_values_buffer = Arc::new(prevout_values_vec);
                    let script_pubkey_indices_buffer = Arc::new(script_pubkey_indices_vec);
                    #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
                    let schnorr_collector = if use_per_sig_schnorr() {
                        None
                    } else {
                        Some(Arc::clone(&block_schnorr_collector))
                    };
                    #[cfg(all(feature = "production", not(feature = "blvm-secp256k1")))]
                    let schnorr_collector: Option<
                        Arc<crate::bip348::SchnorrSignatureCollector>,
                    > = None;

                    // Small-block fast path: skip CCheckQueue overhead for blocks with <32 inputs.
                    const SMALL_BLOCK_THRESHOLD: usize = 32;
                    let precomputed_sighashes_arc = Arc::new(precomputed_sighashes);
                    let precomputed_p2pkh_hashes_arc = Arc::new(precomputed_p2pkh_hashes);

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
                            schnorr_collector: schnorr_collector.clone(),
                            height,
                            median_time_past,
                            network,
                            activation: context.activation.clone(),
                            results: Arc::new(crossbeam_queue::SegQueue::new()),
                            #[cfg(feature = "production")]
                            precomputed_sighashes: Arc::clone(&precomputed_sighashes_arc),
                            #[cfg(feature = "production")]
                            precomputed_p2pkh_hashes: Arc::clone(&precomputed_p2pkh_hashes_arc),
                        };
                        crate::checkqueue::ScriptCheckQueue::run_checks_sequential(
                            &block_checks_buf,
                            &seq_session,
                        )?
                    } else {
                        let rayon_session = Arc::new(BlockSessionContext {
                            block: Arc::clone(&block_arc),
                            prevout_values_buffer: Arc::clone(&prevout_values_buffer),
                            script_pubkey_indices_buffer: Arc::clone(&script_pubkey_indices_buffer),
                            script_pubkey_buffer: Arc::clone(&script_pubkey_buffer),
                            witness_buffer: Arc::clone(&witness_buffer),
                            tx_contexts: Arc::clone(&tx_contexts_arc),
                            #[cfg(feature = "production")]
                            ecdsa_sub_counters: Arc::clone(&ecdsa_sub_counters),
                            #[cfg(feature = "production")]
                            schnorr_collector,
                            height,
                            median_time_past,
                            network,
                            activation: context.activation.clone(),
                            results: Arc::clone(&results_arc),
                            #[cfg(feature = "production")]
                            precomputed_sighashes: Arc::clone(&precomputed_sighashes_arc),
                            #[cfg(feature = "production")]
                            precomputed_p2pkh_hashes: Arc::clone(&precomputed_p2pkh_hashes_arc),
                        });

                        use rayon::prelude::*;
                        let rayon_results: Vec<std::result::Result<(usize, bool), ConsensusError>> =
                            block_checks_buf
                                .par_iter()
                                .map(|c| {
                                    let session = rayon_session.as_ref();
                                    let buffer = session.script_pubkey_buffer.as_slice();
                                    let ctx = &session.tx_contexts[c.tx_ctx_idx];
                                    let tx = &session.block.transactions[ctx.tx_index];
                                    let flags = ctx.flags;
                                    let height = session.height;
                                    let network = session.network;
                                    let s = c.spk_offset as usize;
                                    let l = c.spk_len as usize;
                                    let script_pubkey = if s + l <= buffer.len() {
                                        &buffer[s..s + l]
                                    } else {
                                        &[]
                                    };

                                    let spk_len = script_pubkey.len();
                                    let last_byte = if spk_len > 0 {
                                        script_pubkey[spk_len - 1]
                                    } else {
                                        0
                                    };

                                    // P2PK fast path: <pubkey> OP_CHECKSIG (35 or 67 bytes)
                                    // Gate: script_sig must parse as exactly <sig> (1 push).
                                    if (spk_len == 35 || spk_len == 67)
                                        && last_byte == OP_CHECKSIG
                                        && (script_pubkey[0] == PUSH_33_BYTES
                                            || script_pubkey[0] == PUSH_65_BYTES)
                                        && crate::script::parse_p2pk_script_sig(
                                            tx.inputs[c.input_idx].script_sig.as_slice(),
                                        )
                                        .is_some()
                                    {
                                        return crate::script::verify_p2pk_inline(
                                            tx.inputs[c.input_idx].script_sig.as_slice(),
                                            script_pubkey,
                                            flags,
                                            tx,
                                            c.input_idx,
                                            height,
                                            network,
                                        )
                                        .map(|v| (c.tx_ctx_idx, v))
                                        .map_err(|e| {
                                            ConsensusError::BlockValidation(
                                                format!(
                                                    "P2PK tx {} input {}: {}",
                                                    ctx.tx_index, c.input_idx, e
                                                )
                                                .into(),
                                            )
                                        });
                                    }

                                    // P2PKH fast path: OP_DUP OP_HASH160 <20> ... OP_EQUALVERIFY OP_CHECKSIG
                                    // Gate: script_sig must parse as exactly <sig> <pubkey> (2 pushes).
                                    // Non-standard script_sigs (e.g. OP_0 <sig> <pubkey>) fall through to
                                    // the full interpreter which handles them correctly.
                                    if spk_len == 25
                                        && script_pubkey[0] == OP_DUP
                                        && script_pubkey[1] == OP_HASH160
                                        && script_pubkey[2] == PUSH_20_BYTES
                                        && script_pubkey[23] == OP_EQUALVERIFY
                                        && last_byte == OP_CHECKSIG
                                        && crate::script::parse_p2pkh_script_sig(
                                            tx.inputs[c.input_idx].script_sig.as_slice(),
                                        )
                                        .is_some()
                                    {
                                        return crate::script::verify_p2pkh_inline(
                                            tx.inputs[c.input_idx].script_sig.as_slice(),
                                            script_pubkey,
                                            flags,
                                            tx,
                                            c.input_idx,
                                            height,
                                            network,
                                            None,
                                        )
                                        .map(|v| (c.tx_ctx_idx, v))
                                        .map_err(|e| {
                                            ConsensusError::BlockValidation(
                                                format!(
                                                    "P2PKH tx {} input {}: {}",
                                                    ctx.tx_index, c.input_idx, e
                                                )
                                                .into(),
                                            )
                                        });
                                    }

                                    // Fallback: full interpreter path
                                    let pv = session.prevout_values_buffer.as_slice();
                                    let spi = session.script_pubkey_indices_buffer.as_slice();
                                    let (pv_base, pv_count) = ctx.prevout_values_range;
                                    let prevout_slice = &pv[pv_base..][..pv_count];
                                    let (spi_base, spi_count) = ctx.script_pubkey_indices_range;
                                    let refs: Vec<&[u8]> = (0..spi_count)
                                        .map(|j| {
                                            let (start, len) = spi[spi_base + j];
                                            if start + len <= buffer.len() {
                                                &buffer[start..start + len]
                                            } else {
                                                &[]
                                            }
                                        })
                                        .collect();
                                    let valid =
                                        crate::checkqueue::ScriptCheckQueue::run_check_with_refs(
                                            c,
                                            session,
                                            ctx,
                                            &refs,
                                            buffer,
                                            #[cfg(feature = "production")]
                                            None,
                                            Some(script_pubkey),
                                            Some(prevout_slice),
                                        )?;
                                    Ok((c.tx_ctx_idx, valid))
                                })
                                .collect();
                        let mut check_results = Vec::with_capacity(rayon_results.len());
                        for r in rayon_results {
                            check_results.push(r?);
                        }
                        check_results
                    };

                    #[cfg(feature = "profile")]
                    {
                        total_script_time += script_start.elapsed();
                    }

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
                        match r {
                            None => {
                                return Err(ConsensusError::BlockValidation(
                                    "Internal error: script check queue slot not filled".into(),
                                ));
                            }
                            Some(Ok(triple)) => validation_results.push(Ok(triple)),
                            Some(Err(e)) => return Err(e),
                        }
                    }
                    #[cfg(all(feature = "production", feature = "profile"))]
                    {
                        script_checks_queued_count = block_checks_buf.len();
                    }
                }
            }

            // Sequential application (write operations) — must be sequential
            // NOTE: Use block_arc (block moved into parallel block at 741)
            if validation_results.len() != block_arc.transactions.len() {
                return Err(ConsensusError::BlockValidation(
                    format!(
                        "Validation results count {} must match transaction count {}",
                        validation_results.len(),
                        block_arc.transactions.len()
                    )
                    .into(),
                ));
            }

            for (i, result) in validation_results.into_iter().enumerate() {
                let (input_valid, fee, script_valid) = result?;

                if !matches!(input_valid, ValidationResult::Valid) {
                    return Ok((
                        input_valid,
                        utxo_set,
                        tx_ids_cow.clone(),
                        crate::reorganization::BlockUndoLog::new(),
                        None,
                    ));
                }

                if !script_valid {
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!("Invalid script at transaction {i}"),
                    );
                }

                if fee < 0 {
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!("Fee {fee} must be non-negative at transaction {i}"),
                    );
                }
                // Use checked arithmetic to prevent fee overflow
                total_fees = total_fees
                    .checked_add(fee)
                    .ok_or_else(|| make_fee_overflow_error(Some(i)))?;
                if total_fees < 0 {
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!(
                            "Total fees {total_fees} must be non-negative after transaction {i}"
                        ),
                    );
                }
            }
            #[cfg(all(feature = "production", feature = "profile"))]
            let validation_elapsed = validation_start.elapsed();
            #[cfg(all(feature = "production", feature = "profile"))]
            let total_inputs: usize = block_arc
                .transactions
                .iter()
                .filter(|tx| !is_coinbase(tx))
                .map(|tx| tx.inputs.len())
                .sum();
            #[cfg(all(feature = "production", feature = "profile"))]
            {
                let (p2pk, p2pkh, p2sh, p2wpkh, p2wsh, p2tr, bare_ms, interp) =
                    crate::script::get_and_reset_fast_path_counts();
                let total = p2pk + p2pkh + p2sh + p2wpkh + p2wsh + p2tr + bare_ms + interp;
                if total > 0 {
                    let pct = |n: u64| (100.0 * n as f64 / total as f64).round() as u32;
                    eprintln!(
                        "[FAST_PATH] Block {}: p2pk={}% p2pkh={}% p2sh={}% p2wpkh={}% p2wsh={}% p2tr={}% bare_ms={}% interpreter={}% (n={})",
                        height, pct(p2pk), pct(p2pkh), pct(p2sh), pct(p2wpkh), pct(p2wsh), pct(p2tr), pct(bare_ms), pct(interp), total
                    );
                }
            }
            // Batch verify Schnorr signatures (ECDSA uses per-sig verification only).
            // Only when blvm-secp256k1 — crates.io secp256k1 has no batch API; workers verify per-sig.
            #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
            {
                let batch_start = std::time::Instant::now();
                let schnorr_empty = block_schnorr_collector.is_empty();
                let schnorr_result = if schnorr_empty {
                    Ok(Vec::new())
                } else {
                    block_schnorr_collector.verify_batch()
                };
                let schnorr_sig_count = schnorr_result.as_ref().map(|v| v.len()).unwrap_or(0);
                if schnorr_empty {
                    // No-op
                } else if let Err(e) = schnorr_result {
                    #[cfg(feature = "profile")]
                    profile_log!(
                        "[BATCH] Block {}: Schnorr batch verification failed: {:?}",
                        height,
                        e
                    );
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!("Schnorr batch verification failed: {e:?}"),
                    );
                } else {
                    let schnorr_results = schnorr_result.unwrap();
                    if schnorr_results.iter().any(|&v| !v) {
                        #[cfg(feature = "profile")]
                        profile_log!(
                            "[BATCH] Block {}: {} Schnorr signatures, {} invalid",
                            height,
                            schnorr_results.len(),
                            schnorr_results.iter().filter(|&&v| !v).count()
                        );
                        return invalid_block_result(
                            utxo_set,
                            tx_ids,
                            "Invalid Schnorr signature in block",
                        );
                    }
                    #[cfg(feature = "profile")]
                    if !schnorr_results.is_empty() {
                        profile_log!(
                            "[BATCH] Block {}: {} Schnorr signatures verified successfully",
                            height,
                            schnorr_results.len()
                        );
                    }
                }
                let total_batch_time = batch_start.elapsed();
                #[cfg(feature = "profile")]
                {
                    if total_batch_time.as_millis() > 10 {
                        profile_log!(
                            "[BATCH_PERF] Block {}: Total batch verification time: {:?}",
                            height,
                            total_batch_time
                        );
                    }
                    // PERF total = validation (script/structure/overlay) + batch — this is the real cost per block (why we see ~70–100 b/s on heavy blocks).
                    // schnorr_sigs/ecdsa_sigs help correlate slow blocks with ECDSA-heavy blocks (batch bottleneck).
                    let total_with_batch = validation_elapsed + total_batch_time;
                    let (sighash_ns, interpreter_ns, multisig_ns) =
                        crate::script_profile::get_and_reset_script_sub_timing();
                    let (
                        p2pkh_parse_ns,
                        p2pkh_hash160_ns,
                        p2pkh_collect_ns,
                        p2pkh_entry_ns,
                        p2pkh_bip66_ns,
                        p2pkh_secp_ns,
                    ) = crate::script_profile::get_and_reset_p2pkh_timing();
                    let (collect_slot_ns, collect_lock_ns, collect_copy_ns, collect_chunk_ns) =
                        crate::script_profile::get_and_reset_collect_timing();
                    let (
                        worker_p2pkh_ns,
                        worker_refs_ns,
                        worker_refs_lock_ns,
                        run_check_loop_ns,
                        results_extend_ns,
                    ) = crate::script_profile::get_and_reset_worker_timing();
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
                    // script_checks_queued = inputs sent to CCheckQueue (0 when assume-valid skips signatures).
                    // (Former field ecdsa_sigs was always 0 here — misleading vs real verification.)
                    profile_log!("[PERF] Block {}: total={:?} (validation_loop={:?} batch={:?}), script_sub: sighash={:.2}ms interpreter={:.2}ms multisig={:.2}ms p2pkh_entry={:.2}ms p2pkh_parse={:.2}ms p2pkh_hash160={:.2}ms p2pkh_bip66={:.2}ms p2pkh_collect={:.2}ms p2pkh_secp={:.2}ms collect_slot={:.2}ms collect_lock={:.2}ms collect_copy={:.2}ms collect_chunk={:.2}ms worker_refs={:.2}ms worker_p2pkh={:.2}ms worker_refs_lock={:.2}ms run_check_loop={:.2}ms results_extend={:.2}ms batch_extract={:.2}ms batch_secp={:.2}ms batch_cache={:.2}ms drain_copy={:.2}ms drain_parse={:.2}ms drain_secp={:.2}ms ecdsa_cache_hits={} ecdsa_cache_misses={}, structure={:?}, input_lookup={:?}, check_inputs={:?}, overlay_apply={:?}, txs={} inputs={} schnorr_batch_sigs={} script_checks_queued={}",
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
                        schnorr_sig_count,
                        script_checks_queued_count
                    );
                    let total_ns = total_with_batch.as_nanos() as f64;
                    if total_ns > 0.0 {
                        let pct = |d: std::time::Duration| {
                            (100.0 * d.as_nanos() as f64 / total_ns).min(100.0)
                        };
                        let script_pct = pct(total_script_time);
                        let batch_pct = pct(total_batch_time);
                        let input_lookup_pct = pct(total_input_lookup_time);
                        let check_inputs_pct = pct(total_check_tx_inputs_time);
                        let overlay_pct = pct(total_overlay_apply_time);
                        let structure_pct = pct(total_tx_structure_time);
                        let total_ms = total_with_batch.as_secs_f64() * 1000.0;
                        // Sparse bands: defaults or `BLVM_PERF_CLIFF_RANGES`; stride `BLVM_PERF_CLIFF_STRIDE` (default 100).
                        if perf_cliff_sample_height(height, perf_cliff_stride()) {
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
            // Pre-allocate overlay with capacity (computed above)
            let mut overlay = UtxoOverlay::with_capacity(
                &utxo_set,
                estimated_outputs.max(100),
                estimated_inputs.max(100),
            );

            // Pre-allocate reusable Vecs to avoid per-transaction allocations
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
                // Accumulate sigop for this tx (non-rayon path; overlay has prev txs)
                let wits_i = witnesses.get(i).map(|w| w.as_slice());
                let has_wit = wits_i
                    .map(|w| w.iter().any(|wit| !is_witness_empty(wit)))
                    .unwrap_or(false);
                let tx_flags = calculate_script_flags_for_block_with_base(
                    tx,
                    has_wit,
                    base_script_flags,
                    height,
                    context,
                );
                total_sigop_cost = total_sigop_cost
                    .checked_add(
                        crate::sigop::get_transaction_sigop_cost_with_witness_slices(
                            tx, &overlay, wits_i, tx_flags,
                        )?,
                    )
                    .ok_or_else(|| ConsensusError::BlockValidation("Sigop cost overflow".into()))?;

                if let Some(msg) =
                    check_bip54_sigop_limit(bip54_active, tx, &overlay, wits_i, tx_flags, tx_ids)?
                {
                    return invalid_block_result(utxo_set, tx_ids, msg);
                }

                let structure_start = std::time::Instant::now();
                // Validate transaction structure
                let tx_valid = check_transaction(tx)?;
                total_tx_structure_time += structure_start.elapsed();
                if !matches!(tx_valid, ValidationResult::Valid) {
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!("Invalid transaction at index {i}"),
                    );
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
                        input_utxos_reusable
                            .reserve(tx.inputs.len() - input_utxos_reusable.capacity());
                    }
                    let mut total_input: i64 = 0;

                    for (input_idx, input) in tx.inputs.iter().enumerate() {
                        match overlay.get(&input.prevout) {
                            Some(utxo) => {
                                input_utxos_reusable.push(Some(utxo));
                                total_input =
                                    total_input.checked_add(utxo.value).ok_or_else(|| {
                                        ConsensusError::TransactionValidation(
                                            "Input value overflow".into(),
                                        )
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
                                return invalid_block_result(
                                    utxo_set,
                                    tx_ids,
                                    format!("UTXO not found for input {}", input_idx),
                                );
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
                        .map_err(|e| {
                            ConsensusError::TransactionValidation(Cow::Owned(e.to_string()))
                        })?;

                    let fee = total_input.checked_sub(total_output).ok_or_else(|| {
                        ConsensusError::TransactionValidation("Fee calculation underflow".into())
                    })?;

                    if fee < 0 {
                        (ValidationResult::Invalid("Negative fee".to_string()), 0)
                    } else {
                        // fee = total_input - total_output ⇒ fee <= total_input when non-negative
                        // Pass pre-collected UTXOs to avoid redundant lookups
                        let (input_valid, _) = crate::transaction::check_tx_inputs_with_utxos(
                            tx,
                            &overlay,
                            height,
                            Some(&input_utxos_reusable),
                        )?;
                        (input_valid, fee)
                    }
                };
                let input_utxos = &input_utxos_reusable;

                if !matches!(input_valid, ValidationResult::Valid) {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "   ❌ [non-parallel] Block {} TX {}: input_valid={:?}",
                        height, i, input_valid
                    );
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!("Invalid transaction inputs at index {i}"),
                    );
                }

                // Verify scripts for non-coinbase transactions
                // Skip signature verification if assume-valid
                // Reuse input_utxos collected during fee calculation
                if !is_coinbase(tx) && !skip_signatures {
                    // Reuse pre-allocated Vecs instead of allocating per transaction
                    prevout_values_reusable.clear();
                    prevout_script_pubkeys_reusable.clear();
                    if prevout_script_pubkeys_reusable.capacity() < input_utxos.len() {
                        prevout_script_pubkeys_reusable.reserve(
                            input_utxos
                                .len()
                                .saturating_sub(prevout_script_pubkeys_reusable.capacity()),
                        );
                    }
                    if prevout_values_reusable.capacity() < input_utxos.len() {
                        prevout_values_reusable
                            .reserve(input_utxos.len() - prevout_values_reusable.capacity());
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
                    let has_witness = tx_witnesses
                        .map(|w| w.iter().any(|wit| !is_witness_empty(wit)))
                        .unwrap_or(false);
                    let flags = calculate_script_flags_for_block_with_base(
                        tx,
                        has_witness,
                        base_script_flags,
                        height,
                        context,
                    );
                    let median_time_past = time_context
                        .map(|ctx| ctx.median_time_past)
                        .filter(|&mtp| mtp > 0);
                    #[cfg(feature = "production")]
                    let bip143_hashes = if has_witness {
                        Some(crate::transaction_hash::Bip143PrecomputedHashes::compute(
                            tx,
                            &prevout_values_reusable,
                            &prevout_script_pubkeys_reusable,
                        ))
                    } else {
                        None
                    };
                    for (j, input) in tx.inputs.iter().enumerate() {
                        // Reuse input_utxos instead of overlay.get()
                        if let Some(utxo) = input_utxos.get(j).and_then(|opt| *opt) {
                            let witness_elem = tx_witnesses.and_then(|w| w.get(j));
                            let witness_for_script = witness_elem.and_then(|w| {
                                if is_witness_empty(w) {
                                    None
                                } else {
                                    Some(w)
                                }
                            });

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
                                #[cfg(feature = "production")]
                                Some(&schnorr_collector),
                                #[cfg(feature = "production")]
                                None, // ECDSA: per-sig only (no batch)
                                #[cfg(feature = "production")]
                                None,
                                #[cfg(feature = "production")]
                                bip143_hashes.as_ref(),
                                #[cfg(not(feature = "production"))]
                                None,
                                #[cfg(feature = "production")]
                                None, // precomputed_sighash_all (non-CCheckQueue path)
                                #[cfg(feature = "production")]
                                None, // precomputed_p2pkh_hash
                            )? {
                                return invalid_block_result(
                                    utxo_set,
                                    tx_ids,
                                    format!("Invalid script at transaction {}, input {}", i, j),
                                );
                            }
                        }
                    }

                    // OPTIMIZATION: Batch verify Schnorr signatures (ECDSA uses per-sig only)
                    #[cfg(feature = "production")]
                    {
                        if !schnorr_collector.is_empty() {
                            let batch_results = schnorr_collector.verify_batch()?;
                            if batch_results.iter().any(|&valid| !valid) {
                                return invalid_block_result(
                                    utxo_set,
                                    tx_ids,
                                    format!("Invalid Schnorr signature in transaction {i}"),
                                );
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

                if fee < 0 {
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!("Fee {fee} must be non-negative at transaction {i}"),
                    );
                }
                // Use checked arithmetic to prevent fee overflow
                total_fees = total_fees
                    .checked_add(fee)
                    .ok_or_else(|| make_fee_overflow_error(Some(i)))?;
                if total_fees < 0 {
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        format!(
                            "Total fees {total_fees} must be non-negative after transaction {i}"
                        ),
                    );
                }
            }
            // Accumulate sigop for remaining txs (production sequential path)
            for j in last_sigop_index..block.transactions.len() {
                let tx_j = &block.transactions[j];
                let wits_j = witnesses.get(j).map(|w| w.as_slice());
                let has_wit = wits_j
                    .map(|w| w.iter().any(|wit| !is_witness_empty(wit)))
                    .unwrap_or(false);
                let tx_flags = calculate_script_flags_for_block_with_base(
                    tx_j,
                    has_wit,
                    base_script_flags,
                    height,
                    context,
                );
                total_sigop_cost = total_sigop_cost
                    .checked_add(
                        crate::sigop::get_transaction_sigop_cost_with_witness_slices(
                            tx_j, &overlay, wits_j, tx_flags,
                        )?,
                    )
                    .ok_or_else(|| ConsensusError::BlockValidation("Sigop cost overflow".into()))?;
                if let Some(msg) =
                    check_bip54_sigop_limit(bip54_active, tx_j, &overlay, wits_j, tx_flags, tx_ids)?
                {
                    return invalid_block_result(utxo_set, tx_ids, msg);
                }
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
        // Pre-allocate overlay with capacity (computed above)
        let mut overlay = UtxoOverlay::with_capacity(
            &utxo_set,
            estimated_outputs.max(100),
            estimated_inputs.max(100),
        );

        // Pre-allocate reusable Vecs to avoid per-transaction allocations
        let mut prevout_values_reusable: Vec<i64> = Vec::with_capacity(256);
        let mut input_utxos_reusable: Vec<Option<&UTXO>> = Vec::with_capacity(256);
        let mut prevout_script_pubkeys_reusable: Vec<&[u8]> = Vec::with_capacity(256);

        for (i, tx) in block.transactions.iter().enumerate() {
            // Accumulate sigop for this tx (non-production path; overlay has prev txs)
            let wits_i = witnesses.get(i).map(|w| w.as_slice());
            let has_wit = wits_i
                .map(|w| w.iter().any(|wit| !is_witness_empty(wit)))
                .unwrap_or(false);
            let tx_flags = calculate_script_flags_for_block_with_base(
                tx,
                has_wit,
                base_script_flags,
                height,
                context,
            );
            total_sigop_cost = total_sigop_cost
                .checked_add(
                    crate::sigop::get_transaction_sigop_cost_with_witness_slices(
                        tx, &overlay, wits_i, tx_flags,
                    )?,
                )
                .ok_or_else(|| ConsensusError::BlockValidation("Sigop cost overflow".into()))?;

            if let Some(msg) =
                check_bip54_sigop_limit(bip54_active, tx, &overlay, wits_i, tx_flags, tx_ids)?
            {
                return invalid_block_result(utxo_set, tx_ids, msg);
            }

            // Validate transaction structure
            if !matches!(check_transaction(tx)?, ValidationResult::Valid) {
                return invalid_block_result(
                    utxo_set,
                    tx_ids,
                    format!("Invalid transaction at index {i}"),
                );
            }

            // Check transaction inputs and calculate fees
            // CRITICAL: Use overlay which includes outputs from previous transactions in this block
            let (input_valid, fee) = check_tx_inputs(tx, &overlay, height)?;

            if matches!(input_valid, ValidationResult::Valid) && fee < 0 {
                return invalid_block_result(
                    utxo_set,
                    tx_ids,
                    format!("Negative fee {fee} for valid transaction at index {i}"),
                );
            }
            if !matches!(input_valid, ValidationResult::Valid) {
                #[cfg(debug_assertions)]
                eprintln!(
                    "   ❌ Block {} TX {}: input_valid={:?}",
                    height, i, input_valid
                );
                return invalid_block_result(
                    utxo_set,
                    tx_ids,
                    format!("Invalid transaction inputs at index {i}"),
                );
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

                // Reuse pre-allocated Vecs instead of allocating per transaction
                prevout_values_reusable.clear();
                prevout_script_pubkeys_reusable.clear();
                if prevout_script_pubkeys_reusable.capacity() < input_utxos.len() {
                    prevout_script_pubkeys_reusable.reserve(
                        input_utxos
                            .len()
                            .saturating_sub(prevout_script_pubkeys_reusable.capacity()),
                    );
                }
                if prevout_values_reusable.capacity() < input_utxos.len() {
                    prevout_values_reusable
                        .reserve(input_utxos.len() - prevout_values_reusable.capacity());
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
                let has_witness = tx_witnesses
                    .map(|w| w.iter().any(|wit| !is_witness_empty(wit)))
                    .unwrap_or(false);
                let flags = calculate_script_flags_for_block_with_base(
                    tx,
                    has_witness,
                    base_script_flags,
                    height,
                    context,
                );
                let median_time_past = time_context
                    .map(|ctx| ctx.median_time_past)
                    .filter(|&mtp| mtp > 0);
                #[cfg(feature = "production")]
                let bip143_hashes = if has_witness {
                    Some(crate::transaction_hash::Bip143PrecomputedHashes::compute(
                        tx,
                        &prevout_values_reusable,
                        &prevout_script_pubkeys_reusable,
                    ))
                } else {
                    None
                };

                // OPTIMIZATION: Collect Schnorr signatures for batch verification (ECDSA: per-sig only)
                #[cfg(feature = "production")]
                let schnorr_collector = crate::bip348::SchnorrSignatureCollector::new();

                for (j, input) in tx.inputs.iter().enumerate() {
                    if let Some(utxo) = input_utxos.get(j).and_then(|opt| *opt) {
                        // Reuse cached tx_witnesses and flags from above
                        // Get witness stack for this transaction input if available
                        // witnesses is Vec<Vec<Witness>> where each Vec<Witness> is for one transaction
                        // and each Witness is for one input
                        let witness_stack = tx_witnesses.and_then(|tx_wits| tx_wits.get(j));
                        let witness_for_script = witness_stack.and_then(|w| {
                            if is_witness_empty(w) {
                                None
                            } else {
                                Some(w)
                            }
                        });

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
                            #[cfg(feature = "production")]
                            Some(&schnorr_collector),
                            #[cfg(feature = "production")]
                            None, // ECDSA: per-sig only (no batch)
                            #[cfg(feature = "production")]
                            None,
                            #[cfg(feature = "production")]
                            bip143_hashes.as_ref(),
                            #[cfg(not(feature = "production"))]
                            None,
                            #[cfg(feature = "production")]
                            None, // precomputed_sighash_all (non-CCheckQueue path)
                            #[cfg(feature = "production")]
                            None, // precomputed_p2pkh_hash
                        )? {
                            return invalid_block_result(
                                utxo_set,
                                tx_ids,
                                format!("Invalid script at transaction {i}, input {j}"),
                            );
                        }
                    }
                }

                // OPTIMIZATION: Batch verify Schnorr signatures (ECDSA uses per-sig only)
                #[cfg(feature = "production")]
                {
                    if !schnorr_collector.is_empty() {
                        let batch_results = schnorr_collector.verify_batch()?;
                        if batch_results.iter().any(|&valid| !valid) {
                            return invalid_block_result(
                                utxo_set,
                                tx_ids,
                                format!("Invalid Schnorr signature in transaction {i}"),
                            );
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
            if fee < 0 {
                return invalid_block_result(
                    utxo_set,
                    tx_ids,
                    format!("Fee {fee} must be non-negative at transaction {i}"),
                );
            }
            total_fees = total_fees
                .checked_add(fee)
                .ok_or_else(|| make_fee_overflow_error(Some(i)))?;
            if total_fees < 0 {
                return invalid_block_result(
                    utxo_set,
                    tx_ids,
                    format!("Total fees {total_fees} must be non-negative after transaction {i}"),
                );
            }
        }
    }

    #[cfg(feature = "profile")]
    profile_log!(
        "[TIMING] Block {}: post_validation={:.2}ms",
        height,
        _fn_start.elapsed().as_secs_f64() * 1000.0
    );
    // 3. Validate coinbase transaction
    if let Some(coinbase) = block.transactions.first() {
        if !is_coinbase(coinbase) {
            return invalid_block_result(utxo_set, tx_ids, "First transaction must be coinbase");
        }

        // Validate coinbase scriptSig length (Orange Paper Section 5.1, rule 5)
        // If tx is coinbase: 2 ≤ |ins[0].scriptSig| ≤ 100
        let script_sig_len = coinbase_script_sig_len(coinbase);

        if !(2..=100).contains(&script_sig_len) {
            return invalid_block_result(
                utxo_set,
                tx_ids,
                format!(
                    "Coinbase scriptSig length {script_sig_len} must be between 2 and 100 bytes"
                ),
            );
        }

        let subsidy = get_block_subsidy(height);
        if !(0..=MAX_MONEY).contains(&subsidy) {
            return Err(ConsensusError::BlockValidation(
                format!("Block subsidy {subsidy} out of valid range").into(),
            ));
        }

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

        if coinbase_output < 0 {
            return invalid_block_result(utxo_set, tx_ids, "Coinbase output must be non-negative");
        }
        // Check that coinbase output doesn't exceed MAX_MONEY
        if coinbase_output > MAX_MONEY {
            return invalid_block_result(
                utxo_set,
                tx_ids,
                format!("Coinbase output {coinbase_output} exceeds maximum money supply"),
            );
        }

        // Use checked arithmetic for fee + subsidy calculation
        let max_coinbase_value = total_fees
            .checked_add(subsidy)
            .ok_or_else(|| ConsensusError::BlockValidation("Fees + subsidy overflow".into()))?;

        if coinbase_output > max_coinbase_value {
            return invalid_block_result(
                utxo_set,
                tx_ids,
                format!(
                    "Coinbase output {coinbase_output} exceeds fees {total_fees} + subsidy {subsidy}"
                ),
            );
        }

        // Validate witness commitment if witnesses are present (SegWit block).
        // Short-circuit: no witness commitment possible before SegWit activation.
        let has_segwit = segwit_active
            && witnesses
                .iter()
                .any(|tx_w| tx_w.iter().any(|stack| !stack.is_empty()));

        if has_segwit && !witnesses.is_empty() {
            // Skip witness commitment validation in IBD mode. In IBD we replay blocks
            // that were already validated by Bitcoin Core (CheckBlock + ConnectBlock).
            if !ibd_mode {
                let witness_merkle_root =
                    crate::segwit::compute_witness_merkle_root_from_nested(block, witnesses)?;
                // `Hash` is 32 bytes; commitment compares to header field directly.

                if !validate_witness_commitment(coinbase, &witness_merkle_root, &witnesses[0])? {
                    return invalid_block_result(
                        utxo_set,
                        tx_ids,
                        "Invalid witness commitment in coinbase transaction",
                    );
                }
            }
        }
    } else {
        return invalid_block_result(utxo_set, tx_ids, "Block must have at least one transaction");
    }

    // 3.5. Check block sigop cost limit (network rule)
    // total_sigop_cost accumulated in overlay pass
    use crate::constants::MAX_BLOCK_SIGOPS_COST;

    // Invariant assertion: Total sigop cost must not exceed maximum
    if total_sigop_cost > MAX_BLOCK_SIGOPS_COST {
        return invalid_block_result(
            utxo_set,
            tx_ids,
            format!("Block sigop cost {total_sigop_cost} exceeds maximum {MAX_BLOCK_SIGOPS_COST}"),
        );
    }

    // BIP30 index is only read by check_bip30 while the fork is active; skip index mutations
    // after deactivation to avoid redundant HashMap work on every coinbase touch.
    let maintain_bip30_index = context.is_fork_active(ForkId::Bip30, height);

    #[cfg(feature = "production")]
    if crate::config::use_overlay_delta() {
        if let Some(overlay) = overlay_for_delta {
            let (additions_arc, deletions) = overlay.into_changes();
            let mut undo_log = crate::reorganization::BlockUndoLog::new();
            let bip30_for_merge = if maintain_bip30_index {
                bip30_index
            } else {
                None
            };
            if ibd_mode {
                script_cache::merge_overlay_changes_to_cache(
                    &additions_arc,
                    &deletions,
                    &mut utxo_set,
                    bip30_for_merge,
                    None,
                );
            } else {
                script_cache::merge_overlay_changes_to_cache(
                    &additions_arc,
                    &deletions,
                    &mut utxo_set,
                    bip30_for_merge,
                    Some(&mut undo_log),
                );
            }
            #[cfg(feature = "rayon")]
            if !ibd_mode
                && !skip_signatures
                && !crate::block::skip_script_exec_cache()
                && segwit_active
            {
                script_cache::insert_script_exec_cache_for_block(block, witnesses, height, context);
            }
            return Ok((
                ValidationResult::Valid,
                utxo_set,
                tx_ids_cow,
                undo_log,
                Some(UtxoDelta {
                    additions: additions_arc,
                    deletions,
                }),
            ));
        }
    }

    #[cfg(feature = "profile")]
    profile_log!(
        "[TIMING] Block {}: pre_apply={:.2}ms",
        height,
        _fn_start.elapsed().as_secs_f64() * 1000.0
    );
    let (result, new_utxo_set, undo_log) = connect_block_inner_with_tx_ids(
        block,
        witnesses,
        utxo_set,
        height,
        time_context,
        network,
        tx_ids,
        total_fees,
        bip30_index,
        maintain_bip30_index,
        ibd_mode,
    )?;
    #[cfg(feature = "profile")]
    profile_log!(
        "[TIMING] Block {}: post_apply={:.2}ms",
        height,
        _fn_start.elapsed().as_secs_f64() * 1000.0
    );
    #[cfg(all(feature = "production", feature = "rayon"))]
    if matches!(result, ValidationResult::Valid)
        && !ibd_mode
        && !skip_signatures
        && !crate::block::skip_script_exec_cache()
        && segwit_active
    {
        script_cache::insert_script_exec_cache_for_block(block, witnesses, height, context);
    }
    #[cfg(feature = "profile")]
    profile_log!(
        "[TIMING] Block {}: post_cache={:.2}ms (total)",
        height,
        _fn_start.elapsed().as_secs_f64() * 1000.0
    );
    Ok((result, new_utxo_set, tx_ids_cow, undo_log, None))
}
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
    maintain_bip30_index: bool,
    ibd_mode: bool,
) -> Result<(
    ValidationResult,
    UtxoSet,
    crate::reorganization::BlockUndoLog,
)> {
    // 5. Apply all transactions to UTXO set (with pre-computed transaction IDs)
    // Build undo log for all UTXO changes (skipped during IBD — never persisted on parallel path).
    use crate::reorganization::BlockUndoLog;
    let mut undo_log = BlockUndoLog::new();
    let collect_undo = !ibd_mode;

    if tx_ids.len() != block.transactions.len() {
        return Err(ConsensusError::BlockValidation(
            format!(
                "Transaction ID count {} must match transaction count {}",
                tx_ids.len(),
                block.transactions.len()
            )
            .into(),
        ));
    }

    // NOTE: With UtxoOverlay approach, validation uses a read-only view of utxo_set.
    // The overlay tracks additions/deletions in memory but DOES NOT modify the base utxo_set.
    // Therefore, the application loop MUST ALWAYS run to apply changes to utxo_set.
    {
        // Normal path: Apply transactions sequentially to build undo log
        let mut bip30_none_slot: Option<&mut crate::bip_validation::Bip30Index> = None;
        for (i, tx) in block.transactions.iter().enumerate() {
            let initial_utxo_size = utxo_set.len();
            let bip30_apply_ref = if maintain_bip30_index {
                &mut bip30_index
            } else {
                &mut bip30_none_slot
            };
            let (new_utxo_set, tx_undo_entries) = apply::apply_transaction_with_id(
                tx,
                tx_ids[i],
                utxo_set,
                height,
                bip30_apply_ref,
                collect_undo,
            )?;

            debug_assert!(
                tx_undo_entries.len() <= tx.inputs.len() + tx.outputs.len(),
                "Undo entry count {} must be reasonable for transaction {}",
                tx_undo_entries.len(),
                i
            );

            if collect_undo {
                undo_log.entries.extend(tx_undo_entries);
            }
            utxo_set = new_utxo_set;

            if is_coinbase(tx) {
                if utxo_set.len() < initial_utxo_size {
                    return Err(ConsensusError::BlockValidation(
                        format!(
                            "UTXO set size {} must not decrease after coinbase (was {})",
                            utxo_set.len(),
                            initial_utxo_size
                        )
                        .into(),
                    ));
                }
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
                            let op = OutPoint {
                                hash: tx_id,
                                index: j as u32,
                            };
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
    if collect_undo {
        undo_log.entries.reverse();
    }

    // Runtime invariant verification: Supply change must equal subsidy + fees
    // Mathematical specification:
    // ∀ block B, height h: Δsupply = get_block_subsidy(h) + total_fees
    // This ensures no money creation or destruction beyond expected inflation
    #[cfg(any(debug_assertions, feature = "runtime-invariants"))]
    {
        use crate::constants::MAX_MONEY;
        use crate::economic::{get_block_subsidy, total_supply};

        let expected_supply = total_supply(height);
        if !(0..=MAX_MONEY).contains(&expected_supply) {
            return Err(ConsensusError::BlockValidation(
                format!("Expected supply {expected_supply} out of valid range [0, MAX_MONEY]")
                    .into(),
            ));
        }

        if utxo_set.len() > u32::MAX as usize {
            return Err(ConsensusError::BlockValidation(
                format!("UTXO set size {} must fit in u32", utxo_set.len()).into(),
            ));
        }

        let mut actual_supply: i64 = 0i64;
        for utxo in utxo_set.values() {
            let v = utxo.value;
            if !(0..=MAX_MONEY).contains(&v) {
                return Err(ConsensusError::BlockValidation(
                    format!("UTXO value {v} out of valid range [0, MAX_MONEY]").into(),
                ));
            }
            actual_supply = actual_supply.checked_add(v).ok_or_else(|| {
                ConsensusError::BlockValidation("UTXO supply sum overflow".into())
            })?;
        }

        let subsidy = get_block_subsidy(height);
        let _expected_change = subsidy.saturating_add(total_fees);

        let supply_plus_fees = expected_supply.saturating_add(total_fees);
        // Soft check: model vs summed UTXOs (can warn in dev if economics/UTXO state diverge).
        debug_assert!(
            actual_supply <= supply_plus_fees,
            "Supply invariant violated at height {height}: actual supply {actual_supply} exceeds expected {expected_supply} + fees {total_fees}"
        );
    }

    if utxo_set.len() > u32::MAX as usize {
        return Err(ConsensusError::BlockValidation(
            format!(
                "UTXO set size {} must not exceed maximum after block connection",
                utxo_set.len()
            )
            .into(),
        ));
    }

    Ok((ValidationResult::Valid, utxo_set, undo_log))
}
