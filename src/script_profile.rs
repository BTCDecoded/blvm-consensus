//! Script sub-timing for IBD profiling (sighash, interpreter, multisig, P2PKH fast path).
//! Used by block.rs to extend [PERF] with breakdown when profile feature is enabled.

#![cfg(all(feature = "production", feature = "profile"))]

use std::sync::atomic::{AtomicU64, Ordering};

static SCRIPT_SIGHASH_NS: AtomicU64 = AtomicU64::new(0);
static SCRIPT_INTERPRETER_NS: AtomicU64 = AtomicU64::new(0);
static SCRIPT_MULTISIG_NS: AtomicU64 = AtomicU64::new(0);
static SCRIPT_P2PKH_PARSE_NS: AtomicU64 = AtomicU64::new(0);
static SCRIPT_P2PKH_HASH160_NS: AtomicU64 = AtomicU64::new(0);
static SCRIPT_P2PKH_COLLECT_NS: AtomicU64 = AtomicU64::new(0);
/// Time from collect() entry to before lock (fetch_add, shard_idx)
static COLLECT_SLOT_NS: AtomicU64 = AtomicU64::new(0);
/// Time waiting for EcdsaSoA shard lock (contention indicator)
static COLLECT_SHARD_LOCK_NS: AtomicU64 = AtomicU64::new(0);
/// Time copying into shard after lock acquired
static COLLECT_COPY_NS: AtomicU64 = AtomicU64::new(0);
/// Time in chunk_ready logic (fetch_add, ready_queue push) after lock released
static COLLECT_CHUNK_NS: AtomicU64 = AtomicU64::new(0);
/// Worker: build_p2pkh_hash_map (batch)
static WORKER_P2PKH_MAP_NS: AtomicU64 = AtomicU64::new(0);
/// Worker: refs extraction (batch)
static WORKER_REFS_NS: AtomicU64 = AtomicU64::new(0);
/// Worker: time to acquire tx_contexts, buffer, spi, pv read locks (contention indicator)
static WORKER_REFS_LOCK_NS: AtomicU64 = AtomicU64::new(0);
/// Worker: run_check_with_refs loop (per-check work, excludes refs + p2pkh_map)
static WORKER_RUN_CHECK_LOOP_NS: AtomicU64 = AtomicU64::new(0);
/// Worker: results.push() time (per batch; SegQueue, lock-free)
static WORKER_RESULTS_EXTEND_NS: AtomicU64 = AtomicU64::new(0);
/// Caller: entry to try_verify_p2pkh_fast_path (before parse)
static P2PKH_FAST_PATH_ENTRY_NS: AtomicU64 = AtomicU64::new(0);
/// Caller: BIP66 check (check_bip66)
static P2PKH_BIP66_NS: AtomicU64 = AtomicU64::new(0);
/// Caller: SECP256K1_CONTEXT.with closure (verify_signature body)
static P2PKH_SECP_CONTEXT_NS: AtomicU64 = AtomicU64::new(0);
/// Batch phase: SoA extraction (parse raw → compact, cache lookup)
static BATCH_SOA_EXTRACT_NS: AtomicU64 = AtomicU64::new(0);
/// Batch phase: secp256k1 batch verify
static BATCH_SECP_VERIFY_NS: AtomicU64 = AtomicU64::new(0);
/// Batch phase: sig cache writes after verify
static BATCH_CACHE_WRITE_NS: AtomicU64 = AtomicU64::new(0);
/// Drain threads: copy from shards under lock
static DRAIN_SHARD_COPY_NS: AtomicU64 = AtomicU64::new(0);
/// Drain threads: parse_raw_to_compact + cache lookup (outside lock)
static DRAIN_PARSE_NS: AtomicU64 = AtomicU64::new(0);
/// Drain threads: secp256k1 verify_batch_from_raw
static DRAIN_SECP_NS: AtomicU64 = AtomicU64::new(0);
/// ECDSA sig cache hits (per block)
static ECDSA_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
/// ECDSA sig cache misses (per block)
static ECDSA_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

#[inline(always)]
pub fn add_sighash_ns(ns: u64) {
    SCRIPT_SIGHASH_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_interpreter_ns(ns: u64) {
    SCRIPT_INTERPRETER_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_multisig_ns(ns: u64) {
    SCRIPT_MULTISIG_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_p2pkh_parse_ns(ns: u64) {
    SCRIPT_P2PKH_PARSE_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_p2pkh_hash160_ns(ns: u64) {
    SCRIPT_P2PKH_HASH160_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_p2pkh_collect_ns(ns: u64) {
    SCRIPT_P2PKH_COLLECT_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_collect_slot_ns(ns: u64) {
    COLLECT_SLOT_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_collect_shard_lock_ns(ns: u64) {
    COLLECT_SHARD_LOCK_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_collect_copy_ns(ns: u64) {
    COLLECT_COPY_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_collect_chunk_ns(ns: u64) {
    COLLECT_CHUNK_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_worker_p2pkh_map_ns(ns: u64) {
    WORKER_P2PKH_MAP_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_worker_refs_ns(ns: u64) {
    WORKER_REFS_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_worker_refs_lock_ns(ns: u64) {
    WORKER_REFS_LOCK_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_worker_run_check_loop_ns(ns: u64) {
    WORKER_RUN_CHECK_LOOP_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_worker_results_extend_ns(ns: u64) {
    WORKER_RESULTS_EXTEND_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_p2pkh_fast_path_entry_ns(ns: u64) {
    P2PKH_FAST_PATH_ENTRY_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_p2pkh_bip66_ns(ns: u64) {
    P2PKH_BIP66_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_p2pkh_secp_context_ns(ns: u64) {
    P2PKH_SECP_CONTEXT_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_batch_soa_extract_ns(ns: u64) {
    BATCH_SOA_EXTRACT_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_batch_secp_verify_ns(ns: u64) {
    BATCH_SECP_VERIFY_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_batch_cache_write_ns(ns: u64) {
    BATCH_CACHE_WRITE_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_drain_shard_copy_ns(ns: u64) {
    DRAIN_SHARD_COPY_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_drain_parse_ns(ns: u64) {
    DRAIN_PARSE_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_drain_secp_ns(ns: u64) {
    DRAIN_SECP_NS.fetch_add(ns, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_ecdsa_cache_hit() {
    ECDSA_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
}

#[inline(always)]
pub fn add_ecdsa_cache_miss() {
    ECDSA_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

pub fn get_and_reset_drain_timing() -> (u64, u64, u64) {
    (
        DRAIN_SHARD_COPY_NS.swap(0, Ordering::Relaxed),
        DRAIN_PARSE_NS.swap(0, Ordering::Relaxed),
        DRAIN_SECP_NS.swap(0, Ordering::Relaxed),
    )
}

pub fn get_and_reset_ecdsa_cache_stats() -> (u64, u64) {
    (
        ECDSA_CACHE_HITS.swap(0, Ordering::Relaxed),
        ECDSA_CACHE_MISSES.swap(0, Ordering::Relaxed),
    )
}

pub fn get_and_reset_script_sub_timing() -> (u64, u64, u64) {
    (
        SCRIPT_SIGHASH_NS.swap(0, Ordering::Relaxed),
        SCRIPT_INTERPRETER_NS.swap(0, Ordering::Relaxed),
        SCRIPT_MULTISIG_NS.swap(0, Ordering::Relaxed),
    )
}

pub fn get_and_reset_p2pkh_timing() -> (u64, u64, u64, u64, u64, u64) {
    (
        SCRIPT_P2PKH_PARSE_NS.swap(0, Ordering::Relaxed),
        SCRIPT_P2PKH_HASH160_NS.swap(0, Ordering::Relaxed),
        SCRIPT_P2PKH_COLLECT_NS.swap(0, Ordering::Relaxed),
        P2PKH_FAST_PATH_ENTRY_NS.swap(0, Ordering::Relaxed),
        P2PKH_BIP66_NS.swap(0, Ordering::Relaxed),
        P2PKH_SECP_CONTEXT_NS.swap(0, Ordering::Relaxed),
    )
}

pub fn get_and_reset_collect_timing() -> (u64, u64, u64, u64) {
    (
        COLLECT_SLOT_NS.swap(0, Ordering::Relaxed),
        COLLECT_SHARD_LOCK_NS.swap(0, Ordering::Relaxed),
        COLLECT_COPY_NS.swap(0, Ordering::Relaxed),
        COLLECT_CHUNK_NS.swap(0, Ordering::Relaxed),
    )
}

pub fn get_and_reset_worker_timing() -> (u64, u64, u64, u64, u64) {
    (
        WORKER_P2PKH_MAP_NS.swap(0, Ordering::Relaxed),
        WORKER_REFS_NS.swap(0, Ordering::Relaxed),
        WORKER_REFS_LOCK_NS.swap(0, Ordering::Relaxed),
        WORKER_RUN_CHECK_LOOP_NS.swap(0, Ordering::Relaxed),
        WORKER_RESULTS_EXTEND_NS.swap(0, Ordering::Relaxed),
    )
}

pub fn get_and_reset_batch_phase_timing() -> (u64, u64, u64) {
    (
        BATCH_SOA_EXTRACT_NS.swap(0, Ordering::Relaxed),
        BATCH_SECP_VERIFY_NS.swap(0, Ordering::Relaxed),
        BATCH_CACHE_WRITE_NS.swap(0, Ordering::Relaxed),
    )
}
