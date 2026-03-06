//! Core-style script execution cache: skip all script checks for a tx when (witness_hash, flags) is cached.
//!
//! Key = SHA256(witness_hash || flags) where witness_hash = SHA256d(serialize_transaction_with_witness).
//! Hit rate ~0 during IBD (each tx validated once); helps reorgs and mempool revalidation.
//! Enable with BLVM_SCRIPT_EXEC_CACHE=1.

#![cfg(all(feature = "production", feature = "rayon"))]

use crate::serialization::serialize_transaction_with_witness;
use crate::types::Transaction;
use bitcoin_hashes::{sha256d, Hash as BitcoinHash, HashEngine};
use std::sync::OnceLock;

/// Cache capacity (entries). Core uses ~16MB for script exec; we use 64k entries (~2MB for keys).
const CACHE_CAPACITY: usize = 65536;

static CACHE: OnceLock<std::sync::Mutex<lru::LruCache<[u8; 32], ()>>> = OnceLock::new();

fn get_cache() -> Option<&'static std::sync::Mutex<lru::LruCache<[u8; 32], ()>>> {
    if std::env::var("BLVM_SCRIPT_EXEC_CACHE").map(|s| s == "1" || s.eq_ignore_ascii_case("true")).unwrap_or(false) {
        Some(CACHE.get_or_init(|| std::sync::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(CACHE_CAPACITY).unwrap()))))
    } else {
        None
    }
}

/// Compute cache key: SHA256(witness_hash || flags_le) where witness_hash = SHA256d(serialize_tx_with_witness).
pub fn compute_key(tx: &Transaction, witnesses: &[crate::witness::Witness], flags: u32) -> [u8; 32] {
    let bytes = serialize_transaction_with_witness(tx, witnesses);
    let witness_hash = sha256d::Hash::hash(&bytes);
    let mut hasher = bitcoin_hashes::sha256::Hash::engine();
    hasher.input(&witness_hash);
    hasher.input(&flags.to_le_bytes());
    let result = bitcoin_hashes::sha256::Hash::from_engine(hasher);
    let mut key = [0u8; 32];
    key.copy_from_slice(&result);
    key
}

/// If cache enabled and hit, return true. Otherwise false.
pub fn contains(key: &[u8; 32]) -> bool {
    if let Some(cache) = get_cache() {
        if let Ok(mut c) = cache.lock() {
            return c.get(key).is_some();
        }
    }
    false
}

/// Insert key into cache (call after block validation passes).
pub fn insert(key: &[u8; 32]) {
    if let Some(cache) = get_cache() {
        if let Ok(mut c) = cache.lock() {
            c.put(*key, ());
        }
    }
}
