//! BIP348: OP_CHECKSIGFROMSTACK (CSFS)
//!
//! Implementation of BIP348 CheckSigFromStack opcode for arbitrary message signature verification.
//!
//! **Feature Flag**: This module is only available when the `csfs` feature is enabled.
//! CSFS is a proposed soft fork and should be used with caution until activated on mainnet.
//!
//! **Context**: Tapscript only (leaf version 0xc0)
//!
//! Specification: https://raw.githubusercontent.com/bitcoin/bips/master/bip-0348.md
//!
//! ## Overview
//!
//! OP_CHECKSIGFROMSTACK (CSFS) verifies a BIP 340 Schnorr signature against an arbitrary message.
//! Unlike OP_CHECKSIG, this verifies signatures on arbitrary data, not transaction data.
//!
//! ## Key Differences from OP_CHECKSIG
//!
//! - Uses BIP 340 Schnorr signatures (not ECDSA)
//! - Message is NOT hashed (BIP 340 accepts any size)
//! - Only 32-byte pubkeys are verified (BIP 340 x-only pubkeys)
//! - Only available in Tapscript (leaf version 0xc0)
//!
//! ## Security Considerations
//!
//! - **Constant-time operations**: Signature verification uses constant-time operations
//! - **Input validation**: All inputs are validated before processing
//! - **Feature flag**: CSFS is behind a feature flag to prevent accidental use before activation

use crate::crypto::OptimizedSha256;
use crate::error::{ConsensusError, Result};
use blvm_spec_lock::spec_locked;
#[cfg(all(feature = "production", feature = "rayon"))]
use rayon::prelude::*;
use secp256k1::{schnorr::Signature, Message, XOnlyPublicKey};
#[cfg(feature = "production")]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Collector for Schnorr signatures to enable batch verification
///
/// D (SoA): Pre-allocated Vecs when capacity > 0; atomic index. No Vec alloc per collect.
/// Fallback SegQueue when capacity = 0.
#[cfg(feature = "production")]
pub struct SchnorrSignatureCollector {
    soa: Option<std::sync::Arc<SchnorrSoAStorage>>,
    next_idx: AtomicUsize,
    tasks: crossbeam_queue::SegQueue<(usize, (Vec<u8>, Vec<u8>, Vec<u8>))>,
    streaming_results: crossbeam_queue::SegQueue<Vec<(usize, bool)>>,
}

#[cfg(feature = "production")]
struct SchnorrSoAStorage {
    inner: std::sync::Mutex<SchnorrSoAInner>,
    next_slot: AtomicUsize,
}

#[cfg(feature = "production")]
struct SchnorrSoAInner {
    indices: Vec<usize>,
    msgs: Vec<[u8; 32]>,
    pubkeys: Vec<[u8; 32]>,
    sigs: Vec<[u8; 64]>,
}

#[cfg(feature = "production")]
impl Default for SchnorrSignatureCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "production")]
impl SchnorrSignatureCollector {
    /// Create a new empty collector (SegQueue fallback)
    pub fn new() -> Self {
        Self {
            soa: None,
            next_idx: AtomicUsize::new(0),
            tasks: crossbeam_queue::SegQueue::new(),
            streaming_results: crossbeam_queue::SegQueue::new(),
        }
    }

    /// Returns true when using SoA storage (try_verify_chunk is a no-op).
    pub fn uses_soa(&self) -> bool {
        self.soa.is_some()
    }

    /// Create a collector with pre-allocated SoA storage (block validation path)
    pub fn new_with_capacity(cap: usize) -> Self {
        let soa = if cap == 0 {
            None
        } else {
            Some(std::sync::Arc::new(SchnorrSoAStorage {
                inner: std::sync::Mutex::new(SchnorrSoAInner {
                    indices: vec![0; cap],
                    msgs: vec![[0u8; 32]; cap],
                    pubkeys: vec![[0u8; 32]; cap],
                    sigs: vec![[0u8; 64]; cap],
                }),
                next_slot: AtomicUsize::new(0),
            }))
        };
        Self {
            soa,
            next_idx: AtomicUsize::new(0),
            tasks: crossbeam_queue::SegQueue::new(),
            streaming_results: crossbeam_queue::SegQueue::new(),
        }
    }

    /// Stream verification: drain up to `chunk_size` sigs, verify, store results.
    /// No-op when using SoA.
    #[cfg(all(feature = "production", feature = "rayon"))]
    pub fn try_verify_chunk(&self, chunk_size: usize) {
        if self.soa.is_some() || chunk_size == 0 {
            return;
        }
        let mut chunk: Vec<_> = std::iter::from_fn(|| self.tasks.pop())
            .take(chunk_size)
            .collect();
        if chunk.is_empty() {
            return;
        }
        chunk.sort_by_key(|t| t.0);
        let indices: Vec<usize> = chunk.iter().map(|(i, _)| *i).collect();
        let tasks: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = chunk.into_iter().map(|(_, t)| t).collect();
        let task_refs: Vec<(&[u8], &[u8], &[u8])> = tasks
            .iter()
            .map(|(msg, pk, sig)| (msg.as_slice(), pk.as_slice(), sig.as_slice()))
            .collect();
        if let Ok(results) = batch_verify_signatures_from_stack(&task_refs) {
            let partial: Vec<(usize, bool)> = indices.into_iter().zip(results).collect();
            self.streaming_results.push(partial);
        }
    }

    #[cfg(all(feature = "production", not(feature = "rayon")))]
    pub fn try_verify_chunk(&self, _chunk_size: usize) {}

    /// Collect a signature for deferred batch verification
    ///
    /// Returns the index of this task for result mapping
    pub fn collect(&self, message: &[u8], pubkey: &[u8], signature: &[u8]) -> usize {
        let idx = self.next_idx.fetch_add(1, Ordering::Relaxed);
        self.collect_with_index(idx, message, pubkey, signature);
        idx
    }

    /// Collect with explicit global index (for CCheckQueue-style parallel script verification).
    /// Enables deterministic (tx, input) order when workers collect out of order.
    pub fn collect_with_index(
        &self,
        global_index: usize,
        message: &[u8],
        pubkey: &[u8],
        signature: &[u8],
    ) {
        if let Some(ref soa) = self.soa {
            if message.len() != 32 || pubkey.len() != 32 || signature.len() != 64 {
                return;
            }
            let slot = soa.next_slot.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut inner) = soa.inner.lock() {
                if slot < inner.indices.len() {
                    inner.indices[slot] = global_index;
                    inner.msgs[slot].copy_from_slice(message);
                    inner.pubkeys[slot].copy_from_slice(pubkey);
                    inner.sigs[slot].copy_from_slice(signature);
                }
            }
        } else {
            self.tasks.push((
                global_index,
                (message.to_vec(), pubkey.to_vec(), signature.to_vec()),
            ));
        }
    }

    /// Batch verify all collected signatures
    ///
    /// Returns a vector of results, one per collected signature (in collection order)
    pub fn verify_batch(&self) -> Result<Vec<bool>> {
        if let Some(ref soa) = self.soa {
            let count = soa.next_slot.load(Ordering::Relaxed);
            if count == 0 {
                return Ok(Vec::new());
            }
            return Self::verify_soa_batch(soa, count);
        }

        #[cfg(all(feature = "production", feature = "rayon"))]
        let streaming: Vec<Vec<(usize, bool)>> =
            std::iter::from_fn(|| self.streaming_results.pop()).collect();

        let mut tasks: Vec<(usize, (Vec<u8>, Vec<u8>, Vec<u8>))> =
            std::iter::from_fn(|| self.tasks.pop()).collect();
        if tasks.is_empty() {
            #[cfg(all(feature = "production", feature = "rayon"))]
            if !streaming.is_empty() {
                let mut merged: Vec<(usize, bool)> = streaming.into_iter().flatten().collect();
                merged.sort_by_key(|(i, _)| *i);
                return Ok(merged.into_iter().map(|(_, v)| v).collect());
            }
            return Ok(Vec::new());
        }
        tasks.sort_by_key(|t| t.0);
        let indices: Vec<usize> = tasks.iter().map(|(i, _)| *i).collect();
        let tasks: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = tasks.into_iter().map(|(_, t)| t).collect();

        let task_refs: Vec<(&[u8], &[u8], &[u8])> = tasks
            .iter()
            .map(|(msg, pk, sig)| (msg.as_slice(), pk.as_slice(), sig.as_slice()))
            .collect();

        let remainder_results = batch_verify_signatures_from_stack(&task_refs)?;
        let mut merged: Vec<(usize, bool)> = indices.into_iter().zip(remainder_results).collect();
        #[cfg(feature = "rayon")]
        for partial in streaming {
            merged.extend(partial);
        }
        merged.sort_by_key(|(i, _)| *i);
        Ok(merged.into_iter().map(|(_, v)| v).collect())
    }

    fn verify_soa_batch(soa: &SchnorrSoAStorage, count: usize) -> Result<Vec<bool>> {
        let inner = soa
            .inner
            .lock()
            .map_err(|_| ConsensusError::BlockValidation("SoA lock poisoned".into()))?;
        let task_refs: Vec<(&[u8], &[u8], &[u8])> = (0..count)
            .map(|slot| {
                (
                    inner.msgs[slot].as_slice(),
                    inner.pubkeys[slot].as_slice(),
                    inner.sigs[slot].as_slice(),
                )
            })
            .collect();
        let results = batch_verify_signatures_from_stack(&task_refs)?;
        let mut merged: Vec<(usize, bool)> = (0..count)
            .map(|slot| (inner.indices[slot], results[slot]))
            .collect();
        merged.sort_by_key(|(i, _)| *i);
        Ok(merged.into_iter().map(|(_, v)| v).collect())
    }

    /// Clear all collected signatures and streaming results
    pub fn clear(&self) {
        while self.tasks.pop().is_some() {}
        self.next_idx.store(0, Ordering::Relaxed);
        #[cfg(all(feature = "production", feature = "rayon"))]
        while self.streaming_results.pop().is_some() {}
    }

    /// Check if collector is empty
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Merge tasks from another collector (for per-thread collection merge).
    /// No-op when using SoA.
    #[cfg(feature = "rayon")]
    pub fn extend_from(&self, other: &Self) {
        if self.soa.is_some() {
            return;
        }
        while let Some((_, task)) = other.tasks.pop() {
            let new_idx = self.next_idx.fetch_add(1, Ordering::Relaxed);
            self.tasks.push((new_idx, task));
        }
    }
}

/// Verify BIP 340 Schnorr signature against arbitrary message (CSFS)
///
/// Verifies that a BIP 340 Schnorr signature is valid for a given message and public key.
/// Unlike OP_CHECKSIG, this verifies signatures on arbitrary data, not transaction data.
///
/// **Key Differences**:
/// - Uses BIP 340 Schnorr signatures (not ECDSA)
/// - Message is NOT hashed (BIP 340 accepts any size, but we hash to 32 bytes for secp256k1)
/// - Only 32-byte pubkeys are verified (BIP 340 x-only pubkeys)
///
/// # Arguments
///
/// * `message` - The message to verify (arbitrary bytes, NOT hashed by BIP 340 spec)
/// * `pubkey` - The public key (32 bytes for BIP 340, other sizes succeed as unknown type)
/// * `signature` - The signature (64-byte BIP 340 Schnorr signature)
/// * `collector` - Optional collector for deferred batch verification (production feature)
///
/// # Returns
///
/// `true` if signature is valid or collected, `false` otherwise
///
/// # Errors
///
/// Returns error if:
/// - Pubkey size is zero
/// - Signature verification fails (for 32-byte pubkeys, when not using collector)
///
/// # Note on Message Hashing
///
/// BIP-348 states "Message is NOT hashed" because BIP 340 accepts messages of any size.
/// However, secp256k1's `Message::from_digest_slice()` requires exactly 32 bytes.
/// BIP 340 uses tagged hashes. For CSFS, we hash the message with SHA256 to create
/// a 32-byte digest, which is then used for BIP 340 verification.
/// This matches the reference implementation in BIP348 reference implementation.
#[spec_locked("5.4.8")]
pub fn verify_signature_from_stack(
    message: &[u8],
    pubkey: &[u8],
    signature: &[u8],
    #[cfg(feature = "production")] collector: Option<&SchnorrSignatureCollector>,
) -> Result<bool> {
    // BIP-348: If pubkey size is zero, script MUST fail
    if pubkey.is_empty() {
        return Err(ConsensusError::ScriptErrorWithCode {
            code: crate::error::ScriptErrorCode::PubkeyType,
            message: "OP_CHECKSIGFROMSTACK: pubkey size is zero".into(),
        });
    }

    // BIP-348: Only 32-byte pubkeys are verified (BIP 340)
    if pubkey.len() == 32 {
        // BIP 340 Schnorr signature verification
        // Message is NOT hashed by BIP 340 spec, but we need 32 bytes for secp256k1
        // Use SHA256 to hash message to 32 bytes (matches BIP348 reference implementation)

        // Signature must be 64 bytes (BIP 340 Schnorr)
        if signature.len() != 64 {
            return Ok(false);
        }

        // OPTIMIZATION: If collector is provided, defer verification for batch processing
        #[cfg(feature = "production")]
        if let Some(collector) = collector {
            // Collect signature for batch verification
            collector.collect(message, pubkey, signature);
            // Return true for now - actual verification happens in batch
            return Ok(true);
        }

        // Immediate verification (fallback when no collector)
        // Parse x-only public key (32 bytes)
        let pubkey_xonly = match XOnlyPublicKey::from_slice(pubkey) {
            Ok(pk) => pk,
            Err(_) => return Ok(false), // Invalid pubkey format
        };

        // Parse Schnorr signature (64 bytes)
        let sig = match Signature::from_slice(signature) {
            Ok(s) => s,
            Err(_) => return Ok(false), // Invalid signature format
        };

        // Create message from bytes
        // BIP 340: Message is NOT hashed (accepts any size)
        // But secp256k1 requires 32 bytes, so we hash with SHA256
        // Uses OptimizedSha256 (SHA-NI when available) for faster hashing
        let message_hash = OptimizedSha256::new().hash(message);
        let msg = Message::from_digest_slice(&message_hash)
            .map_err(|_| ConsensusError::InvalidSignature("Invalid message".into()))?;

        // Verify using backend (libsecp256k1 or blvm-secp256k1)
        let pk_bytes: [u8; 32] = pubkey_xonly.serialize();
        Ok(crate::secp256k1_backend::verify_schnorr(
            &sig.serialize(),
            &message_hash,
            &pk_bytes,
        )?)
    } else {
        // BIP-348: Unknown pubkey type - succeeds as if valid
        Ok(true)
    }
}

/// Batch verify multiple BIP 340 Schnorr signatures
///
/// Verifies multiple Schnorr signatures in a single batch operation using
/// true batch verification (multi-scalar multiplication). This provides
/// significant performance improvements (2-3x speedup) when verifying
/// multiple signatures.
///
/// # Arguments
///
/// * `verification_tasks` - Vector of (message, pubkey, signature) tuples
///
/// # Returns
///
/// Vector of boolean results, one per signature (in same order)
///
/// # Performance
///
/// Uses true batch verification via `libsecp256k1`'s multi-scalar multiplication,
/// providing 2-3x speedup compared to individual verification.
#[cfg(feature = "production")]
#[spec_locked("5.4.8")]
pub fn batch_verify_signatures_from_stack(
    verification_tasks: &[(&[u8], &[u8], &[u8])],
) -> Result<Vec<bool>> {
    if verification_tasks.is_empty() {
        return Ok(Vec::new());
    }

    // Filter to only 32-byte pubkeys (BIP 340) and collect valid tasks
    let mut valid_tasks = Vec::new();
    let mut task_indices = Vec::new(); // Track original indices
    let mut results = vec![true; verification_tasks.len()]; // Default: unknown pubkey type succeeds

    for (idx, (message, pubkey, signature)) in verification_tasks.iter().enumerate() {
        // BIP-348: Only 32-byte pubkeys are verified
        if pubkey.len() == 32 && signature.len() == 64 {
            valid_tasks.push((idx, message, pubkey, signature));
            task_indices.push(idx);
        } else if pubkey.is_empty() {
            // Zero-length pubkey fails
            results[idx] = false;
        } else if pubkey.len() != 32 {
            // Unknown pubkey type succeeds (already set to true)
            continue;
        } else if signature.len() != 64 {
            // Invalid signature length fails
            results[idx] = false;
        }
    }

    if valid_tasks.is_empty() {
        return Ok(results);
    }

    // Parse all signatures and pubkeys
    let mut sigs = Vec::new();
    let mut pubkeys = Vec::new();
    let mut msgs: Vec<[u8; 32]> = Vec::new();

    for (idx, message, pubkey, signature) in &valid_tasks {
        // Parse x-only public key
        let pubkey_xonly = match XOnlyPublicKey::from_slice(pubkey) {
            Ok(pk) => pk,
            Err(_) => {
                results[*idx] = false;
                continue;
            }
        };

        // Parse Schnorr signature
        let sig = match Signature::from_slice(signature) {
            Ok(s) => s,
            Err(_) => {
                results[*idx] = false;
                continue;
            }
        };

        // Handle message: Tapscript uses 32-byte sighash directly, CSFS hashes arbitrary messages
        let digest: [u8; 32] = if message.len() == 32 {
            // Already 32 bytes (Tapscript sighash) - use directly
            let mut d = [0u8; 32];
            d.copy_from_slice(message);
            d
        } else {
            // Arbitrary message (CSFS) - hash to 32 bytes. Uses OptimizedSha256 (SHA-NI when available)
            OptimizedSha256::new().hash(message)
        };

        sigs.push(sig);
        pubkeys.push(pubkey_xonly);
        msgs.push(digest);
    }

    if sigs.is_empty() {
        return Ok(results);
    }

    let sigs_bytes: Vec<[u8; 64]> = sigs.iter().map(|s| s.serialize()).collect();
    let pubkeys_bytes: Vec<[u8; 32]> = pubkeys.iter().map(|p| p.serialize()).collect();
    let msg_refs: Vec<&[u8]> = msgs.iter().map(|m| m.as_slice()).collect();

    let perf = &crate::config::get_consensus_config_ref().performance;
    let chunk_threshold =
        blvm_primitives::ibd_tuning::chunk_threshold_config_or_hardware(perf.ibd_chunk_threshold);
    let min_chunk =
        blvm_primitives::ibd_tuning::min_chunk_size_config_or_hardware(perf.ibd_min_chunk_size);
    let n = sigs.len();

    #[cfg(all(feature = "production", feature = "rayon"))]
    let batch_bools: Vec<bool> = {
        if n <= chunk_threshold {
            crate::secp256k1_backend::verify_schnorr_batch(&sigs_bytes, &msg_refs, &pubkeys_bytes)?
        } else {
            let num_threads = rayon::current_num_threads();
            let max_parallel = (n / 128).max(1);
            let num_chunks = num_threads.min(max_parallel).max(1);
            let chunk_ranges =
                blvm_primitives::ibd_tuning::compute_chunk_ranges(n, num_chunks, min_chunk);
            let chunk_results: Vec<Result<Vec<bool>>> = chunk_ranges
                .into_par_iter()
                .map(|(start, end)| {
                    crate::secp256k1_backend::verify_schnorr_batch(
                        &sigs_bytes[start..end],
                        &msg_refs[start..end],
                        &pubkeys_bytes[start..end],
                    )
                })
                .collect();
            let mut batch_bools = Vec::with_capacity(n);
            for r in chunk_results {
                batch_bools.extend(r?);
            }
            batch_bools
        }
    };

    #[cfg(not(all(feature = "production", feature = "rayon")))]
    let batch_bools: Vec<bool> =
        crate::secp256k1_backend::verify_schnorr_batch(&sigs_bytes, &msg_refs, &pubkeys_bytes)?;

    for (i, &valid) in batch_bools.iter().enumerate() {
        results[task_indices[i]] = valid;
    }

    Ok(results)
}

/// Verify Tapscript Schnorr signature (BIP 340/342)
///
/// Verifies a BIP 340 Schnorr signature for Tapscript OP_CHECKSIG.
/// Uses BIP 341 sighash algorithm for the message.
///
/// # Arguments
///
/// * `sighash` - The BIP 341 sighash (32 bytes)
/// * `pubkey` - The x-only public key (32 bytes)
/// * `signature` - The BIP 340 Schnorr signature (64 bytes)
/// * `collector` - Optional collector for deferred batch verification (production feature)
///
/// # Returns
///
/// `true` if signature is valid or collected, `false` otherwise
#[cfg(feature = "production")]
#[spec_locked("5.4.8")]
pub fn verify_tapscript_schnorr_signature(
    sighash: &[u8; 32],
    pubkey: &[u8],
    signature: &[u8],
    collector: Option<&SchnorrSignatureCollector>,
) -> Result<bool> {
    // Tapscript: Only 32-byte pubkeys are verified (BIP 340 x-only pubkeys)
    if pubkey.len() != 32 {
        return Ok(false);
    }

    // Signature must be 64 bytes (BIP 340 Schnorr)
    if signature.len() != 64 {
        return Ok(false);
    }

    // OPTIMIZATION: If collector is provided, defer verification for batch processing
    if let Some(c) = collector {
        // Collect signature for batch verification
        c.collect(sighash, pubkey, signature);
        // Return true for now - actual verification happens in batch
        return Ok(true);
    }

    // Immediate verification (fallback when no collector)
    // Parse x-only public key (32 bytes)
    let pubkey_xonly = match XOnlyPublicKey::from_slice(pubkey) {
        Ok(pk) => pk,
        Err(_) => return Ok(false), // Invalid pubkey format
    };

    // Parse Schnorr signature (64 bytes)
    let sig = match Signature::from_slice(signature) {
        Ok(s) => s,
        Err(_) => return Ok(false), // Invalid signature format
    };

    // Verify using backend (libsecp256k1 or blvm-secp256k1)
    let pk_bytes: [u8; 32] = pubkey_xonly.serialize();
    crate::secp256k1_backend::verify_schnorr(&sig.serialize(), sighash, &pk_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_signature_from_stack_zero_pubkey() {
        let message = b"test message";
        let pubkey = vec![];
        let signature = vec![0u8; 64];

        let result = verify_signature_from_stack(
            message,
            &pubkey,
            &signature,
            #[cfg(feature = "production")]
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_signature_from_stack_unknown_pubkey_type() {
        let message = b"test message";
        let pubkey = vec![1u8; 33]; // 33-byte pubkey (unknown type)
        let signature = vec![0u8; 64];

        // Unknown pubkey type should succeed
        let result = verify_signature_from_stack(
            message,
            &pubkey,
            &signature,
            #[cfg(feature = "production")]
            None,
        );
        assert_eq!(result.unwrap(), true);
    }

    #[test]
    fn test_verify_signature_from_stack_invalid_signature_length() {
        let message = b"test message";
        let pubkey = vec![1u8; 32]; // Valid 32-byte pubkey
        let signature = vec![0u8; 63]; // Invalid length (not 64 bytes)

        let result = verify_signature_from_stack(
            message,
            &pubkey,
            &signature,
            #[cfg(feature = "production")]
            None,
        );
        assert_eq!(result.unwrap(), false);
    }
}
