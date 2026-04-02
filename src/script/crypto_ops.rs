//! Hash and signature opcodes for script execution.
//!
//! OP_HASH160, OP_HASH256, OP_RIPEMD160, OP_SHA1, OP_SHA256, OP_CHECKSIG, OP_CHECKSIGVERIFY.

use crate::crypto::OptimizedSha256;
use crate::error::{ConsensusError, Result, ScriptErrorCode};
use crate::opcodes::{OP_HASH160, OP_HASH256};
use crate::types::Network;
use digest::Digest;
use ripemd::Ripemd160;
#[cfg(not(feature = "production"))]
use secp256k1::Secp256k1;
use sha1::Sha1;

use super::signature;
use super::stack::{to_stack_element, StackElement};
use super::SigVersion;

const SCRIPT_VERIFY_NULLFAIL: u32 = 0x4000;

/// Check if hash caching is disabled (delegates to script mod).
#[cfg(feature = "production")]
fn is_hash_caching_disabled() -> bool {
    super::is_caching_disabled()
}

/// Clear hash operation cache. Re-exported by script mod for benchmarks.
#[cfg(feature = "production")]
pub(crate) fn clear_hash_cache() {
    HASH_CACHE.with(|cell| cell.borrow_mut().clear());
}

#[cfg(feature = "production")]
thread_local! {
    static HASH_CACHE: std::cell::RefCell<lru::LruCache<[u8; 32], Vec<u8>>> = std::cell::RefCell::new({
        use lru::LruCache;
        use std::num::NonZeroUsize;
        LruCache::new(NonZeroUsize::new(25_000).unwrap())
    });
}

#[cfg(feature = "production")]
fn with_hash_cache<F, R>(f: F) -> R
where
    F: FnOnce(&mut lru::LruCache<[u8; 32], Vec<u8>>) -> R,
{
    HASH_CACHE.with(|cell| f(&mut cell.borrow_mut()))
}

#[cfg(feature = "production")]
fn compute_hash_cache_key(input: &[u8], op_hash160: bool) -> [u8; 32] {
    let mut data = input.to_vec();
    data.push(if op_hash160 { OP_HASH160 } else { OP_HASH256 });
    let hash = OptimizedSha256::new().hash(&data);
    let mut key = [0u8; 32];
    key.copy_from_slice(&hash);
    key
}

/// OP_RIPEMD160 - RIPEMD160(x)
pub(crate) fn op_ripemd160(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        let hash = Ripemd160::digest(&item);
        #[cfg(feature = "production")]
        {
            let mut hash_vec = Vec::with_capacity(20);
            hash_vec.extend_from_slice(&hash);
            stack.push(to_stack_element(&hash_vec));
        }
        #[cfg(not(feature = "production"))]
        {
            stack.push(to_stack_element(&hash));
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// OP_SHA1 - SHA1(x)
pub(crate) fn op_sha1(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        let hash = Sha1::digest(&item);
        #[cfg(feature = "production")]
        {
            let mut hash_vec = Vec::with_capacity(20);
            hash_vec.extend_from_slice(&hash);
            stack.push(to_stack_element(&hash_vec));
        }
        #[cfg(not(feature = "production"))]
        {
            stack.push(to_stack_element(&hash));
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// OP_SHA256 - SHA256(x)
pub(crate) fn op_sha256(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        let hash = OptimizedSha256::new().hash(&item);
        #[cfg(feature = "production")]
        {
            let mut hash_vec = Vec::with_capacity(32);
            hash_vec.extend_from_slice(&hash);
            stack.push(to_stack_element(&hash_vec));
        }
        #[cfg(not(feature = "production"))]
        {
            stack.push(to_stack_element(&hash));
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// OP_HASH160 - RIPEMD160(SHA256(x))
///
/// Production: pubkeys (33/65 bytes) skip cache to avoid alloc and lock contention.
pub(crate) fn op_hash160(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        #[cfg(feature = "production")]
        {
            if item.len() == 33 || item.len() == 65 {
                let sha256_hash = OptimizedSha256::new().hash(&item);
                let ripemd160_hash = Ripemd160::digest(sha256_hash);
                stack.push(to_stack_element(&ripemd160_hash));
            } else if !is_hash_caching_disabled() {
                let cache_key = compute_hash_cache_key(&item, true);
                if let Some(cached_result) = with_hash_cache(|c| c.peek(&cache_key).cloned()) {
                    if cached_result.len() == 20 {
                        stack.push(to_stack_element(&cached_result));
                        return Ok(true);
                    }
                }
                let sha256_hash = OptimizedSha256::new().hash(&item);
                let ripemd160_hash = Ripemd160::digest(sha256_hash);
                let mut hash_vec = Vec::with_capacity(20);
                hash_vec.extend_from_slice(&ripemd160_hash);
                with_hash_cache(|c| c.put(cache_key, hash_vec.clone()));
                stack.push(to_stack_element(&hash_vec));
            } else {
                let sha256_hash = OptimizedSha256::new().hash(&item);
                let ripemd160_hash = Ripemd160::digest(sha256_hash);
                stack.push(to_stack_element(&ripemd160_hash));
            }
        }
        #[cfg(not(feature = "production"))]
        {
            let sha256_hash = OptimizedSha256::new().hash(&item);
            let ripemd160_hash = Ripemd160::digest(sha256_hash);
            stack.push(to_stack_element(&ripemd160_hash));
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// OP_HASH256 - SHA256(SHA256(x))
pub(crate) fn op_hash256(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        #[cfg(feature = "production")]
        {
            if !is_hash_caching_disabled() {
                let cache_key = compute_hash_cache_key(&item, false);
                if let Some(cached_result) = with_hash_cache(|c| c.peek(&cache_key).cloned()) {
                    if cached_result.len() == 32 {
                        stack.push(to_stack_element(&cached_result));
                        return Ok(true);
                    }
                }
            }
            let hasher = OptimizedSha256::new();
            let result = hasher.hash256(&item).to_vec();
            if !is_hash_caching_disabled() {
                let cache_key = compute_hash_cache_key(&item, false);
                with_hash_cache(|c| c.put(cache_key, result.clone()));
            }
            stack.push(to_stack_element(&result));
        }
        #[cfg(not(feature = "production"))]
        {
            let result = OptimizedSha256::new().hash256(&item);
            stack.push(to_stack_element(&result));
        }
        Ok(true)
    } else {
        Ok(false)
    }
}

/// OP_CHECKSIG (simple path) - verify ECDSA signature with dummy hash.
/// Used by execute_opcode when no transaction context is available (e.g. basic tests).
pub(crate) fn op_checksig_simple(stack: &mut Vec<StackElement>, flags: u32) -> Result<bool> {
    if stack.len() < 2 {
        return Err(ConsensusError::ScriptErrorWithCode {
            code: ScriptErrorCode::InvalidStackOperation,
            message: "OP_CHECKSIG: insufficient stack items".into(),
        });
    }
    let pubkey_bytes = stack.pop().unwrap();
    let signature_bytes = stack.pop().unwrap();

    let dummy_hash = [0u8; 32];
    #[cfg(feature = "production")]
    let result = signature::with_secp_context(|secp| {
        signature::verify_signature(
            secp,
            &pubkey_bytes,
            &signature_bytes,
            &dummy_hash,
            flags,
            0,
            Network::Regtest,
            SigVersion::Base,
        )
    });

    #[cfg(not(feature = "production"))]
    let result = {
        let secp = Secp256k1::new();
        signature::verify_signature(
            &secp,
            &pubkey_bytes,
            &signature_bytes,
            &dummy_hash,
            flags,
            0,
            Network::Regtest,
            SigVersion::Base,
        )
    };

    let ok = result.unwrap_or(false);

    if !ok && (flags & SCRIPT_VERIFY_NULLFAIL) != 0 && !signature_bytes.is_empty() {
        return Err(ConsensusError::ScriptErrorWithCode {
            code: ScriptErrorCode::SigNullFail,
            message: "OP_CHECKSIG: non-null signature must not fail under NULLFAIL".into(),
        });
    }

    stack.push(to_stack_element(&[if ok { 1 } else { 0 }]));
    Ok(true)
}

/// OP_CHECKSIGVERIFY (simple path) - verify ECDSA signature and fail if invalid.
/// Used by execute_opcode when no transaction context is available.
pub(crate) fn op_checksigverify_simple(stack: &mut Vec<StackElement>, flags: u32) -> Result<bool> {
    if stack.len() < 2 {
        return Ok(false);
    }
    let pubkey_bytes = stack.pop().unwrap();
    let signature_bytes = stack.pop().unwrap();

    let dummy_hash = [0u8; 32];
    #[cfg(feature = "production")]
    let result = signature::with_secp_context(|secp| {
        signature::verify_signature(
            secp,
            &pubkey_bytes,
            &signature_bytes,
            &dummy_hash,
            flags,
            0,
            Network::Regtest,
            SigVersion::Base,
        )
    });

    #[cfg(not(feature = "production"))]
    let result = {
        let secp = Secp256k1::new();
        signature::verify_signature(
            &secp,
            &pubkey_bytes,
            &signature_bytes,
            &dummy_hash,
            flags,
            0,
            Network::Regtest,
            SigVersion::Base,
        )
    };

    let ok = result.unwrap_or(false);

    if !ok && (flags & SCRIPT_VERIFY_NULLFAIL) != 0 && !signature_bytes.is_empty() {
        return Err(ConsensusError::ScriptErrorWithCode {
            code: ScriptErrorCode::SigNullFail,
            message: "OP_CHECKSIGVERIFY: non-null signature must not fail under NULLFAIL".into(),
        });
    }

    Ok(ok)
}
