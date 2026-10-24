//! Hash and signature opcodes for script execution.
//!
//! OP_HASH160, OP_HASH256, OP_RIPEMD160, OP_SHA1, OP_SHA256, OP_CHECKSIG, OP_CHECKSIGVERIFY.

use crate::crypto::OptimizedSha256;
use crate::error::{ConsensusError, Result, ScriptErrorCode};
use crate::opcodes::{OP_HASH160, OP_HASH256};
use crate::types::Network;
use digest::Digest;
use ripemd::Ripemd160;
use sha1::Sha1;

use super::SigVersion;
use super::signature;
use super::stack::{StackElement, to_stack_element};

const SCRIPT_VERIFY_NULLFAIL: u32 = 0x4000;

/// Check if hash caching is disabled (delegates to script mod).
#[cfg(feature = "production")]
fn is_hash_caching_disabled() -> bool {
    super::is_caching_disabled()
}

/// Clear hash operation cache. Used by benchmark cache helpers in script mod.
#[cfg(all(feature = "production", feature = "benchmarking"))]
pub(crate) fn clear_hash_cache() {
    HASH_CACHE.with(|cell| cell.borrow_mut().clear());
}

/// HASH160 is 20 bytes, HASH256 is 32. Stored inline so a miss does not allocate.
#[cfg(feature = "production")]
struct CachedHash {
    bytes: [u8; 32],
    len: u8,
}

#[cfg(feature = "production")]
impl CachedHash {
    fn from_20(src: impl AsRef<[u8]>) -> Self {
        let src = src.as_ref();
        let mut bytes = [0u8; 32];
        bytes[..20].copy_from_slice(&src[..20]);
        Self { bytes, len: 20 }
    }

    fn from_32(src: &[u8; 32]) -> Self {
        Self {
            bytes: *src,
            len: 32,
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }
}

#[cfg(feature = "production")]
thread_local! {
    static HASH_CACHE: std::cell::RefCell<lru::LruCache<[u8; 32], CachedHash>> = std::cell::RefCell::new({
        use lru::LruCache;
        use std::num::NonZeroUsize;
        LruCache::new(NonZeroUsize::new(25_000).unwrap())
    });
}

#[cfg(feature = "production")]
fn with_hash_cache<F, R>(f: F) -> R
where
    F: FnOnce(&mut lru::LruCache<[u8; 32], CachedHash>) -> R,
{
    HASH_CACHE.with(|cell| f(&mut cell.borrow_mut()))
}

/// Distinguish HASH160 from HASH256 without hashing the script item again.
/// `first_sha` is SHA256(item), which both opcodes already need.
#[cfg(feature = "production")]
fn cache_key_from_first_sha(first_sha: &[u8; 32], op_hash160: bool) -> [u8; 32] {
    let mut buf = [0u8; 33];
    buf[..32].copy_from_slice(first_sha);
    buf[32] = if op_hash160 { OP_HASH160 } else { OP_HASH256 };
    OptimizedSha256::new().hash(&buf)
}

/// OP_RIPEMD160 - RIPEMD160(x)
pub(crate) fn op_ripemd160(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        let hash = Ripemd160::digest(&item);
        stack.push(to_stack_element(&hash));
        Ok(true)
    } else {
        Ok(false)
    }
}

/// OP_SHA1 - SHA1(x)
pub(crate) fn op_sha1(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        let hash = Sha1::digest(&item);
        stack.push(to_stack_element(&hash));
        Ok(true)
    } else {
        Ok(false)
    }
}

/// OP_SHA256 - SHA256(x)
pub(crate) fn op_sha256(stack: &mut Vec<StackElement>) -> Result<bool> {
    if let Some(item) = stack.pop() {
        let hash = OptimizedSha256::new().hash(&item);
        stack.push(to_stack_element(&hash));
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
                let sha256_hash = OptimizedSha256::new().hash(&item);
                let cache_key = cache_key_from_first_sha(&sha256_hash, true);
                let hit = with_hash_cache(|c| {
                    c.peek(&cache_key)
                        .filter(|cached| cached.len == 20)
                        .map(|cached| to_stack_element(cached.as_slice()))
                });
                if let Some(elem) = hit {
                    stack.push(elem);
                    return Ok(true);
                }
                let ripemd160_hash = Ripemd160::digest(sha256_hash);
                stack.push(to_stack_element(&ripemd160_hash));
                with_hash_cache(|c| c.put(cache_key, CachedHash::from_20(ripemd160_hash)));
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
            let first = OptimizedSha256::new().hash(&item);
            let cache_key = if !is_hash_caching_disabled() {
                let cache_key = cache_key_from_first_sha(&first, false);
                let hit = with_hash_cache(|c| {
                    c.peek(&cache_key)
                        .filter(|cached| cached.len == 32)
                        .map(|cached| to_stack_element(cached.as_slice()))
                });
                if let Some(elem) = hit {
                    stack.push(elem);
                    return Ok(true);
                }
                Some(cache_key)
            } else {
                None
            };
            let result = OptimizedSha256::new().hash(&first);
            stack.push(to_stack_element(&result));
            if let Some(cache_key) = cache_key {
                with_hash_cache(|c| c.put(cache_key, CachedHash::from_32(&result)));
            }
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
        let secp = signature::new_secp();
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

    let ok = result?;

    if !ok && (flags & SCRIPT_VERIFY_NULLFAIL) != 0 && !signature_bytes.is_empty() {
        return Err(ConsensusError::ScriptErrorWithCode {
            code: ScriptErrorCode::SigNullFail,
            message: "OP_CHECKSIG: non-null signature must not fail under NULLFAIL".into(),
        });
    }

    stack.push(to_stack_element(super::script_bool_bytes(ok)));
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
        let secp = signature::new_secp();
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

    let ok = result?;

    if !ok && (flags & SCRIPT_VERIFY_NULLFAIL) != 0 && !signature_bytes.is_empty() {
        return Err(ConsensusError::ScriptErrorWithCode {
            code: ScriptErrorCode::SigNullFail,
            message: "OP_CHECKSIGVERIFY: non-null signature must not fail under NULLFAIL".into(),
        });
    }

    Ok(ok)
}
