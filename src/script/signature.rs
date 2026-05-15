//! ECDSA signature verification for script execution.
//!
//! BIP66 strict DER, BIP62 LOW_S, assumevalid optimization.
//! Supports two backends (exactly one must be enabled):
//!   - `blvm-secp256k1`: pure-Rust, no FFI, handles compressed and uncompressed pubkeys
//!   - `secp256k1-fallback`: crates.io secp256k1 0.28 (libsecp256k1 FFI)

use crate::error::Result;
use crate::types::Natural;

use super::SigVersion;

// ---------- backend-specific imports & thread-local context ----------

#[cfg(all(feature = "secp256k1-fallback", not(feature = "blvm-secp256k1")))]
use secp256k1::Secp256k1;

/// Opaque secp context type. Under `blvm-secp256k1` (default) this is a zero-sized placeholder;
/// under `secp256k1-fallback` (and NOT `blvm-secp256k1`) it is `secp256k1::Secp256k1<All>`.
/// Callers treat it identically — they pass it to `verify_signature` which ignores / uses it.
/// When both features are active `blvm-secp256k1` takes priority.
#[cfg(all(feature = "secp256k1-fallback", not(feature = "blvm-secp256k1")))]
pub(crate) type SecpCtx = secp256k1::Secp256k1<secp256k1::All>;

#[cfg(not(all(feature = "secp256k1-fallback", not(feature = "blvm-secp256k1"))))]
#[derive(Default, Clone, Copy)]
pub(crate) struct SecpCtx;

/// Construct a fresh context.
pub(crate) fn new_secp() -> SecpCtx {
    #[cfg(all(feature = "secp256k1-fallback", not(feature = "blvm-secp256k1")))]
    {
        secp256k1::Secp256k1::new()
    }
    #[cfg(not(all(feature = "secp256k1-fallback", not(feature = "blvm-secp256k1"))))]
    {
        SecpCtx
    }
}

#[cfg(feature = "production")]
use std::thread_local;

#[cfg(feature = "production")]
thread_local! {
    static SECP256K1_CONTEXT: SecpCtx = new_secp();
}

/// Run a closure with the thread-local context (avoids repeated allocation under secp256k1-fallback).
#[cfg(feature = "production")]
pub(crate) fn with_secp_context<F, R>(f: F) -> R
where
    F: FnOnce(&SecpCtx) -> R,
{
    SECP256K1_CONTEXT.with(f)
}

// ---------- core verifier ----------

/// Verify ECDSA signature.
/// BIP66: strict DER. BIP62: LOW_S, STRICTENC.
/// The `_secp` parameter is accepted for call-site uniformity but is ignored when
/// the `blvm-secp256k1` backend is active.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_signature(
    _secp: &SecpCtx,
    pubkey_bytes: &[u8],
    signature_bytes: &[u8],
    sighash: &[u8; 32],
    flags: u32,
    height: Natural,
    network: crate::types::Network,
    sigversion: SigVersion,
) -> Result<bool> {
    if signature_bytes.is_empty() {
        return Ok(false);
    }
    let sig_len = signature_bytes.len();
    let sighash_byte = signature_bytes[sig_len - 1];
    let der_sig = &signature_bytes[..sig_len - 1];

    if flags & 0x04 != 0
        && !crate::bip_validation::check_bip66_network(signature_bytes, height, network)?
    {
        return Ok(false);
    }

    if flags & 0x02 != 0 {
        let base_sighash = sighash_byte & !0x80;
        if !(0x01..=0x03).contains(&base_sighash) {
            return Ok(false);
        }
    }

    if flags & 0x02 != 0 {
        if pubkey_bytes.len() < 33 {
            return Ok(false);
        }
        if pubkey_bytes[0] == 0x04 {
            if pubkey_bytes.len() != 65 {
                return Ok(false);
            }
        } else if pubkey_bytes[0] == 0x02 || pubkey_bytes[0] == 0x03 {
            if pubkey_bytes.len() != 33 {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
    }

    const SCRIPT_VERIFY_WITNESS_PUBKEYTYPE: u32 = 0x8000;
    if (flags & SCRIPT_VERIFY_WITNESS_PUBKEYTYPE) != 0
        && sigversion == SigVersion::WitnessV0
        && !(pubkey_bytes.len() == 33 && (pubkey_bytes[0] == 0x02 || pubkey_bytes[0] == 0x03))
    {
        return Ok(false);
    }

    let strict_der = flags & 0x04 != 0;
    let enforce_low_s = flags & 0x08 != 0;

    #[cfg(feature = "blvm-secp256k1")]
    {
        return Ok(blvm_secp256k1::ecdsa::verify_ecdsa_direct(
            der_sig,
            pubkey_bytes,
            sighash,
            strict_der,
            enforce_low_s,
        )
        .unwrap_or(false));
    }

    #[cfg(all(feature = "secp256k1-fallback", not(feature = "blvm-secp256k1")))]
    {
        use secp256k1::{ecdsa::Signature, PublicKey};

        let signature = if strict_der {
            match Signature::from_der(der_sig) {
                Ok(sig) => sig,
                Err(_) => return Ok(false),
            }
        } else {
            match Signature::from_der_lax(der_sig) {
                Ok(sig) => sig,
                Err(_) => return Ok(false),
            }
        };

        if enforce_low_s {
            let before = signature.serialize_compact();
            let mut normalized = signature;
            normalized.normalize_s();
            if before != normalized.serialize_compact() {
                return Ok(false);
            }
        }

        let pubkey = match PublicKey::from_slice(pubkey_bytes) {
            Ok(pk) => pk,
            Err(_) => return Ok(false),
        };

        let normalized_signature = if enforce_low_s {
            signature
        } else {
            let mut s = signature;
            s.normalize_s();
            s
        };

        let sig_compact = normalized_signature.serialize_compact();
        let pk_compressed = pubkey.serialize();
        return crate::secp256k1_backend::verify_ecdsa(sighash, &sig_compact, &pk_compressed);
    }

    #[allow(unreachable_code)]
    Ok(false)
}

/// Verify pre-extracted ECDSA (P2PKH/P2PK) inline without re-parsing script_sig.
#[cfg(feature = "production")]
pub fn verify_pre_extracted_ecdsa(
    pubkey_bytes: &[u8],
    signature_bytes: &[u8],
    sighash: &[u8; 32],
    flags: u32,
    height: Natural,
    network: crate::types::Network,
) -> Result<bool> {
    with_secp_context(|secp| {
        verify_signature(
            secp,
            pubkey_bytes,
            signature_bytes,
            sighash,
            flags,
            height,
            network,
            SigVersion::Base,
        )
    })
}

#[cfg(feature = "production")]
pub fn batch_verify_signatures(
    verification_tasks: &[(&[u8], &[u8], [u8; 32])],
    flags: u32,
    height: Natural,
    network: crate::types::Network,
) -> Result<Vec<bool>> {
    #[cfg(feature = "profile")]
    let _t0 = std::time::Instant::now();

    if verification_tasks.is_empty() {
        #[cfg(feature = "profile")]
        crate::script_profile::add_multisig_ns(_t0.elapsed().as_nanos() as u64);
        return Ok(Vec::new());
    }

    // Serial verification. The previous rayon `par_iter` path was a
    // hidden hotspot: it fires inside script execution (per OP_CHECKMULTISIG), and IBD
    // runs N script verifiers in parallel on N worker threads. Each multisig opcode
    // pushed up to 20 sig-checks into the global rayon pool, oversubscribing the CPU
    // and causing IBD workers to stall waiting for their own rayon tasks. CHECKMULTISIG
    // caps at 20 sigs so serial cost is small relative to the per-worker thread budget.
    let mut results = Vec::with_capacity(verification_tasks.len());
    with_secp_context(|secp| -> Result<()> {
        for (pubkey_bytes, signature_bytes, sighash) in verification_tasks {
            let result = verify_signature(
                secp,
                pubkey_bytes,
                signature_bytes,
                sighash,
                flags,
                height,
                network,
                SigVersion::Base,
            )?;
            results.push(result);
        }
        Ok(())
    })?;
    #[cfg(feature = "profile")]
    crate::script_profile::add_multisig_ns(_t0.elapsed().as_nanos() as u64);
    Ok(results)
}
