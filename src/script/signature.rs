//! ECDSA signature verification for script execution.
//!
//! BIP66 strict DER, BIP62 LOW_S, assumevalid optimization.

use crate::error::Result;
use crate::types::Natural;
use blvm_spec_lock::spec_locked;
use secp256k1::{ecdsa::Signature, Context, Message, PublicKey, Secp256k1, Verification};

use super::SigVersion;

#[cfg(feature = "production")]
use std::thread_local;

#[cfg(feature = "production")]
thread_local! {
    static SECP256K1_CONTEXT: Secp256k1<secp256k1::All> = Secp256k1::new();
}

/// Run a closure with the thread-local secp256k1 context. Used by mod.rs for verify_signature calls.
#[cfg(feature = "production")]
pub(crate) fn with_secp_context<F, R>(f: F) -> R
where
    F: FnOnce(&Secp256k1<secp256k1::All>) -> R,
{
    SECP256K1_CONTEXT.with(f)
}

pub(crate) fn get_assumevalid_height() -> u64 {
    crate::config::get_assume_valid_height()
}

/// Verify ECDSA signature using secp256k1.
/// BIP66: strict DER. BIP62: LOW_S, STRICTENC.
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_signature<C: Context + Verification>(
    secp: &Secp256k1<C>,
    pubkey_bytes: &[u8],
    signature_bytes: &[u8],
    sighash: &[u8; 32],
    flags: u32,
    height: Natural,
    network: crate::types::Network,
    sigversion: SigVersion,
) -> Result<bool> {
    // ECDSA short-circuit when height < assume_valid_height (no two-week / chainwork check here).
    // connect_block_inner uses `skip_signatures` with two_week_ok && chainwork_ok; when that is
    // false but height < av, the interpreter still runs and this path may skip ECDSA anyway.
    // Keep in sync with consensus review if Core parity for assume-valid is tightened.
    let assumevalid_height = get_assumevalid_height();
    if assumevalid_height > 0 && height < assumevalid_height {
        if signature_bytes.is_empty() {
            return Ok(false);
        }
        return Ok(true);
    }

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

    let signature = if flags & 0x04 != 0 {
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

    if flags & 0x08 != 0 {
        let before = signature.serialize_compact();
        let mut normalized = signature;
        normalized.normalize_s();
        if before != normalized.serialize_compact() {
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

    let pubkey = match PublicKey::from_slice(pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => return Ok(false),
    };

    let normalized_signature = if flags & 0x08 != 0 {
        signature
    } else {
        let mut s = signature;
        s.normalize_s();
        s
    };

    let sig_compact = normalized_signature.serialize_compact();
    let pk_compressed = pubkey.serialize();
    crate::secp256k1_backend::verify_ecdsa(sighash, &sig_compact, &pk_compressed)
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
    SECP256K1_CONTEXT.with(|secp| {
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
#[allow(dead_code)]
fn parse_task_for_batch(
    pubkey_bytes: &[u8],
    signature_bytes: &[u8],
    sighash: &[u8; 32],
    flags: u32,
    height: Natural,
    network: crate::types::Network,
) -> Result<Option<(PublicKey, secp256k1::ecdsa::Signature, Message)>> {
    use secp256k1::ecdsa::Signature;

    if signature_bytes.is_empty() {
        return Ok(None);
    }
    let sig_len = signature_bytes.len();
    let sighash_byte = signature_bytes[sig_len - 1];
    let der_sig = &signature_bytes[..sig_len - 1];

    if flags & 0x04 != 0
        && !crate::bip_validation::check_bip66_network(signature_bytes, height, network)?
    {
        return Ok(None);
    }
    if flags & 0x02 != 0 {
        let base_sighash = sighash_byte & !0x80;
        if !(0x01..=0x03).contains(&base_sighash) {
            return Ok(None);
        }
    }

    let signature = if flags & 0x04 != 0 {
        match Signature::from_der(der_sig) {
            Ok(sig) => sig,
            Err(_) => return Ok(None),
        }
    } else {
        match Signature::from_der_lax(der_sig) {
            Ok(sig) => sig,
            Err(_) => return Ok(None),
        }
    };

    let original_compact = signature.serialize_compact();
    let mut normalized_signature = signature;
    normalized_signature.normalize_s();
    let norm_compact = normalized_signature.serialize_compact();
    if flags & 0x08 != 0 && original_compact != norm_compact {
        return Ok(None);
    }

    if flags & 0x02 != 0 {
        if pubkey_bytes.len() < 33 {
            return Ok(None);
        }
        if pubkey_bytes[0] == 0x04 {
            if pubkey_bytes.len() != 65 {
                return Ok(None);
            }
        } else if pubkey_bytes[0] == 0x02 || pubkey_bytes[0] == 0x03 {
            if pubkey_bytes.len() != 33 {
                return Ok(None);
            }
        } else {
            return Ok(None);
        }
    }

    let pubkey = match PublicKey::from_slice(pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => return Ok(None),
    };
    let message = match Message::from_digest_slice(sighash) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };

    Ok(Some((pubkey, normalized_signature, message)))
}

#[cfg(feature = "production")]
#[spec_locked("5.2")]
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

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let r: Result<Vec<bool>> = verification_tasks
            .par_iter()
            .map(|(pubkey_bytes, signature_bytes, sighash)| {
                SECP256K1_CONTEXT.with(|secp| {
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
            })
            .collect();
        #[cfg(feature = "profile")]
        crate::script_profile::add_multisig_ns(_t0.elapsed().as_nanos() as u64);
        r
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut results = Vec::with_capacity(verification_tasks.len());
        for (pubkey_bytes, signature_bytes, sighash) in verification_tasks {
            let secp = Secp256k1::new();
            let result = verify_signature(
                &secp,
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
        #[cfg(feature = "profile")]
        crate::script_profile::add_multisig_ns(_t0.elapsed().as_nanos() as u64);
        Ok(results)
    }
}
