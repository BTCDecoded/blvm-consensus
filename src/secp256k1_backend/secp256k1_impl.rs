//! secp256k1-fork / libsecp256k1 backend implementation.

use crate::error::{ConsensusError, Result};
use crate::types::Hash;
use secp256k1::{ecdsa, schnorr, Message, PublicKey, Secp256k1, XOnlyPublicKey};

pub fn verify_ecdsa(
    msg_hash: &[u8; 32],
    sig_compact: &[u8; 64],
    pubkey_compressed: &[u8; 33],
) -> Result<bool> {
    let msg = Message::from_digest(*msg_hash);
    let sig = match ecdsa::Signature::from_compact(sig_compact) {
        Ok(s) => s,
        Err(_) => return Ok(false),
    };
    let pk = match PublicKey::from_slice(pubkey_compressed) {
        Ok(p) => p,
        Err(_) => return Ok(false),
    };
    let secp = Secp256k1::verification_only();
    Ok(secp.verify_ecdsa(&msg, &sig, &pk).is_ok())
}

pub fn verify_schnorr(sig: &[u8; 64], msg: &[u8], pubkey: &[u8; 32]) -> Result<bool> {
    if msg.len() != 32 {
        return Ok(false);
    }
    let msg_secp = match Message::from_digest_slice(msg) {
        Ok(m) => m,
        Err(_) => return Ok(false),
    };
    let pk = match XOnlyPublicKey::from_slice(pubkey) {
        Ok(p) => p,
        Err(_) => return Ok(false),
    };
    let sig_parsed = match schnorr::Signature::from_slice(sig) {
        Ok(s) => s,
        Err(_) => return Ok(false),
    };
    let secp = Secp256k1::verification_only();
    Ok(secp.verify_schnorr(&sig_parsed, &msg_secp, &pk).is_ok())
}

/// Per-sig fallback when using crates.io secp256k1 (no batch API). Batch path uses blvm_impl when blvm-secp256k1.
pub fn verify_schnorr_batch(
    sigs: &[[u8; 64]],
    msgs: &[&[u8]],
    pubkeys: &[[u8; 32]],
) -> Result<Vec<bool>> {
    let n = sigs.len().min(msgs.len()).min(pubkeys.len());
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        results.push(verify_schnorr(&sigs[i], msgs[i], &pubkeys[i])?);
    }
    Ok(results)
}

pub fn taproot_output_key(internal_pubkey: &[u8; 32], merkle_root: &Hash) -> Result<[u8; 32]> {
    use secp256k1::{Parity, Scalar, XOnlyPublicKey};
    use sha2::{Digest, Sha256};

    let internal_pk = match XOnlyPublicKey::from_slice(internal_pubkey) {
        Ok(pk) => pk,
        Err(_) => {
            return Err(ConsensusError::InvalidSignature(
                "Invalid internal public key".into(),
            ))
        }
    };

    let mut tweak_data = Vec::new();
    tweak_data.extend_from_slice(b"TapTweak");
    tweak_data.extend_from_slice(internal_pubkey);
    tweak_data.extend_from_slice(merkle_root);
    let tweak_hash = Sha256::digest(&tweak_data);
    let tweak_scalar = match Scalar::from_be_bytes(tweak_hash.into()) {
        Ok(s) => s,
        Err(_) => {
            return Err(ConsensusError::InvalidSignature(
                "Invalid tweak scalar".into(),
            ))
        }
    };

    let secp = Secp256k1::verification_only();
    let full_pk = PublicKey::from_x_only_public_key(internal_pk, Parity::Even);
    let tweaked_pk = full_pk.add_exp_tweak(&secp, &tweak_scalar).map_err(|_| {
        ConsensusError::InvalidSignature("Failed to compute tweaked public key".into())
    })?;
    let xonly_pk = XOnlyPublicKey::from(tweaked_pk);
    Ok(xonly_pk.serialize())
}

/// BIP 341 tagged hash: SHA256(SHA256(tag) || SHA256(tag) || data).
fn bip341_tagged_hash(tag: &[u8], data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let tag_hash = Sha256::digest(tag);
    let mut hasher = Sha256::new();
    hasher.update(tag_hash);
    hasher.update(tag_hash);
    hasher.update(data);
    hasher.finalize().into()
}

fn compact_size_encode(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(9);
    if n < 0xfd {
        out.push(n as u8);
    } else if n <= 0xffff {
        out.push(0xfd);
        out.extend_from_slice(&(n as u16).to_le_bytes());
    } else if n <= 0xffff_ffff {
        out.push(0xfe);
        out.extend_from_slice(&(n as u32).to_le_bytes());
    } else {
        out.push(0xff);
        out.extend_from_slice(&(n as u64).to_le_bytes());
    }
    out
}

pub fn tap_leaf_hash(leaf_version: u8, script: &[u8]) -> [u8; 32] {
    let mut data = Vec::with_capacity(1 + 9 + script.len());
    data.push(leaf_version);
    data.extend(compact_size_encode(script.len()));
    data.extend_from_slice(script);
    bip341_tagged_hash(b"TapLeaf", &data)
}

pub fn tap_branch_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut data = [0u8; 64];
    data[..32].copy_from_slice(left);
    data[32..].copy_from_slice(right);
    bip341_tagged_hash(b"TapBranch", &data)
}

pub fn tap_sighash_hash(data: &[u8]) -> [u8; 32] {
    bip341_tagged_hash(b"TapSighash", data)
}
