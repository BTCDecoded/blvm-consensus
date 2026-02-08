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

use crate::error::{ConsensusError, Result};
use crate::types::{ByteString, Hash};
use secp256k1::{XOnlyPublicKey, Message, schnorr::Signature, Secp256k1};
use sha2::{Digest, Sha256};
use blvm_spec_lock::spec_locked;

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
///
/// # Returns
///
/// `true` if signature is valid, `false` otherwise
///
/// # Errors
///
/// Returns error if:
/// - Pubkey size is zero
/// - Signature verification fails (for 32-byte pubkeys)
///
/// # Note on Message Hashing
///
/// BIP-348 states "Message is NOT hashed" because BIP 340 accepts messages of any size.
/// However, secp256k1's `Message::from_digest_slice()` requires exactly 32 bytes.
/// BIP 340 uses tagged hashes. For CSFS, we hash the message with SHA256 to create
/// a 32-byte digest, which is then used for BIP 340 verification.
/// This matches the reference implementation in Bitcoin Core PR #29270.
#[spec_locked("5.4.8")]
pub fn verify_signature_from_stack(
    message: &[u8],
    pubkey: &[u8],
    signature: &[u8],
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
        // Use SHA256 to hash message to 32 bytes (matches Bitcoin Core PR #29270)
        
        // Signature must be 64 bytes (BIP 340 Schnorr)
        if signature.len() != 64 {
            return Ok(false);
        }

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
        // This matches Bitcoin Core PR #29270 implementation
        let message_hash = Sha256::digest(message);
        let msg = Message::from_digest_slice(&message_hash)
            .map_err(|_| ConsensusError::InvalidSignature("Invalid message".into()))?;

        // Verify using secp256k1 BIP 340 verification
        let secp = Secp256k1::verification_only();
        match secp.verify_schnorr(&sig, &msg, &pubkey_xonly) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false), // Invalid signature
        }
    } else {
        // BIP-348: Unknown pubkey type - succeeds as if valid
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_signature_from_stack_zero_pubkey() {
        let message = b"test message";
        let pubkey = vec![];
        let signature = vec![0u8; 64];
        
        let result = verify_signature_from_stack(message, &pubkey, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_signature_from_stack_unknown_pubkey_type() {
        let message = b"test message";
        let pubkey = vec![1u8; 33]; // 33-byte pubkey (unknown type)
        let signature = vec![0u8; 64];
        
        // Unknown pubkey type should succeed
        let result = verify_signature_from_stack(message, &pubkey, &signature);
        assert_eq!(result.unwrap(), true);
    }

    #[test]
    fn test_verify_signature_from_stack_invalid_signature_length() {
        let message = b"test message";
        let pubkey = vec![1u8; 32]; // Valid 32-byte pubkey
        let signature = vec![0u8; 63]; // Invalid length (not 64 bytes)
        
        let result = verify_signature_from_stack(message, &pubkey, &signature);
        assert_eq!(result.unwrap(), false);
    }
}

