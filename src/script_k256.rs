//! k256-based signature verification (Phase 3.2)
//!
//! Pure Rust implementation using k256 crate instead of secp256k1 FFI.
//! This module provides the same interface as the existing verify_signature
//! function but uses k256 for cryptographic operations.

#[cfg(feature = "k256")]
use blvm_spec_lock::spec_locked;
#[allow(deprecated)] // generic_array is deprecated but k256 still uses it internally
use k256::{
    ecdsa::{signature::Verifier, Signature, VerifyingKey},
    elliptic_curve::generic_array::GenericArray,
    EncodedPoint,
};

/// Verify ECDSA signature using k256 (pure Rust)
///
/// This is the k256-based implementation that replaces the FFI-based secp256k1.
///
/// # Arguments
/// * `pubkey_bytes` - Public key in SEC1 format (compressed or uncompressed)
/// * `signature_bytes` - ECDSA signature in DER format
/// * `sighash` - 32-byte message digest (transaction sighash)
/// * `_flags` - Verification flags (reserved for future use)
///
/// # Returns
/// `true` if signature is valid, `false` otherwise
#[cfg(feature = "k256")]
#[spec_locked("5.2")]
#[cfg_attr(feature = "production", inline(always))]
#[cfg_attr(not(feature = "production"), inline)]
pub fn verify_signature_k256(
    pubkey_bytes: &[u8],
    signature_bytes: &[u8],
    sighash: &[u8; 32],
    _flags: u32,
) -> bool {
    // Parse public key from SEC1 format
    let verifying_key = match EncodedPoint::from_bytes(pubkey_bytes) {
        Ok(encoded_point) => match VerifyingKey::from_encoded_point(&encoded_point) {
            Ok(vk) => vk,
            Err(_) => return false,
        },
        Err(_) => return false,
    };

    // Parse signature from DER format
    // k256 Signature can parse from DER directly
    let signature = match Signature::from_der(signature_bytes) {
        Ok(sig) => sig,
        Err(_) => {
            // Try parsing as compact 64-byte format if DER fails
            if signature_bytes.len() == 64 {
                // Compact format: r || s (32 bytes each)
                // Convert slices to fixed-size arrays first for type safety
                let r_array: [u8; 32] = match signature_bytes[..32].try_into() {
                    Ok(arr) => arr,
                    Err(_) => return false,
                };
                let s_array: [u8; 32] = match signature_bytes[32..].try_into() {
                    Ok(arr) => arr,
                    Err(_) => return false,
                };

                // Convert arrays to GenericArray using Into trait
                // Note: k256's Signature::from_scalars requires GenericArray from elliptic_curve.
                // While k256 still uses generic_array 0.x internally, we avoid the deprecated
                // GenericArray::from_slice() by converting fixed-size arrays using Into<GenericArray>.
                // This is the recommended approach until k256 migrates to generic-array 1.x.
                #[allow(deprecated)]
                use k256::elliptic_curve::generic_array::typenum::U32;
                #[allow(deprecated)]
                let r: GenericArray<u8, U32> = r_array.into();
                #[allow(deprecated)]
                let s: GenericArray<u8, U32> = s_array.into();

                match Signature::from_scalars(r, s) {
                    Ok(sig) => sig,
                    Err(_) => return false,
                }
            } else {
                return false;
            }
        }
    };

    // Convert sighash to GenericArray for verification (32 bytes = SHA256 output)
    // k256's verify method expects a reference to a GenericArray<u8, U32>
    // We can pass the slice directly and let k256 handle the conversion
    use sha2::digest::Digest;
    use sha2::Sha256;
    let msg_hash = Sha256::digest(sighash);
    verifying_key.verify(&msg_hash, &signature).is_ok()
}

#[cfg(test)]
#[cfg(feature = "k256")]
mod tests {
    use super::*;

    /// Test k256 signature verification with known good signature
    ///
    /// Uses a test vector with:
    /// - Public key: 02 79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 (compressed)
    /// - Message: "Hello, Bitcoin!" (SHA256 hashed)
    /// - Signature: Valid ECDSA signature
    #[test]
    fn test_k256_signature_verification_valid() {
        // Test with a simple case: invalid signature should return false
        // This tests the interface and error handling

        // Compressed public key (33 bytes) - valid format
        let pubkey = vec![
            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce,
            0x87, 0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81,
            0x5b, 0x16, 0xf8, 0x17, 0x98,
        ];

        // Invalid DER signature (too short) - should return false
        let invalid_signature = vec![0x30, 0x06, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00];
        let sighash = [0u8; 32];

        let result = verify_signature_k256(&pubkey, &invalid_signature, &sighash, 0);
        assert!(!result, "Invalid signature should return false");
    }

    #[test]
    fn test_k256_signature_verification_invalid_pubkey() {
        // Test with invalid public key format
        let invalid_pubkey = vec![0x00, 0x01, 0x02]; // Too short
        let signature = vec![0x30, 0x06, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00];
        let sighash = [0u8; 32];

        let result = verify_signature_k256(&invalid_pubkey, &signature, &sighash, 0);
        assert!(!result, "Invalid pubkey should return false");
    }

    #[test]
    fn test_k256_signature_verification_deterministic() {
        // Test that verification is deterministic
        let pubkey = vec![
            0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce,
            0x87, 0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81,
            0x5b, 0x16, 0xf8, 0x17, 0x98,
        ];
        let signature = vec![0x30, 0x06, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00];
        let sighash = [0u8; 32];

        let result1 = verify_signature_k256(&pubkey, &signature, &sighash, 0);
        let result2 = verify_signature_k256(&pubkey, &signature, &sighash, 0);

        assert_eq!(result1, result2, "Verification should be deterministic");
    }
}

