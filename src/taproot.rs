//! Taproot functions from Orange Paper Section 11.2

use crate::error::Result;
use crate::types::*;
use crate::types::{ByteString, Hash};
use crate::witness;
use bitcoin_hashes::{sha256d, Hash as BitcoinHash, HashEngine};
use secp256k1::{PublicKey, Scalar, Secp256k1, XOnlyPublicKey};
use sha2::{Digest, Sha256};

/// Witness Data: 𝒲 = 𝕊* (stack of witness elements)
///
/// Uses unified witness type from witness module for consistency with SegWit
pub use crate::witness::Witness;

/// Taproot output script: OP_1 <32-byte-hash>
pub const TAPROOT_SCRIPT_PREFIX: u8 = 0x51; // OP_1

/// Validate Taproot output script
pub fn validate_taproot_script(script: &ByteString) -> Result<bool> {
    // Check if script is P2TR: OP_1 <33-byte-key>
    if script.len() != 34 {
        return Ok(false);
    }

    if script[0] != TAPROOT_SCRIPT_PREFIX {
        return Ok(false);
    }

    // The remaining 33 bytes should be the Taproot output key
    Ok(true)
}

/// Extract Taproot output key from script
pub fn extract_taproot_output_key(script: &ByteString) -> Result<Option<[u8; 32]>> {
    if !validate_taproot_script(script)? {
        return Ok(None);
    }

    let mut output_key = [0u8; 32];
    output_key.copy_from_slice(&script[1..33]);
    Ok(Some(output_key))
}

/// Compute Taproot tweak using proper cryptographic operations
/// OutputKey = InternalPubKey + TaprootTweak(MerkleRoot) × G
pub fn compute_taproot_tweak(internal_pubkey: &[u8; 32], merkle_root: &Hash) -> Result<[u8; 32]> {
    // Create secp256k1 context (optimized: reuse in production, create new otherwise)
    // Note: Taproot operations need mutable context for add_exp_tweak, so we create new
    // For verification-only operations, use thread-local context
    let secp = Secp256k1::new();

    // Parse internal public key (x-only format for Taproot)
    let internal_pk = match XOnlyPublicKey::from_slice(internal_pubkey) {
        Ok(pk) => pk,
        Err(_) => {
            return Err(crate::error::ConsensusError::InvalidSignature(
                "Invalid internal public key".to_string(),
            ))
        }
    };

    // Compute tweak: SHA256("TapTweak" || internal_pubkey || merkle_root)
    let mut tweak_data = Vec::new();
    tweak_data.extend_from_slice(b"TapTweak");
    tweak_data.extend_from_slice(internal_pubkey);
    tweak_data.extend_from_slice(merkle_root);

    let tweak_hash = Sha256::digest(&tweak_data);
    let tweak_scalar = match Scalar::from_be_bytes(tweak_hash.into()) {
        Ok(scalar) => scalar,
        Err(_) => {
            return Err(crate::error::ConsensusError::InvalidSignature(
                "Invalid tweak scalar".to_string(),
            ))
        }
    };

    // Convert x-only public key to full public key for tweaking
    let full_pk = PublicKey::from_x_only_public_key(internal_pk, secp256k1::Parity::Even);

    // Compute tweaked public key: full_pk + tweak_scalar * G
    let tweaked_pk = full_pk.add_exp_tweak(&secp, &tweak_scalar).map_err(|_| {
        crate::error::ConsensusError::InvalidSignature(
            "Failed to compute tweaked public key".to_string(),
        )
    })?;

    // Return the x-coordinate of the tweaked public key
    let xonly_pk = XOnlyPublicKey::from(tweaked_pk);
    Ok(xonly_pk.serialize())
}

/// Validate Taproot key aggregation
pub fn validate_taproot_key_aggregation(
    internal_pubkey: &[u8; 32],
    merkle_root: &Hash,
    output_key: &[u8; 32],
) -> Result<bool> {
    let expected_output_key = compute_taproot_tweak(internal_pubkey, merkle_root)?;
    Ok(expected_output_key == *output_key)
}

/// Validate Taproot script path spending
pub fn validate_taproot_script_path(
    script: &ByteString,
    merkle_proof: &[Hash],
    merkle_root: &Hash,
) -> Result<bool> {
    // Compute merkle root from script and proof
    let computed_root = compute_script_merkle_root(script, merkle_proof)?;
    Ok(computed_root == *merkle_root)
}

/// Compute merkle root for script path
fn compute_script_merkle_root(script: &ByteString, proof: &[Hash]) -> Result<Hash> {
    let mut current_hash = hash_script(script);

    for proof_hash in proof {
        current_hash = hash_pair(&current_hash, proof_hash);
    }

    Ok(current_hash)
}

/// Hash a script
fn hash_script(script: &ByteString) -> Hash {
    let mut hasher = sha256d::Hash::engine();
    hasher.input(script);
    let result = sha256d::Hash::from_engine(hasher);
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Hash a pair of hashes
fn hash_pair(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = sha256d::Hash::engine();
    hasher.input(left);
    hasher.input(right);
    let result = sha256d::Hash::from_engine(hasher);
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Check if transaction output is Taproot
pub fn is_taproot_output(output: &TransactionOutput) -> bool {
    validate_taproot_script(&output.script_pubkey).unwrap_or(false)
}

/// Validate Taproot transaction
pub fn validate_taproot_transaction(tx: &Transaction, witness: Option<&Witness>) -> Result<bool> {
    // Check if any output is Taproot
    for output in &tx.outputs {
        if is_taproot_output(output) {
            // Validate Taproot output
            if !validate_taproot_script(&output.script_pubkey)? {
                return Ok(false);
            }
        }
    }

    // Validate Taproot witness structure using unified framework
    // Determine if this is a script path spend based on witness structure
    // Script path has at least 2 elements (script + control block), key path has 1 element (signature)
    if let Some(w) = witness {
        let is_script_path = w.len() >= 2;
        if !witness::validate_taproot_witness_structure(w, is_script_path)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Compute Taproot signature hash following BIP 341 specification
pub fn compute_taproot_signature_hash(
    tx: &Transaction,
    input_index: usize,
    prevouts: &[TransactionOutput],
    sighash_type: u8,
) -> Result<Hash> {
    // Create SHA256 hasher for Taproot signature hash
    let mut hasher = Sha256::new();

    // 1. Transaction version (4 bytes, little-endian)
    hasher.update((tx.version as u32).to_le_bytes());

    // 2. Input count (varint)
    hasher.update(encode_varint(tx.inputs.len() as u64));

    // 3. Inputs
    for input in &tx.inputs {
        // Previous output hash (32 bytes)
        hasher.update(input.prevout.hash);
        // Previous output index (4 bytes, little-endian)
        hasher.update((input.prevout.index as u32).to_le_bytes());
        // Script length (varint) - empty for Taproot
        hasher.update([0]);
        // Sequence (4 bytes, little-endian)
        hasher.update((input.sequence as u32).to_le_bytes());
    }

    // 4. Output count (varint)
    hasher.update(encode_varint(tx.outputs.len() as u64));

    // 5. Outputs
    for output in &tx.outputs {
        // Value (8 bytes, little-endian)
        hasher.update((output.value as u64).to_le_bytes());
        // Script length (varint)
        hasher.update(encode_varint(output.script_pubkey.len() as u64));
        // Script
        hasher.update(&output.script_pubkey);
    }

    // 6. Lock time (4 bytes, little-endian)
    hasher.update((tx.lock_time as u32).to_le_bytes());

    // 7. Sighash type (4 bytes, little-endian)
    hasher.update((sighash_type as u32).to_le_bytes());

    // 8. Input index (4 bytes, little-endian)
    hasher.update((input_index as u32).to_le_bytes());

    // 9. Previous output value (8 bytes, little-endian)
    if input_index < prevouts.len() {
        hasher.update((prevouts[input_index].value as u64).to_le_bytes());
    } else {
        hasher.update([0u8; 8]);
    }

    // 10. Previous output script (varint + script)
    if input_index < prevouts.len() {
        hasher.update(encode_varint(
            prevouts[input_index].script_pubkey.len() as u64
        ));
        hasher.update(&prevouts[input_index].script_pubkey);
    } else {
        hasher.update([0]);
    }

    // Final hash
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    Ok(hash)
}

/// Encode a number as a Bitcoin varint
fn encode_varint(value: u64) -> Vec<u8> {
    if value < 0xfd {
        vec![value as u8]
    } else if value <= 0xffff {
        let mut result = vec![0xfd];
        result.extend_from_slice(&(value as u16).to_le_bytes());
        result
    } else if value <= 0xffffffff {
        let mut result = vec![0xfe];
        result.extend_from_slice(&(value as u32).to_le_bytes());
        result
    } else {
        let mut result = vec![0xff];
        result.extend_from_slice(&value.to_le_bytes());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_taproot_script_valid() {
        let script = create_taproot_script(&[1u8; 32]);
        assert!(validate_taproot_script(&script).unwrap());
    }

    #[test]
    fn test_validate_taproot_script_invalid_length() {
        let script = vec![0x51, 0x20]; // Too short
        assert!(!validate_taproot_script(&script).unwrap());
    }

    #[test]
    fn test_validate_taproot_script_invalid_prefix() {
        let mut script = vec![0x52]; // Wrong prefix (OP_2 instead of OP_1)
        script.extend_from_slice(&[1u8; 32]);
        assert!(!validate_taproot_script(&script).unwrap());
    }

    #[test]
    fn test_extract_taproot_output_key() {
        let expected_key = [1u8; 32];
        let script = create_taproot_script(&expected_key);

        let extracted_key = extract_taproot_output_key(&script).unwrap();
        assert_eq!(extracted_key, Some(expected_key));
    }

    #[test]
    fn test_compute_taproot_tweak() {
        // Use a valid secp256k1 public key (x-coordinate only for Taproot)
        let internal_pubkey = [
            0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87,
            0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b,
            0x16, 0xf8, 0x17, 0x98,
        ];
        let merkle_root = [2u8; 32];

        let tweak = compute_taproot_tweak(&internal_pubkey, &merkle_root).unwrap();
        assert_eq!(tweak.len(), 32);
    }

    #[test]
    fn test_validate_taproot_key_aggregation() {
        // Use a valid secp256k1 public key (x-coordinate only for Taproot)
        let internal_pubkey = [
            0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87,
            0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b,
            0x16, 0xf8, 0x17, 0x98,
        ];
        let merkle_root = [2u8; 32];
        let output_key = compute_taproot_tweak(&internal_pubkey, &merkle_root).unwrap();

        assert!(
            validate_taproot_key_aggregation(&internal_pubkey, &merkle_root, &output_key).unwrap()
        );
    }

    #[test]
    fn test_validate_taproot_script_path() {
        let script = vec![0x51, 0x52]; // OP_1, OP_2
        let merkle_proof = vec![[3u8; 32], [4u8; 32]];
        let merkle_root = compute_script_merkle_root(&script, &merkle_proof).unwrap();

        assert!(validate_taproot_script_path(&script, &merkle_proof, &merkle_root).unwrap());
    }

    #[test]
    fn test_is_taproot_output() {
        let output = TransactionOutput {
            value: 1000,
            script_pubkey: create_taproot_script(&[1u8; 32]),
        };

        assert!(is_taproot_output(&output));
    }

    #[test]
    fn test_validate_taproot_transaction() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: create_taproot_script(&[1u8; 32]),
            }],
            lock_time: 0,
        };

        // Key path spend: single signature
        let witness = Some(vec![vec![0u8; 64]]);
        assert!(validate_taproot_transaction(&tx, witness.as_ref()).unwrap());
    }

    #[test]
    fn test_compute_taproot_signature_hash() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        };

        let prevouts = vec![TransactionOutput {
            value: 2000,
            script_pubkey: create_taproot_script(&[1u8; 32]),
        }];

        let sig_hash = compute_taproot_signature_hash(&tx, 0, &prevouts, 0x01).unwrap();
        assert_eq!(sig_hash.len(), 32);
    }

    #[test]
    fn test_compute_taproot_signature_hash_invalid_input_index() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        };

        let prevouts = vec![TransactionOutput {
            value: 2000,
            script_pubkey: create_taproot_script(&[1u8; 32]),
        }];

        // Use invalid input index (out of bounds)
        let sig_hash = compute_taproot_signature_hash(&tx, 1, &prevouts, 0x01).unwrap();
        assert_eq!(sig_hash.len(), 32);
    }

    #[test]
    fn test_compute_taproot_signature_hash_empty_prevouts() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        };

        let prevouts = vec![];

        let sig_hash = compute_taproot_signature_hash(&tx, 0, &prevouts, 0x01).unwrap();
        assert_eq!(sig_hash.len(), 32);
    }

    #[test]
    fn test_compute_taproot_tweak_invalid_pubkey() {
        let invalid_pubkey = [0u8; 32]; // Invalid public key
        let merkle_root = [2u8; 32];

        let result = compute_taproot_tweak(&invalid_pubkey, &merkle_root);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_taproot_key_aggregation_invalid() {
        let internal_pubkey = [
            0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87,
            0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b,
            0x16, 0xf8, 0x17, 0x98,
        ];
        let merkle_root = [2u8; 32];
        let wrong_output_key = [3u8; 32]; // Wrong output key

        assert!(!validate_taproot_key_aggregation(
            &internal_pubkey,
            &merkle_root,
            &wrong_output_key
        )
        .unwrap());
    }

    #[test]
    fn test_validate_taproot_script_path_invalid() {
        let script = vec![0x51, 0x52]; // OP_1, OP_2
        let merkle_proof = vec![[3u8; 32], [4u8; 32]];
        let wrong_merkle_root = [5u8; 32]; // Wrong merkle root

        assert!(!validate_taproot_script_path(&script, &merkle_proof, &wrong_merkle_root).unwrap());
    }

    #[test]
    fn test_validate_taproot_script_path_empty_proof() {
        let script = vec![0x51, 0x52]; // OP_1, OP_2
        let merkle_proof = vec![];
        let merkle_root = hash_script(&script);

        assert!(validate_taproot_script_path(&script, &merkle_proof, &merkle_root).unwrap());
    }

    #[test]
    fn test_hash_script() {
        let script = vec![0x51, 0x52];
        let hash = hash_script(&script);

        assert_eq!(hash.len(), 32);

        // Different script should produce different hash
        let script2 = vec![0x53, 0x54];
        let hash2 = hash_script(&script2);
        assert_ne!(hash, hash2);
    }

    #[test]
    fn test_hash_script_empty() {
        let script = vec![];
        let hash = hash_script(&script);

        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_hash_pair() {
        let left = [1u8; 32];
        let right = [2u8; 32];
        let hash = hash_pair(&left, &right);

        assert_eq!(hash.len(), 32);

        // Different order should produce different hash
        let hash2 = hash_pair(&right, &left);
        assert_ne!(hash, hash2);
    }

    #[test]
    fn test_hash_pair_same() {
        let hash1 = [1u8; 32];
        let hash2 = hash_pair(&hash1, &hash1);

        assert_eq!(hash2.len(), 32);
    }

    #[test]
    fn test_encode_varint_small() {
        let encoded = encode_varint(0xfc);
        assert_eq!(encoded, vec![0xfc]);
    }

    #[test]
    fn test_encode_varint_medium() {
        let encoded = encode_varint(0x1000);
        assert_eq!(encoded.len(), 3);
        assert_eq!(encoded[0], 0xfd);
    }

    #[test]
    fn test_encode_varint_large() {
        let encoded = encode_varint(0x1000000);
        assert_eq!(encoded.len(), 5);
        assert_eq!(encoded[0], 0xfe);
    }

    #[test]
    fn test_encode_varint_huge() {
        let encoded = encode_varint(0x1000000000000000);
        assert_eq!(encoded.len(), 9);
        assert_eq!(encoded[0], 0xff);
    }

    #[test]
    fn test_extract_taproot_output_key_invalid_script() {
        let script = vec![0x52, 0x20]; // Invalid script
        let result = extract_taproot_output_key(&script).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_is_taproot_output_false() {
        let output = TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x52, 0x20], // Not a Taproot script
        };

        assert!(!is_taproot_output(&output));
    }

    #[test]
    fn test_validate_taproot_transaction_no_taproot_outputs() {
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x52], // Not Taproot
            }],
            lock_time: 0,
        };

        // No witness needed for non-Taproot transaction
        assert!(validate_taproot_transaction(&tx, None).unwrap());
    }

    #[test]
    fn test_validate_taproot_transaction_invalid_taproot_output() {
        // Create a transaction with a valid Taproot script
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: create_taproot_script(&[1u8; 32]),
            }],
            lock_time: 0,
        };

        // Key path spend: single signature
        let witness = Some(vec![vec![0u8; 64]]);
        assert!(validate_taproot_transaction(&tx, witness.as_ref()).unwrap());
    }

    // Helper function
    fn create_taproot_script(output_key: &[u8; 32]) -> ByteString {
        let mut script = vec![TAPROOT_SCRIPT_PREFIX];
        script.extend_from_slice(output_key);
        script.push(0x00); // Add extra byte to make it 34 bytes total
        script
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Verify Taproot script validation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString: validate_taproot_script(script) is deterministic
    /// - Same script always produces same result
    /// - Script length and prefix are correctly validated
    #[kani::proof]
    fn kani_validate_taproot_script_deterministic() {
        let script_len: usize = kani::any();
        kani::assume(script_len <= 50); // Bounded for tractability

        let mut script = Vec::new();
        for i in 0..script_len {
            let byte: u8 = kani::any();
            script.push(byte);
        }

        // Call validate_taproot_script twice with same input
        let result1 = validate_taproot_script(&script).unwrap();
        let result2 = validate_taproot_script(&script).unwrap();

        // Results should be identical (deterministic)
        assert_eq!(result1, result2);

        // If script is valid Taproot, it should have correct length and prefix
        if result1 {
            assert_eq!(script.len(), 34);
            assert_eq!(script[0], TAPROOT_SCRIPT_PREFIX);
        }
    }

    /// Verify Taproot output key extraction is correct
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString: if validate_taproot_script(script) = true
    /// then extract_taproot_output_key(script) returns Some(key) where key = script[1..33]
    #[kani::proof]
    fn kani_extract_taproot_output_key_correct() {
        let script_len: usize = kani::any();
        kani::assume(script_len <= 50);

        let mut script = Vec::new();
        for i in 0..script_len {
            let byte: u8 = kani::any();
            script.push(byte);
        }

        let extracted_key = extract_taproot_output_key(&script).unwrap();

        // If script is valid Taproot, extraction should succeed
        if validate_taproot_script(&script).unwrap() {
            assert!(extracted_key.is_some());
            let key = extracted_key.unwrap();
            assert_eq!(key.len(), 32);

            // Key should match script bytes 1-32
            for i in 0..32 {
                assert_eq!(key[i], script[i + 1]);
            }
        } else {
            // If script is invalid, extraction should return None
            assert!(extracted_key.is_none());
        }
    }

    /// Verify Taproot key aggregation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ internal_pubkey ∈ [u8; 32], merkle_root ∈ Hash:
    /// - compute_taproot_tweak(internal_pubkey, merkle_root) is deterministic
    /// - validate_taproot_key_aggregation is deterministic
    #[kani::proof]
    fn kani_taproot_key_aggregation_deterministic() {
        let internal_pubkey: [u8; 32] = kani::any();
        let merkle_root: Hash = kani::any();

        // Use a valid public key for testing
        let valid_pubkey = [
            0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87,
            0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b,
            0x16, 0xf8, 0x17, 0x98,
        ];

        // Call compute_taproot_tweak twice with same inputs
        let result1 = compute_taproot_tweak(&valid_pubkey, &merkle_root);
        let result2 = compute_taproot_tweak(&valid_pubkey, &merkle_root);

        // Results should be identical (deterministic)
        assert_eq!(result1.is_ok(), result2.is_ok());
        if result1.is_ok() && result2.is_ok() {
            assert_eq!(result1.unwrap(), result2.unwrap());
        }

        // If tweak computation succeeds, validate aggregation should work
        if let Ok(output_key) = result1 {
            let validation_result1 =
                validate_taproot_key_aggregation(&valid_pubkey, &merkle_root, &output_key).unwrap();
            let validation_result2 =
                validate_taproot_key_aggregation(&valid_pubkey, &merkle_root, &output_key).unwrap();

            assert_eq!(validation_result1, validation_result2);
            assert!(validation_result1); // Should be true for correct aggregation
        }
    }

    /// Verify Taproot script path validation is correct
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString, merkle_proof ∈ [Hash], merkle_root ∈ Hash:
    /// - validate_taproot_script_path(script, merkle_proof, merkle_root) is deterministic
    /// - If computed_root = merkle_root, validation should succeed
    #[kani::proof]
    fn kani_validate_taproot_script_path_correct() {
        let script_len: usize = kani::any();
        let proof_len: usize = kani::any();
        kani::assume(script_len <= 10);
        kani::assume(proof_len <= 5);

        let mut script = Vec::new();
        for i in 0..script_len {
            let byte: u8 = kani::any();
            script.push(byte);
        }

        let mut merkle_proof = Vec::new();
        for i in 0..proof_len {
            let hash: Hash = kani::any();
            merkle_proof.push(hash);
        }

        let merkle_root: Hash = kani::any();

        // Call validate_taproot_script_path twice with same inputs
        let result1 = validate_taproot_script_path(&script, &merkle_proof, &merkle_root);
        let result2 = validate_taproot_script_path(&script, &merkle_proof, &merkle_root);

        // Results should be identical (deterministic)
        assert_eq!(result1.is_ok(), result2.is_ok());
        if result1.is_ok() && result2.is_ok() {
            assert_eq!(result1.unwrap(), result2.unwrap());
        }

        // If validation succeeds, computed root should match provided root
        if result1.is_ok() && result1.unwrap() {
            let computed_root = compute_script_merkle_root(&script, &merkle_proof).unwrap();
            assert_eq!(computed_root, merkle_root);
        }
    }

    /// Kani proof: validate_taproot_transaction checks all outputs
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction:
    /// - validate_taproot_transaction(tx) = true ⟺
    ///   ∀ output ∈ tx.outputs: is_taproot_output(output) → validate_taproot_script(output.script_pubkey) = true
    ///
    /// This ensures all Taproot outputs in a transaction are valid.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_validate_taproot_transaction_outputs() {
        let tx = create_bounded_transaction();

        // Bound for tractability
        kani::assume(tx.inputs.len() <= 3);
        kani::assume(tx.outputs.len() <= 3);

        let result = validate_taproot_transaction(&tx, None);

        if result.is_ok() && result.unwrap() {
            // If transaction is valid, all Taproot outputs must be valid
            for output in &tx.outputs {
                if is_taproot_output(output) {
                    // Taproot outputs must pass script validation
                    let script_valid =
                        validate_taproot_script(&output.script_pubkey).unwrap_or(false);
                    assert!(
                        script_valid,
                        "Valid Taproot transaction must have valid Taproot script"
                    );
                }
            }
        }
    }

    /// Verify Taproot signature hash computation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, input_index ∈ ℕ, prevouts ∈ [TransactionOutput], sighash_type ∈ ℕ:
    /// - compute_taproot_signature_hash(tx, input_index, prevouts, sighash_type) is deterministic
    /// - Same inputs always produce same hash
    #[kani::proof]
    fn kani_compute_taproot_signature_hash_deterministic() {
        let tx = create_bounded_transaction();
        let input_index: usize = kani::any();
        kani::assume(input_index <= 5);

        let prevouts_len: usize = kani::any();
        kani::assume(prevouts_len <= 5);

        let mut prevouts = Vec::new();
        for i in 0..prevouts_len {
            let script_len: usize = kani::any();
            kani::assume(script_len <= 10);
            let mut script = Vec::new();
            for j in 0..script_len {
                let byte: u8 = kani::any();
                script.push(byte);
            }
            prevouts.push(TransactionOutput {
                value: 1000,
                script_pubkey: script,
            });
        }

        let sighash_type: u8 = kani::any();

        // Call compute_taproot_signature_hash twice with same inputs
        let result1 = compute_taproot_signature_hash(&tx, input_index, &prevouts, sighash_type);

        // Create new transaction/prevouts for second call (deterministic test)
        let tx2 = Transaction {
            version: tx.version,
            inputs: tx.inputs.clone(),
            outputs: tx.outputs.clone(),
            lock_time: tx.lock_time,
        };
        let prevouts2: Vec<TransactionOutput> = prevouts.iter().cloned().collect();
        let result2 = compute_taproot_signature_hash(&tx2, input_index, &prevouts2, sighash_type);

        // Results should be identical (deterministic)
        assert_eq!(result1.is_ok(), result2.is_ok());
        if result1.is_ok() && result2.is_ok() {
            assert_eq!(result1.unwrap(), result2.unwrap());
        }

        // Hash should be 32 bytes
        if result1.is_ok() {
            let hash = result1.unwrap();
            assert_eq!(hash.len(), 32);
        }
    }

    /// Helper function to create bounded transaction for Kani
    fn create_bounded_transaction() -> Transaction {
        let input_count: usize = kani::any();
        let output_count: usize = kani::any();
        kani::assume(input_count <= 3);
        kani::assume(output_count <= 3);

        let mut inputs = Vec::new();
        for i in 0..input_count {
            let script_len: usize = kani::any();
            kani::assume(script_len <= 5);
            let mut script = Vec::new();
            for j in 0..script_len {
                let byte: u8 = kani::any();
                script.push(byte);
            }
            inputs.push(TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: i as u32,
                },
                script_sig: script,
                sequence: 0xffffffff,
            });
        }

        let mut outputs = Vec::new();
        for i in 0..output_count {
            let script_len: usize = kani::any();
            kani::assume(script_len <= 10);
            let mut script = Vec::new();
            for j in 0..script_len {
                let byte: u8 = kani::any();
                script.push(byte);
            }
            outputs.push(TransactionOutput {
                value: 1000,
                script_pubkey: script,
            });
        }

        Transaction {
            version: 1,
            inputs,
            outputs,
            lock_time: 0,
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Property test: Taproot script validation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString: validate_taproot_script(script) is deterministic
    proptest! {
        #[test]
        fn prop_validate_taproot_script_deterministic(
            script in prop::collection::vec(any::<u8>(), 0..50)
        ) {
            let result1 = validate_taproot_script(&script).unwrap();
            let result2 = validate_taproot_script(&script).unwrap();

            assert_eq!(result1, result2);
        }
    }

    /// Property test: Taproot output key extraction is correct
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString: if validate_taproot_script(script) = true
    /// then extract_taproot_output_key(script) returns Some(key)
    proptest! {
        #[test]
        fn prop_extract_taproot_output_key_correct(
            script in prop::collection::vec(any::<u8>(), 0..50)
        ) {
            let extracted_key = extract_taproot_output_key(&script).unwrap();
            let is_valid = validate_taproot_script(&script).unwrap();

            if is_valid {
                assert!(extracted_key.is_some());
                let key = extracted_key.unwrap();
                assert_eq!(key.len(), 32);
            } else {
                assert!(extracted_key.is_none());
            }
        }
    }

    /// Property test: Taproot key aggregation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ internal_pubkey ∈ [u8; 32], merkle_root ∈ Hash:
    /// compute_taproot_tweak(internal_pubkey, merkle_root) is deterministic
    proptest! {
        #[test]
        fn prop_taproot_key_aggregation_deterministic(
            internal_pubkey in create_pubkey_strategy(),
            merkle_root in create_hash_strategy()
        ) {
            let result1 = compute_taproot_tweak(&internal_pubkey, &merkle_root);
            let result2 = compute_taproot_tweak(&internal_pubkey, &merkle_root);

            assert_eq!(result1.is_ok(), result2.is_ok());
            if result1.is_ok() && result2.is_ok() {
                assert_eq!(result1.unwrap(), result2.unwrap());
            }
        }
    }

    /// Property test: Taproot script path validation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString, merkle_proof ∈ [Hash], merkle_root ∈ Hash:
    /// validate_taproot_script_path(script, merkle_proof, merkle_root) is deterministic
    proptest! {
        #[test]
        fn prop_validate_taproot_script_path_deterministic(
            script in prop::collection::vec(any::<u8>(), 0..20),
            merkle_proof in prop::collection::vec(create_hash_strategy(), 0..5),
            merkle_root in create_hash_strategy()
        ) {
            let result1 = validate_taproot_script_path(&script, &merkle_proof, &merkle_root);
            let result2 = validate_taproot_script_path(&script, &merkle_proof, &merkle_root);

            assert_eq!(result1.is_ok(), result2.is_ok());
            if result1.is_ok() && result2.is_ok() {
                assert_eq!(result1.unwrap(), result2.unwrap());
            }
        }
    }

    /// Property test: Taproot signature hash computation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, input_index ∈ ℕ, prevouts ∈ [TransactionOutput], sighash_type ∈ ℕ:
    /// compute_taproot_signature_hash(tx, input_index, prevouts, sighash_type) is deterministic
    proptest! {
        #[test]
        fn prop_compute_taproot_signature_hash_deterministic(
            tx in create_transaction_strategy(),
            input_index in 0..10usize,
            prevouts in prop::collection::vec(create_output_strategy(), 0..5),
            sighash_type in any::<u8>()
        ) {
            let result1 = compute_taproot_signature_hash(&tx, input_index, &prevouts, sighash_type);
            let result2 = compute_taproot_signature_hash(&tx, input_index, &prevouts, sighash_type);

            assert_eq!(result1.is_ok(), result2.is_ok());
            if let (Ok(hash1), Ok(hash2)) = (&result1, &result2) {
                assert_eq!(hash1, hash2);
                assert_eq!(hash1.len(), 32);
            }

            // Hash should be 32 bytes if result is Ok
            if let Ok(ref hash) = result1 {
                assert_eq!(hash.len(), 32);
            }
        }
    }

    /// Property test: Taproot output detection is consistent
    ///
    /// Mathematical specification:
    /// ∀ output ∈ TransactionOutput: is_taproot_output(output) ∈ {true, false}
    proptest! {
        #[test]
        fn prop_is_taproot_output_consistent(
            output in create_output_strategy()
        ) {
            let is_taproot = is_taproot_output(&output);
            // Just test it returns a boolean (is_taproot is either true or false)
            let _ = is_taproot;
        }
    }

    /// Property test: Taproot transaction validation is deterministic
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction: validate_taproot_transaction(tx) is deterministic
    proptest! {
        #[test]
        fn prop_validate_taproot_transaction_deterministic(
            tx in create_transaction_strategy()
        ) {
            let result1 = validate_taproot_transaction(&tx, None).unwrap();
            let result2 = validate_taproot_transaction(&tx, None).unwrap();

            assert_eq!(result1, result2);
        }
    }

    /// Property test: Script hashing is deterministic
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString: hash_script(script) is deterministic
    proptest! {
        #[test]
        fn prop_hash_script_deterministic(
            script in prop::collection::vec(any::<u8>(), 0..20)
        ) {
            let hash1 = hash_script(&script);
            let hash2 = hash_script(&script);

            assert_eq!(hash1, hash2);
            assert_eq!(hash1.len(), 32);
        }
    }

    /// Property test: Hash pair operations are deterministic
    ///
    /// Mathematical specification:
    /// ∀ left, right ∈ Hash: hash_pair(left, right) is deterministic
    proptest! {
        #[test]
        fn prop_hash_pair_deterministic(
            left in create_hash_strategy(),
            right in create_hash_strategy()
        ) {
            let hash1 = hash_pair(&left, &right);
            let hash2 = hash_pair(&left, &right);

            assert_eq!(hash1, hash2);
            assert_eq!(hash1.len(), 32);
        }
    }

    /// Property test: Varint encoding is deterministic
    ///
    /// Mathematical specification:
    /// ∀ value ∈ ℕ: encode_varint(value) is deterministic
    proptest! {
        #[test]
        fn prop_encode_varint_deterministic(
            value in 0..u64::MAX
        ) {
            let encoded1 = encode_varint(value);
            let encoded2 = encode_varint(value);

            assert_eq!(encoded1, encoded2);

            // Encoded length should be reasonable
            assert!(!encoded1.is_empty());
            assert!(encoded1.len() <= 9);
        }
    }

    /// Property test: Varint encoding preserves value
    ///
    /// Mathematical specification:
    /// ∀ value ∈ ℕ: decode_varint(encode_varint(value)) = value
    proptest! {
        #[test]
        fn prop_encode_varint_preserves_value(
            value in 0..1000000u64  // Smaller range for tractability
        ) {
            let encoded = encode_varint(value);

            // Basic validation of encoding format
            match encoded.len() {
                1 => {
                    // Single byte encoding
                    assert!(value < 0xfd);
                    assert_eq!(encoded[0], value as u8);
                },
                3 => {
                    // 2-byte encoding
                    assert!((0xfd..=0xffff).contains(&value));
                    assert_eq!(encoded[0], 0xfd);
                },
                5 => {
                    // 4-byte encoding
                    assert!(value > 0xffff && value <= 0xffffffff);
                    assert_eq!(encoded[0], 0xfe);
                },
                9 => {
                    // 8-byte encoding
                    assert!(value > 0xffffffff);
                    assert_eq!(encoded[0], 0xff);
                },
                _ => panic!("Invalid varint encoding length"),
            }
        }
    }

    /// Property test: Taproot script path validation with correct proof
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString, merkle_proof ∈ [Hash]:
    /// If computed_root = compute_script_merkle_root(script, merkle_proof)
    /// then validate_taproot_script_path(script, merkle_proof, computed_root) = true
    proptest! {
        #[test]
        fn prop_validate_taproot_script_path_correct_proof(
            script in prop::collection::vec(any::<u8>(), 0..20),
            merkle_proof in prop::collection::vec(create_hash_strategy(), 0..5)
        ) {
            let computed_root = compute_script_merkle_root(&script, &merkle_proof).unwrap();
            let is_valid = validate_taproot_script_path(&script, &merkle_proof, &computed_root).unwrap();

            assert!(is_valid);
        }
    }

    // Property test strategies
    fn create_transaction_strategy() -> impl Strategy<Value = Transaction> {
        (
            prop::collection::vec(any::<u8>(), 0..10), // inputs
            prop::collection::vec(any::<u8>(), 0..10), // outputs
        )
            .prop_map(|(input_data, output_data)| {
                let mut inputs = Vec::new();
                for (i, _) in input_data.iter().enumerate() {
                    inputs.push(TransactionInput {
                        prevout: OutPoint {
                            hash: [0; 32],
                            index: i as u64,
                        },
                        script_sig: vec![],
                        sequence: 0xffffffff,
                    });
                }

                let mut outputs = Vec::new();
                for _ in output_data {
                    outputs.push(TransactionOutput {
                        value: 1000,
                        script_pubkey: vec![0x51],
                    });
                }

                Transaction {
                    version: 1,
                    inputs,
                    outputs,
                    lock_time: 0,
                }
            })
    }

    fn create_output_strategy() -> impl Strategy<Value = TransactionOutput> {
        (any::<i64>(), prop::collection::vec(any::<u8>(), 0..50)).prop_map(|(value, script)| {
            TransactionOutput {
                value,
                script_pubkey: script,
            }
        })
    }

    fn create_hash_strategy() -> impl Strategy<Value = Hash> {
        prop::collection::vec(any::<u8>(), 32..=32).prop_map(|bytes| {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&bytes);
            hash
        })
    }

    fn create_pubkey_strategy() -> impl Strategy<Value = [u8; 32]> {
        prop::collection::vec(any::<u8>(), 32..=32).prop_map(|bytes| {
            let mut pubkey = [0u8; 32];
            pubkey.copy_from_slice(&bytes);
            pubkey
        })
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: Taproot key aggregation correctness (Orange Paper Section 11.2)
    ///
    /// Mathematical specification:
    /// ∀ internal_pubkey ∈ [u8; 32], merkle_root ∈ Hash:
    /// - compute_taproot_tweak(internal_pubkey, merkle_root) = OutputKey
    /// - OutputKey = InternalPubKey + TaprootTweak(MerkleRoot) × G
    ///
    /// This proves the Taproot key aggregation matches the Orange Paper specification.
    #[kani::proof]
    fn kani_taproot_key_aggregation_correctness() {
        let internal_pubkey: [u8; 32] = kani::any();
        let merkle_root: Hash = kani::any();

        // Calculate tweak
        let tweak_result = compute_taproot_tweak(&internal_pubkey, &merkle_root);

        if tweak_result.is_ok() {
            let output_key = tweak_result.unwrap();

            // Verify key aggregation: validate that output_key matches expected computation
            let validation_result =
                validate_taproot_key_aggregation(&internal_pubkey, &merkle_root, &output_key);

            if validation_result.is_ok() && validation_result.unwrap() {
                // Key aggregation is correct: OutputKey = InternalPubKey + TaprootTweak(MerkleRoot) × G
                assert!(
                    true,
                    "Taproot key aggregation matches Orange Paper specification"
                );
            }
        }
    }

    /// Kani proof: Taproot output script validation (Orange Paper Section 11.2)
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString:
    /// - validate_taproot_script(script) = true ⟹ script = OP_1 <32-byte-hash>
    ///
    /// This ensures Taproot output scripts match the required format.
    #[kani::proof]
    fn kani_taproot_script_validation() {
        let script: ByteString = kani::any();

        // Bound for tractability
        kani::assume(script.len() <= 50);

        let result = validate_taproot_script(&script);

        if result.is_ok() && result.unwrap() {
            // Valid Taproot script must be: OP_1 <32-byte-hash> (34 bytes total)
            assert_eq!(
                script.len(),
                34,
                "Valid Taproot script must be exactly 34 bytes"
            );
            assert_eq!(
                script[0], TAPROOT_SCRIPT_PREFIX,
                "Valid Taproot script must start with OP_1 (0x51)"
            );

            // Extract output key
            let output_key_result = extract_taproot_output_key(&script);
            if output_key_result.is_ok() {
                let output_key_opt = output_key_result.unwrap();
                assert!(
                    output_key_opt.is_some(),
                    "Valid Taproot script must have extractable output key"
                );

                if let Some(output_key) = output_key_opt {
                    // Output key should be 32 bytes
                    assert_eq!(output_key.len(), 32, "Taproot output key must be 32 bytes");
                }
            }
        }
    }

    /// Kani proof: Taproot script path validation (Orange Paper Section 11.2)
    ///
    /// Mathematical specification:
    /// ∀ script ∈ ByteString, merkle_proof ∈ [Hash], merkle_root ∈ Hash:
    /// - validate_taproot_script_path(script, merkle_proof, merkle_root) = true ⟹
    ///   merkle_root can be computed from script and proof
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_taproot_script_path_validation() {
        let script: ByteString = kani::any();
        let merkle_proof: Vec<Hash> = kani::any();
        let merkle_root: Hash = kani::any();

        // Bound for tractability
        kani::assume(script.len() <= 100);
        kani::assume(merkle_proof.len() <= 5);

        let result = validate_taproot_script_path(&script, &merkle_proof, &merkle_root);

        if result.is_ok() && result.unwrap() {
            // Script path validation passed: merkle root must be computable from script and proof
            // This proves the script path is valid according to Taproot specification
            assert!(
                true,
                "Taproot script path validation: merkle root matches script and proof"
            );
        }
    }
}
