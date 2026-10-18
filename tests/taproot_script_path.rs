//! Taproot script path validation edge cases
//!
//! Tests for Taproot script path spending validation:
//! - Merkle proof validation
//! - Invalid merkle proof rejection
//! - Empty scripts
//! - Depth limits
//! - Key aggregation edge cases
//! - Witness size limits
//!
//! Consensus-critical: Taproot validation bugs can cause consensus divergence.

use blvm_consensus::opcodes::{OP_1, OP_2};
use blvm_consensus::taproot::{
    TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root, compute_taproot_tweak,
    validate_taproot_key_aggregation, validate_taproot_script_path,
};
use blvm_consensus::types::Hash;
use blvm_consensus::witness::validate_taproot_witness_structure;

/// Test Taproot script path merkle proof validation
#[test]
fn test_taproot_script_path_merkle_proof() {
    // Create a script
    let script = vec![OP_1];
    let merkle_proof: Vec<Hash> = vec![];
    let merkle_root =
        compute_script_merkle_root(&script, &merkle_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    assert!(validate_taproot_script_path(&script, &merkle_proof, &merkle_root).unwrap());
    assert!(!validate_taproot_script_path(&script, &merkle_proof, &[0x42; 32]).unwrap());
}

/// Test Taproot script path with invalid merkle proof
#[test]
fn test_taproot_script_path_invalid_proof() {
    let script = vec![OP_1]; // OP_1

    // Create invalid merkle proof (wrong hash)
    let invalid_proof: Vec<Hash> = vec![[0xff; 32]];

    // Merkle root doesn't match
    let merkle_root: Hash = [0x42; 32];

    assert!(!validate_taproot_script_path(&script, &invalid_proof, &merkle_root).unwrap());
}

/// Test Taproot script path with empty scripts
#[test]
fn test_taproot_script_path_empty_script() {
    // Empty script should still validate if merkle proof is correct
    let empty_script = vec![];

    let merkle_proof: Vec<Hash> = vec![];
    let merkle_root =
        compute_script_merkle_root(&empty_script, &merkle_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT)
            .unwrap();
    assert!(validate_taproot_script_path(&empty_script, &merkle_proof, &merkle_root).unwrap());
    assert!(!validate_taproot_script_path(&empty_script, &merkle_proof, &[0x42; 32]).unwrap());
}

/// Test Taproot script path depth limits
#[test]
fn test_taproot_script_path_depth_limits() {
    // Taproot merkle tree can have up to 128 levels (2^128 scripts)
    // Test with various proof depths

    let script = vec![OP_1]; // OP_1

    // Test with shallow proof (few levels)
    let shallow_proof: Vec<Hash> = vec![[0x01; 32], [0x02; 32]];

    // Test with deep proof (many levels)
    let mut deep_proof = Vec::new();
    for i in 0..10 {
        deep_proof.push([i as u8; 32]);
    }

    let shallow_root =
        compute_script_merkle_root(&script, &shallow_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT)
            .unwrap();
    let deep_root =
        compute_script_merkle_root(&script, &deep_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    assert!(validate_taproot_script_path(&script, &shallow_proof, &shallow_root).unwrap());
    assert!(validate_taproot_script_path(&script, &deep_proof, &deep_root).unwrap());
    assert!(!validate_taproot_script_path(&script, &shallow_proof, &deep_root).unwrap());
}

/// Test Taproot key aggregation edge cases
#[test]
fn test_taproot_key_aggregation() {
    // Internal public key (32 bytes, x-only)
    let internal_pubkey: [u8; 32] = [0x01; 32];

    // Merkle root (32 bytes)
    let merkle_root: Hash = [0x02; 32];

    // Compute tweaked output key
    let output_key = compute_taproot_tweak(&internal_pubkey, &merkle_root);

    // Should compute tweak successfully
    assert!(output_key.is_ok());
    let output_key_value = output_key.unwrap();

    // Output key should be 32 bytes
    assert_eq!(output_key_value.len(), 32);

    // Validate key aggregation — use with_parity to get correct parity for this key
    // just to satisfy imports
    let (expected_key, parity) = blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(
        &internal_pubkey,
        &merkle_root,
    )
    .unwrap();
    let result =
        validate_taproot_key_aggregation(&internal_pubkey, &merkle_root, &expected_key, parity);
    assert!(result.is_ok());
    assert!(result.unwrap()); // Should match
}

/// Test Taproot key aggregation with wrong output key
#[test]
fn test_taproot_key_aggregation_wrong_key() {
    let internal_pubkey: [u8; 32] = [0x01; 32];
    let merkle_root: Hash = [0x02; 32];
    let wrong_output_key: [u8; 32] = [0xff; 32]; // Wrong key

    // Should fail validation
    let result =
        validate_taproot_key_aggregation(&internal_pubkey, &merkle_root, &wrong_output_key, 0);
    assert!(result.is_ok());
    assert!(!result.unwrap()); // Should not match
}

/// Test Taproot witness size limits
#[test]
fn test_taproot_witness_size_limits() {
    // Taproot witness can include:
    // - Control block (33 bytes minimum)
    // - Script execution stack items
    // - Signature(s)

    // Test with maximum witness size
    let script = vec![0x51];
    let control = vec![0xc0; 33];
    assert!(validate_taproot_witness_structure(&vec![script.clone(), control], true).unwrap());
    assert!(!validate_taproot_witness_structure(&vec![script, vec![0x42; 520]], true).unwrap());
}

/// Test Taproot script path with multiple scripts
#[test]
fn test_taproot_script_path_multiple_scripts() {
    // Taproot can have multiple script paths in merkle tree
    // Each script needs its own merkle proof

    let script1 = vec![OP_1];
    let script2 = vec![OP_2];
    let proof1: Vec<Hash> = vec![[0x01; 32]];
    let proof2: Vec<Hash> = vec![[0x02; 32]];
    let root1 =
        compute_script_merkle_root(&script1, &proof1, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    let root2 =
        compute_script_merkle_root(&script2, &proof2, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    assert!(validate_taproot_script_path(&script1, &proof1, &root1).unwrap());
    assert!(validate_taproot_script_path(&script2, &proof2, &root2).unwrap());
    assert!(!validate_taproot_script_path(&script1, &proof1, &root2).unwrap());
}

/// Test Taproot key aggregation with empty merkle root
#[test]
fn test_taproot_key_aggregation_empty_merkle() {
    // Taproot key-only spending (no script paths) uses empty merkle root
    let internal_pubkey: [u8; 32] = [0x01; 32];
    let empty_merkle_root: Hash = [0x00; 32];

    // Compute tweak with empty merkle root
    let output_key = compute_taproot_tweak(&internal_pubkey, &empty_merkle_root);

    // Should compute successfully
    assert!(output_key.is_ok());

    // Validate key aggregation
    let (expected_key, parity) = blvm_consensus::secp256k1_backend::taproot_output_key_with_parity(
        &internal_pubkey,
        &empty_merkle_root,
    )
    .unwrap();
    let result = validate_taproot_key_aggregation(
        &internal_pubkey,
        &empty_merkle_root,
        &expected_key,
        parity,
    );
    assert!(result.unwrap());
}

/// Test Taproot script path boundary conditions
#[test]
fn test_taproot_script_path_boundaries() {
    // Test merkle proof with single hash
    let script = vec![OP_1];
    let single_proof: Vec<Hash> = vec![[0x01; 32]];
    let merkle_root: Hash = [0x42; 32];

    let single_root =
        compute_script_merkle_root(&script, &single_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    assert!(validate_taproot_script_path(&script, &single_proof, &single_root).unwrap());
    assert!(!validate_taproot_script_path(&script, &single_proof, &merkle_root).unwrap());

    let empty_proof: Vec<Hash> = vec![];
    let empty_root =
        compute_script_merkle_root(&script, &empty_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();
    assert!(validate_taproot_script_path(&script, &empty_proof, &empty_root).unwrap());
}

/// Test Taproot control block validation
#[test]
fn test_taproot_control_block() {
    // Control block format:
    // - Version byte (0xc0 or 0xc1)
    // - Internal public key (32 bytes)
    // - Merkle proof (variable length)

    // Minimum control block: 33 bytes (version + internal key, no proof)
    let script = vec![0x51];
    assert!(
        validate_taproot_witness_structure(&vec![script.clone(), vec![0xc0; 33]], true).unwrap()
    );
    assert!(
        validate_taproot_witness_structure(&vec![script.clone(), vec![0xc0; 33 + 32]], true)
            .unwrap()
    );
    let wide = 33 + 5 * 32;
    let within = wide <= blvm_consensus::taproot::TAPROOT_CONTROL_MAX_SIZE;
    assert_eq!(
        validate_taproot_witness_structure(&vec![script, vec![0xc0; wide]], true).unwrap(),
        within
    );
}
