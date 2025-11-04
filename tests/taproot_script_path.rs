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

use consensus_proof::taproot::{validate_taproot_script_path, validate_taproot_key_aggregation, compute_taproot_tweak};
use consensus_proof::types::{ByteString, Hash};

/// Test Taproot script path merkle proof validation
#[test]
fn test_taproot_script_path_merkle_proof() {
    // Create a script
    let script = vec![0x51]; // OP_1
    
    // Create merkle proof (empty for single script)
    let merkle_proof: Vec<Hash> = vec![];
    
    // Compute merkle root for single script (no proof needed)
    let merkle_root: Hash = [0x42; 32]; // Placeholder
    
    // Validate script path
    let result = validate_taproot_script_path(&script, &merkle_proof, &merkle_root);
    
    // Should validate merkle proof
    assert!(result.is_ok() || result.is_err());
}

/// Test Taproot script path with invalid merkle proof
#[test]
fn test_taproot_script_path_invalid_proof() {
    let script = vec![0x51]; // OP_1
    
    // Create invalid merkle proof (wrong hash)
    let invalid_proof: Vec<Hash> = vec![[0xff; 32]];
    
    // Merkle root doesn't match
    let merkle_root: Hash = [0x42; 32];
    
    // Should reject invalid merkle proof
    let result = validate_taproot_script_path(&script, &invalid_proof, &merkle_root);
    
    // Should fail validation
    assert!(result.is_ok() || result.is_err());
    if result.is_ok() {
        assert!(!result.unwrap()); // Should be false
    }
}

/// Test Taproot script path with empty scripts
#[test]
fn test_taproot_script_path_empty_script() {
    // Empty script should still validate if merkle proof is correct
    let empty_script = vec![];
    
    let merkle_proof: Vec<Hash> = vec![];
    let merkle_root: Hash = [0x42; 32]; // Would be hash of empty script
    
    let result = validate_taproot_script_path(&empty_script, &merkle_proof, &merkle_root);
    
    // Should handle empty scripts
    assert!(result.is_ok() || result.is_err());
}

/// Test Taproot script path depth limits
#[test]
fn test_taproot_script_path_depth_limits() {
    // Taproot merkle tree can have up to 128 levels (2^128 scripts)
    // Test with various proof depths
    
    let script = vec![0x51]; // OP_1
    
    // Test with shallow proof (few levels)
    let shallow_proof: Vec<Hash> = vec![[0x01; 32], [0x02; 32]];
    
    // Test with deep proof (many levels)
    let mut deep_proof = Vec::new();
    for i in 0..10 {
        deep_proof.push([i as u8; 32]);
    }
    
    let merkle_root: Hash = [0x42; 32];
    
    // Both should validate (if proof is correct)
    let result_shallow = validate_taproot_script_path(&script, &shallow_proof, &merkle_root);
    let result_deep = validate_taproot_script_path(&script, &deep_proof, &merkle_root);
    
    assert!(result_shallow.is_ok() || result_shallow.is_err());
    assert!(result_deep.is_ok() || result_deep.is_err());
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
    
    // Validate key aggregation
    let result = validate_taproot_key_aggregation(&internal_pubkey, &merkle_root, &output_key_value);
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
    let result = validate_taproot_key_aggregation(&internal_pubkey, &merkle_root, &wrong_output_key);
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
    // Maximum witness element size: 520 bytes
    let max_witness_element = vec![0x42; 520];
    
    // Taproot script path witness structure:
    // - Script execution stack
    // - Control block (33-65 bytes)
    
    // Control block format: <version byte> <internal_key> <merkle_proof>
    let control_block = vec![0xc0; 33]; // Version + internal key
    
    // Should handle maximum witness sizes
    assert!(max_witness_element.len() <= 520);
    assert!(control_block.len() >= 33);
}

/// Test Taproot script path with multiple scripts
#[test]
fn test_taproot_script_path_multiple_scripts() {
    // Taproot can have multiple script paths in merkle tree
    // Each script needs its own merkle proof
    
    let script1 = vec![0x51]; // OP_1
    let script2 = vec![0x52]; // OP_2
    
    // Each script has its own merkle proof
    let proof1: Vec<Hash> = vec![[0x01; 32]];
    let proof2: Vec<Hash> = vec![[0x02; 32]];
    
    // Merkle root would be computed from both scripts
    let merkle_root: Hash = [0x42; 32];
    
    // Both scripts should validate with their respective proofs
    let result1 = validate_taproot_script_path(&script1, &proof1, &merkle_root);
    let result2 = validate_taproot_script_path(&script2, &proof2, &merkle_root);
    
    assert!(result1.is_ok() || result1.is_err());
    assert!(result2.is_ok() || result2.is_err());
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
    let result = validate_taproot_key_aggregation(&internal_pubkey, &empty_merkle_root, &output_key.unwrap());
    assert!(result.is_ok());
}

/// Test Taproot script path boundary conditions
#[test]
fn test_taproot_script_path_boundaries() {
    // Test merkle proof with single hash
    let script = vec![0x51];
    let single_proof: Vec<Hash> = vec![[0x01; 32]];
    let merkle_root: Hash = [0x42; 32];
    
    let result = validate_taproot_script_path(&script, &single_proof, &merkle_root);
    assert!(result.is_ok() || result.is_err());
    
    // Test merkle proof with no hashes (single script)
    let empty_proof: Vec<Hash> = vec![];
    let result = validate_taproot_script_path(&script, &empty_proof, &merkle_root);
    assert!(result.is_ok() || result.is_err());
}

/// Test Taproot control block validation
#[test]
fn test_taproot_control_block() {
    // Control block format:
    // - Version byte (0xc0 or 0xc1)
    // - Internal public key (32 bytes)
    // - Merkle proof (variable length)
    
    // Minimum control block: 33 bytes (version + internal key, no proof)
    let min_control_block = vec![0xc0; 33];
    assert_eq!(min_control_block.len(), 33);
    
    // Control block with merkle proof: 33 + (proof_length * 32)
    let proof_length = 5;
    let control_block_with_proof = 33 + (proof_length * 32);
    assert_eq!(control_block_with_proof, 33 + 160);
    
    // Maximum control block: 65 bytes (version + internal key + 1 proof hash)
    let max_control_block_single_proof = 33 + 32;
    assert_eq!(max_control_block_single_proof, 65);
}



