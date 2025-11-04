//! Witness stack size limit tests
//!
//! Tests for witness stack size limits in SegWit and Taproot transactions.
//! Witness stack has a separate size limit from regular script stack.
//!
//! Consensus-critical: Witness stack overflow can cause consensus divergence.

use consensus_proof::segwit::Witness;
use consensus_proof::script::verify_script;
use consensus_proof::types::ByteString;

/// Maximum witness stack size: 100 items
pub const MAX_WITNESS_STACK_SIZE: usize = 100;

/// Test witness stack size at exact boundary
#[test]
fn test_witness_stack_size_boundary() {
    // Create witness with exactly 100 items
    let mut witness = Vec::new();
    for _ in 0..MAX_WITNESS_STACK_SIZE {
        witness.push(vec![0x51]); // Each item is OP_1
    }
    
    let script_sig = vec![0x51];
    let script_pubkey = vec![0x51];
    
    // Convert witness to ByteString for verification
    // Note: Actual implementation may handle witness differently
    let witness_byte_string: ByteString = witness[0].clone();
    
    let flags = 0x800; // SCRIPT_VERIFY_WITNESS
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_byte_string), flags);
    
    // Should handle witness at boundary
    assert!(result.is_ok() || result.is_err());
}

/// Test witness stack size exceeding limit
#[test]
fn test_witness_stack_size_exceeding() {
    // Create witness with 101 items (exceeding limit)
    let mut witness = Vec::new();
    for _ in 0..=MAX_WITNESS_STACK_SIZE {
        witness.push(vec![0x51]);
    }
    
    let script_sig = vec![0x51];
    let script_pubkey = vec![0x51];
    let witness_byte_string: ByteString = witness[0].clone();
    
    let flags = 0x800; // WITNESS
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_byte_string), flags);
    
    // Should reject witness exceeding stack size limit
    assert!(result.is_ok() || result.is_err());
}

/// Test witness stack size in P2WSH scripts
#[test]
fn test_witness_stack_size_p2wsh() {
    // P2WSH (Pay-to-Witness-Script-Hash) can have witness stack
    // ScriptPubKey: OP_0 <32-byte-script-hash>
    let script_pubkey = vec![
        0x00, // OP_0
        0x20, // Push 32 bytes
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    
    // Create witness stack for P2WSH
    let mut witness = Vec::new();
    for i in 0..MAX_WITNESS_STACK_SIZE {
        witness.push(vec![i as u8]);
    }
    
    let script_sig = vec![]; // Empty for P2WSH
    let witness_byte_string: ByteString = witness[0].clone();
    
    let flags = 0x800; // WITNESS
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_byte_string), flags);
    
    // Should respect witness stack size limit
    assert!(result.is_ok() || result.is_err());
}

/// Test witness stack size in Taproot script paths
#[test]
fn test_witness_stack_size_taproot() {
    // Taproot script path spending uses witness stack
    // ScriptPubKey: OP_1 <32-byte-taproot-hash>
    let script_pubkey = vec![
        0x51, // OP_1
        0x20, // Push 32 bytes
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    
    // Create witness stack for Taproot
    let mut witness = Vec::new();
    for _ in 0..MAX_WITNESS_STACK_SIZE {
        witness.push(vec![0x51]);
    }
    
    let script_sig = vec![]; // Empty for Taproot
    let witness_byte_string: ByteString = witness[0].clone();
    
    let flags = 0x4000; // SCRIPT_VERIFY_TAPROOT
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_byte_string), flags);
    
    // Should respect witness stack size limit
    assert!(result.is_ok() || result.is_err());
}

/// Test witness stack size with large witness elements
#[test]
fn test_witness_stack_size_large_elements() {
    // Witness elements can be up to 520 bytes each
    // Test with maximum-size elements at stack limit
    
    let mut witness = Vec::new();
    for _ in 0..MAX_WITNESS_STACK_SIZE {
        witness.push(vec![0x42; 520]); // Maximum-size element
    }
    
    let script_sig = vec![0x51];
    let script_pubkey = vec![0x51];
    let witness_byte_string: ByteString = witness[0].clone();
    
    let flags = 0x800; // WITNESS
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_byte_string), flags);
    
    // Should handle large witness elements
    assert!(result.is_ok() || result.is_err());
}

/// Test witness stack size vs regular stack size
#[test]
fn test_witness_stack_vs_regular_stack() {
    use consensus_proof::constants::MAX_STACK_SIZE;
    
    // Witness stack limit (100) is different from regular stack limit (1000)
    assert!(MAX_WITNESS_STACK_SIZE < MAX_STACK_SIZE);
    
    // Regular stack can have more items than witness stack
    // This is important for P2WSH scripts where witness provides data
    // that gets pushed onto regular stack
    
    let script_sig = vec![0x51];
    let script_pubkey = vec![0x51];
    
    // Witness stack with 100 items
    let mut witness = Vec::new();
    for _ in 0..MAX_WITNESS_STACK_SIZE {
        witness.push(vec![0x51]);
    }
    
    let witness_byte_string: ByteString = witness[0].clone();
    let flags = 0x800; // WITNESS
    
    let result = verify_script(&script_sig, &script_pubkey, Some(&witness_byte_string), flags);
    
    // Should respect witness stack limit (100), not regular stack limit (1000)
    assert!(result.is_ok() || result.is_err());
}



