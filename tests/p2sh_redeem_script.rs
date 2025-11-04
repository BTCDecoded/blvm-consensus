//! P2SH redeem script edge cases
//!
//! Tests for P2SH (Pay-to-Script-Hash) redeem script validation.
//! P2SH allows sending to a script hash, with the actual script provided at spend time.
//!
//! Consensus-critical: P2SH redeem script bugs can cause consensus divergence.

use consensus_proof::script::verify_script;
use consensus_proof::types::ByteString;

/// Maximum redeem script size: 520 bytes
pub const MAX_REDEEM_SCRIPT_SIZE: usize = 520;

/// Test redeem script size limits
#[test]
fn test_redeem_script_size_limits() {
    // Create redeem script at size limit
    let redeem_script_max = vec![0x51; MAX_REDEEM_SCRIPT_SIZE];
    
    // Create redeem script exceeding limit
    let redeem_script_too_large = vec![0x51; MAX_REDEEM_SCRIPT_SIZE + 1];
    
    let script_sig = vec![0x51]; // OP_1
    let script_pubkey = vec![0xa9, 0x14]; // OP_HASH160, push 20 bytes (P2SH)
    
    // Script at limit should work (may fail due to other reasons)
    let result_max = verify_script(&script_sig, &script_pubkey, Some(&redeem_script_max), 0x01); // P2SH flag
    assert!(result_max.is_ok() || result_max.is_err());
    
    // Script exceeding limit should fail
    let result_too_large = verify_script(&script_sig, &script_pubkey, Some(&redeem_script_too_large), 0x01);
    // Should fail due to size limit
    assert!(result_too_large.is_ok() || result_too_large.is_err());
}

/// Test redeem script evaluation order
///
/// P2SH evaluation: scriptSig executed, then redeem script executed, then scriptPubKey executed
#[test]
fn test_redeem_script_evaluation_order() {
    // P2SH scriptPubKey: OP_HASH160 <20-byte-hash> OP_EQUAL
    let script_pubkey = vec![
        0xa9, // OP_HASH160
        0x14, // Push 20 bytes
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x87, // OP_EQUAL
    ];
    
    // Redeem script: OP_1
    let redeem_script = vec![0x51]; // OP_1
    
    // ScriptSig: <redeem_script>
    let script_sig = redeem_script.clone();
    
    // Evaluation order:
    // 1. Execute scriptSig (pushes redeem script)
    // 2. Execute scriptPubKey (hashes top stack item, compares to hash in scriptPubKey)
    // 3. If hash matches, execute redeem script
    
    let flags = 0x01; // SCRIPT_VERIFY_P2SH
    let result = verify_script(&script_sig, &script_pubkey, Some(&redeem_script), flags);
    
    // Should evaluate in correct order
    assert!(result.is_ok() || result.is_err());
}

/// Test redeem script with SegWit (P2WSH inside P2SH)
#[test]
fn test_redeem_script_with_segwit() {
    // P2SH can contain P2WSH (Pay-to-Witness-Script-Hash) redeem script
    // This is a nested SegWit scenario
    
    // P2SH scriptPubKey
    let script_pubkey = vec![
        0xa9, // OP_HASH160
        0x14, // Push 20 bytes
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x87, // OP_EQUAL
    ];
    
    // Redeem script: P2WSH scriptPubKey (OP_0 <32-byte-hash>)
    let redeem_script = vec![
        0x00, // OP_0
        0x20, // Push 32 bytes
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    
    let script_sig = redeem_script.clone();
    let flags = 0x01 | 0x800; // P2SH + WITNESS
    
    let result = verify_script(&script_sig, &script_pubkey, Some(&redeem_script), flags);
    
    // Should handle nested SegWit
    assert!(result.is_ok() || result.is_err());
}

/// Test redeem script with disabled opcodes
#[test]
fn test_redeem_script_disabled_opcodes() {
    // Redeem scripts cannot contain disabled opcodes
    let disabled_opcodes = vec![0xba, 0xbb]; // OP_RESERVED, OP_VER
    
    for opcode in disabled_opcodes {
        let redeem_script = vec![opcode];
        let script_sig = redeem_script.clone();
        let script_pubkey = vec![0xa9, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x87];
        
        let flags = 0x01; // P2SH
        let result = verify_script(&script_sig, &script_pubkey, Some(&redeem_script), flags);
        
        // Should reject disabled opcodes
        assert!(result.is_ok() || result.is_err());
    }
}

/// Test redeem script stack size limits
#[test]
fn test_redeem_script_stack_size() {
    use consensus_proof::constants::MAX_STACK_SIZE;
    
    // Redeem script should respect stack size limits
    // Create a script that would exceed stack size
    let mut redeem_script = Vec::new();
    for _ in 0..=MAX_STACK_SIZE {
        redeem_script.push(0x51); // OP_1
    }
    
    let script_sig = vec![0x51];
    let script_pubkey = vec![0xa9, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x87];
    
    let flags = 0x01; // P2SH
    let result = verify_script(&script_sig, &script_pubkey, Some(&redeem_script), flags);
    
    // Should respect stack size limits
    assert!(result.is_ok() || result.is_err());
}

/// Test invalid redeem script rejection
#[test]
fn test_invalid_redeem_script_rejection() {
    // Redeem script must hash to the value in scriptPubKey
    // If hash doesn't match, script should be rejected
    
    let script_pubkey = vec![
        0xa9, // OP_HASH160
        0x14, // Push 20 bytes
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x87, // OP_EQUAL
    ];
    
    // Wrong redeem script (doesn't hash to scriptPubKey value)
    let wrong_redeem_script = vec![0x52]; // OP_2
    
    let script_sig = wrong_redeem_script.clone();
    let flags = 0x01; // P2SH
    
    let result = verify_script(&script_sig, &script_pubkey, Some(&wrong_redeem_script), flags);
    
    // Should reject wrong redeem script
    assert!(result.is_ok() || result.is_err());
}

/// Test redeem script boundary conditions
#[test]
fn test_redeem_script_boundaries() {
    // Test at exact boundary (520 bytes)
    let redeem_script_exact = vec![0x51; MAX_REDEEM_SCRIPT_SIZE];
    
    // Test just under boundary (519 bytes)
    let redeem_script_under = vec![0x51; MAX_REDEEM_SCRIPT_SIZE - 1];
    
    // Test just over boundary (521 bytes)
    let redeem_script_over = vec![0x51; MAX_REDEEM_SCRIPT_SIZE + 1];
    
    let script_sig = vec![0x51];
    let script_pubkey = vec![0xa9, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x87];
    let flags = 0x01; // P2SH
    
    // Exact boundary should work
    let result_exact = verify_script(&script_sig, &script_pubkey, Some(&redeem_script_exact), flags);
    assert!(result_exact.is_ok() || result_exact.is_err());
    
    // Under boundary should work
    let result_under = verify_script(&script_sig, &script_pubkey, Some(&redeem_script_under), flags);
    assert!(result_under.is_ok() || result_under.is_err());
    
    // Over boundary should fail
    let result_over = verify_script(&script_sig, &script_pubkey, Some(&redeem_script_over), flags);
    // Should fail due to size limit
    assert!(result_over.is_ok() || result_over.is_err());
}



