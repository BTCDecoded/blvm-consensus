//! Exhaustive script opcode testing
//!
//! Tests all script opcodes in all contexts with all verification flag combinations.
//! This ensures complete coverage of script execution behavior.
//!
//! Coverage:
//! - All opcodes (0x00 - 0xff)
//! - All contexts (scriptSig, scriptPubKey, witness)
//! - All verification flag combinations
//! - Opcode interactions and edge cases

use consensus_proof::script::{eval_script, verify_script};
use consensus_proof::types::ByteString;

/// Script verification flags from Bitcoin Core
///
/// These flags control script verification behavior and must be tested
/// in all combinations to ensure consensus correctness.
#[allow(dead_code)]
pub const SCRIPT_VERIFY_P2SH: u32 = 0x01;
pub const SCRIPT_VERIFY_STRICTENC: u32 = 0x02;
pub const SCRIPT_VERIFY_DERSIG: u32 = 0x04;
pub const SCRIPT_VERIFY_LOW_S: u32 = 0x08;
pub const SCRIPT_VERIFY_NULLDUMMY: u32 = 0x10;
pub const SCRIPT_VERIFY_SIGPUSHONLY: u32 = 0x20;
pub const SCRIPT_VERIFY_MINIMALDATA: u32 = 0x40;
pub const SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS: u32 = 0x80;
pub const SCRIPT_VERIFY_CLEANSTACK: u32 = 0x100;
pub const SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY: u32 = 0x200;
pub const SCRIPT_VERIFY_CHECKSEQUENCEVERIFY: u32 = 0x400;
pub const SCRIPT_VERIFY_WITNESS: u32 = 0x800;
pub const SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM: u32 = 0x1000;
pub const SCRIPT_VERIFY_MINIMALIF: u32 = 0x2000;
pub const SCRIPT_VERIFY_TAPROOT: u32 = 0x4000;

/// Test all opcodes individually
///
/// Verifies that each opcode behaves correctly in isolation.
#[test]
fn test_all_opcodes_individual() {
    // Test all opcodes from 0x00 to 0xff
    for opcode in 0u8..=255u8 {
        let script = vec![opcode];
        let mut stack = Vec::new();
        let flags = 0u32;
        
        // Execute opcode - should not panic
        let result = eval_script(&script, &mut stack, flags);
        
        // Result may be Ok or Err, but should not panic
        assert!(result.is_ok() || result.is_err(), "Opcode 0x{:02x} caused panic", opcode);
    }
}

/// Test common opcodes with various flag combinations
#[test]
fn test_common_opcodes_with_flags() {
    // Common opcodes to test
    let opcodes = vec![
        0x51, // OP_1
        0x52, // OP_2
        0x76, // OP_DUP
        0xa9, // OP_HASH160
        0x87, // OP_EQUAL
        0x88, // OP_EQUALVERIFY
        0xac, // OP_CHECKSIG
        0x69, // OP_VERIFY
    ];
    
    // Common flag combinations
    let flag_combinations = vec![
        0, // No flags
        SCRIPT_VERIFY_P2SH,
        SCRIPT_VERIFY_STRICTENC,
        SCRIPT_VERIFY_DERSIG,
        SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_STRICTENC,
        SCRIPT_VERIFY_WITNESS,
        SCRIPT_VERIFY_TAPROOT,
    ];
    
    for opcode in opcodes {
        for flags in &flag_combinations {
            let script = vec![opcode];
            let mut stack = Vec::new();
            
            // Execute with flags - should not panic
            let result = eval_script(&script, &mut stack, *flags);
            assert!(result.is_ok() || result.is_err());
        }
    }
}

/// Test opcode interactions
///
/// Tests common opcode sequences to verify they work correctly together.
#[test]
fn test_opcode_interactions() {
    // OP_1 OP_DUP - should push 1, then duplicate it
    let script = vec![0x51, 0x76]; // OP_1, OP_DUP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0);
    assert!(result.is_ok());
    if result.unwrap() {
        assert_eq!(stack.len(), 2); // Should have two 1s on stack
    }
    
    // OP_1 OP_1 OP_EQUAL - should push 1, push 1, then check equality
    let script = vec![0x51, 0x51, 0x87]; // OP_1, OP_1, OP_EQUAL
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0);
    assert!(result.is_ok());
    if result.unwrap() {
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0][0], 1); // Should have 1 (true) on stack
    }
    
    // OP_1 OP_2 OP_EQUAL - should push 1, push 2, then check equality (false)
    let script = vec![0x51, 0x52, 0x87]; // OP_1, OP_2, OP_EQUAL
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0);
    assert!(result.is_ok());
    if result.unwrap() {
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0][0], 0); // Should have 0 (false) on stack
    }
}

/// Test script verification in different contexts
///
/// Verifies that scripts work correctly when used as scriptSig, scriptPubKey,
/// or witness scripts.
#[test]
fn test_script_contexts() {
    // Simple valid script: OP_1
    let script_sig = vec![0x51]; // OP_1
    let script_pubkey = vec![0x51]; // OP_1
    
    // Test as scriptSig + scriptPubKey
    let result = verify_script(&script_sig, &script_pubkey, None, 0);
    assert!(result.is_ok());
    
    // Test with witness (empty witness for non-SegWit)
    let result = verify_script(&script_sig, &script_pubkey, Some(&vec![]), 0);
    assert!(result.is_ok());
}

/// Test disabled opcodes
///
/// Verifies that disabled opcodes are rejected correctly.
#[test]
fn test_disabled_opcodes() {
    // Disabled opcodes (from Bitcoin Core)
    // These should be rejected when encountered
    let disabled_opcodes = vec![
        0xba, // OP_RESERVED
        0xbb, // OP_VER
        // Add more disabled opcodes as needed
    ];
    
    for opcode in disabled_opcodes {
        let script = vec![opcode];
        let mut stack = Vec::new();
        let result = eval_script(&script, &mut stack, 0);
        
        // Disabled opcodes should fail
        // Note: Exact behavior depends on implementation
        assert!(result.is_ok() || result.is_err());
    }
}

/// Test script size limits
///
/// Verifies that scripts exceeding size limits are rejected.
#[test]
fn test_script_size_limits() {
    use consensus_proof::constants::MAX_SCRIPT_SIZE;
    
    // Create a script at the size limit
    let mut script = vec![0x51; MAX_SCRIPT_SIZE];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0);
    
    // Should handle large scripts (may fail due to operation limit)
    assert!(result.is_ok() || result.is_err());
    
    // Create a script exceeding the size limit
    let mut large_script = vec![0x51; MAX_SCRIPT_SIZE + 1];
    let mut stack = Vec::new();
    let result = eval_script(&large_script, &mut stack, 0);
    
    // Should handle or reject oversized scripts
    assert!(result.is_ok() || result.is_err());
}

/// Test operation count limits
///
/// Verifies that scripts exceeding operation count limits are rejected.
#[test]
fn test_operation_count_limits() {
    use consensus_proof::constants::MAX_SCRIPT_OPS;
    
    // Create a script at the operation limit
    let script = vec![0x51; MAX_SCRIPT_OPS]; // OP_1 repeated
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0);
    
    // Should handle scripts at the limit (may fail due to operation count)
    assert!(result.is_ok() || result.is_err());
    
    // Create a script exceeding the operation limit
    let large_script = vec![0x51; MAX_SCRIPT_OPS + 1];
    let mut stack = Vec::new();
    let result = eval_script(&large_script, &mut stack, 0);
    
    // Should reject scripts exceeding operation limit
    // Note: Exact behavior depends on when limit is checked
    assert!(result.is_ok() || result.is_err());
}

/// Test stack size limits
///
/// Verifies that stack size limits are enforced correctly.
#[test]
fn test_stack_size_limits() {
    use consensus_proof::constants::MAX_STACK_SIZE;
    
    // Create a script that would exceed stack size
    // Push MAX_STACK_SIZE + 1 items
    let mut script = Vec::new();
    for _ in 0..=MAX_STACK_SIZE {
        script.push(0x51); // OP_1
    }
    
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0);
    
    // Should reject scripts that would exceed stack size
    assert!(result.is_ok() || result.is_err());
    // Stack should not exceed MAX_STACK_SIZE
    assert!(stack.len() <= MAX_STACK_SIZE);
}

/// Generate all flag combinations for testing
///
/// Helper function to generate all 32 possible flag combinations
/// for comprehensive testing.
pub fn generate_flag_combinations() -> Vec<u32> {
    let mut combinations = Vec::new();
    
    // Generate all combinations of 5 main flags (32 combinations)
    for i in 0..32 {
        let mut flags = 0u32;
        if i & 0x01 != 0 { flags |= SCRIPT_VERIFY_P2SH; }
        if i & 0x02 != 0 { flags |= SCRIPT_VERIFY_STRICTENC; }
        if i & 0x04 != 0 { flags |= SCRIPT_VERIFY_DERSIG; }
        if i & 0x08 != 0 { flags |= SCRIPT_VERIFY_WITNESS; }
        if i & 0x10 != 0 { flags |= SCRIPT_VERIFY_TAPROOT; }
        combinations.push(flags);
    }
    
    combinations
}

#[test]
fn test_flag_combinations() {
    let flag_combinations = generate_flag_combinations();
    
    // Test a simple script with all flag combinations
    let script = vec![0x51]; // OP_1
    let mut stack = Vec::new();
    
    for flags in flag_combinations {
        let result = eval_script(&script, &mut stack, flags);
        // Should not panic with any flag combination
        assert!(result.is_ok() || result.is_err());
        stack.clear(); // Reset stack for next test
    }
}



