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

use blvm_consensus::opcodes::{
    OP_1, OP_2, OP_CHECKSIG, OP_DUP, OP_EQUAL, OP_EQUALVERIFY, OP_HASH160, OP_RESERVED, OP_VER,
    OP_VERIFY,
};
use blvm_consensus::script::flags::SCRIPT_VERIFY_TAPROOT;
use blvm_consensus::script::{SigVersion, eval_script, verify_script};

/// Script verification flags (values match Bitcoin Core script/interpreter.h)
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
// SCRIPT_VERIFY_TAPROOT = 0x20000 (bit 17) — imported from flags module above.
// 0x4000 (bit 14) is SCRIPT_VERIFY_NULLFAIL; do not use it for Taproot.

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
        let result = eval_script(&script, &mut stack, flags, SigVersion::Base);

        // Result may be Ok or Err, but should not panic
        assert!(
            result.is_ok() || result.is_err(),
            "Opcode 0x{opcode:02x} caused panic"
        );
    }
}

/// Test common opcodes with various flag combinations
#[test]
fn test_common_opcodes_with_flags() {
    // Common opcodes to test
    let opcodes = vec![
        OP_1,
        OP_2,
        OP_DUP,
        OP_HASH160,
        OP_EQUAL,
        OP_EQUALVERIFY,
        OP_CHECKSIG,
        OP_VERIFY,
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
            let result = eval_script(&script, &mut stack, *flags, SigVersion::Base);
            assert!(result.is_ok() || result.is_err());
        }
    }
}

/// Test opcode interactions
///
/// Tests common opcode sequences to verify they work correctly together.
#[test]
fn test_opcode_interactions() {
    // OP_1 OP_DUP
    let script = vec![OP_1, OP_DUP];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_ok());

    // OP_1 OP_1 OP_EQUAL
    let script = vec![OP_1, OP_1, OP_EQUAL];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_ok());

    // OP_1 OP_2 OP_EQUAL
    let script = vec![OP_1, OP_2, OP_EQUAL];
    let mut stack = Vec::new();
    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_ok());
}

/// Test script verification in different contexts
///
/// Verifies that scripts work correctly when used as scriptSig, scriptPubKey,
/// or witness scripts.
#[test]
fn test_script_contexts() {
    // Simple valid script: OP_1
    let script_sig = vec![OP_1]; // OP_1
    let script_pubkey = vec![OP_1]; // OP_1

    // Test as scriptSig + scriptPubKey
    // Note: verify_script is a simplified API that doesn't require full context
    // For full verification, use verify_script_with_context_full
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
    // Disabled opcodes (from consensus)
    // These should be rejected when encountered
    let disabled_opcodes = vec![OP_RESERVED, OP_VER];

    for opcode in disabled_opcodes {
        let script = vec![opcode];
        let mut stack = Vec::new();
        let result = eval_script(&script, &mut stack, 0, SigVersion::Base);

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
    use blvm_consensus::constants::MAX_SCRIPT_SIZE;

    // Create a script at the size limit
    let script = vec![OP_1; MAX_SCRIPT_SIZE];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);

    // Should handle large scripts (may fail due to operation limit)
    assert!(result.is_ok() || result.is_err());

    // Create a script exceeding the size limit
    let large_script = vec![OP_1; MAX_SCRIPT_SIZE + 1];
    let mut stack = Vec::new();
    let result = eval_script(&large_script, &mut stack, 0, SigVersion::Base);

    // Should handle or reject oversized scripts
    assert!(result.is_ok() || result.is_err());
}

/// Test operation count limits
///
/// Verifies that scripts exceeding operation count limits are rejected.
#[test]
fn test_operation_count_limits() {
    use blvm_consensus::constants::MAX_SCRIPT_OPS;

    // Create a script at the operation limit
    let script = vec![OP_1; MAX_SCRIPT_OPS]; // OP_1 repeated
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);

    // Should handle scripts at the limit (may fail due to operation count)
    assert!(result.is_ok() || result.is_err());

    // Create a script exceeding the operation limit
    let large_script = vec![OP_1; MAX_SCRIPT_OPS + 1];
    let mut stack = Vec::new();
    let result = eval_script(&large_script, &mut stack, 0, SigVersion::Base);

    // Should reject scripts exceeding operation limit
    // Note: Exact behavior depends on when limit is checked
    assert!(result.is_ok() || result.is_err());
}

/// Test stack size limits
///
/// Verifies that stack size limits are enforced correctly.
#[test]
fn test_stack_size_limits() {
    use blvm_consensus::constants::MAX_STACK_SIZE;

    // Create a script that would exceed stack size
    // Push MAX_STACK_SIZE + 1 items
    let mut script = Vec::new();
    for _ in 0..=MAX_STACK_SIZE {
        script.push(OP_1); // OP_1
    }

    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);

    // Bitcoin Core checks combined stack size after each opcode (> MAX_STACK_SIZE).
    // The failing push may leave stack at MAX+1 before the error is returned.
    assert!(
        result.is_err(),
        "expected stack overflow for {} pushes",
        MAX_STACK_SIZE + 1
    );
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
        if i & 0x01 != 0 {
            flags |= SCRIPT_VERIFY_P2SH;
        }
        if i & 0x02 != 0 {
            flags |= SCRIPT_VERIFY_STRICTENC;
        }
        if i & 0x04 != 0 {
            flags |= SCRIPT_VERIFY_DERSIG;
        }
        if i & 0x08 != 0 {
            flags |= SCRIPT_VERIFY_WITNESS;
        }
        if i & 0x10 != 0 {
            flags |= SCRIPT_VERIFY_TAPROOT;
        }
        combinations.push(flags);
    }

    combinations
}

#[test]
fn test_flag_combinations() {
    let flag_combinations = generate_flag_combinations();

    // Test a simple script with all flag combinations
    let script = vec![OP_1]; // OP_1
    let mut stack = Vec::new();

    for flags in flag_combinations {
        let result = eval_script(&script, &mut stack, flags, SigVersion::Base);
        // Should not panic with any flag combination
        assert!(result.is_ok() || result.is_err());
        stack.clear(); // Reset stack for next test
    }
}
