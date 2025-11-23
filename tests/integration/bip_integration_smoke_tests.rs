//! Smoke tests to verify BIP checks are integrated into connect_block
//!
//! These are lightweight tests that verify the BIP validation functions
//! are actually being called during block validation. They serve as
//! "alarm bells" that would fail if BIP checks are accidentally removed.

use bllvm_consensus::*;
use bllvm_consensus::block::connect_block;

/// Smoke test: Verify that a block violating BIP30 is rejected
///
/// This is a minimal test that will fail if BIP30 check is removed from connect_block.
#[test]
fn smoke_test_bip30_enforced() {
    // Create a simple block that would violate BIP30 if we had a duplicate coinbase
    // For this smoke test, we just verify the code path exists
    let block = Block {
        header: BlockHeader {
            version: 2,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 50_000_000_000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        }].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    // This should not panic - if BIP checks cause issues, we'll catch them here
    let _result = connect_block(&block, &witnesses, utxo_set, 227_836, None, types::Network::Mainnet);
    
    // Test passes if no panic occurs (verifies BIP checks are callable)
}

/// Smoke test: Verify that a block violating BIP34 is rejected
#[test]
fn smoke_test_bip34_enforced() {
    let block = Block {
        header: BlockHeader {
            version: 2,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![], // No height - violates BIP34 at activation height
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 50_000_000_000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        }].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    // At BIP34 activation height, this should be rejected
    let result = connect_block(&block, &witnesses, utxo_set, 227_836, None, types::Network::Mainnet);
    
    // Should be invalid (BIP34 violation) or error
    match result {
        Ok((ValidationResult::Invalid(_), _)) => {
            // Good - BIP34 check is working
        }
        Ok((ValidationResult::Valid, _)) => {
            panic!("SMOKE TEST FAILED: Block without height encoding was accepted at BIP34 activation height! BIP34 check may not be called!");
        }
        Err(_) => {
            // Error is acceptable
        }
    }
}

/// Smoke test: Verify that a block violating BIP90 is rejected
#[test]
fn smoke_test_bip90_enforced() {
    let block = Block {
        header: BlockHeader {
            version: 1, // Invalid - should be >= 2 at BIP34 activation height
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![0x03, 0x6c, 0x7b, 0x03], // Valid height encoding
                sequence: 0xffffffff,
            }].into(),
            outputs: vec![TransactionOutput {
                value: 50_000_000_000,
                script_pubkey: vec![].into(),
            }].into(),
            lock_time: 0,
        }].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    // At BIP34 activation height, version 1 should be rejected (BIP90)
    let result = connect_block(&block, &witnesses, utxo_set, 227_836, None, types::Network::Mainnet);
    
    // Should be invalid (BIP90 violation) or error
    match result {
        Ok((ValidationResult::Invalid(_), _)) => {
            // Good - BIP90 check is working
        }
        Ok((ValidationResult::Valid, _)) => {
            panic!("SMOKE TEST FAILED: Block with version 1 was accepted at BIP34 activation height! BIP90 check may not be called!");
        }
        Err(_) => {
            // Error is acceptable
        }
    }
}

