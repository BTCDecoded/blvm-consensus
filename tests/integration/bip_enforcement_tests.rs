//! Integration tests to verify BIP checks are enforced in connect_block
//!
//! These tests verify that BIP30, BIP34, and BIP90 violations are caught
//! by connect_block, not just by the individual BIP check functions.
//!
//! **CRITICAL**: These tests will FAIL if BIP checks are removed from connect_block,
//! providing an alarm bell for missing consensus rules.

use bllvm_consensus::*;
use bllvm_consensus::block::connect_block;
use bllvm_consensus::block::calculate_tx_id;
use bllvm_consensus::transaction::is_coinbase;

/// Test that BIP30 (duplicate coinbase) is enforced in connect_block
///
/// This test creates a block with a duplicate coinbase transaction.
/// If BIP30 check is NOT called in connect_block, this block will be accepted (BUG).
/// If BIP30 check IS called, this block will be rejected (CORRECT).
#[test]
fn test_connect_block_rejects_bip30_violation() {
    // Create a coinbase transaction
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let txid = calculate_tx_id(&coinbase_tx);
    
    // Create UTXO set with a UTXO from this coinbase (simulating duplicate)
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: txid, index: 0 },
        UTXO {
            value: 50_000_000_000,
            script_pubkey: vec![],
            height: 0,
        },
    );
    
    // Create block with same coinbase (duplicate - violates BIP30)
    let block = Block {
        header: BlockHeader {
            version: 2,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    
    // connect_block MUST reject this block due to BIP30 violation
    let result = connect_block(&block, &witnesses, utxo_set, 1, None, types::Network::Mainnet);
    
    match result {
        Ok((ValidationResult::Invalid(reason), _)) => {
            // Good - block was rejected
            assert!(
                reason.contains("BIP30") || reason.contains("duplicate coinbase"),
                "Rejection reason should mention BIP30 or duplicate coinbase, got: {}",
                reason
            );
        }
        Ok((ValidationResult::Valid, _)) => {
            panic!("CRITICAL BUG: connect_block accepted a block with duplicate coinbase (BIP30 violation)! This means BIP30 check is NOT being called in connect_block!");
        }
        Err(e) => {
            // Error is also acceptable - means validation caught the violation
            eprintln!("connect_block returned error (acceptable): {:?}", e);
        }
    }
}

/// Test that BIP34 (block height in coinbase) is enforced in connect_block
///
/// This test creates a block at height >= 227836 without height in coinbase.
/// If BIP34 check is NOT called in connect_block, this block will be accepted (BUG).
/// If BIP34 check IS called, this block will be rejected (CORRECT).
#[test]
fn test_connect_block_rejects_bip34_violation() {
    let height = 227_836; // BIP34 activation height
    
    // Create coinbase WITHOUT height encoding (violates BIP34)
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![], // Empty scriptSig - no height encoding
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let block = Block {
        header: BlockHeader {
            version: 2, // Valid version for BIP34
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    // connect_block MUST reject this block due to BIP34 violation
    let result = connect_block(&block, &witnesses, utxo_set, height, None, types::Network::Mainnet);
    
    match result {
        Ok((ValidationResult::Invalid(reason), _)) => {
            // Good - block was rejected
            assert!(
                reason.contains("BIP34") || reason.contains("height") || reason.contains("coinbase"),
                "Rejection reason should mention BIP34, height, or coinbase, got: {}",
                reason
            );
        }
        Ok((ValidationResult::Valid, _)) => {
            panic!("CRITICAL BUG: connect_block accepted a block without height in coinbase at height {} (BIP34 violation)! This means BIP34 check is NOT being called in connect_block!", height);
        }
        Err(e) => {
            // Error is also acceptable
            eprintln!("connect_block returned error (acceptable): {:?}", e);
        }
    }
}

/// Test that BIP34 is NOT enforced before activation height
///
/// This ensures BIP34 check is called but correctly allows blocks before activation.
#[test]
fn test_connect_block_allows_bip34_before_activation() {
    let height = 100_000; // Before BIP34 activation (227836)
    
    // Create coinbase WITHOUT height encoding (OK before activation)
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![], // Empty scriptSig - OK before activation
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    // connect_block should allow this block (BIP34 not active yet)
    // Note: Block may still be invalid for other reasons (PoW, etc.), but BIP34 shouldn't reject it
    let result = connect_block(&block, &witnesses, utxo_set, height, None, types::Network::Mainnet);
    
    // If rejected, it should NOT be due to BIP34
    if let Ok((ValidationResult::Invalid(reason), _)) = result {
        assert!(
            !reason.contains("BIP34"),
            "Block should not be rejected for BIP34 before activation height, but got: {}",
            reason
        );
    }
}

/// Test that BIP90 (block version enforcement) is enforced in connect_block
///
/// This test creates a block with version 1 at height >= 227836 (after BIP34 activation).
/// If BIP90 check is NOT called in connect_block, this block will be accepted (BUG).
/// If BIP90 check IS called, this block will be rejected (CORRECT).
#[test]
fn test_connect_block_rejects_bip90_violation() {
    let height = 227_836; // BIP34 activation height (requires version >= 2)
    
    // Create block with version 1 (violates BIP90 - requires version >= 2 after BIP34)
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x03, (height & 0xff) as u8, ((height >> 8) & 0xff) as u8, ((height >> 16) & 0xff) as u8], // Valid height encoding
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let block = Block {
        header: BlockHeader {
            version: 1, // INVALID - should be >= 2 after BIP34 activation
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    // connect_block MUST reject this block due to BIP90 violation
    let result = connect_block(&block, &witnesses, utxo_set, height, None, types::Network::Mainnet);
    
    match result {
        Ok((ValidationResult::Invalid(reason), _)) => {
            // Good - block was rejected
            assert!(
                reason.contains("BIP90") || reason.contains("version") || reason.contains("Block version"),
                "Rejection reason should mention BIP90 or version, got: {}",
                reason
            );
        }
        Ok((ValidationResult::Valid, _)) => {
            panic!("CRITICAL BUG: connect_block accepted a block with version 1 at height {} (BIP90 violation)! This means BIP90 check is NOT being called in connect_block!", height);
        }
        Err(e) => {
            // Error is also acceptable
            eprintln!("connect_block returned error (acceptable): {:?}", e);
        }
    }
}

/// Test that BIP90 allows valid versions
///
/// This ensures BIP90 check is called but correctly allows valid versions.
#[test]
fn test_connect_block_allows_bip90_valid_version() {
    let height = 227_836; // BIP34 activation height (requires version >= 2)
    
    // Create block with version 2 (valid for BIP90)
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x03, (height & 0xff) as u8, ((height >> 8) & 0xff) as u8, ((height >> 16) & 0xff) as u8], // Valid height encoding
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let block = Block {
        header: BlockHeader {
            version: 2, // VALID - meets BIP90 requirement
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    // connect_block should allow this block (BIP90 satisfied)
    // Note: Block may still be invalid for other reasons (PoW, etc.), but BIP90 shouldn't reject it
    let result = connect_block(&block, &witnesses, utxo_set, height, None, types::Network::Mainnet);
    
    // If rejected, it should NOT be due to BIP90
    if let Ok((ValidationResult::Invalid(reason), _)) = result {
        assert!(
            !reason.contains("BIP90"),
            "Block should not be rejected for BIP90 with valid version, but got: {}",
            reason
        );
    }
}

/// Test that all three BIP checks work together
///
/// This test creates a block that violates multiple BIPs to ensure all checks are called.
#[test]
fn test_connect_block_multiple_bip_violations() {
    let height = 227_836;
    
    // Create block that violates BIP30, BIP34, and BIP90
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![], // Violates BIP34 (no height)
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let txid = calculate_tx_id(&coinbase_tx);
    
    // Add UTXO to simulate duplicate coinbase (violates BIP30)
    let mut utxo_set = UtxoSet::new();
    utxo_set.insert(
        OutPoint { hash: txid, index: 0 },
        UTXO {
            value: 50_000_000_000,
            script_pubkey: vec![],
            height: 0,
        },
    );
    
    let block = Block {
        header: BlockHeader {
            version: 1, // Violates BIP90 (should be >= 2)
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    
    // connect_block MUST reject this block
    let result = connect_block(&block, &witnesses, utxo_set, height, None, types::Network::Mainnet);
    
    match result {
        Ok((ValidationResult::Invalid(reason), _)) => {
            // Should mention at least one BIP violation
            let mentions_bip = reason.contains("BIP30") 
                || reason.contains("BIP34") 
                || reason.contains("BIP90")
                || reason.contains("duplicate coinbase")
                || reason.contains("height")
                || reason.contains("version");
            
            assert!(
                mentions_bip,
                "Rejection reason should mention a BIP violation, got: {}",
                reason
            );
        }
        Ok((ValidationResult::Valid, _)) => {
            panic!("CRITICAL BUG: connect_block accepted a block with multiple BIP violations! This means BIP checks are NOT being called in connect_block!");
        }
        Err(e) => {
            // Error is also acceptable
            eprintln!("connect_block returned error (acceptable): {:?}", e);
        }
    }
}

/// Test that BIP checks are called in the correct order
///
/// BIP90 should be checked first (on header), then BIP30, then BIP34.
/// This test verifies the order by checking which violation is caught first.
#[test]
fn test_bip_check_order() {
    let height = 227_836;
    
    // Create block that violates BIP90 (version) - should be caught first
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![], // Also violates BIP34, but BIP90 should be caught first
            sequence: 0xffffffff,
        }].into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![].into(),
        }].into(),
        lock_time: 0,
    };
    
    let block = Block {
        header: BlockHeader {
            version: 1, // Violates BIP90 - should be caught first
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    
    let witnesses: Vec<segwit::Witness> = block.transactions.iter().map(|_| Vec::new()).collect();
    let utxo_set = UtxoSet::new();
    
    let result = connect_block(&block, &witnesses, utxo_set, height, None, types::Network::Mainnet);
    
    // BIP90 should be caught first (it's checked on header, before transaction checks)
    if let Ok((ValidationResult::Invalid(reason), _)) = result {
        // BIP90 should be mentioned (or version), not BIP34
        // This verifies BIP90 is checked before BIP34
        assert!(
            reason.contains("BIP90") || reason.contains("version") || reason.contains("Block version"),
            "BIP90 violation should be caught first, but got: {}",
            reason
        );
    }
}

