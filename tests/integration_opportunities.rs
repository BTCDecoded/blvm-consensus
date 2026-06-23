//! Integration tests between different consensus systems
//!
//! These tests verify that different modules work together correctly
//! and catch integration bugs that unit tests might miss.

use blvm_consensus::opcodes::OP_1;
use blvm_consensus::transaction::is_coinbase;
use blvm_consensus::*;

mod test_helpers;
use test_helpers::{
    adjusted_timeout, create_invalid_transaction, create_test_tx, create_test_utxo_set, is_ci,
};

/// Test integration between mempool and block creation
#[test]
fn test_mempool_to_block_integration() {
    let consensus = ConsensusProof::new();

    // 1. Create a valid transaction
    let tx = create_test_tx(1000, None, None, None);
    let utxo_set = create_test_utxo_set();
    let mempool = mempool::Mempool::new();

    // 2. Accept transaction to mempool
    let time_context = None; // No time context for this test
    let result = consensus
        .accept_to_memory_pool(
            &tx,
            &utxo_set,
            &mempool,
            100,
            time_context,
            Network::Regtest,
        )
        .unwrap();
    assert_eq!(result, mempool::MempoolResult::Accepted);

    // 3. Create block from mempool (coinbase-only when mempool list is empty)
    const HEIGHT: u64 = 100;
    let prev_header = create_valid_block_header();
    let prev_headers = vec![prev_header.clone(), prev_header.clone()];
    let coinbase_script = encode_bip34_height(HEIGHT);
    let coinbase_address = vec![OP_1];

    let mut block = consensus
        .create_new_block(
            &utxo_set,
            &[], // Empty mempool
            HEIGHT,
            &prev_header,
            &prev_headers,
            &coinbase_script,
            &coinbase_address,
        )
        .unwrap();
    block.header.version = 4;

    // 4. Verify block structure
    assert_eq!(block.transactions.len(), 1); // Only coinbase
    assert!(is_coinbase(&block.transactions[0]));

    // 5. Validate the created block (Regtest skips PoW; BIP34 height encoded in coinbase)
    let witnesses: Vec<Vec<blvm_consensus::segwit::Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let time_context = Some(TimeContext {
        network_time: block.header.timestamp,
        median_time_past: block.header.timestamp,
    });
    let network = blvm_consensus::types::Network::Regtest;
    let (validation_result, _new_utxo_set) = consensus
        .validate_block_with_time_context(
            &block,
            &witnesses,
            utxo_set,
            HEIGHT,
            time_context,
            network,
        )
        .unwrap();
    assert_eq!(validation_result, ValidationResult::Valid);
}

/// Test integration between economic model and mining
#[test]
fn test_economic_mining_integration() {
    let consensus = ConsensusProof::new();

    // 1. Test subsidy calculation at different heights
    // Using Orange Paper constant H (halving interval = 210,000)
    use blvm_consensus::orange_paper_constants::H;
    let heights = vec![0, H, H * 2, H * 3]; // Different halving periods

    for height in heights {
        let subsidy = consensus.get_block_subsidy(height);
        let total_supply = consensus.total_supply(height);

        // 2. Create coinbase transaction with calculated subsidy
        // Coinbase script_sig must be between 2 and 100 bytes (Orange Paper Section 5.1, rule 8)
        let coinbase_script = vec![OP_1, OP_1]; // At least 2 bytes
        let coinbase_address = vec![OP_1];

        let block = consensus
            .create_new_block(
                &UtxoSet::default(),
                &[],
                height,
                &create_valid_block_header(),
                &[create_valid_block_header(), create_valid_block_header()],
                &coinbase_script,
                &coinbase_address,
            )
            .unwrap();

        // 3. Verify coinbase output matches subsidy
        assert_eq!(block.transactions[0].outputs[0].value, subsidy);

        // 4. Verify total supply is reasonable
        assert!(total_supply > 0);
        assert!(total_supply <= MAX_MONEY);
    }
}

/// Test integration between script execution and transaction validation
#[test]
fn test_script_transaction_integration() {
    let consensus = ConsensusProof::new();

    // 1. Create transaction with specific script
    let mut tx = create_test_tx(1000, None, None, None);
    tx.inputs[0].script_sig = vec![OP_1]; // OP_1
    tx.outputs[0].script_pubkey = vec![OP_1]; // OP_1

    // 2. Create UTXO with matching script
    let mut utxo_set = UtxoSet::default();
    let outpoint = tx.inputs[0].prevout;
    let utxo = UTXO {
        value: 10000,
        script_pubkey: vec![OP_1].into(), // OP_1
        height: 0,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    // 3. Validate transaction inputs (should pass script validation)
    let (result, fee) = consensus.validate_tx_inputs(&tx, &utxo_set, 100).unwrap();
    assert_eq!(result, ValidationResult::Valid);
    assert!(fee > 0);

    // 4. Test script verification directly
    let script_result = consensus
        .verify_script(
            &tx.inputs[0].script_sig,
            &tx.outputs[0].script_pubkey,
            None,
            0,
        )
        .unwrap();

    // OP_1 on both sides should evaluate to true with the current script engine.
    assert!(script_result);
}

/// Test integration between proof of work and block validation
#[test]
fn test_pow_block_integration() {
    let consensus = ConsensusProof::new();

    // 1. Create block with specific difficulty
    let mut block = create_valid_block();
    block.header.bits = 0x1800ffff; // Smaller target

    // 2. Test proof of work validation (expected to fail due to target expansion)
    let pow_result = consensus.check_proof_of_work(&block.header);
    // Expected to fail due to target expansion issues
    // With improved implementation, this should return a boolean result
    assert!(pow_result.is_ok());
    let is_valid = pow_result.unwrap();
    // The header should be invalid (hash >= target)
    assert!(!is_valid);

    // 3. Test difficulty adjustment
    let prev_headers = vec![block.header.clone(), block.header.clone()];
    let next_work = consensus
        .get_next_work_required(&block.header, &prev_headers)
        .unwrap();
    assert!(next_work > 0); // Should return valid target

    // 4. Validate block structure on regtest (PoW skipped; tests non-PoW rules only)
    let utxo_set = UtxoSet::default();
    let mut regtest_block = create_valid_regtest_block(0);
    regtest_block.header.bits = 0x207fffff;
    regtest_block.header.version = 4;
    let witnesses: Vec<Vec<blvm_consensus::segwit::Witness>> = regtest_block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let time_context = Some(TimeContext {
        network_time: regtest_block.header.timestamp,
        median_time_past: regtest_block.header.timestamp,
    });
    let (validation_result, _new_utxo_set) = consensus
        .validate_block_with_time_context(
            &regtest_block,
            &witnesses,
            utxo_set,
            0,
            time_context,
            Network::Regtest,
        )
        .unwrap();
    assert_eq!(validation_result, ValidationResult::Valid);
}

/// Test cross-system error handling
#[test]
fn test_cross_system_error_handling() {
    let consensus = ConsensusProof::new();

    // 1. Test invalid transaction in mempool
    let invalid_tx = create_invalid_transaction();
    let utxo_set = UtxoSet::default();
    let mempool = mempool::Mempool::new();

    let time_context = None;
    let mempool_result = consensus
        .accept_to_memory_pool(
            &invalid_tx,
            &utxo_set,
            &mempool,
            100,
            time_context,
            Network::Mainnet,
        )
        .unwrap();
    assert!(
        matches!(mempool_result, mempool::MempoolResult::Rejected(_)),
        "empty-input transaction must be rejected by mempool policy"
    );

    // 2. Invalid block creation: create_new_block must skip rejected mempool txs
    let result = consensus.create_new_block(
        &utxo_set,
        &[invalid_tx],
        100,
        &create_valid_block_header(),
        &[create_valid_block_header(), create_valid_block_header()],
        &vec![OP_1],
        &vec![OP_1],
    );

    // Should succeed but create block without invalid transactions
    assert!(result.is_ok());
    let block = result.unwrap();
    assert_eq!(block.transactions.len(), 1); // Only coinbase
}

/// Test performance integration between systems
#[test]
fn test_performance_integration() {
    let consensus = ConsensusProof::new();

    // 1. Create large UTXO set
    let mut utxo_set = UtxoSet::default();
    for i in 0..1000 {
        let outpoint = OutPoint {
            hash: [i as u8; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        };
        utxo_set.insert(outpoint, std::sync::Arc::new(utxo));
    }

    // 2. Create multiple transactions
    let mut mempool_txs = Vec::new();
    for i in 0..100 {
        let mut tx = create_test_tx(1000, None, None, None);
        tx.inputs[0].prevout = OutPoint {
            hash: [(i % 1000) as u8; 32],
            index: 0,
        };
        mempool_txs.push(tx);
    }

    // 3. Test mempool acceptance performance
    let start = std::time::Instant::now();
    let mut accepted = 0;
    let mempool = mempool::Mempool::new();

    let time_context = None; // No time context for this test
    for tx in &mempool_txs {
        let result = consensus
            .accept_to_memory_pool(tx, &utxo_set, &mempool, 100, time_context, Network::Mainnet)
            .unwrap();
        if matches!(result, mempool::MempoolResult::Accepted) {
            accepted += 1;
        }
    }

    let duration = start.elapsed();
    let max_duration_ms = adjusted_timeout(1000); // Adjust for CI environment
    assert!(
        duration.as_millis() < max_duration_ms as u128,
        "Performance test: accepted {}/{} transactions in {:?} (max: {}ms, CI: {})",
        accepted,
        mempool_txs.len(),
        duration,
        max_duration_ms,
        is_ci()
    );
    println!(
        "Accepted {}/{} transactions in {:?} (CI: {})",
        accepted,
        mempool_txs.len(),
        duration,
        is_ci()
    );

    // 4. Test block creation performance
    let start = std::time::Instant::now();
    let block = consensus
        .create_new_block(
            &utxo_set,
            &mempool_txs,
            100,
            &create_valid_block_header(),
            &[create_valid_block_header(), create_valid_block_header()],
            &vec![OP_1, OP_1], // 2 bytes for coinbase script_sig
            &vec![OP_1],
        )
        .unwrap();

    let duration = start.elapsed();
    let max_duration_ms = adjusted_timeout(1000); // Adjust for CI environment
    assert!(
        duration.as_millis() < max_duration_ms as u128,
        "Performance test: created block with {} transactions in {:?} (max: {}ms, CI: {})",
        block.transactions.len(),
        duration,
        max_duration_ms,
        is_ci()
    );
    println!(
        "Created block with {} transactions in {:?}",
        block.transactions.len(),
        duration
    );
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Transaction and UTXO creation helpers are now in test_helpers.rs

fn encode_bip34_height(height: u64) -> Vec<u8> {
    if height == 0 {
        return vec![0x00, 0xff];
    }
    let mut height_bytes = Vec::new();
    let mut n = height;
    while n > 0 {
        height_bytes.push((n & 0xff) as u8);
        n >>= 8;
    }
    if height_bytes.last().is_some_and(|&b| b & 0x80 != 0) {
        height_bytes.push(0x00);
    }
    let mut script_sig = Vec::with_capacity(1 + height_bytes.len() + 1);
    script_sig.push(height_bytes.len() as u8);
    script_sig.extend_from_slice(&height_bytes);
    if script_sig.len() < 2 {
        script_sig.push(0xff);
    }
    script_sig
}

fn create_valid_block_header() -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x207fffff,
        nonce: 0,
    }
}

fn create_valid_regtest_block(height: u64) -> Block {
    let coinbase = Transaction {
        version: 1,
        inputs: tx_inputs![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: encode_bip34_height(height),
            sequence: 0xffffffff,
        }],
        outputs: tx_outputs![TransactionOutput {
            value: 50 * blvm_consensus::orange_paper_constants::C as i64,
            script_pubkey: vec![OP_1],
        }],
        lock_time: 0,
    };
    let merkle_root = mining::calculate_merkle_root(&[coinbase.clone()]).unwrap();
    Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp: 1_700_000_000,
            bits: 0x207fffff,
            nonce: 0,
        },
        transactions: vec![coinbase].into_boxed_slice(),
    }
}

fn create_valid_block() -> Block {
    Block {
        header: create_valid_block_header(),
        transactions: vec![Transaction {
            version: 1,
            inputs: tx_inputs![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![OP_1],
                sequence: 0xffffffff,
            }],
            outputs: tx_outputs![TransactionOutput {
                value: 50 * blvm_consensus::orange_paper_constants::C as i64, // Initial subsidy = 50 BTC
                script_pubkey: vec![OP_1],
            }],
            lock_time: 0,
        }]
        .into_boxed_slice(),
    }
}
