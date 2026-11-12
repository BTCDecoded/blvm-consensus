use super::*;
use crate::opcodes::*;

#[test]
fn sigop_adjusted_vsize_matches_core_rule() {
    // Weight wins: 400 weight, 1 sigop * 20 = 20. vsize = ceil(400/4) = 100.
    assert_eq!(sigop_adjusted_vsize(400, 1, DEFAULT_BYTES_PER_SIGOP), 100);
    // Sigop term wins: 100 weight, 10 sigops * 20 = 200. vsize = ceil(200/4) = 50.
    assert_eq!(sigop_adjusted_vsize(100, 10, DEFAULT_BYTES_PER_SIGOP), 50);
    // Ceil: 201 weight is not divisible by 4.
    assert_eq!(sigop_adjusted_vsize(201, 0, DEFAULT_BYTES_PER_SIGOP), 51);
    // Default multiplier is 20 weight units per sigop (5 vbytes).
    assert_eq!(DEFAULT_BYTES_PER_SIGOP, 20);
    assert_eq!(MAX_STANDARD_TX_SIGOPS_COST, 16_000);
}

#[test]
fn test_disabled_opcode_spend_is_rejected() {
    let prevout = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        prevout,
        std::sync::Arc::new(UTXO {
            value: 10_000,
            script_pubkey: vec![OP_MUL].into(),
            height: 0,
            is_coinbase: false,
        }),
    );
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    };
    let result = accept_to_memory_pool(
        &tx,
        None,
        &utxo_set,
        &Mempool::new(),
        200,
        Some(TimeContext {
            network_time: 1_700_000_000,
            median_time_past: 0,
        }),
        Network::Mainnet,
        None,
    )
    .expect("a disabled opcode is a rejection");
    assert!(matches!(result, MempoolResult::Rejected(_)));
}

#[test]
fn test_accept_to_memory_pool_valid() {
    // Skip script validation for now - focus on mempool logic
    let tx = create_valid_transaction();
    let utxo_set = create_test_utxo_set();
    let mempool = Mempool::new();

    // Script: sig=OP_1, spk=OP_1 (UTXO). Stack after: [[1],[1]], top truthy → Accepted.
    let time_context = Some(TimeContext {
        network_time: 1234567890,
        median_time_past: 1234567890,
    });
    let result = accept_to_memory_pool(
        &tx,
        None,
        &utxo_set,
        &mempool,
        100,
        time_context,
        Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(matches!(result, MempoolResult::Accepted));
}

#[test]
fn relative_height_lock_is_not_accepted_until_it_is_old_enough() {
    let mut tx = create_valid_transaction();
    tx.version = 2;
    tx.inputs[0].sequence = 10;
    let prevout = tx.inputs[0].prevout;
    let time_context = Some(TimeContext {
        network_time: 1_700_000_000,
        median_time_past: 1_700_000_000,
    });
    let mempool = Mempool::new();
    let young = utxo_confirmed_at(prevout, 499_995);
    let result = accept_to_memory_pool(
        &tx,
        None,
        &young,
        &mempool,
        500_000,
        time_context,
        Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(matches!(result, MempoolResult::Rejected(_)));

    let old = utxo_confirmed_at(prevout, 499_000);
    let result = accept_to_memory_pool(
        &tx,
        None,
        &old,
        &mempool,
        500_000,
        time_context,
        Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(matches!(result, MempoolResult::Accepted));

    tx.version = 1;
    let result = accept_to_memory_pool(
        &tx,
        None,
        &young,
        &mempool,
        500_000,
        time_context,
        Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(matches!(result, MempoolResult::Accepted));
}

#[test]
fn time_relative_lock_uses_prior_median() {
    const SEQUENCE_LOCKTIME_TYPE_FLAG: u64 = 0x0040_0000;
    let mut tx = create_valid_transaction();
    tx.version = 2;
    tx.inputs[0].sequence = SEQUENCE_LOCKTIME_TYPE_FLAG | 2;
    let prevout = tx.inputs[0].prevout;
    let utxo = utxo_confirmed_at(prevout, 499_000);
    let mempool = Mempool::new();
    let prior = [1_000_000i64];
    let inside = Some(TimeContext {
        network_time: 1_001_023,
        median_time_past: 1_001_023,
    });
    let rejected = accept_to_memory_pool(
        &tx,
        None,
        &utxo,
        &mempool,
        500_000,
        inside,
        Network::Mainnet,
        Some(&prior),
    )
    .unwrap();
    assert!(matches!(rejected, MempoolResult::Rejected(_)));

    let old_enough = Some(TimeContext {
        network_time: 1_001_024,
        median_time_past: 1_001_024,
    });
    let accepted = accept_to_memory_pool(
        &tx,
        None,
        &utxo,
        &mempool,
        500_000,
        old_enough,
        Network::Mainnet,
        Some(&prior),
    )
    .unwrap();
    assert!(matches!(accepted, MempoolResult::Accepted));
}

#[test]
fn test_accept_to_memory_pool_duplicate() {
    let tx = create_valid_transaction();
    let utxo_set = create_test_utxo_set();
    let mut mempool = Mempool::new();
    mempool.insert(crate::block::calculate_tx_id(&tx));

    let time_context = Some(TimeContext {
        network_time: 1234567890,
        median_time_past: 1234567890,
    });
    let result = accept_to_memory_pool(
        &tx,
        None,
        &utxo_set,
        &mempool,
        100,
        time_context,
        Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(matches!(result, MempoolResult::Rejected(_)));
}

#[test]
fn test_is_standard_tx_valid() {
    let tx = create_valid_transaction();
    assert!(is_standard_tx(&tx).unwrap());
}

#[test]
fn test_is_standard_tx_too_large() {
    let mut tx = create_valid_transaction();
    // Make transaction too large by adding many inputs
    // MAX_INPUTS (100,000) * ~42 bytes/input >> MAX_TX_SIZE (1,000,000)
    for _ in 0..MAX_INPUTS {
        tx.inputs.push(create_dummy_input());
    }
    // Transaction exceeds MAX_TX_SIZE so it should NOT be standard
    assert!(!is_standard_tx(&tx).unwrap());
}

#[test]
fn test_replacement_checks_all_requirements() {
    let utxo_set = create_test_utxo_set();
    let mempool = Mempool::new();

    // Create existing transaction with RBF signaling and lower fee
    let mut existing_tx = create_valid_transaction();
    // BIP125 opts in strictly below SEQUENCE_RBF (0xfffffffe).
    existing_tx.inputs[0].sequence = (SEQUENCE_RBF as u64) - 1;
    existing_tx.outputs[0].value = 9000; // Fee = 10000 - 9000 = 1000 sats

    // Create new transaction that:
    // 1. Signals RBF (or doesn't - per BIP125 only existing needs to signal)
    // 2. Conflicts with existing (same input)
    // 3. Has higher fee rate and absolute fee
    let mut new_tx = existing_tx.clone();
    new_tx.outputs[0].value = 8000; // Fee = 10000 - 8000 = 2000 sats
    // Higher fee rate and absolute fee bump (2000 > 1000 + 1000 = 2000, needs >)
    new_tx.outputs[0].value = 7999; // Fee = 10000 - 7999 = 2001 sats

    // Should pass all BIP125 checks
    let result = replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap();
    assert!(result, "Valid RBF replacement should be accepted");
}

#[test]
fn test_replacement_checks_no_rbf_signal() {
    let utxo_set = create_test_utxo_set();
    let mempool = Mempool::new();

    let existing_tx = create_valid_transaction();
    let mut new_tx = existing_tx.clone();
    new_tx.outputs[0].value = 500;

    assert!(!signals_rbf(&existing_tx));
    assert!(replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
}

#[test]
fn test_replacement_checks_no_conflict() {
    let mut utxo_set = create_test_utxo_set();
    // Add UTXO for the new transaction's input
    let new_outpoint = OutPoint {
        hash: [2; 32],
        index: 0,
    };
    let new_utxo = UTXO {
        value: 10000,
        script_pubkey: vec![OP_1].into(),
        height: 0,
        is_coinbase: false,
    };
    utxo_set.insert(new_outpoint, std::sync::Arc::new(new_utxo));

    let mempool = Mempool::new();

    let mut existing_tx = create_valid_transaction();
    existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;

    // New transaction with different input (no conflict)
    let mut new_tx = create_valid_transaction();
    new_tx.inputs[0].prevout.hash = [2; 32]; // Different input
    new_tx.inputs[0].sequence = SEQUENCE_RBF as u64;
    // Ensure output value doesn't exceed input value to avoid negative fee
    new_tx.outputs[0].value = 5000; // Less than input value of 10000

    // Should fail: no conflict (requirement #4)
    assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
}

#[test]
fn test_replacement_checks_fee_rate_too_low() {
    let utxo_set = create_test_utxo_set();
    let mempool = Mempool::new();

    // Existing transaction with higher fee rate
    let mut existing_tx = create_valid_transaction();
    existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;
    existing_tx.outputs[0].value = 5000; // Fee = 5000 sats, size = small

    // New transaction with same or lower fee rate (but higher absolute fee)
    let mut new_tx = existing_tx.clone();
    new_tx.outputs[0].value = 4999; // Fee = 5001 sats, but same size so same fee rate

    // Should fail: fee rate not higher (requirement #2)
    assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
}

#[test]
fn test_replacement_checks_absolute_fee_insufficient() {
    let utxo_set = create_test_utxo_set();
    let mempool = Mempool::new();

    // Existing transaction. Sequence is one below SEQUENCE_RBF, so it signals.
    let mut existing_tx = create_valid_transaction();
    existing_tx.inputs[0].sequence = (SEQUENCE_RBF as u64) - 1;
    existing_tx.outputs[0].value = 9000; // Fee = 1000 sats

    // One sat short of the 100 sat/kvB relay bump. The fee rate is higher.
    let bump = incremental_relay_bump(calculate_transaction_size_vbytes(&existing_tx, None) as u64);
    let mut new_tx = existing_tx.clone();
    new_tx.outputs[0].value = 10_000 - (1_000 + bump - 1);

    assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
}

// ============================================================================
// COMPREHENSIVE MEMPOOL TESTS
// ============================================================================

#[test]
fn test_accept_to_memory_pool_coinbase() {
    let coinbase_tx = create_coinbase_transaction();
    let utxo_set = UtxoSet::default();
    let mempool = Mempool::new();
    // Coinbase transactions should be rejected from mempool
    let time_context = Some(TimeContext {
        network_time: 0,
        median_time_past: 0,
    });
    let result = accept_to_memory_pool(
        &coinbase_tx,
        None,
        &utxo_set,
        &mempool,
        100,
        time_context,
        Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(matches!(result, MempoolResult::Rejected(_)));
}

#[test]
fn test_is_standard_tx_large_script() {
    let mut tx = create_valid_transaction();
    // Create a script that's too large
    tx.inputs[0].script_sig = vec![OP_1; MAX_SCRIPT_SIZE + 1];

    let result = is_standard_tx(&tx).unwrap();
    assert!(!result);
}

#[test]
fn test_is_standard_tx_large_output_script() {
    let mut tx = create_valid_transaction();
    // Create an output script that's too large
    tx.outputs[0].script_pubkey = vec![OP_1; MAX_SCRIPT_SIZE + 1];

    let result = is_standard_tx(&tx).unwrap();
    assert!(!result);
}

#[test]
fn test_replacement_checks_new_unconfirmed_dependency() {
    let utxo_set = create_test_utxo_set();
    let mempool = Mempool::new();

    // Existing transaction
    let mut existing_tx = create_valid_transaction();
    existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;

    // New transaction that adds a new unconfirmed input
    let mut new_tx = existing_tx.clone();
    new_tx.inputs.push(TransactionInput {
        prevout: OutPoint {
            hash: [99; 32],
            index: 0,
        }, // Not in UTXO set
        script_sig: vec![],
        sequence: SEQUENCE_RBF as u64,
    });
    new_tx.outputs[0].value = 7000; // Higher fee

    // Should fail: creates new unconfirmed dependency (requirement #5)
    assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
}

#[test]
fn test_has_conflict_with_tx_true() {
    let tx1 = create_valid_transaction();
    let mut tx2 = create_valid_transaction();
    tx2.inputs[0].prevout = tx1.inputs[0].prevout; // Same input = conflict

    assert!(has_conflict_with_tx(&tx2, &tx1));
}

#[test]
fn test_has_conflict_with_tx_false() {
    let tx1 = create_valid_transaction();
    let mut tx2 = create_valid_transaction();
    tx2.inputs[0].prevout.hash = [2; 32]; // Different input = no conflict

    assert!(!has_conflict_with_tx(&tx2, &tx1));
}

#[test]
fn test_replacement_checks_minimum_relay_fee() {
    let utxo_set = create_test_utxo_set();
    let mempool = Mempool::new();

    // Existing transaction
    let mut existing_tx = create_valid_transaction();
    existing_tx.inputs[0].sequence = (SEQUENCE_RBF as u64) - 1;
    existing_tx.outputs[0].value = 9500; // Fee = 500 sats

    let bump = incremental_relay_bump(calculate_transaction_size_vbytes(&existing_tx, None) as u64);
    let mut new_tx = existing_tx.clone();
    new_tx.outputs[0].value = 10_000 - (500 + bump - 1);
    assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
    new_tx.outputs[0].value = 10_000 - (500 + bump);
    assert!(replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
}

#[test]
fn test_check_mempool_rules_low_fee() {
    let tx = create_valid_transaction();
    let fee = 1; // Very low fee
    let mempool = Mempool::new();

    let result = check_mempool_rules(&tx, None, fee, &mempool).unwrap();
    assert!(!result);
}

#[test]
fn test_check_mempool_rules_high_fee() {
    let tx = create_valid_transaction();
    let fee = 10000; // High fee
    let mempool = Mempool::new();

    let result = check_mempool_rules(&tx, None, fee, &mempool).unwrap();
    assert!(result);
}

#[test]
fn test_mempool_total_vbytes_tracking() {
    let mut mempool = Mempool::new();
    let tx = create_valid_transaction();
    let vsize = calculate_transaction_size_vbytes(&tx, None);
    let txid = crate::block::calculate_tx_id(&tx);

    mempool.insert_transaction(&tx);
    assert_eq!(mempool.total_vbytes(), vsize);

    mempool.remove(&txid);
    assert_eq!(mempool.total_vbytes(), 0);
}

#[test]
fn mempool_vsize_includes_witness_stacks() {
    let tx = create_valid_transaction();
    let bare = calculate_transaction_size_vbytes(&tx, None);
    let empty_stacks: Vec<Witness> = vec![vec![]];
    assert_eq!(
        calculate_transaction_size_vbytes(&tx, Some(&empty_stacks)),
        bare
    );

    let heavy_stacks: Vec<Witness> = vec![vec![vec![0u8; 4_000]]];
    let heavy = calculate_transaction_size_vbytes(&tx, Some(&heavy_stacks));
    assert!(heavy > bare);

    let fee = (heavy as i64) - 1;
    assert!(fee >= 1_000);
    let mempool = Mempool::new();
    assert!(check_mempool_rules(&tx, None, fee, &mempool).unwrap());
    assert!(!check_mempool_rules(&tx, Some(&heavy_stacks), fee, &mempool).unwrap());

    let mut pool = Mempool::new();
    assert!(pool.insert_transaction_with_witness(&tx, Some(&heavy_stacks)));
    assert_eq!(pool.total_vbytes(), heavy);
}

#[test]
fn test_mempool_byte_limit_exceeded() {
    let mut mempool = Mempool::new();
    let tx = create_valid_transaction();
    let vsize = calculate_transaction_size_vbytes(&tx, None);
    mempool.insert_transaction(&tx);

    let at_limit = mempool.total_vbytes();
    assert!(!mempool_size_limits_exceeded(
        &mempool, 0, 1_000_000, at_limit
    ));
    assert!(mempool_size_limits_exceeded(
        &mempool, 1, 1_000_000, at_limit
    ));
    assert!(!mempool_size_limits_exceeded(
        &mempool,
        vsize,
        1_000_000,
        at_limit.saturating_add(vsize)
    ));
}

#[test]
fn test_check_mempool_rules_full_mempool() {
    let tx = create_valid_transaction();
    let fee = 10000;
    let mut mempool = Mempool::new();

    // Fill mempool to the tx-count limit (default max_mempool_txs is 100,000).
    for i in 0..100_000 {
        let mut hash = [0u8; 32];
        hash[0] = (i & 0xff) as u8;
        hash[1] = ((i >> 8) & 0xff) as u8;
        hash[2] = ((i >> 16) & 0xff) as u8;
        hash[3] = ((i >> 24) & 0xff) as u8;
        mempool.insert(hash);
    }

    assert_eq!(mempool.len(), 100_000);

    let result = check_mempool_rules(&tx, None, fee, &mempool).unwrap();
    assert!(!result);
}

#[test]
fn test_has_conflicts_no_conflicts() {
    let tx = create_valid_transaction();
    let mempool = Mempool::new();

    let result = has_conflicts(&tx, &mempool).unwrap();
    assert!(!result);
}

#[test]
fn test_has_conflicts_with_conflicts() {
    let tx = create_valid_transaction();
    let mut mempool = Mempool::new();

    // Add a conflicting transaction to mempool (same outpoint, different txid)
    let mut pool_tx = tx.clone();
    pool_tx.version = 2;
    mempool.insert_transaction(&pool_tx);

    let result = has_conflicts(&tx, &mempool).unwrap();
    assert!(result);
}

#[test]
fn test_signals_rbf_true() {
    let mut tx = create_valid_transaction();
    tx.inputs[0].sequence = 0xfffffffd; // below 0xfffffffe

    assert!(signals_rbf(&tx));
}

#[test]
fn test_signals_rbf_locktime_sequence_opts_in() {
    let mut tx = create_valid_transaction();
    tx.inputs[0].sequence = 0xfffffffe;

    assert!(signals_rbf(&tx));
}

#[test]
fn test_signals_rbf_false() {
    let tx = create_valid_transaction(); // sequence = 0xffffffff (final)

    assert!(!signals_rbf(&tx));
}

#[test]
fn test_calculate_fee_rate() {
    let tx = create_valid_transaction();
    let utxo_set = create_test_utxo_set();
    let fee = calculate_fee(&tx, &utxo_set);

    // Fee should be calculable (may be 0 for valid transactions)
    assert!(fee.is_ok());
}

#[test]
fn test_creates_new_dependencies_no_new() {
    let new_tx = create_valid_transaction();
    let existing_tx = create_valid_transaction();
    let mempool = Mempool::new();

    let utxo_set = create_test_utxo_set();
    let result = creates_new_dependencies(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap();
    assert!(!result);
}

#[test]
fn test_creates_new_dependencies_with_new() {
    let mut new_tx = create_valid_transaction();
    let existing_tx = create_valid_transaction();
    let mempool = Mempool::new();

    // Make new_tx spend a different input
    new_tx.inputs[0].prevout.hash = [2; 32];

    let utxo_set = create_test_utxo_set();
    let result = creates_new_dependencies(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap();
    assert!(result);
}

#[test]
fn test_is_standard_script_empty() {
    let script = vec![];
    let result = is_standard_script(&script).unwrap();
    assert!(!result);
}

#[test]
fn test_is_standard_script_too_large() {
    let script = vec![OP_1; MAX_SCRIPT_SIZE + 1];
    let result = is_standard_script(&script).unwrap();
    assert!(!result);
}

#[test]
fn test_is_standard_script_non_standard_opcode() {
    let script = vec![OP_VERIF]; // Non-standard opcode (disabled)
    let result = is_standard_script(&script).unwrap();
    assert!(!result);
}

#[test]
fn test_is_standard_script_valid() {
    let script = vec![OP_1];
    let result = is_standard_script(&script).unwrap();
    assert!(result);
}

#[test]
fn test_calculate_tx_id() {
    let tx = create_valid_transaction();
    let tx_id = crate::block::calculate_tx_id(&tx);

    // Should be a 32-byte hash
    assert_eq!(tx_id.len(), 32);

    // Same transaction should produce same ID
    let tx_id2 = crate::block::calculate_tx_id(&tx);
    assert_eq!(tx_id, tx_id2);
}

#[test]
fn test_calculate_tx_id_different_txs() {
    let tx1 = create_valid_transaction();
    let mut tx2 = tx1.clone();
    tx2.version = 2; // Different version

    let id1 = crate::block::calculate_tx_id(&tx1);
    let id2 = crate::block::calculate_tx_id(&tx2);

    assert_ne!(id1, id2);
}

#[test]
fn test_calculate_transaction_size() {
    let tx = create_valid_transaction();
    let size = calculate_transaction_size(&tx);

    assert!(size > 0);

    // Size should be deterministic
    let size2 = calculate_transaction_size(&tx);
    assert_eq!(size, size2);
}

#[test]
fn test_calculate_transaction_size_multiple_inputs_outputs() {
    let mut tx = create_valid_transaction();
    tx.inputs.push(create_dummy_input());
    tx.outputs.push(create_dummy_output());

    let size = calculate_transaction_size(&tx);
    assert!(size > 0);
}

#[test]
fn test_is_coinbase_true() {
    let coinbase_tx = create_coinbase_transaction();
    assert!(is_coinbase(&coinbase_tx));
}

#[test]
fn test_is_coinbase_false() {
    let regular_tx = create_valid_transaction();
    assert!(!is_coinbase(&regular_tx));
}

// Helper functions for tests
fn create_valid_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![create_dummy_input()].into(),
        outputs: vec![create_dummy_output()].into(),
        lock_time: 0,
    }
}

fn create_dummy_input() -> TransactionInput {
    TransactionInput {
        prevout: OutPoint {
            hash: [1; 32],
            index: 0,
        },
        script_sig: vec![OP_1],
        sequence: 0xffffffff,
    }
}

fn create_dummy_output() -> TransactionOutput {
    TransactionOutput {
        value: 1000,
        script_pubkey: vec![OP_1], // OP_1 for valid script
    }
}

fn utxo_confirmed_at(prevout: OutPoint, height: u64) -> UtxoSet {
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        prevout,
        std::sync::Arc::new(UTXO {
            value: 10000,
            script_pubkey: vec![OP_1].into(),
            height,
            is_coinbase: false,
        }),
    );
    utxo_set
}

fn create_test_utxo_set() -> UtxoSet {
    let mut utxo_set = UtxoSet::default();
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 10000,
        script_pubkey: vec![OP_1].into(), // OP_1 for valid script
        height: 0,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));
    utxo_set
}

fn create_coinbase_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    }
}
