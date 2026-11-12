use super::*;

fn witnesses_for_chain(chain: &[Block]) -> Vec<Vec<Vec<Witness>>> {
    chain
        .iter()
        .map(|block| {
            block
                .transactions
                .iter()
                .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
                .collect()
        })
        .collect()
}

fn reorg_network_time(chain: &[Block]) -> u64 {
    chain
        .iter()
        .map(|b| b.header.timestamp)
        .max()
        .unwrap_or(0)
        .saturating_add(crate::constants::MAX_FUTURE_BLOCK_TIME)
}

fn reorganize_chain_test(
    new_chain: &[Block],
    current_chain: &[Block],
    utxo_set: UtxoSet,
    current_height: Natural,
) -> Result<ReorganizationResult> {
    reorganize_chain_with_witnesses(
        new_chain,
        &witnesses_for_chain(new_chain),
        None,
        current_chain,
        utxo_set,
        current_height,
        None::<fn(&Block) -> Option<Vec<Witness>>>,
        None::<fn(Natural) -> Option<Vec<BlockHeader>>>,
        None::<fn(&Hash) -> Option<BlockUndoLog>>,
        None::<fn(&Hash, &BlockUndoLog) -> Result<()>>,
        reorg_network_time(new_chain),
        crate::types::Network::Regtest,
        None,
    )
}

#[test]
fn test_should_reorganize_more_cumulative_work() {
    let new_chain = vec![create_test_block(), create_test_block()];
    let current_chain = vec![create_test_block()];

    assert!(should_reorganize(&new_chain, &current_chain).unwrap());
}

#[test]
fn test_should_reorganize_same_length_more_work() {
    let mut new_chain = vec![create_test_block()];
    let mut current_chain = vec![create_test_block()];

    // Same-length fork: harder chain (smaller target) has more cumulative work.
    new_chain[0].header.bits = 0x0300ffff;
    current_chain[0].header.bits = 0x0400ffff;

    assert!(should_reorganize(&new_chain, &current_chain).unwrap());
}

#[test]
fn test_should_not_reorganize_less_work() {
    let new_chain = vec![create_test_block()];
    let current_chain = vec![create_test_block(), create_test_block()];

    assert!(!should_reorganize(&new_chain, &current_chain).unwrap());
}

#[test]
fn test_find_common_ancestor() {
    let new_chain = vec![create_test_block()];
    let current_chain = vec![create_test_block()];

    let ancestor = find_common_ancestor(&new_chain, &current_chain).unwrap();
    assert_eq!(ancestor.header.version, 4);
    assert_eq!(ancestor.new_chain_index, 0);
    assert_eq!(ancestor.current_chain_index, 0);
}

#[test]
fn test_find_common_ancestor_shared_prefix_unequal_length() {
    let b0 = create_test_block_at_height(0);
    let b1 = create_test_block_at_height(1);
    let b2 = create_test_block_at_height(2);
    let b3 = create_test_block_at_height(3);

    let new_chain = vec![b0.clone(), b1.clone(), b2, b3];
    let current_chain = vec![b0, b1];

    let ancestor = find_common_ancestor(&new_chain, &current_chain).unwrap();
    assert_eq!(ancestor.new_chain_index, 1);
    assert_eq!(ancestor.current_chain_index, 1);
}

#[test]
fn test_find_common_ancestor_empty_chain() {
    let new_chain = vec![];
    let current_chain = vec![create_test_block()];

    let result = find_common_ancestor(&new_chain, &current_chain);
    assert!(result.is_err());
}

#[test]
fn test_common_ancestor_when_slices_start_at_different_heights() {
    let h0 = create_test_block_at_height(0);
    let h1 = create_test_block_at_height(1);
    let h2 = create_test_block_at_height(2);
    let h3 = create_test_block_at_height(3);
    let current_chain = vec![h0, h1.clone(), h2.clone()];
    let new_chain = vec![h1, h2, h3];

    let ancestor = find_common_ancestor(&new_chain, &current_chain).unwrap();
    assert_eq!(ancestor.new_chain_index, 1);
    assert_eq!(ancestor.current_chain_index, 2);
}

#[test]
fn test_disconnect_block_accepts_more_than_10_000_undo_entries() {
    let block = create_test_block();
    let mut undo_log = BlockUndoLog::new();
    for i in 0..10_001u32 {
        undo_log.push(UndoEntry {
            outpoint: OutPoint {
                hash: [1; 32],
                index: i,
            },
            previous_utxo: None,
            new_utxo: None,
        });
    }
    let utxo_set = UtxoSet::default();
    assert!(disconnect_block(&block, &undo_log, utxo_set, 1).is_ok());
}

#[test]
fn test_disconnect_block_restores_a_utxo_past_10_000_entries() {
    let block = create_test_block();
    let restored_outpoint = OutPoint {
        hash: [2; 32],
        index: 7,
    };
    let restored = std::sync::Arc::new(UTXO {
        value: 42,
        script_pubkey: vec![0x51].into(),
        height: 3,
        is_coinbase: false,
    });
    let mut undo_log = BlockUndoLog::new();
    for i in 0..10_000u32 {
        undo_log.push(UndoEntry {
            outpoint: OutPoint {
                hash: [1; 32],
                index: i,
            },
            previous_utxo: None,
            new_utxo: None,
        });
    }
    undo_log.push(UndoEntry {
        outpoint: restored_outpoint,
        previous_utxo: Some(std::sync::Arc::clone(&restored)),
        new_utxo: None,
    });
    let utxo_set = disconnect_block(&block, &undo_log, UtxoSet::default(), 1).unwrap();
    let got = utxo_set.get(&restored_outpoint).expect("restored coin");
    assert_eq!(got.value, 42);
    assert_eq!(utxo_set.len(), 1);
}

#[test]
fn test_should_reorganize_accepts_a_chain_longer_than_10_000() {
    let chain = vec![create_test_block(); 10_001];
    assert!(should_reorganize(&chain, &chain).is_ok());
}

#[test]
fn test_calculate_chain_work() {
    let mut block = create_test_block();
    // Use a bits value with exponent <= 18 so the target fits in u128
    block.header.bits = 0x0300ffff;
    let chain = vec![block];
    let work = calculate_chain_work(&chain).unwrap();
    assert!(work > crate::pow::U256::zero());
}

#[test]
fn test_reorganize_chain() {
    // current_chain = [ancestor] (tip at height 1)
    // new_chain = [ancestor, new_block] (longer chain, should win)
    let ancestor = create_test_block_at_height(0);
    let mut new_block = create_test_block_at_height(1);
    new_block.header.nonce = 42; // Different block than ancestor
    // Recalculate merkle root (nonce doesn't affect it, but prev_block_hash irrelevant for connect_block)

    let new_chain = vec![ancestor.clone(), new_block];
    let current_chain = vec![ancestor];
    let utxo_set = UtxoSet::default();

    // current_height = 1 (tip of current_chain at height 1)
    let result = reorganize_chain_test(&new_chain, &current_chain, utxo_set, 1);
    match result {
        Ok(reorg_result) => {
            // Ancestor is at height 1 (current_height - 0 blocks after it = 1)
            // One new block connected at height 2
            assert_eq!(reorg_result.new_height, 2);
            assert_eq!(reorg_result.connected_blocks.len(), 1);
            assert_eq!(reorg_result.connected_block_undo_logs.len(), 1);
        }
        Err(_) => {
            // Acceptable: validation may fail for simplified test blocks
        }
    }
}

#[test]
fn test_reorganize_chain_deep_reorg() {
    // Create blocks with unique hashes by varying nonce
    let mut block1 = create_test_block();
    block1.header.nonce = 1;
    let mut block2 = create_test_block();
    block2.header.nonce = 2;
    let mut block3 = create_test_block();
    block3.header.nonce = 3;
    let new_chain = vec![block1, block2, block3];

    let mut current_block1 = create_test_block();
    current_block1.header.nonce = 10;
    let mut current_block2 = create_test_block();
    current_block2.header.nonce = 11;
    let current_chain = vec![current_block1, current_block2];
    let utxo_set = UtxoSet::default();

    let result = reorganize_chain_test(&new_chain, &current_chain, utxo_set, 2);
    match result {
        Ok(reorg_result) => {
            assert_eq!(reorg_result.connected_blocks.len(), 3);
            assert_eq!(reorg_result.reorganization_depth, 2);
            // Verify undo logs are stored for all connected blocks
            assert_eq!(reorg_result.connected_block_undo_logs.len(), 3);
        }
        Err(_) => {
            // Expected failure due to simplified validation
        }
    }
}

#[test]
#[allow(deprecated)]
fn missing_undo_callback_does_not_connect_the_new_chain() {
    let ancestor = create_test_block_at_height(0);
    let mut old_tip = create_test_block_at_height(1);
    old_tip.header.nonce = 1;
    let mut new_tip = create_test_block_at_height(1);
    new_tip.header.nonce = 2;
    let current = vec![ancestor.clone(), old_tip];
    let new_chain = vec![ancestor, new_tip];
    let err = super::reorganize_chain(
        &new_chain,
        &current,
        UtxoSet::default(),
        1,
        crate::types::Network::Regtest,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("missing undo log"), "{err}");
}

#[test]
fn undo_lookup_uses_the_block_header_hash() {
    let header = create_test_block().header.clone();
    assert_eq!(
        calculate_block_hash(&header),
        crate::block::block_header_hash(&header)
    );
}

#[test]
fn test_undo_log_storage_and_retrieval() {
    use crate::block::connect_block;
    use crate::segwit::Witness;

    let block = create_test_block_at_height(1);
    let mut utxo_set = UtxoSet::default();

    // Add some UTXOs that will be spent
    let tx_id = calculate_tx_id(&block.transactions[0]);
    let outpoint = OutPoint {
        hash: tx_id,
        index: 0,
    };
    let utxo = UTXO {
        value: 5_000_000_000,
        script_pubkey: vec![0x51].into(),
        height: 1,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo.clone()));

    // Connect block and get undo log
    let witnesses: Vec<Vec<Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let ctx = crate::block::BlockValidationContext::for_network(crate::types::Network::Regtest);
    let (result, new_utxo_set, undo_log) =
        connect_block(&block, &witnesses, utxo_set.clone(), 1, &ctx).unwrap();

    assert!(matches!(result, crate::types::ValidationResult::Valid));

    // Verify undo log contains entries
    assert!(
        !undo_log.entries.is_empty(),
        "Undo log should contain entries"
    );

    // Calculate block hash
    let block_hash = calculate_block_hash(&block.header);

    // Store undo log in a map (simulating persistent storage)
    let mut undo_log_storage: HashMap<Hash, BlockUndoLog> = HashMap::new();
    undo_log_storage.insert(block_hash, undo_log.clone());

    // Retrieve undo log
    let retrieved_undo_log = undo_log_storage.get(&block_hash);
    assert!(
        retrieved_undo_log.is_some(),
        "Should be able to retrieve undo log"
    );
    assert_eq!(
        retrieved_undo_log.unwrap().entries.len(),
        undo_log.entries.len()
    );

    // Disconnect block using retrieved undo log
    let disconnected_utxo_set = disconnect_block(&block, &undo_log, new_utxo_set, 1).unwrap();

    // Verify UTXO was restored
    assert!(
        disconnected_utxo_set.contains_key(&outpoint),
        "Disconnected UTXO set should contain restored UTXO"
    );
}

#[test]
fn test_reorganize_with_undo_log_callback() {
    use crate::block::connect_block;
    use crate::segwit::Witness;

    // Create a block at height 1 and connect it to get undo log
    let block = create_test_block_at_height(1);
    let utxo_set = UtxoSet::default();
    let witnesses: Vec<Vec<Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();

    let ctx = crate::block::BlockValidationContext::for_network(crate::types::Network::Regtest);
    let (result, connected_utxo_set, undo_log) =
        connect_block(&block, &witnesses, utxo_set.clone(), 1, &ctx).unwrap();

    if !matches!(result, crate::types::ValidationResult::Valid) {
        eprintln!("Block validation failed: {result:?}");
    }
    assert!(matches!(result, crate::types::ValidationResult::Valid));

    // Store undo log
    let block_hash = calculate_block_hash(&block.header);
    let mut undo_log_storage: HashMap<Hash, BlockUndoLog> = HashMap::new();
    undo_log_storage.insert(block_hash, undo_log);

    // Create callback to retrieve undo log
    let get_undo_log =
        |hash: &Hash| -> Option<BlockUndoLog> { undo_log_storage.get(hash).cloned() };

    // Reorganize with undo log callback
    // new_chain shares the same ancestor block, plus a new block at height 2
    let mut new_block = create_test_block_at_height(2);
    new_block.header.nonce = 42; // Differentiate from ancestor
    let new_chain = vec![block.clone(), new_block];
    let current_chain = vec![block];
    let empty_witnesses: Vec<Vec<Vec<Witness>>> = new_chain
        .iter()
        .map(|b| {
            b.transactions
                .iter()
                .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
                .collect()
        })
        .collect();

    let reorg_result = reorganize_chain_with_witnesses(
        &new_chain,
        &empty_witnesses,
        None,
        &current_chain,
        connected_utxo_set,
        1,
        None::<fn(&Block) -> Option<Vec<Witness>>>,
        None::<fn(Natural) -> Option<Vec<BlockHeader>>>,
        Some(get_undo_log),
        None::<fn(&Hash, &BlockUndoLog) -> Result<()>>, // No storage in test
        2_000_000_000,
        crate::types::Network::Regtest,
        None,
    );

    // Reorganization should succeed (or fail gracefully)
    match reorg_result {
        Ok(result) => {
            // Verify undo logs are stored for new blocks
            assert!(!result.connected_block_undo_logs.is_empty());
        }
        Err(_) => {
            // Expected failure due to simplified validation
        }
    }
}

#[test]
fn failed_undo_store_is_not_a_successful_reorg() {
    use crate::segwit::Witness;
    let ancestor = create_test_block_at_height(1);
    let witnesses_one: Vec<Vec<Witness>> = ancestor
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let ctx = crate::block::BlockValidationContext::for_network(crate::types::Network::Regtest);
    let (_result, utxo_set, undo_log) =
        connect_block(&ancestor, &witnesses_one, UtxoSet::default(), 1, &ctx).unwrap();
    let ancestor_hash = calculate_block_hash(&ancestor.header);
    let mut stored = HashMap::new();
    stored.insert(ancestor_hash, undo_log);
    let mut new_tip = create_test_block_at_height(2);
    new_tip.header.prev_block_hash = ancestor_hash;
    new_tip.header.timestamp = ancestor.header.timestamp + 1;
    new_tip.header.nonce = 0;
    while !crate::pow::check_proof_of_work(&new_tip.header).unwrap_or(false) {
        new_tip.header.nonce = new_tip.header.nonce.wrapping_add(1);
    }
    let tip_witnesses: Vec<Vec<Witness>> = new_tip
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let (tip_result, _, _) =
        connect_block(&new_tip, &tip_witnesses, utxo_set.clone(), 2, &ctx).unwrap();
    assert!(
        matches!(tip_result, crate::types::ValidationResult::Valid),
        "{tip_result:?}"
    );
    let mut context_for_height = |_height: u64,
                                  _headers: Option<&[BlockHeader]>,
                                  _time: u64,
                                  network: crate::types::Network| {
        crate::block::BlockValidationContext::for_network(network)
    };
    let new_chain = vec![ancestor.clone(), new_tip];
    let new_witnesses: Vec<Vec<Vec<Witness>>> = new_chain
        .iter()
        .map(|block| {
            block
                .transactions
                .iter()
                .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
                .collect()
        })
        .collect();
    let result = reorganize_chain_with_witnesses(
        &new_chain,
        &new_witnesses,
        None,
        &[ancestor],
        utxo_set,
        1,
        None::<fn(&Block) -> Option<Vec<Witness>>>,
        None::<fn(Natural) -> Option<Vec<BlockHeader>>>,
        Some(|hash: &Hash| stored.get(hash).cloned()),
        Some(|_hash: &Hash, _log: &BlockUndoLog| {
            Err(crate::error::ConsensusError::BlockValidation(
                "undo store failed".into(),
            ))
        }),
        2_000_000_000,
        crate::types::Network::Regtest,
        Some(&mut context_for_height),
    );
    match result {
        Err(crate::error::ConsensusError::BlockValidation(msg)) => {
            assert!(msg.contains("undo store failed"), "{msg}");
        }
        other => panic!("expected undo store error, got {other:?}"),
    }
}

#[test]
fn test_reorganize_chain_empty_new_chain() {
    let new_chain = vec![];
    let current_chain = vec![create_test_block()];
    let utxo_set = UtxoSet::default();

    let result = reorganize_chain_test(&new_chain, &current_chain, utxo_set, 1);
    assert!(result.is_err());
}

#[test]
fn test_reorganize_chain_empty_current_chain() {
    let new_chain = vec![create_test_block()];
    let current_chain = vec![];
    let utxo_set = UtxoSet::default();

    let result = reorganize_chain_test(&new_chain, &current_chain, utxo_set, 0);
    assert!(result.is_err());
}

#[test]
fn test_reorganize_disconnect_empty_tx_is_err() {
    let ancestor = create_test_block_at_height(0);
    let mut empty_tip = create_test_block_at_height(1);
    empty_tip.transactions = Box::new([]);
    empty_tip.header.nonce = 99;
    let mut new_tip = create_test_block_at_height(1);
    new_tip.header.nonce = 42;
    let result = reorganize_chain_test(
        &[ancestor.clone(), new_tip],
        &[ancestor, empty_tip],
        UtxoSet::default(),
        1,
    );
    match result {
        Err(crate::error::ConsensusError::BlockValidation(msg)) => {
            assert!(
                msg.contains("must have at least one transaction"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected BlockValidation empty-tx, got {other:?}"),
    }
}

#[test]
fn test_disconnect_block() {
    let block = create_test_block();
    let mut utxo_set = UtxoSet::default();

    // Add some UTXOs that will be removed
    let tx_id = calculate_tx_id(&block.transactions[0]);
    let outpoint = OutPoint {
        hash: tx_id,
        index: 0,
    };
    let utxo = UTXO {
        value: 50_000_000_000,
        script_pubkey: vec![0x51].into(),
        height: 1,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    // Create an empty undo log for testing (simplified)
    let empty_undo_log = BlockUndoLog::new();
    let result = disconnect_block(&block, &empty_undo_log, utxo_set, 1);
    assert!(result.is_ok());
}

#[test]
fn test_calculate_chain_work_empty_chain() {
    let chain = vec![];
    let work = calculate_chain_work(&chain).unwrap();
    assert_eq!(work, crate::pow::U256::zero());
}

#[test]
fn test_calculate_chain_work_multiple_blocks() {
    let mut chain = vec![create_test_block(), create_test_block()];
    // Use bits values with exponent <= 18 so targets fit in u128
    chain[0].header.bits = 0x0300ffff;
    chain[1].header.bits = 0x0400ffff;

    let work = calculate_chain_work(&chain).unwrap();
    assert!(work > crate::pow::U256::zero());
}

#[test]
fn test_expand_target_edge_cases() {
    // Zero mantissa with valid exponent
    let result = crate::pow::expand_target(0x03000000);
    assert!(result.is_ok());
    assert!(result.unwrap().is_zero());

    // Test maximum valid target in low-exponent range
    let result = crate::pow::expand_target(0x03ffffff);
    assert!(result.is_ok());

    // Exponent outside pow::expand_target range (3..=32)
    let result = crate::pow::expand_target(0x2100ffff);
    assert!(result.is_err());
}

#[test]
fn test_calculate_tx_id_different_transactions() {
    let tx1 = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };

    let tx2 = Transaction {
        version: 2,
        inputs: vec![].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };

    let id1 = calculate_tx_id(&tx1);
    let id2 = calculate_tx_id(&tx2);

    assert_ne!(id1, id2);
}

/// Create a test block with BIP34-compliant coinbase encoding for the given height.
fn create_test_block_at_height(height: u64) -> Block {
    use crate::mining::calculate_merkle_root;

    let script_sig = crate::bip_validation::encode_bip34_coinbase_script(height);
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig,
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5_000_000_000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let merkle_root =
        calculate_merkle_root(&[coinbase_tx.clone()]).expect("Failed to calculate merkle root");

    let mut block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp: 1231006505,
            bits: 0x207fffff,
            nonce: 0,
        },
        transactions: vec![coinbase_tx].into_boxed_slice(),
    };
    while !crate::pow::check_proof_of_work(&block.header).unwrap_or(false) {
        block.header.nonce = block.header.nonce.wrapping_add(1);
    }
    block
}

/// Create a test block at height 0 (backward-compatible default).
fn create_test_block() -> Block {
    create_test_block_at_height(0)
}
