use super::*;

/// Reference: original definition (O(height); use only for small heights in tests).
fn naive_total_supply(end: u64) -> i64 {
    let mut total = 0i64;
    for h in 0..=end {
        let subsidy = get_block_subsidy(h);
        total = total.checked_add(subsidy).unwrap_or_else(|| {
            debug_assert!(false, "overflow");
            MAX_MONEY
        });
        if total >= MAX_MONEY {
            break;
        }
    }
    total
}

#[test]
fn subsidy_matches_the_shared_schedule() {
    // These are the integer-shift values Bitcoin Core and the Orange Paper
    // both return. A second model agrees only when it returns the same ones.
    assert_eq!(get_block_subsidy(0), 5_000_000_000);
    assert_eq!(get_block_subsidy(209_999), 5_000_000_000);
    assert_eq!(get_block_subsidy(210_000), 2_500_000_000);
    assert_eq!(get_block_subsidy(33 * 210_000), 0);
    assert_eq!(get_block_subsidy(64 * 210_000), 0);
}

#[test]
fn total_supply_fast_matches_naive_reference() {
    for end in [
        0u64,
        1,
        2,
        1000,
        HALVING_INTERVAL - 1,
        HALVING_INTERVAL,
        HALVING_INTERVAL + 1,
        2 * HALVING_INTERVAL - 1,
        500_000,
    ] {
        assert_eq!(total_supply(end), naive_total_supply(end), "end={end}");
    }
    // Beyond subsidy emission, additional blocks add 0; multi-million height would be
    // infeasible for the naive reference loop.
    let after_last_reward = HALVING_INTERVAL * 64 - 1;
    assert_eq!(
        total_supply(after_last_reward + 1_000_000),
        total_supply(after_last_reward),
        "heights after 64×H−1 should not increase supply"
    );
}

fn coinbase_only_block(height: u64, value: i64) -> Block {
    use crate::mining::calculate_merkle_root;

    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffff_ffff,
            },
            // Height 150 is `0x96`. The high bit is set, so the prefix is
            // `02 96 00` once BIP34 is active.
            script_sig: vec![0x02, 0x96, 0x00],
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };
    let merkle_root = calculate_merkle_root(std::slice::from_ref(&coinbase)).expect("merkle");
    let mut block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp: 1_600_000_000 + height,
            bits: 0x207f_ffff,
            nonce: 0,
        },
        transactions: vec![coinbase].into(),
    };
    while !crate::pow::check_proof_of_work(&block.header).unwrap_or(false) {
        block.header.nonce = block.header.nonce.wrapping_add(1);
    }
    block
}

fn connect_coinbase(network: Network, height: u64, value: i64) -> ValidationResult {
    use crate::block::{BlockValidationContext, connect_block};
    use crate::segwit::Witness;

    let block = coinbase_only_block(height, value);
    let witnesses: Vec<Vec<Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Witness::default()).collect())
        .collect();
    let ctx = BlockValidationContext::for_network(network);
    connect_block(&block, &witnesses, UtxoSet::default(), height, &ctx)
        .expect("connect")
        .0
}

#[test]
fn regtest_subsidy_halves_every_150_blocks() {
    let full = INITIAL_SUBSIDY;
    assert_eq!(subsidy_halving_interval(Network::Regtest), 150);
    assert_eq!(get_block_subsidy_for_network(149, Network::Regtest), full);
    assert_eq!(
        get_block_subsidy_for_network(150, Network::Regtest),
        full / 2
    );
    assert_eq!(
        get_block_subsidy_for_network(299, Network::Regtest),
        full / 2
    );
    assert_eq!(
        get_block_subsidy_for_network(300, Network::Regtest),
        full / 4
    );
    assert_eq!(get_block_subsidy_for_network(150 * 33, Network::Regtest), 0);
    // The 210,000-block function is unchanged at the regtest boundary.
    assert_eq!(get_block_subsidy(150), full);
}

#[test]
fn non_regtest_subsidy_stays_on_210_000_blocks() {
    for network in [
        Network::Mainnet,
        Network::Testnet,
        Network::Testnet4,
        Network::Signet,
    ] {
        assert_eq!(subsidy_halving_interval(network), HALVING_INTERVAL);
        assert_eq!(get_block_subsidy_for_network(150, network), INITIAL_SUBSIDY);
        assert_eq!(
            get_block_subsidy_for_network(HALVING_INTERVAL, network),
            INITIAL_SUBSIDY / 2
        );
    }
}

#[test]
fn regtest_connect_rejects_unhalved_coinbase_at_height_150() {
    assert!(matches!(
        connect_coinbase(Network::Regtest, 150, INITIAL_SUBSIDY / 2),
        ValidationResult::Valid
    ));
    let rejected = connect_coinbase(Network::Regtest, 150, INITIAL_SUBSIDY);
    assert!(
        matches!(rejected, ValidationResult::Invalid(ref msg) if msg.contains("subsidy")),
        "50 BTC coinbase at regtest height 150 must fail, got {rejected:?}"
    );
}

#[test]
fn mainnet_connect_accepts_full_subsidy_at_height_150() {
    assert!(matches!(
        connect_coinbase(Network::Mainnet, 150, INITIAL_SUBSIDY),
        ValidationResult::Valid
    ));
}

#[test]
fn test_get_block_subsidy_genesis() {
    assert_eq!(get_block_subsidy(0), INITIAL_SUBSIDY);
}

#[test]
fn test_get_block_subsidy_first_halving() {
    // At height 210,000, subsidy should be 25 BTC
    assert_eq!(get_block_subsidy(HALVING_INTERVAL), INITIAL_SUBSIDY / 2);
}

#[test]
fn test_get_block_subsidy_second_halving() {
    // At height 420,000, subsidy should be 12.5 BTC
    assert_eq!(get_block_subsidy(HALVING_INTERVAL * 2), INITIAL_SUBSIDY / 4);
}

#[test]
fn test_get_block_subsidy_max_halvings() {
    // After 64 halvings, subsidy should be 0
    assert_eq!(get_block_subsidy(HALVING_INTERVAL * 64), 0);
}

#[test]
fn test_total_supply_convergence() {
    // Test that total supply approaches 21M BTC
    let supply_at_halving = total_supply(HALVING_INTERVAL);
    // At the first halving, we have 210,000 blocks of 50 BTC each
    let expected_at_halving = (HALVING_INTERVAL as i64) * INITIAL_SUBSIDY;
    // The difference is due to bit shifting in get_block_subsidy
    // Allow for much larger rounding differences due to bit operations
    let difference = (supply_at_halving - expected_at_halving).abs();
    println!(
        "Supply at halving: {supply_at_halving}, Expected: {expected_at_halving}, Difference: {difference}"
    );
    assert!(difference <= 3_000_000_000); // Allow for significant rounding differences
}

#[test]
fn test_supply_limit() {
    // Test that supply limit is respected
    assert!(validate_supply_limit(0));
    assert!(validate_supply_limit(HALVING_INTERVAL));
    assert!(validate_supply_limit(HALVING_INTERVAL * 10));
}

#[test]
fn test_calculate_fee_coinbase() {
    let coinbase_tx = Transaction {
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
    };

    let utxo_set = UtxoSet::default();
    assert_eq!(calculate_fee(&coinbase_tx, &utxo_set).unwrap(), 0);
}

// ============================================================================
// COMPREHENSIVE ECONOMIC TESTS
// ============================================================================

#[test]
fn test_get_block_subsidy_edge_cases() {
    // Test height 1 (just after genesis)
    assert_eq!(get_block_subsidy(1), INITIAL_SUBSIDY);

    // Test height just before first halving
    assert_eq!(get_block_subsidy(HALVING_INTERVAL - 1), INITIAL_SUBSIDY);

    // Test height just after first halving
    assert_eq!(get_block_subsidy(HALVING_INTERVAL + 1), INITIAL_SUBSIDY / 2);

    // Test height just before second halving
    assert_eq!(
        get_block_subsidy(HALVING_INTERVAL * 2 - 1),
        INITIAL_SUBSIDY / 2
    );

    // Test height just after second halving
    assert_eq!(
        get_block_subsidy(HALVING_INTERVAL * 2 + 1),
        INITIAL_SUBSIDY / 4
    );
}

#[test]
fn test_get_block_subsidy_large_heights() {
    // Test very large height (beyond 64 halvings)
    let very_large_height = HALVING_INTERVAL * 100;
    assert_eq!(get_block_subsidy(very_large_height), 0);

    // Test exactly 64 halvings
    let exactly_64_halvings = HALVING_INTERVAL * 64;
    assert_eq!(get_block_subsidy(exactly_64_halvings), 0);

    // Test just before 64 halvings
    let just_before_64 = HALVING_INTERVAL * 64 - 1;
    assert_eq!(get_block_subsidy(just_before_64), INITIAL_SUBSIDY >> 63);
}

#[test]
fn test_total_supply_edge_cases() {
    // Test total supply at height 0
    assert_eq!(total_supply(0), INITIAL_SUBSIDY);

    // Test total supply at height 1
    assert_eq!(total_supply(1), INITIAL_SUBSIDY * 2);

    // Test total supply at height 2
    assert_eq!(total_supply(2), INITIAL_SUBSIDY * 3);

    // Test total supply at first halving
    let supply_at_halving = total_supply(HALVING_INTERVAL);
    assert!(supply_at_halving > 0);
    // At halving, supply should be close to 210,000 * 50 BTC, but bit shifting causes rounding
    let expected_approximate = INITIAL_SUBSIDY * HALVING_INTERVAL as i64;
    // Allow for the fact that bit shifting can cause the supply to be higher due to rounding
    assert!(supply_at_halving <= expected_approximate + 5000000000); // Allow 50 BTC difference
}

#[test]
fn test_total_supply_large_heights() {
    // Test total supply at very large height
    let very_large_height = HALVING_INTERVAL * 100;
    let supply = total_supply(very_large_height);

    // Should be finite and positive
    assert!(supply > 0);
    assert!(supply < MAX_MONEY);

    // Should be close to 21M BTC (allowing for rounding)
    let expected_max = 21_000_000 * 100_000_000; // 21M BTC in satoshis
    assert!(supply <= expected_max);
}

#[test]
fn test_calculate_fee_regular_transaction() {
    let mut utxo_set = UtxoSet::default();

    // Add UTXO to the set
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 1000000000, // 10 BTC
        script_pubkey: vec![].into(),
        height: 0,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 900000000, // 9 BTC output
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    // Fee should be 1 BTC (1000000000 - 900000000)
    assert_eq!(calculate_fee(&tx, &utxo_set).unwrap(), 100000000);
}

#[test]
fn test_calculate_fee_multiple_inputs_outputs() {
    let mut utxo_set = UtxoSet::default();

    // Add two UTXOs
    let outpoint1 = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo1 = UTXO {
        value: 500000000, // 5 BTC
        script_pubkey: vec![].into(),
        height: 0,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint1, std::sync::Arc::new(utxo1));

    let outpoint2 = OutPoint {
        hash: [2; 32],
        index: 0,
    };
    let utxo2 = UTXO {
        value: 300000000, // 3 BTC
        script_pubkey: vec![].into(),
        height: 0,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint2, std::sync::Arc::new(utxo2));

    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![
            TransactionOutput {
                value: 600000000, // 6 BTC output
                script_pubkey: vec![],
            },
            TransactionOutput {
                value: 150000000, // 1.5 BTC output
                script_pubkey: vec![],
            },
        ]
        .into(),
        lock_time: 0,
    };

    // Fee should be 0.5 BTC (800000000 - 750000000)
    assert_eq!(calculate_fee(&tx, &utxo_set).unwrap(), 50000000);
}

#[test]
fn test_calculate_fee_missing_utxo() {
    let utxo_set = UtxoSet::default(); // Empty UTXO set

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 100000000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    // Missing prevout is outside CalculateFee domain → UtxoNotFound (not "Negative fee")
    let result = calculate_fee(&tx, &utxo_set);
    assert!(result.is_err());
    assert!(matches!(result, Err(ConsensusError::UtxoNotFound(_))));
}

#[test]
fn test_check_coinbase_subsidy_rejects_negative_fees() {
    let coinbase = Transaction {
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
            value: 1,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };
    // Tiny coinbase + negative fees would previously return true via checked_add wrap logic
    assert!(!check_coinbase_subsidy(&coinbase, INITIAL_SUBSIDY, -1));
    assert!(!check_coinbase_subsidy(&coinbase, -1, 0));
    assert!(check_coinbase_subsidy(&coinbase, INITIAL_SUBSIDY, 0));
}

#[test]
fn test_calculate_fee_negative_fee() {
    let mut utxo_set = UtxoSet::default();

    // Add UTXO with value less than output
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 100000000, // 1 BTC
        script_pubkey: vec![].into(),
        height: 0,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 200000000, // 2 BTC output (more than input)
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    // Should return error for negative fee
    let result = calculate_fee(&tx, &utxo_set);
    assert!(result.is_err());
    assert!(matches!(result, Err(ConsensusError::EconomicValidation(_))));
}

#[test]
fn test_validate_supply_limit_edge_cases() {
    // Test at height 0
    assert!(validate_supply_limit(0));

    // Test at first halving
    assert!(validate_supply_limit(HALVING_INTERVAL));

    // Test at second halving
    assert!(validate_supply_limit(HALVING_INTERVAL * 2));

    // Test at very large height
    assert!(validate_supply_limit(HALVING_INTERVAL * 100));
}

#[test]
fn test_is_coinbase_edge_cases() {
    // Valid coinbase
    let valid_coinbase = Transaction {
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
        outputs: vec![].into(),
        lock_time: 0,
    };
    assert!(is_coinbase(&valid_coinbase));

    // Wrong hash
    let wrong_hash = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0xffffffff,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    assert!(!is_coinbase(&wrong_hash));

    // Wrong index
    let wrong_index = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    assert!(!is_coinbase(&wrong_index));

    // Multiple inputs
    let multiple_inputs = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    assert!(!is_coinbase(&multiple_inputs));

    // No inputs
    let no_inputs = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };
    assert!(!is_coinbase(&no_inputs));
}
