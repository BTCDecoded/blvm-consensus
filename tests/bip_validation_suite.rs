//! COV-C-06b: BIP validation helpers exercised via direct API calls.

use blvm_consensus::activation::IsForkActive;
use blvm_consensus::bip_validation::{
    build_bip30_index, check_bip147, check_bip147_network, check_bip30, check_bip30_network,
    check_bip34, check_bip34_network, check_bip54_coinbase, check_bip66, check_bip66_network,
    check_bip90, check_bip90_network, is_bip54_active, is_bip54_active_at, Bip147Network,
};
use blvm_consensus::block::{calculate_tx_id, BlockValidationContext};
use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::opcodes::{OP_0, OP_1, OP_CHECKSIG};
use blvm_consensus::types::{ForkId, Network};
use blvm_consensus::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet,
    BIP147_ACTIVATION_TESTNET, BIP30_DEACTIVATION_MAINNET, BIP34_ACTIVATION_MAINNET,
    BIP54_ACTIVATION_MAINNET, BIP65_ACTIVATION_MAINNET, BIP66_ACTIVATION_MAINNET,
    SEGWIT_ACTIVATION_MAINNET, UTXO,
};
use std::sync::Arc;

fn ctx() -> BlockValidationContext {
    BlockValidationContext::for_network(Network::Mainnet)
}

fn coinbase_at_height(height: u64) -> Transaction {
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
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: script_sig.into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

fn block_with_coinbase(coinbase: Transaction) -> Block {
    block_with_transactions(vec![coinbase])
}

fn block_with_transactions(transactions: Vec<Transaction>) -> Block {
    let merkle_root = calculate_merkle_root(&transactions).unwrap();
    Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp: 1_500_353_985,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into(),
    }
}

fn coinbase_with_pushdata2_height(height: u64) -> Transaction {
    let mut height_bytes = Vec::new();
    let mut n = height;
    while n > 0 {
        height_bytes.push((n & 0xff) as u8);
        n >>= 8;
    }
    if height_bytes.last().is_some_and(|&b| b & 0x80 != 0) {
        height_bytes.push(0x00);
    }
    let len = height_bytes.len();
    let mut script_sig = vec![0x4d, (len & 0xff) as u8, ((len >> 8) & 0xff) as u8];
    script_sig.extend_from_slice(&height_bytes);
    if script_sig.len() < 2 {
        script_sig.push(0xff);
    }
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: script_sig.into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

#[test]
fn test_check_bip34_valid_at_activation() {
    let height = BIP34_ACTIVATION_MAINNET;
    let block = block_with_coinbase(coinbase_at_height(height));
    assert!(check_bip34(&block, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip34_rejects_missing_height() {
    let height = BIP34_ACTIVATION_MAINNET;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs[0].script_sig = vec![OP_1, OP_1].into();
    let block = block_with_coinbase(coinbase);
    let result = check_bip34(&block, height, &ctx());
    match result {
        Ok(false) | Err(_) => {}
        Ok(true) => panic!("BIP34 violation must not pass"),
    }
}

#[test]
fn test_check_bip90_rejects_version_one_at_bip34() {
    let height = BIP34_ACTIVATION_MAINNET;
    let mut block = block_with_coinbase(coinbase_at_height(height));
    block.header.version = 1;
    assert!(!check_bip90(block.header.version, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip54_coinbase_compliant() {
    let height = 500u64;
    let mut coinbase = coinbase_at_height(height);
    coinbase.lock_time = height.saturating_sub(13);
    coinbase.inputs[0].sequence = 0xfffffffe;
    assert!(check_bip54_coinbase(&coinbase, height));
}

#[test]
fn test_check_bip54_coinbase_rejects_wrong_locktime() {
    let height = 500u64;
    let coinbase = coinbase_at_height(height);
    assert!(!check_bip54_coinbase(&coinbase, height));
}

#[test]
fn test_fork_id_bip34_active_after_activation() {
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    assert!(ctx.is_fork_active(ForkId::Bip34, BIP34_ACTIVATION_MAINNET));
    assert!(!ctx.is_fork_active(ForkId::Bip34, BIP34_ACTIVATION_MAINNET - 1));
}

#[test]
fn test_check_bip66_rejects_non_der_before_activation_skipped() {
    let bad_sig = vec![OP_0; 10];
    let height = BIP66_ACTIVATION_MAINNET - 1;
    assert!(check_bip66(&bad_sig, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip66_rejects_non_der_at_activation() {
    let bad_sig = vec![OP_0; 10];
    let height = BIP66_ACTIVATION_MAINNET;
    assert!(!check_bip66(&bad_sig, height, &ctx()).unwrap());
}

#[test]
fn test_is_bip54_active_with_override() {
    let activation = 100u64;
    assert!(!is_bip54_active_at(99, Network::Mainnet, Some(activation)));
    assert!(is_bip54_active_at(100, Network::Mainnet, Some(activation)));
}

#[test]
fn test_build_bip30_index_counts_coinbase_utxos() {
    let mut set = UtxoSet::default();
    let coinbase_id = [0xcc; 32];
    set.insert(
        OutPoint {
            hash: coinbase_id,
            index: 0,
        },
        Arc::new(UTXO {
            value: 50_000_000_000,
            script_pubkey: vec![OP_1].into(),
            height: 1,
            is_coinbase: true,
        }),
    );
    let index = build_bip30_index(&set);
    assert_eq!(index.get(&coinbase_id), Some(&1));
}

#[test]
fn test_check_bip30_rejects_duplicate_coinbase_txid_in_utxo() {
    let height = 50_000u64;
    assert!(height <= BIP30_DEACTIVATION_MAINNET);
    let coinbase = coinbase_at_height(height);
    let txid = calculate_tx_id(&coinbase);
    let block = block_with_coinbase(coinbase);

    let mut set = UtxoSet::default();
    set.insert(
        OutPoint {
            hash: txid,
            index: 0,
        },
        Arc::new(UTXO {
            value: 1,
            script_pubkey: vec![OP_1].into(),
            height: 1,
            is_coinbase: true,
        }),
    );

    let index = build_bip30_index(&set);
    assert!(!check_bip30(&block, &set, Some(&index), height, &ctx(), Some(&txid)).unwrap());
}

#[test]
fn test_check_bip30_network_wrapper() {
    let height = 50_000u64;
    let block = block_with_coinbase(coinbase_at_height(height));
    assert!(check_bip30_network(
        &block,
        &UtxoSet::default(),
        None,
        height,
        Network::Mainnet,
        None
    )
    .unwrap());
}

#[test]
fn test_check_bip34_network_wrapper() {
    let height = BIP34_ACTIVATION_MAINNET;
    let block = block_with_coinbase(coinbase_at_height(height));
    assert!(check_bip34_network(&block, height, Network::Mainnet).unwrap());
}

#[test]
fn test_check_bip90_network_accepts_version_four() {
    assert!(check_bip90_network(4, BIP66_ACTIVATION_MAINNET, Network::Mainnet).unwrap());
}

#[test]
fn test_check_bip147_null_dummy_before_activation_skips() {
    let script_pubkey = vec![0xae]; // OP_CHECKMULTISIG
    let script_sig = vec![0x01, 0x51]; // non-null dummy
    let height = SEGWIT_ACTIVATION_MAINNET - 1;
    assert!(check_bip147(&script_sig, &script_pubkey, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip147_requires_null_dummy_at_activation() {
    let script_pubkey = vec![0xae];
    let good_sig = vec![0x01, 0x00];
    let bad_sig = vec![0x01, 0x51];
    let height = SEGWIT_ACTIVATION_MAINNET;
    assert!(check_bip147(&good_sig, &script_pubkey, height, &ctx()).unwrap());
    assert!(!check_bip147(&bad_sig, &script_pubkey, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip66_network_rejects_non_der_at_activation() {
    let bad_sig = vec![OP_0; 10];
    let height = BIP66_ACTIVATION_MAINNET;
    assert!(!check_bip66_network(&bad_sig, height, Network::Mainnet).unwrap());
}

#[test]
fn test_check_bip147_network_requires_null_dummy_at_activation() {
    let script_pubkey = vec![0xae];
    let bad_sig = vec![0x01, 0x51];
    let height = SEGWIT_ACTIVATION_MAINNET;
    assert!(
        !check_bip147_network(&bad_sig, &script_pubkey, height, Bip147Network::Mainnet).unwrap()
    );
}

#[test]
fn test_is_bip54_active_mainnet_activation_height() {
    assert!(!is_bip54_active(
        BIP54_ACTIVATION_MAINNET - 1,
        Network::Mainnet
    ));
    assert!(is_bip54_active(BIP54_ACTIVATION_MAINNET, Network::Mainnet));
}

#[test]
fn test_check_bip147_skips_non_multisig_at_activation() {
    let height = SEGWIT_ACTIVATION_MAINNET;
    let script_pubkey = vec![OP_1, OP_CHECKSIG];
    let script_sig = vec![0x01, 0x51];
    assert!(check_bip147(&script_sig, &script_pubkey, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip54_coinbase_rejects_final_sequence() {
    let height = 500u64;
    let mut coinbase = coinbase_at_height(height);
    coinbase.lock_time = height.saturating_sub(13);
    coinbase.inputs[0].sequence = 0xffffffff;
    assert!(!check_bip54_coinbase(&coinbase, height));
}

#[test]
fn test_check_bip54_coinbase_rejects_empty_inputs() {
    let height = 500u64;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs = vec![].into();
    assert!(!check_bip54_coinbase(&coinbase, height));
}

#[test]
fn test_check_bip34_skips_before_activation() {
    let height = BIP34_ACTIVATION_MAINNET - 1;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs[0].script_sig = vec![OP_1].into();
    let block = block_with_coinbase(coinbase);
    assert!(check_bip34(&block, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip90_rejects_version_two_at_bip66_activation() {
    assert!(!check_bip90(2, BIP66_ACTIVATION_MAINNET, &ctx()).unwrap());
}

#[test]
fn test_check_bip90_rejects_version_three_at_bip65_activation() {
    assert!(!check_bip90(3, BIP65_ACTIVATION_MAINNET, &ctx()).unwrap());
    assert!(check_bip90(4, BIP65_ACTIVATION_MAINNET, &ctx()).unwrap());
}

#[test]
fn test_check_bip34_rejects_wrong_height_at_activation() {
    let height = BIP34_ACTIVATION_MAINNET;
    let block = block_with_coinbase(coinbase_at_height(height + 1));
    assert!(!check_bip34(&block, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip34_errors_on_truncated_pushdata() {
    let height = BIP34_ACTIVATION_MAINNET;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs[0].script_sig = vec![0x4c, 0x05].into();
    let block = block_with_coinbase(coinbase);
    assert!(check_bip34(&block, height, &ctx()).is_err());
}

#[test]
fn test_check_bip34_network_rejects_wrong_height_at_activation() {
    let height = BIP34_ACTIVATION_MAINNET;
    let block = block_with_coinbase(coinbase_at_height(height + 1));
    assert!(!check_bip34_network(&block, height, Network::Mainnet).unwrap());
}

#[test]
fn test_check_bip66_network_skips_before_activation() {
    let bad_sig = vec![OP_0; 10];
    let height = BIP66_ACTIVATION_MAINNET - 1;
    assert!(check_bip66_network(&bad_sig, height, Network::Mainnet).unwrap());
}

#[test]
fn test_check_bip90_network_rejects_version_three_at_bip65() {
    assert!(!check_bip90_network(3, BIP65_ACTIVATION_MAINNET, Network::Mainnet).unwrap());
    assert!(check_bip90_network(4, BIP65_ACTIVATION_MAINNET, Network::Mainnet).unwrap());
}

#[test]
fn test_check_bip34_accepts_pushdata2_height_encoding() {
    let height = BIP34_ACTIVATION_MAINNET;
    let block = block_with_coinbase(coinbase_with_pushdata2_height(height));
    assert!(check_bip34(&block, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip34_accepts_height_zero_on_regtest() {
    let ctx = BlockValidationContext::for_network(Network::Regtest);
    let block = block_with_coinbase(coinbase_at_height(0));
    assert!(check_bip34(&block, 0, &ctx).unwrap());
}

#[test]
fn test_check_bip34_skips_when_first_tx_not_coinbase() {
    let height = BIP34_ACTIVATION_MAINNET;
    let spend = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x55; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let block = block_with_transactions(vec![spend, coinbase_at_height(height)]);
    assert!(check_bip34(&block, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip34_rejects_empty_script_sig_at_activation() {
    let height = BIP34_ACTIVATION_MAINNET;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs[0].script_sig = vec![].into();
    let block = block_with_coinbase(coinbase);
    assert!(!check_bip34(&block, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip34_errors_on_invalid_height_encoding() {
    let height = BIP34_ACTIVATION_MAINNET;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs[0].script_sig = vec![0x4e, 0x01, 0x00, 0x00, 0x01].into();
    let block = block_with_coinbase(coinbase);
    assert!(check_bip34(&block, height, &ctx()).is_err());
}

#[test]
fn test_check_bip34_errors_on_height_value_too_large() {
    let height = BIP34_ACTIVATION_MAINNET;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs[0].script_sig = vec![0x09, 0, 0, 0, 0, 0, 0, 0, 0, 0].into();
    let block = block_with_coinbase(coinbase);
    assert!(check_bip34(&block, height, &ctx()).is_err());
}

#[test]
fn test_check_bip30_skips_non_coinbase_transaction() {
    let spend = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x66; 32],
                index: 0,
            },
            script_sig: vec![OP_1].into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let block = block_with_transactions(vec![spend]);
    assert!(check_bip30(&block, &UtxoSet::default(), None, 100, &ctx(), None).unwrap());
}

#[test]
fn test_check_bip147_network_requires_null_dummy_on_testnet() {
    let script_pubkey = vec![0xae];
    let bad_sig = vec![0x01, 0x51];
    let height = BIP147_ACTIVATION_TESTNET;
    assert!(
        !check_bip147_network(&bad_sig, &script_pubkey, height, Bip147Network::Testnet).unwrap()
    );
}

#[test]
fn test_check_bip34_errors_on_truncated_direct_push() {
    let height = BIP34_ACTIVATION_MAINNET;
    let mut coinbase = coinbase_at_height(height);
    coinbase.inputs[0].script_sig = vec![0x03, 0x01, 0x02].into();
    let block = block_with_coinbase(coinbase);
    assert!(check_bip34(&block, height, &ctx()).is_err());
}

#[test]
fn test_check_bip30_index_fast_path_accepts_new_coinbase() {
    let height = 50_000u64;
    let block = block_with_coinbase(coinbase_at_height(height));
    let index = build_bip30_index(&UtxoSet::default());
    assert!(check_bip30(
        &block,
        &UtxoSet::default(),
        Some(&index),
        height,
        &ctx(),
        None
    )
    .unwrap());
}

#[test]
fn test_check_bip147_rejects_empty_script_sig_at_activation() {
    let script_pubkey = vec![0xae];
    let height = SEGWIT_ACTIVATION_MAINNET;
    assert!(!check_bip147(&[], &script_pubkey, height, &ctx()).unwrap());
}

#[test]
fn test_check_bip147_network_requires_null_dummy_on_regtest() {
    let script_pubkey = vec![0xae];
    let bad_sig = vec![0x01, 0x51];
    assert!(!check_bip147_network(&bad_sig, &script_pubkey, 100, Bip147Network::Regtest).unwrap());
}

#[test]
fn test_is_bip54_active_at_uses_network_constant_without_override() {
    assert!(!is_bip54_active_at(
        BIP54_ACTIVATION_MAINNET - 1,
        Network::Mainnet,
        None
    ));
    assert!(is_bip54_active_at(
        BIP54_ACTIVATION_MAINNET,
        Network::Mainnet,
        None
    ));
}
