use super::*;
use crate::constants::BIP147_ACTIVATION_MAINNET;

use crate::opcodes::{OP_0, OP_1, OP_2, OP_CHECKMULTISIG, OP_CHECKSIG};

#[test]
fn bip34_height_1_through_16_is_a_single_opcode() {
    // Testnet4 block 1 coinbase scriptSig (mempool.space): OP_1, OP_0, then a 6-byte push.
    let block1: &[u8] = &[0x51, 0x00, 0x06, 0x2f, 0x40, 0x77, 0x69, 0x7a, 0x2f];
    assert!(script_sig_starts_with_height(block1, 1));
    assert!(!script_sig_starts_with_height(&[0x01, 0x01], 1));
    assert!(script_sig_starts_with_height(&[0x52, 0x00], 2));
    assert!(script_sig_starts_with_height(&[0x60], 16));
    // Height 17 is a one-byte push, not OP_17.
    assert!(script_sig_starts_with_height(&[0x01, 0x11, 0xff], 17));
    assert!(!script_sig_starts_with_height(&[0x51], 17));
    assert!(script_sig_starts_with_height(&[0x00], 0));
    assert_eq!(OP_1, 0x51);
}

#[test]
fn encode_bip34_coinbase_script_satisfies_the_prefix_check() {
    for height in [0u64, 1, 16, 17, 127, 128, 150, 255, 256, 210_000] {
        let script = encode_bip34_coinbase_script(height);
        assert!(
            script_sig_starts_with_height(&script, height),
            "height {height} script {script:?}"
        );
        assert!(
            (2..=100).contains(&script.len()),
            "height {height} len {}",
            script.len()
        );
    }
    assert_eq!(encode_bip34_coinbase_script(0), vec![0x00, 0xff]);
    assert_eq!(encode_bip34_coinbase_script(1), vec![0x51, 0xff]);
    assert_eq!(encode_bip34_coinbase_script(16), vec![0x60, 0xff]);
    assert_eq!(encode_bip34_coinbase_script(17), vec![0x01, 0x11]);
    assert_eq!(encode_bip34_coinbase_script(127), vec![0x01, 0x7f]);
    assert_eq!(encode_bip34_coinbase_script(128), vec![0x02, 0x80, 0x00]);
    assert_eq!(encode_bip34_coinbase_script(150), vec![0x02, 0x96, 0x00]);
    assert_eq!(encode_bip34_coinbase_script(255), vec![0x02, 0xff, 0x00]);
    assert_eq!(encode_bip34_coinbase_script(256), vec![0x02, 0x00, 0x01]);
    assert_eq!(
        encode_bip34_coinbase_script(210_000),
        vec![0x03, 0x50, 0x34, 0x03]
    );
}

#[test]
fn test_bip30_repeat_hashes_are_display_ids_reversed() {
    fn digest(display: &str) -> Hash {
        let mut bytes = hex::decode(display).unwrap();
        bytes.reverse();
        bytes.try_into().unwrap()
    }
    assert!(is_bip30_repeat_block(
        91_842,
        &digest("00000000000a4d0a398161ffc163c503763b1f4360639393e0e4c8e300e0caec")
    ));
    assert!(is_bip30_repeat_block(
        91_880,
        &digest("00000000000743f190a18c5577a3c2d2a1f610ae9601ac046a38084ccb7cd721")
    ));
    assert!(!is_bip30_repeat_block(91_842, &[0; 32]));
    assert!(!is_bip30_repeat_block(
        91_722,
        &digest("00000000000a4d0a398161ffc163c503763b1f4360639393e0e4c8e300e0caec")
    ));
    assert!(is_known_bip34_block(
        Network::Mainnet,
        crate::constants::BIP34_ACTIVATION_MAINNET,
        &digest("000000000000024b89b42a942fe0d9fea3bb44ab7bd1b19115dd6a759c0808b8")
    ));
    assert!(!is_known_bip34_block(
        Network::Mainnet,
        crate::constants::BIP34_ACTIVATION_MAINNET,
        &[0; 32]
    ));
}

#[test]
fn test_bip30_basic() {
    // Test that BIP30 check passes for new coinbase
    let transactions: Vec<Transaction> = vec![Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    }];
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into_boxed_slice(),
    };

    let utxo_set = UtxoSet::default();
    let result = check_bip30_network(
        &block,
        &utxo_set,
        None,
        0,
        crate::types::Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(result, "BIP30 should pass for new coinbase");
}

#[test]
fn test_bip34_before_activation() {
    let transactions: Vec<Transaction> = vec![Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    }];
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into_boxed_slice(),
    };

    // Before activation, BIP34 should pass
    let result = check_bip34_network(&block, 100_000, crate::types::Network::Mainnet).unwrap();
    assert!(result, "BIP34 should pass before activation");
}

#[test]
fn test_bip34_after_activation() {
    let height = crate::constants::BIP34_ACTIVATION_MAINNET;
    let transactions: Vec<Transaction> = vec![Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            // Height encoded as CScriptNum: 0x03 (push 3 bytes) + height in little-endian
            script_sig: vec![
                0x03,
                (height & 0xff) as u8,
                ((height >> 8) & 0xff) as u8,
                ((height >> 16) & 0xff) as u8,
            ],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    }];
    let block = Block {
        header: BlockHeader {
            version: 2, // BIP34 requires version >= 2
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into_boxed_slice(),
    };

    let result = check_bip34_network(&block, height, crate::types::Network::Mainnet).unwrap();
    assert!(result, "BIP34 should pass with correct height encoding");
}

#[test]
fn test_bip90_version_enforcement() {
    // Test version 1 before BIP34 activation
    let result = check_bip90_network(1, 100_000, crate::types::Network::Mainnet).unwrap();
    assert!(result, "Version 1 should be valid before BIP34");

    // Test version 1 after BIP34 activation (should fail)
    let result = check_bip90_network(
        1,
        crate::constants::BIP34_ACTIVATION_MAINNET,
        crate::types::Network::Mainnet,
    )
    .unwrap();
    assert!(
        !result,
        "Version 1 should be invalid after BIP34 activation"
    );

    // Test version 2 after BIP34 activation (should pass)
    let result = check_bip90_network(
        2,
        crate::constants::BIP34_ACTIVATION_MAINNET,
        crate::types::Network::Mainnet,
    )
    .unwrap();
    assert!(result, "Version 2 should be valid after BIP34 activation");

    // Test version 2 after BIP66 activation (should fail)
    // BIP66 activates at block 363,725, so we test at that height
    let result = check_bip90_network(2, 363_725, crate::types::Network::Mainnet).unwrap();
    assert!(
        !result,
        "Version 2 should be invalid after BIP66 activation"
    );

    // Test version 3 after BIP66 activation (should pass)
    // BIP66 activates at block 363,725, so we test at that height
    let result = check_bip90_network(3, 363_725, crate::types::Network::Mainnet).unwrap();
    assert!(result, "Version 3 should be valid after BIP66 activation");
}

#[test]
fn test_bip30_duplicate_coinbase() {
    use crate::block::calculate_tx_id;

    // Create a coinbase transaction
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    let txid = calculate_tx_id(&coinbase_tx);

    // Create UTXO set with a UTXO from this coinbase
    let mut utxo_set = UtxoSet::default();
    utxo_set.insert(
        OutPoint {
            hash: txid,
            index: 0,
        },
        std::sync::Arc::new(UTXO {
            value: 50_0000_0000,
            script_pubkey: vec![].into(),
            height: 0,
            is_coinbase: false,
        }),
    );

    // Create block with same coinbase (duplicate)
    let transactions: Vec<Transaction> = vec![coinbase_tx];
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into_boxed_slice(),
    };

    // BIP30 should fail for duplicate coinbase
    let result = check_bip30_network(
        &block,
        &utxo_set,
        None,
        0,
        crate::types::Network::Mainnet,
        None,
    )
    .unwrap();
    assert!(!result, "BIP30 should fail for duplicate coinbase");
}

#[test]
fn test_bip34_invalid_height() {
    let height = crate::constants::BIP34_ACTIVATION_MAINNET;
    let transactions: Vec<Transaction> = vec![Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            // Wrong height encoding
            script_sig: vec![0x03, 0x00, 0x00, 0x00], // Height 0 instead of activation height
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    }];
    let block = Block {
        header: BlockHeader {
            version: 2,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into_boxed_slice(),
    };

    // BIP34 should fail with wrong height
    let result = check_bip34_network(&block, height, crate::types::Network::Mainnet).unwrap();
    assert!(!result, "BIP34 should fail with incorrect height encoding");
}

#[test]
fn test_bip66_strict_der() {
    use crate::constants::BIP66_ACTIVATION_MAINNET;
    use crate::types::Network;

    // 30 06 02 01 01 02 01 01 01 — tag, length 6, R = 1, S = 1, sighash 01.
    let accept: Vec<u8> = vec![0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01];
    assert_eq!(
        check_bip66_network(&accept, BIP66_ACTIVATION_MAINNET, Network::Mainnet).unwrap(),
        true,
        "9-byte strict DER must pass at activation"
    );

    let mut high_bit_r = accept.clone();
    high_bit_r[4] = 0x81;
    // R length 2, leading 0x00, next byte high bit clear.
    let leading_zero_r: Vec<u8> = vec![0x30, 0x07, 0x02, 0x02, 0x00, 0x01, 0x02, 0x01, 0x01, 0x01];
    let mut bad_length = accept.clone();
    bad_length[1] = 0x05;
    let rejects = [
        vec![0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01],
        vec![0u8; 74],
        high_bit_r.clone(),
        leading_zero_r,
        bad_length,
    ];
    for sig in &rejects {
        assert_eq!(
            check_bip66_network(sig, BIP66_ACTIVATION_MAINNET, Network::Mainnet).unwrap(),
            false,
            "non-strict encoding must fail at activation: {sig:?}"
        );
    }

    assert_eq!(
        check_bip66_network(&high_bit_r, BIP66_ACTIVATION_MAINNET - 1, Network::Mainnet).unwrap(),
        true,
        "pre-activation must accept a non-strict signature"
    );
}

#[test]
fn test_bip147_skips_checkmultisig_byte_in_push_data() {
    // Push data contains 0xae; executable path is OP_CHECKSIG only.
    let script_pubkey = vec![0x01, OP_CHECKMULTISIG, OP_CHECKSIG];
    let script_sig = vec![1, OP_1];
    let result = check_bip147_network(
        &script_sig,
        &script_pubkey,
        BIP147_ACTIVATION_MAINNET,
        Bip147Network::Mainnet,
    )
    .unwrap();
    assert!(
        result,
        "BIP147 must not apply when CHECKMULTISIG is only in push data"
    );
}

#[test]
fn test_bip147_null_dummy() {
    // Executable OP_CHECKMULTISIG (matches bip_validation_suite fixture).
    let script_pubkey = vec![OP_CHECKMULTISIG];
    let script_sig_valid = vec![OP_0];
    let result = check_bip147_network(
        &script_sig_valid,
        &script_pubkey,
        BIP147_ACTIVATION_MAINNET,
        Bip147Network::Mainnet,
    )
    .unwrap();
    assert!(result, "BIP147 should pass with NULLDUMMY");

    let script_sig_invalid = vec![1, OP_1];
    let result = check_bip147_network(
        &script_sig_invalid,
        &script_pubkey,
        BIP147_ACTIVATION_MAINNET,
        Bip147Network::Mainnet,
    )
    .unwrap();
    assert!(!result, "BIP147 should fail without NULLDUMMY");

    // Before activation, should always pass
    let result = check_bip147_network(
        &script_sig_invalid,
        &script_pubkey,
        100_000,
        Bip147Network::Mainnet,
    )
    .unwrap();
    assert!(result, "BIP147 should pass before activation");
}
