use super::*;
use crate::opcodes::*;

#[test]
fn test_sighash_type_parsing() {
    // Standard types
    assert_eq!(SighashType::from_byte(0x01), SighashType::ALL);
    assert_eq!(SighashType::from_byte(0x02), SighashType::NONE);
    assert_eq!(SighashType::from_byte(0x03), SighashType::SINGLE);
    assert_eq!(SighashType::from_byte(0x00), SighashType::ALL_LEGACY);
    assert_eq!(SighashType::from_byte(0x81), SighashType::ALL_ANYONECANPAY);
    assert_eq!(SighashType::from_byte(0x82), SighashType::NONE_ANYONECANPAY);
    assert_eq!(
        SighashType::from_byte(0x83),
        SighashType::SINGLE_ANYONECANPAY
    );
    // Verify the byte values are preserved correctly for sighash preimage
    assert_eq!(SighashType::ALL_ANYONECANPAY.as_u32(), 0x81);
    assert_eq!(SighashType::NONE_ANYONECANPAY.as_u32(), 0x82);
    assert_eq!(SighashType::SINGLE_ANYONECANPAY.as_u32(), 0x83);
    // Every byte is accepted. The raw byte is what the preimage stores.
    let st = SighashType::from_byte(0x04);
    assert!(st.is_all()); // base_type 0x04 acts as ALL
    assert_eq!(st.as_u32(), 0x04); // raw byte preserved in preimage
    let st84 = SighashType::from_byte(0x84);
    assert!(st84.is_all());
    assert!(st84.is_anyonecanpay());
    assert_eq!(st84.as_u32(), 0x84);
}

#[test]
fn legacy_sighash_cache_is_bound_to_the_transaction() {
    use crate::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    let prevout = OutPoint {
        hash: [9u8; 32],
        index: 0,
    };
    let script = [0x51u8];
    let tx_paying = |value: i64| Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            sequence: 0xffff_ffff,
            script_sig: vec![0x01],
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };
    let sighash = |value: i64| {
        calculate_transaction_sighash_single_input(
            &tx_paying(value),
            0,
            &script,
            value,
            SighashType::ALL,
            #[cfg(feature = "production")]
            None,
        )
        .unwrap()
    };

    let first = sighash(1_000);
    for value in [2_000i64, 3, 50_000_000, i64::MAX / 4] {
        assert_ne!(first, sighash(value), "output {value}");
    }
    // Same transaction again, after the other spends filled the cache.
    assert_eq!(first, sighash(1_000));

    let mut later_locktime = tx_paying(1_000);
    later_locktime.lock_time = 50;
    let locktime_hash = calculate_transaction_sighash_single_input(
        &later_locktime,
        0,
        &script,
        1_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .unwrap();
    assert_ne!(first, locktime_hash);

    let mut extra_input = tx_paying(1_000);
    extra_input.inputs.push(TransactionInput {
        prevout: OutPoint {
            hash: [8u8; 32],
            index: 1,
        },
        sequence: 0xffff_ffff,
        script_sig: vec![],
    });
    let extra_hash = calculate_transaction_sighash_single_input(
        &extra_input,
        0,
        &script,
        1_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .unwrap();
    assert_ne!(first, extra_hash);

    // scriptSig is blanked for every input except the signing script code.
    let mut other_script_sig = tx_paying(1_000);
    other_script_sig.inputs[0].script_sig = vec![0x02, 0x03];
    let script_sig_hash = calculate_transaction_sighash_single_input(
        &other_script_sig,
        0,
        &script,
        1_000,
        SighashType::ALL,
        #[cfg(feature = "production")]
        None,
    )
    .unwrap();
    assert_eq!(first, script_sig_hash);
}

/// Two different 16-byte strings with this tail share an `FxHasher` digest.
/// The sighash cache must still return different digests.
fn fxhash_colliding_tail() -> [u8; 8] {
    0x1319_8a2e_0370_7344u64.to_le_bytes()
}

fn fxhash_colliding_scripts() -> ([u8; 16], [u8; 16]) {
    let tail = fxhash_colliding_tail();
    let mut first = [0u8; 16];
    let mut second = [0u8; 16];
    let mut head_a = [0u8; 8];
    head_a[0] = OP_1;
    head_a[1] = OP_CHECKSIG;
    head_a[3] = 1;
    head_a[4] = 2;
    head_a[5] = 3;
    head_a[6] = 4;
    head_a[7] = 5;
    let mut head_b = [0u8; 8];
    head_b[0] = OP_2;
    head_b[1] = OP_CHECKSIG;
    head_b[2] = 9;
    head_b[3] = 8;
    head_b[4] = 7;
    head_b[5] = 6;
    head_b[6] = 5;
    head_b[7] = 4;
    first[..8].copy_from_slice(&head_a);
    second[..8].copy_from_slice(&head_b);
    first[8..].copy_from_slice(&tail);
    second[8..].copy_from_slice(&tail);
    (first, second)
}

#[test]
fn legacy_sighash_cache_does_not_reuse_an_fxhash_script_collision() {
    use crate::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    let (script_a, script_b) = fxhash_colliding_scripts();
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [9u8; 32],
                index: 0,
            },
            sequence: 0xffff_ffff,
            script_sig: vec![],
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    };
    let sighash = |script: &[u8]| {
        calculate_transaction_sighash_single_input(
            &tx,
            0,
            script,
            1_000,
            SighashType::ALL,
            #[cfg(feature = "production")]
            None,
        )
        .unwrap()
    };
    let first = sighash(&script_a);
    let other = sighash(&script_b);
    assert_ne!(first, other);
    assert_eq!(first, sighash(&script_a));
}

#[test]
fn legacy_sighash_cache_does_not_reuse_an_fxhash_output_collision() {
    use crate::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    let (script_a, script_b) = fxhash_colliding_scripts();
    let signing = [OP_1, OP_CHECKSIG];
    let tx_paying = |spk: &[u8]| Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [9u8; 32],
                index: 0,
            },
            sequence: 0xffff_ffff,
            script_sig: vec![],
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: spk.to_vec(),
        }]
        .into(),
        lock_time: 0,
    };
    let sighash = |spk: &[u8]| {
        calculate_transaction_sighash_single_input(
            &tx_paying(spk),
            0,
            &signing,
            1_000,
            SighashType::ALL,
            #[cfg(feature = "production")]
            None,
        )
        .unwrap()
    };
    let first = sighash(&script_a);
    assert_ne!(first, sighash(&script_b));
    assert_eq!(first, sighash(&script_a));
}

#[test]
fn test_varint_encoding() {
    use crate::serialization::varint::encode_varint;
    assert_eq!(encode_varint(0), vec![0]);
    assert_eq!(encode_varint(252), vec![252]);
    assert_eq!(encode_varint(253), vec![0xfd, 253, 0]);
    assert_eq!(encode_varint(65535), vec![0xfd, 255, 255]);
    assert_eq!(encode_varint(65536), vec![0xfe, 0, 0, 1, 0]);
}

#[test]
fn test_sighash_calculation() {
    // Create a simple transaction for testing
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32],
                index: 0,
            },
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![
                OP_DUP,
                OP_HASH160,
                PUSH_20_BYTES,
                0x89,
                0xab,
                0xcd,
                0xef,
                0x12,
                0x34,
                0x56,
                0x78,
                0x9a,
                0xbc,
                0xde,
                0xf0,
                0x12,
                0x34,
                0x56,
                0x78,
                0x9a,
                OP_EQUALVERIFY,
                OP_CHECKSIG,
            ], // P2PKH
        }]
        .into(),
        lock_time: 0,
    };

    let prevouts = vec![TransactionOutput {
        value: 10000000000,
        script_pubkey: vec![
            OP_DUP,
            OP_HASH160,
            PUSH_20_BYTES,
            0x89,
            0xab,
            0xcd,
            0xef,
            0x12,
            0x34,
            0x56,
            0x78,
            0x9a,
            0xbc,
            0xde,
            0xf0,
            0x12,
            0x34,
            0x56,
            0x78,
            0x9a,
            OP_EQUALVERIFY,
            OP_CHECKSIG,
        ],
    }];

    // Test SIGHASH_ALL
    let sighash = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL).unwrap();
    assert_eq!(sighash.len(), 32);

    // Test SIGHASH_NONE
    let sighash_none = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::NONE).unwrap();
    assert_ne!(sighash, sighash_none);

    // Test SIGHASH_SINGLE
    let sighash_single =
        calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::SINGLE).unwrap();
    assert_ne!(sighash, sighash_single);
}

#[test]
fn test_sighash_invalid_input_index() {
    let tx = Transaction {
        version: 1,
        inputs: vec![].into(),
        outputs: vec![].into(),
        lock_time: 0,
    };

    let result = calculate_transaction_sighash(&tx, 0, &[], SighashType::ALL);
    assert!(result.is_err());
}

#[test]
fn test_bip143_sighash() {
    // Create a SegWit transaction for testing
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1u8; 32],
                    index: 0,
                },
                script_sig: vec![], // Empty for SegWit
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2u8; 32],
                    index: 1,
                },
                script_sig: vec![],
                sequence: 0xfffffffe,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![
                OP_0,
                PUSH_20_BYTES,
                0x89,
                0xab,
                0xcd,
                0xef,
                0x12,
                0x34,
                0x56,
                0x78,
                0x9a,
                0xbc,
                0xde,
                0xf0,
                0x12,
                0x34,
                0x56,
                0x78,
                0x9a,
                0xbc,
                0xde,
                0xf0,
            ],
        }]
        .into(),
        lock_time: 0,
    };

    let prevouts = [
        TransactionOutput {
            value: 10000000000,
            script_pubkey: vec![
                OP_0,
                PUSH_20_BYTES,
                0x11,
                0x22,
                0x33,
                0x44,
                0x55,
                0x66,
                0x77,
                0x88,
                0x99,
                0xaa,
                0xbb,
                0xcc,
                0xdd,
                0xee,
                0xff,
                0x00,
                0x11,
                0x22,
                0x33,
                0x44,
            ],
        },
        TransactionOutput {
            value: 8000000000,
            script_pubkey: vec![
                OP_0,
                PUSH_20_BYTES,
                0xaa,
                0xbb,
                0xcc,
                0xdd,
                0xee,
                0xff,
                0x00,
                0x11,
                0x22,
                0x33,
                0x44,
                0x55,
                0x66,
                0x77,
                0x88,
                0x99,
                0xaa,
                0xbb,
                0xcc,
                0xdd,
            ],
        },
    ];

    // P2WPKH scriptCode is OP_DUP OP_HASH160 <20-byte-hash> OP_EQUALVERIFY OP_CHECKSIG
    let script_code = vec![
        OP_DUP,
        OP_HASH160,
        PUSH_20_BYTES,
        0x11,
        0x22,
        0x33,
        0x44,
        0x55,
        0x66,
        0x77,
        0x88,
        0x99,
        0xaa,
        0xbb,
        0xcc,
        0xdd,
        0xee,
        0xff,
        0x00,
        0x11,
        0x22,
        0x33,
        OP_EQUALVERIFY,
        OP_CHECKSIG,
    ];

    // Test BIP143 sighash for first input
    let sighash0 =
        calculate_bip143_sighash(&tx, 0, &script_code, prevouts[0].value, 0x01, None).unwrap();
    assert_eq!(sighash0.len(), 32);

    // Test BIP143 sighash for second input (should be different)
    let sighash1 =
        calculate_bip143_sighash(&tx, 1, &script_code, prevouts[1].value, 0x01, None).unwrap();
    assert_ne!(sighash0, sighash1);

    // Test with precomputed hashes (should match)
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_script_pubkeys: Vec<&[u8]> =
        prevouts.iter().map(|p| p.script_pubkey.as_ref()).collect();
    let precomputed =
        Bip143PrecomputedHashes::compute(&tx, &prevout_values, &prevout_script_pubkeys);
    let sighash0_precomputed = calculate_bip143_sighash(
        &tx,
        0,
        &script_code,
        prevout_values[0],
        0x01,
        Some(&precomputed),
    )
    .unwrap();
    assert_eq!(sighash0, sighash0_precomputed);

    // N7: midstate batch == per-input BIP143 for SIGHASH_ALL.
    let script_codes = [script_code.as_slice(), script_code.as_slice()];
    let batched = batch_compute_bip143_sighashes(
        &tx,
        &prevout_values,
        &prevout_script_pubkeys,
        &script_codes,
        0x01,
    )
    .unwrap();
    assert_eq!(batched[0], sighash0);
    assert_eq!(batched[1], sighash1);
}

#[test]
fn test_bip143_anyonecanpay() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_0, PUSH_20_BYTES],
        }]
        .into(),
        lock_time: 0,
    };

    let script_code = {
        let mut s = vec![OP_DUP, OP_HASH160, PUSH_20_BYTES];
        s.extend_from_slice(&[0u8; 20]); // 20 zero bytes (pubkey hash)
        s.push(OP_EQUALVERIFY);
        s.push(OP_CHECKSIG);
        s // 25 bytes total
    };
    let amount = 10000000000i64;

    // SIGHASH_ALL
    let sighash_all = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x01, None).unwrap();

    // SIGHASH_ALL | ANYONECANPAY (0x81)
    let sighash_anyonecanpay =
        calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x81, None).unwrap();

    // Should be different (ANYONECANPAY zeroes hashPrevouts and hashSequence)
    assert_ne!(sighash_all, sighash_anyonecanpay);
}

/// Regression: 2-input legacy sighash must use EMPTY script for non-signing input.
/// Per consensus, only the signing input's scriptCode is included; others get 0x00.
/// Bug: build_preimage_2in1out/2in2out used full scriptPubKey for non-signing input → wrong sighash → IBD failure.
#[test]
fn test_2input_legacy_sighash_non_signing_empty_script() {
    // 2-in-1-out tx: when signing input 0, input 1's script must be empty in preimage
    let script_a = vec![
        PUSH_33_BYTES,
        0x02,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        OP_CHECKSIG,
    ]; // P2PK 35 bytes
    let script_b = vec![
        PUSH_33_BYTES,
        0x03,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        0x11,
        OP_CHECKSIG,
    ]; // Different P2PK
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1u8; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2u8; 32],
                    index: 1,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_DUP, OP_HASH160, PUSH_20_BYTES],
        }]
        .into(),
        lock_time: 0,
    };
    let pv: Vec<i64> = vec![10_000_000_000, 8_000_000_000];
    let psp_ab: Vec<&[u8]> = vec![script_a.as_slice(), script_b.as_slice()];
    let psp_aa: Vec<&[u8]> = vec![script_a.as_slice(), script_a.as_slice()];

    // Signing input 0: input 1's script must be empty. So (script_a, script_b) and (script_a, script_a)
    // must produce the SAME sighash for input 0 — because input 1 is empty in preimage.
    let sighash_ab = calculate_transaction_sighash_with_script_code(
        &tx,
        0,
        &pv,
        &psp_ab,
        SighashType::ALL,
        None,
        #[cfg(feature = "production")]
        None,
    )
    .unwrap();
    let sighash_aa = calculate_transaction_sighash_with_script_code(
        &tx,
        0,
        &pv,
        &psp_aa,
        SighashType::ALL,
        None,
        #[cfg(feature = "production")]
        None,
    )
    .unwrap();
    assert_eq!(
        sighash_ab, sighash_aa,
        "2-input legacy: signing input 0 — input 1 script must be empty; changing input 1 scriptPubKey must not change sighash"
    );
}

#[cfg(feature = "production")]
#[test]
fn test_batch_sighash_single_input_index_ge_outputs() {
    // 2 inputs, 1 output: SIGHASH_SINGLE for input_index=1 has no corresponding output.
    // Must not panic; must return consensus hash 0x0000...0001.
    let tx = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: OutPoint {
                    hash: [1u8; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [2u8; 32],
                    index: 1,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![OP_DUP, OP_HASH160, PUSH_20_BYTES],
        }]
        .into(),
        lock_time: 0,
    };
    let prevout_values = vec![10_000_000_000i64, 8_000_000_000i64];
    let script = vec![OP_DUP, OP_HASH160, PUSH_20_BYTES];
    let prevout_script_pubkeys: Vec<&[u8]> = vec![script.as_slice(), script.as_slice()];
    let specs = vec![(1usize, 0x03u8, script.as_slice() as &[u8])]; // SIGHASH_SINGLE, input 1
    let hashes = super::batch_compute_legacy_sighashes(
        &tx,
        &prevout_values,
        &prevout_script_pubkeys,
        &specs,
    )
    .unwrap();
    assert_eq!(hashes.len(), 1);
    let mut expected = [0u8; 32];
    expected[0] = 1;
    assert_eq!(
        hashes[0], expected,
        "SIGHASH_SINGLE with input_index>=outputs.len() must return 0x0000...0001"
    );
}
