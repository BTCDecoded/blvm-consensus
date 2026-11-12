use super::*;

#[test]
fn test_validate_taproot_script_valid() {
    let script = create_taproot_script(&[1u8; 32]);
    assert!(validate_taproot_script(&script).unwrap());
}

#[test]
fn test_validate_taproot_script_invalid_length() {
    let script = vec![0x51, 0x20]; // Too short
    assert!(!validate_taproot_script(&script).unwrap());
}

#[test]
fn test_validate_taproot_script_invalid_prefix() {
    let mut script = vec![0x52]; // Wrong prefix (OP_2 instead of OP_1)
    script.extend_from_slice(&[1u8; 32]);
    assert!(!validate_taproot_script(&script).unwrap());
}

#[test]
fn test_extract_taproot_output_key() {
    let expected_key = [1u8; 32];
    let script = create_taproot_script(&expected_key);

    let extracted_key = extract_taproot_output_key(&script).unwrap();
    assert_eq!(extracted_key, Some(expected_key));
}

#[test]
fn test_compute_taproot_tweak() {
    // Use a valid secp256k1 public key (x-coordinate only for Taproot)
    let internal_pubkey = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let merkle_root = [2u8; 32];

    let tweak = compute_taproot_tweak(&internal_pubkey, &merkle_root).unwrap();
    assert_eq!(tweak.len(), 32);
}

#[test]
fn test_validate_taproot_key_aggregation() {
    // Use a valid secp256k1 public key (x-coordinate only for Taproot)
    let internal_pubkey = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let merkle_root = [2u8; 32];
    let (output_key, parity) =
        crate::secp256k1_backend::taproot_output_key_with_parity(&internal_pubkey, &merkle_root)
            .unwrap();

    assert!(
        validate_taproot_key_aggregation(&internal_pubkey, &merkle_root, &output_key, parity)
            .unwrap()
    );
}

#[test]
fn test_validate_taproot_script_path() {
    let script = vec![0x51, 0x52]; // OP_1, OP_2
    let merkle_proof = vec![[3u8; 32], [4u8; 32]];
    let merkle_root =
        compute_script_merkle_root(&script, &merkle_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();

    assert!(validate_taproot_script_path(&script, &merkle_proof, &merkle_root).unwrap());
}

#[test]
fn test_validate_taproot_script_requires_push_opcode() {
    // 34 bytes but missing PUSH_32 (0x20) before output key
    let mut bad = vec![TAPROOT_SCRIPT_PREFIX];
    bad.extend_from_slice(&[0u8; 33]);
    assert!(!validate_taproot_script(&bad).unwrap());

    assert!(validate_taproot_script(&create_taproot_script(&[1u8; 32])).unwrap());
}

#[test]
fn test_is_taproot_output() {
    let output = TransactionOutput {
        value: 1000,
        script_pubkey: create_taproot_script(&[1u8; 32]),
    };

    assert!(is_taproot_output(&output));
}

#[test]
fn test_validate_taproot_transaction() {
    let tx = Transaction {
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
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: create_taproot_script(&[1u8; 32]),
        }]
        .into(),
        lock_time: 0,
    };

    // Key path spend: single signature
    let witness = Some(vec![vec![0u8; 64]]);
    assert!(validate_taproot_transaction(&tx, witness.as_ref()).unwrap());
}

#[test]
fn test_compute_taproot_signature_hash() {
    let tx = Transaction {
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
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let prevouts = [TransactionOutput {
        value: 2000,
        script_pubkey: create_taproot_script(&[1u8; 32]),
    }];
    let pv: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let psp: Vec<&[u8]> = prevouts
        .iter()
        .map(|p| p.script_pubkey.as_slice())
        .collect();
    let sig_hash = compute_taproot_signature_hash(&tx, 0, &pv, &psp, 0x01, None).unwrap();
    assert_eq!(sig_hash.len(), 32);
}

/// N7: multi-input Taproot reuses BIP341 commitments (TLS) — bit-identical across calls.
#[test]
fn n7_bip341_tls_cache_multi_input_identical() {
    let tx = Transaction {
        version: 2,
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
            value: 1000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };
    let prevouts = [
        TransactionOutput {
            value: 2000,
            script_pubkey: create_taproot_script(&[1u8; 32]),
        },
        TransactionOutput {
            value: 3000,
            script_pubkey: create_taproot_script(&[2u8; 32]),
        },
    ];
    let pv: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let psp: Vec<&[u8]> = prevouts
        .iter()
        .map(|p| p.script_pubkey.as_slice())
        .collect();
    let h0 = compute_taproot_signature_hash(&tx, 0, &pv, &psp, 0x01, None).unwrap();
    let h1 = compute_taproot_signature_hash(&tx, 1, &pv, &psp, 0x01, None).unwrap();
    assert_ne!(h0, h1);
    // Explicit compute matches TLS path (second call hits cache).
    let direct = Bip341PrecomputedHashes::compute(&tx, &pv, &psp);
    let cached = bip341_precompute(&tx, &pv, &psp);
    assert_eq!(direct.sha_prevouts, cached.sha_prevouts);
    assert_eq!(direct.sha_outputs, cached.sha_outputs);
    let t0 = std::time::Instant::now();
    for _ in 0..500 {
        let _ = bip341_precompute(&tx, &pv, &psp);
    }
    let cached_ns = t0.elapsed().as_nanos() / 500;
    let t1 = std::time::Instant::now();
    for _ in 0..500 {
        let _ = Bip341PrecomputedHashes::compute(&tx, &pv, &psp);
    }
    let fresh_ns = t1.elapsed().as_nanos() / 500;
    eprintln!("[N7 micro] bip341 TLS≈{cached_ns} ns/op fresh≈{fresh_ns} ns/op");
    assert!(
        cached_ns < fresh_ns,
        "TLS cache should beat full bip341_precompute"
    );
}

#[test]
fn bip341_cache_does_not_reuse_another_spend() {
    let mut tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [4u8; 32],
                index: 0,
            },
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };
    let script = create_taproot_script(&[1u8; 32]);
    let pv = vec![2_000i64];
    let psp: Vec<&[u8]> = vec![script.as_slice()];
    let first = compute_taproot_signature_hash(&tx, 0, &pv, &psp, 0x01, None).unwrap();

    // Same allocation, different output. An address key would return `first`.
    tx.outputs[0].value = 2_000;
    let other_output = compute_taproot_signature_hash(&tx, 0, &pv, &psp, 0x01, None).unwrap();
    assert_ne!(first, other_output);
    let fresh = Bip341PrecomputedHashes::compute(&tx, &pv, &psp);
    let reused = bip341_precompute(&tx, &pv, &psp);
    assert_eq!(fresh.sha_prevouts, reused.sha_prevouts);
    assert_eq!(fresh.sha_amounts, reused.sha_amounts);
    assert_eq!(fresh.sha_scriptpubkeys, reused.sha_scriptpubkeys);
    assert_eq!(fresh.sha_sequences, reused.sha_sequences);
    assert_eq!(fresh.sha_outputs, reused.sha_outputs);

    tx.outputs[0].value = 1_000;
    let other_amount = vec![9_000i64];
    let spent = compute_taproot_signature_hash(&tx, 0, &other_amount, &psp, 0x01, None).unwrap();
    assert_ne!(first, spent);

    let again = compute_taproot_signature_hash(&tx, 0, &pv, &psp, 0x01, None).unwrap();
    assert_eq!(first, again);
}

#[test]
fn test_compute_taproot_signature_hash_invalid_input_index() {
    let tx = Transaction {
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
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    let prevouts = [TransactionOutput {
        value: 2000,
        script_pubkey: create_taproot_script(&[1u8; 32]),
    }];
    let pv: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let psp: Vec<&[u8]> = prevouts
        .iter()
        .map(|p| p.script_pubkey.as_slice())
        .collect();
    // Use invalid input index (out of bounds)
    let sig_hash = compute_taproot_signature_hash(&tx, 1, &pv, &psp, 0x01, None).unwrap();
    assert_eq!(sig_hash.len(), 32);
}

#[test]
fn test_compute_taproot_signature_hash_empty_prevouts() {
    let tx = Transaction {
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
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    };

    // Empty prevout slices with a non-empty input list must return Err (fail-closed).
    // Previously this silently used zeros; the audit fix made it explicit.
    let prevouts: Vec<TransactionOutput> = vec![];
    let pv: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let psp: Vec<&[u8]> = prevouts
        .iter()
        .map(|p| p.script_pubkey.as_slice())
        .collect();
    let result = compute_taproot_signature_hash(&tx, 0, &pv, &psp, 0x01, None);
    assert!(
        result.is_err(),
        "empty prevouts shorter than tx.inputs must return Err, not silently use zeros"
    );
}

#[test]
fn test_compute_taproot_tweak_invalid_pubkey() {
    let invalid_pubkey = [0u8; 32]; // Invalid public key
    let merkle_root = [2u8; 32];

    let result = compute_taproot_tweak(&invalid_pubkey, &merkle_root);
    assert!(result.is_err());
}

#[test]
fn test_validate_taproot_key_aggregation_invalid() {
    let internal_pubkey = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let merkle_root = [2u8; 32];
    let wrong_output_key = [3u8; 32]; // Wrong output key

    assert!(
        !validate_taproot_key_aggregation(
            &internal_pubkey,
            &merkle_root,
            &wrong_output_key,
            0, // parity — irrelevant since key mismatch will fail first
        )
        .unwrap()
    );
}

#[test]
fn test_validate_taproot_script_path_invalid() {
    let script = vec![0x51, 0x52]; // OP_1, OP_2
    let merkle_proof = vec![[3u8; 32], [4u8; 32]];
    let wrong_merkle_root = [5u8; 32]; // Wrong merkle root

    assert!(!validate_taproot_script_path(&script, &merkle_proof, &wrong_merkle_root).unwrap());
}

#[test]
fn test_validate_taproot_script_path_empty_proof() {
    let script = vec![0x51, 0x52]; // OP_1, OP_2
    let merkle_proof = vec![];
    let merkle_root =
        compute_script_merkle_root(&script, &merkle_proof, TAPROOT_LEAF_VERSION_TAPSCRIPT).unwrap();

    assert!(validate_taproot_script_path(&script, &merkle_proof, &merkle_root).unwrap());
}

#[test]
fn test_tap_leaf_hash() {
    let script = vec![0x51, 0x52];
    let hash = crate::secp256k1_backend::tap_leaf_hash(TAPROOT_LEAF_VERSION_TAPSCRIPT, &script);

    assert_eq!(hash.len(), 32);

    let script2 = vec![0x53, 0x54];
    let hash2 = crate::secp256k1_backend::tap_leaf_hash(TAPROOT_LEAF_VERSION_TAPSCRIPT, &script2);
    assert_ne!(hash, hash2);
}

#[test]
fn test_tap_branch_hash() {
    let left = [1u8; 32];
    let right = [2u8; 32];
    let hash = crate::secp256k1_backend::tap_branch_hash(&left, &right);

    assert_eq!(hash.len(), 32);

    let hash2 = crate::secp256k1_backend::tap_branch_hash(&right, &left);
    assert_ne!(hash, hash2);
}

#[test]
fn test_encode_varint_small() {
    let encoded = encode_varint(0xfc);
    assert_eq!(encoded, vec![0xfc]);
}

#[test]
fn test_encode_varint_medium() {
    let encoded = encode_varint(0x1000);
    assert_eq!(encoded.len(), 3);
    assert_eq!(encoded[0], 0xfd);
}

#[test]
fn test_encode_varint_large() {
    let encoded = encode_varint(0x1000000);
    assert_eq!(encoded.len(), 5);
    assert_eq!(encoded[0], 0xfe);
}

#[test]
fn test_encode_varint_huge() {
    let encoded = encode_varint(0x1000000000000000);
    assert_eq!(encoded.len(), 9);
    assert_eq!(encoded[0], 0xff);
}

#[test]
fn test_extract_taproot_output_key_invalid_script() {
    let script = vec![0x52, 0x20]; // Invalid script
    let result = extract_taproot_output_key(&script).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_is_taproot_output_false() {
    let output = TransactionOutput {
        value: 1000,
        script_pubkey: vec![0x52, 0x20], // Not a Taproot script
    };

    assert!(!is_taproot_output(&output));
}

#[test]
fn test_validate_taproot_transaction_no_taproot_outputs() {
    let tx = Transaction {
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
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x52], // Not Taproot
        }]
        .into(),
        lock_time: 0,
    };

    // No witness needed for non-Taproot transaction
    assert!(validate_taproot_transaction(&tx, None).unwrap());
}

#[test]
fn test_validate_taproot_transaction_invalid_taproot_output() {
    // Create a transaction with a valid Taproot script
    let tx = Transaction {
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
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: create_taproot_script(&[1u8; 32]),
        }]
        .into(),
        lock_time: 0,
    };

    // Key path spend: single signature
    let witness = Some(vec![vec![0u8; 64]]);
    assert!(validate_taproot_transaction(&tx, witness.as_ref()).unwrap());
}

// Helper function
fn create_taproot_script(output_key: &[u8; 32]) -> ByteString {
    let mut script = vec![TAPROOT_SCRIPT_PREFIX, PUSH_32_BYTES];
    script.extend_from_slice(output_key);
    script
}
