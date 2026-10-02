//! Mainnet golden vectors (wire tier, plus a small consensus leaf).
//!
//! Each wire test decodes real chain bytes, spot-checks known facts, and
//! re-encodes byte-for-byte. Consensus here is only `check_tx_inputs` /
//! `apply_transaction` on the first-payment fixtures — not `connect_block`.
//!
//! Fixtures live in `golden_vectors/fixtures.rs`. Large era blocks are fetched
//! via `scripts/fetch_golden_block_481824.sh` (CI sets `BLVM_REQUIRE_GOLDEN_FIXTURES=1`).

#[path = "golden_vectors/mod.rs"]
mod golden_vectors;

use blvm_consensus::block::{apply_transaction, calculate_tx_id};
use blvm_consensus::crypto::OptimizedSha256;
use blvm_consensus::locktime::{LocktimeType, get_locktime_type, get_locktime_type_timestamp};
use blvm_consensus::mempool::is_final_tx;
use blvm_consensus::mining::compute_merkle_root_and_mutated;
use blvm_consensus::segwit::{
    compute_witness_merkle_root_from_nested, validate_witness_commitment,
};
use blvm_consensus::serialization::transaction::deserialize_transaction_with_witness;
use blvm_consensus::serialization::varint::{decode_varint, encode_varint};
use blvm_consensus::transaction::{check_transaction, check_tx_inputs, is_coinbase};
use blvm_consensus::types::{OutPoint, UTXO, UtxoSet, ValidationResult};
use golden_vectors::fixtures::*;
use golden_vectors::harness::*;
use std::path::Path;
use std::sync::Arc;

fn synth_hash(n: u8) -> blvm_consensus::types::Hash {
    let mut h = [0u8; 32];
    h[0] = n;
    h
}

// --- Transactions (mainnet wire bytes) ---

/// Block 170: first non-coinbase spend (10 BTC to Hal Finney, 40 BTC change).
#[test]
fn golden_first_bitcoin_payment_wire_and_txid() {
    let decoded = assert_tx_roundtrip(FIRST_BITCOIN_PAYMENT_HEX);
    let tx = &decoded.tx;

    assert_eq!(tx.version, 1);
    assert_eq!(tx.lock_time, 0);
    assert_eq!(tx.outputs.len(), 2);
    assert_eq!(tx.inputs.len(), 1);

    let prevout = &tx.inputs[0].prevout;
    assert_eq!(prevout.index, 0);
    assert_eq!(
        prevout.hash,
        hash_from_display(BLOCK9_COINBASE_TXID_DISPLAY)
    );
    assert_eq!(tx.inputs[0].sequence, 0xffff_ffff);
    assert_eq!(tx.outputs[0].value, 1_000_000_000);
    assert_eq!(tx.outputs[1].value, 4_000_000_000);

    let txid = calculate_tx_id(tx);
    assert_eq!(txid, hash_from_display(FIRST_PAYMENT_TXID_DISPLAY));
    assert_eq!(calculate_wtxid(tx, &decoded.witnesses), txid);
}

/// First payment at height 170: mature spend, fee 0, apply conserves 50 BTC.
#[test]
fn golden_first_payment_check_tx_inputs_and_apply() {
    let coinbase = assert_tx_roundtrip(BLOCK9_COINBASE_HEX).tx;
    let payment = assert_tx_roundtrip(FIRST_BITCOIN_PAYMENT_HEX).tx;

    assert!(matches!(
        check_transaction(&payment).unwrap(),
        ValidationResult::Valid
    ));

    let coinbase_txid = calculate_tx_id(&coinbase);
    let spent = OutPoint {
        hash: coinbase_txid,
        index: 0,
    };
    let mut utxos = UtxoSet::default();
    utxos.insert(
        spent,
        Arc::new(UTXO {
            value: coinbase.outputs[0].value,
            script_pubkey: coinbase.outputs[0].script_pubkey.as_slice().into(),
            height: 9,
            is_coinbase: true,
        }),
    );
    assert!(utxos.contains_key(&spent));

    let (at_170, fee) = check_tx_inputs(&payment, &utxos, 170).unwrap();
    assert!(matches!(at_170, ValidationResult::Valid));
    assert_eq!(fee, 0);

    let (at_105, _) = check_tx_inputs(&payment, &utxos, 105).unwrap();
    assert!(matches!(at_105, ValidationResult::Invalid(_)));

    let (next, _) = apply_transaction(&payment, utxos, 170).unwrap();
    let payment_txid = calculate_tx_id(&payment);
    assert!(next.get(&spent).is_none());
    let out0 = next
        .get(&OutPoint {
            hash: payment_txid,
            index: 0,
        })
        .expect("10 BTC output");
    let out1 = next
        .get(&OutPoint {
            hash: payment_txid,
            index: 1,
        })
        .expect("40 BTC output");
    assert_eq!(out0.value, 1_000_000_000);
    assert_eq!(out1.value, 4_000_000_000);
    assert_eq!(out0.height, 170);
    assert_eq!(out1.height, 170);
    assert!(!out0.is_coinbase);
    assert!(!out1.is_coinbase);
    assert_eq!(next.values().map(|u| u.value).sum::<i64>(), 5_000_000_000);
}

/// Duplicate prevout is rejected by CheckTransaction (rule 4).
#[test]
fn golden_first_payment_duplicate_input_rejected() {
    let payment = assert_tx_roundtrip(FIRST_BITCOIN_PAYMENT_HEX).tx;
    let mut doubled = payment.clone();
    let input = doubled.inputs[0].clone();
    doubled.inputs.push(input);
    assert!(matches!(
        check_transaction(&doubled).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

/// Block 9 coinbase; output 0 funds the first payment.
#[test]
fn golden_block9_coinbase_funds_first_payment() {
    let decoded = assert_tx_roundtrip(BLOCK9_COINBASE_HEX);
    let tx = &decoded.tx;

    assert_eq!(
        calculate_tx_id(tx),
        hash_from_display(BLOCK9_COINBASE_TXID_DISPLAY)
    );
    assert_eq!(tx.outputs.len(), 1);
    assert_eq!(tx.outputs[0].value, 5_000_000_000);
    assert!(is_coinbase(tx));
    assert!(matches!(
        check_transaction(tx).unwrap(),
        ValidationResult::Valid
    ));
}

/// Block 481824: SegWit activation coinbase (witness reserved value + commitment output).
#[test]
fn golden_segwit_activation_coinbase() {
    let decoded = assert_tx_roundtrip(SEGWIT_COINBASE_HEX);
    let tx = &decoded.tx;

    assert_eq!(tx.lock_time, 0);
    assert_eq!(tx.outputs.len(), 2);
    assert_eq!(tx.inputs.len(), 1);
    assert_eq!(tx.inputs[0].prevout.index, 0xffff_ffff);
    assert_eq!(tx.inputs[0].prevout.hash, [0u8; 32]);

    let witness = &decoded.witnesses[0];
    assert_eq!(witness.len(), 1);
    assert_eq!(witness[0].len(), 32);
    assert!(witness[0].iter().all(|&b| b == 0));

    let txid = calculate_tx_id(tx);
    assert_eq!(txid, hash_from_display(SEGWIT_COINBASE_TXID_DISPLAY));
    assert_ne!(calculate_wtxid(tx, &decoded.witnesses), txid);
}

/// Block 481824: first SegWit spend (P2SH-wrapped P2WPKH).
#[test]
fn golden_first_segwit_spend() {
    let decoded = assert_tx_roundtrip(FIRST_SEGWIT_SPEND_HEX);
    let tx = &decoded.tx;

    assert_eq!(tx.version, 2);
    assert_eq!(tx.lock_time, 0);
    assert_eq!(tx.outputs.len(), 1);
    assert_eq!(tx.inputs.len(), 1);
    assert_eq!(
        tx.inputs[0].prevout.hash,
        hash_from_display("42f7d0545ef45bd3b9cfee6b170cf6314a3bd8b3f09b610eeb436d92993ad440")
    );
    assert_eq!(tx.inputs[0].prevout.index, 1);

    let witness = &decoded.witnesses[0];
    assert_eq!(witness.len(), 2);
    assert_eq!(witness[0].len(), 72);
    assert_eq!(witness[1].len(), 33);

    assert!(matches!(
        check_transaction(tx).unwrap(),
        ValidationResult::Valid
    ));
}

/// Core empty tx: `version ‖ 00 ‖ 00 ‖ locktime`; decodes but fails CheckTransaction.
#[test]
fn golden_core_empty_tx() {
    let decoded = assert_tx_roundtrip(CORE_EMPTY_TX_HEX);
    let tx = &decoded.tx;

    assert!(tx.inputs.is_empty());
    assert!(tx.outputs.is_empty());
    assert_eq!(tx.version, 1);
    assert_eq!(tx.lock_time, 0);
    assert!(matches!(
        check_transaction(tx).unwrap(),
        ValidationResult::Invalid(_)
    ));
}

// --- BIP144 edge case (known wire-parser gap) ---

#[test]
#[ignore = "wire parser accepts superfluous witness; Core rejects at decode"]
fn golden_superfluous_witness_rejected_at_decode() {
    let bytes = hex_decode(SUPERFLUOUS_WITNESS_HEX);
    let result = deserialize_transaction_with_witness(&bytes);
    assert!(
        result.is_err(),
        "BIP144 superfluous-witness marker/flag tx with empty stacks must not decode \
         (Core v28 transaction.h)"
    );
}

/// Documents current wire-parser behavior until superfluous-witness rejection lands.
#[test]
fn golden_superfluous_witness_currently_decodes() {
    let bytes = hex_decode(SUPERFLUOUS_WITNESS_HEX);
    let (tx, witnesses, consumed) =
        deserialize_transaction_with_witness(&bytes).expect("currently decodes");
    assert_eq!(consumed, bytes.len());
    assert_eq!(tx.inputs.len(), 1);
    assert_eq!(tx.outputs.len(), 1);
    assert!(witnesses[0].is_empty());
}

// --- Blocks and headers ---

#[test]
fn golden_genesis_block() {
    let decoded = assert_block_roundtrip(GENESIS_BLOCK_HEX);
    let block = &decoded.block;
    let header = &block.header;

    assert_eq!(header.version, 1);
    assert_eq!(header.prev_block_hash, [0u8; 32]);
    assert_eq!(
        header.merkle_root,
        hash_from_display(GENESIS_COINBASE_TXID_DISPLAY)
    );
    assert_eq!(header.timestamp, 1_231_006_505);
    assert_eq!(header.bits, 0x1d00_ffff);
    assert_eq!(header.nonce, 2_083_236_893);
    assert_eq!(
        block_hash(header),
        hash_from_display(GENESIS_BLOCK_HASH_DISPLAY)
    );

    assert_eq!(block.transactions.len(), 1);
    let coinbase = &block.transactions[0];
    assert_eq!(coinbase.inputs[0].script_sig.len(), 77);
    assert_eq!(coinbase.outputs[0].value, 5_000_000_000);
    assert_eq!(
        calculate_tx_id(coinbase),
        hash_from_display(GENESIS_COINBASE_TXID_DISPLAY)
    );
    assert_eq!(calculate_tx_id(coinbase), header.merkle_root);
}

/// First block containing a non-coinbase tx; embeds the first-payment vector.
#[test]
fn golden_block_170_embeds_first_payment() {
    let decoded = assert_block_roundtrip(BLOCK_170_HEX);
    let block = &decoded.block;
    let header = &block.header;

    assert_eq!(header.version, 1);
    assert_eq!(
        header.prev_block_hash,
        hash_from_display("000000002a22cfee1f2c846adbd12b3e183d4f97683f85dad08a79780a84bd55")
    );
    assert_eq!(
        header.merkle_root,
        hash_from_display("7dac2c5666815c17a3b36427de37bb9d2e2c5ccec3f8633eb91a4205cb4c10ff")
    );
    assert_eq!(header.timestamp, 1_231_731_025);
    assert_eq!(header.bits, 0x1d00_ffff);

    assert_eq!(block.transactions.len(), 2);
    let payment_bytes = tx_roundtrip_bytes(
        &hex_decode(FIRST_BITCOIN_PAYMENT_HEX),
        &block.transactions[1],
        &decoded.witnesses[1],
    );
    assert_eq!(payment_bytes, hex_decode(FIRST_BITCOIN_PAYMENT_HEX));
}

/// Block 1 header extends genesis (chain linkage at the wire layer).
#[test]
fn golden_block1_header_extends_genesis() {
    let genesis = assert_block_roundtrip(GENESIS_BLOCK_HEX);
    let header = assert_header_roundtrip(BLOCK_1_HEADER_HEX);

    assert_eq!(block_hash(&header), hash_from_display(BLOCK_1_HASH_DISPLAY));
    assert_eq!(header.prev_block_hash, block_hash(&genesis.block.header));
}

// --- Encoding and crypto ---

#[test]
fn golden_compact_size_boundaries() {
    assert_eq!(encode_varint(0), vec![0x00]);
    assert_eq!(encode_varint(252), vec![0xfc]);
    assert_eq!(encode_varint(253), vec![0xfd, 0xfd, 0x00]);
    assert_eq!(encode_varint(0xffff), vec![0xfd, 0xff, 0xff]);
    assert_eq!(encode_varint(0x1_0000), vec![0xfe, 0x00, 0x00, 0x01, 0x00]);
    assert_eq!(
        encode_varint(0x1_0000_0000),
        vec![0xff, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
    );

    assert!(decode_varint(&[0xfd, 0xfc, 0x00]).is_err());
    assert!(decode_varint(&[0xfe, 0xff, 0xff, 0x00, 0x00]).is_err());
}

/// FIPS 180-4 and Bitcoin hash256 empty-string vectors.
#[test]
fn golden_sha256_known_answer_vectors() {
    let sha256 = |data: &[u8]| blvm_consensus::crypto::sha256(data);
    let sha256d = |data: &[u8]| OptimizedSha256::new().hash256(data);

    assert_eq!(
        sha256(&[]),
        hash_from_fips_hex("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    );
    assert_eq!(
        sha256(b"abc"),
        hash_from_fips_hex("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    );
    assert_eq!(
        sha256d(&[]),
        hash_from_fips_hex("5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456")
    );
    assert_eq!(
        sha256(&vec![0x61; 55]),
        hash_from_fips_hex("9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318")
    );
    assert_eq!(
        sha256(&vec![0x61; 56]),
        hash_from_fips_hex("b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a")
    );
    assert_eq!(
        sha256(&vec![0x61; 64]),
        hash_from_fips_hex("ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb")
    );
}

/// CVE-2012-2459: equal roots from padded dupes; Core `mutated` flag behavior.
#[test]
fn golden_merkle_cve_2012_2459_core_mutated_flag() {
    let h123 = [synth_hash(1), synth_hash(2), synth_hash(3)];
    let h1233 = [synth_hash(1), synth_hash(2), synth_hash(3), synth_hash(3)];

    let (root123, mut123) = compute_merkle_root_and_mutated(&h123).unwrap();
    let (root1233, mut1233) = compute_merkle_root_and_mutated(&h1233).unwrap();
    assert_eq!(root123, root1233);
    assert!(!mut123);
    assert!(mut1233);

    let h56dup = [
        synth_hash(1),
        synth_hash(2),
        synth_hash(3),
        synth_hash(4),
        synth_hash(5),
        synth_hash(6),
        synth_hash(5),
        synth_hash(6),
    ];
    let (_, mut_dup) = compute_merkle_root_and_mutated(&h56dup).unwrap();
    assert!(mut_dup);

    let (_, mut_pair) = compute_merkle_root_and_mutated(&[synth_hash(7), synth_hash(7)]).unwrap();
    assert!(mut_pair);
}

/// Header merkle root matches txid tree for genesis and block 170.
#[test]
fn golden_block_merkle_commitments() {
    for hex in [GENESIS_BLOCK_HEX, BLOCK_170_HEX] {
        let decoded = decode_block_full(&hex_decode(hex));
        let tx_ids: Vec<_> = decoded
            .block
            .transactions
            .iter()
            .map(calculate_tx_id)
            .collect();
        let (root, mutated) = compute_merkle_root_and_mutated(&tx_ids).unwrap();
        assert!(!mutated);
        assert_eq!(root, decoded.block.header.merkle_root);
    }
}

/// Full SegWit activation block (1866 txs). Requires fetched fixture; see module docs.
#[test]
fn golden_block_481824_segwit_activation_fixture() {
    let path = Path::new(BLOCK_481824_FIXTURE);
    if !path.exists() {
        if std::env::var("BLVM_REQUIRE_GOLDEN_FIXTURES").is_ok() {
            panic!(
                "missing {path:?}; run scripts/fetch_golden_block_481824.sh \
                 (required when BLVM_REQUIRE_GOLDEN_FIXTURES=1)"
            );
        }
        eprintln!(
            "skipping golden_block_481824: missing {path:?} \
             (run scripts/fetch_golden_block_481824.sh)"
        );
        return;
    }

    let bytes = std::fs::read(path).expect("read block 481824 fixture");
    let decoded = decode_block_full(&bytes);
    let header = &decoded.block.header;

    assert_eq!(header.version, 0x2000_0002);
    assert_eq!(
        header.prev_block_hash,
        hash_from_display("000000000000000000cbeff0b533f8e1189cf09dfbebf57a8ebe349362811b80")
    );
    assert_eq!(
        header.merkle_root,
        hash_from_display("6438250cad442b982801ae6994edb8a9ec63c0a0ba117779fbe7ef7f07cad140")
    );
    assert_eq!(header.timestamp, 1_503_539_857);
    assert_eq!(header.bits, 0x1801_3ce9);
    assert_eq!(header.nonce, 575_995_682);
    assert_eq!(
        block_hash(header),
        hash_from_display(BLOCK_481824_HASH_DISPLAY)
    );
    assert_eq!(decoded.block.transactions.len(), 1866);

    let coinbase_bytes = tx_roundtrip_bytes(
        &hex_decode(SEGWIT_COINBASE_HEX),
        &decoded.block.transactions[0],
        &decoded.witnesses[0],
    );
    assert_eq!(coinbase_bytes, hex_decode(SEGWIT_COINBASE_HEX));

    let first_spend_bytes = hex_decode(FIRST_SEGWIT_SPEND_HEX);
    let first_spend_txid =
        hash_from_display("c586389e5e4b3acb9d6c8be1c19ae8ab2795397633176f5a6442a261bbdefc3a");
    let embedded = decoded
        .block
        .transactions
        .iter()
        .zip(decoded.witnesses.iter())
        .find(|(tx, _w)| calculate_tx_id(tx) == first_spend_txid)
        .expect("block 481824 must contain first SegWit spend");
    let embedded_bytes = tx_roundtrip_bytes(&first_spend_bytes, embedded.0, embedded.1);
    assert_eq!(embedded_bytes, first_spend_bytes);

    let tx_ids: Vec<_> = decoded
        .block
        .transactions
        .iter()
        .map(calculate_tx_id)
        .collect();
    let (root, mutated) = compute_merkle_root_and_mutated(&tx_ids).unwrap();
    assert!(!mutated);
    assert_eq!(root, header.merkle_root);

    let coinbase = &decoded.block.transactions[0];
    let commitment = coinbase
        .outputs
        .iter()
        .map(|o| o.script_pubkey.as_slice())
        .find(|s| s.len() == 38 && s.starts_with(&[0x6a, 0x24, 0xaa, 0x21, 0xa9, 0xed]))
        .expect("481824 coinbase must have BIP141 witness commitment");
    assert_eq!(commitment.len(), 38);
    let witness_root =
        compute_witness_merkle_root_from_nested(&decoded.block, &decoded.witnesses, None)
            .expect("witness merkle root");
    assert!(
        validate_witness_commitment(coinbase, &witness_root, &decoded.witnesses[0])
            .expect("validate witness commitment")
    );
}

/// Taproot activation block (2043 txs). Requires fetched fixture; see module docs.
#[test]
fn golden_block_709632_taproot_activation_fixture() {
    let path = Path::new(BLOCK_709632_FIXTURE);
    if !path.exists() {
        if std::env::var("BLVM_REQUIRE_GOLDEN_FIXTURES").is_ok() {
            panic!(
                "missing {path:?}; run scripts/fetch_golden_block_481824.sh \
                 (required when BLVM_REQUIRE_GOLDEN_FIXTURES=1)"
            );
        }
        eprintln!(
            "skipping golden_block_709632: missing {path:?} \
             (run scripts/fetch_golden_block_481824.sh)"
        );
        return;
    }

    let bytes = std::fs::read(path).expect("read block 709632 fixture");
    let decoded = decode_block_full(&bytes);
    let header = &decoded.block.header;

    assert_eq!(header.version, 538_968_068);
    assert_eq!(
        header.prev_block_hash,
        hash_from_display("000000000000000000013712fc242ee6dd28476d0e9c931c75f83e6974c6bccc")
    );
    assert_eq!(
        header.merkle_root,
        hash_from_display("6ada3b10082068de09f7e819b65113d3c58969fd857aab2980c65f374714ec77")
    );
    assert_eq!(header.timestamp, 1_636_866_927);
    assert_eq!(header.bits, 386_689_514);
    assert_eq!(header.nonce, 1_410_298_626);
    assert_eq!(
        block_hash(header),
        hash_from_display(BLOCK_709632_HASH_DISPLAY)
    );
    assert_eq!(decoded.block.transactions.len(), 2043);

    let tx_ids: Vec<_> = decoded
        .block
        .transactions
        .iter()
        .map(calculate_tx_id)
        .collect();
    let (root, mutated) = compute_merkle_root_and_mutated(&tx_ids).unwrap();
    assert!(!mutated);
    assert_eq!(root, header.merkle_root);
}

/// Block 709635 P2TR key-path coinjoin (wire only; script-verify is elsewhere).
#[test]
fn golden_block709635_taproot_keypath_wire() {
    let decoded = assert_tx_roundtrip(BLOCK_709635_P2TR_KEYPATH_HEX);
    let tx = &decoded.tx;

    assert_eq!(tx.inputs.len(), 4);
    assert_eq!(tx.outputs.len(), 1);
    assert_eq!(
        calculate_tx_id(tx),
        hash_from_display(BLOCK_709635_P2TR_KEYPATH_TXID_DISPLAY)
    );
    assert_eq!(decoded.witnesses.len(), 4);
    for stack in &decoded.witnesses {
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].len(), 65);
    }
}

/// Block 170 clocks: payment is always final; BIP113 split on a mutated copy.
#[test]
fn golden_block170_locktime_mtp_clocks() {
    let payment = assert_tx_roundtrip(FIRST_BITCOIN_PAYMENT_HEX).tx;
    assert_eq!(payment.lock_time, 0);
    assert!(is_final_tx(&payment, 170, BLOCK_170_HEADER_TIME));

    assert_eq!(get_locktime_type(0), LocktimeType::BlockHeight);
    assert_eq!(get_locktime_type(499_999_999), LocktimeType::BlockHeight);
    assert_eq!(
        get_locktime_type_timestamp(500_000_000),
        LocktimeType::Timestamp
    );

    let mut copy = payment.clone();
    copy.inputs[0].sequence = 0xffff_fffe;
    copy.lock_time = 1_231_716_000;
    assert!(is_final_tx(&copy, 170, BLOCK_170_HEADER_TIME));
    assert!(!is_final_tx(&copy, 170, BLOCK_170_MTP));
}
