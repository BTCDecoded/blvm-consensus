use super::*;

#[test]
fn test_count_sigops_checksig() {
    // Script with OP_CHECKSIG
    let script = vec![OP_1, OP_1, OP_CHECKSIG]; // OP_1, OP_1, OP_CHECKSIG
    assert_eq!(count_sigops_in_script(&script, false), 1);
}

#[test]
fn test_count_sigops_checksigverify() {
    // Script with OP_CHECKSIGVERIFY
    let script = vec![OP_1, OP_1, OP_CHECKSIGVERIFY]; // OP_1, OP_1, OP_CHECKSIGVERIFY
    assert_eq!(count_sigops_in_script(&script, false), 1);
}

#[test]
fn test_count_sigops_multisig() {
    // Script with OP_CHECKMULTISIG (defaults to 20)
    let script = vec![OP_1, OP_2, OP_CHECKMULTISIG]; // OP_1, OP_2, OP_CHECKMULTISIG
    assert_eq!(count_sigops_in_script(&script, false), 20);

    // Accurate mode: use OP_2 value (2 sigops)
    assert_eq!(count_sigops_in_script(&script, true), 2);
}

#[test]
fn test_get_legacy_sigop_count() {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0,
            },
            script_sig: vec![OP_1, OP_CHECKSIG], // OP_1, OP_CHECKSIG
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1, OP_CHECKSIGVERIFY], // OP_1, OP_CHECKSIGVERIFY
        }]
        .into(),
        lock_time: 0,
    };

    assert_eq!(get_legacy_sigop_count(&tx), 2);
}

#[test]
fn test_is_pay_to_script_hash() {
    // Valid P2SH script: OP_HASH160 <20 bytes> OP_EQUAL
    let mut p2sh_script = vec![OP_HASH160, 0x14]; // OP_HASH160, push 20
    p2sh_script.extend_from_slice(&[0u8; 20]);
    p2sh_script.push(OP_EQUAL); // OP_EQUAL

    assert!(is_pay_to_script_hash(&p2sh_script));

    // Invalid: wrong length
    assert!(!is_pay_to_script_hash(&[OP_HASH160, 0x14]));

    // Invalid: not P2SH format
    let p2pkh = vec![OP_DUP, OP_HASH160, 0x14]; // OP_DUP OP_HASH160
    assert!(!is_pay_to_script_hash(&p2pkh));
}

// ==========================================================================
// REGRESSION TESTS: Push data must not be counted as sigops (block 310357 fix)
// ==========================================================================
// These tests prevent regression of the bug where bytes inside push data
// (e.g., 0xAC = OP_CHECKSIG) were incorrectly counted as sigops.
// This caused valid blocks to be rejected with "sigop cost exceeds maximum".

#[test]
fn test_pushdata1_containing_checksig_byte_not_counted() {
    // OP_PUSHDATA1 <len=3> <0xAC 0xAC 0xAC>
    // The 0xAC bytes are push DATA, not OP_CHECKSIG opcodes.
    // Sigop count must be 0.
    let script = vec![OP_PUSHDATA1, 0x03, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG];
    assert_eq!(
        count_sigops_in_script(&script, false),
        0,
        "Push data containing 0xAC must NOT be counted as sigops"
    );
}

#[test]
fn test_pushdata2_containing_checksig_byte_not_counted() {
    // OP_PUSHDATA2 <len=4 as u16 LE> <0xAC 0xAD 0xAE 0xAF>
    // These are push DATA bytes, not opcodes.
    let script = vec![
        OP_PUSHDATA2,
        0x04,
        0x00,
        OP_CHECKSIG,
        OP_CHECKSIGVERIFY,
        OP_CHECKMULTISIG,
        OP_CHECKMULTISIGVERIFY,
    ];
    assert_eq!(
        count_sigops_in_script(&script, false),
        0,
        "Push data containing sigop-like bytes must NOT be counted"
    );
}

#[test]
fn test_pushdata4_containing_checksig_byte_not_counted() {
    // OP_PUSHDATA4 <len=2 as u32 LE> <0xAC 0xAC>
    let script = vec![
        OP_PUSHDATA4,
        0x02,
        0x00,
        0x00,
        0x00,
        OP_CHECKSIG,
        OP_CHECKSIG,
    ];
    assert_eq!(
        count_sigops_in_script(&script, false),
        0,
        "OP_PUSHDATA4 data containing 0xAC must NOT be counted"
    );
}

#[test]
fn test_direct_push_containing_checksig_byte_not_counted() {
    // Direct push: opcode 0x05 means "push next 5 bytes"
    // Data contains OP_CHECKSIG byte which must NOT be counted.
    let script = vec![
        0x05,
        OP_CHECKSIG,
        OP_CHECKSIG,
        OP_CHECKSIG,
        OP_CHECKSIG,
        OP_CHECKSIG,
    ];
    assert_eq!(
        count_sigops_in_script(&script, false),
        0,
        "Direct push data containing 0xAC must NOT be counted as sigops"
    );
}

#[test]
fn test_push_data_then_real_checksig() {
    // Direct push of 3 bytes (containing 0xAC), then a REAL OP_CHECKSIG
    // Only the real OP_CHECKSIG (after push data) should count.
    let script = vec![0x03, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG, OP_CHECKSIG]; // push 3, data, then OP_CHECKSIG
    assert_eq!(
        count_sigops_in_script(&script, false),
        1,
        "Only real OP_CHECKSIG after push data should count"
    );
}

#[test]
fn test_pushdata1_then_real_multisig() {
    // OP_PUSHDATA1 <len=2> <OP_CHECKSIG OP_CHECKSIG> then OP_2 OP_CHECKMULTISIG
    // The OP_CHECKSIG bytes in push data don't count. Only the real OP_CHECKMULTISIG counts.
    let script = vec![
        OP_PUSHDATA1,
        0x02,
        OP_CHECKSIG,
        OP_CHECKSIG,
        OP_2,
        OP_CHECKMULTISIG,
    ];
    // Inaccurate mode: multisig = 20
    assert_eq!(
        count_sigops_in_script(&script, false),
        20,
        "Only real OP_CHECKMULTISIG should count (inaccurate=20)"
    );
    // Accurate mode: OP_2 before OP_CHECKMULTISIG = 2
    assert_eq!(
        count_sigops_in_script(&script, true),
        2,
        "Accurate mode: OP_2 before OP_CHECKMULTISIG = 2 sigops"
    );
}

#[test]
fn test_empty_script_zero_sigops() {
    let script: Vec<u8> = vec![];
    assert_eq!(count_sigops_in_script(&script, false), 0);
    assert_eq!(count_sigops_in_script(&script, true), 0);
}

#[test]
fn test_truncated_pushdata1_does_not_panic() {
    // OP_PUSHDATA1 at end of script (no length byte)
    let script = vec![OP_PUSHDATA1];
    assert_eq!(count_sigops_in_script(&script, false), 0);
}

#[test]
fn test_truncated_pushdata2_does_not_panic() {
    // OP_PUSHDATA2 with only 1 of 2 length bytes
    let script = vec![OP_PUSHDATA2, 0x01];
    assert_eq!(count_sigops_in_script(&script, false), 0);
}

#[test]
fn test_truncated_pushdata4_does_not_panic() {
    // OP_PUSHDATA4 with only 3 of 4 length bytes
    let script = vec![OP_PUSHDATA4, 0x01, 0x00, 0x00];
    assert_eq!(count_sigops_in_script(&script, false), 0);
}

#[test]
fn test_large_push_data_with_many_checksig_bytes() {
    // Simulate a realistic script where push data contains many OP_CHECKSIG bytes
    // This is the kind of script that caused the block 310357 failure.
    // OP_PUSHDATA2 <len=100 as u16 LE> <100 bytes of OP_CHECKSIG>
    let mut script = vec![OP_PUSHDATA2, 100, 0x00]; // OP_PUSHDATA2, length=100
    script.extend_from_slice(&[OP_CHECKSIG; 100]); // 100 bytes of OP_CHECKSIG data
    assert_eq!(
        count_sigops_in_script(&script, false),
        0,
        "100 bytes of OP_CHECKSIG in push data must count as 0 sigops"
    );
}

#[test]
fn test_multiple_sigop_opcodes() {
    // OP_CHECKSIG OP_CHECKSIG OP_CHECKSIGVERIFY
    let script = vec![0xac, 0xac, 0xad];
    assert_eq!(count_sigops_in_script(&script, false), 3);
}

#[test]
fn test_multisig_accurate_op_16() {
    // OP_16 (0x60) OP_CHECKMULTISIG
    let script = vec![0x60, 0xae];
    assert_eq!(count_sigops_in_script(&script, true), 16);
}

#[test]
fn test_multisig_accurate_op_1() {
    // OP_1 (0x51) OP_CHECKMULTISIG
    let script = vec![0x51, 0xae];
    assert_eq!(count_sigops_in_script(&script, true), 1);
}

/// Prefetched-UTXO sigop cost must match overlay-based counting (used on assume-valid path).
/// P2TR witness does not contribute to legacy block sigop cost (witness v1 = 0).
#[test]
fn get_transaction_sigop_cost_with_utxos_matches_witness_slices_p2tr_excluded_from_block_cost() {
    use crate::segwit::Witness;

    let prev = OutPoint {
        hash: [7u8; 32],
        index: 0,
    };
    let mut spk = vec![OP_1, 0x20];
    spk.extend_from_slice(&[9u8; 32]);

    let utxo = UTXO {
        value: 50_000,
        script_pubkey: spk.into(),
        height: 700_000,
        is_coinbase: false,
    };

    let mut set: UtxoSet = Default::default();
    utxo_set_insert(&mut set, prev, utxo);

    let tapscript = vec![OP_CHECKSIG];
    let witness_two: Witness = vec![tapscript.clone(), vec![0u8; 32]];
    let witness_three: Witness = vec![tapscript, vec![0x50], vec![0u8; 32]];

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: prev,
            script_sig: vec![],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 10_000,
            script_pubkey: vec![OP_0],
        }]
        .into(),
        lock_time: 0,
    };

    let flags = 0x800 | 0x8000 | 0x01;
    let uref = set.get(&prev).map(|a| a.as_ref());
    let utxo_refs: Vec<Option<&UTXO>> = vec![uref];

    for witnesses in [&witness_two, &witness_three] {
        let w = vec![witnesses.clone()];
        let with_slices =
            get_transaction_sigop_cost_with_witness_slices(&tx, &set, Some(w.as_slice()), flags)
                .unwrap();
        let with_utxos =
            get_transaction_sigop_cost_with_utxos(&tx, &utxo_refs, Some(w.as_slice()), flags)
                .unwrap();
        assert_eq!(
            with_utxos, with_slices,
            "sigop cost must match between utxo prefetch and overlay lookup"
        );
        assert_eq!(
            with_slices, 0,
            "tapscript in P2TR witness must not add to block sigop cost"
        );
    }
}

fn p2sh_script_pubkey() -> Vec<u8> {
    let mut script = vec![OP_HASH160, 0x14];
    script.extend_from_slice(&[0x11; 20]);
    script.push(OP_EQUAL);
    script
}

fn direct_push(payload: &[u8]) -> Vec<u8> {
    let mut script = vec![payload.len() as u8];
    script.extend_from_slice(payload);
    script
}

fn spend_cost(script_pubkey: Vec<u8>, script_sig: Vec<u8>, witness: Witness, flags: u32) -> u64 {
    let prev = OutPoint {
        hash: [3u8; 32],
        index: 1,
    };
    let utxo = UTXO {
        value: 50_000,
        script_pubkey: script_pubkey.into(),
        height: 500_000,
        is_coinbase: false,
    };
    let mut set: UtxoSet = Default::default();
    crate::utxo_set_insert(&mut set, prev, utxo);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: prev,
            script_sig,
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 10_000,
            script_pubkey: vec![OP_1],
        }]
        .into(),
        lock_time: 0,
    };
    let witnesses = vec![witness];
    let with_slices =
        get_transaction_sigop_cost_with_witness_slices(&tx, &set, Some(&witnesses), flags).unwrap();
    let uref = set.get(&prev).map(|a| a.as_ref());
    let with_utxos =
        get_transaction_sigop_cost_with_utxos(&tx, &[uref], Some(&witnesses), flags).unwrap();
    assert_eq!(with_utxos, with_slices);
    with_slices
}

#[test]
fn nested_v0_witness_sigops_count_toward_block_cost() {
    let flags = 0x01 | 0x800;
    let mut p2wpkh = vec![OP_0, 0x14];
    p2wpkh.extend_from_slice(&[0x22; 20]);
    assert_eq!(
        spend_cost(p2sh_script_pubkey(), direct_push(&p2wpkh), vec![], flags),
        1,
        "nested 20-byte program costs 1 even with an empty witness"
    );

    let mut p2wsh = vec![OP_0, 0x20];
    p2wsh.extend_from_slice(&[0x33; 32]);
    let witness_script = vec![OP_CHECKSIG, OP_CHECKSIG];
    assert_eq!(
        spend_cost(
            p2sh_script_pubkey(),
            direct_push(&p2wsh),
            vec![witness_script],
            flags,
        ),
        2,
        "nested 32-byte program costs the witness script's sigops"
    );
    assert_eq!(
        spend_cost(p2sh_script_pubkey(), direct_push(&p2wsh), vec![], flags),
        0,
        "nested 32-byte program with an empty witness costs 0"
    );

    let mut native = vec![OP_0, 0x14];
    native.extend_from_slice(&[0x44; 20]);
    assert_eq!(
        spend_cost(native, vec![], vec![], flags),
        1,
        "native 20-byte program costs 1 even with an empty witness"
    );
}

#[test]
fn p2sh_sigops_keep_the_redeem_after_small_pushes() {
    let flags = 0x01 | 0x800;
    let redeem = vec![OP_CHECKSIG];
    let plain = spend_cost(p2sh_script_pubkey(), direct_push(&redeem), vec![], flags);
    assert_eq!(plain, 4, "one redeem CHECKSIG costs 4");

    for prefix in [OP_1NEGATE, OP_RESERVED, OP_1] {
        let mut script_sig = vec![prefix];
        script_sig.extend(direct_push(&redeem));
        assert_eq!(
            spend_cost(p2sh_script_pubkey(), script_sig, vec![], flags),
            4,
            "prefix 0x{prefix:02x} is a push and must not drop the redeem sigop"
        );
    }

    assert_eq!(
        spend_cost(p2sh_script_pubkey(), vec![OP_1NEGATE], vec![], flags),
        0,
        "a lone OP_1NEGATE has an empty immediate and costs 0"
    );

    let mut then_non_push = direct_push(&redeem);
    then_non_push.push(OP_DUP);
    assert_eq!(
        spend_cost(p2sh_script_pubkey(), then_non_push, vec![], flags),
        0,
        "an opcode above OP_16 drops the P2SH sigop count"
    );

    let mut p2wpkh = vec![OP_0, 0x14];
    p2wpkh.extend_from_slice(&[0x22; 20]);
    for prefix in [OP_1NEGATE, OP_RESERVED] {
        let mut script_sig = vec![prefix];
        script_sig.extend(direct_push(&p2wpkh));
        assert_eq!(
            spend_cost(p2sh_script_pubkey(), script_sig, vec![], flags),
            1,
            "prefix 0x{prefix:02x} must still count a nested 20-byte program"
        );
    }
}
