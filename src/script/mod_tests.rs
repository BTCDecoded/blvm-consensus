use super::*;

#[test]
fn tapscript_truncated_push_is_not_op_success() {
    use crate::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};

    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            sequence: 0xffff_ffff,
            script_sig: vec![],
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };
    let eval = |script: &[u8]| {
        let mut stack = Vec::new();
        eval_script_with_context_full(
            script,
            &mut stack,
            0,
            &tx,
            0,
            &[0],
            &[&[]],
            None,
            None,
            Network::Mainnet,
            SigVersion::Tapscript,
            None,
            None,
            None,
            Some(0),
            #[cfg(feature = "production")]
            None,
            None,
            #[cfg(feature = "production")]
            None,
        )
    };

    // Opcode 80 is an immediate success. A bare one succeeds.
    assert!(eval(&[0x50]).unwrap());
    // OP_PUSHDATA2 with a one-byte header. 0x50 is leftover length, not an opcode.
    assert!(!eval(&[0x4d, 0x50]).unwrap());
    // OP_PUSHDATA4 with a one-byte header. 0x7e is opcode 126, also an immediate success.
    assert!(!eval(&[0x4e, 0x7e]).unwrap());
    // 0x50 inside a real push is data. The following OP_0 leaves two stack items.
    assert!(!eval(&[0x01, 0x50, 0x00]).unwrap());

    // A push whose payload does not fit is not an immediate-success opcode,
    // whatever byte is left over. One supplied byte is short of every
    // direct push except a one-byte push, which is truncated by itself.
    assert!(!eval(&[0x01]).unwrap(), "direct push 1");
    for opcode in 2u8..=0x4b {
        assert!(!eval(&[opcode, 0x50]).unwrap(), "direct push {opcode}");
    }
    for (header, leftover) in [(0x4cu8, 0x50u8), (0x4d, 0x50), (0x4e, 0x50), (0x4e, 0x7e)] {
        assert!(!eval(&[header, leftover]).unwrap(), "header {header:#x}");
    }
}

fn eval_tapscript(
    script: &[u8],
    network: crate::types::Network,
    block_height: Option<u64>,
) -> Result<bool> {
    use crate::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            sequence: 0xffff_ffff,
            script_sig: vec![],
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };
    let mut stack = Vec::new();
    eval_script_with_context_full(
        script,
        &mut stack,
        0,
        &tx,
        0,
        &[0],
        &[&[]],
        block_height,
        None,
        network,
        SigVersion::Tapscript,
        None,
        None,
        None,
        Some(0),
        #[cfg(feature = "production")]
        None,
        None,
        #[cfg(feature = "production")]
        None,
    )
}

#[test]
fn mainnet_checksigfromstack_is_op_success() {
    let result = eval_tapscript(&[OP_CHECKSIGFROMSTACK], Network::Mainnet, Some(800_000));
    assert_eq!(result.unwrap(), true);
}

#[cfg(feature = "csfs")]
#[test]
fn active_csfs_is_not_an_immediate_success() {
    let mainnet = eval_tapscript(&[OP_CHECKSIGFROMSTACK], Network::Mainnet, Some(0));
    assert_eq!(mainnet.unwrap(), true);
    let regtest = eval_tapscript(&[OP_CHECKSIGFROMSTACK], Network::Regtest, Some(0));
    assert!(
        !matches!(regtest, Ok(true)),
        "regtest CSFS must run the opcode, got {regtest:?}"
    );
}

#[cfg(not(feature = "production"))]
#[test]
fn nonproduction_tapscript_schnorr_spend_is_accepted() {
    use crate::taproot::{
        TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root,
        compute_tapscript_signature_hash,
    };
    use crate::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[0x42; 32]).expect("secret");
    let keypair = Keypair::from_secret_key(&secp, &secret);
    let leaf_pk = keypair.x_only_public_key().0.serialize();
    let mut tapscript = vec![0x20];
    tapscript.extend_from_slice(&leaf_pk);
    tapscript.push(OP_CHECKSIG);
    let internal = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
    let (output_key, parity) =
        crate::secp256k1_backend::taproot_output_key_with_parity(&internal, &root).expect("tweak");
    let mut script_pubkey = vec![0x51, 0x20];
    script_pubkey.extend_from_slice(&output_key);
    let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control.extend_from_slice(&internal);
    let prevout_value = 10_000i64;
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x71; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let sighash = compute_tapscript_signature_hash(
        &tx,
        0,
        &[prevout_value],
        &[script_pubkey.as_slice()],
        &tapscript,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffff_ffff,
        0x00,
        None,
    )
    .expect("sighash");
    let msg = Message::from_digest_slice(&sighash).expect("digest");
    let sig = secp.sign_schnorr_no_aux_rand(&msg, &keypair);
    let witness = vec![sig.as_ref().to_vec(), tapscript, control];
    let accepted = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        Some(&witness),
        flags::SCRIPT_VERIFY_WITNESS | flags::SCRIPT_VERIFY_TAPROOT,
        &tx,
        0,
        &[prevout_value],
        &[script_pubkey.as_slice()],
        Some(crate::constants::TAPROOT_ACTIVATION_MAINNET),
        None,
        Network::Mainnet,
        SigVersion::Base,
        None,
    );
    assert_eq!(accepted.unwrap(), true);
}

#[test]
fn checkmultisig_pubkeys_count_toward_opcode_limit() {
    let mut within = vec![0x61; 160];
    within.push(0x00);
    within.push(0x00);
    within.extend(std::iter::repeat(0x00).take(20));
    within.extend_from_slice(&[0x01, 20, 0xae]);
    let mut stack = Vec::new();
    assert!(eval_script(&within, &mut stack, 0, SigVersion::Base).unwrap());

    let mut over = vec![0x61; 181];
    over.push(0x00);
    over.push(0x00);
    over.extend(std::iter::repeat(0x00).take(20));
    over.extend_from_slice(&[0x01, 20, 0xae]);
    let mut stack = Vec::new();
    assert!(eval_script(&over, &mut stack, 0, SigVersion::Base).is_err());

    // n outside 0..=20 is not added. 181 NOPs plus the opcode stay within 201.
    let mut ignored = vec![0x61; 181];
    ignored.extend_from_slice(&[0x00, 0x00, 0x01, 21, 0xae]);
    let mut stack = Vec::new();
    assert!(!eval_script(&ignored, &mut stack, 0, SigVersion::Base).unwrap());
}

#[test]
fn false_script_result_is_an_empty_vector() {
    let mut stack = Vec::new();
    assert!(eval_script(&[0x51, 0x52, 0x87], &mut stack, 0, SigVersion::Base).unwrap());
    assert!(stack.last().unwrap().is_empty());

    let mut stack = Vec::new();
    assert!(eval_script(&[0x51, 0x51, 0x87], &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack[0].as_slice(), &[1]);

    let mut stack = Vec::new();
    assert!(
        eval_script(
            &[0x00, 0x01, 0x02, 0xac, 0x82],
            &mut stack,
            0,
            SigVersion::Base
        )
        .unwrap()
    );
    assert!(stack.last().unwrap().is_empty());

    let mut stack = Vec::new();
    assert!(
        eval_script(
            &[0x00, 0x00, 0x51, 0x00, 0x51, 0xae, 0x82],
            &mut stack,
            0,
            SigVersion::Base
        )
        .unwrap()
    );
    assert!(stack.last().unwrap().is_empty());
}

#[test]
fn non_der_signature_aborts_the_script() {
    use crate::constants::BIP66_ACTIVATION_MAINNET;
    use crate::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            sequence: 0xffff_ffff,
            script_sig: vec![],
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };
    let eval = |script: &[u8], flags: u32| {
        let mut stack = Vec::new();
        eval_script_with_context_full(
            script,
            &mut stack,
            flags,
            &tx,
            0,
            &[0],
            &[&[]],
            Some(BIP66_ACTIVATION_MAINNET),
            None,
            Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            #[cfg(feature = "production")]
            None,
            None,
            #[cfg(feature = "production")]
            None,
        )
    };

    let bad = [0x03, 0x30, 0x01, 0x01, 0x01, 0x02, 0xac, 0x75, 0x51];
    assert!(eval(&bad, 0x04).is_err());
    let empty = [0x00, 0x01, 0x02, 0xac, 0x75, 0x51];
    assert!(eval(&empty, 0x04).unwrap());
    assert!(eval(&bad, 0).unwrap());
}

#[test]
fn nested_v0_redeem_push_must_be_canonical() {
    let mut redeem = vec![0x00, 0x14];
    redeem.extend_from_slice(&[0xab; 20]);
    let canonical = serialize_push_data(&redeem);
    let mut pushdata1 = vec![0x4c, redeem.len() as u8];
    pushdata1.extend_from_slice(&redeem);
    assert!(redeem_push_is_canonical(&canonical, &redeem));
    assert!(!redeem_push_is_canonical(&pushdata1, &redeem));
    assert_eq!(canonical[0], 22);
}

#[test]
fn keypath_spend_reaches_the_sighash_check() {
    use crate::script::flags::SCRIPT_VERIFY_WITNESS;
    use crate::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};

    let mut script_pubkey = vec![OP_1, PUSH_32_BYTES];
    script_pubkey.extend_from_slice(&[0x11u8; 32]);
    let tx = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x22; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let witness = vec![vec![0x40u8; 64]];
    let result = verify_script_with_context_full(
        &vec![].into(),
        &script_pubkey,
        Some(&witness),
        SCRIPT_VERIFY_WITNESS,
        &tx,
        0,
        &[],
        &[],
        Some(1),
        None,
        Network::Regtest,
        SigVersion::Tapscript,
        #[cfg(feature = "production")]
        None,
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(
        result.is_err(),
        "a key-path witness with no prevouts must reach the sighash check, got {result:?}"
    );
}

#[test]
fn p2sh_wrapped_unknown_witness_program_succeeds() {
    use digest::Digest;

    let redeem = vec![0x52, 0x02, 0x11, 0x22];
    let hash = ripemd::Ripemd160::digest(sha2::Sha256::digest(&redeem));
    let mut script_pubkey = vec![OP_HASH160, 0x14];
    script_pubkey.extend_from_slice(&hash);
    script_pubkey.push(OP_EQUAL);
    let script_sig = serialize_push_data(&redeem);
    let witness = vec![vec![1u8]];
    let (tx, pv, psp) = minimal_tx_and_prevouts(&script_sig, &script_pubkey);
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
    let verify = |flags: u32, wit: Option<&Vec<Vec<u8>>>| {
        verify_script_with_context_full(
            &script_sig,
            &script_pubkey,
            wit,
            flags,
            &tx,
            0,
            &pv,
            &psp_refs,
            Some(500_000),
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            #[cfg(feature = "production")]
            None,
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
            None,
        )
    };
    assert!(verify(0x801, Some(&witness)).unwrap());
    assert!(!verify(0x801 | 0x1000, Some(&witness)).unwrap());
    let mut pushdata1 = vec![0x4c, redeem.len() as u8];
    pushdata1.extend_from_slice(&redeem);
    let (tx_pd, pv_pd, psp_pd) = minimal_tx_and_prevouts(&pushdata1, &script_pubkey);
    let psp_pd_refs: Vec<&[u8]> = psp_pd.iter().map(|b| b.as_ref()).collect();
    let noncanonical = verify_script_with_context_full(
        &pushdata1,
        &script_pubkey,
        Some(&witness),
        0x801,
        &tx_pd,
        0,
        &pv_pd,
        &psp_pd_refs,
        Some(500_000),
        None,
        crate::types::Network::Mainnet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(!noncanonical.unwrap());

    let invalid_v0 = vec![0x00, 0x02, 0x11, 0x22];
    let hash = ripemd::Ripemd160::digest(sha2::Sha256::digest(&invalid_v0));
    let mut script_pubkey = vec![OP_HASH160, 0x14];
    script_pubkey.extend_from_slice(&hash);
    script_pubkey.push(OP_EQUAL);
    let script_sig = serialize_push_data(&invalid_v0);
    let (tx, pv, psp) = minimal_tx_and_prevouts(&script_sig, &script_pubkey);
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
    let result = verify_script_with_context_full(
        &script_sig,
        &script_pubkey,
        Some(&witness),
        0x801,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(500_000),
        None,
        crate::types::Network::Mainnet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(!result.unwrap());
}

/// scriptPubKey is executed before the witness-program shortcut. An
/// all-zero or negative-zero program leaves a false stack and fails. A
/// nonzero program of a future version succeeds.
#[test]
fn witness_program_false_payload_is_rejected() {
    use crate::constants::TAPROOT_ACTIVATION_MAINNET;
    use digest::Digest;

    let verify = |script_pubkey: &[u8], script_sig: &[u8], height: u64| {
        let script_sig = script_sig.to_vec();
        let (tx, pv, psp) = minimal_tx_and_prevouts(&script_sig, script_pubkey);
        let refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
        verify_script_with_context_full(
            &script_sig,
            script_pubkey,
            None,
            0x801,
            &tx,
            0,
            &pv,
            &refs,
            Some(height),
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            #[cfg(feature = "production")]
            None,
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
            None,
        )
    };

    let payloads: &[(&[u8], bool)] = &[
        (&[0x00, 0x00], false),
        (&[0x00, 0x80], false),
        (&[0x01, 0x00], true),
    ];
    for version in 0x51u8..=0x60 {
        for &(payload, truthy) in payloads {
            let mut script = vec![version, payload.len() as u8];
            script.extend_from_slice(payload);
            let got = verify(&script, &[], 400_000).unwrap();
            let taproot_v1 = version == 0x51 && payload.len() == 32;
            let expect = truthy && !taproot_v1;
            assert_eq!(got, expect, "native {script:02x?}");
        }
    }

    let zeros = vec![0u8; 32];
    let mut neg = vec![0u8; 32];
    neg[31] = 0x80;
    let mut nonzero = vec![0u8; 32];
    nonzero[0] = 1;
    for (payload, truthy) in [(&zeros[..], false), (&neg[..], false), (&nonzero[..], true)] {
        let mut script = vec![0x51, 0x20];
        script.extend_from_slice(payload);
        assert!(!verify(&script, &[], TAPROOT_ACTIVATION_MAINNET).unwrap());
        let before = verify(&script, &[], 400_000).unwrap();
        assert_eq!(before, truthy, "v1-32 before taproot {payload:02x?}");
    }

    let redeem = vec![0x52, 0x02, 0x00, 0x00];
    let hash = ripemd::Ripemd160::digest(sha2::Sha256::digest(&redeem));
    let mut p2sh = vec![OP_HASH160, 0x14];
    p2sh.extend_from_slice(&hash);
    p2sh.push(OP_EQUAL);
    let push = serialize_push_data(&redeem);
    assert!(!verify(&p2sh, &push, 400_000).unwrap());

    let redeem_ok = vec![0x52, 0x02, 0x01, 0x00];
    let hash_ok = ripemd::Ripemd160::digest(sha2::Sha256::digest(&redeem_ok));
    let mut p2sh_ok = vec![OP_HASH160, 0x14];
    p2sh_ok.extend_from_slice(&hash_ok);
    p2sh_ok.push(OP_EQUAL);
    assert!(verify(&p2sh_ok, &serialize_push_data(&redeem_ok), 400_000).unwrap());
}

/// BIP341 taproot is the native `OP_1 PUSH_32_BYTES` program. The same
/// program inside pay-to-script-hash stays an upgradable witness program
/// after activation: a truthy program succeeds, and an all-zero program fails.
#[test]
fn p2sh_wrapped_v1_32_stays_upgradable_after_taproot() {
    use crate::constants::TAPROOT_ACTIVATION_MAINNET;
    use crate::script::flags::{SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_WITNESS};
    use digest::Digest;

    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS;
    let verify = |script_pubkey: &[u8], script_sig: &[u8]| {
        let script_sig = script_sig.to_vec();
        let (tx, pv, psp) = minimal_tx_and_prevouts(&script_sig, script_pubkey);
        let refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
        verify_script_with_context_full(
            &script_sig,
            script_pubkey,
            None,
            flags,
            &tx,
            0,
            &pv,
            &refs,
            Some(TAPROOT_ACTIVATION_MAINNET),
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            #[cfg(feature = "production")]
            None,
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
            None,
        )
    };

    let program_len = PUSH_32_BYTES as usize;
    let mut truthy_redeem = vec![OP_1, PUSH_32_BYTES];
    truthy_redeem.extend(std::iter::repeat(1u8).take(program_len));
    let truthy_hash = ripemd::Ripemd160::digest(sha2::Sha256::digest(&truthy_redeem));
    let mut truthy_p2sh = vec![OP_HASH160, PUSH_20_BYTES];
    truthy_p2sh.extend_from_slice(&truthy_hash);
    truthy_p2sh.push(OP_EQUAL);
    assert!(verify(&truthy_p2sh, &serialize_push_data(&truthy_redeem)).unwrap());
    assert!(!verify(&truthy_redeem, &[]).unwrap());

    let mut zero_redeem = vec![OP_1, PUSH_32_BYTES];
    zero_redeem.extend(std::iter::repeat(0u8).take(program_len));
    let zero_hash = ripemd::Ripemd160::digest(sha2::Sha256::digest(&zero_redeem));
    let mut zero_p2sh = vec![OP_HASH160, PUSH_20_BYTES];
    zero_p2sh.extend_from_slice(&zero_hash);
    zero_p2sh.push(OP_EQUAL);
    assert!(!verify(&zero_p2sh, &serialize_push_data(&zero_redeem)).unwrap());
}

#[test]
fn test_eval_script_simple() {
    let script = vec![OP_1]; // OP_1
    let mut stack = Vec::new();

    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap());
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_eval_script_overflow() {
    let script = vec![0x51; MAX_STACK_SIZE + 1]; // Too many pushes
    let mut stack = Vec::new();

    assert!(eval_script(&script, &mut stack, 0, SigVersion::Base).is_err());
}

#[test]
fn test_verify_script_simple() {
    let _script_sig = [0x51]; // OP_1
    let _script_pubkey = [0x51]; // OP_1

    // This should work: OP_1 pushes 1, then OP_1 pushes another 1
    // Final stack has [1, 1], which is not exactly one non-zero value
    // Let's use a script that results in exactly one value on stack
    let script_sig = vec![0x51]; // OP_1
    let script_pubkey = vec![0x76, 0x88]; // OP_DUP, OP_EQUALVERIFY

    // This should fail because OP_EQUALVERIFY removes both values
    assert!(!verify_script(&script_sig, &script_pubkey, None, 0).unwrap());
}

// ============================================================================
// COMPREHENSIVE OPCODE TESTS
// ============================================================================

#[test]
fn test_op_0() {
    let script = vec![OP_0]; // OP_0
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script always succeeds; stack has empty item (falsy but not an error)
    assert_eq!(stack.len(), 1);
    assert!(stack[0].is_empty());
}

#[test]
fn test_op_1_to_op_16() {
    // Test OP_1 through OP_16
    for i in 1..=16 {
        let opcode = 0x50 + i;
        let script = vec![opcode];
        let mut stack = Vec::new();
        let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
        assert!(result);
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0].as_slice(), &[i]);
    }
}

#[test]
fn test_op_dup() {
    let script = vec![0x51, 0x76]; // OP_1, OP_DUP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 2 items (EvalScript doesn't check count)
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[1]);
}

#[test]
fn test_op_dup_empty_stack() {
    let script = vec![OP_DUP]; // OP_DUP on empty stack
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
}

#[test]
fn test_op_hash160() {
    let script = vec![OP_1, OP_HASH160]; // OP_1, OP_HASH160
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].len(), 20); // RIPEMD160 output is 20 bytes
}

#[test]
fn test_op_hash160_empty_stack() {
    let script = vec![OP_HASH160]; // OP_HASH160 on empty stack
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
}

#[test]
fn test_op_hash256() {
    let script = vec![OP_1, OP_HASH256]; // OP_1, OP_HASH256
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].len(), 32); // SHA256 output is 32 bytes
}

#[test]
fn test_op_hash256_empty_stack() {
    let script = vec![OP_HASH256]; // OP_HASH256 on empty stack
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
}

#[test]
fn test_op_equal() {
    let script = vec![0x51, 0x51, 0x87]; // OP_1, OP_1, OP_EQUAL
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result);
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]); // True
}

#[test]
fn test_op_equal_false() {
    let script = vec![0x51, 0x52, 0x87]; // OP_1, OP_2, OP_EQUAL
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; OP_EQUAL pushes [] (empty = falsy)
    assert_eq!(stack.len(), 1);
    assert!(stack[0].is_empty()); // False is [] (empty), not [0x00]
}

#[test]
fn test_op_equal_insufficient_stack() {
    let script = vec![0x51, 0x87]; // OP_1, OP_EQUAL (need 2 items)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(
        result.is_err(),
        "OP_EQUAL with insufficient stack should return error"
    );
    if let Err(crate::error::ConsensusError::ScriptErrorWithCode { code, .. }) = result {
        assert_eq!(
            code,
            crate::error::ScriptErrorCode::InvalidStackOperation,
            "Should return InvalidStackOperation"
        );
    }
}

#[test]
fn test_op_verify() {
    let script = vec![0x51, 0x69]; // OP_1, OP_VERIFY
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; OP_VERIFY consumed the truthy item
    assert_eq!(stack.len(), 0); // OP_VERIFY consumes the top item
}

#[test]
fn test_op_verify_false() {
    let script = vec![0x00, 0x69]; // OP_0, OP_VERIFY (false)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
}

#[test]
fn test_op_verify_empty_stack() {
    let script = vec![OP_VERIFY]; // OP_VERIFY on empty stack
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
}

#[test]
fn test_op_equalverify() {
    let script = vec![0x51, 0x51, 0x88]; // OP_1, OP_1, OP_EQUALVERIFY
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; OP_EQUALVERIFY consumed both equal items
    assert_eq!(stack.len(), 0); // OP_EQUALVERIFY consumes both items
}

#[test]
fn test_op_equalverify_false() {
    let script = vec![0x51, 0x52, 0x88]; // OP_1, OP_2, OP_EQUALVERIFY (false)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(
        result.is_err(),
        "OP_EQUALVERIFY with false condition should return error"
    );
    if let Err(crate::error::ConsensusError::ScriptErrorWithCode { code, .. }) = result {
        assert_eq!(
            code,
            crate::error::ScriptErrorCode::EqualVerify,
            "Should return EqualVerify"
        );
    }
}

#[test]
fn test_op_checksig() {
    // Note: This test uses simplified inputs. Production code performs full signature verification.
    // The test verifies that OP_CHECKSIG executes without panicking, not that signatures are valid.
    let script = vec![OP_1, OP_1, OP_CHECKSIG]; // OP_1, OP_1, OP_CHECKSIG
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; CHECKSIG pushes [] (invalid sig = falsy) but no error
    assert_eq!(stack.len(), 1);
    // Production code validates signatures using secp256k1; test uses simplified inputs
}

#[test]
fn test_op_checksig_insufficient_stack() {
    let script = vec![OP_1, OP_CHECKSIG]; // OP_1, OP_CHECKSIG (need 2 items)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(
        result.is_err(),
        "OP_CHECKSIG with insufficient stack should return error"
    );
    if let Err(crate::error::ConsensusError::ScriptErrorWithCode { code, .. }) = result {
        assert_eq!(
            code,
            crate::error::ScriptErrorCode::InvalidStackOperation,
            "Should return InvalidStackOperation"
        );
    }
}

#[test]
fn test_unknown_opcode() {
    let script = vec![0xff]; // Unknown opcode (0xff is not a defined opcode constant)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
}

#[test]
fn test_script_size_limit() {
    let script = vec![0x51; MAX_SCRIPT_SIZE + 1]; // Exceed size limit
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err());
}

#[test]
fn test_operation_count_limit() {
    // Use OP_NOP (0x61) - non-push opcodes count toward limit
    let script = vec![0x61; MAX_SCRIPT_OPS + 1]; // Exceed operation limit
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err());
}

#[test]
fn test_stack_underflow_multiple_ops() {
    let script = vec![0x51, 0x87, 0x87]; // OP_1, OP_EQUAL, OP_EQUAL (second OP_EQUAL will underflow)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base);
    assert!(result.is_err(), "Stack underflow should return error");
    if let Err(crate::error::ConsensusError::ScriptErrorWithCode { code, .. }) = result {
        assert_eq!(
            code,
            crate::error::ScriptErrorCode::InvalidStackOperation,
            "Should return InvalidStackOperation"
        );
    }
}

#[test]
fn test_final_stack_empty() {
    // eval_script (EvalScript) does not check the final stack — VerifyScript does.
    let script = vec![0x51, 0x52]; // OP_1, OP_2 (two items on final stack)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; final stack check is VerifyScript's responsibility
    assert_eq!(stack.len(), 2);
}

#[test]
fn test_final_stack_false() {
    // eval_script (EvalScript) does not check top-of-stack truthiness — VerifyScript does.
    let script = vec![OP_0]; // OP_0 (false on final stack)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; verify_script would return false here
}

#[test]
fn test_verify_script_with_witness() {
    // OP_1 sig + OP_1 pubkey + OP_1 witness leaves 3 truthy items; top is truthy → success.
    let script_sig = vec![OP_1]; // OP_1
    let script_pubkey = vec![OP_1]; // OP_1
    let witness = vec![OP_1]; // OP_1
    let flags = 0;

    let result = verify_script(&script_sig, &script_pubkey, Some(&witness), flags).unwrap();
    assert!(result); // VerifyScript: top of stack is truthy regardless of stack size
}

#[test]
fn test_verify_script_failure() {
    // OP_0 pubkey pushes empty (falsy) item on top; verify_script returns false.
    let script_sig = vec![OP_1]; // OP_1 (truthy)
    let script_pubkey = vec![OP_0]; // OP_0 (pushes empty = falsy)
    let witness = None;
    let flags = 0;

    let result = verify_script(&script_sig, &script_pubkey, witness, flags).unwrap();
    assert!(!result); // VerifyScript: top of stack is [] (falsy) → fail
}

// ============================================================================
// COMPREHENSIVE SCRIPT TESTS
// ============================================================================

#[test]
fn test_op_ifdup_true() {
    let script = vec![OP_1, OP_IFDUP]; // OP_1, OP_IFDUP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 2 items after dup
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[1]);
}

#[test]
fn test_op_ifdup_false() {
    let script = vec![OP_0, OP_IFDUP]; // OP_0, OP_IFDUP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; IFDUP doesn't dup falsy items
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[] as &[u8]);
}

#[test]
fn test_op_depth() {
    let script = vec![OP_1, OP_1, OP_DEPTH]; // OP_1, OP_1, OP_DEPTH
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 3 items
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[2].as_slice(), &[2]); // Depth should be 2 (before OP_DEPTH)
}

#[test]
fn test_op_drop() {
    let script = vec![OP_1, OP_2, OP_DROP]; // OP_1, OP_2, OP_DROP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // Final stack has 1 item [1]
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_op_drop_empty_stack() {
    let script = vec![OP_DROP]; // OP_DROP on empty stack
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 0);
}

#[test]
fn test_op_nip() {
    let script = vec![OP_1, OP_2, OP_NIP]; // OP_1, OP_2, OP_NIP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // Final stack has 1 item [2]
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[2]);
}

#[test]
fn test_op_nip_insufficient_stack() {
    let script = vec![OP_1, OP_NIP]; // OP_1, OP_NIP (only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_over() {
    let script = vec![OP_1, OP_2, OP_OVER]; // OP_1, OP_2, OP_OVER
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 3 items
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[2]);
    assert_eq!(stack[2].as_slice(), &[1]);
}

#[test]
fn test_op_over_insufficient_stack() {
    let script = vec![OP_1, OP_OVER]; // OP_1, OP_OVER (only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_pick() {
    let script = vec![OP_1, OP_2, OP_3, OP_1, OP_PICK]; // OP_1, OP_2, OP_3, OP_1, OP_PICK
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 4 items
    assert_eq!(stack.len(), 4);
    assert_eq!(stack[3].as_slice(), &[2]); // Should pick index 1 (OP_2)
}

#[test]
fn test_op_pick_empty_n() {
    // OP_1, OP_0, OP_PICK: n=0 picks top item (duplicates it), stack [1,1]
    let script = vec![OP_1, OP_0, OP_PICK];
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 2 items
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[1].as_slice(), &[1]); // Picked the top (OP_1 value)
}

#[test]
fn test_op_pick_invalid_index() {
    let script = vec![OP_1, OP_2, OP_PICK]; // OP_1, OP_2, OP_PICK (n=2, but only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_roll() {
    let script = vec![OP_1, OP_2, OP_3, OP_1, OP_ROLL]; // OP_1, OP_2, OP_3, OP_1, OP_ROLL
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 3 items
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[3]);
    assert_eq!(stack[2].as_slice(), &[2]); // Should roll index 1 (OP_2) to top
}

#[test]
fn test_op_roll_zero_n() {
    // OP_0 pushes empty bytes (CScriptNum 0), OP_ROLL(0) is a valid no-op
    let script = vec![OP_1, OP_0, OP_ROLL]; // OP_1, OP_0, OP_ROLL (n=0, no-op)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // Stack has [1], which is truthy
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_op_roll_invalid_index() {
    let script = vec![OP_1, OP_2, OP_ROLL]; // OP_1, OP_2, OP_ROLL (n=2, but only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_rot() {
    let script = vec![OP_1, OP_2, OP_3, OP_ROT]; // OP_1, OP_2, OP_3, OP_ROT
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 3 items
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[3]);
    assert_eq!(stack[2].as_slice(), &[1]);
}

#[test]
fn test_op_rot_insufficient_stack() {
    let script = vec![OP_1, OP_2, OP_ROT]; // OP_1, OP_2, OP_ROT (only 2 items)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 2);
}

#[test]
fn test_op_swap() {
    let script = vec![OP_1, OP_2, OP_SWAP]; // OP_1, OP_2, OP_SWAP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 2 items
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[1]);
}

#[test]
fn test_op_swap_insufficient_stack() {
    let script = vec![OP_1, OP_SWAP]; // OP_1, OP_SWAP (only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_tuck() {
    let script = vec![OP_1, OP_2, OP_TUCK]; // OP_1, OP_2, OP_TUCK
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 3 items
    assert_eq!(stack.len(), 3);
    assert_eq!(stack[0].as_slice(), &[2]);
    assert_eq!(stack[1].as_slice(), &[1]);
    assert_eq!(stack[2].as_slice(), &[2]);
}

#[test]
fn test_op_tuck_insufficient_stack() {
    let script = vec![OP_1, OP_TUCK]; // OP_1, OP_TUCK (only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_2drop() {
    let script = vec![OP_1, OP_2, OP_3, OP_2DROP]; // OP_1, OP_2, OP_3, OP_2DROP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // Final stack has 1 item [1]
    assert_eq!(stack.len(), 1);
    assert_eq!(stack[0].as_slice(), &[1]);
}

#[test]
fn test_op_2drop_insufficient_stack() {
    let script = vec![OP_1, OP_2DROP]; // OP_1, OP_2DROP (only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_2dup() {
    let script = vec![OP_1, OP_2, OP_2DUP]; // OP_1, OP_2, OP_2DUP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 4 items
    assert_eq!(stack.len(), 4);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[2]);
    assert_eq!(stack[2].as_slice(), &[1]);
    assert_eq!(stack[3].as_slice(), &[2]);
}

#[test]
fn test_op_2dup_insufficient_stack() {
    let script = vec![OP_1, OP_2DUP]; // OP_1, OP_2DUP (only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_3dup() {
    let script = vec![OP_1, OP_2, OP_3, OP_3DUP]; // OP_1, OP_2, OP_3, OP_3DUP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 6 items
    assert_eq!(stack.len(), 6);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[2]);
    assert_eq!(stack[2].as_slice(), &[3]);
    assert_eq!(stack[3].as_slice(), &[1]);
    assert_eq!(stack[4].as_slice(), &[2]);
    assert_eq!(stack[5].as_slice(), &[3]);
}

#[test]
fn test_op_3dup_insufficient_stack() {
    let script = vec![OP_1, OP_2, OP_3DUP]; // OP_1, OP_2, OP_3DUP (only 2 items)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 2);
}

#[test]
fn test_op_2over() {
    let script = vec![OP_1, OP_2, OP_3, OP_4, OP_2OVER]; // OP_1, OP_2, OP_3, OP_4, OP_2OVER
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 6 items
    assert_eq!(stack.len(), 6);
    assert_eq!(stack[4].as_slice(), &[1]); // Should copy second pair
    assert_eq!(stack[5].as_slice(), &[2]);
}

#[test]
fn test_op_2over_insufficient_stack() {
    let script = vec![OP_1, OP_2, OP_3, OP_2OVER]; // OP_1, OP_2, OP_3, OP_2OVER (only 3 items)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 3);
}

#[test]
fn test_op_2rot() {
    // Stack before: [1,2,3,4,5,6] (1=bottom, 6=top)
    // 2ROT moves 6th and 5th from top (=1 and 2) to top preserving order.
    // Stack after:  [3,4,5,6,1,2] (2=top, 1=second)
    let script = vec![OP_1, OP_2, OP_3, OP_4, OP_5, OP_6, OP_2ROT]; // 6 items, OP_2ROT
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 6 items
    assert_eq!(stack.len(), 6);
    assert_eq!(stack[4].as_slice(), &[1]); // OP_1 moved to second-from-top
    assert_eq!(stack[5].as_slice(), &[2]); // OP_2 moved to top
}

#[test]
fn test_op_2rot_insufficient_stack() {
    let script = vec![OP_1, OP_2, OP_3, OP_4, OP_2ROT]; // OP_1, OP_2, OP_3, OP_4, OP_2ROT (only 4 items)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 4);
}

#[test]
fn test_op_2swap() {
    let script = vec![OP_1, OP_2, OP_3, OP_4, OP_2SWAP]; // OP_1, OP_2, OP_3, OP_4, OP_2SWAP
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 4 items
    assert_eq!(stack.len(), 4);
    assert_eq!(stack[0].as_slice(), &[3]); // Should swap second pair
    assert_eq!(stack[1].as_slice(), &[4]);
    assert_eq!(stack[2].as_slice(), &[1]);
    assert_eq!(stack[3].as_slice(), &[2]);
}

#[test]
fn test_op_2swap_insufficient_stack() {
    let script = vec![OP_1, OP_2, OP_3, OP_2SWAP]; // OP_1, OP_2, OP_3, OP_2SWAP (only 3 items)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 3);
}

#[test]
fn test_op_size() {
    let script = vec![OP_1, OP_SIZE]; // OP_1, OP_SIZE
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(result); // eval_script succeeds; stack has 2 items (original + size)
    assert_eq!(stack.len(), 2);
    assert_eq!(stack[0].as_slice(), &[1]);
    assert_eq!(stack[1].as_slice(), &[1]); // Size of [1] is 1
}

#[test]
fn test_op_size_empty_stack() {
    let script = vec![OP_SIZE]; // OP_SIZE on empty stack
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 0);
}

#[test]
fn test_op_return() {
    let script = vec![OP_1, OP_RETURN]; // OP_1, OP_RETURN
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result); // OP_RETURN always fails
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_op_checksigverify() {
    let script = vec![OP_1, OP_2, OP_CHECKSIGVERIFY]; // OP_1, OP_2, OP_CHECKSIGVERIFY
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result); // Should fail due to invalid signature
    assert_eq!(stack.len(), 0);
}

#[test]
fn test_op_checksigverify_insufficient_stack() {
    let script = vec![OP_1, OP_CHECKSIGVERIFY]; // OP_1, OP_CHECKSIGVERIFY (only 1 item)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result);
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_unknown_opcode_comprehensive() {
    let script = vec![OP_1, 0xff]; // OP_1, unknown opcode (0xff is not a defined opcode constant)
    let mut stack = Vec::new();
    let result = eval_script(&script, &mut stack, 0, SigVersion::Base).unwrap();
    assert!(!result); // Unknown opcode should fail
    assert_eq!(stack.len(), 1);
}

#[test]
fn test_verify_signature_invalid_pubkey() {
    let secp = signature::new_secp();
    let invalid_pubkey = vec![0x00]; // Invalid pubkey
    let signature = vec![0x30, 0x06, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00]; // Valid DER signature
    let dummy_hash = [0u8; 32];
    let result = signature::verify_signature(
        &secp,
        &invalid_pubkey,
        &signature,
        &dummy_hash,
        0,
        0,
        crate::types::Network::Regtest,
        SigVersion::Base,
    );
    assert!(!result.unwrap_or(false));
}

#[test]
fn test_verify_signature_invalid_signature() {
    let secp = signature::new_secp();
    let pubkey = vec![
        0x02, 0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87,
        0x0b, 0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16,
        0xf8, 0x17, 0x98,
    ]; // Valid pubkey
    let invalid_signature = vec![0x00]; // Invalid signature
    let dummy_hash = [0u8; 32];
    let result = signature::verify_signature(
        &secp,
        &pubkey,
        &invalid_signature,
        &dummy_hash,
        0,
        0,
        crate::types::Network::Regtest,
        SigVersion::Base,
    );
    assert!(!result.unwrap_or(false));
}

// ============================================================================
// Fast-path and verify_script_with_context_full tests
// ============================================================================

/// Build a minimal transaction and prevout slices for verify_script_with_context_full.
fn minimal_tx_and_prevouts(
    script_sig: &[u8],
    script_pubkey: &[u8],
) -> (
    crate::types::Transaction,
    Vec<i64>,
    Vec<crate::types::ByteString>,
) {
    use crate::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            sequence: 0xffff_ffff,
            script_sig: script_sig.to_vec(),
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script_pubkey.to_vec(),
        }]
        .into(),
        lock_time: 0,
    };
    let prevout_values = vec![0i64];
    let prevout_script_pubkeys_vec = vec![script_pubkey.to_vec()];
    let prevout_script_pubkeys: Vec<&ByteString> = prevout_script_pubkeys_vec.iter().collect();
    (tx, prevout_values, prevout_script_pubkeys_vec)
}

#[test]
fn canonical_p2pk_requires_the_push_opcode_to_match_the_key_length() {
    let compressed = [2u8; PUSH_33_BYTES as usize];
    let mut matched = vec![PUSH_33_BYTES];
    matched.extend_from_slice(&compressed);
    matched.push(OP_CHECKSIG);
    assert_eq!(canonical_p2pk_pubkey(&matched), Some(compressed.as_slice()));

    let mut truncated = vec![PUSH_65_BYTES];
    truncated.extend_from_slice(&compressed);
    truncated.push(OP_CHECKSIG);
    assert!(canonical_p2pk_pubkey(&truncated).is_none());

    let uncompressed = [4u8; PUSH_65_BYTES as usize];
    let mut matched_long = vec![PUSH_65_BYTES];
    matched_long.extend_from_slice(&uncompressed);
    matched_long.push(OP_CHECKSIG);
    assert_eq!(
        canonical_p2pk_pubkey(&matched_long),
        Some(uncompressed.as_slice())
    );

    let mut short_push_on_long_script = vec![PUSH_33_BYTES];
    short_push_on_long_script.extend_from_slice(&uncompressed);
    short_push_on_long_script.push(OP_CHECKSIG);
    assert!(canonical_p2pk_pubkey(&short_push_on_long_script).is_none());
}

#[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
#[test]
fn p2pk_fast_path_does_not_accept_a_push_that_does_not_fit() {
    use crate::transaction_hash::{SighashType, calculate_transaction_sighash_single_input};
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[52u8; 32]).expect("secret");
    let pubkey = PublicKey::from_secret_key(&secp, &secret).serialize();

    let spend = |script_pubkey: &[u8]| {
        let (mut tx, prevout_values, prevout_scripts) = minimal_tx_and_prevouts(&[], script_pubkey);
        let sighash = calculate_transaction_sighash_single_input(
            &tx,
            0,
            script_pubkey,
            prevout_values[0],
            SighashType::ALL,
            None,
        )
        .expect("sighash");
        let msg = Message::from_digest_slice(&sighash).expect("digest");
        let mut der = secp.sign_ecdsa(&msg, &secret).serialize_der().to_vec();
        der.push(SighashType::ALL.0);
        let script_sig = serialize_push_data(&der);
        tx.inputs[0].script_sig = script_sig.clone();
        let refs: Vec<&[u8]> = prevout_scripts.iter().map(|s| s.as_slice()).collect();
        verify_script_with_context_full(
            &script_sig,
            script_pubkey,
            None,
            0,
            &tx,
            0,
            &prevout_values,
            &refs,
            None,
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("verify")
    };

    let mut mismatched = vec![PUSH_65_BYTES];
    mismatched.extend_from_slice(&pubkey);
    mismatched.push(OP_CHECKSIG);
    assert!(!spend(&mismatched));

    let mut matched = vec![PUSH_33_BYTES];
    matched.extend_from_slice(&pubkey);
    matched.push(OP_CHECKSIG);
    assert!(spend(&matched));
}

#[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
#[test]
fn raw_key_multisig_is_not_accepted_by_the_fast_path() {
    use crate::transaction_hash::{SighashType, calculate_transaction_sighash_single_input};
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[52u8; 32]).expect("secret");
    let pubkey = PublicKey::from_secret_key(&secp, &secret).serialize();

    let spend = |script_pubkey: &[u8]| {
        let (mut tx, prevout_values, prevout_scripts) = minimal_tx_and_prevouts(&[], script_pubkey);
        let sighash = calculate_transaction_sighash_single_input(
            &tx,
            0,
            script_pubkey,
            prevout_values[0],
            SighashType::ALL,
            None,
        )
        .expect("sighash");
        let msg = Message::from_digest_slice(&sighash).expect("digest");
        let mut der = secp.sign_ecdsa(&msg, &secret).serialize_der().to_vec();
        der.push(SighashType::ALL.0);
        let mut script_sig = vec![OP_0];
        script_sig.extend(serialize_push_data(&der));
        tx.inputs[0].script_sig = script_sig.clone();
        let refs: Vec<&[u8]> = prevout_scripts.iter().map(|s| s.as_slice()).collect();
        verify_script_with_context_full(
            &script_sig,
            script_pubkey,
            None,
            0,
            &tx,
            0,
            &prevout_values,
            &refs,
            None,
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    };

    let mut pushed = vec![OP_1, PUSH_33_BYTES];
    pushed.extend_from_slice(&pubkey);
    pushed.push(OP_1);
    pushed.push(OP_CHECKMULTISIG);
    assert!(matches!(spend(&pushed), Ok(true)));

    let mut raw = vec![OP_1];
    raw.extend_from_slice(&pubkey);
    raw.push(OP_1);
    raw.push(OP_CHECKMULTISIG);
    assert!(!matches!(spend(&raw), Ok(true)));
}

#[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
#[test]
fn reserved_opcode_multisig_is_not_accepted_by_the_fast_path() {
    use crate::transaction_hash::{SighashType, calculate_transaction_sighash_single_input};
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[52u8; 32]).expect("secret");
    let pubkey = PublicKey::from_secret_key(&secp, &secret).serialize();

    let mut redeem = vec![OP_1, PUSH_33_BYTES];
    redeem.extend_from_slice(&pubkey);
    redeem.push(OP_1);
    redeem.push(OP_CHECKMULTISIG);

    let redeem_hash = Ripemd160::digest(OptimizedSha256::new().hash(&redeem));
    let mut p2sh = vec![OP_HASH160, PUSH_20_BYTES];
    p2sh.extend_from_slice(&redeem_hash);
    p2sh.push(OP_EQUAL);

    let spend = |script_pubkey: &[u8], script_code: &[u8], prefix: &[u8], flags: u32| {
        let (mut tx, prevout_values, prevout_scripts) = minimal_tx_and_prevouts(&[], script_pubkey);
        let sighash = calculate_transaction_sighash_single_input(
            &tx,
            0,
            script_code,
            prevout_values[0],
            SighashType::ALL,
            None,
        )
        .expect("sighash");
        let msg = Message::from_digest_slice(&sighash).expect("digest");
        let mut der = secp.sign_ecdsa(&msg, &secret).serialize_der().to_vec();
        der.push(SighashType::ALL.0);
        let mut script_sig = prefix.to_vec();
        script_sig.extend(serialize_push_data(&der));
        if script_pubkey.len() == 23 {
            script_sig.extend(serialize_push_data(script_code));
        }
        tx.inputs[0].script_sig = script_sig.clone();
        let refs: Vec<&[u8]> = prevout_scripts.iter().map(|s| s.as_slice()).collect();
        verify_script_with_context_full(
            &script_sig,
            script_pubkey,
            None,
            flags,
            &tx,
            0,
            &prevout_values,
            &refs,
            None,
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    };

    assert!(matches!(spend(&redeem, &redeem, &[OP_0], 0), Ok(true)));
    assert!(!matches!(
        spend(&redeem, &redeem, &[OP_RESERVED], 0),
        Ok(true)
    ));
    assert!(matches!(
        spend(&p2sh, &redeem, &[OP_0], flags::SCRIPT_VERIFY_P2SH),
        Ok(true)
    ));
    assert!(!matches!(
        spend(&p2sh, &redeem, &[OP_RESERVED], flags::SCRIPT_VERIFY_P2SH),
        Ok(true)
    ));
}

#[test]
#[allow(deprecated)]
fn verify_script_runs_the_p2sh_redeem_script() {
    let bad = vec![OP_RESERVED];
    let bad_hash = Ripemd160::digest(OptimizedSha256::new().hash(&bad));
    let mut bad_pubkey = vec![OP_HASH160, PUSH_20_BYTES];
    bad_pubkey.extend_from_slice(&bad_hash);
    bad_pubkey.push(OP_EQUAL);
    assert_eq!(
        verify_script(
            &serialize_push_data(&bad),
            &bad_pubkey,
            None,
            flags::SCRIPT_VERIFY_P2SH
        )
        .unwrap(),
        false
    );

    let ok = vec![OP_1];
    let ok_hash = Ripemd160::digest(OptimizedSha256::new().hash(&ok));
    let mut ok_pubkey = vec![OP_HASH160, PUSH_20_BYTES];
    ok_pubkey.extend_from_slice(&ok_hash);
    ok_pubkey.push(OP_EQUAL);
    assert_eq!(
        verify_script(
            &serialize_push_data(&ok),
            &ok_pubkey,
            None,
            flags::SCRIPT_VERIFY_P2SH
        )
        .unwrap(),
        true
    );
}

#[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
#[test]
fn hybrid_pubkey_is_accepted_without_strictenc() {
    use crate::transaction_hash::{SighashType, calculate_transaction_sighash_single_input};
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

    const PUBKEY_UNCOMPRESSED: u8 = 0x04;
    const PUBKEY_HYBRID_EVEN: u8 = 0x06;
    const PUBKEY_HYBRID_ODD: u8 = 0x07;

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[53u8; 32]).expect("secret");
    let mut uncompressed = PublicKey::from_secret_key(&secp, &secret).serialize_uncompressed();
    assert_eq!(uncompressed[0], PUBKEY_UNCOMPRESSED);
    let y_odd = uncompressed[64] & 1 == 1;
    let matching = if y_odd {
        PUBKEY_HYBRID_ODD
    } else {
        PUBKEY_HYBRID_EVEN
    };
    let mismatched = if y_odd {
        PUBKEY_HYBRID_EVEN
    } else {
        PUBKEY_HYBRID_ODD
    };

    let spend = |pubkey: &[u8], flags: u32| {
        let mut script_pubkey = vec![PUSH_65_BYTES];
        script_pubkey.extend_from_slice(pubkey);
        script_pubkey.push(OP_CHECKSIG);
        let (mut tx, prevout_values, prevout_scripts) =
            minimal_tx_and_prevouts(&[], &script_pubkey);
        let sighash = calculate_transaction_sighash_single_input(
            &tx,
            0,
            &script_pubkey,
            prevout_values[0],
            SighashType::ALL,
            None,
        )
        .expect("sighash");
        let msg = Message::from_digest_slice(&sighash).expect("digest");
        let mut der = secp.sign_ecdsa(&msg, &secret).serialize_der().to_vec();
        der.push(SighashType::ALL.0);
        let script_sig = serialize_push_data(&der);
        tx.inputs[0].script_sig = script_sig.clone();
        let refs: Vec<&[u8]> = prevout_scripts.iter().map(|s| s.as_slice()).collect();
        verify_script_with_context_full(
            &script_sig,
            &script_pubkey,
            None,
            flags,
            &tx,
            0,
            &prevout_values,
            &refs,
            None,
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    };

    let block_flags = flags::SCRIPT_VERIFY_P2SH
        | flags::SCRIPT_VERIFY_WITNESS
        | flags::SCRIPT_VERIFY_TAPROOT
        | flags::SCRIPT_VERIFY_DERSIG;
    uncompressed[0] = matching;
    assert!(matches!(spend(&uncompressed, 0), Ok(true)));
    assert!(matches!(spend(&uncompressed, block_flags), Ok(true)));
    assert!(!matches!(
        spend(&uncompressed, flags::SCRIPT_VERIFY_STRICTENC),
        Ok(true)
    ));
    uncompressed[0] = mismatched;
    assert!(!matches!(spend(&uncompressed, 0), Ok(true)));
    uncompressed[0] = PUBKEY_UNCOMPRESSED;
    assert!(matches!(spend(&uncompressed, 0), Ok(true)));
}

#[test]
fn discouraged_nop_uses_the_upgradable_nop_flag() {
    let discourage = flags::SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS;
    let reserved = [OP_NOP1, OP_NOP5, OP_NOP10, OP_CHECKTEMPLATEVERIFY];

    let eval = |script: &[u8], flags: u32| {
        let mut stack = Vec::new();
        eval_script(script, &mut stack, flags, SigVersion::Base)
    };
    let with_context = |opcode: u8, flags: u32| {
        let script_pubkey = vec![OP_1, opcode];
        let script_sig = Vec::new();
        let (tx, pv, psp) = minimal_tx_and_prevouts(&script_sig, &script_pubkey);
        let refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
        verify_script_with_context_full(
            &script_sig,
            &script_pubkey,
            None,
            flags,
            &tx,
            0,
            &pv,
            &refs,
            None,
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            #[cfg(feature = "production")]
            None,
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(feature = "production")]
            None,
            #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
            None,
        )
    };
    let is_discouraged = |result: Result<bool>| match result {
        Err(ConsensusError::ScriptErrorWithCode { code, .. }) => {
            code == ScriptErrorCode::DiscourageUpgradableNops
        }
        _ => false,
    };

    for opcode in reserved {
        let script = [OP_1, opcode];
        assert!(eval(&script, 0).unwrap());
        assert!(is_discouraged(eval(&script, discourage)));
        assert!(eval(&script, flags::SCRIPT_VERIFY_CONST_SCRIPTCODE).unwrap());
        assert!(eval(&script, flags::SCRIPT_VERIFY_WITNESS_PUBKEYTYPE).unwrap());
        assert!(with_context(opcode, 0).unwrap());
        assert!(is_discouraged(with_context(opcode, discourage)));
        assert!(with_context(opcode, flags::SCRIPT_VERIFY_CONST_SCRIPTCODE).unwrap());
    }

    assert!(eval(&[OP_1, OP_NOP], discourage).unwrap());
    assert!(eval(&[OP_1, OP_CHECKLOCKTIMEVERIFY], discourage).unwrap());
    assert!(eval(&[OP_0, OP_IF, OP_NOP1, OP_ENDIF, OP_1], discourage,).unwrap());
    assert!(with_context(OP_NOP, discourage).unwrap());
    assert!(with_context(OP_CHECKLOCKTIMEVERIFY, discourage).unwrap());
}

#[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
#[test]
fn nullfail_waits_until_the_multisig_fails() {
    use crate::transaction_hash::{SighashType, calculate_transaction_sighash_single_input};
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let signing = SecretKey::from_slice(&[53u8; 32]).expect("secret");
    let other = SecretKey::from_slice(&[54u8; 32]).expect("secret");
    let outsider = SecretKey::from_slice(&[55u8; 32]).expect("secret");
    let signing_pk = PublicKey::from_secret_key(&secp, &signing).serialize();
    let other_pk = PublicKey::from_secret_key(&secp, &other).serialize();

    let multisig = |prefix: &[u8], first: &[u8], second: &[u8]| {
        let mut script = prefix.to_vec();
        script.push(OP_1);
        script.push(PUSH_33_BYTES);
        script.extend_from_slice(first);
        script.push(PUSH_33_BYTES);
        script.extend_from_slice(second);
        script.push(OP_2);
        script.push(OP_CHECKMULTISIG);
        script
    };
    let spend = |script_pubkey: &[u8], secret: &SecretKey, flags: u32| {
        let (mut tx, prevout_values, prevout_scripts) = minimal_tx_and_prevouts(&[], script_pubkey);
        let sighash = calculate_transaction_sighash_single_input(
            &tx,
            0,
            script_pubkey,
            prevout_values[0],
            SighashType::ALL,
            None,
        )
        .expect("sighash");
        let msg = Message::from_digest_slice(&sighash).expect("digest");
        let mut der = secp.sign_ecdsa(&msg, secret).serialize_der().to_vec();
        der.push(SighashType::ALL.0);
        let mut script_sig = vec![OP_0];
        script_sig.extend(serialize_push_data(&der));
        tx.inputs[0].script_sig = script_sig.clone();
        let refs: Vec<&[u8]> = prevout_scripts.iter().map(|s| s.as_slice()).collect();
        verify_script_with_context_full(
            &script_sig,
            script_pubkey,
            None,
            flags,
            &tx,
            0,
            &prevout_values,
            &refs,
            None,
            None,
            crate::types::Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    };
    let nullfail_error = |result: Result<bool>| {
        matches!(
            result,
            Err(ConsensusError::ScriptErrorWithCode {
                code: ScriptErrorCode::SigNullFail,
                ..
            })
        )
    };

    let nullfail = flags::SCRIPT_VERIFY_NULLFAIL;
    // Leading OP_NOP skips the bare-multisig fast path. The interpreter tries the
    // last key first, so this signature misses once and then matches.
    let interpreter = multisig(&[OP_NOP], &signing_pk, &other_pk);
    assert!(matches!(spend(&interpreter, &signing, nullfail), Ok(true)));
    assert!(matches!(spend(&interpreter, &signing, 0), Ok(true)));
    assert!(nullfail_error(spend(&interpreter, &outsider, nullfail)));
    assert!(matches!(spend(&interpreter, &outsider, 0), Ok(false)));

    // Fast path tries keys in script order. The signature misses the first key.
    let fast_path = multisig(&[], &other_pk, &signing_pk);
    assert!(matches!(spend(&fast_path, &signing, nullfail), Ok(true)));
}

#[test]
fn test_verify_with_context_p2pkh_hash_mismatch() {
    // P2PKH pattern but pubkey hash does not match script_pubkey -> false (fast-path or interpreter).
    let pubkey = vec![0x02u8; 33]; // dummy compressed pubkey
    let sig = vec![0x30u8; 70]; // dummy sig (with sighash byte)
    let mut script_sig = Vec::new();
    script_sig.push(sig.len() as u8);
    script_sig.extend(&sig);
    script_sig.push(pubkey.len() as u8);
    script_sig.extend(&pubkey);

    let mut script_pubkey = vec![OP_DUP, OP_HASH160, PUSH_20_BYTES];
    script_pubkey.extend(&[0u8; 20]); // wrong hash (not HASH160(pubkey))
    script_pubkey.push(OP_EQUALVERIFY);
    script_pubkey.push(OP_CHECKSIG);

    let (tx, pv, psp) = minimal_tx_and_prevouts(&script_sig, &script_pubkey);
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
    let result = verify_script_with_context_full(
        &script_sig,
        &script_pubkey,
        None,
        0,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(500_000),
        None,
        crate::types::Network::Mainnet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_verify_with_context_p2sh_hash_mismatch() {
    // P2SH pattern but redeem script hash does not match -> false.
    let redeem = vec![OP_1, OP_1, OP_ADD]; // minimal redeem
    let mut script_sig = Vec::new();
    script_sig.push(redeem.len() as u8);
    script_sig.extend(&redeem);

    let mut script_pubkey = vec![OP_HASH160, PUSH_20_BYTES];
    script_pubkey.extend(&[0u8; 20]); // wrong hash (not HASH160(redeem))
    script_pubkey.push(OP_EQUAL);

    let (tx, pv, psp) = minimal_tx_and_prevouts(&script_sig, &script_pubkey);
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
    let result = verify_script_with_context_full(
        &script_sig,
        &script_pubkey,
        None,
        0x01, // P2SH
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(500_000),
        None,
        crate::types::Network::Mainnet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_verify_with_context_p2wpkh_wrong_witness_size() {
    // P2WPKH script_pubkey but witness has 1 element (need 2) -> false.
    let mut script_pubkey = vec![OP_0, PUSH_20_BYTES];
    script_pubkey.extend(&[0u8; 20]);
    let witness: Vec<Vec<u8>> = vec![vec![0x30; 70]]; // only sig, no pubkey
    let (tx, pv, psp) = minimal_tx_and_prevouts(&[], &script_pubkey);
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
    let empty: Vec<u8> = vec![];
    let result = verify_script_with_context_full(
        &empty,
        &script_pubkey,
        Some(&witness),
        0,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(500_000),
        None,
        crate::types::Network::Mainnet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_verify_with_context_p2wsh_wrong_witness_script_hash() {
    // P2WSH script_pubkey but SHA256(witness_script) != program -> false.
    let witness_script = vec![OP_1];
    let mut script_pubkey = vec![OP_0, PUSH_32_BYTES];
    script_pubkey.extend(&[0u8; 32]); // wrong hash (not SHA256(witness_script))
    let witness: Vec<Vec<u8>> = vec![witness_script];
    let (tx, pv, psp) = minimal_tx_and_prevouts(&[], &script_pubkey);
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
    let empty: Vec<u8> = vec![];
    let result = verify_script_with_context_full(
        &empty,
        &script_pubkey,
        Some(&witness),
        0,
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(500_000),
        None,
        crate::types::Network::Mainnet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
#[cfg(feature = "production")]
fn test_p2wsh_multisig_fast_path() {
    // P2WSH 2-of-2 multisig: fast path parses and validates; placeholder sigs fail -> Ok(false).
    use crate::constants::BIP147_ACTIVATION_MAINNET;
    use crate::crypto::OptimizedSha256;

    let pk1 = [0x02u8; PUSH_33_BYTES as usize];
    let pk2 = [0x03u8; PUSH_33_BYTES as usize];
    let mut witness_script = vec![OP_2, PUSH_33_BYTES];
    witness_script.extend_from_slice(&pk1);
    witness_script.push(PUSH_33_BYTES);
    witness_script.extend_from_slice(&pk2);
    witness_script.push(OP_2);
    witness_script.push(OP_CHECKMULTISIG);

    let wsh_hash = OptimizedSha256::new().hash(&witness_script);
    let mut script_pubkey = vec![OP_0, PUSH_32_BYTES];
    script_pubkey.extend_from_slice(&wsh_hash);

    let witness: Vec<Vec<u8>> = vec![
        vec![],           // zero-length NULLDUMMY
        vec![0x30u8; 72], // placeholder sig 1
        vec![0x30u8; 72], // placeholder sig 2
        witness_script.clone(),
    ];

    let (tx, pv, psp) = minimal_tx_and_prevouts(&[], &script_pubkey);
    let psp_refs: Vec<&[u8]> = psp.iter().map(|b| b.as_ref()).collect();
    let empty: Vec<u8> = vec![];
    let result = verify_script_with_context_full(
        &empty,
        &script_pubkey,
        Some(&witness),
        0x810, // SIGHASH_ALL | VERIFY_NULLDUMMY | VERIFY_NULLFAIL
        &tx,
        0,
        &pv,
        &psp_refs,
        Some(BIP147_ACTIVATION_MAINNET + 1),
        None,
        crate::types::Network::Mainnet,
        SigVersion::Base,
        #[cfg(feature = "production")]
        None,
        None, // precomputed_bip143
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(feature = "production")]
        None,
        #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
        None,
    );
    assert!(result.is_ok());
    assert!(!result.unwrap());
}
