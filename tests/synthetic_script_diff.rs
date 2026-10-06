//! Off-chain script verdicts against `libbitcoinconsensus`.
//!
//! The chain differential only scores scripts that were mined. These cases
//! are scripts the chain does not show. The same flag integer is passed to
//! both sides. Height is what turns Taproot on in this interpreter; the
//! library uses `SCRIPT_VERIFY_TAPROOT`, so a pre-taproot height is paired
//! with the flag clear and a post-taproot height with the flag set.
//!
//! Activation boundaries (the block before, the activation block, and the
//! block after) live in `height_script_differential.rs`. The heights here
//! are only "taproot off" and "taproot on".

use bitcoinconsensus::{
    Error, Utxo, VERIFY_DERSIG, VERIFY_P2SH, VERIFY_TAPROOT, VERIFY_WITNESS, verify_with_flags,
};
use blvm_consensus::crypto::OptimizedSha256;
use blvm_consensus::opcodes::{OP_EQUAL, OP_HASH160};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_TAPROOT, SCRIPT_VERIFY_WITNESS,
};
use blvm_consensus::script::{SigVersion, verify_script_with_context_full};
use blvm_consensus::serialization::transaction::serialize_transaction;
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use blvm_consensus::{tx_inputs, tx_outputs};
use ripemd::{Digest, Ripemd160};

/// Before mainnet Taproot (709632). Witness and DERSIG are flags, not this height.
const PRE_TAPROOT_HEIGHT: u64 = 400_000;
/// After mainnet Taproot.
const TAPROOT_HEIGHT: u64 = 800_000;

struct Case {
    name: String,
    script_sig: Vec<u8>,
    script_pubkey: Vec<u8>,
    flags: u32,
    height: u64,
}

fn consensus_flags(taproot: bool) -> u32 {
    let mut flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_DERSIG;
    if taproot {
        flags |= SCRIPT_VERIFY_TAPROOT;
    }
    flags
}

fn p2sh_of(redeem: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let hash = Ripemd160::digest(OptimizedSha256::new().hash(redeem));
    let mut script_pubkey = vec![OP_HASH160, 0x14];
    script_pubkey.extend_from_slice(&hash);
    script_pubkey.push(OP_EQUAL);
    let mut script_sig = Vec::with_capacity(redeem.len() + 1);
    script_sig.push(redeem.len() as u8);
    script_sig.extend_from_slice(redeem);
    (script_sig, script_pubkey)
}

fn witness_program(version: u8, payload: &[u8]) -> Vec<u8> {
    let mut script = Vec::with_capacity(payload.len() + 2);
    script.push(version);
    script.push(payload.len() as u8);
    script.extend_from_slice(payload);
    script
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let flags = consensus_flags(false);
    let payloads: &[(&[u8], &str)] = &[
        (&[0x00, 0x00], "zeros"),
        (&[0x00, 0x80], "negative-zero"),
        (&[0x01, 0x00], "nonzero"),
    ];

    for version in 0x51u8..=0x60 {
        for &(payload, label) in payloads {
            let script_pubkey = witness_program(version, payload);
            out.push(Case {
                name: format!("native v{version:#x} {label} len{}", payload.len()),
                script_sig: Vec::new(),
                script_pubkey,
                flags,
                height: PRE_TAPROOT_HEIGHT,
            });
        }
    }

    let wide_zero = vec![0u8; 32];
    let mut wide_neg = vec![0u8; 32];
    wide_neg[31] = 0x80;
    let mut wide_one = vec![0u8; 32];
    wide_one[0] = 1;
    for (payload, label) in [
        (&wide_zero[..], "zeros"),
        (&wide_neg[..], "negative-zero"),
        (&wide_one[..], "nonzero"),
    ] {
        let script_pubkey = witness_program(0x51, payload);
        out.push(Case {
            name: format!("v1-32 {label} before taproot"),
            script_sig: Vec::new(),
            script_pubkey: script_pubkey.clone(),
            flags,
            height: PRE_TAPROOT_HEIGHT,
        });
        out.push(Case {
            name: format!("v1-32 {label} at taproot"),
            script_sig: Vec::new(),
            script_pubkey,
            flags: consensus_flags(true),
            height: TAPROOT_HEIGHT,
        });
    }

    // Version 0 of the wrong length is invalid. A 2-byte program is not P2WPKH.
    for payload in [&[0x00, 0x00][..], &[0x11, 0x22][..]] {
        out.push(Case {
            name: format!("v0 short {payload:02x?}"),
            script_sig: Vec::new(),
            script_pubkey: witness_program(0x00, payload),
            flags,
            height: PRE_TAPROOT_HEIGHT,
        });
    }

    for redeem in [
        vec![0x52, 0x02, 0x00, 0x00],
        vec![0x52, 0x02, 0x00, 0x80],
        vec![0x52, 0x02, 0x01, 0x00],
    ] {
        let (script_sig, script_pubkey) = p2sh_of(&redeem);
        out.push(Case {
            name: format!("p2sh redeem {redeem:02x?}"),
            script_sig,
            script_pubkey,
            flags,
            height: PRE_TAPROOT_HEIGHT,
        });
        let mut pushdata1 = vec![0x4c, redeem.len() as u8];
        pushdata1.extend_from_slice(&redeem);
        let (_, script_pubkey) = p2sh_of(&redeem);
        out.push(Case {
            name: format!("p2sh pushdata1 {redeem:02x?}"),
            script_sig: pushdata1,
            script_pubkey,
            flags,
            height: PRE_TAPROOT_HEIGHT,
        });
    }

    // A false result is an empty vector. OP_SIZE of that result is 0.
    out.push(Case {
        name: "false result has size zero".to_string(),
        script_sig: Vec::new(),
        script_pubkey: vec![0x51, 0x52, 0x87, 0x82, 0x00, 0x87],
        flags,
        height: PRE_TAPROOT_HEIGHT,
    });

    // Non-DER aborts, so OP_NOT never runs. An empty signature does not abort.
    let der_flags = consensus_flags(false);
    out.push(Case {
        name: "non-der checksig then not".to_string(),
        script_sig: vec![0x03, 0x30, 0x01, 0x01, 0x01, 0x02],
        script_pubkey: vec![0xac, 0x91],
        flags: der_flags,
        height: PRE_TAPROOT_HEIGHT,
    });
    out.push(Case {
        name: "empty checksig then not under dersig".to_string(),
        script_sig: vec![0x00, 0x01, 0x02],
        script_pubkey: vec![0xac, 0x91],
        flags: der_flags,
        height: PRE_TAPROOT_HEIGHT,
    });
    out.push(Case {
        name: "non-der checksig then not without dersig".to_string(),
        script_sig: vec![0x03, 0x30, 0x01, 0x01, 0x01, 0x02],
        script_pubkey: vec![0xac, 0x91],
        flags: SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS,
        height: PRE_TAPROOT_HEIGHT,
    });

    // Pubkeys executed by CHECKMULTISIG count toward the 201-opcode limit.
    let mut within = vec![0x61; 160];
    within.extend_from_slice(&[0x00, 0x00]);
    within.extend(std::iter::repeat(0x00).take(20));
    within.extend_from_slice(&[0x01, 20, 0xae]);
    out.push(Case {
        name: "multisig pubkey count within limit".to_string(),
        script_sig: Vec::new(),
        script_pubkey: within,
        flags,
        height: PRE_TAPROOT_HEIGHT,
    });
    let mut over = vec![0x61; 181];
    over.extend_from_slice(&[0x00, 0x00]);
    over.extend(std::iter::repeat(0x00).take(20));
    over.extend_from_slice(&[0x01, 20, 0xae]);
    out.push(Case {
        name: "multisig pubkey count over limit".to_string(),
        script_sig: Vec::new(),
        script_pubkey: over,
        flags,
        height: PRE_TAPROOT_HEIGHT,
    });

    out
}

fn make_tx(script_sig: &[u8]) -> Transaction {
    Transaction {
        version: 1,
        inputs: tx_inputs![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32],
                index: 0,
            },
            script_sig: script_sig.to_vec(),
            sequence: 0xffff_ffff,
        }],
        outputs: tx_outputs![TransactionOutput {
            value: 0,
            script_pubkey: Vec::new(),
        }],
        lock_time: 0,
    }
}

fn blvm_accepts(tx: &Transaction, script_pubkey: &[u8], flags: u32, height: u64) -> bool {
    let prevout_values = [0i64];
    let prevouts: [&[u8]; 1] = [script_pubkey];
    #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        None,
        flags,
        tx,
        0,
        &prevout_values,
        &prevouts,
        Some(height),
        None,
        Network::Mainnet,
        SigVersion::Base,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    #[cfg(all(feature = "production", not(feature = "blvm-secp256k1")))]
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        None,
        flags,
        tx,
        0,
        &prevout_values,
        &prevouts,
        Some(height),
        None,
        Network::Mainnet,
        SigVersion::Base,
        None,
        None,
        None,
        None,
        None,
    );
    #[cfg(not(feature = "production"))]
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        None,
        flags,
        tx,
        0,
        &prevout_values,
        &prevouts,
        Some(height),
        None,
        Network::Mainnet,
        SigVersion::Base,
        None,
    );
    matches!(result, Ok(true))
}

fn library_accepts(tx: &Transaction, script_pubkey: &[u8], flags: u32) -> bool {
    let raw = serialize_transaction(tx);
    let spent = [Utxo {
        script_pubkey: script_pubkey.as_ptr(),
        script_pubkey_len: script_pubkey.len() as u32,
        value: 0,
    }];
    match verify_with_flags(script_pubkey, 0, &raw, Some(&spent), 0, flags) {
        Ok(()) => true,
        Err(Error::ERR_SCRIPT) => false,
        Err(err) => panic!("libbitcoinconsensus api error: {err}"),
    }
}

#[test]
fn synthetic_scripts_match_libbitcoinconsensus() {
    assert_eq!(SCRIPT_VERIFY_P2SH, VERIFY_P2SH);
    assert_eq!(SCRIPT_VERIFY_WITNESS, VERIFY_WITNESS);
    assert_eq!(SCRIPT_VERIFY_DERSIG, VERIFY_DERSIG);
    assert_eq!(SCRIPT_VERIFY_TAPROOT, VERIFY_TAPROOT);

    let cases = cases();
    assert!(
        cases.len() > 50,
        "the sweep shrank to {} cases",
        cases.len()
    );

    let mut mismatches = Vec::new();
    for case in &cases {
        let tx = make_tx(&case.script_sig);
        let ours = blvm_accepts(&tx, &case.script_pubkey, case.flags, case.height);
        let theirs = library_accepts(&tx, &case.script_pubkey, case.flags);
        if ours != theirs {
            mismatches.push(format!(
                "{}: blvm={ours} libbitcoinconsensus={theirs} flags={:#x} height={}",
                case.name, case.flags, case.height
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} script verdict mismatch(es):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
