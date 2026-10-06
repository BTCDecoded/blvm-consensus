//! Script verdicts against `libbitcoinconsensus` at mainnet activation heights.
//!
//! Each sample spends one script the new rule changes, at height−1, the
//! activation block, and height+1. The flag word and the block height passed
//! into `verify_script_with_context_full` are the pair for that block.
//!
//! Two flag sources:
//! - `Connect` is `get_block_script_verify_flags_core` (what `connect_block`
//!   passes). P2SH, witness, and taproot stay on from genesis except two
//!   exception block hashes; DERSIG, CLTV, CSV, and NULLDUMMY flip here.
//! - `HeightFaithful` is the activation table. P2SH, witness, and taproot
//!   flip here, because the connect mask does not turn them off before
//!   their heights. Taproot enforcement in this crate is the height
//!   (`taproot_active_at_height`), so the taproot bit is set only once that
//!   height is reached — the same condition the interpreter uses.
//!
//! Not scored: BIP30, BIP34, halvings, BIP113 median time, NULLFAIL. Those
//! are block-connect rules. `libbitcoinconsensus` does not see them.

use bitcoinconsensus::{Error, Utxo, verify_with_flags};
use blvm_consensus::activation::ForkActivationTable;
use blvm_consensus::block::{
    calculate_base_script_flags_for_block_network, get_block_script_verify_flags_core,
};
use blvm_consensus::constants::{
    BIP16_P2SH_ACTIVATION_MAINNET, BIP65_ACTIVATION_MAINNET, BIP66_ACTIVATION_MAINNET,
    BIP112_CSV_ACTIVATION_MAINNET, BIP147_ACTIVATION_MAINNET, SEGWIT_ACTIVATION_MAINNET,
    TAPROOT_ACTIVATION_MAINNET,
};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY, SCRIPT_VERIFY_CHECKSEQUENCEVERIFY, SCRIPT_VERIFY_DERSIG,
    SCRIPT_VERIFY_NULLDUMMY, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_TAPROOT, SCRIPT_VERIFY_WITNESS,
};
use blvm_consensus::script::{SigVersion, verify_script_with_context_full};
use blvm_consensus::serialization::transaction::serialize_transaction;
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use blvm_consensus::{tx_inputs, tx_outputs};

/// Bits both this interpreter and `libbitcoinconsensus` 0.106 honor.
/// Drops feature-gated bits (CTV) that the C library does not know.
const COMPARED_FLAGS: u32 = SCRIPT_VERIFY_P2SH
    | SCRIPT_VERIFY_DERSIG
    | SCRIPT_VERIFY_NULLDUMMY
    | SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY
    | SCRIPT_VERIFY_CHECKSEQUENCEVERIFY
    | SCRIPT_VERIFY_WITNESS
    | SCRIPT_VERIFY_TAPROOT;

#[derive(Clone, Copy)]
enum FlagSource {
    Connect,
    /// `witness`: OR `SCRIPT_VERIFY_WITNESS` once SegWit is active.
    HeightFaithful {
        witness: bool,
    },
}

struct Sample {
    name: &'static str,
    height: u64,
    bit: u32,
    source: FlagSource,
    script_sig: Vec<u8>,
    script_pubkey: Vec<u8>,
    sequence: u64,
    lock_time: u64,
}

fn flags_for(source: FlagSource, height: u64) -> u32 {
    let flags = match source {
        FlagSource::Connect => {
            let table = ForkActivationTable::from_network(Network::Mainnet);
            // Not a script-flag exception hash.
            let block_hash = [0x11u8; 32];
            get_block_script_verify_flags_core(&block_hash, height, &table, Network::Mainnet)
        }
        FlagSource::HeightFaithful { witness } => {
            let mut flags = calculate_base_script_flags_for_block_network(height, Network::Mainnet);
            if witness && height >= SEGWIT_ACTIVATION_MAINNET {
                flags |= SCRIPT_VERIFY_WITNESS;
            }
            if height >= TAPROOT_ACTIVATION_MAINNET {
                flags |= SCRIPT_VERIFY_TAPROOT;
            }
            flags
        }
    };
    flags & COMPARED_FLAGS
}

fn make_tx(script_sig: &[u8], sequence: u64, lock_time: u64) -> Transaction {
    Transaction {
        version: 1,
        inputs: tx_inputs![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32],
                index: 0,
            },
            script_sig: script_sig.to_vec(),
            sequence,
        }],
        outputs: tx_outputs![TransactionOutput {
            value: 0,
            script_pubkey: Vec::new(),
        }],
        lock_time,
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
    // Taproot flags require the spent output even when this input is not a
    // taproot program. `connect_block` sets that flag from genesis.
    let spent = [Utxo {
        script_pubkey: script_pubkey.as_ptr(),
        script_pubkey_len: script_pubkey.len() as u32,
        value: 0,
    }];
    match verify_with_flags(script_pubkey, 0, &raw, Some(&spent), 0, flags) {
        Ok(()) => true,
        // Script failure. The C API leaves the error at ERR_OK / ERR_SCRIPT.
        Err(Error::ERR_SCRIPT) => false,
        Err(err) => panic!("libbitcoinconsensus api error: {err}"),
    }
}

fn assert_boundary(sample: &Sample) {
    let heights = [sample.height - 1, sample.height, sample.height + 1];
    let mut rows = Vec::with_capacity(3);
    for height in heights {
        let flags = flags_for(sample.source, height);
        let tx = make_tx(&sample.script_sig, sample.sequence, sample.lock_time);
        let ours = blvm_accepts(&tx, &sample.script_pubkey, flags, height);
        let theirs = library_accepts(&tx, &sample.script_pubkey, flags);
        assert_eq!(
            ours, theirs,
            "{} height {height}: blvm={ours} libbitcoinconsensus={theirs} flags={flags:#x}",
            sample.name
        );
        rows.push((height, flags, ours));
    }

    let before_bit = rows[0].1 & sample.bit;
    let at_bit = rows[1].1 & sample.bit;
    let after_bit = rows[2].1 & sample.bit;
    assert_eq!(
        before_bit, 0,
        "{}: flag {:#x} set at height {}",
        sample.name, sample.bit, rows[0].0
    );
    assert_ne!(
        at_bit, 0,
        "{}: flag {:#x} clear at activation height {}",
        sample.name, sample.bit, rows[1].0
    );
    assert_ne!(
        after_bit, 0,
        "{}: flag {:#x} clear at height {}",
        sample.name, sample.bit, rows[2].0
    );

    assert!(
        rows[0].2,
        "{}: expected accept at height {} (before the rule)",
        sample.name, rows[0].0
    );
    assert!(
        !rows[1].2,
        "{}: expected reject at activation height {}",
        sample.name, rows[1].0
    );
    assert!(
        !rows[2].2,
        "{}: expected reject at height {} (after the rule)",
        sample.name, rows[2].0
    );
}

/// P2SH redeem is `OP_0`. The hash check passes; executing the redeem does not.
fn p2sh_false_redeem() -> (Vec<u8>, Vec<u8>) {
    // hash160(0x00) = 9f7fd096d37ed2c0e3f7f0cfc924beef4ffceb68
    let mut script_pubkey = vec![0xa9, 0x14];
    script_pubkey.extend_from_slice(&hex_20("9f7fd096d37ed2c0e3f7f0cfc924beef4ffceb68"));
    script_pubkey.push(0x87); // OP_EQUAL
    (vec![0x01, 0x00], script_pubkey)
}

fn hex_20(s: &str) -> [u8; 20] {
    let mut out = [0u8; 20];
    for i in 0..20 {
        out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
    }
    out
}

#[test]
fn p2sh_boundary_matches_libbitcoinconsensus() {
    let (script_sig, script_pubkey) = p2sh_false_redeem();
    assert_boundary(&Sample {
        name: "BIP16 P2SH",
        height: BIP16_P2SH_ACTIVATION_MAINNET,
        bit: SCRIPT_VERIFY_P2SH,
        source: FlagSource::HeightFaithful { witness: false },
        script_sig,
        script_pubkey,
        sequence: 0xffff_ffff,
        lock_time: 0,
    });
}

#[test]
fn dersig_boundary_matches_libbitcoinconsensus() {
    // Non-DER signature. Without DERSIG, CHECKSIG is false and OP_NOT succeeds.
    // With DERSIG the non-DER encoding aborts before OP_NOT.
    assert_boundary(&Sample {
        name: "BIP66 DERSIG",
        height: BIP66_ACTIVATION_MAINNET,
        bit: SCRIPT_VERIFY_DERSIG,
        source: FlagSource::Connect,
        script_sig: vec![0x03, 0x30, 0x01, 0x01, 0x01, 0x02],
        script_pubkey: vec![0xac, 0x91],
        sequence: 0xffff_ffff,
        lock_time: 0,
    });
}

#[test]
fn cltv_boundary_matches_libbitcoinconsensus() {
    // Locktime 1 on the stack, tx lock_time 0. NOP2 before BIP65; CLTV fails after.
    assert_boundary(&Sample {
        name: "BIP65 CLTV",
        height: BIP65_ACTIVATION_MAINNET,
        bit: SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY,
        source: FlagSource::Connect,
        script_sig: Vec::new(),
        script_pubkey: vec![0x51, 0xb1, 0x75, 0x51],
        sequence: 0xffff_ffff,
        lock_time: 0,
    });
}

#[test]
fn csv_boundary_matches_libbitcoinconsensus() {
    // Relative lock 1, input sequence disabled. NOP3 before CSV; CSV fails after.
    assert_boundary(&Sample {
        name: "BIP112 CSV",
        height: BIP112_CSV_ACTIVATION_MAINNET,
        bit: SCRIPT_VERIFY_CHECKSEQUENCEVERIFY,
        source: FlagSource::Connect,
        script_sig: Vec::new(),
        script_pubkey: vec![0x51, 0xb2, 0x75, 0x51],
        sequence: 0xffff_ffff,
        lock_time: 0,
    });
}

#[test]
fn nulldummy_boundary_matches_libbitcoinconsensus() {
    // 0-of-0 CHECKMULTISIG with a non-zero dummy. Allowed before BIP147.
    assert_boundary(&Sample {
        name: "BIP147 NULLDUMMY",
        height: BIP147_ACTIVATION_MAINNET,
        bit: SCRIPT_VERIFY_NULLDUMMY,
        source: FlagSource::Connect,
        script_sig: vec![0x51],
        script_pubkey: vec![0x00, 0x00, 0xae],
        sequence: 0xffff_ffff,
        lock_time: 0,
    });
}

#[test]
fn segwit_boundary_matches_libbitcoinconsensus() {
    // v0 program, empty witness. Bare script succeeds before the witness flag;
    // empty witness fails once the flag is on. The program bytes are non-zero
    // so the pre-witness stack top is true.
    let mut script_pubkey = vec![0x00, 0x14];
    script_pubkey.extend(std::iter::repeat(0x11).take(20));
    assert_boundary(&Sample {
        name: "SegWit witness",
        height: SEGWIT_ACTIVATION_MAINNET,
        bit: SCRIPT_VERIFY_WITNESS,
        source: FlagSource::HeightFaithful { witness: true },
        script_sig: Vec::new(),
        script_pubkey,
        sequence: 0xffff_ffff,
        lock_time: 0,
    });
}

#[test]
fn taproot_boundary_matches_libbitcoinconsensus() {
    // v1 32-byte program, empty witness. Anyone-can-spend until taproot;
    // empty witness fails at and after the height. Program bytes are non-zero.
    let mut script_pubkey = vec![0x51, 0x20];
    script_pubkey.extend(std::iter::repeat(0x02).take(32));
    assert_boundary(&Sample {
        name: "Taproot",
        height: TAPROOT_ACTIVATION_MAINNET,
        bit: SCRIPT_VERIFY_TAPROOT,
        source: FlagSource::HeightFaithful { witness: true },
        script_sig: Vec::new(),
        script_pubkey,
        sequence: 0xffff_ffff,
        lock_time: 0,
    });
}
