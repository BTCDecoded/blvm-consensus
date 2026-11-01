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
use blvm_consensus::serialization::transaction::{
    serialize_transaction, serialize_transaction_with_witness,
};
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use blvm_consensus::witness::Witness;
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

fn blvm_accepts(
    tx: &Transaction,
    script_pubkey: &[u8],
    flags: u32,
    height: u64,
    witness: Option<&Witness>,
) -> bool {
    let prevout_values = [0i64];
    let prevouts: [&[u8]; 1] = [script_pubkey];
    #[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        witness,
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
        witness,
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
        witness,
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

fn library_accepts(
    tx: &Transaction,
    script_pubkey: &[u8],
    flags: u32,
    witness: Option<&[Witness]>,
) -> bool {
    let raw = match witness {
        Some(stacks) => serialize_transaction_with_witness(tx, stacks).expect("witness encoding"),
        None => serialize_transaction(tx),
    };
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
        let ours = blvm_accepts(&tx, &sample.script_pubkey, flags, height, None);
        let theirs = library_accepts(&tx, &sample.script_pubkey, flags, None);
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
fn nulldummy_follows_the_flag_below_activation() {
    // Same spend, one block before BIP147, with the flag forced on.
    // libbitcoinconsensus rejects from the flag alone. A second height
    // gate inside the opcode would still accept.
    let height = BIP147_ACTIVATION_MAINNET - 1;
    let flags = flags_for(FlagSource::Connect, height) | SCRIPT_VERIFY_NULLDUMMY;
    let tx = make_tx(&[0x51], 0xffff_ffff, 0);
    let script_pubkey = [0x00, 0x00, 0xae];
    let ours = blvm_accepts(&tx, &script_pubkey, flags, height, None);
    let theirs = library_accepts(&tx, &script_pubkey, flags, None);
    assert!(
        !theirs,
        "libbitcoinconsensus accepted a non-empty dummy with NULLDUMMY set"
    );
    assert_eq!(
        ours, theirs,
        "NULLDUMMY at height {height} flags {flags:#x}: ours={ours} library={theirs}"
    );
}

#[cfg(feature = "production")]
#[test]
fn nulldummy_fast_paths_follow_the_flag_below_activation() {
    use blvm_consensus::transaction_hash::{
        calculate_bip143_sighash, compute_legacy_sighash_nocache,
    };
    use ripemd::Ripemd160;
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};
    use sha2::{Digest, Sha256};

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[0x41; 32]).expect("key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret).serialize();
    let mut redeem = vec![0x51, 0x21];
    redeem.extend_from_slice(&pubkey);
    redeem.extend_from_slice(&[0x51, 0xae]);

    let height = BIP147_ACTIVATION_MAINNET - 1;
    let flags = flags_for(FlagSource::Connect, height) | SCRIPT_VERIFY_NULLDUMMY;
    let unsigned = make_tx(&[], 0xffff_ffff, 0);
    let legacy = compute_legacy_sighash_nocache(&unsigned, 0, &redeem, 0x01);
    let mut legacy_sig = secp
        .sign_ecdsa(
            &Message::from_digest_slice(&legacy).expect("digest"),
            &secret,
        )
        .serialize_der()
        .to_vec();
    legacy_sig.push(0x01);
    let witness_hash =
        calculate_bip143_sighash(&unsigned, 0, &redeem, 0, 0x01, None).expect("bip143");
    let mut witness_sig = secp
        .sign_ecdsa(
            &Message::from_digest_slice(&witness_hash).expect("digest"),
            &secret,
        )
        .serialize_der()
        .to_vec();
    witness_sig.push(0x01);

    let push = |data: &[u8]| {
        let mut out = Vec::with_capacity(data.len() + 1);
        out.push(data.len() as u8);
        out.extend_from_slice(data);
        out
    };
    let mut legacy_push = push(&legacy_sig);
    let redeem_push = push(&redeem);

    let mut bare_empty = vec![0x00];
    bare_empty.append(&mut legacy_push.clone());
    let mut bare_full = vec![0x51];
    bare_full.append(&mut legacy_push);

    let mut p2sh_empty = vec![0x00];
    p2sh_empty.extend_from_slice(&push(&legacy_sig));
    p2sh_empty.extend_from_slice(&redeem_push);
    let mut p2sh_full = vec![0x51];
    p2sh_full.extend_from_slice(&push(&legacy_sig));
    p2sh_full.extend_from_slice(&redeem_push);
    let hash160 = Ripemd160::digest(Sha256::digest(&redeem));
    let mut p2sh_program = vec![0xa9, 0x14];
    p2sh_program.extend_from_slice(&hash160);
    p2sh_program.push(0x87);

    let program = Sha256::digest(&redeem);
    let mut witness_program = vec![0x00, 0x20];
    witness_program.extend_from_slice(&program);
    let empty_stack: Witness = vec![vec![], witness_sig.clone(), redeem.clone()];
    let full_stack: Witness = vec![vec![0x01], witness_sig, redeem.clone()];

    let agree = |label: &str,
                 script_sig: &[u8],
                 script_pubkey: &[u8],
                 stack: Option<&Witness>,
                 expect_library: bool| {
        let tx = make_tx(script_sig, 0xffff_ffff, 0);
        let owned;
        let library_witness = match stack {
            Some(item) => {
                owned = [item.clone()];
                Some(&owned[..])
            }
            None => None,
        };
        let ours = blvm_accepts(&tx, script_pubkey, flags, height, stack);
        let theirs = library_accepts(&tx, script_pubkey, flags, library_witness);
        assert_eq!(
            theirs, expect_library,
            "{label}: library={theirs}, expected {expect_library}"
        );
        assert_eq!(ours, theirs, "{label}: ours={ours} library={theirs}");
    };

    agree("bare empty dummy", &bare_empty, &redeem, None, true);
    agree("bare non-empty dummy", &bare_full, &redeem, None, false);
    agree("p2sh empty dummy", &p2sh_empty, &p2sh_program, None, true);
    agree(
        "p2sh non-empty dummy",
        &p2sh_full,
        &p2sh_program,
        None,
        false,
    );
    agree(
        "witness empty dummy",
        &[],
        &witness_program,
        Some(&empty_stack),
        true,
    );
    agree(
        "witness non-empty dummy",
        &[],
        &witness_program,
        Some(&full_stack),
        false,
    );
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

#[cfg(feature = "production")]
#[test]
fn checksig_script_code_is_the_executing_subscript() {
    use blvm_consensus::transaction_hash::compute_legacy_sighash_nocache;
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let secret = SecretKey::from_slice(&[0x5a; 32]).expect("key");
    let pubkey = PublicKey::from_secret_key(&secp, &secret).serialize();
    let mut script_code = vec![0x21];
    script_code.extend_from_slice(&pubkey);
    script_code.push(0xac);

    let height = BIP66_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let unsigned = make_tx(&[], 0xffff_ffff, 0);
    let digest = compute_legacy_sighash_nocache(&unsigned, 0, &script_code, 0x01);
    let mut sig = secp
        .sign_ecdsa(
            &Message::from_digest_slice(&digest).expect("digest"),
            &secret,
        )
        .serialize_der()
        .to_vec();
    sig.push(0x01);
    let mut push_sig = vec![sig.len() as u8];
    push_sig.extend_from_slice(&sig);
    let mut push_key = vec![0x21];
    push_key.extend_from_slice(&pubkey);

    let mut in_script_sig = push_sig.clone();
    in_script_sig.extend_from_slice(&push_key);
    in_script_sig.push(0xac);
    let mut after_separator = push_sig.clone();
    after_separator.push(0xab);
    after_separator.extend_from_slice(&push_key);
    after_separator.push(0xac);

    let spend = |script_sig: &[u8], script_pubkey: &[u8]| {
        let tx = make_tx(script_sig, 0xffff_ffff, 0);
        (
            blvm_accepts(&tx, script_pubkey, flags, height, None),
            library_accepts(&tx, script_pubkey, flags, None),
        )
    };

    let (ours, theirs) = spend(&in_script_sig, &[]);
    assert!(
        theirs,
        "library rejected a scriptSig OP_CHECKSIG over its own subscript"
    );
    assert_eq!(
        ours, theirs,
        "scriptSig CHECKSIG: ours={ours} library={theirs}"
    );

    let (ours, theirs) = spend(&after_separator, &[]);
    assert!(
        theirs,
        "library rejected OP_CHECKSIG after OP_CODESEPARATOR"
    );
    assert_eq!(
        ours, theirs,
        "CODESEPARATOR subscript: ours={ours} library={theirs}"
    );

    let (ours, theirs) = spend(&push_sig, &script_code);
    assert!(
        theirs,
        "library rejected an ordinary pay-to-pubkey OP_CHECKSIG"
    );
    assert_eq!(
        ours, theirs,
        "scriptPubKey CHECKSIG: ours={ours} library={theirs}"
    );
}

#[cfg(feature = "production")]
#[test]
fn checkmultisig_matches_later_keys_in_order() {
    use blvm_consensus::transaction_hash::compute_legacy_sighash_nocache;
    use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let secret = |byte: u8| SecretKey::from_slice(&[byte; 32]).expect("key");
    let key_of = |byte: u8| PublicKey::from_secret_key(&secp, &secret(byte)).serialize();
    let pk1 = key_of(0x11);
    let pk2 = key_of(0x22);
    let pk3 = key_of(0x33);

    let mut script_pubkey = vec![0x52, 0x21];
    script_pubkey.extend_from_slice(&pk1);
    script_pubkey.push(0x21);
    script_pubkey.extend_from_slice(&pk2);
    script_pubkey.push(0x21);
    script_pubkey.extend_from_slice(&pk3);
    script_pubkey.extend_from_slice(&[0x53, 0xae]);

    let height = BIP66_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let unsigned = make_tx(&[], 0xffff_ffff, 0);
    let digest = compute_legacy_sighash_nocache(&unsigned, 0, &script_pubkey, 0x01);
    let sign = |byte: u8| {
        let mut sig = secp
            .sign_ecdsa(
                &Message::from_digest_slice(&digest).expect("digest"),
                &secret(byte),
            )
            .serialize_der()
            .to_vec();
        sig.push(0x01);
        let mut push = vec![sig.len() as u8];
        push.extend_from_slice(&sig);
        push
    };
    let mut script_sig = vec![0x00];
    script_sig.extend_from_slice(&sign(0x11));
    script_sig.extend_from_slice(&sign(0x33));

    let tx = make_tx(&script_sig, 0xffff_ffff, 0);
    let ours = blvm_accepts(&tx, &script_pubkey, flags, height, None);
    let theirs = library_accepts(&tx, &script_pubkey, flags, None);
    assert!(
        theirs,
        "library rejected sigs for the first and third keys of a 2-of-3"
    );
    assert_eq!(ours, theirs, "2-of-3 order: ours={ours} library={theirs}");
}

#[cfg(feature = "production")]
#[test]
fn tapscript_sighash_single_matches_libbitcoinconsensus() {
    use blvm_consensus::secp256k1_backend::taproot_output_key_with_parity;
    use blvm_consensus::taproot::{
        TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root,
        compute_tapscript_signature_hash,
    };
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let leaf_secret = SecretKey::from_slice(&[0x42; 32]).expect("leaf key");
    let leaf = Keypair::from_secret_key(&secp, &leaf_secret);
    let leaf_xonly = leaf.x_only_public_key().0.serialize();

    let mut tapscript = vec![0x20];
    tapscript.extend_from_slice(&leaf_xonly);
    tapscript.push(0xac);

    let internal = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
    let (output_key, parity) =
        taproot_output_key_with_parity(&internal, &merkle_root).expect("output key");
    let mut script_pubkey = vec![0x51, 0x20];
    script_pubkey.extend_from_slice(&output_key);
    let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control.extend_from_slice(&internal);

    let height = TAPROOT_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let tx = make_tx(&[], 0xffff_ffff, 0);
    let sighash = compute_tapscript_signature_hash(
        &tx,
        0,
        &[0],
        &[script_pubkey.as_slice()],
        &tapscript,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffff_ffff,
        0x03,
        None,
    )
    .expect("sighash");
    let mut sig = secp
        .sign_schnorr_no_aux_rand(
            &Message::from_digest_slice(&sighash).expect("digest"),
            &leaf,
        )
        .as_ref()
        .to_vec();
    sig.push(0x03);
    let witness = vec![sig, tapscript, control];

    let ours = blvm_accepts(&tx, &script_pubkey, flags, height, Some(&witness));
    let theirs = library_accepts(&tx, &script_pubkey, flags, Some(&[witness]));
    assert!(theirs, "library rejected a tapscript SIGHASH_SINGLE spend");
    assert_eq!(
        ours, theirs,
        "tapscript SIGHASH_SINGLE: ours={ours} library={theirs}"
    );
}

#[cfg(feature = "production")]
#[test]
fn tapscript_codesep_position_is_the_opcode_index() {
    use blvm_consensus::secp256k1_backend::taproot_output_key_with_parity;
    use blvm_consensus::taproot::{
        TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root,
        compute_tapscript_signature_hash,
    };
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let leaf_secret = SecretKey::from_slice(&[0x43; 32]).expect("leaf key");
    let leaf = Keypair::from_secret_key(&secp, &leaf_secret);
    let leaf_xonly = leaf.x_only_public_key().0.serialize();

    // One-byte push, DROP, CODESEPARATOR, then CHECKSIG.
    // The separator is opcode 2 and byte 3. The signature commits to opcode 2.
    let mut tapscript = vec![0x01, 0x11, 0x75, 0xab, 0x20];
    tapscript.extend_from_slice(&leaf_xonly);
    tapscript.push(0xac);

    let internal = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
    let (output_key, parity) =
        taproot_output_key_with_parity(&internal, &merkle_root).expect("output key");
    let mut script_pubkey = vec![0x51, 0x20];
    script_pubkey.extend_from_slice(&output_key);
    let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control.extend_from_slice(&internal);

    let height = TAPROOT_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let tx = make_tx(&[], 0xffff_ffff, 0);
    let sighash = compute_tapscript_signature_hash(
        &tx,
        0,
        &[0],
        &[script_pubkey.as_slice()],
        &tapscript,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        2,
        0x00,
        None,
    )
    .expect("sighash");
    let sig = secp
        .sign_schnorr_no_aux_rand(
            &Message::from_digest_slice(&sighash).expect("digest"),
            &leaf,
        )
        .as_ref()
        .to_vec();
    let witness = vec![sig, tapscript, control];

    let ours = blvm_accepts(&tx, &script_pubkey, flags, height, Some(&witness));
    let theirs = library_accepts(&tx, &script_pubkey, flags, Some(&[witness]));
    assert!(
        theirs,
        "library rejected a tapscript signature over codeseparator opcode index 2"
    );
    assert_eq!(
        ours, theirs,
        "tapscript codesep position: ours={ours} library={theirs}"
    );
}

#[cfg(feature = "production")]
#[test]
fn tapscript_sigop_budget_matches_libbitcoinconsensus() {
    use blvm_consensus::secp256k1_backend::taproot_output_key_with_parity;
    use blvm_consensus::taproot::{TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root};

    let internal = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let height = TAPROOT_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let tx = make_tx(&[], 0xffff_ffff, 0);

    let spend = |tapscript: &[u8], stack: &[Vec<u8>]| {
        let script = tapscript.to_vec();
        let merkle_root =
            compute_script_merkle_root(&script, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
        let (output_key, parity) =
            taproot_output_key_with_parity(&internal, &merkle_root).expect("output key");
        let mut script_pubkey = vec![0x51, 0x20];
        script_pubkey.extend_from_slice(&output_key);
        let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
        control.extend_from_slice(&internal);
        let mut witness = stack.to_vec();
        witness.push(tapscript.to_vec());
        witness.push(control);
        let ours = blvm_accepts(&tx, &script_pubkey, flags, height, Some(&witness));
        let theirs = library_accepts(&tx, &script_pubkey, flags, Some(&[witness]));
        (ours, theirs)
    };

    // One-byte pubkey is an upgradable tapscript key. One check stays inside the budget.
    let one_sig = vec![0x01];
    let one_key = vec![0x02];
    let (ours, theirs) = spend(&[0xac], &[one_sig.clone(), one_key.clone()]);
    assert!(theirs, "library rejected one upgradable tapscript CHECKSIG");
    assert_eq!(
        ours, theirs,
        "one tapscript sigop: ours={ours} library={theirs}"
    );

    // Two checks of one-byte signatures in this witness cost more than the budget.
    let (ours, theirs) = spend(
        &[0xac, 0x75, 0xac],
        &[one_sig.clone(), one_key.clone(), one_sig, one_key],
    );
    assert!(
        !theirs,
        "library accepted two tapscript checks past the validation weight"
    );
    assert_eq!(
        ours, theirs,
        "tapscript sigop budget: ours={ours} library={theirs}"
    );
}

#[cfg(feature = "production")]
#[test]
fn tapscript_op_success_skips_the_element_limit() {
    use blvm_consensus::secp256k1_backend::taproot_output_key_with_parity;
    use blvm_consensus::taproot::{TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root};

    let internal = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let tapscript = vec![0x50];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
    let (output_key, parity) =
        taproot_output_key_with_parity(&internal, &merkle_root).expect("output key");
    let mut script_pubkey = vec![0x51, 0x20];
    script_pubkey.extend_from_slice(&output_key);
    let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control.extend_from_slice(&internal);
    let witness = vec![vec![0x11; 521], tapscript, control];

    let height = TAPROOT_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let tx = make_tx(&[], 0xffff_ffff, 0);
    let ours = blvm_accepts(&tx, &script_pubkey, flags, height, Some(&witness));
    let theirs = library_accepts(&tx, &script_pubkey, flags, Some(&[witness]));
    assert!(
        theirs,
        "library rejected OP_SUCCESS with a 521-byte witness element"
    );
    assert_eq!(
        ours, theirs,
        "tapscript OP_SUCCESS element limit: ours={ours} library={theirs}"
    );
}

#[cfg(feature = "production")]
#[test]
fn tapscript_anyonecanpay_ignores_other_inputs() {
    use blvm_consensus::secp256k1_backend::taproot_output_key_with_parity;
    use blvm_consensus::taproot::{
        TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root,
        compute_tapscript_signature_hash,
    };
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let leaf_secret = SecretKey::from_slice(&[0x44; 32]).expect("leaf key");
    let leaf = Keypair::from_secret_key(&secp, &leaf_secret);
    let leaf_xonly = leaf.x_only_public_key().0.serialize();
    let mut tapscript = vec![0x20];
    tapscript.extend_from_slice(&leaf_xonly);
    tapscript.push(0xac);

    let internal = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[], TAPROOT_LEAF_VERSION_TAPSCRIPT).expect("root");
    let (output_key, parity) =
        taproot_output_key_with_parity(&internal, &merkle_root).expect("output key");
    let mut script_pubkey = vec![0x51, 0x20];
    script_pubkey.extend_from_slice(&output_key);
    let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control.extend_from_slice(&internal);

    let other_script = vec![0x51];
    let mut tx = Transaction {
        version: 2,
        inputs: tx_inputs![
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x11; 32],
                    index: 0,
                },
                script_sig: Vec::new(),
                sequence: 0xffff_fffe,
            },
            TransactionInput {
                prevout: OutPoint {
                    hash: [0x22; 32],
                    index: 1,
                },
                script_sig: Vec::new(),
                sequence: 7,
            },
        ],
        outputs: tx_outputs![TransactionOutput {
            value: 9_000,
            script_pubkey: vec![0x51],
        }],
        lock_time: 0,
    };
    let values = [50_000i64, 1];
    let prevouts: [&[u8]; 2] = [script_pubkey.as_slice(), other_script.as_slice()];
    let sighash = compute_tapscript_signature_hash(
        &tx,
        0,
        &values,
        &prevouts,
        &tapscript,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffff_ffff,
        0x81,
        None,
    )
    .expect("sighash");
    let mut sig = secp
        .sign_schnorr_no_aux_rand(
            &Message::from_digest_slice(&sighash).expect("digest"),
            &leaf,
        )
        .as_ref()
        .to_vec();
    sig.push(0x81);
    let witness = vec![sig, tapscript, control];

    let height = TAPROOT_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let agree = |tx: &Transaction| {
        let ours = accepts_first_input(
            tx,
            &script_pubkey,
            &other_script,
            &values,
            flags,
            height,
            &witness,
        );
        let theirs =
            library_first_input(tx, &script_pubkey, &other_script, &values, flags, &witness);
        (ours, theirs)
    };

    let (ours, theirs) = agree(&tx);
    assert!(
        theirs,
        "library rejected a tapscript ANYONECANPAY|ALL spend"
    );
    assert_eq!(
        ours, theirs,
        "tapscript ANYONECANPAY: ours={ours} library={theirs}"
    );

    tx.inputs[1].prevout.hash = [0x33; 32];
    tx.inputs[1].sequence = 9;
    let (ours, theirs) = agree(&tx);
    assert!(
        theirs,
        "library rejected ANYONECANPAY after the other input changed"
    );
    assert_eq!(
        ours, theirs,
        "other input changed: ours={ours} library={theirs}"
    );

    tx.outputs[0].value = 9_001;
    let (ours, theirs) = agree(&tx);
    assert!(
        !theirs,
        "library accepted ANYONECANPAY|ALL after the output changed"
    );
    assert_eq!(ours, theirs, "output changed: ours={ours} library={theirs}");
}

#[cfg(all(feature = "production", feature = "blvm-secp256k1"))]
fn accepts_first_input(
    tx: &Transaction,
    script_pubkey: &[u8],
    other_script: &[u8],
    values: &[i64],
    flags: u32,
    height: u64,
    witness: &Witness,
) -> bool {
    let prevouts = [script_pubkey, other_script];
    matches!(
        verify_script_with_context_full(
            &tx.inputs[0].script_sig,
            script_pubkey,
            Some(witness),
            flags,
            tx,
            0,
            values,
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
        ),
        Ok(true)
    )
}

#[cfg(all(feature = "production", not(feature = "blvm-secp256k1")))]
fn accepts_first_input(
    tx: &Transaction,
    script_pubkey: &[u8],
    other_script: &[u8],
    values: &[i64],
    flags: u32,
    height: u64,
    witness: &Witness,
) -> bool {
    let prevouts = [script_pubkey, other_script];
    matches!(
        verify_script_with_context_full(
            &tx.inputs[0].script_sig,
            script_pubkey,
            Some(witness),
            flags,
            tx,
            0,
            values,
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
        ),
        Ok(true)
    )
}

fn library_first_input(
    tx: &Transaction,
    script_pubkey: &[u8],
    other_script: &[u8],
    values: &[i64],
    flags: u32,
    witness: &Witness,
) -> bool {
    let raw = serialize_transaction_with_witness(tx, &[witness.clone(), Vec::new()])
        .expect("witness encoding");
    let spent = [
        Utxo {
            script_pubkey: script_pubkey.as_ptr(),
            script_pubkey_len: script_pubkey.len() as u32,
            value: values[0],
        },
        Utxo {
            script_pubkey: other_script.as_ptr(),
            script_pubkey_len: other_script.len() as u32,
            value: values[1],
        },
    ];
    match verify_with_flags(
        script_pubkey,
        values[0] as u64,
        &raw,
        Some(&spent),
        0,
        flags,
    ) {
        Ok(()) => true,
        Err(Error::ERR_SCRIPT) => false,
        Err(err) => panic!("libbitcoinconsensus api error: {err}"),
    }
}

#[cfg(feature = "production")]
#[test]
fn tapscript_merkle_branch_matches_libbitcoinconsensus() {
    use blvm_consensus::secp256k1_backend::{tap_leaf_hash, taproot_output_key_with_parity};
    use blvm_consensus::taproot::{
        TAPROOT_LEAF_VERSION_TAPSCRIPT, compute_script_merkle_root,
        compute_tapscript_signature_hash,
    };
    use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

    let secp = Secp256k1::new();
    let leaf_secret = SecretKey::from_slice(&[0x45; 32]).expect("leaf key");
    let leaf = Keypair::from_secret_key(&secp, &leaf_secret);
    let leaf_xonly = leaf.x_only_public_key().0.serialize();
    let mut tapscript = vec![0x20];
    tapscript.extend_from_slice(&leaf_xonly);
    tapscript.push(0xac);

    let sibling = vec![0x51];
    let sibling_hash = tap_leaf_hash(TAPROOT_LEAF_VERSION_TAPSCRIPT, &sibling);
    let internal = [
        0x79, 0xbe, 0x66, 0x7e, 0xf9, 0xdc, 0xbb, 0xac, 0x55, 0xa0, 0x62, 0x95, 0xce, 0x87, 0x0b,
        0x07, 0x02, 0x9b, 0xfc, 0xdb, 0x2d, 0xce, 0x28, 0xd9, 0x59, 0xf2, 0x81, 0x5b, 0x16, 0xf8,
        0x17, 0x98,
    ];
    let merkle_root =
        compute_script_merkle_root(&tapscript, &[sibling_hash], TAPROOT_LEAF_VERSION_TAPSCRIPT)
            .expect("root");
    let (output_key, parity) =
        taproot_output_key_with_parity(&internal, &merkle_root).expect("output key");
    let mut script_pubkey = vec![0x51, 0x20];
    script_pubkey.extend_from_slice(&output_key);
    let mut control = vec![TAPROOT_LEAF_VERSION_TAPSCRIPT | parity];
    control.extend_from_slice(&internal);
    control.extend_from_slice(&sibling_hash);

    let height = TAPROOT_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let tx = make_tx(&[], 0xffff_ffff, 0);
    let sighash = compute_tapscript_signature_hash(
        &tx,
        0,
        &[0],
        &[script_pubkey.as_slice()],
        &tapscript,
        TAPROOT_LEAF_VERSION_TAPSCRIPT,
        0xffff_ffff,
        0x00,
        None,
    )
    .expect("sighash");
    let sig = secp
        .sign_schnorr_no_aux_rand(
            &Message::from_digest_slice(&sighash).expect("digest"),
            &leaf,
        )
        .as_ref()
        .to_vec();
    let witness = vec![sig, tapscript, control];
    let ours = blvm_accepts(&tx, &script_pubkey, flags, height, Some(&witness));
    let theirs = library_accepts(&tx, &script_pubkey, flags, Some(&[witness]));
    assert!(
        theirs,
        "library rejected a tapscript spend with one merkle branch"
    );
    assert_eq!(
        ours, theirs,
        "tapscript merkle branch: ours={ours} library={theirs}"
    );
}

#[test]
fn verif_fails_in_both_branches() {
    let height = BIP66_ACTIVATION_MAINNET;
    let flags = flags_for(FlagSource::Connect, height);
    let tx = make_tx(&[], 0xffff_ffff, 0);
    // OP_0 OP_IF OP_VERIF OP_ENDIF OP_1, and the same with the true branch taken.
    for (script_pubkey, label) in [
        (vec![0x00, 0x63, 0x65, 0x68, 0x51], "false branch"),
        (vec![0x51, 0x63, 0x65, 0x68, 0x51], "true branch"),
    ] {
        let ours = blvm_accepts(&tx, &script_pubkey, flags, height, None);
        let theirs = library_accepts(&tx, &script_pubkey, flags, None);
        assert!(!theirs, "library accepted OP_VERIF in the {label}");
        assert_eq!(
            ours, theirs,
            "OP_VERIF {label}: ours={ours} library={theirs}"
        );
    }
}
