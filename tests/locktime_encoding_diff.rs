//! CLTV comparison, CSV comparison, and strict-DER encoding against libbitcoinconsensus.
//!
//! The activation-height bits are locked in `height_script_differential`. These
//! cases are the comparisons those bits turn on.

use bitcoinconsensus::{Error, Utxo, verify_with_flags};
use blvm_consensus::constants::{
    BIP65_ACTIVATION_MAINNET, BIP66_ACTIVATION_MAINNET, BIP112_CSV_ACTIVATION_MAINNET,
};
use blvm_consensus::opcodes::{OP_CHECKSIG, PUSH_33_BYTES};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY, SCRIPT_VERIFY_CHECKSEQUENCEVERIFY, SCRIPT_VERIFY_DERSIG,
    SCRIPT_VERIFY_P2SH,
};
use blvm_consensus::script::{SigVersion, verify_script_with_context_full};
use blvm_consensus::serialization::transaction::serialize_transaction;
use blvm_consensus::transaction_hash::compute_legacy_sighash_nocache;
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

const N: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B, 0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41,
];

fn script_num(n: i64) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let mut abs = n.unsigned_abs();
    let mut out = Vec::new();
    while abs > 0 {
        out.push((abs & 0xff) as u8);
        abs >>= 8;
    }
    if out.last().copied().unwrap() & 0x80 != 0 {
        out.push(if n < 0 { 0x80 } else { 0x00 });
    } else if n < 0 {
        *out.last_mut().unwrap() |= 0x80;
    }
    out
}

fn push_num(n: i64) -> Vec<u8> {
    let body = script_num(n);
    let mut out = vec![body.len() as u8];
    out.extend(body);
    out
}

fn make_tx(version: u64, sequence: u64, lock_time: u64, script_sig: Vec<u8>) -> Transaction {
    Transaction {
        version,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32],
                index: 0,
            },
            script_sig: script_sig.into(),
            sequence,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: Vec::new(),
        }]
        .into(),
        lock_time,
    }
}

fn blvm_accepts(tx: &Transaction, script_pubkey: &[u8], flags: u32, height: u64) -> bool {
    let prevout_values = [0i64];
    let prevouts: [&[u8]; 1] = [script_pubkey];
    matches!(
        verify_script_with_context_full(
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
        ),
        Ok(true)
    )
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

fn expect(name: &str, tx: &Transaction, script_pubkey: &[u8], flags: u32, height: u64, want: bool) {
    let ours = blvm_accepts(tx, script_pubkey, flags, height);
    let theirs = library_accepts(tx, script_pubkey, flags);
    assert_eq!(ours, theirs, "{name}: blvm={ours} lib={theirs}");
    assert_eq!(ours, want, "{name}");
}

/// `<push> <op> <same push> OP_EQUAL` keeps the original stack number.
fn keep_script(push: &[u8], op: u8) -> Vec<u8> {
    let mut script = push.to_vec();
    script.push(op);
    script.extend_from_slice(push);
    script.push(0x87);
    script
}

fn drop_script(push: &[u8], op: u8) -> Vec<u8> {
    let mut script = push.to_vec();
    script.push(op);
    script.extend_from_slice(&[0x75, 0x51]);
    script
}

#[test]
fn cltv_comparison_matches_libbitcoinconsensus() {
    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY;
    let clear = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_DERSIG;
    let height = BIP65_ACTIVATION_MAINNET;
    let open = 0xffff_fffe;
    let one = [0x51u8];

    let cases = [
        (
            "equal height locktimes",
            keep_script(&one, 0xb1),
            flags,
            open,
            1,
            true,
        ),
        (
            "locktime strictly above",
            keep_script(&one, 0xb1),
            flags,
            open,
            2,
            true,
        ),
        (
            "stack locktime greater",
            drop_script(&one, 0xb1),
            flags,
            open,
            0,
            false,
        ),
        (
            "final sequence",
            drop_script(&one, 0xb1),
            flags,
            0xffff_ffff,
            1,
            false,
        ),
        (
            "negative stack",
            drop_script(&[0x4f], 0xb1),
            flags,
            open,
            0,
            false,
        ),
        (
            "unsatisfied without the flag",
            drop_script(&one, 0xb1),
            clear,
            open,
            0,
            true,
        ),
    ];
    for (name, spk, flags, sequence, lock_time, want) in cases {
        let tx = make_tx(1, sequence, lock_time, Vec::new());
        expect(name, &tx, &spk, flags, height, want);
    }

    let time_push = push_num(500_000_000);
    let mixed = drop_script(&time_push, 0xb1);
    let tx = make_tx(1, open, 499_999_999, Vec::new());
    expect(
        "time stack against height tx",
        &tx,
        &mixed,
        flags,
        height,
        false,
    );
}

#[test]
fn csv_comparison_matches_libbitcoinconsensus() {
    let flags = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_CHECKSEQUENCEVERIFY;
    let clear = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_DERSIG;
    let height = BIP112_CSV_ACTIVATION_MAINNET;
    let one = [0x51u8];
    let two = [0x52u8];

    let rows = [
        (
            "equal block locks",
            keep_script(&one, 0xb2),
            flags,
            2,
            1,
            true,
        ),
        (
            "input mask strictly above",
            keep_script(&one, 0xb2),
            flags,
            2,
            2,
            true,
        ),
        (
            "stack lock greater",
            drop_script(&two, 0xb2),
            flags,
            2,
            1,
            false,
        ),
        ("version 1", drop_script(&one, 0xb2), flags, 1, 1, false),
        ("version 0", drop_script(&one, 0xb2), flags, 0, 1, false),
        (
            "input disable bit",
            drop_script(&one, 0xb2),
            flags,
            2,
            1 << 31,
            false,
        ),
        (
            "unsatisfied without the flag",
            drop_script(&two, 0xb2),
            clear,
            2,
            1,
            true,
        ),
    ];
    for (name, spk, flags, version, sequence, want) in rows {
        let tx = make_tx(version, sequence, 0, Vec::new());
        expect(name, &tx, &spk, flags, height, want);
    }

    let disable = push_num(1 << 31);
    let skipped = drop_script(&disable, 0xb2);
    let tx = make_tx(1, 0, 0, Vec::new());
    expect(
        "operand disable bit skips version 1",
        &tx,
        &skipped,
        flags,
        height,
        true,
    );

    let time_one = push_num((1 << 22) | 1);
    let mismatch = drop_script(&time_one, 0xb2);
    let tx = make_tx(2, 1, 0, Vec::new());
    expect(
        "time stack against block input",
        &tx,
        &mismatch,
        flags,
        height,
        false,
    );
}

fn n_minus(s: [u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut borrow = 0i16;
    for i in (0..32).rev() {
        let diff = N[i] as i16 - s[i] as i16 - borrow;
        if diff < 0 {
            out[i] = (diff + 256) as u8;
            borrow = 1;
        } else {
            out[i] = diff as u8;
            borrow = 0;
        }
    }
    out
}

fn to_32(be: &[u8]) -> [u8; 32] {
    let v = if be.first() == Some(&0x00) {
        &be[1..]
    } else {
        be
    };
    let mut out = [0u8; 32];
    out[32 - v.len()..].copy_from_slice(v);
    out
}

fn minimal_positive(n: [u8; 32]) -> Vec<u8> {
    let mut start = 0;
    while start < 31 && n[start] == 0 {
        start += 1;
    }
    let mut body = n[start..].to_vec();
    if body[0] & 0x80 != 0 {
        body.insert(0, 0x00);
    }
    body
}

fn split_der(sig: &[u8]) -> (Vec<u8>, Vec<u8>, u8) {
    let ht = *sig.last().unwrap();
    let der = &sig[..sig.len() - 1];
    let mut i = 2;
    i += 1;
    let rlen = der[i] as usize;
    i += 1;
    let r = der[i..i + rlen].to_vec();
    i += rlen;
    i += 1;
    let slen = der[i] as usize;
    i += 1;
    (r, der[i..i + slen].to_vec(), ht)
}

fn join_der(r: &[u8], s: &[u8], ht: u8) -> Vec<u8> {
    let mut body = vec![0x02, r.len() as u8];
    body.extend_from_slice(r);
    body.push(0x02);
    body.push(s.len() as u8);
    body.extend_from_slice(s);
    let mut out = vec![0x30, body.len() as u8];
    out.extend(body);
    out.push(ht);
    out
}

#[test]
fn dersig_encoding_matches_libbitcoinconsensus() {
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x51; 32]).expect("secret");
    let pk = PublicKey::from_secret_key(&secp, &sk).serialize();
    let mut script_pubkey = vec![PUSH_33_BYTES];
    script_pubkey.extend_from_slice(&pk);
    script_pubkey.push(OP_CHECKSIG);

    let bare = make_tx(1, 0xffff_ffff, 0, Vec::new());
    let sighash = compute_legacy_sighash_nocache(&bare, 0, &script_pubkey, 0x01);
    let mut low = secp
        .sign_ecdsa(&Message::from_digest_slice(&sighash).expect("digest"), &sk)
        .serialize_der()
        .to_vec();
    low.push(0x01);
    let (r, s, ht) = split_der(&low);
    assert!(to_32(&s)[0] < 0x80);
    let high = join_der(&r, &minimal_positive(n_minus(to_32(&s))), ht);
    let mut padded_r = vec![0x00];
    padded_r.extend_from_slice(&r);
    let padded = join_der(&padded_r, &s, ht);

    let height = BIP66_ACTIVATION_MAINNET;
    let with = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_DERSIG;
    let without = SCRIPT_VERIFY_P2SH;
    for (name, sig, flags, want) in [
        ("low S", low, with, true),
        ("high S", high, with, true),
        ("padded R", padded.clone(), with, false),
        ("padded R without DERSIG", padded, without, true),
    ] {
        let mut script_sig = vec![sig.len() as u8];
        script_sig.extend_from_slice(&sig);
        let tx = make_tx(1, 0xffff_ffff, 0, script_sig);
        expect(name, &tx, &script_pubkey, flags, height, want);
    }
}
