//! BIP325 signet solution against `libbitcoinconsensus`.
//!
//! The script library checks the spend. It does not build the signet
//! transaction. This test builds that transaction from the header image,
//! asks the library to verify the challenge, and requires
//! `check_signet_block_solution` to return the same verdict.

#![cfg(feature = "production")]

use bitcoinconsensus::{
    Error, Utxo, VERIFY_DERSIG, VERIFY_NULLDUMMY, VERIFY_P2SH, VERIFY_WITNESS, verify_with_flags,
};
use blvm_consensus::block::calculate_tx_id;
use blvm_consensus::mining::compute_merkle_root_and_mutated;
use blvm_consensus::opcodes::{OP_0, OP_CHECKSIG, OP_PUSHDATA1, OP_RETURN};
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_NULLDUMMY, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_WITNESS,
};
use blvm_consensus::script::{SigVersion, verify_script_with_context_full};
use blvm_consensus::serialization::transaction::serialize_transaction;
use blvm_consensus::signet::check_signet_block_solution;
use blvm_consensus::transaction_hash::compute_legacy_sighash_nocache;
use blvm_consensus::types::{
    Block, BlockHeader, Hash, Network, OutPoint, Transaction, TransactionInput, TransactionOutput,
};

const SIGNET_HEADER: [u8; 4] = [0xec, 0xc7, 0xda, 0xa2];
const SIGNET_FLAGS: u32 =
    SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_NULLDUMMY;

fn library_accepts(tx: &Transaction, script_pubkey: &[u8]) -> bool {
    let raw = serialize_transaction(tx);
    let spent = [Utxo {
        script_pubkey: script_pubkey.as_ptr(),
        script_pubkey_len: script_pubkey.len() as u32,
        value: 0,
    }];
    match verify_with_flags(script_pubkey, 0, &raw, Some(&spent), 0, SIGNET_FLAGS) {
        Ok(()) => true,
        Err(Error::ERR_SCRIPT) => false,
        Err(err) => panic!("libbitcoinconsensus api error: {err}"),
    }
}

fn blvm_accepts(tx: &Transaction, script_pubkey: &[u8]) -> bool {
    let prevout_values = [0i64];
    let prevouts: [&[u8]; 1] = [script_pubkey];
    let result = verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        script_pubkey,
        None,
        SIGNET_FLAGS,
        tx,
        0,
        &prevout_values,
        &prevouts,
        Some(1),
        None,
        Network::Signet,
        SigVersion::Base,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    matches!(result, Ok(true))
}

fn header() -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: Hash::from([0u8; 32]),
        merkle_root: Hash::from([0u8; 32]),
        timestamp: 1_600_000_000,
        bits: 0x1e0377ae,
        nonce: 0,
    }
}

fn coinbase(script: Vec<u8>) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: Hash::from([0u8; 32]),
                index: 0xffff_ffff,
            },
            script_sig: Vec::new(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: script,
        }]
        .into(),
        lock_time: 0,
    }
}

fn dummy() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: Hash::from([1u8; 32]),
                index: 0,
            },
            script_sig: Vec::new(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: Vec::new(),
        }]
        .into(),
        lock_time: 0,
    }
}

/// Header image committed by the to_spend scriptSig: version, prev, modified merkle, time.
fn header_image(header: &BlockHeader, modified: &Transaction, other: &Transaction) -> Vec<u8> {
    let ids = [calculate_tx_id(modified), calculate_tx_id(other)];
    let (merkle, _) = compute_merkle_root_and_mutated(&ids).unwrap();
    let mut image = Vec::with_capacity(72);
    image.extend_from_slice(&(header.version as i32).to_le_bytes());
    image.extend_from_slice(&header.prev_block_hash);
    image.extend_from_slice(&merkle);
    image.extend_from_slice(&(header.timestamp as u32).to_le_bytes());
    image
}

fn to_spend(image: &[u8], challenge: &[u8]) -> Transaction {
    let mut script_sig = Vec::with_capacity(2 + image.len());
    script_sig.push(OP_0);
    script_sig.push(image.len() as u8);
    script_sig.extend_from_slice(image);
    Transaction {
        version: 0,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: Hash::from([0u8; 32]),
                index: 0xffff_ffff,
            },
            script_sig,
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: challenge.to_vec(),
        }]
        .into(),
        lock_time: 0,
    }
}

fn to_sign(spend_id: Hash, script_sig: Vec<u8>) -> Transaction {
    Transaction {
        version: 0,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: spend_id,
                index: 0,
            },
            script_sig,
            sequence: 0,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![OP_RETURN],
        }]
        .into(),
        lock_time: 0,
    }
}

fn solution_block(script_sig: &[u8]) -> Block {
    let mut solution = Vec::new();
    solution.push(script_sig.len() as u8);
    solution.extend_from_slice(script_sig);
    solution.push(0);
    let mut payload = SIGNET_HEADER.to_vec();
    payload.extend_from_slice(&solution);
    let mut commit = vec![OP_RETURN, 0x24, 0xaa, 0x21, 0xa9, 0xed];
    commit.extend(std::iter::repeat_n(0xff, 32));
    commit.push(OP_PUSHDATA1);
    commit.push(payload.len() as u8);
    commit.extend_from_slice(&payload);
    Block {
        header: header(),
        transactions: vec![coinbase(commit), dummy()].into(),
    }
}

fn p2pk_challenge() -> (Vec<u8>, Vec<u8>) {
    let seckey = [0x11u8; 32];
    let mut sec = blvm_secp256k1::scalar::Scalar::zero();
    assert!(!sec.set_b32(&seckey));
    let pubkey =
        blvm_secp256k1::ecdsa::ge_to_compressed(&blvm_secp256k1::ecdsa::pubkey_from_secret(&sec));
    let mut challenge = Vec::with_capacity(35);
    challenge.push(33);
    challenge.extend_from_slice(&pubkey);
    challenge.push(OP_CHECKSIG);

    let header = header();
    let mut stripped = vec![OP_RETURN, 0x24, 0xaa, 0x21, 0xa9, 0xed];
    stripped.extend(std::iter::repeat_n(0xff, 32));
    stripped.push(4);
    stripped.extend_from_slice(&SIGNET_HEADER);
    let image = header_image(&header, &coinbase(stripped), &dummy());
    let spend = to_spend(&image, &challenge);
    let signed = to_sign(calculate_tx_id(&spend), Vec::new());
    let sighash = compute_legacy_sighash_nocache(&signed, 0, &challenge, 0x01);
    let mut sig = blvm_secp256k1::ecdsa::ecdsa_sign_der_rfc6979(&sighash, &seckey).unwrap();
    sig.push(0x01);
    let mut script_sig = Vec::with_capacity(1 + sig.len());
    script_sig.push(sig.len() as u8);
    script_sig.extend_from_slice(&sig);
    (challenge, script_sig)
}

fn verdicts(script_sig: &[u8], challenge: &[u8]) -> (bool, bool, bool) {
    let header = header();
    let mut stripped = vec![OP_RETURN, 0x24, 0xaa, 0x21, 0xa9, 0xed];
    stripped.extend(std::iter::repeat_n(0xff, 32));
    stripped.push(4);
    stripped.extend_from_slice(&SIGNET_HEADER);
    let image = header_image(&header, &coinbase(stripped), &dummy());
    let spend = to_spend(&image, challenge);
    let signed = to_sign(calculate_tx_id(&spend), script_sig.to_vec());
    let library = library_accepts(&signed, challenge);
    let ours = blvm_accepts(&signed, challenge);
    // A non-DER signature is a script error. That rejects the block.
    let block_ok =
        check_signet_block_solution(&solution_block(script_sig), challenge, 1).unwrap_or(false);
    (library, ours, block_ok)
}

#[test]
fn signet_solution_matches_libbitcoinconsensus() {
    assert_eq!(SCRIPT_VERIFY_P2SH, VERIFY_P2SH);
    assert_eq!(SCRIPT_VERIFY_WITNESS, VERIFY_WITNESS);
    assert_eq!(SCRIPT_VERIFY_DERSIG, VERIFY_DERSIG);
    assert_eq!(SCRIPT_VERIFY_NULLDUMMY, VERIFY_NULLDUMMY);

    let (challenge, script_sig) = p2pk_challenge();
    let (library, ours, block_ok) = verdicts(&script_sig, &challenge);
    assert!(
        library && ours && block_ok,
        "valid signet solution: libbitcoinconsensus={library} blvm={ours} check_signet={block_ok}"
    );

    let mut bad = script_sig;
    bad[1] ^= 0xff;
    let (library, ours, block_ok) = verdicts(&bad, &challenge);
    assert!(
        !library && !ours && !block_ok,
        "bad signet solution: libbitcoinconsensus={library} blvm={ours} check_signet={block_ok}"
    );
}
