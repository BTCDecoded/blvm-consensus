//! A pay-to-public-key inside pay-to-witness-script-hash joins the block batch.
//! A bad signature is rejected when the batch runs, not only by the inline check.

use bitcoin_hashes::{sha256, Hash as _};
use blvm_consensus::constants::SEGWIT_ACTIVATION_MAINNET;
use blvm_consensus::ecdsa_batch::EcdsaSignatureCollector;
use blvm_consensus::script::flags::{
    SCRIPT_VERIFY_DERSIG, SCRIPT_VERIFY_P2SH, SCRIPT_VERIFY_WITNESS,
};
use blvm_consensus::script::{SigVersion, verify_script_with_context_full};
use blvm_consensus::transaction_hash::calculate_bip143_sighash;
use blvm_consensus::types::{Network, OutPoint, Transaction, TransactionInput, TransactionOutput};
use secp256k1::{Message, PublicKey, Secp256k1, SecretKey};

const FLAGS: u32 = SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_DERSIG | SCRIPT_VERIFY_WITNESS;
const AMOUNT: i64 = 100_000;

fn spend() -> Transaction {
    Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x11; 32],
                index: 0,
            },
            script_sig: Vec::new(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000,
            script_pubkey: vec![0x51],
        }]
        .into(),
        lock_time: 0,
    }
}

fn p2pk_witness() -> (Vec<u8>, Vec<Vec<u8>>, Vec<u8>) {
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&[0x51; 32]).expect("secret");
    let pk = PublicKey::from_secret_key(&secp, &sk).serialize();
    let mut witness_script = vec![0x21];
    witness_script.extend_from_slice(&pk);
    witness_script.push(0xac);
    let tx = spend();
    let digest =
        calculate_bip143_sighash(&tx, 0, &witness_script, AMOUNT, 0x01, None).expect("sighash");
    let mut sig = secp
        .sign_ecdsa(&Message::from_digest_slice(&digest).expect("digest"), &sk)
        .serialize_der()
        .to_vec();
    sig.push(0x01);
    let program = sha256::Hash::hash(&witness_script).into_inner();
    let mut script_pubkey = vec![0x00, 0x20];
    script_pubkey.extend_from_slice(&program);
    let witness = vec![sig, witness_script];
    (script_pubkey, witness, pk.to_vec())
}

fn verify(
    script_pubkey: &[u8],
    witness: &Vec<Vec<u8>>,
    collector: Option<(&EcdsaSignatureCollector, usize)>,
) -> bool {
    let tx = spend();
    let values = [AMOUNT];
    let prevouts: [&[u8]; 1] = [script_pubkey];
    matches!(
        verify_script_with_context_full(
            &tx.inputs[0].script_sig,
            script_pubkey,
            Some(witness),
            FLAGS,
            &tx,
            0,
            &values,
            &prevouts,
            Some(SEGWIT_ACTIVATION_MAINNET),
            None,
            Network::Mainnet,
            SigVersion::Base,
            None,
            None,
            None,
            None,
            None,
            collector,
        ),
        Ok(true)
    )
}

#[test]
fn buried_p2pk_joins_the_block_batch() {
    let (script_pubkey, mut witness, _pk) = p2pk_witness();
    let good = EcdsaSignatureCollector::new_with_capacity(4);
    assert!(verify(&script_pubkey, &witness, Some((&good, 7))));
    assert!(!good.is_empty());
    let rows = good.verify_batch_indexed().expect("batch");
    assert_eq!(rows, vec![(7, true)]);

    let hashtype = witness[0].len() - 1;
    witness[0][hashtype] = 0x03;
    let bad = EcdsaSignatureCollector::new_with_capacity(4);
    assert!(
        verify(&script_pubkey, &witness, Some((&bad, 7))),
        "a deferred signature accepts until the batch runs"
    );
    assert!(bad.verify_batch().is_err());

    assert!(
        !verify(&script_pubkey, &witness, None),
        "without a collector the same signature fails inline"
    );
}
