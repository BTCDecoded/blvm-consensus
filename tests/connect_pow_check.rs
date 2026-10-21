//! ConnectBlock must reject a block whose header hash is above the target its own
//! `nBits` encodes (Core: `CheckBlockHeader` -> `CheckProofOfWork`, run from
//! `CheckBlock` inside `ConnectBlock`). The `nBits == required` retarget check alone
//! does not hash the header. Since bbdaccb the check runs on every connect, with or
//! without the ancestor lookup; these vectors pin both cases.
//!
//! Vector: mainnet block 1 (hash
//! 00000000839a8e6886ab5951d76f411475428afc90947ee320161bbf18eb6048), rebuilt from its
//! coinbase so the merkle root is checked too.

use blvm_consensus::block::{BlockValidationContext, calculate_tx_id, connect_block};
use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::pow::check_proof_of_work;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::Network;
use blvm_consensus::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet,
    ValidationResult,
};
use std::sync::Arc;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Display (RPC) hex to internal byte order.
fn hash_from_display(s: &str) -> [u8; 32] {
    let mut v = unhex(s);
    v.reverse();
    v.try_into().unwrap()
}

const GENESIS: &str = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";
const BLOCK1_MERKLE: &str = "0e3e2357e806b6cdb1f70b54c3a3a17b6714ee1f0e68bebb44a74b1efd512098";
const BLOCK1_HASH: &str = "00000000839a8e6886ab5951d76f411475428afc90947ee320161bbf18eb6048";

fn mainnet_block_1() -> Block {
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffff_ffff,
            },
            script_sig: unhex("04ffff001d0104").into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000,
            script_pubkey: unhex(
                "410496b538e853519c726a2c91e61ec11600ae1390813a627c66fb8be7947be63c52da7589379515d4e0a604f8141781e62294721166bf621e73a82cbf2342c858eeac",
            )
            .into(),
        }]
        .into(),
        lock_time: 0,
    };
    let txs = vec![coinbase];
    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: hash_from_display(GENESIS),
            merkle_root: calculate_merkle_root(&txs).unwrap(),
            timestamp: 1_231_469_665,
            bits: 0x1d00ffff,
            nonce: 2_573_394_689,
        },
        transactions: txs.into(),
    }
}

/// Production context: the node passes an ancestor lookup, so `nBits` is checked
/// against the retarget rule. Height 0 is genesis.
fn mainnet_ctx() -> BlockValidationContext {
    let mut ctx = BlockValidationContext::for_network(Network::Mainnet);
    ctx.difficulty_ancestor = Some(Arc::new(|h| {
        if h == 0 {
            Some((0x1d00ffff, 1_231_006_505))
        } else {
            None
        }
    }));
    ctx
}

fn connect(block: &Block, ctx: &BlockValidationContext) -> ValidationResult {
    let witnesses: Vec<Vec<Witness>> = vec![vec![Vec::new()]; block.transactions.len()];
    connect_block(block, &witnesses, UtxoSet::default(), 1, ctx)
        .expect("connect")
        .0
}

#[test]
fn vector_is_mainnet_block_1() {
    let block = mainnet_block_1();
    assert_eq!(
        calculate_tx_id(&block.transactions[0]),
        hash_from_display(BLOCK1_MERKLE)
    );
    assert_eq!(block.header.merkle_root, hash_from_display(BLOCK1_MERKLE));
    assert_eq!(
        blvm_consensus::block::block_header_hash(&block.header),
        hash_from_display(BLOCK1_HASH)
    );
    assert!(check_proof_of_work(&block.header).unwrap());
}

#[test]
fn connect_accepts_mainnet_block_1() {
    let result = connect(&mainnet_block_1(), &mainnet_ctx());
    assert!(matches!(result, ValidationResult::Valid), "{result:?}");
}

/// Same header with a different nonce: `nBits` still equals the required value, the
/// merkle root still matches, and the hash is above the target.
#[test]
fn connect_rejects_hash_above_target_with_correct_bits() {
    let mut block = mainnet_block_1();
    block.header.nonce = 0;
    assert!(!check_proof_of_work(&block.header).unwrap());
    let result = connect(&block, &mainnet_ctx());
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("proof of work")),
        "a header above its target must not connect: {result:?}"
    );
}

/// Library callers that pass no ancestor lookup get the same check (Core runs
/// `CheckProofOfWork` regardless of retarget context).
#[test]
fn connect_rejects_hash_above_target_without_ancestor_lookup() {
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    assert!(ctx.difficulty_ancestor.is_none());
    assert!(matches!(
        connect(&mainnet_block_1(), &ctx),
        ValidationResult::Valid
    ));
    let mut block = mainnet_block_1();
    block.header.nonce = 0;
    assert!(!check_proof_of_work(&block.header).unwrap());
    let result = connect(&block, &ctx);
    assert!(
        matches!(result, ValidationResult::Invalid(ref r) if r.contains("proof of work")),
        "{result:?}"
    );
}

/// Boundary on regtest (target just under 2^255): grind one nonce on each side.
#[test]
fn connect_regtest_pow_boundary() {
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffff_ffff,
            },
            script_sig: vec![0x51, 0x01].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1,
            script_pubkey: vec![0x51].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let txs = vec![coinbase];
    let mut block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [7; 32],
            merkle_root: calculate_merkle_root(&txs).unwrap(),
            timestamp: 1_296_688_700,
            bits: 0x207fffff,
            nonce: 0,
        },
        transactions: txs.into(),
    };
    let mut ctx = BlockValidationContext::for_network(Network::Regtest);
    ctx.difficulty_ancestor = Some(Arc::new(|_| Some((0x207fffff, 1_296_688_602))));

    let mut seen_pass = false;
    let mut seen_fail = false;
    for nonce in 0..64u64 {
        block.header.nonce = nonce;
        let meets = check_proof_of_work(&block.header).unwrap();
        let result = connect(&block, &ctx);
        if meets {
            seen_pass = true;
            assert!(
                matches!(result, ValidationResult::Valid),
                "nonce {nonce}: {result:?}"
            );
        } else {
            seen_fail = true;
            assert!(
                matches!(result, ValidationResult::Invalid(ref r) if r.contains("proof of work")),
                "nonce {nonce}: {result:?}"
            );
        }
    }
    assert!(seen_pass && seen_fail, "grind did not hit both sides");
}
