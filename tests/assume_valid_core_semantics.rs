//! Assume-valid with Bitcoin Core semantics (`ConnectBlock`, fScriptChecks):
//! scripts are skipped only for an ancestor of the assume-valid block on the header
//! chain; any other block, including a competing block at the assume-valid height, is
//! fully validated and never rejected merely for not being the assume-valid block.
#![cfg(feature = "production")]

use blvm_consensus::block::{BlockValidationContext, connect_block_ibd};
use blvm_consensus::config::{ConsensusConfig, init_consensus_config};
use blvm_consensus::economic::get_block_subsidy;
use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::opcodes::{OP_0, OP_1};
use blvm_consensus::pow::U256;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::{Hash, Network};
use blvm_consensus::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput, UTXO, UtxoSet,
    ValidationResult,
};
use std::sync::Arc;

/// Below BIP34, P2SH and SegWit on mainnet, so blocks need no height push or commitment.
const AV_HEIGHT: u64 = 2_000;
const BELOW_AV: u64 = 1_000;
/// Some block hash we were told is valid. None of the test blocks hash to it.
const AV_HASH: Hash = [0xa5; 32];

fn setup() {
    let mut cfg = ConsensusConfig::default();
    cfg.block_validation.assume_valid_height = AV_HEIGHT;
    cfg.block_validation.assume_valid_hash = Some(AV_HASH);
    cfg.block_validation.n_minimum_chain_work = 0;
    init_consensus_config(cfg);
    assert_eq!(
        blvm_consensus::config::get_assume_valid_hash(),
        Some(AV_HASH)
    );
}

/// One coin locked by `OP_0`: any spend of it fails script verification.
const BAD: OutPoint = OutPoint {
    hash: [0x77; 32],
    index: 0,
};
/// One coin locked by `OP_1`: anyone can spend it.
const GOOD: OutPoint = OutPoint {
    hash: [0x78; 32],
    index: 0,
};

fn utxos() -> UtxoSet {
    let mut set = UtxoSet::default();
    for (op, script) in [(BAD, OP_0), (GOOD, OP_1)] {
        set.insert(
            op,
            Arc::new(UTXO {
                value: 10_000,
                script_pubkey: vec![script].into(),
                height: 1,
                is_coinbase: false,
            }),
        );
    }
    set
}

fn tx(prevout: OutPoint, value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout,
            script_sig: vec![OP_1].into(),
            sequence: 0xffff_ffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    }
}

/// Coinbase plus one spend of `spent`.
fn block(height: u64, spent: OutPoint) -> Block {
    let spend = tx(spent, 9_000);
    let mut coinbase = tx(
        OutPoint {
            hash: [0; 32],
            index: 0xffff_ffff,
        },
        get_block_subsidy(height) + 1_000,
    );
    coinbase.inputs[0].script_sig = vec![OP_1, 0x07].into();
    let txs = vec![coinbase, spend];
    let mut block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: calculate_merkle_root(&txs).unwrap(),
            timestamp: 1_300_000_000,
            bits: 0x207fffff,
            nonce: 0,
        },
        transactions: txs.into(),
    };
    blvm_consensus::pow::meet_regtest_proof_of_work(&mut block.header);
    block
}

fn hash(block: &Block) -> Hash {
    blvm_consensus::block::block_header_hash(&block.header)
}

/// Header chain whose ancestor at every height is `ancestor_at` (None = assume-valid block
/// not on the header chain).
fn ctx_with(ancestor_at: Option<Hash>) -> BlockValidationContext {
    let mut ctx = BlockValidationContext::for_network(Network::Mainnet);
    ctx.assume_valid_ancestor = Some(Arc::new(move |_h| ancestor_at));
    ctx
}

fn connect(block: &Block, height: u64, ctx: &BlockValidationContext) -> ValidationResult {
    let witnesses: Vec<Vec<Witness>> = block
        .transactions
        .iter()
        .map(|t| vec![Vec::new(); t.inputs.len()])
        .collect();
    connect_block_ibd(
        block,
        &witnesses,
        utxos(),
        height,
        ctx,
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
        Some(U256::from_u128(u128::MAX)),
    )
    .expect("connect")
    .0
}

/// The block is the assume-valid block's ancestor: its scripts are not run, so a spend
/// that would fail script verification is accepted (that is what assume-valid trusts).
#[test]
fn ancestor_of_assume_valid_block_skips_scripts() {
    setup();
    let b = block(BELOW_AV, BAD);
    let r = connect(&b, BELOW_AV, &ctx_with(Some(hash(&b))));
    assert!(matches!(r, ValidationResult::Valid), "{r:?}");
}

/// Same block, but the header chain's block at that height is a different one: the
/// block is on a competing chain and gets full script validation.
#[test]
fn competing_chain_below_assume_valid_is_fully_validated() {
    setup();
    let b = block(BELOW_AV, BAD);
    let r = connect(&b, BELOW_AV, &ctx_with(Some([0x11; 32])));
    assert!(matches!(r, ValidationResult::Invalid(_)), "{r:?}");
    let r = connect(&b, BELOW_AV, &ctx_with(None));
    assert!(matches!(r, ValidationResult::Invalid(_)), "{r:?}");
    // The same chain spending a valid coin is accepted: the rejection is the script.
    let ok = block(BELOW_AV, GOOD);
    let r = connect(&ok, BELOW_AV, &ctx_with(Some([0x11; 32])));
    assert!(matches!(r, ValidationResult::Valid), "{r:?}");
}

/// No ancestry information at all (library caller): fail closed, verify scripts.
#[test]
fn no_ancestry_lookup_verifies_scripts() {
    setup();
    let b = block(BELOW_AV, BAD);
    let r = connect(
        &b,
        BELOW_AV,
        &BlockValidationContext::for_network(Network::Mainnet),
    );
    assert!(matches!(r, ValidationResult::Invalid(_)), "{r:?}");
}

/// A valid block at the assume-valid height that is not the assume-valid block is
/// accepted (fully validated), not rejected for a hash mismatch.
#[test]
fn competing_block_at_assume_valid_height_is_not_rejected() {
    setup();
    let b = block(AV_HEIGHT, GOOD);
    assert_ne!(hash(&b), AV_HASH);
    let r = connect(&b, AV_HEIGHT, &ctx_with(Some([0x11; 32])));
    assert!(matches!(r, ValidationResult::Valid), "{r:?}");
    // ...and an invalid one there is rejected for its script, not for its hash.
    let bad = block(AV_HEIGHT, BAD);
    match connect(&bad, AV_HEIGHT, &ctx_with(Some([0x11; 32]))) {
        ValidationResult::Invalid(msg) => {
            assert!(!msg.contains("Assume-valid block hash mismatch"), "{msg}")
        }
        r => panic!("{r:?}"),
    }
}
