//! Parallel IBD stores the undo log that `connect_block_ibd_with_undo` returns, and a
//! reorg replays it to disconnect the block. Law: connect then disconnect leaves the
//! UTXO set exactly as it was, whichever connect path built the log.

use blvm_consensus::block::{
    BlockValidationContext, calculate_tx_id, connect_block, connect_block_ibd_with_undo,
};
use blvm_consensus::economic::get_block_subsidy;
use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::reorganization::{BlockUndoLog, UndoEntry};
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::Network;
use blvm_consensus::utxo_overlay::UtxoLookup;
use blvm_consensus::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput, UTXO, UtxoSet,
    ValidationResult,
};
use std::sync::Arc;

/// Below BIP34, P2SH, and SegWit, so the block needs no height push or commitment.
const HEIGHT: u64 = 1_000;

fn op1_coin(value: i64) -> Arc<UTXO> {
    Arc::new(UTXO {
        value,
        script_pubkey: vec![OP_1].into(),
        height: 1,
        is_coinbase: false,
    })
}

fn pre_state() -> UtxoSet {
    let mut set = UtxoSet::default();
    for b in [0x51u8, 0x52, 0x53] {
        set.insert(
            OutPoint {
                hash: [b; 32],
                index: 0,
            },
            op1_coin(10_000),
        );
    }
    set
}

fn tx(inputs: Vec<OutPoint>, outputs: Vec<i64>) -> Transaction {
    Transaction {
        version: 1,
        inputs: inputs
            .into_iter()
            .map(|prevout| TransactionInput {
                prevout,
                script_sig: vec![OP_1].into(),
                sequence: 0xffff_ffff,
            })
            .collect::<Vec<_>>()
            .into(),
        outputs: outputs
            .into_iter()
            .map(|value| TransactionOutput {
                value,
                script_pubkey: vec![OP_1].into(),
            })
            .collect::<Vec<_>>()
            .into(),
        lock_time: 0,
    }
}

/// Coinbase, a spend of two pre-existing coins, and a spend of an output created
/// earlier in the same block (that output must not appear in the undo log).
fn block() -> Block {
    let spend_a = tx(
        vec![
            OutPoint {
                hash: [0x51; 32],
                index: 0,
            },
            OutPoint {
                hash: [0x52; 32],
                index: 0,
            },
        ],
        vec![15_000, 4_000],
    );
    let spend_a_id = calculate_tx_id(&spend_a);
    let spend_b = tx(
        vec![OutPoint {
            hash: spend_a_id,
            index: 1,
        }],
        vec![3_000],
    );
    let fees = 20_000 - 19_000 + 1_000;
    let mut coinbase = tx(
        vec![OutPoint {
            hash: [0; 32],
            index: 0xffff_ffff,
        }],
        vec![get_block_subsidy(HEIGHT) + fees],
    );
    coinbase.inputs[0].script_sig = vec![OP_1, 0x07].into();
    let txs = vec![coinbase, spend_a, spend_b];
    let merkle_root = calculate_merkle_root(&txs).expect("merkle root");
    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp: 1_300_000_000,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: txs.into(),
    }
}

fn witnesses(block: &Block) -> Vec<Vec<Witness>> {
    block
        .transactions
        .iter()
        .map(|t| vec![Vec::new(); t.inputs.len()])
        .collect()
}

/// Same steps as `reorganization::disconnect_block`.
fn disconnect(mut set: UtxoSet, log: &BlockUndoLog) -> UtxoSet {
    for entry in &log.entries {
        if entry.new_utxo.is_some() {
            set.remove(&entry.outpoint);
        }
        if let Some(prev) = &entry.previous_utxo {
            set.insert(entry.outpoint, Arc::clone(prev));
        }
    }
    set
}

fn key(e: &UndoEntry) -> (OutPoint, Option<UTXO>, Option<UTXO>) {
    (
        e.outpoint,
        e.previous_utxo.as_deref().cloned(),
        e.new_utxo.as_deref().cloned(),
    )
}

fn sorted(log: &BlockUndoLog) -> Vec<(OutPoint, Option<UTXO>, Option<UTXO>)> {
    let mut v: Vec<_> = log.entries.iter().map(key).collect();
    v.sort_by(|a, b| (a.0.hash, a.0.index).cmp(&(b.0.hash, b.0.index)));
    v
}

fn assert_sets_equal(a: &UtxoSet, b: &UtxoSet) {
    assert_eq!(a.len(), b.len(), "UTXO set sizes differ");
    for (op, coin) in a.iter() {
        let other = b
            .get(op)
            .unwrap_or_else(|| panic!("missing {op:?} after disconnect"));
        assert_eq!(coin.as_ref(), other.as_ref(), "coin {op:?} differs");
    }
}

fn connect_ibd(block: &Block, set: UtxoSet) -> BlockUndoLog {
    connect_ibd_with_lookup(block, set, None)
}

fn connect_ibd_with_lookup(
    block: &Block,
    set: UtxoSet,
    lookup: Option<&dyn UtxoLookup>,
) -> BlockUndoLog {
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let w = witnesses(block);
    let (result, _, _, _, undo) = connect_block_ibd_with_undo(
        block,
        &w,
        set,
        HEIGHT,
        &ctx,
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
        None,
        lookup,
    )
    .expect("ibd connect");
    assert!(matches!(result, ValidationResult::Valid), "{result:?}");
    undo
}

#[test]
fn ibd_undo_log_matches_sequential_undo_log() {
    let block = block();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (result, _, seq_undo) =
        connect_block(&block, &witnesses(&block), pre_state(), HEIGHT, &ctx).expect("connect");
    assert!(matches!(result, ValidationResult::Valid), "{result:?}");

    let ibd_undo = connect_ibd(&block, pre_state());
    assert!(
        !ibd_undo.entries.is_empty(),
        "connect_block_ibd_with_undo returned an empty undo log"
    );
    assert_eq!(sorted(&ibd_undo), sorted(&seq_undo));
}

#[test]
fn reorg_after_ibd_connect_restores_utxo_set_exactly() {
    let block = block();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (_, post, _) =
        connect_block(&block, &witnesses(&block), pre_state(), HEIGHT, &ctx).expect("connect");
    assert!(post.len() != pre_state().len() || post.keys().any(|k| !pre_state().contains_key(k)));

    let ibd_undo = connect_ibd(&block, pre_state());
    let restored = disconnect(post, &ibd_undo);
    assert_sets_equal(&restored, &pre_state());
}

/// The node validates parallel IBD blocks against a lookup over its UTXO store and an
/// empty scratch set. The spent coins come from that lookup.
#[test]
fn ibd_undo_log_reads_spent_coins_from_the_ibd_lookup() {
    let block = block();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (_, post, _) =
        connect_block(&block, &witnesses(&block), pre_state(), HEIGHT, &ctx).expect("connect");
    let store = pre_state();
    let ibd_undo = connect_ibd_with_lookup(&block, UtxoSet::default(), Some(&store));
    let restored = disconnect(post, &ibd_undo);
    assert_sets_equal(&restored, &pre_state());
}
