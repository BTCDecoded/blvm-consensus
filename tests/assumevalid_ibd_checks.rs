//! Assume-valid IBD still runs the checks Bitcoin Core runs with script checks off.
//!
//! `skip_signatures` may skip script execution. It must not skip transaction
//! structure, signature-operation cost, or the witness commitment.

use std::sync::Arc;

use blvm_consensus::block::{
    BlockValidationContext, connect_block, connect_block_ibd, connect_block_ibd_with_undo,
};
use blvm_consensus::config::{ConsensusConfig, init_consensus_config};
use blvm_consensus::economic::get_block_subsidy;
use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::opcodes::{OP_1, OP_CHECKSIG};
use blvm_consensus::pow::{U256, meet_regtest_proof_of_work};
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::Network;
use blvm_consensus::{
    Block, BlockHeader, OutPoint, SEGWIT_ACTIVATION_MAINNET, Transaction, TransactionInput,
    TransactionOutput, UTXO, UtxoSet, ValidationResult,
};

fn arm_assume_valid() {
    let mut config = ConsensusConfig::default();
    config.block_validation.assume_valid_height = 1_000_000;
    config.block_validation.assume_valid_hash = Some([0xab; 32]);
    config.block_validation.n_minimum_chain_work = 0;
    init_consensus_config(config);
}

fn ctx() -> BlockValidationContext {
    BlockValidationContext::for_network(Network::Mainnet)
}

fn coinbase(height: u64, value: i64) -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: blvm_consensus::bip_validation::encode_bip34_coinbase_script(height),
            sequence: 0xffffffff,
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

fn block_from(txs: Vec<Transaction>, timestamp: u64, version: i64) -> Block {
    let merkle_root = calculate_merkle_root(&txs).expect("merkle");
    let mut block = Block {
        header: BlockHeader {
            version,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp,
            bits: 0x207f_ffff,
            nonce: 0,
        },
        transactions: txs.into(),
    };
    meet_regtest_proof_of_work(&mut block.header);
    block
}

fn empty_witnesses(block: &Block) -> Vec<Vec<Witness>> {
    block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect()
}

fn chainwork() -> Option<U256> {
    Some(U256::from_u128(1))
}

fn seed(set: &mut UtxoSet, byte: u8, value: i64) {
    set.insert(
        OutPoint {
            hash: [byte; 32],
            index: 0,
        },
        Arc::new(UTXO {
            value,
            script_pubkey: vec![OP_1].into(),
            height: 0,
            is_coinbase: false,
        }),
    );
}

#[test]
fn sibling_below_the_pin_does_not_skip_scripts() {
    arm_assume_valid();
    let height = 50u64;
    let input_value = 50_000i64;
    let mut utxo_set = UtxoSet::default();
    seed_false(&mut utxo_set, 0x61, input_value);
    let spend = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0x61; 32],
                index: 0,
            },
            script_sig: Vec::new().into(),
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: input_value - 1_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let mut txs = vec![coinbase(height, get_block_subsidy(height) + 1_000)];
    txs.push(spend);
    let block = block_from(txs, 1_231_006_505, 1);
    let witnesses = empty_witnesses(&block);
    let (sibling, _, _, _, _) = connect_block_ibd_with_undo(
        &block,
        &witnesses,
        utxo_set.clone(),
        height,
        &ctx(),
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
        chainwork(),
        None,
        false,
    )
    .expect("sibling connect");
    assert!(
        matches!(sibling, ValidationResult::Invalid(ref r) if r.contains("Invalid script")),
        "a sibling below the pin must run scripts, got {sibling:?}"
    );
    let (on_pin_chain, _, _, _, _) = connect_block_ibd_with_undo(
        &block,
        &witnesses,
        utxo_set,
        height,
        &ctx(),
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
        chainwork(),
        None,
        true,
    )
    .expect("pin chain connect");
    assert!(
        matches!(on_pin_chain, ValidationResult::Valid),
        "the same block skips scripts once the pin hash is in the index, got {on_pin_chain:?}"
    );
}

fn seed_false(set: &mut UtxoSet, byte: u8, value: i64) {
    set.insert(
        OutPoint {
            hash: [byte; 32],
            index: 0,
        },
        Arc::new(UTXO {
            value,
            script_pubkey: vec![0x00].into(),
            height: 0,
            is_coinbase: false,
        }),
    );
}

#[test]
fn assumevalid_ibd_rejects_duplicate_prevout() {
    arm_assume_valid();
    let height = 1u64;
    let input_value = 50_000i64;
    let mut utxo_set = UtxoSet::default();
    seed(&mut utxo_set, 0x44, input_value);
    let prevout = OutPoint {
        hash: [0x44; 32],
        index: 0,
    };
    let spend = Transaction {
        version: 1,
        inputs: vec![
            TransactionInput {
                prevout: prevout.clone(),
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            },
            TransactionInput {
                prevout,
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            },
        ]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    // Subsidy plus twice the output. That coinbase is legal only if the prevout is counted twice.
    let coinbase_value = get_block_subsidy(height) + input_value * 2;
    let block = block_from(
        vec![coinbase(height, coinbase_value), spend],
        1_231_006_505,
        1,
    );
    let witnesses = empty_witnesses(&block);
    let (non_ibd, _, _) = connect_block(&block, &witnesses, utxo_set.clone(), height, &ctx())
        .expect("non-ibd connect");
    assert!(
        matches!(non_ibd, ValidationResult::Invalid(ref r) if r.contains("Duplicate")),
        "non-IBD connect must reject the duplicate prevout, got {non_ibd:?}"
    );
    let (ibd, _, _, _) = connect_block_ibd(
        &block,
        &witnesses,
        utxo_set,
        height,
        &ctx(),
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
        chainwork(),
    )
    .expect("ibd connect");
    assert!(
        matches!(ibd, ValidationResult::Invalid(ref r) if r.contains("Duplicate")),
        "assume-valid IBD must reject the duplicate prevout, got {ibd:?}"
    );
}

#[test]
fn assumevalid_ibd_rejects_sigop_cost_over_limit() {
    arm_assume_valid();
    let height = 1u64;
    let per_tx = 7_000usize;
    let mut utxo_set = UtxoSet::default();
    let mut spends = Vec::new();
    for i in 0..3u8 {
        let byte = 0x50 + i;
        seed(&mut utxo_set, byte, 1_000);
        spends.push(Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [byte; 32],
                    index: 0,
                },
                script_sig: vec![OP_1].into(),
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 500,
                script_pubkey: vec![OP_CHECKSIG; per_tx].into(),
            }]
            .into(),
            lock_time: 0,
        });
    }
    let fees = 500 * 3;
    let mut txs = vec![coinbase(height, get_block_subsidy(height) + fees)];
    txs.extend(spends);
    let block = block_from(txs, 1_231_006_505, 1);
    let witnesses = empty_witnesses(&block);
    let (non_ibd, _, _) =
        connect_block(&block, &witnesses, utxo_set.clone(), height, &ctx()).expect("non-ibd");
    assert!(
        matches!(non_ibd, ValidationResult::Invalid(ref r) if r.contains("sigop")),
        "non-IBD connect must reject the sigop cost, got {non_ibd:?}"
    );
    let (ibd, _, _, _) = connect_block_ibd(
        &block,
        &witnesses,
        utxo_set,
        height,
        &ctx(),
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
        chainwork(),
    )
    .expect("ibd");
    assert!(
        matches!(ibd, ValidationResult::Invalid(ref r) if r.contains("sigop")),
        "assume-valid IBD must reject the sigop cost, got {ibd:?}"
    );
}

#[test]
fn assumevalid_ibd_rejects_witness_without_commitment() {
    arm_assume_valid();
    let height = SEGWIT_ACTIVATION_MAINNET;
    let block = block_from(
        vec![coinbase(height, get_block_subsidy(height))],
        1_500_353_985,
        4,
    );
    let witnesses: Vec<Vec<Witness>> = vec![vec![vec![vec![0x11u8; 32]]]];
    let (non_ibd, _, _) =
        connect_block(&block, &witnesses, UtxoSet::default(), height, &ctx()).expect("non-ibd");
    assert!(
        matches!(non_ibd, ValidationResult::Invalid(ref r) if r.contains("witness commitment")),
        "non-IBD connect must reject a witness with no commitment, got {non_ibd:?}"
    );
    let (ibd, _, _, _) = connect_block_ibd(
        &block,
        &witnesses,
        UtxoSet::default(),
        height,
        &ctx(),
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
        chainwork(),
    )
    .expect("ibd");
    assert!(
        matches!(ibd, ValidationResult::Invalid(ref r) if r.contains("witness commitment")),
        "assume-valid IBD must reject a witness with no commitment, got {ibd:?}"
    );
}
