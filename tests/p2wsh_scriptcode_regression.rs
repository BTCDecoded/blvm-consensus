//! Regression: P2WSH-in-P2SH spends must not fail after witness script execution.
//!
//! Mainnet block 954486 tx 66 hit `Invalid script at transaction 66` when
//! `verify_script_with_context_full` rejected any input whose witness parameter
//! was still non-empty after the P2WSH-in-P2SH path ran (witness is never cleared).
//! Populate dump: `/tmp/blvm_ibd_failure/height_954486/` from node IBD failure dump.

use blvm_consensus::ValidationResult;
use blvm_consensus::block::connect_block_ibd;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::{Block, Network, OutPoint, UTXO, UtxoSet};
use std::path::Path;
use std::sync::Arc;

fn load_dump(height: u64) -> Option<(Block, Vec<Vec<Witness>>, UtxoSet)> {
    let dir = std::env::var("BLVM_IBD_DUMP_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("blvm_ibd_failure"))
        .join(format!("height_{height}"));
    if !dir.join("block.bin").exists() {
        return None;
    }
    let block: Block = bincode::deserialize_from(std::io::BufReader::new(
        std::fs::File::open(dir.join("block.bin")).ok()?,
    ))
    .ok()?;
    let witnesses: Vec<Vec<Witness>> = bincode::deserialize_from(std::io::BufReader::new(
        std::fs::File::open(dir.join("witnesses.bin")).ok()?,
    ))
    .ok()?;
    let raw: std::collections::HashMap<OutPoint, UTXO> = bincode::deserialize_from(
        std::io::BufReader::new(std::fs::File::open(dir.join("utxo_set.bin")).ok()?),
    )
    .ok()?;
    let utxo_set: UtxoSet = raw.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    Some((block, witnesses, utxo_set))
}

#[test]
fn mainnet_954486_p2wsh_in_p2sh_witness_not_rejected_after_exec() {
    const HEIGHT: u64 = 954486;
    let Some((block, witnesses, utxo_set)) = load_dump(HEIGHT) else {
        eprintln!(
            "skip: no dump at {} (run node until IBD failure dump or copy height_{HEIGHT})",
            Path::new("/tmp/blvm_ibd_failure")
                .join(format!("height_{HEIGHT}"))
                .display()
        );
        return;
    };

    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        0,
        Network::Mainnet,
    );
    let (result, _, _, _) = connect_block_ibd(
        &block,
        &witnesses,
        utxo_set,
        HEIGHT,
        &ctx,
        None,
        None,
        Some(std::sync::Arc::new(block.clone())),
        None,
        None,
    )
    .expect("connect_block_ibd");

    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => {
            panic!("block {HEIGHT} must validate (P2WSH-in-P2SH witness leftover check): {reason}")
        }
    }
}
