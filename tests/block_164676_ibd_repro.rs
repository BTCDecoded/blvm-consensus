//! Reproduce block 164676 validation using dumped IBD failure data.
//!
//! When IBD fails at a block, the node dumps block.bin, witnesses.bin, utxo_set.bin to
//! `$BLVM_IBD_DUMP_DIR/height_{height}/` (default: platform temp dir + blvm_ibd_failure).
//! This test loads that data and runs connect_block_ibd so we can fix the ECDSA batch
//! issue without running full IBD.
//!
//! Run with: `cargo test --test block_164676_ibd_repro -- --ignored` (or run without
//! --ignored if the dump is present and you want it to run in CI).
//! Set BLVM_IBD_DUMP_DIR to point at the dump root if not using default.

use blvm_consensus::block::connect_block_ibd;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::{Block, Network, UtxoSet, UTXO};
use blvm_consensus::ValidationResult;
use std::path::Path;
use std::sync::Arc;

/// Backup in repo so it survives cleanup; used when BLVM_IBD_DUMP_DIR is not set.
const REPO_DUMP_SUBDIR: &str = "tests/test_data/ibd_failure_height_164676";
const HEIGHT: u64 = 164676;

fn dump_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("BLVM_IBD_DUMP_DIR") {
        return std::path::PathBuf::from(d).join(format!("height_{}", HEIGHT));
    }
    // Prefer repo backup (survives temp cleanup)
    let repo = std::path::PathBuf::from(REPO_DUMP_SUBDIR);
    if repo.join("block.bin").exists() {
        return repo;
    }
    std::env::temp_dir()
        .join("blvm_ibd_failure")
        .join(format!("height_{}", HEIGHT))
}

fn load_dump(
    dir: &Path,
) -> Result<(Block, Vec<Vec<Witness>>, UtxoSet), Box<dyn std::error::Error + Send + Sync>> {
    let block_path = dir.join("block.bin");
    let witnesses_path = dir.join("witnesses.bin");
    let utxo_path = dir.join("utxo_set.bin");

    let block: Block =
        bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(&block_path)?))?;
    let witnesses: Vec<Vec<Witness>> = bincode::deserialize_from(std::io::BufReader::new(
        std::fs::File::open(&witnesses_path)?,
    ))?;
    // Dump format: HashMap<OutPoint, UTXO> (no Arc). UtxoSet uses Arc<UTXO>.
    let raw: std::collections::HashMap<_, UTXO> =
        bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(&utxo_path)?))?;
    let utxo_set: UtxoSet = raw.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();

    Ok((block, witnesses, utxo_set))
}

#[test]
#[ignore = "Slow: loads ~113MB UTXO set; run with --ignored when iterating on ECDSA fix"]
fn block_164676_connect_block_ibd_repro() {
    let dir = dump_dir();
    if !dir.join("block.bin").exists() {
        eprintln!("Skip: dump not found at {} (set BLVM_IBD_DUMP_DIR or use repo tests/test_data/ibd_failure_height_164676)", dir.display());
        return;
    }

    let (block, mut witnesses, utxo_set) = load_dump(&dir).expect("load dump");
    // Pre-SegWit block 164676: signatures are in scriptSig. If dump has no witnesses, use one empty witness per input.
    if witnesses.len() != block.transactions.len() {
        witnesses = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::new()).collect())
            .collect();
    }
    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        0u64,
        Network::Mainnet,
    );
    let (result, _new_utxo_set, _tx_ids, _utxo_delta) = connect_block_ibd(
        &block,
        &witnesses,
        utxo_set,
        HEIGHT,
        &ctx,
        None,
        None,
        Some(std::sync::Arc::new(block.clone())),
        None,
    )
    .expect("connect_block_ibd");

    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => panic!("Block {} should be valid: {}", HEIGHT, reason),
    }
}
