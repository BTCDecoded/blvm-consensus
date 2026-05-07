//! Replay an IBD failure dump under **full script verification** (assume-valid off).
//!
//! Exercises production paths including inline P2PKH/P2PK verification and legacy batch
//! sighash precompute when applicable — the same code that is skipped when
//! `height < BLVM_ASSUME_VALID_HEIGHT` with a positive checkpoint height.
//!
//! **Requires** populated binaries next to `info.txt`:
//! `tests/test_data/ibd_failure_height_{H}/block.bin`, `witnesses.bin`, `utxo_set.bin`
//! (see `scripts/ibd_failure_to_repro_test.sh` / `block_ibd_repro.rs`).
//!
//! Run:
//! ```text
//! ./scripts/run-full-script-verify-ibd-dump.sh [HEIGHT]
//! ```
//! Or manually (from repo root):
//! ```text
//! BLVM_ASSUME_VALID_HEIGHT=0 BLVM_IBD_FAILURE_HEIGHT=481824 \
//!   cargo test --manifest-path blvm-consensus/Cargo.toml --test full_script_verify_ibd_dump -- \
//!   --ignored --exact ibd_dump_connect_full_script_verify --nocapture
//! ```

use blvm_consensus::block::connect_block_ibd;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::{Block, Network, OutPoint, UtxoSet, UTXO};
use blvm_consensus::ValidationResult;
use std::path::Path;
use std::sync::Arc;

fn height() -> u64 {
    std::env::var("BLVM_IBD_FAILURE_HEIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(481_824)
}

fn dump_dir(h: u64) -> std::path::PathBuf {
    if let Ok(d) = std::env::var("BLVM_IBD_DUMP_DIR") {
        return Path::new(&d).join(format!("height_{h}"));
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/test_data")
        .join(format!("ibd_failure_height_{h}"))
}

fn load_dump(
    dir: &Path,
) -> Result<(Block, Vec<Vec<Witness>>, UtxoSet), Box<dyn std::error::Error + Send + Sync>> {
    let block: Block = bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(
        dir.join("block.bin"),
    )?))?;
    let witnesses: Vec<Vec<Witness>> = bincode::deserialize_from(std::io::BufReader::new(
        std::fs::File::open(dir.join("witnesses.bin"))?,
    )?)?;
    let raw: std::collections::HashMap<OutPoint, UTXO> = bincode::deserialize_from(
        std::io::BufReader::new(std::fs::File::open(dir.join("utxo_set.bin"))?),
    )?;
    let utxo_set: UtxoSet = raw.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    Ok((block, witnesses, utxo_set))
}

#[test]
#[ignore = "slow; requires ibd_failure_height_* binaries; forces full script verification"]
fn ibd_dump_connect_full_script_verify() {
    // Must run before any consensus code touches GLOBAL_CONSENSUS_CONFIG (OnceLock).
    std::env::set_var("BLVM_ASSUME_VALID_HEIGHT", "0");

    let h = height();
    let dir = dump_dir(h);
    if !dir.join("block.bin").exists() {
        eprintln!(
            "skip full_script_verify_ibd_dump: no dump at {} (populate via scripts/ibd_failure_to_repro_test.sh {})",
            dir.display(),
            h
        );
        return;
    }

    let (block, mut witnesses, utxo_set) =
        load_dump(&dir).expect("load dump; check block.bin witnesses.bin utxo_set.bin");

    if witnesses.len() != block.transactions.len() {
        witnesses = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::new()).collect())
            .collect();
    }

    assert_eq!(
        blvm_consensus::block::get_assume_valid_height(),
        0,
        "assume-valid height must be 0 so scripts are verified (config read from env)"
    );

    let network_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        network_time,
        Network::Mainnet,
    );

    let (result, _new_utxo, _tx_ids, _delta) = connect_block_ibd(
        &block,
        &witnesses,
        utxo_set,
        h,
        &ctx,
        None,
        None,
        Some(Arc::new(block.clone())),
        None,
    )
    .expect("connect_block_ibd");

    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => panic!("full script verify failed at h={h}: {reason}"),
    }
}
