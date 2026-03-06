//! Validate IBD snapshots from BLVM_IBD_SNAPSHOT_DIR.
//!
//! Snapshots are dumped during `./scripts/ibd-profile.sh` at heights 50k, 75k, 100k, ...
//! Run: BLVM_IBD_SNAPSHOT_DIR=/path/to/ibd-snapshots-YYYYMMDD-HHMMSS cargo test -p blvm-consensus --test block_ibd_snapshot_tests -- --ignored

use blvm_consensus::block::connect_block_ibd;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::{Block, Network, UTXO, UtxoSet};
use blvm_consensus::ValidationResult;
use std::path::Path;
use std::sync::Arc;

fn load_dump(
    dir: &Path,
) -> Result<(Block, Vec<Vec<Witness>>, UtxoSet), Box<dyn std::error::Error + Send + Sync>> {
    let block: Block =
        bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(dir.join("block.bin"))?))?;
    let witnesses: Vec<Vec<Witness>> =
        bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(dir.join("witnesses.bin"))?))?;
    // Dump format: HashMap<OutPoint, UTXO> (no Arc). UtxoSet uses Arc<UTXO>.
    let raw: std::collections::HashMap<_, UTXO> =
        bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(dir.join("utxo_set.bin"))?))?;
    let utxo_set: UtxoSet = raw.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    Ok((block, witnesses, utxo_set))
}

#[test]
#[ignore = "Requires BLVM_IBD_SNAPSHOT_DIR with snapshot data from ibd-profile.sh"]
fn block_ibd_snapshot_tests() {
    let base = match std::env::var("BLVM_IBD_SNAPSHOT_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            eprintln!("Skip: BLVM_IBD_SNAPSHOT_DIR not set. Run ibd-profile.sh first.");
            return;
        }
    };
    if !base.exists() {
        eprintln!("Skip: {} does not exist", base.display());
        return;
    }
    let mut heights: Vec<u64> = match std::fs::read_dir(&base) {
        Ok(rd) => rd,
        Err(_) => {
            eprintln!("Skip: Cannot read {}", base.display());
            return;
        }
    }
    .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let s = name.to_str()?;
            if s.starts_with("height_") {
                s.strip_prefix("height_")?.parse().ok()
            } else {
                None
            }
        })
        .collect();
    heights.sort_unstable();
    if heights.is_empty() {
        eprintln!("Skip: No height_* dirs in {}", base.display());
        return;
    }
    for h in heights {
        let dir = base.join(format!("height_{}", h));
        if !dir.join("block.bin").exists() {
            eprintln!("Skip {}: block.bin missing", h);
            continue;
        }
        let (block, mut witnesses, utxo_set) = match load_dump(&dir) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("Skip {}: load failed: {}", h, e);
                continue;
            }
        };
        if witnesses.len() != block.transactions.len() {
            witnesses = block
                .transactions
                .iter()
                .map(|tx| (0..tx.inputs.len()).map(|_| Vec::new()).collect())
                .collect();
        }
        let (result, _new_utxo_set, _tx_ids, _utxo_delta) = connect_block_ibd(
            &block,
            &witnesses,
            utxo_set,
            h,
            None::<&[blvm_consensus::types::BlockHeader]>,
            0u64,
            Network::Mainnet,
            None,
            None,
            Some(std::sync::Arc::new(block.clone())),
            None,
        )
        .expect("connect_block_ibd");
        match result {
            ValidationResult::Valid => {}
            ValidationResult::Invalid(reason) => panic!("Block {} should be valid: {}", h, reason),
        }
    }
}
