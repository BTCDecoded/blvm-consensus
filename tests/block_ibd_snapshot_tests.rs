//! Validate and benchmark IBD snapshots.
//!
//! Correctness: cargo test -p blvm-consensus --test block_ibd_snapshot_tests --features production -- --ignored
//! Benchmark:   cargo test -p blvm-consensus --test block_ibd_snapshot_tests --features production --release -- --ignored bench_ibd_snapshots --nocapture
//!
//! Set BLVM_IBD_SNAPSHOT_DIR or defaults to ../ibd-snapshots-20260307-192410 relative to blvm-consensus.

use blvm_consensus::block::connect_block_ibd;
use blvm_consensus::segwit::Witness;
use blvm_consensus::types::{Block, Network, UtxoSet, UTXO};
use blvm_consensus::ValidationResult;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn snapshot_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("BLVM_IBD_SNAPSHOT_DIR") {
        let p = PathBuf::from(d);
        if p.exists() {
            return Some(p);
        }
    }
    let default =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ibd-snapshots-20260307-192410");
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

fn load_dump(
    dir: &Path,
) -> Result<(Block, Vec<Vec<Witness>>, UtxoSet), Box<dyn std::error::Error + Send + Sync>> {
    let block: Block = bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(
        dir.join("block.bin"),
    )?))?;
    let witnesses: Vec<Vec<Witness>> = bincode::deserialize_from(std::io::BufReader::new(
        std::fs::File::open(dir.join("witnesses.bin"))?,
    ))?;
    let raw: std::collections::HashMap<_, UTXO> = bincode::deserialize_from(
        std::io::BufReader::new(std::fs::File::open(dir.join("utxo_set.bin"))?),
    )?;
    let utxo_set: UtxoSet = raw.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    Ok((block, witnesses, utxo_set))
}

fn prepare(
    dir: &Path,
) -> Option<(
    Block,
    Arc<Block>,
    Vec<Vec<Witness>>,
    Arc<Vec<Vec<Witness>>>,
    UtxoSet,
)> {
    if !dir.join("block.bin").exists() {
        return None;
    }
    let (block, mut witnesses, utxo_set) = load_dump(dir).ok()?;
    if witnesses.len() != block.transactions.len() {
        witnesses = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::new()).collect())
            .collect();
    }
    let block_arc = Arc::new(block.clone());
    let witnesses_arc = Arc::new(witnesses.clone());
    Some((block, block_arc, witnesses, witnesses_arc, utxo_set))
}

#[inline(never)]
fn validate_once(
    block: &Block,
    witnesses: &[Vec<Witness>],
    utxo_set: UtxoSet,
    height: u64,
    block_arc: &Arc<Block>,
) -> f64 {
    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        0u64,
        Network::Mainnet,
    );
    let t = Instant::now();
    let (result, _new_utxo, _txids, _delta) = connect_block_ibd(
        block,
        witnesses,
        utxo_set,
        height,
        &ctx,
        None,
        None,
        Some(Arc::clone(block_arc)),
        None,
    )
    .expect("connect_block_ibd");
    let elapsed = t.elapsed().as_secs_f64() * 1000.0;
    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => panic!("height {} invalid: {}", height, reason),
    }
    elapsed
}

/// Measure only the connect_block_ibd call time, excluding return value drop.
/// Passes precomputed tx_ids and witnesses_arc to match real IBD.
#[inline(never)]
fn validate_call_only(
    block: &Block,
    witnesses: &[Vec<Witness>],
    utxo_set: UtxoSet,
    height: u64,
    block_arc: &Arc<Block>,
    precomputed_tx_ids: Option<&[blvm_consensus::types::Hash]>,
    witnesses_arc: Option<&Arc<Vec<Vec<Witness>>>>,
) -> (f64, UtxoSet) {
    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        0u64,
        Network::Mainnet,
    );
    let t = Instant::now();
    let (result, new_utxo, _txids, _delta) = connect_block_ibd(
        block,
        witnesses,
        utxo_set,
        height,
        &ctx,
        None,
        precomputed_tx_ids,
        Some(Arc::clone(block_arc)),
        witnesses_arc,
    )
    .expect("connect_block_ibd");
    let elapsed = t.elapsed().as_secs_f64() * 1000.0;
    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => panic!("height {} invalid: {}", height, reason),
    }
    (elapsed, new_utxo)
}

#[inline(never)]
fn validate_once_timed(
    block: &Block,
    witnesses: &[Vec<Witness>],
    utxo_set: UtxoSet,
    height: u64,
    block_arc: &Arc<Block>,
) -> f64 {
    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        0u64,
        Network::Mainnet,
    );
    let t0 = Instant::now();
    let clone_done = Instant::now(); // utxo_set already cloned by caller
    let (result, _new_utxo, _txids, _delta) = connect_block_ibd(
        block,
        witnesses,
        utxo_set,
        height,
        &ctx,
        None,
        None,
        Some(Arc::clone(block_arc)),
        None,
    )
    .expect("connect_block_ibd");
    let call_done = Instant::now();
    let drop_start = Instant::now();
    drop(_new_utxo);
    drop(_txids);
    drop(_delta);
    let drop_done = Instant::now();
    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => panic!("height {} invalid: {}", height, reason),
    }
    let total = t0.elapsed().as_secs_f64() * 1000.0;
    let call_ms = (call_done - clone_done).as_secs_f64() * 1000.0;
    let drop_ms = (drop_done - drop_start).as_secs_f64() * 1000.0;
    eprintln!(
        "  [TIMING] h={} total={:.2}ms call={:.2}ms drop={:.2}ms",
        height, total, call_ms, drop_ms
    );
    total
}

#[inline(never)]
fn validate_with_txids(
    block: &Block,
    witnesses: &[Vec<Witness>],
    utxo_set: UtxoSet,
    height: u64,
    block_arc: &Arc<Block>,
    tx_ids: &[blvm_consensus::types::Hash],
) -> f64 {
    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        0u64,
        Network::Mainnet,
    );
    let t = Instant::now();
    let (result, _, _, _) = connect_block_ibd(
        block,
        witnesses,
        utxo_set,
        height,
        &ctx,
        None,
        Some(tx_ids),
        Some(Arc::clone(block_arc)),
        None,
    )
    .expect("connect_block_ibd");
    let elapsed = t.elapsed().as_secs_f64() * 1000.0;
    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => panic!("height {} invalid: {}", height, reason),
    }
    elapsed
}

#[test]
#[ignore = "Requires snapshot data"]
fn block_ibd_snapshot_tests() {
    let base = match snapshot_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skip: no snapshot dir");
            return;
        }
    };
    let mut heights: Vec<u64> = std::fs::read_dir(&base)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.file_name()
                .to_str()?
                .strip_prefix("height_")?
                .parse()
                .ok()
        })
        .collect();
    heights.sort_unstable();
    for h in heights {
        let dir = base.join(format!("height_{}", h));
        let (block, block_arc, witnesses, _witnesses_arc, utxo_set) = match prepare(&dir) {
            Some(x) => x,
            None => continue,
        };
        validate_once(&block, &witnesses, utxo_set, h, &block_arc);
        eprintln!("  height={}: OK", h);
    }
}

#[test]
#[ignore = "Requires snapshot data"]
fn bench_ibd_snapshots() {
    let base = match snapshot_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skip: no snapshot dir");
            return;
        }
    };
    let iterations: u32 = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let height_filter: Option<u64> = std::env::var("BENCH_HEIGHT")
        .ok()
        .and_then(|s| s.parse().ok());

    let mut heights: Vec<u64> = std::fs::read_dir(&base)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.file_name()
                .to_str()?
                .strip_prefix("height_")?
                .parse()
                .ok()
        })
        .collect();
    heights.sort_unstable();
    if let Some(hf) = height_filter {
        heights.retain(|h| *h == hf);
    }

    eprintln!("=== IBD Snapshot Benchmark ({} iters) ===", iterations);
    eprintln!("height,txs,inputs,min_ms,median_ms,mean_ms,p95_ms,max_ms,bps");

    let mut focus_medians: Vec<f64> = Vec::new();

    for &h in &heights {
        let dir = base.join(format!("height_{}", h));
        let (block, block_arc, witnesses, witnesses_arc, utxo_set_template) = match prepare(&dir) {
            Some(x) => x,
            None => continue,
        };
        let n_txs = block.transactions.len();
        let n_inputs: usize = block.transactions.iter().map(|tx| tx.inputs.len()).sum();

        eprintln!("  h={}: utxo_set entries={}", h, utxo_set_template.len());

        let tx_ids = blvm_consensus::block::compute_block_tx_ids(&block);

        // warmup
        let (_warmup_ms, warmup_utxo) = validate_call_only(
            &block,
            &witnesses,
            utxo_set_template.clone(),
            h,
            &block_arc,
            Some(&tx_ids),
            Some(&witnesses_arc),
        );
        drop(warmup_utxo);

        let mut times: Vec<f64> = Vec::with_capacity(iterations as usize);
        for _ in 0..iterations {
            let (ms, returned_utxo) = validate_call_only(
                &block,
                &witnesses,
                utxo_set_template.clone(),
                h,
                &block_arc,
                Some(&tx_ids),
                Some(&witnesses_arc),
            );
            times.push(ms);
            drop(returned_utxo);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = times[0];
        let median = times[times.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let p95 = times[(times.len() as f64 * 0.95) as usize];
        let max = *times.last().unwrap();
        let bps = 1000.0 / median;

        eprintln!(
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.0}",
            h, n_txs, n_inputs, min, median, mean, p95, max, bps
        );

        if h >= 100_000 {
            focus_medians.push(median);
        }
    }

    if !focus_medians.is_empty() {
        let avg = focus_medians.iter().sum::<f64>() / focus_medians.len() as f64;
        eprintln!("\n=== 100k+ Summary ===");
        eprintln!("avg median={:.2}ms ({:.0} bps)", avg, 1000.0 / avg);
        eprintln!("4x target={:.2}ms ({:.0} bps)", avg / 4.0, 4000.0 / avg);
        eprintln!("5x target={:.2}ms ({:.0} bps)", avg / 5.0, 5000.0 / avg);
    }
}

#[test]
#[ignore = "Requires snapshot data"]
fn bench_ibd_snapshots_no_txid() {
    let base = match snapshot_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skip: no snapshot dir");
            return;
        }
    };
    let iterations: u32 = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let mut heights: Vec<u64> = std::fs::read_dir(&base)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            e.file_name()
                .to_str()?
                .strip_prefix("height_")?
                .parse()
                .ok()
        })
        .collect();
    heights.sort_unstable();
    heights.retain(|h| *h >= 100_000);

    eprintln!(
        "=== IBD Validation-Only Benchmark ({} iters, precomputed tx_ids) ===",
        iterations
    );
    eprintln!("height,txs,inputs,min_ms,median_ms,mean_ms,max_ms,bps");

    let mut focus_medians: Vec<f64> = Vec::new();

    for &h in &heights {
        let dir = base.join(format!("height_{}", h));
        let (block, block_arc, witnesses, _witnesses_arc, utxo_set_template) = match prepare(&dir) {
            Some(x) => x,
            None => continue,
        };
        let n_txs = block.transactions.len();
        let n_inputs: usize = block.transactions.iter().map(|tx| tx.inputs.len()).sum();

        let tx_ids = blvm_consensus::block::compute_block_tx_ids(&block);
        std::thread::sleep(std::time::Duration::from_millis(50));

        let _ = validate_with_txids(
            &block,
            &witnesses,
            utxo_set_template.clone(),
            h,
            &block_arc,
            &tx_ids,
        );

        let mut times: Vec<f64> = (0..iterations)
            .map(|_| {
                validate_with_txids(
                    &block,
                    &witnesses,
                    utxo_set_template.clone(),
                    h,
                    &block_arc,
                    &tx_ids,
                )
            })
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = times[0];
        let median = times[times.len() / 2];
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let max = *times.last().unwrap();
        let bps = 1000.0 / median;

        eprintln!(
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.0}",
            h, n_txs, n_inputs, min, median, mean, max, bps
        );
        focus_medians.push(median);
    }

    if !focus_medians.is_empty() {
        let avg = focus_medians.iter().sum::<f64>() / focus_medians.len() as f64;
        eprintln!("\n=== Validation-Only Summary ===");
        eprintln!("avg median={:.2}ms ({:.0} bps)", avg, 1000.0 / avg);
    }
}
