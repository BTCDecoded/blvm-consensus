//! Reproduce IBD failure using dumped block data.
//!
//! When IBD fails, the node dumps to $BLVM_IBD_DUMP_DIR/height_{N}/
//! Run: `scripts/ibd_failure_to_repro_test.sh [HEIGHT]` to copy dump to repo.
//!
//! Run test: BLVM_IBD_FAILURE_HEIGHT=N cargo test --test block_ibd_repro -- --ignored
//!
//! Repo layout under `tests/test_data/ibd_failure_height_{h}/`: commit **`info.txt`** only;
//! large `*.bin` payloads are **gitignored** (see workspace `.gitignore`). Populate bins locally
//! or via `scripts/ibd_failure_to_repro_test.sh`.

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
        .expect("Set BLVM_IBD_FAILURE_HEIGHT to the failing block height")
}

fn dump_dir() -> std::path::PathBuf {
    let h = height();
    if let Ok(d) = std::env::var("BLVM_IBD_DUMP_DIR") {
        return std::path::PathBuf::from(d).join(format!("height_{h}"));
    }
    // Repo backup: blvm-consensus/tests/test_data/ibd_failure_height_{h}
    let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/test_data")
        .join(format!("ibd_failure_height_{h}"));
    if repo.join("block.bin").exists() {
        return repo;
    }
    std::env::temp_dir()
        .join("blvm_ibd_failure")
        .join(format!("height_{h}"))
}

/// Dump format: HashMap<OutPoint, UTXO> (no Arc in serialized form)
fn load_dump(
    dir: &Path,
) -> Result<(Block, Vec<Vec<Witness>>, UtxoSet), Box<dyn std::error::Error + Send + Sync>> {
    let block: Block = bincode::deserialize_from(std::io::BufReader::new(std::fs::File::open(
        dir.join("block.bin"),
    )?))?;
    let witnesses: Vec<Vec<Witness>> = bincode::deserialize_from(std::io::BufReader::new(
        std::fs::File::open(dir.join("witnesses.bin"))?,
    ))?;
    let raw: std::collections::HashMap<OutPoint, UTXO> = bincode::deserialize_from(
        std::io::BufReader::new(std::fs::File::open(dir.join("utxo_set.bin"))?),
    )?;
    let utxo_set: UtxoSet = raw.into_iter().map(|(k, v)| (k, Arc::new(v))).collect();
    Ok((block, witnesses, utxo_set))
}

#[test]
#[ignore = "Slow: loads UTXO set; run with --ignored and BLVM_IBD_FAILURE_HEIGHT set"]
fn block_ibd_repro() {
    let h = height();
    let dir = dump_dir();
    if !dir.join("block.bin").exists() {
        eprintln!(
            "Skip: dump not found at {}. Run: ./scripts/ibd_failure_to_repro_test.sh {}",
            dir.display(),
            h
        );
        return;
    }

    let (block, mut witnesses, utxo_set) = load_dump(&dir).expect("load dump");

    // Debug: Check tx 1 (first non-coinbase) prevouts in UTXO set
    if h == 546 && block.transactions.len() > 1 {
        let tx1 = &block.transactions[1];
        eprintln!("Block 546 tx 1: {} inputs", tx1.inputs.len());
        for (i, input) in tx1.inputs.iter().enumerate() {
            let found = utxo_set.get(&input.prevout);
            eprintln!(
                "  input {}: prevout {:02x?}..:{}, in_utxo={}, script_pubkey_len={}",
                i,
                &input.prevout.hash[..4],
                input.prevout.index,
                found.is_some(),
                found.map(|u| u.script_pubkey.len()).unwrap_or(0),
            );
        }
    }

    if witnesses.len() != block.transactions.len() {
        witnesses = block
            .transactions
            .iter()
            .map(|tx| (0..tx.inputs.len()).map(|_| Vec::new()).collect())
            .collect();
    }

    // Diagnostic: individually verify each input of the failing transaction
    eprintln!(
        "Block {}: {} txs, utxo_set size {}",
        h,
        block.transactions.len(),
        utxo_set.len()
    );
    for (tx_idx, tx) in block.transactions.iter().enumerate() {
        if blvm_consensus::transaction::is_coinbase(tx) {
            continue;
        }
        for (inp_idx, input) in tx.inputs.iter().enumerate() {
            let utxo = match utxo_set.get(&input.prevout) {
                Some(u) => u,
                None => {
                    eprintln!("  tx {tx_idx} input {inp_idx}: UTXO not found for prevout");
                    continue;
                }
            };
            let script_pubkey = &utxo.script_pubkey;
            let witness = witnesses.get(tx_idx).and_then(|w| w.get(inp_idx));
            let wit_ref = witness.and_then(|w| if w.is_empty() { None } else { Some(w) });
            let has_witness = witness
                .map(|w| w.iter().any(|x| !x.is_empty()))
                .unwrap_or(false);
            let flags = 0u32; // height 139758 is before BIP16/66/65 activations
            let prevout_values: Vec<i64> = tx
                .inputs
                .iter()
                .map(|i| utxo_set.get(&i.prevout).map(|u| u.value).unwrap_or(0))
                .collect();
            let prevout_scripts: Vec<&[u8]> = tx
                .inputs
                .iter()
                .map(|i| {
                    utxo_set
                        .get(&i.prevout)
                        .map(|u| u.script_pubkey.as_ref())
                        .unwrap_or(&[])
                })
                .collect();
            let result = blvm_consensus::script::verify_script_with_context_full(
                &input.script_sig,
                script_pubkey.as_ref(),
                wit_ref,
                flags,
                tx,
                inp_idx,
                &prevout_values,
                &prevout_scripts,
                Some(h),
                None,
                Network::Mainnet,
                blvm_consensus::script::SigVersion::Base,
                #[cfg(feature = "production")]
                None, // schnorr_collector
                #[cfg(not(feature = "production"))]
                None,
                None, // precomputed_bip143
                #[cfg(feature = "production")]
                None, // precomputed_sighash_all
                #[cfg(feature = "production")]
                None, // sighash_cache
                #[cfg(feature = "production")]
                None, // precomputed_p2pkh_hash
            );
            match &result {
                Ok(valid) if !valid => {
                    eprintln!(
                        "  tx {tx_idx} input {inp_idx}: SCRIPT INVALID (verify returned false)"
                    );
                    eprintln!(
                        "    script_pubkey ({} bytes): {:02x?}",
                        script_pubkey.len(),
                        &script_pubkey[..script_pubkey.len().min(40)]
                    );
                    eprintln!(
                        "    script_sig ({} bytes): {:02x?}",
                        input.script_sig.len(),
                        &input.script_sig[..input.script_sig.len().min(40)]
                    );
                    if let Some(w) = witness {
                        eprintln!("    witness: {} items", w.len());
                    }
                }
                Err(e) => {
                    eprintln!("  tx {tx_idx} input {inp_idx}: SCRIPT ERROR: {e}");
                }
                Ok(true) => {
                    if tx_idx == 8 {
                        eprintln!(
                            "  tx {} input {}: OK (spk {} bytes)",
                            tx_idx,
                            inp_idx,
                            script_pubkey.len()
                        );
                    }
                }
                _ => {}
            }
        }
    }
    // Print all prevouts for transactions with ≥2 inputs (to identify which "input 1" might fail)
    eprintln!(
        "--- Transactions with multiple inputs (to identify 'input 1' in skip_signatures path) ---"
    );
    for (tx_idx, tx) in block.transactions.iter().enumerate() {
        if blvm_consensus::transaction::is_coinbase(tx) {
            continue;
        }
        if tx.inputs.len() >= 2 {
            for (inp_idx, input) in tx.inputs.iter().enumerate() {
                let found = utxo_set.get(&input.prevout).is_some();
                eprintln!(
                    "  tx {} input {}: prevout {:02x?}..:{} in_utxo_set={}",
                    tx_idx,
                    inp_idx,
                    &input.prevout.hash[..4],
                    input.prevout.index,
                    found,
                );
            }
        }
    }
    eprintln!("--- Individual verification complete, now running connect_block_ibd ---");

    // Use current time so two_week_ok=true, enabling skip_signatures if BLVM_ASSUME_VALID_HEIGHT is set
    let network_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ctx = blvm_consensus::block::block_validation_context_for_connect_ibd(
        None::<&[blvm_consensus::types::BlockHeader]>,
        network_time,
        Network::Mainnet,
    );
    let (result, _new_utxo_set, _tx_ids, _utxo_delta) = connect_block_ibd(
        &block,
        &witnesses,
        utxo_set,
        h,
        &ctx,
        None,
        None,
        Some(std::sync::Arc::new(block.clone())),
        None,
    )
    .expect("connect_block_ibd");

    match result {
        ValidationResult::Valid => {}
        ValidationResult::Invalid(reason) => panic!("Block {h} should be valid: {reason}"),
    }
}
