//! Mining and block creation functions from Orange Paper Section 10.1

#[cfg(test)]
use crate::economic::get_block_subsidy;
use crate::economic::get_block_subsidy_for_network;
use crate::error::Result;
use crate::pow::{check_proof_of_work, next_required_bits};
use crate::transaction::check_transaction;
use crate::types::*;
#[cfg(feature = "spec-lock")]
use blvm_spec_lock::spec_locked;

#[cfg(test)]
use crate::transaction::is_coinbase;

/// Compact bits for a block template.
///
/// `prev_headers` is oldest-to-newest and ends at `prev_header`'s height.
/// Height 0 uses the network proof-of-work limit.
fn template_bits(
    network: Network,
    height: Natural,
    block_timestamp: u64,
    parent: &BlockHeader,
    prev_headers: &[BlockHeader],
) -> Result<Natural> {
    if height == 0 {
        return Ok(match network {
            Network::Regtest => 0x207fffff,
            Network::Signet => 0x1e0377ae,
            Network::Mainnet | Network::Testnet | Network::Testnet4 => {
                crate::constants::MAX_TARGET as u64
            }
        });
    }
    let parent_height = height - 1;
    let window_end = parent_height;
    let window_start = if prev_headers.is_empty() {
        window_end
    } else {
        window_end.saturating_sub(prev_headers.len() as u64 - 1)
    };
    next_required_bits(network, height, block_timestamp, &|h| {
        if h == parent_height {
            return Some((parent.bits, parent.timestamp));
        }
        if prev_headers.is_empty() || h < window_start || h > window_end {
            return None;
        }
        prev_headers
            .get((h - window_start) as usize)
            .map(|hdr| (hdr.bits, hdr.timestamp))
    })?
    .ok_or_else(|| {
        crate::error::ConsensusError::InvalidProofOfWork(
            "genesis template has no parent bits".into(),
        )
    })
}

/// Minimum header version `check_bip90` accepts at `height`.
///
/// Height 1 on mainnet is still version 1. After BIP65 the minimum is 4.
pub fn template_block_version(network: Network, height: Natural) -> i64 {
    use crate::activation::{ForkActivationTable, IsForkActive};
    let table = ForkActivationTable::from_network(network);
    if table.is_fork_active(ForkId::Bip65, height) {
        4
    } else if table.is_fork_active(ForkId::Bip66, height) {
        3
    } else if table.is_fork_active(ForkId::Bip34, height) {
        2
    } else {
        1
    }
}

/// CreateNewBlock: 𝒰𝒮 × 𝒯𝒳* → ℬ
///
/// For UTXO set us and mempool transactions txs:
/// 1. Create coinbase transaction with appropriate subsidy
/// 2. Select transactions from mempool based on fee rate
/// 3. Calculate merkle root
/// 4. Create block header with appropriate difficulty
/// 5. Return new block
pub fn create_new_block(
    utxo_set: &UtxoSet,
    mempool_txs: &[Transaction],
    height: Natural,
    prev_header: &BlockHeader,
    prev_headers: &[BlockHeader],
    coinbase_script: &ByteString,
    coinbase_address: &ByteString,
) -> Result<Block> {
    // For backward compatibility, derive block_time from system clock here.
    let block_time = get_current_timestamp();
    create_new_block_with_time(
        utxo_set,
        mempool_txs,
        height,
        prev_header,
        prev_headers,
        coinbase_script,
        coinbase_address,
        block_time,
        Network::Mainnet,
        None,
    )
}

/// CreateNewBlock variant that accepts an explicit block_time.
///
/// This allows callers (e.g., node layer) to provide a median time-past or
/// adjusted network time instead of relying on `SystemTime::now()` inside
/// consensus code.
#[allow(clippy::too_many_arguments)]
pub fn create_new_block_with_time(
    utxo_set: &UtxoSet,
    mempool_txs: &[Transaction],
    height: Natural,
    prev_header: &BlockHeader,
    prev_headers: &[BlockHeader],
    coinbase_script: &ByteString,
    coinbase_address: &ByteString,
    block_time: u64,
    network: Network,
    mempool_witnesses: Option<&[Option<Vec<crate::segwit::Witness>>]>,
) -> Result<Block> {
    use crate::bip113::get_median_time_past;
    use crate::mempool::{Mempool, MempoolResult, accept_to_memory_pool};

    let median_time_past = get_median_time_past(prev_headers);
    let block_time = block_time.max(median_time_past.saturating_add(1));

    // 1. Create coinbase transaction
    let coinbase_tx = create_coinbase_transaction(
        height,
        get_block_subsidy_for_network(height, network),
        coinbase_script,
        coinbase_address,
    )?;

    // Header (80 bytes plus a 3-byte transaction count) and the coinbase are in the
    // block before any mempool transaction. The witness commitment is appended later.
    const WITNESS_COMMITMENT_RESERVE: u64 = 4_000;
    let header_weight = 83u64.saturating_mul(4);
    let coinbase_weight =
        crate::segwit::transaction_weight_from_stacks(&coinbase_tx, None).unwrap_or(0);
    let mut running_weight = header_weight
        .saturating_add(coinbase_weight)
        .saturating_add(WITNESS_COMMITMENT_RESERVE);
    // Block sigop cost counts witness programs. P2SH is 0x01. Witness is 0x800.
    const BLOCK_SIGOP_FLAGS: u32 = 0x01 | 0x800;
    let mut running_sigops = crate::sigop::get_transaction_sigop_cost_with_witness_slices(
        &coinbase_tx,
        utxo_set,
        None,
        BLOCK_SIGOP_FLAGS,
    )
    .unwrap_or(0);
    let max_weight = crate::constants::MAX_BLOCK_WEIGHT as u64;
    let max_sigops = crate::constants::MAX_BLOCK_SIGOPS_COST;

    // 2. Select transactions from mempool with proper validation.
    // Outputs of a selected transaction are spendable by a later one in this template.
    // A child that sorts ahead of its parent is tried again after that parent is selected.
    let mut selected_txs = Vec::new();
    let mut temp_mempool = Mempool::new();
    let mut available = utxo_set.clone();
    let mut waiting: Vec<usize> = (0..mempool_txs.len()).collect();

    while !waiting.is_empty() {
        let mut still_waiting = Vec::new();
        let selected_before = selected_txs.len();
        for idx in waiting {
            let tx = &mempool_txs[idx];
            if check_transaction(tx)? != ValidationResult::Valid {
                continue;
            }

            let time_context = Some(TimeContext {
                network_time: block_time,
                median_time_past,
            });
            let tx_witnesses = mempool_witnesses
                .and_then(|all| all.get(idx))
                .and_then(|w| w.as_deref());
            match accept_to_memory_pool(
                tx,
                tx_witnesses,
                &available,
                &temp_mempool,
                height,
                time_context,
                network,
                None,
            )? {
                MempoolResult::Accepted => {
                    let tx_weight =
                        match crate::segwit::transaction_weight_from_stacks(tx, tx_witnesses) {
                            Ok(weight) => weight,
                            Err(_) => continue,
                        };
                    let tx_sigops =
                        match crate::sigop::get_transaction_sigop_cost_with_witness_slices(
                            tx,
                            &available,
                            tx_witnesses,
                            BLOCK_SIGOP_FLAGS,
                        ) {
                            Ok(cost) => cost,
                            Err(_) => continue,
                        };
                    if running_weight.saturating_add(tx_weight) > max_weight
                        || running_sigops.saturating_add(tx_sigops) > max_sigops
                    {
                        continue;
                    }
                    running_weight = running_weight.saturating_add(tx_weight);
                    running_sigops = running_sigops.saturating_add(tx_sigops);
                    temp_mempool.insert_transaction_with_witness(tx, tx_witnesses);
                    let txid = crate::block::calculate_tx_id(tx);
                    for (index, output) in tx.outputs.iter().enumerate() {
                        available.insert(
                            OutPoint {
                                hash: txid,
                                index: index as u32,
                            },
                            std::sync::Arc::new(UTXO {
                                value: output.value,
                                script_pubkey: output.script_pubkey.clone().into(),
                                height,
                                is_coinbase: false,
                            }),
                        );
                    }
                    selected_txs.push(tx.clone());
                }
                MempoolResult::Rejected(_reason) => {
                    // The parent may appear later in this fee order.
                    #[cfg(test)]
                    eprintln!("Transaction rejected: {_reason}");
                    still_waiting.push(idx);
                }
            }
        }
        if selected_txs.len() == selected_before {
            break;
        }
        waiting = still_waiting;
    }

    // 3. Build transaction list (coinbase first)
    let mut transactions = vec![coinbase_tx];
    transactions.extend(selected_txs);

    // 4. Calculate merkle root
    let merkle_root = calculate_merkle_root(&transactions)?;

    // 5. Required compact bits for this height. Off a period boundary this is the
    // parent's bits (or the minimum-difficulty exception). The retarget formula
    // runs only when this height starts a new difficulty period.
    let next_work = template_bits(network, height, block_time, prev_header, prev_headers)?;

    // 6. Create block header
    let header = BlockHeader {
        version: template_block_version(network, height),
        prev_block_hash: calculate_block_hash(prev_header),
        merkle_root,
        timestamp: block_time,
        bits: next_work,
        nonce: 0, // Will be set during mining
    };

    Ok(Block {
        header,
        transactions: transactions.into_boxed_slice(),
    })
}

/// MineBlock: ℬ × ℕ → ℬ × {success, failure}
///
/// Attempt to mine a block by finding a valid nonce:
/// 1. Try different nonce values
/// 2. Check if resulting hash meets difficulty target
/// 3. Return mined block or failure
#[track_caller] // Better error messages showing caller location
pub fn mine_block(mut block: Block, max_attempts: Natural) -> Result<(Block, MiningResult)> {
    for nonce in 0..max_attempts {
        block.header.nonce = nonce;

        if check_proof_of_work(&block.header).unwrap_or(false) {
            return Ok((block, MiningResult::Success));
        }
    }

    Ok((block, MiningResult::Failure))
}

/// BlockTemplate: Interface for mining software
///
/// Provides a template for mining software to work with:
/// 1. Block header with current difficulty
/// 2. Coinbase transaction template
/// 3. Selected transactions
/// 4. Mining parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockTemplate {
    pub header: BlockHeader,
    pub coinbase_tx: Transaction,
    pub transactions: Vec<Transaction>,
    pub target: crate::pow::U256,
    pub height: Natural,
    pub timestamp: Natural,
}

/// Create a block template for mining
pub fn create_block_template(
    utxo_set: &UtxoSet,
    mempool_txs: &[Transaction],
    height: Natural,
    prev_header: &BlockHeader,
    prev_headers: &[BlockHeader],
    coinbase_script: &ByteString,
    coinbase_address: &ByteString,
    network: Network,
    mempool_witnesses: Option<&[Option<Vec<crate::segwit::Witness>>]>,
) -> Result<BlockTemplate> {
    let block_time = get_current_timestamp();
    let block = create_new_block_with_time(
        utxo_set,
        mempool_txs,
        height,
        prev_header,
        prev_headers,
        coinbase_script,
        coinbase_address,
        block_time,
        network,
        mempool_witnesses,
    )?;

    let target = crate::pow::expand_target(block.header.bits)?;

    let header = block.header.clone();

    // Use proven bounds for coinbase access (block has been validated)
    #[cfg(feature = "production")]
    let coinbase_tx = {
        use crate::optimizations::_optimized_access::get_proven_by_;
        get_proven_by_(&block.transactions, 0)
            .ok_or_else(|| {
                crate::error::ConsensusError::BlockValidation("Block has no transactions".into())
            })?
            .clone()
    };

    #[cfg(not(feature = "production"))]
    let coinbase_tx = block.transactions[0].clone();

    Ok(BlockTemplate {
        header: block.header.clone(),
        coinbase_tx,
        transactions: block.transactions[1..].to_vec(),
        target,
        height,
        timestamp: header.timestamp,
    })
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Result of mining attempt
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MiningResult {
    Success,
    Failure,
}

/// Create coinbase transaction
/// Orange Paper 12.2: Coinbase transaction structure.
/// BIP54: When BIP54 is active, coinbase must have nLockTime = height - 1 and nSequence != 0xffff_ffff.
/// This implementation sets those so that blocks are valid under BIP54 when activated.
fn create_coinbase_transaction(
    height: Natural,
    subsidy: Integer,
    script: &ByteString,
    address: &ByteString,
) -> Result<Transaction> {
    create_coinbase_with_outputs(height, script, &[(subsidy, address.clone())])
}

/// Multi-output coinbase. Callers set values (subsidy + fees). Empty list is an error.
/// Witness commitment is appended by `create_block_template_with_outputs` / Stratum.
pub fn create_coinbase_with_outputs(
    height: Natural,
    script: &ByteString,
    outputs: &[(Integer, ByteString)],
) -> Result<Transaction> {
    if outputs.is_empty() {
        return Err(crate::error::ConsensusError::InvalidProofOfWork(
            "coinbase requires at least one output".into(),
        ));
    }
    let lock_time = height.saturating_sub(1);
    let coinbase_input = TransactionInput {
        prevout: OutPoint {
            hash: [0u8; 32],
            index: 0xffffffff,
        },
        script_sig: script.clone(),
        sequence: 0xfffffffe, // BIP54: not 0xffffffff so coinbase is unique
    };
    let outs: Vec<TransactionOutput> = outputs
        .iter()
        .map(|(value, script_pubkey)| TransactionOutput {
            value: *value,
            script_pubkey: script_pubkey.clone(),
        })
        .collect();

    Ok(Transaction {
        version: 1,
        inputs: crate::tx_inputs![coinbase_input],
        outputs: outs.into(),
        lock_time,
    })
}

/// Same as `create_block_template`, then replace the coinbase and merkle root.
#[allow(clippy::too_many_arguments)]
pub fn create_block_template_with_outputs(
    utxo_set: &UtxoSet,
    mempool_txs: &[Transaction],
    height: Natural,
    prev_header: &BlockHeader,
    prev_headers: &[BlockHeader],
    coinbase_script: &ByteString,
    coinbase_outputs: &[(Integer, ByteString)],
    network: Network,
    mempool_witnesses: Option<&[Option<Vec<crate::segwit::Witness>>]>,
) -> Result<BlockTemplate> {
    let address = coinbase_outputs
        .first()
        .map(|(_, s)| s.clone())
        .unwrap_or_default();
    let mut tmpl = create_block_template(
        utxo_set,
        mempool_txs,
        height,
        prev_header,
        prev_headers,
        coinbase_script,
        &address,
        network,
        mempool_witnesses,
    )?;
    let subsidy = get_block_subsidy_for_network(height, network);
    let fees = sum_selected_fees(utxo_set, &tmpl.transactions);
    let fitted = fit_payouts_to_reward(coinbase_outputs, subsidy, fees)?;
    tmpl.coinbase_tx = create_coinbase_with_outputs(height, coinbase_script, &fitted)?;
    let mut txs = Vec::with_capacity(1 + tmpl.transactions.len());
    txs.push(tmpl.coinbase_tx.clone());
    txs.extend(tmpl.transactions.iter().cloned());
    let nested = nested_witnesses_for_selected(&txs, mempool_txs, mempool_witnesses)?;
    append_witness_commitment_from_nested(&mut tmpl.coinbase_tx, &mut txs, Some(&nested))?;
    tmpl.header.merkle_root = calculate_merkle_root(&txs)?;
    Ok(tmpl)
}

/// Resolve named txids from a mempool list, in declaration order.
/// Unknown or duplicate txids are errors. An empty list is allowed (coinbase-only).
pub fn resolve_declared_txs(
    mempool_txs: &[Transaction],
    txids: &[Hash],
) -> Result<Vec<Transaction>> {
    use crate::block::calculate_tx_id;
    use std::collections::{HashMap, HashSet};

    let mut by_id: HashMap<Hash, &Transaction> = HashMap::new();
    for tx in mempool_txs {
        by_id.entry(calculate_tx_id(tx)).or_insert(tx);
    }
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(txids.len());
    for id in txids {
        if !seen.insert(*id) {
            return Err(crate::error::ConsensusError::BlockValidation(
                format!("duplicate declared txid {}", hex::encode(id)).into(),
            ));
        }
        let tx = by_id.get(id).ok_or_else(|| {
            crate::error::ConsensusError::BlockValidation(
                format!("unknown declared txid {}", hex::encode(id)).into(),
            )
        })?;
        out.push((*tx).clone());
    }
    Ok(out)
}

fn declared_tx_spends_witness_utxo(tx: &Transaction, utxo_set: &UtxoSet) -> bool {
    tx.inputs.iter().any(|input| {
        utxo_set.get(&input.prevout).is_some_and(|utxo| {
            crate::witness::is_standard_witness_program(utxo.script_pubkey.as_ref())
        })
    })
}

/// Align mempool witness slots to the declared subset by txid.
/// A witness spend without stored stacks is an error (no empty-stack BIP141).
fn align_declared_witnesses(
    declared: &[Transaction],
    mempool_txs: &[Transaction],
    mempool_witnesses: Option<&[Option<Vec<crate::segwit::Witness>>]>,
    utxo_set: &UtxoSet,
) -> Result<Vec<Option<Vec<crate::segwit::Witness>>>> {
    use crate::block::calculate_tx_id;
    use std::collections::HashMap;

    let mut by_txid = HashMap::new();
    if let Some(wits) = mempool_witnesses {
        for (tx, w) in mempool_txs.iter().zip(wits.iter()) {
            by_txid.insert(calculate_tx_id(tx), w.clone());
        }
    }
    let mut out = Vec::with_capacity(declared.len());
    for tx in declared {
        let id = calculate_tx_id(tx);
        let slot = by_txid.get(&id).cloned().flatten();
        if let Some(ref stacks) = slot {
            if stacks.len() != tx.inputs.len() {
                return Err(crate::error::ConsensusError::BlockValidation(
                    format!(
                        "witness count {} != input count {} for declared tx {}",
                        stacks.len(),
                        tx.inputs.len(),
                        hex::encode(id)
                    )
                    .into(),
                ));
            }
        } else if declared_tx_spends_witness_utxo(tx, utxo_set) {
            return Err(crate::error::ConsensusError::BlockValidation(
                format!(
                    "missing mempool witnesses for declared witness spend {}",
                    hex::encode(id)
                )
                .into(),
            ));
        }
        // A legacy spend has no witness. Store empty stacks so the commitment
        // lookup finds this txid. A None slot stays an error for a witness spend.
        let slot = slot.or_else(|| Some(vec![Vec::new(); tx.inputs.len()]));
        out.push(slot);
    }
    Ok(out)
}

/// Stage 3b slice 1: miner-declared mempool subset.
/// Resolves `declared_txids`, builds the Stage 3a template, and errors if the
/// selected non-coinbase txids are not exactly that list (no silent skip).
/// Empty `declared_txids` is coinbase-only. Spec-locked builders are unchanged.
/// Overweight uses Bitcoin `MAX_BLOCK_WEIGHT` (not a pool-invented cap).
#[allow(clippy::too_many_arguments)]
pub fn create_block_template_declared(
    utxo_set: &UtxoSet,
    mempool_txs: &[Transaction],
    declared_txids: &[Hash],
    height: Natural,
    prev_header: &BlockHeader,
    prev_headers: &[BlockHeader],
    coinbase_script: &ByteString,
    coinbase_outputs: &[(Integer, ByteString)],
    network: Network,
    mempool_witnesses: Option<&[Option<Vec<crate::segwit::Witness>>]>,
) -> Result<BlockTemplate> {
    use crate::block::calculate_tx_id;

    let declared = resolve_declared_txs(mempool_txs, declared_txids)?;
    let aligned = align_declared_witnesses(&declared, mempool_txs, mempool_witnesses, utxo_set)?;
    let tmpl = create_block_template_with_outputs(
        utxo_set,
        &declared,
        height,
        prev_header,
        prev_headers,
        coinbase_script,
        coinbase_outputs,
        network,
        Some(aligned.as_slice()),
    )?;
    let selected: Vec<Hash> = tmpl.transactions.iter().map(calculate_tx_id).collect();
    if selected.as_slice() != declared_txids {
        return Err(crate::error::ConsensusError::BlockValidation(
            "declared transaction was not selected (rejected or skipped)".into(),
        ));
    }
    check_template_weight(&tmpl, crate::constants::MAX_BLOCK_WEIGHT as u64)?;
    Ok(tmpl)
}

/// Weight of coinbase + selected txs (empty stacks if none stored).
/// Uses Bitcoin `MAX_BLOCK_WEIGHT` (4_000_000). Not an invented pool cap.
pub fn check_template_weight(tmpl: &BlockTemplate, max_weight: u64) -> Result<()> {
    let mut txs = Vec::with_capacity(1 + tmpl.transactions.len());
    txs.push(tmpl.coinbase_tx.clone());
    txs.extend(tmpl.transactions.iter().cloned());
    let block = Block {
        header: tmpl.header.clone(),
        transactions: txs.clone().into_boxed_slice(),
    };
    let nested: Vec<Vec<crate::segwit::Witness>> = txs
        .iter()
        .map(|tx| vec![Vec::new(); tx.inputs.len()])
        .collect();
    let weight = crate::segwit::calculate_block_weight_from_nested(&block, &nested)?;
    if weight > max_weight {
        return Err(crate::error::ConsensusError::BlockValidation(
            format!("declared template weight {weight} exceeds max {max_weight}").into(),
        ));
    }
    Ok(())
}

/// Coinbase empty stacks, then mempool witnesses aligned to selected txs by txid.
fn nested_witnesses_for_selected(
    txs: &[Transaction],
    mempool_txs: &[Transaction],
    mempool_witnesses: Option<&[Option<Vec<crate::segwit::Witness>>]>,
) -> Result<Vec<Vec<crate::segwit::Witness>>> {
    use crate::block::calculate_tx_id;
    use std::collections::HashMap;

    let mut by_txid = HashMap::new();
    if let Some(wits) = mempool_witnesses {
        for (tx, w) in mempool_txs.iter().zip(wits.iter()) {
            if let Some(stacks) = w {
                by_txid.insert(calculate_tx_id(tx), stacks.clone());
            }
        }
    }
    let mut nested = Vec::with_capacity(txs.len());
    for (i, tx) in txs.iter().enumerate() {
        if i == 0 || mempool_witnesses.is_none() {
            nested.push(vec![Vec::new(); tx.inputs.len()]);
            continue;
        }
        let stacks = by_txid.get(&calculate_tx_id(tx)).cloned().ok_or_else(|| {
            crate::error::ConsensusError::BlockValidation(
                "missing witness for selected transaction".into(),
            )
        })?;
        nested.push(stacks);
    }
    Ok(nested)
}

/// Fees of selected (non-coinbase) txs.
/// An input missing from the chain set is priced from an earlier selected transaction.
/// Any other missing input contributes nothing.
pub fn sum_selected_fees(utxo_set: &UtxoSet, txs: &[Transaction]) -> Integer {
    use crate::block::calculate_tx_id;

    let mut created: std::collections::HashMap<Hash, usize> = std::collections::HashMap::new();
    let mut total = 0i64;
    for (i, tx) in txs.iter().enumerate() {
        if !crate::transaction::is_coinbase(tx) {
            total = total.saturating_add(selected_tx_fee(tx, utxo_set, txs, &created));
        }
        created.insert(calculate_tx_id(tx), i);
    }
    total
}

fn selected_tx_fee(
    tx: &Transaction,
    utxo_set: &UtxoSet,
    txs: &[Transaction],
    created: &std::collections::HashMap<Hash, usize>,
) -> Integer {
    let mut total_input = 0i64;
    for input in &tx.inputs {
        let value = if let Some(utxo) = utxo_set.get(&input.prevout) {
            utxo.value
        } else if let Some(&parent_i) = created.get(&input.prevout.hash) {
            let Some(output) = txs[parent_i].outputs.get(input.prevout.index as usize) else {
                return 0;
            };
            output.value
        } else {
            return 0;
        };
        total_input = match total_input.checked_add(value) {
            Some(sum) => sum,
            None => return 0,
        };
    }
    let total_output = tx
        .outputs
        .iter()
        .try_fold(0i64, |acc, output| acc.checked_add(output.value))
        .unwrap_or(0);
    total_input.saturating_sub(total_output).max(0)
}

/// If payouts exceed subsidy+fees, error. Shortfall is added to the first output.
pub fn fit_payouts_to_reward(
    payouts: &[(Integer, ByteString)],
    subsidy: Integer,
    fees: Integer,
) -> Result<Vec<(Integer, ByteString)>> {
    if payouts.is_empty() {
        return Err(crate::error::ConsensusError::InvalidProofOfWork(
            "coinbase requires at least one output".into(),
        ));
    }
    let max = subsidy.saturating_add(fees);
    let sum: Integer = payouts
        .iter()
        .map(|(v, _)| *v)
        .fold(0, |a, b| a.saturating_add(b));
    if sum > max {
        return Err(crate::error::ConsensusError::EconomicValidation(
            format!("payouts {sum} exceed subsidy+fees {max}").into(),
        ));
    }
    let mut out = payouts.to_vec();
    if sum < max {
        out[0].0 = out[0].0.saturating_add(max - sum);
    }
    Ok(out)
}

/// BIP141: coinbase wtxid is 0, so appending the OP_RETURN does not change the witness root.
/// Empty stacks when `nested` is `None` (legacy). Prefer
/// [`append_witness_commitment_from_nested`] when mempool witnesses are known.
pub fn append_witness_commitment(
    coinbase: &mut Transaction,
    txs: &mut [Transaction],
) -> Result<()> {
    append_witness_commitment_from_nested(coinbase, txs, None)
}

/// Same as [`append_witness_commitment`], using per-tx per-input stacks.
/// `nested[i]` is the witness for `txs[i]`. Coinbase wtxid stays 0.
pub fn append_witness_commitment_from_nested(
    coinbase: &mut Transaction,
    txs: &mut [Transaction],
    nested: Option<&[Vec<crate::segwit::Witness>]>,
) -> Result<()> {
    if txs.is_empty() {
        return Ok(());
    }
    let owned: Vec<Vec<crate::segwit::Witness>>;
    let nested = if let Some(n) = nested {
        if n.len() < txs.len() {
            return Err(crate::error::ConsensusError::BlockValidation(
                "missing witness for selected transaction".into(),
            ));
        }
        n
    } else {
        owned = txs
            .iter()
            .map(|tx| vec![Vec::new(); tx.inputs.len()])
            .collect();
        &owned
    };
    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0u8; 32],
        merkle_root: [0u8; 32],
        timestamp: 0,
        bits: 0,
        nonce: 0,
    };
    let block = Block {
        header,
        transactions: txs.to_vec().into_boxed_slice(),
    };
    let root = crate::segwit::compute_witness_merkle_root_from_nested(&block, nested, None)?;
    let script = crate::segwit::witness_commitment_script(&root, &[0u8; 32]);
    let mut outs = coinbase.outputs.to_vec();
    outs.push(TransactionOutput {
        value: 0,
        script_pubkey: script,
    });
    coinbase.outputs = outs.into();
    txs[0] = coinbase.clone();
    Ok(())
}

/// Calculate merkle root using proper Bitcoin Merkle tree construction
#[track_caller] // Better error messages showing caller location
#[cfg_attr(feature = "production", inline(always))]
#[cfg_attr(not(feature = "production"), inline)]
#[cfg_attr(feature = "spec-lock", spec_locked("8.4.1", "ComputeMerkleRoot"))]
pub fn calculate_merkle_root(transactions: &[Transaction]) -> Result<Hash> {
    if transactions.is_empty() {
        return Err(crate::error::ConsensusError::InvalidProofOfWork(
            "Cannot calculate merkle root for empty transaction list".into(),
        ));
    }

    // Calculate transaction hashes with batch optimization (if available)
    // Uses SIMD vectorization + proven bounds for optimal performance
    // Use cache-aligned structures throughout merkle tree building
    #[cfg(feature = "production")]
    let mut hashes: Vec<crate::optimizations::CacheAlignedHash> = {
        use crate::optimizations::simd_vectorization;

        // Serialize all transactions in parallel (if rayon available)
        // Then batch hash all serialized forms using double SHA256
        // Pre-allocate serialization buffers
        let serialized_txs: Vec<Vec<u8>> = {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                transactions
                    .par_iter()
                    .map(serialize_tx_for_hash) // Uses prealloc_tx_buffer internally
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                transactions
                    .iter()
                    .map(serialize_tx_for_hash) // Uses prealloc_tx_buffer internally
                    .collect()
            }
        };

        // Batch hash all serialized transactions using double SHA256
        // Keep cache-aligned structures for better cache locality
        let tx_data_refs: Vec<&[u8]> = serialized_txs.iter().map(|v| v.as_slice()).collect();
        simd_vectorization::batch_double_sha256_aligned(&tx_data_refs)
    };

    #[cfg(not(feature = "production"))]
    let mut hashes: Vec<Hash> = {
        // Sequential fallback for non-production builds
        let mut hashes = Vec::with_capacity(transactions.len());
        for tx in transactions {
            hashes.push(calculate_tx_hash(tx));
        }
        hashes
    };

    // Build Merkle tree bottom-up
    // Pre-allocate next level and combined buffers
    // Optimization: Process multiple tree levels in parallel where safe
    // Use cache-aligned structures in production mode for better cache locality
    #[cfg(feature = "production")]
    {
        use crate::optimizations::CacheAlignedHash;

        let mut mutated = false;

        while hashes.len() > 1 {
            // CVE-2012-2459: Detect mutations (duplicate hashes at same level)
            let mut level_mutated = false;
            let mut pos = 0;
            while pos + 1 < hashes.len() {
                if hashes[pos].as_bytes() == hashes[pos + 1].as_bytes() {
                    level_mutated = true;
                }
                pos += 2;
            }
            if level_mutated {
                mutated = true;
            }

            // Duplicate last hash if odd number of hashes (Bitcoin's special rule)
            if hashes.len() & 1 != 0 {
                let last = hashes[hashes.len() - 1].clone();
                hashes.push(last);
            }

            // Stack-allocated 64-byte buffer, double SHA256 at each level
            let next_level: Vec<CacheAlignedHash> = hashes
                .chunks(2)
                .map(|chunk| {
                    let mut combined = [0u8; 64];
                    combined[..32].copy_from_slice(chunk[0].as_bytes());
                    combined[32..].copy_from_slice(if chunk.len() == 2 {
                        chunk[1].as_bytes()
                    } else {
                        chunk[0].as_bytes()
                    });
                    CacheAlignedHash::new(double_sha256_hash(&combined))
                })
                .collect();

            hashes = next_level;
        }

        if mutated {
            return Err(crate::error::ConsensusError::InvalidProofOfWork(
                "Merkle root mutation detected (CVE-2012-2459)".into(),
            ));
        }

        Ok(*hashes[0].as_bytes())
    }

    #[cfg(not(feature = "production"))]
    {
        let (root, mutated) = merkle_tree_from_hashes(&mut hashes)?;
        if mutated {
            return Err(crate::error::ConsensusError::InvalidProofOfWork(
                "Merkle root mutation detected (CVE-2012-2459)".into(),
            ));
        }
        Ok(root)
    }
}

/// Build merkle tree from pre-computed leaf hashes.
///
/// Also usable directly when tx_ids are already computed, avoiding
/// redundant serialization and hashing.
/// Implements ComputeMerkleRoot (Orange Paper 8.4.1).
///
/// Returns `Err` if `tx_ids` is empty **or** if a CVE-2012-2459 mutation is detected
/// (duplicate adjacent hashes at any merkle level). Use [`compute_merkle_root_and_mutated`]
/// when you need the root regardless of mutation (Orange Paper §8.4.1 merkle root).
#[cfg_attr(feature = "spec-lock", spec_locked("8.4.1", "ComputeMerkleRoot"))]
pub fn calculate_merkle_root_from_tx_ids(tx_ids: &[Hash]) -> Result<Hash> {
    let (root, mutated) = compute_merkle_root_and_mutated(tx_ids)?;
    if mutated {
        return Err(crate::error::ConsensusError::InvalidProofOfWork(
            "Merkle root mutation detected (CVE-2012-2459)".into(),
        ));
    }
    Ok(root)
}

/// Compute merkle root **and** CVE-2012-2459 mutation flag without aborting on mutation.
/// Computes merkle root from leaves with optional mutation detection — always returns a root.
/// Callers decide policy: `CheckBlock` rejects mutated blocks; `ConnectBlock` may skip
/// the check for already-accepted blocks.
#[cfg_attr(feature = "spec-lock", spec_locked("8.4.1", "ComputeMerkleRoot"))]
pub fn compute_merkle_root_and_mutated(tx_ids: &[Hash]) -> Result<(Hash, bool)> {
    if tx_ids.is_empty() {
        return Err(crate::error::ConsensusError::InvalidProofOfWork(
            "Cannot calculate merkle root for empty transaction list".into(),
        ));
    }
    let mut hashes = tx_ids.to_vec();
    merkle_tree_from_hashes(&mut hashes)
}

/// Merkle tree building logic. Uses double SHA256 at each level (Bitcoin standard).
/// Stack-allocates the 64-byte pair buffer to avoid heap allocation per node.
/// Orange Paper 8.4.1: ComputeMerkleRoot pair-and-hash construction.
/// Returns `(root, mutated)` — caller decides policy on mutation.
#[cfg_attr(feature = "spec-lock", spec_locked("8.4.1", "ComputeMerkleRoot"))]
fn merkle_tree_from_hashes(hashes: &mut Vec<Hash>) -> Result<(Hash, bool)> {
    let mut mutated = false;

    while hashes.len() > 1 {
        // CVE-2012-2459: Detect mutations (duplicate hashes at same level)
        for pos in (0..hashes.len().saturating_sub(1)).step_by(2) {
            if hashes[pos] == hashes[pos + 1] {
                mutated = true;
            }
        }

        // Duplicate last hash if odd number of hashes (Bitcoin's special rule)
        if hashes.len() & 1 != 0 {
            hashes.push(hashes[hashes.len() - 1]);
        }

        let mut next_level = Vec::with_capacity(hashes.len() / 2);

        // Stack-allocated 64-byte buffer for combining hash pairs
        for chunk in hashes.chunks(2) {
            let mut combined = [0u8; 64];
            combined[..32].copy_from_slice(&chunk[0]);
            combined[32..].copy_from_slice(if chunk.len() == 2 {
                &chunk[1]
            } else {
                &chunk[0]
            });
            next_level.push(double_sha256_hash(&combined));
        }

        *hashes = next_level;
    }

    Ok((hashes[0], mutated))
}

/// Serialize transaction for hashing (used for batch hashing optimization)
///
/// This is the same serialization as calculate_tx_hash but returns the serialized bytes
/// instead of hashing them, allowing batch hashing to be applied.
fn serialize_tx_for_hash(tx: &Transaction) -> Vec<u8> {
    // Pre-allocate buffer using proven maximum size
    #[cfg(feature = "production")]
    let mut data = {
        use crate::optimizations::prealloc_tx_buffer;
        prealloc_tx_buffer()
    };

    #[cfg(not(feature = "production"))]
    let mut data = Vec::new();

    // Version (4 bytes, little-endian)
    data.extend_from_slice(&(tx.version as u32).to_le_bytes());

    // Input count (varint)
    data.extend_from_slice(&encode_varint(tx.inputs.len() as u64));

    // Inputs
    for input in &tx.inputs {
        // Previous output hash (32 bytes)
        data.extend_from_slice(&input.prevout.hash);
        // Previous output index (4 bytes, little-endian)
        data.extend_from_slice(&input.prevout.index.to_le_bytes());
        // Script length (varint)
        data.extend_from_slice(&encode_varint(input.script_sig.len() as u64));
        // Script
        data.extend_from_slice(&input.script_sig);
        // Sequence (4 bytes, little-endian)
        data.extend_from_slice(&(input.sequence as u32).to_le_bytes());
    }

    // Output count (varint)
    data.extend_from_slice(&encode_varint(tx.outputs.len() as u64));

    // Outputs
    for output in &tx.outputs {
        // Value (8 bytes, little-endian)
        data.extend_from_slice(&(output.value as u64).to_le_bytes());
        // Script length (varint)
        data.extend_from_slice(&encode_varint(output.script_pubkey.len() as u64));
        // Script
        data.extend_from_slice(&output.script_pubkey);
    }

    // Lock time (4 bytes, little-endian)
    data.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());

    data
}

/// Calculate transaction hash using proper Bitcoin serialization
///
/// This function computes the double SHA256 hash of the serialized transaction.
/// For batch operations, use serialize_tx_for_hash + batch_double_sha256 instead.
#[allow(dead_code)] // Used in tests
fn calculate_tx_hash(tx: &Transaction) -> Hash {
    let data = serialize_tx_for_hash(tx);
    // Double SHA256 (Bitcoin standard)
    let hash1 = sha256_hash(&data);
    sha256_hash(&hash1)
}

/// Encode a number as a Bitcoin varint
fn encode_varint(value: u64) -> Vec<u8> {
    if value < 0xfd {
        vec![value as u8]
    } else if value <= 0xffff {
        let mut result = vec![0xfd];
        result.extend_from_slice(&(value as u16).to_le_bytes());
        result
    } else if value <= 0xffffffff {
        let mut result = vec![0xfe];
        result.extend_from_slice(&(value as u32).to_le_bytes());
        result
    } else {
        let mut result = vec![0xff];
        result.extend_from_slice(&value.to_le_bytes());
        result
    }
}

/// Orange Paper 7.2: Block hash = SHA256d(serialize(header)) — same as blockstore / PoW.
fn calculate_block_hash(header: &BlockHeader) -> Hash {
    let wire = crate::serialization::block::serialize_block_header(header);
    double_sha256_hash(&wire)
}

/// Simple SHA256 hash function
///
/// Performance optimization: Uses OptimizedSha256 (SHA-NI or AVX2) instead of sha2 crate
/// for faster hashing in Merkle tree construction.
#[inline(always)]
fn sha256_hash(data: &[u8]) -> Hash {
    use crate::crypto::OptimizedSha256;
    OptimizedSha256::new().hash(data)
}

/// Double SHA256 hash (Bitcoin standard for merkle tree nodes and txids)
#[inline(always)]
fn double_sha256_hash(data: &[u8]) -> Hash {
    use crate::crypto::OptimizedSha256;
    OptimizedSha256::new().hash256(data)
}

/// Wall-clock Unix timestamp for block template creation.
fn get_current_timestamp() -> Natural {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(std::time::Duration::ZERO)
        .as_secs()
}

#[cfg(test)]
#[path = "mining_tests.rs"]
mod tests;
