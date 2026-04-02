//! Apply block effects: apply_transaction, apply_transaction_with_id, calculate_tx_id.
//!
//! Clear "apply block effects" API; used by connect_block and external callers.

use crate::bip_validation::Bip30Index;
use crate::constants::MAX_MONEY;
use crate::error::Result;
use crate::reorganization::UndoEntry;
use crate::transaction::is_coinbase;
use crate::types::{Hash, Natural, OutPoint, Transaction, UtxoSet, UTXO};
use blvm_spec_lock::spec_locked;

/// ApplyTransaction (Orange Paper 5.3.1)
///
/// For transaction tx and UTXO set us:
/// 1. If tx is coinbase: us' = us ∪ {(tx.id, i) ↦ tx.outputs\[i\] : i ∈ \[0, |tx.outputs|)}
/// 2. Otherwise: us' = (us \ {i.prevout : i ∈ tx.inputs}) ∪ {(tx.id, i) ↦ tx.outputs\[i\] : i ∈ \[0, |tx.outputs|)}
/// 3. Return us'
///
/// This function computes the transaction ID internally.
/// For batch operations, use `apply_transaction_with_id` instead.
///
/// Returns both the new UTXO set and undo entries for all UTXO changes.
#[spec_locked("5.3.1")]
#[track_caller]
pub fn apply_transaction(
    tx: &Transaction,
    utxo_set: UtxoSet,
    height: Natural,
) -> Result<(UtxoSet, Vec<UndoEntry>)> {
    let tx_id = calculate_tx_id(tx);
    let mut no_index = None;
    apply_transaction_with_id(tx, tx_id, utxo_set, height, &mut no_index, true)
}

/// ApplyTransaction with pre-computed transaction ID
///
/// Same as `apply_transaction` but accepts a pre-computed transaction ID
/// to avoid redundant computation when transaction IDs are batch-computed.
///
/// Returns both the new UTXO set and undo entries for all UTXO changes.
/// When `bip30_index` is Some, updates it for coinbase add/remove (O(1) BIP30 checks).
#[spec_locked("5.3.1")]
pub(crate) fn apply_transaction_with_id(
    tx: &Transaction,
    tx_id: Hash,
    mut utxo_set: UtxoSet,
    height: Natural,
    bip30_index: &mut Option<&mut Bip30Index>,
    collect_undo: bool,
) -> Result<(UtxoSet, Vec<UndoEntry>)> {
    assert!(
        !tx.inputs.is_empty() || is_coinbase(tx),
        "Transaction must have inputs unless it's a coinbase"
    );
    assert!(
        !tx.outputs.is_empty(),
        "Transaction must have at least one output"
    );
    assert!(
        height <= i64::MAX as u64,
        "Block height {height} must fit in i64"
    );

    let mut undo_entries = if collect_undo {
        Vec::with_capacity(tx.inputs.len().saturating_add(tx.outputs.len()))
    } else {
        Vec::new()
    };
    let initial_utxo_count = utxo_set.len();

    #[cfg(feature = "production")]
    {
        let estimated_new_size = utxo_set
            .len()
            .saturating_add(tx.outputs.len())
            .saturating_sub(if is_coinbase(tx) { 0 } else { tx.inputs.len() });
        if estimated_new_size > utxo_set.capacity() {
            utxo_set.reserve(estimated_new_size.saturating_sub(utxo_set.len()));
        }
    }

    if !is_coinbase(tx) {
        assert!(
            !tx.inputs.is_empty(),
            "Non-coinbase transaction must have inputs"
        );

        for input in &tx.inputs {
            assert!(
                input.prevout.hash != [0u8; 32] || input.prevout.index != 0xffffffff,
                "Prevout must be valid for non-coinbase input"
            );

            if let Some(arc) = utxo_set.remove(&input.prevout) {
                let previous_utxo = arc.as_ref();
                if let Some(idx) = bip30_index.as_deref_mut() {
                    if previous_utxo.is_coinbase {
                        if let std::collections::hash_map::Entry::Occupied(mut o) =
                            idx.entry(input.prevout.hash)
                        {
                            *o.get_mut() = o.get().saturating_sub(1);
                            if *o.get() == 0 {
                                o.remove();
                            }
                        }
                    }
                }

                assert!(
                    previous_utxo.value >= 0,
                    "Previous UTXO value {} must be non-negative",
                    previous_utxo.value
                );
                assert!(
                    previous_utxo.value <= MAX_MONEY,
                    "Previous UTXO value {} must not exceed MAX_MONEY",
                    previous_utxo.value
                );

                if collect_undo {
                    undo_entries.push(UndoEntry {
                        outpoint: input.prevout,
                        previous_utxo: Some(std::sync::Arc::clone(&arc)),
                        new_utxo: None,
                    });
                    assert!(
                        undo_entries.len() <= tx.inputs.len() + tx.outputs.len(),
                        "Undo entry count {} must be reasonable",
                        undo_entries.len()
                    );
                }
            }
        }
    }

    for (i, output) in tx.outputs.iter().enumerate() {
        assert!(
            i < tx.outputs.len(),
            "Output index {} out of bounds (transaction has {} outputs)",
            i,
            tx.outputs.len()
        );
        assert!(
            output.value >= 0,
            "Output value {} must be non-negative",
            output.value
        );
        assert!(
            output.value <= MAX_MONEY,
            "Output value {} must not exceed MAX_MONEY",
            output.value
        );

        let outpoint = OutPoint {
            hash: tx_id,
            index: i as u32,
        };
        assert!(
            i <= u32::MAX as usize,
            "Output index {i} must fit in Natural"
        );

        let utxo = UTXO {
            value: output.value,
            script_pubkey: output.script_pubkey.as_slice().into(),
            height,
            is_coinbase: is_coinbase(tx),
        };
        assert!(
            utxo.value == output.value,
            "UTXO value {} must match output value {}",
            utxo.value,
            output.value
        );

        let utxo_arc = std::sync::Arc::new(utxo);
        if collect_undo {
            undo_entries.push(UndoEntry {
                outpoint,
                previous_utxo: None,
                new_utxo: Some(std::sync::Arc::clone(&utxo_arc)),
            });
            assert!(
                undo_entries.len() <= tx.outputs.len() + tx.inputs.len(),
                "Undo entry count {} must be reasonable",
                undo_entries.len()
            );
        }

        utxo_set.insert(outpoint, utxo_arc);

        if let Some(idx) = bip30_index.as_deref_mut() {
            if is_coinbase(tx) {
                *idx.entry(tx_id).or_insert(0) += 1;
            }
        }
    }

    if !is_coinbase(tx) {
        let current_count = utxo_set.len();
        let expected_count = initial_utxo_count
            .saturating_sub(tx.inputs.len())
            .saturating_add(tx.outputs.len());
        if current_count < expected_count {
            for (j, output) in tx.outputs.iter().enumerate() {
                let op = OutPoint {
                    hash: tx_id,
                    index: j as u32,
                };
                utxo_set.entry(op).or_insert_with(|| {
                    let utxo = UTXO {
                        value: output.value,
                        script_pubkey: output.script_pubkey.as_slice().into(),
                        height,
                        is_coinbase: false,
                    };
                    std::sync::Arc::new(utxo)
                });
            }
        }
    }

    let final_utxo_count = utxo_set.len();
    if is_coinbase(tx) {
        assert!(
            final_utxo_count >= initial_utxo_count,
            "UTXO set size {final_utxo_count} must not decrease after coinbase (was {initial_utxo_count})"
        );
        assert!(
            final_utxo_count <= initial_utxo_count + tx.outputs.len(),
            "UTXO set size {} must not exceed initial {} + outputs {}",
            final_utxo_count,
            initial_utxo_count,
            tx.outputs.len()
        );
    } else {
        let expected_change = tx.outputs.len() as i64 - tx.inputs.len() as i64;
        let actual_change = final_utxo_count as i64 - initial_utxo_count as i64;
        let lower = -(tx.inputs.len() as i64);
        debug_assert!(
            actual_change >= lower,
            "UTXO set size change {actual_change} must be reasonable (expected ~{expected_change})"
        );
    }
    assert!(
        utxo_set.len() <= u32::MAX as usize,
        "UTXO set size {} must not exceed maximum",
        utxo_set.len()
    );

    Ok((utxo_set, undo_entries))
}

/// Calculate transaction ID using proper Bitcoin double SHA256
///
/// Transaction ID is SHA256(SHA256(serialized_tx)) where serialized_tx
/// is the transaction in Bitcoin wire format.
///
/// For batch operations, use serialize_transaction + batch_double_sha256 instead.
#[inline(always)]
#[spec_locked("5.1")]
pub fn calculate_tx_id(tx: &Transaction) -> Hash {
    use crate::crypto::OptimizedSha256;
    use crate::serialization::transaction::serialize_transaction;

    let serialized = serialize_transaction(tx);
    OptimizedSha256::new().hash256(&serialized)
}
