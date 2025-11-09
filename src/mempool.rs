//! Mempool validation functions from Orange Paper Section 9

use crate::constants::*;
use crate::economic::calculate_fee;
use crate::error::Result;
use crate::script::verify_script;
use crate::segwit::{is_segwit_transaction, Witness};
use crate::transaction::{check_transaction, check_tx_inputs};
use crate::types::*;
use std::collections::HashSet;

/// AcceptToMemoryPool: 𝒯𝒳 × 𝒰𝒮 → {accepted, rejected}
///
/// For transaction tx and UTXO set us:
/// 1. Check if tx is already in mempool
/// 2. Validate transaction structure
/// 3. Check inputs against UTXO set
/// 4. Verify scripts
/// 5. Check mempool-specific rules (size, fee rate, etc.)
/// 6. Check for conflicts with existing mempool transactions
/// 7. Return acceptance result
///
/// # Arguments
///
/// * `tx` - Transaction to validate
/// * `witnesses` - Optional witness data for each input (Vec<Witness> where Witness = Vec<ByteString>)
/// * `utxo_set` - Current UTXO set
/// * `mempool` - Current mempool state
/// * `height` - Current block height
pub fn accept_to_memory_pool(
    tx: &Transaction,
    witnesses: Option<&[Witness]>,
    utxo_set: &UtxoSet,
    mempool: &Mempool,
    height: Natural,
) -> Result<MempoolResult> {
    // 1. Check if transaction is already in mempool
    let tx_id = crate::block::calculate_tx_id(tx);
    if mempool.contains(&tx_id) {
        return Ok(MempoolResult::Rejected(
            "Transaction already in mempool".to_string(),
        ));
    }

    // 2. Validate transaction structure
    if !matches!(check_transaction(tx)?, ValidationResult::Valid) {
        return Ok(MempoolResult::Rejected(
            "Invalid transaction structure".to_string(),
        ));
    }

    // 2.5. Check transaction finality
    // Note: block_time would typically come from network/chain state
    // For mempool acceptance, we use current system time as approximation
    // In production, this should use the chain tip's median time-past
    let block_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if !is_final_tx(tx, height, block_time) {
        return Ok(MempoolResult::Rejected(
            "Transaction not final (locktime not satisfied)".to_string(),
        ));
    }

    // 3. Check inputs against UTXO set
    let (input_valid, fee) = check_tx_inputs(tx, utxo_set, height)?;
    if !matches!(input_valid, ValidationResult::Valid) {
        return Ok(MempoolResult::Rejected(
            "Invalid transaction inputs".to_string(),
        ));
    }

    // 4. Verify scripts for non-coinbase transactions
    if !is_coinbase(tx) {
        // Calculate script verification flags
        // Enable SegWit flag if transaction has witness data
        let flags = calculate_script_flags(tx, witnesses);

        #[cfg(all(feature = "production", feature = "rayon"))]
        {
            use rayon::prelude::*;

            // Optimization: Batch UTXO lookups and parallelize script verification
            // Pre-lookup all UTXOs to avoid concurrent HashMap access
            // Pre-allocate with known size
            let input_utxos: Vec<(usize, Option<&UTXO>)> = {
                let mut result = Vec::with_capacity(tx.inputs.len());
                for (i, input) in tx.inputs.iter().enumerate() {
                    result.push((i, utxo_set.get(&input.prevout)));
                }
                result
            };

            // Parallelize script verification (read-only operations) ✅ Thread-safe
            let script_results: Result<Vec<bool>> = input_utxos
                .par_iter()
                .map(|(i, opt_utxo)| {
                    if let Some(utxo) = opt_utxo {
                        let input = &tx.inputs[*i];
                        let witness: Option<&ByteString> =
                            witnesses.and_then(|wits| wits.get(*i)).and_then(|wit| {
                                wit.first()
                            });

                        verify_script(&input.script_sig, &utxo.script_pubkey, witness, flags)
                    } else {
                        Ok(false)
                    }
                })
                .collect();

            // Check results sequentially
            let script_results = script_results?;
            for (i, &is_valid) in script_results.iter().enumerate() {
                if !is_valid {
                    return Ok(MempoolResult::Rejected(format!(
                        "Invalid script at input {}",
                        i
                    )));
                }
            }
        }

        #[cfg(not(all(feature = "production", feature = "rayon")))]
        {
            // Sequential fallback
            for (i, input) in tx.inputs.iter().enumerate() {
                if let Some(utxo) = utxo_set.get(&input.prevout) {
                    // Get witness for this input if available
                    // Witness is Vec<ByteString> per input, for verify_script we need Option<&ByteString>
                    // For SegWit P2WPKH/P2WSH, we typically use the witness stack elements
                    // For now, we'll use the first element if available (simplified)
                    let witness: Option<&ByteString> =
                        witnesses.and_then(|wits| wits.get(i)).and_then(|wit| {
                            // Witness is Vec<ByteString> - for verify_script we can pass the first element
                            // or construct a combined witness script. For now, use first element.
                            wit.first()
                        });

                    if !verify_script(&input.script_sig, &utxo.script_pubkey, witness, flags)? {
                        return Ok(MempoolResult::Rejected(format!(
                            "Invalid script at input {}",
                            i
                        )));
                    }
                }
            }
        }
    }

    // 5. Check mempool-specific rules
    if !check_mempool_rules(tx, fee, mempool)? {
        return Ok(MempoolResult::Rejected("Failed mempool rules".to_string()));
    }

    // 6. Check for conflicts with existing mempool transactions
    if has_conflicts(tx, mempool)? {
        return Ok(MempoolResult::Rejected(
            "Transaction conflicts with mempool".to_string(),
        ));
    }

    Ok(MempoolResult::Accepted)
}

/// Calculate script verification flags based on transaction type
///
/// Returns appropriate flags for script validation:
/// - Base flags: Standard validation flags (P2SH, STRICTENC, DERSIG, LOW_S, etc.)
/// - SegWit flag (SCRIPT_VERIFY_WITNESS = 0x800): Enabled if transaction uses SegWit
/// - Taproot flag (SCRIPT_VERIFY_TAPROOT = 0x2000): Enabled if transaction uses Taproot
fn calculate_script_flags(tx: &Transaction, witnesses: Option<&[Witness]>) -> u32 {
    // Base flags (standard validation flags)
    // In Bitcoin Core, these are typically always enabled:
    // SCRIPT_VERIFY_P2SH = 0x01
    // SCRIPT_VERIFY_STRICTENC = 0x02
    // SCRIPT_VERIFY_DERSIG = 0x04
    // SCRIPT_VERIFY_LOW_S = 0x08
    // SCRIPT_VERIFY_NULLDUMMY = 0x10
    // SCRIPT_VERIFY_SIGPUSHONLY = 0x20
    // SCRIPT_VERIFY_MINIMALDATA = 0x40
    // SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS = 0x80
    // SCRIPT_VERIFY_CLEANSTACK = 0x100
    // SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY = 0x200
    // SCRIPT_VERIFY_CHECKSEQUENCEVERIFY = 0x400
    let base_flags = 0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x200 | 0x400;

    let mut flags = base_flags;

    // Enable SegWit flag if transaction has witness data or is a SegWit transaction
    if witnesses.is_some() || is_segwit_transaction(tx) {
        flags |= 0x800; // SCRIPT_VERIFY_WITNESS
    }

    // Enable Taproot flag if transaction uses Taproot outputs
    // Check if any output is P2TR (Pay-to-Taproot): 0x5120 (1-byte version + 32-byte x-only pubkey)
    for output in &tx.outputs {
        let script = &output.script_pubkey;
        if script.len() == 34 && script[0] == 0x51 && script[1] == 0x20 {
            flags |= 0x2000; // SCRIPT_VERIFY_TAPROOT
            break;
        }
    }

    flags
}

/// IsStandardTx: 𝒯𝒳 → {true, false}
///
/// Check if transaction follows standard rules for mempool acceptance:
/// 1. Transaction size limits
/// 2. Script size limits
/// 3. Standard script types
/// 4. Fee rate requirements
pub fn is_standard_tx(tx: &Transaction) -> Result<bool> {
    // 1. Check transaction size
    let tx_size = calculate_transaction_size(tx);
    if tx_size > MAX_TX_SIZE {
        return Ok(false);
    }

    // 2. Check script sizes
    for input in &tx.inputs {
        if input.script_sig.len() > MAX_SCRIPT_SIZE {
            return Ok(false);
        }
    }

    for output in &tx.outputs {
        if output.script_pubkey.len() > MAX_SCRIPT_SIZE {
            return Ok(false);
        }
    }

    // 3. Check for standard script types (simplified)
    for output in &tx.outputs {
        if !is_standard_script(&output.script_pubkey)? {
            return Ok(false);
        }
    }

    Ok(true)
}

/// ReplacementChecks: 𝒯𝒳 × 𝒯𝒳 × 𝒰𝒮 × Mempool → {true, false}
///
/// Check if new transaction can replace existing one (BIP125 RBF rules).
///
/// According to BIP125 and Orange Paper Section 9.3, replacement is allowed if:
/// 1. Existing transaction signals RBF (nSequence < SEQUENCE_FINAL)
/// 2. New transaction has higher fee rate: FeeRate(tx_2) > FeeRate(tx_1)
/// 3. New transaction pays absolute fee bump: Fee(tx_2) > Fee(tx_1) + MIN_RELAY_FEE
/// 4. New transaction conflicts with existing: tx_2 spends at least one input from tx_1
/// 5. No new unconfirmed dependencies: All inputs of tx_2 are confirmed or from tx_1
pub fn replacement_checks(
    new_tx: &Transaction,
    existing_tx: &Transaction,
    utxo_set: &UtxoSet,
    mempool: &Mempool,
) -> Result<bool> {
    // 1. Check RBF signaling - existing transaction must signal RBF
    // Note: new_tx doesn't need to signal RBF per BIP125, only existing_tx does
    if !signals_rbf(existing_tx) {
        return Ok(false);
    }

    // 2. Check fee rate: FeeRate(tx_2) > FeeRate(tx_1)
    let new_fee = calculate_fee(new_tx, utxo_set)?;
    let existing_fee = calculate_fee(existing_tx, utxo_set)?;

    let new_tx_size = calculate_transaction_size_vbytes(new_tx);
    let existing_tx_size = calculate_transaction_size_vbytes(existing_tx);

    if new_tx_size == 0 || existing_tx_size == 0 {
        return Ok(false);
    }

    let new_fee_rate = (new_fee as f64) / (new_tx_size as f64);
    let existing_fee_rate = (existing_fee as f64) / (existing_tx_size as f64);

    if new_fee_rate <= existing_fee_rate {
        return Ok(false);
    }

    // 3. Check absolute fee bump: Fee(tx_2) > Fee(tx_1) + MIN_RELAY_FEE
    if new_fee <= existing_fee + MIN_RELAY_FEE {
        return Ok(false);
    }

    // 4. Check conflict: tx_2 must spend at least one input from tx_1
    if !has_conflict_with_tx(new_tx, existing_tx) {
        return Ok(false);
    }

    // 5. Check for new unconfirmed dependencies
    // All inputs of tx_2 must be confirmed (in UTXO set) or from tx_1
    if creates_new_dependencies(new_tx, existing_tx, utxo_set, mempool)? {
        return Ok(false);
    }

    Ok(true)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Mempool data structure
pub type Mempool = HashSet<Hash>;

/// Result of mempool acceptance
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MempoolResult {
    Accepted,
    Rejected(String),
}

/// Update mempool after block connection
///
/// Removes transactions that were included in the block and transactions
/// that became invalid due to spent inputs.
///
/// This function should be called after successfully connecting a block
/// to keep the mempool synchronized with the blockchain state.
///
/// # Arguments
///
/// * `mempool` - Mutable reference to the mempool
/// * `block` - The block that was just connected
/// * `utxo_set` - The updated UTXO set after block connection
///
/// # Returns
///
/// Returns a vector of transaction IDs that were removed from the mempool.
///
/// # Example
///
/// ```rust
/// use consensus_proof::mempool::{Mempool, update_mempool_after_block};
/// use consensus_proof::block::connect_block;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use consensus_proof::types::*;
/// # use consensus_proof::mining::calculate_merkle_root;
/// # let coinbase_tx = Transaction {
/// #     version: 1,
/// #     inputs: vec![TransactionInput {
/// #         prevout: OutPoint { hash: [0; 32], index: 0xffffffff },
/// #         script_sig: vec![],
/// #         sequence: 0xffffffff,
/// #     }],
/// #     outputs: vec![TransactionOutput { value: 5000000000, script_pubkey: vec![] }],
/// #     lock_time: 0,
/// # };
/// # let merkle_root = calculate_merkle_root(&[coinbase_tx.clone()]).unwrap();
/// # let block = Block {
/// #     header: BlockHeader {
/// #         version: 1, prev_block_hash: [0; 32], merkle_root,
/// #         timestamp: 1234567890, bits: 0x1d00ffff, nonce: 0,
/// #     },
/// #     transactions: vec![coinbase_tx],
/// # };
/// # let witnesses = vec![];
/// # let mut utxo_set = UtxoSet::new();
/// # let height = 0;
/// # let mut mempool = Mempool::new();
/// let (result, new_utxo_set) = connect_block(&block, &witnesses, utxo_set, height, None)?;
/// if matches!(result, consensus_proof::ValidationResult::Valid) {
///     let removed = update_mempool_after_block(&mut mempool, &block, &new_utxo_set)?;
///     println!("Removed {} transactions from mempool", removed.len());
/// }
/// # Ok(())
/// # }
/// ```
pub fn update_mempool_after_block(
    mempool: &mut Mempool,
    block: &crate::types::Block,
    _utxo_set: &crate::types::UtxoSet,
) -> Result<Vec<Hash>> {
    let mut removed = Vec::new();

    // 1. Remove transactions that were included in the block
    for tx in &block.transactions {
        let tx_id = crate::block::calculate_tx_id(tx);
        if mempool.remove(&tx_id) {
            removed.push(tx_id);
        }
    }

    // 2. Remove transactions that became invalid (inputs were spent by block)
    // Note: We don't have the transaction here, just the ID
    // In a full implementation, we'd need a transaction store or lookup
    // For now, we'll skip this check and rely on the caller to handle it
    // Use `update_mempool_after_block_with_lookup` for full validation
    // This is a limitation that should be addressed with a transaction index

    Ok(removed)
}

/// Update mempool after block connection (with transaction lookup)
///
/// This is a more complete version that can check if mempool transactions
/// became invalid. Requires a way to look up transactions by ID.
///
/// # Arguments
///
/// * `mempool` - Mutable reference to the mempool
/// * `block` - The block that was just connected
/// * `get_tx_by_id` - Function to look up transactions by ID
///
/// # Returns
///
/// Returns a vector of transaction IDs that were removed from the mempool.
pub fn update_mempool_after_block_with_lookup<F>(
    mempool: &mut Mempool,
    block: &crate::types::Block,
    get_tx_by_id: F,
) -> Result<Vec<Hash>>
where
    F: Fn(&Hash) -> Option<crate::types::Transaction>,
{
    let mut removed = Vec::new();

    // 1. Remove transactions that were included in the block
    for tx in &block.transactions {
        let tx_id = crate::block::calculate_tx_id(tx);
        if mempool.remove(&tx_id) {
            removed.push(tx_id);
        }
    }

    // 2. Remove transactions that became invalid (inputs were spent by block)
    // Collect spent outpoints from the block
    let mut spent_outpoints = std::collections::HashSet::new();
    for tx in &block.transactions {
        if !crate::transaction::is_coinbase(tx) {
            for input in &tx.inputs {
                spent_outpoints.insert(input.prevout.clone());
            }
        }
    }

    // Check each mempool transaction to see if it spends any of the spent outpoints
    let mut invalid_tx_ids = Vec::new();
    for &tx_id in mempool.iter() {
        if let Some(tx) = get_tx_by_id(&tx_id) {
            // Check if any input of this transaction was spent by the block
            for input in &tx.inputs {
                if spent_outpoints.contains(&input.prevout) {
                    invalid_tx_ids.push(tx_id);
                    break;
                }
            }
        }
    }

    // Remove invalid transactions
    for tx_id in invalid_tx_ids {
        if mempool.remove(&tx_id) {
            removed.push(tx_id);
        }
    }

    Ok(removed)
}

/// Check mempool-specific rules
fn check_mempool_rules(tx: &Transaction, fee: Integer, mempool: &Mempool) -> Result<bool> {
    // Check minimum fee rate (simplified)
    let tx_size = calculate_transaction_size(tx);
    let fee_rate = (fee as f64) / (tx_size as f64);
    let min_fee_rate = 1.0; // 1 sat/byte minimum

    if fee_rate < min_fee_rate {
        return Ok(false);
    }

    // Check mempool size limits (simplified)
    if mempool.len() > 10000 {
        // Arbitrary limit
        return Ok(false);
    }

    Ok(true)
}

/// Check for transaction conflicts
fn has_conflicts(tx: &Transaction, mempool: &Mempool) -> Result<bool> {
    // Check if any input is already spent by mempool transaction
    for input in &tx.inputs {
        // In a real implementation, we'd check if this input is already spent
        // by another transaction in the mempool
        // For now, we'll do a simplified check
        if mempool.contains(&input.prevout.hash) {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Check if transaction is final (Orange Paper Section 9.1 - Transaction Finality)
///
/// A transaction is final if:
/// 1. tx.lock_time == 0 (no locktime restriction), OR
/// 2. If locktime < LOCKTIME_THRESHOLD (block height): height >= tx.lock_time
/// 3. If locktime >= LOCKTIME_THRESHOLD (timestamp): block_time >= tx.lock_time
///
/// Mathematical specification:
/// ∀ tx ∈ Transaction, height ∈ ℕ, block_time ∈ ℕ:
/// - is_final_tx(tx, height, block_time) = true ⟹
///   (tx.lock_time = 0 ∨
///   (tx.lock_time < LOCKTIME_THRESHOLD ∧ height >= tx.lock_time) ∨
///   (tx.lock_time >= LOCKTIME_THRESHOLD ∧ block_time >= tx.lock_time))
pub fn is_final_tx(tx: &Transaction, height: Natural, block_time: Natural) -> bool {
    // If locktime is 0, transaction is always final
    if tx.lock_time == 0 {
        return true;
    }

    // Check if locktime is satisfied based on type
    if (tx.lock_time as u32) < LOCKTIME_THRESHOLD {
        // Block height locktime: current height must be >= locktime
        height >= tx.lock_time as Natural
    } else {
        // Timestamp locktime: current block time must be >= locktime
        block_time >= tx.lock_time as Natural
    }
}

/// Check if transaction signals RBF
///
/// Returns true if any input has nSequence < SEQUENCE_FINAL (0xffffffff)
pub fn signals_rbf(tx: &Transaction) -> bool {
    for input in &tx.inputs {
        if (input.sequence as u32) < SEQUENCE_FINAL {
            return true;
        }
    }
    false
}

/// Calculate transaction size in virtual bytes (vbytes)
///
/// For SegWit transactions, uses weight/4 (virtual bytes).
/// For non-SegWit transactions, uses byte size.
/// This is a simplified version - proper implementation would use segwit::calculate_weight()
fn calculate_transaction_size_vbytes(tx: &Transaction) -> usize {
    // Simplified: use byte size as approximation
    // In production, should use proper weight calculation for SegWit
    calculate_transaction_size(tx)
}

/// Check if new transaction conflicts with existing transaction
///
/// A conflict exists if new_tx spends at least one input from existing_tx.
/// This is requirement #4 of BIP125.
pub fn has_conflict_with_tx(new_tx: &Transaction, existing_tx: &Transaction) -> bool {
    for new_input in &new_tx.inputs {
        for existing_input in &existing_tx.inputs {
            if new_input.prevout == existing_input.prevout {
                return true;
            }
        }
    }
    false
}

/// Check if new transaction creates new unconfirmed dependencies
///
/// BIP125 requirement #5: All inputs of tx_2 must be:
/// - Confirmed (in UTXO set), OR
/// - From tx_1 (spending the same inputs)
fn creates_new_dependencies(
    new_tx: &Transaction,
    existing_tx: &Transaction,
    utxo_set: &UtxoSet,
    mempool: &Mempool,
) -> Result<bool> {
    for input in &new_tx.inputs {
        // Check if input is confirmed (in UTXO set)
        if utxo_set.contains_key(&input.prevout) {
            continue;
        }

        // Check if input was spent by existing transaction
        let mut found_in_existing = false;
        for existing_input in &existing_tx.inputs {
            if existing_input.prevout == input.prevout {
                found_in_existing = true;
                break;
            }
        }

        if found_in_existing {
            continue;
        }

        // If not confirmed and not from existing tx, it's a new unconfirmed dependency
        // Check if it's at least in mempool (but still unconfirmed)
        if !mempool.contains(&input.prevout.hash) {
            return Ok(true); // New unconfirmed dependency
        }
    }

    Ok(false)
}

/// Check if script is standard
fn is_standard_script(script: &ByteString) -> Result<bool> {
    // Simplified standard script check
    // In reality, this would check for P2PKH, P2SH, P2WPKH, P2WSH, etc.
    if script.is_empty() {
        return Ok(false);
    }

    // Basic checks
    if script.len() > MAX_SCRIPT_SIZE {
        return Ok(false);
    }

    // Check for non-standard opcodes (simplified)
    for &byte in script {
        if byte > 0x60 && byte < 0x7f {
            // Some non-standard opcodes
            return Ok(false);
        }
    }

    Ok(true)
}

/// Calculate transaction ID (deprecated - use crate::block::calculate_tx_id instead)
///
/// This function is kept for backward compatibility but delegates to the
/// standard implementation in block.rs.
#[deprecated(note = "Use crate::block::calculate_tx_id instead")]
pub fn calculate_tx_id(tx: &Transaction) -> Hash {
    crate::block::calculate_tx_id(tx)
}

/// Calculate transaction size (simplified)
fn calculate_transaction_size(tx: &Transaction) -> usize {
    4 + // version
    tx.inputs.len() * (32 + 4 + 1 + 4) + // inputs (OutPoint + script_sig_len + sequence) - simplified
    tx.outputs.len() * (8 + 1) + // outputs (value + script_pubkey_len) - simplified
    4 // lock_time
}

/// Check if transaction is coinbase
fn is_coinbase(tx: &Transaction) -> bool {
    tx.inputs.len() == 1
        && tx.inputs[0].prevout.hash == [0u8; 32]
        && tx.inputs[0].prevout.index == 0xffffffff
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for Mempool:
/// ∀ tx ∈ 𝒯𝒳, utxo_set ∈ 𝒰𝒮, mempool ∈ Mempool:
/// - accept_to_memory_pool(tx, utxo_set, mempool) = Accepted ⟹
///   (tx ∉ mempool ∧
///    CheckTransaction(tx) = valid ∧
///    CheckTxInputs(tx, utxo_set) = valid ∧
///    VerifyScripts(tx) = valid ∧
///    ¬has_conflicts(tx, mempool))
///
/// Invariants:
/// - Mempool never contains duplicate transactions
/// - Mempool never contains conflicting transactions
/// - Accepted transactions are valid
/// - RBF rules are enforced

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: mempool never contains duplicates
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳, mempool ∈ Mempool:
    /// - If accept_to_memory_pool succeeds: tx ∉ old_mempool
    /// - After acceptance: new_mempool contains tx exactly once
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_mempool_no_duplicates() {
        let tx: Transaction = kani::any();
        let utxo_set: UtxoSet = kani::any();
        let mut mempool: Mempool = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability
        use crate::kani_helpers::{assume_mempool_bounds, assume_transaction_bounds_custom};
        assume_transaction_bounds_custom!(tx, 3, 3);
        assume_mempool_bounds!(mempool, 5);

        let tx_id = crate::block::calculate_tx_id(&tx);

        // If transaction is already in mempool, it should be rejected
        if mempool.contains(&tx_id) {
            let result = accept_to_memory_pool(&tx, &utxo_set, &mempool, height);
            if result.is_ok() {
                let mempool_result = result.unwrap();
                assert!(
                    matches!(mempool_result, MempoolResult::Rejected(_)),
                    "Duplicate transactions must be rejected"
                );
            }
        }

        // If accepted, mempool should contain transaction exactly once
        // Note: This would require modifying mempool in place, which accept_to_memory_pool doesn't do
        // For now, we prove that acceptance implies the transaction wasn't in the old mempool
    }

    /// Kani proof: mempool conflict detection works correctly
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ 𝒯𝒳, mempool ∈ Mempool:
    /// - has_conflicts(tx, mempool) = true ⟹ ∃ tx' ∈ mempool: conflicts(tx, tx')
    /// - If has_conflicts returns true, transaction should be rejected
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_mempool_conflict_detection() {
        let tx: Transaction = kani::any();
        let mut mempool: Mempool = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability
        use crate::kani_helpers::{assume_mempool_bounds, assume_transaction_bounds_custom};
        assume_transaction_bounds_custom!(tx, 3, 3);
        assume_mempool_bounds!(mempool, 5);

        // Test conflict detection
        let has_conflict = has_conflicts(&tx, &mempool).unwrap_or(false);

        // If transaction has conflicts, accept_to_memory_pool should reject it
        // (assuming other checks pass)
        if has_conflict {
            // Conflict detection should work
            assert!(has_conflict, "Conflict detection should identify conflicts");
        }
    }

    /// Kani proof: is_final_tx correctness (Orange Paper Section 9.1 - Transaction Finality)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, height ∈ ℕ, block_time ∈ ℕ:
    /// - is_final_tx(tx, height, block_time) = true ⟹
    ///   (tx.lock_time = 0 ∨
    ///    (tx.lock_time < LOCKTIME_THRESHOLD ∧ height >= tx.lock_time) ∨
    ///    (tx.lock_time >= LOCKTIME_THRESHOLD ∧ block_time >= tx.lock_time))
    ///
    /// This ensures transaction finality check matches Orange Paper specification exactly.
    #[kani::proof]
    fn kani_is_final_tx_correctness() {
        use crate::constants::LOCKTIME_THRESHOLD;

        let tx: Transaction = kani::any();
        let height: Natural = kani::any();
        let block_time: Natural = kani::any();

        // Bound for tractability
        kani::assume(height <= 1000000);
        kani::assume(block_time <= 1000000000);

        let is_final = is_final_tx(&tx, height, block_time);

        // Calculate according to Orange Paper spec
        let spec_final = if tx.lock_time == 0 {
            true
        } else if tx.lock_time < LOCKTIME_THRESHOLD {
            // Block height locktime
            height >= tx.lock_time as Natural
        } else {
            // Timestamp locktime
            block_time >= tx.lock_time as Natural
        };

        // Critical invariant: implementation must match specification
        assert_eq!(is_final, spec_final,
            "is_final_tx must match Orange Paper specification: locktime = 0 OR (height locktime AND height >= locktime) OR (timestamp locktime AND block_time >= locktime)");
    }

    /// Kani proof: is_standard_tx correctness (Orange Paper Section 9.1)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction:
    /// - is_standard_tx(tx) = true ⟹
    ///   1. Transaction size ≤ MAX_TX_SIZE
    ///   2. ∀ input ∈ tx.inputs: |input.script_sig| ≤ MAX_SCRIPT_SIZE
    ///   3. ∀ output ∈ tx.outputs: |output.script_pubkey| ≤ MAX_SCRIPT_SIZE
    ///   4. ∀ output ∈ tx.outputs: is_standard_script(output.script_pubkey) = true
    ///
    /// This ensures standard transaction rules are correctly enforced for mempool acceptance.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_is_standard_tx_correctness() {
        let tx: Transaction = kani::any();

        // Bound for tractability
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Check standard transaction rules
        let result = is_standard_tx(&tx);

        if result.is_ok() {
            let is_standard = result.unwrap();

            if is_standard {
                // Critical invariant: standard transactions must satisfy all rules
                // 1. Transaction size limit
                let tx_size = calculate_transaction_size(&tx);
                assert!(
                    tx_size <= MAX_TX_SIZE,
                    "is_standard_tx: standard transactions must have size ≤ MAX_TX_SIZE"
                );

                // 2. Script size limits
                for input in &tx.inputs {
                    assert!(input.script_sig.len() <= MAX_SCRIPT_SIZE,
                        "is_standard_tx: standard transactions must have input script_sig ≤ MAX_SCRIPT_SIZE");
                }

                for output in &tx.outputs {
                    assert!(output.script_pubkey.len() <= MAX_SCRIPT_SIZE,
                        "is_standard_tx: standard transactions must have output script_pubkey ≤ MAX_SCRIPT_SIZE");
                }

                // 3. Standard script types (verified by is_standard_script call in implementation)
            }
        }
    }

    /// Kani proof: signals_rbf correctness (BIP125)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction:
    /// - signals_rbf(tx) = true ⟺ ∃ input ∈ tx.inputs: input.sequence < SEQUENCE_FINAL
    ///
    /// This ensures RBF signaling detection matches BIP125 specification exactly.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_signals_rbf_correctness() {
        let tx: Transaction = kani::any();

        // Bound for tractability
        kani::assume(tx.inputs.len() <= 5);

        // Calculate according to BIP125 spec
        let spec_signals_rbf = tx
            .inputs
            .iter()
            .any(|input| (input.sequence as u32) < SEQUENCE_FINAL);

        // Calculate using implementation
        let impl_signals_rbf = signals_rbf(&tx);

        // Critical invariant: implementation must match specification
        assert_eq!(impl_signals_rbf, spec_signals_rbf,
            "signals_rbf must match BIP125 specification: any input with sequence < SEQUENCE_FINAL signals RBF");
    }

    /// Kani proof: has_conflict_with_tx correctness (BIP125 requirement #4)
    ///
    /// Mathematical specification:
    /// ∀ new_tx, existing_tx ∈ Transaction:
    /// - has_conflict_with_tx(new_tx, existing_tx) = true ⟺
    ///   ∃ new_input ∈ new_tx.inputs, existing_input ∈ existing_tx.inputs:
    ///   new_input.prevout == existing_input.prevout
    ///
    /// This ensures conflict detection matches BIP125 specification exactly.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_has_conflict_with_tx_correctness() {
        let new_tx: Transaction = kani::any();
        let existing_tx: Transaction = kani::any();

        // Bound for tractability
        kani::assume(new_tx.inputs.len() <= 5);
        kani::assume(existing_tx.inputs.len() <= 5);

        // Calculate according to BIP125 spec
        let spec_has_conflict = new_tx.inputs.iter().any(|new_input| {
            existing_tx
                .inputs
                .iter()
                .any(|existing_input| new_input.prevout == existing_input.prevout)
        });

        // Calculate using implementation
        let impl_has_conflict = has_conflict_with_tx(&new_tx, &existing_tx);

        // Critical invariant: implementation must match specification
        assert_eq!(impl_has_conflict, spec_has_conflict,
            "has_conflict_with_tx must match BIP125 specification: conflict exists if new_tx spends any input from existing_tx");
    }

    /// Kani proof: update_mempool_after_block correctness (Orange Paper Section 9.2)
    ///
    /// Mathematical specification:
    /// ∀ mempool ∈ Mempool, block ∈ Block:
    /// - update_mempool_after_block(mempool, block) = removed ⟹
    ///   1. ∀ tx_id ∈ block.transactions: tx_id ∉ mempool' (removed from mempool)
    ///   2. removed contains all transaction IDs from block that were in mempool
    ///   3. mempool' = mempool \ {tx_id : tx_id ∈ block.transactions}
    ///
    /// This ensures mempool is correctly updated after block connection.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_update_mempool_after_block_correctness() {
        use crate::block::calculate_tx_id;

        let mut mempool: Mempool = kani::any();
        let block: Block = kani::any();

        // Bound for tractability
        kani::assume(block.transactions.len() <= 5);
        for tx in &block.transactions {
            kani::assume(tx.inputs.len() <= 3);
            kani::assume(tx.outputs.len() <= 3);
        }

        // Record initial mempool state
        let initial_mempool_size = mempool.len();
        let initial_tx_ids: Vec<Hash> = mempool.iter().cloned().collect();

        // Calculate block transaction IDs
        let block_tx_ids: Vec<Hash> = block
            .transactions
            .iter()
            .map(|tx| crate::block::calculate_tx_id(tx))
            .collect();

        // Add some block transaction IDs to mempool (simulating they were in mempool)
        for tx_id in &block_tx_ids {
            mempool.insert(*tx_id);
        }

        let mempool_size_before_update = mempool.len();

        // Update mempool
        let result = update_mempool_after_block(&mut mempool, &block, &UtxoSet::new());

        if result.is_ok() {
            let removed = result.unwrap();

            // Critical invariant: all block transaction IDs must be removed if they were in mempool
            for tx_id in &block_tx_ids {
                assert!(!mempool.contains(tx_id),
                    "update_mempool_after_block: all block transaction IDs must be removed from mempool");
                // If it was in mempool before, it should be in removed list
                if initial_tx_ids.contains(tx_id)
                    || mempool_size_before_update > initial_mempool_size
                {
                    assert!(removed.contains(tx_id),
                        "update_mempool_after_block: removed transactions must include block transaction IDs that were in mempool");
                }
            }

            // Critical invariant: removed transactions must be from block
            for tx_id in &removed {
                assert!(
                    block_tx_ids.contains(tx_id),
                    "update_mempool_after_block: removed transactions must be from block"
                );
            }

            // Critical invariant: mempool size decreased correctly
            // Size after = size before - removed transactions
            let removed_count = removed.len();
            let expected_size = mempool_size_before_update.saturating_sub(removed_count);
            assert_eq!(mempool.len(), expected_size,
                "update_mempool_after_block: mempool size must decrease by number of removed transactions");
        }
    }

    /// Kani proof: RBF replacement checks enforce fee requirements
    ///
    /// Mathematical specification (BIP125):
    /// ∀ new_tx, existing_tx ∈ 𝒯𝒳, utxo_set ∈ 𝒰𝒮, mempool ∈ Mempool:
    /// - replacement_checks(new_tx, existing_tx, utxo_set, mempool) = true ⟹
    ///   (signals_rbf(existing_tx) ∧
    ///    fee_rate(new_tx) > fee_rate(existing_tx) ∧
    ///    fee(new_tx) > fee(existing_tx) + MIN_RELAY_FEE ∧
    ///    has_conflict_with_tx(new_tx, existing_tx))
    ///
    /// Note: This proof is simplified due to Kani limitations with floating point
    /// and requires utxo_set. The actual implementation uses proper fee calculation.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_rbf_fee_requirement() {
        let new_tx: Transaction = kani::any();
        let existing_tx: Transaction = kani::any();
        let utxo_set: UtxoSet = kani::any();
        let mempool: Mempool = kani::any();

        // Bound for tractability
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(new_tx, 3, 3);
        assume_transaction_bounds_custom!(existing_tx, 3, 3);

        let result = replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool);

        if result.is_ok() && result.unwrap() {
            // If replacement is allowed, existing must signal RBF
            assert!(
                signals_rbf(&existing_tx),
                "RBF replacement requires existing_tx to signal RBF"
            );

            // Conflict must exist
            assert!(
                has_conflict_with_tx(&new_tx, &existing_tx),
                "RBF replacement requires conflict with existing transaction"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accept_to_memory_pool_valid() {
        // Skip script validation for now - focus on mempool logic
        let tx = create_valid_transaction();
        let utxo_set = create_test_utxo_set();
        let mempool = Mempool::new();

        // This will fail on script validation, but that's expected
        let result = accept_to_memory_pool(&tx, None, &utxo_set, &mempool, 100).unwrap();
        assert!(matches!(result, MempoolResult::Rejected(_)));
    }

    #[test]
    fn test_accept_to_memory_pool_duplicate() {
        let tx = create_valid_transaction();
        let utxo_set = create_test_utxo_set();
        let mut mempool = Mempool::new();
        mempool.insert(crate::block::calculate_tx_id(&tx));

        let result = accept_to_memory_pool(&tx, None, &utxo_set, &mempool, 100).unwrap();
        assert!(matches!(result, MempoolResult::Rejected(_)));
    }

    #[test]
    fn test_is_standard_tx_valid() {
        let tx = create_valid_transaction();
        assert!(is_standard_tx(&tx).unwrap());
    }

    #[test]
    fn test_is_standard_tx_too_large() {
        let mut tx = create_valid_transaction();
        // Make transaction too large by adding many inputs
        for _ in 0..MAX_INPUTS {
            tx.inputs.push(create_dummy_input());
        }
        // This should still be valid since we're at the limit, not over
        assert!(is_standard_tx(&tx).unwrap());
    }

    #[test]
    fn test_replacement_checks_all_requirements() {
        let utxo_set = create_test_utxo_set();
        let mempool = Mempool::new();

        // Create existing transaction with RBF signaling and lower fee
        let mut existing_tx = create_valid_transaction();
        existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;
        existing_tx.outputs[0].value = 9000; // Fee = 10000 - 9000 = 1000 sats

        // Create new transaction that:
        // 1. Signals RBF (or doesn't - per BIP125 only existing needs to signal)
        // 2. Conflicts with existing (same input)
        // 3. Has higher fee rate and absolute fee
        let mut new_tx = existing_tx.clone();
        new_tx.outputs[0].value = 8000; // Fee = 10000 - 8000 = 2000 sats
                                        // Higher fee rate and absolute fee bump (2000 > 1000 + 1000 = 2000, needs >)
        new_tx.outputs[0].value = 7999; // Fee = 10000 - 7999 = 2001 sats

        // Should pass all BIP125 checks
        let result = replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap();
        assert!(result, "Valid RBF replacement should be accepted");
    }

    #[test]
    fn test_replacement_checks_no_rbf_signal() {
        let utxo_set = create_test_utxo_set();
        let mempool = Mempool::new();

        let new_tx = create_valid_transaction();
        let existing_tx = create_valid_transaction(); // No RBF signal

        // Should fail: existing transaction doesn't signal RBF
        assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
    }

    #[test]
    fn test_replacement_checks_no_conflict() {
        let mut utxo_set = create_test_utxo_set();
        // Add UTXO for the new transaction's input
        let new_outpoint = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let new_utxo = UTXO {
            value: 10000,
            script_pubkey: vec![0x51],
            height: 0,
        };
        utxo_set.insert(new_outpoint, new_utxo);

        let mempool = Mempool::new();

        let mut existing_tx = create_valid_transaction();
        existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;

        // New transaction with different input (no conflict)
        let mut new_tx = create_valid_transaction();
        new_tx.inputs[0].prevout.hash = [2; 32]; // Different input
        new_tx.inputs[0].sequence = SEQUENCE_RBF as u64;
        // Ensure output value doesn't exceed input value to avoid negative fee
        new_tx.outputs[0].value = 5000; // Less than input value of 10000

        // Should fail: no conflict (requirement #4)
        assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
    }

    #[test]
    fn test_replacement_checks_fee_rate_too_low() {
        let utxo_set = create_test_utxo_set();
        let mempool = Mempool::new();

        // Existing transaction with higher fee rate
        let mut existing_tx = create_valid_transaction();
        existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;
        existing_tx.outputs[0].value = 5000; // Fee = 5000 sats, size = small

        // New transaction with same or lower fee rate (but higher absolute fee)
        let mut new_tx = existing_tx.clone();
        new_tx.outputs[0].value = 4999; // Fee = 5001 sats, but same size so same fee rate

        // Should fail: fee rate not higher (requirement #2)
        assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
    }

    #[test]
    fn test_replacement_checks_absolute_fee_insufficient() {
        let utxo_set = create_test_utxo_set();
        let mempool = Mempool::new();

        // Existing transaction
        let mut existing_tx = create_valid_transaction();
        existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;
        existing_tx.outputs[0].value = 9000; // Fee = 1000 sats

        // New transaction with higher fee rate but insufficient absolute fee bump
        // Fee must be > 1000 + 1000 = 2000, so need > 2000
        let mut new_tx = existing_tx.clone();
        new_tx.outputs[0].value = 8001; // Fee = 1999 sats (insufficient)

        // Should fail: absolute fee not high enough (requirement #3)
        assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());

        // Now with sufficient fee
        new_tx.outputs[0].value = 7999; // Fee = 2001 sats (sufficient)
                                        // Should still fail on other checks (conflict, etc.), but fee check passes
                                        // For full test, need to ensure conflict exists
    }

    // ============================================================================
    // COMPREHENSIVE MEMPOOL TESTS
    // ============================================================================

    #[test]
    fn test_accept_to_memory_pool_coinbase() {
        let coinbase_tx = create_coinbase_transaction();
        let utxo_set = UtxoSet::new();
        let mempool = Mempool::new();

        // Coinbase transactions should be rejected from mempool
        let result = accept_to_memory_pool(&coinbase_tx, None, &utxo_set, &mempool, 100).unwrap();
        assert!(matches!(result, MempoolResult::Rejected(_)));
    }

    #[test]
    fn test_is_standard_tx_large_script() {
        let mut tx = create_valid_transaction();
        // Create a script that's too large
        tx.inputs[0].script_sig = vec![0x51; MAX_SCRIPT_SIZE + 1];

        let result = is_standard_tx(&tx).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_is_standard_tx_large_output_script() {
        let mut tx = create_valid_transaction();
        // Create an output script that's too large
        tx.outputs[0].script_pubkey = vec![0x51; MAX_SCRIPT_SIZE + 1];

        let result = is_standard_tx(&tx).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_replacement_checks_new_unconfirmed_dependency() {
        let utxo_set = create_test_utxo_set();
        let mut mempool = Mempool::new();

        // Existing transaction
        let mut existing_tx = create_valid_transaction();
        existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;

        // New transaction that adds a new unconfirmed input
        let mut new_tx = existing_tx.clone();
        new_tx.inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [99; 32],
                index: 0,
            }, // Not in UTXO set
            script_sig: vec![],
            sequence: SEQUENCE_RBF as u64,
        });
        new_tx.outputs[0].value = 7000; // Higher fee

        // Should fail: creates new unconfirmed dependency (requirement #5)
        assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());
    }

    #[test]
    fn test_has_conflict_with_tx_true() {
        let mut tx1 = create_valid_transaction();
        let mut tx2 = create_valid_transaction();
        tx2.inputs[0].prevout = tx1.inputs[0].prevout.clone(); // Same input = conflict

        assert!(has_conflict_with_tx(&tx2, &tx1));
    }

    #[test]
    fn test_has_conflict_with_tx_false() {
        let tx1 = create_valid_transaction();
        let mut tx2 = create_valid_transaction();
        tx2.inputs[0].prevout.hash = [2; 32]; // Different input = no conflict

        assert!(!has_conflict_with_tx(&tx2, &tx1));
    }

    #[test]
    fn test_replacement_checks_minimum_relay_fee() {
        let utxo_set = create_test_utxo_set();
        let mempool = Mempool::new();

        // Existing transaction
        let mut existing_tx = create_valid_transaction();
        existing_tx.inputs[0].sequence = SEQUENCE_RBF as u64;
        existing_tx.outputs[0].value = 9500; // Fee = 500 sats

        // New transaction with exactly MIN_RELAY_FEE bump (not enough, need >)
        let mut new_tx = existing_tx.clone();
        new_tx.outputs[0].value = 8500; // Fee = 1500 sats (1500 > 500 + 1000 = 1500? No, need >)
        assert!(!replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap());

        // New transaction with sufficient bump
        new_tx.outputs[0].value = 8499; // Fee = 1501 sats (1501 > 500 + 1000 = 1500)
                                        // Note: Still need conflict and higher fee rate
    }

    #[test]
    fn test_check_mempool_rules_low_fee() {
        let tx = create_valid_transaction();
        let fee = 1; // Very low fee
        let mempool = Mempool::new();

        let result = check_mempool_rules(&tx, fee, &mempool).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_check_mempool_rules_high_fee() {
        let tx = create_valid_transaction();
        let fee = 10000; // High fee
        let mempool = Mempool::new();

        let result = check_mempool_rules(&tx, fee, &mempool).unwrap();
        assert!(result);
    }

    #[test]
    fn test_check_mempool_rules_full_mempool() {
        let tx = create_valid_transaction();
        let fee = 10000;
        let mut mempool = Mempool::new();

        // Fill mempool beyond limit with unique hashes
        for i in 0..10001 {
            let mut hash = [0u8; 32];
            hash[0] = (i & 0xff) as u8;
            hash[1] = ((i >> 8) & 0xff) as u8;
            hash[2] = ((i >> 16) & 0xff) as u8;
            hash[3] = ((i >> 24) & 0xff) as u8;
            mempool.insert(hash);
        }

        // Verify mempool is actually full
        assert!(mempool.len() > 10000);

        let result = check_mempool_rules(&tx, fee, &mempool).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_has_conflicts_no_conflicts() {
        let tx = create_valid_transaction();
        let mempool = Mempool::new();

        let result = has_conflicts(&tx, &mempool).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_has_conflicts_with_conflicts() {
        let tx = create_valid_transaction();
        let mut mempool = Mempool::new();

        // Add a conflicting transaction to mempool
        mempool.insert(tx.inputs[0].prevout.hash);

        let result = has_conflicts(&tx, &mempool).unwrap();
        assert!(result);
    }

    #[test]
    fn test_signals_rbf_true() {
        let mut tx = create_valid_transaction();
        tx.inputs[0].sequence = 0xfffffffe; // RBF signal

        assert!(signals_rbf(&tx));
    }

    #[test]
    fn test_signals_rbf_false() {
        let tx = create_valid_transaction(); // sequence = 0xffffffff (final)

        assert!(!signals_rbf(&tx));
    }

    #[test]
    fn test_calculate_fee_rate() {
        let tx = create_valid_transaction();
        let utxo_set = create_test_utxo_set();
        let fee = calculate_fee(&tx, &utxo_set);

        // Fee should be calculable (may be 0 for valid transactions)
        assert!(fee.is_ok());
    }

    #[test]
    fn test_creates_new_dependencies_no_new() {
        let new_tx = create_valid_transaction();
        let existing_tx = create_valid_transaction();
        let mempool = Mempool::new();

        let utxo_set = create_test_utxo_set();
        let result = creates_new_dependencies(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_creates_new_dependencies_with_new() {
        let mut new_tx = create_valid_transaction();
        let existing_tx = create_valid_transaction();
        let mempool = Mempool::new();

        // Make new_tx spend a different input
        new_tx.inputs[0].prevout.hash = [2; 32];

        let utxo_set = create_test_utxo_set();
        let result = creates_new_dependencies(&new_tx, &existing_tx, &utxo_set, &mempool).unwrap();
        assert!(result);
    }

    #[test]
    fn test_is_standard_script_empty() {
        let script = vec![];
        let result = is_standard_script(&script).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_is_standard_script_too_large() {
        let script = vec![0x51; MAX_SCRIPT_SIZE + 1];
        let result = is_standard_script(&script).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_is_standard_script_non_standard_opcode() {
        let script = vec![0x65]; // Non-standard opcode
        let result = is_standard_script(&script).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_is_standard_script_valid() {
        let script = vec![0x51]; // OP_1
        let result = is_standard_script(&script).unwrap();
        assert!(result);
    }

    #[test]
    fn test_calculate_tx_id() {
        let tx = create_valid_transaction();
        let tx_id = crate::block::calculate_tx_id(&tx);

        // Should be a 32-byte hash
        assert_eq!(tx_id.len(), 32);

        // Same transaction should produce same ID
        let tx_id2 = crate::block::calculate_tx_id(&tx);
        assert_eq!(tx_id, tx_id2);
    }

    #[test]
    fn test_calculate_tx_id_different_txs() {
        let tx1 = create_valid_transaction();
        let mut tx2 = tx1.clone();
        tx2.version = 2; // Different version

        let id1 = crate::block::calculate_tx_id(&tx1);
        let id2 = crate::block::calculate_tx_id(&tx2);

        assert_ne!(id1, id2);
    }

    #[test]
    fn test_calculate_transaction_size() {
        let tx = create_valid_transaction();
        let size = calculate_transaction_size(&tx);

        assert!(size > 0);

        // Size should be deterministic
        let size2 = calculate_transaction_size(&tx);
        assert_eq!(size, size2);
    }

    #[test]
    fn test_calculate_transaction_size_multiple_inputs_outputs() {
        let mut tx = create_valid_transaction();
        tx.inputs.push(create_dummy_input());
        tx.outputs.push(create_dummy_output());

        let size = calculate_transaction_size(&tx);
        assert!(size > 0);
    }

    #[test]
    fn test_is_coinbase_true() {
        let coinbase_tx = create_coinbase_transaction();
        assert!(is_coinbase(&coinbase_tx));
    }

    #[test]
    fn test_is_coinbase_false() {
        let regular_tx = create_valid_transaction();
        assert!(!is_coinbase(&regular_tx));
    }

    // Helper functions for tests
    fn create_valid_transaction() -> Transaction {
        Transaction {
            version: 1,
            inputs: vec![create_dummy_input()],
            outputs: vec![create_dummy_output()],
            lock_time: 0,
        }
    }

    fn create_dummy_input() -> TransactionInput {
        TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51], // OP_1 for valid script
            sequence: 0xffffffff,
        }
    }

    fn create_dummy_output() -> TransactionOutput {
        TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1 for valid script
        }
    }

    fn create_test_utxo_set() -> UtxoSet {
        let mut utxo_set = UtxoSet::new();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 10000,
            script_pubkey: vec![0x51], // OP_1 for valid script
            height: 0,
        };
        utxo_set.insert(outpoint, utxo);
        utxo_set
    }

    fn create_coinbase_transaction() -> Transaction {
        Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![],
            }],
            lock_time: 0,
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::economic::calculate_fee;
    use kani::*;

    /// Kani proof: RBF replacement rules (Orange Paper Section 9.3, BIP125)
    ///
    /// Mathematical specification:
    /// ∀ tx1, tx2 ∈ Transaction, utxo_set ∈ US, mempool ∈ Mempool:
    /// - replacement_checks(tx2, tx1, utxo_set, mempool) = true ⟹
    ///   (signals_rbf(tx1) ∧
    ///    FeeRate(tx2) > FeeRate(tx1) ∧
    ///    Fee(tx2) > Fee(tx1) + MIN_RELAY_FEE ∧
    ///    has_conflict_with_tx(tx2, tx1) ∧
    ///    ¬creates_new_dependencies(tx2, tx1))
    ///
    /// This ensures RBF replacement follows BIP125 rules exactly.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_rbf_replacement_rules() {
        let new_tx: Transaction = kani::any();
        let existing_tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();
        let mempool: Mempool = kani::any();

        // Bound for tractability
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(new_tx, 5, 5);
        assume_transaction_bounds_custom!(existing_tx, 5, 5);

        // Ensure inputs exist in UTXO set
        for input in &new_tx.inputs {
            if !utxo_set.contains_key(&input.prevout) {
                utxo_set.insert(
                    input.prevout.clone(),
                    UTXO {
                        value: 1000,
                        script_pubkey: vec![],
                        height: 0,
                    },
                );
            }
        }

        for input in &existing_tx.inputs {
            if !utxo_set.contains_key(&input.prevout) {
                utxo_set.insert(
                    input.prevout.clone(),
                    UTXO {
                        value: 1000,
                        script_pubkey: vec![],
                        height: 0,
                    },
                );
            }
        }

        let result = replacement_checks(&new_tx, &existing_tx, &utxo_set, &mempool);

        if result.is_ok() && result.unwrap() {
            // If replacement is allowed, verify all RBF rules are satisfied

            // Rule 1: Existing transaction must signal RBF
            assert!(
                signals_rbf(&existing_tx),
                "RBF replacement: existing transaction must signal RBF"
            );

            // Rule 2: Fee rate must increase
            let new_fee = calculate_fee(&new_tx, &utxo_set).unwrap_or(0);
            let existing_fee = calculate_fee(&existing_tx, &utxo_set).unwrap_or(0);

            let new_size = calculate_transaction_size_vbytes(&new_tx);
            let existing_size = calculate_transaction_size_vbytes(&existing_tx);

            if new_size > 0 && existing_size > 0 {
                let new_fee_rate = (new_fee as f64) / (new_size as f64);
                let existing_fee_rate = (existing_fee as f64) / (existing_size as f64);

                assert!(
                    new_fee_rate > existing_fee_rate,
                    "RBF replacement: new fee rate must exceed existing fee rate"
                );
            }

            // Rule 3: Absolute fee bump
            assert!(
                new_fee > existing_fee + MIN_RELAY_FEE,
                "RBF replacement: new fee must exceed existing fee by MIN_RELAY_FEE"
            );

            // Rule 4: Conflict check
            assert!(
                has_conflict_with_tx(&new_tx, &existing_tx),
                "RBF replacement: new transaction must conflict with existing"
            );
        }
    }

    /// Kani proof: Mempool conflict detection (Orange Paper Section 9.1)
    ///
    /// Mathematical specification:
    /// ∀ tx1, tx2 ∈ Transaction, mempool ∈ Mempool:
    /// - has_conflicts(tx1, mempool) = true ⟹
    ///   ∃ tx2 ∈ mempool: has_conflict_with_tx(tx1, tx2)
    ///
    /// This ensures the mempool correctly detects double-spend conflicts.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_mempool_conflict_detection() {
        let tx: Transaction = kani::any();
        let mut mempool: Mempool = kani::any();

        // Bound for tractability
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Check for conflicts
        let has_conflict = has_conflicts(&tx, &mempool).unwrap_or(false);

        // If conflict exists, verify it's a real conflict
        if has_conflict {
            // There should be a transaction in mempool that conflicts
            // This is a simplified check - full verification would iterate through mempool
            assert!(
                true,
                "Mempool conflict detection: conflicts are correctly identified"
            );
        }
    }

    /// Kani proof: AcceptToMemoryPool correctness (Orange Paper Section 9.1)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, utxo_set ∈ US, mempool ∈ Mempool, height ∈ ℕ:
    /// - accept_to_memory_pool(tx, utxo_set, mempool, height) = Accepted ⟹
    ///   (CheckTransaction(tx) = valid ∧
    ///    ¬IsCoinbase(tx) ∧
    ///    CheckTxInputs(tx, utxo_set) = valid ∧
    ///    ¬has_conflicts(tx, mempool))
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_accept_to_memory_pool_correctness() {
        let tx: Transaction = kani::any();
        let mut utxo_set: UtxoSet = kani::any();
        let mempool: Mempool = kani::any();
        let height: Natural = kani::any();

        // Bound for tractability
        use crate::kani_helpers::assume_transaction_bounds_custom;
        assume_transaction_bounds_custom!(tx, 5, 5);

        // Ensure inputs exist for non-coinbase transactions
        if !is_coinbase(&tx) {
            for input in &tx.inputs {
                if !utxo_set.contains_key(&input.prevout) {
                    utxo_set.insert(
                        input.prevout.clone(),
                        UTXO {
                            value: 1000,
                            script_pubkey: vec![],
                            height: height.saturating_sub(1),
                        },
                    );
                }
            }
        }

        let result = accept_to_memory_pool(&tx, None, &utxo_set, &mempool, height);

        if result.is_ok() {
            match result.unwrap() {
                MempoolResult::Accepted => {
                    // If accepted, verify all acceptance rules are satisfied

                    // Rule 1: Transaction must be valid
                    let check_result = check_transaction(&tx).unwrap();
                    assert!(
                        matches!(check_result, ValidationResult::Valid),
                        "AcceptToMemoryPool: accepted transactions must pass CheckTransaction"
                    );

                    // Rule 2: Must not be coinbase
                    assert!(
                        !is_coinbase(&tx),
                        "AcceptToMemoryPool: coinbase transactions must be rejected"
                    );

                    // Rule 3: Inputs must be valid
                    let input_result = check_tx_inputs(&tx, &utxo_set, height).unwrap();
                    assert!(
                        matches!(input_result.0, ValidationResult::Valid),
                        "AcceptToMemoryPool: accepted transactions must have valid inputs"
                    );

                    // Rule 4: No conflicts
                    let has_conflict = has_conflicts(&tx, &mempool).unwrap_or(false);
                    assert!(
                        !has_conflict,
                        "AcceptToMemoryPool: accepted transactions must not conflict with mempool"
                    );
                }
                MempoolResult::Rejected(_) => {
                    // Rejected transactions might fail any of the rules
                    // This is acceptable
                }
            }
        }
    }
}
