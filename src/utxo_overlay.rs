//! Zero-Copy UTXO Overlay
//!
//! Provides a copy-on-write view over the UTXO set for block validation.
//! Instead of cloning the entire UTXO set (O(n) where n = millions of UTXOs),
//! we maintain a small overlay of changes.
//!
//! This is critical for performance at high block heights where the UTXO set
//! contains ~75 million entries.
//!
//! ## Design
//!
//! ```text
//! ┌─────────────────┐
//! │  UtxoOverlay    │  ← Small (block-sized changes)
//! │  - additions    │
//! │  - deletions    │
//! └────────┬────────┘
//!          │ fallback
//! ┌────────▼────────┐
//! │  Base UtxoSet   │  ← Large (millions of UTXOs), read-only
//! └─────────────────┘
//! ```
//!
//! ## Performance
//!
//! - Clone: O(1) instead of O(n)
//! - Lookup: O(1) with small constant overhead
//! - Insert: O(1)
//! - Memory: O(block_tx_count) instead of O(utxo_set_size)

use crate::types::{utxo_set_insert, OutPoint, UTXO, UtxoSet};
#[cfg(feature = "production")]
use rustc_hash::{FxHashMap, FxHashSet};
#[cfg(not(feature = "production"))]
use std::collections::{HashMap as FxHashMap, HashSet as FxHashSet};

/// Fixed-size key for deletions set — avoids OutPoint clone in remove() (~3k inputs/block).
type OutPointKey = [u8; 36];

#[inline]
fn outpoint_to_key(op: &OutPoint) -> OutPointKey {
    let mut key = [0u8; 36];
    key[..32].copy_from_slice(&op.hash);
    key[32..36].copy_from_slice(&op.index.to_be_bytes());
    key
}

#[inline]
fn key_to_outpoint(key: &OutPointKey) -> OutPoint {
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&key[..32]);
    let index = u32::from_be_bytes(key[32..36].try_into().unwrap());
    OutPoint { hash, index }
}

/// Trait for UTXO lookups - implemented by both UtxoSet and UtxoOverlay.
///
/// This allows check_tx_inputs and other validation functions to work
/// with either type without code duplication.
pub trait UtxoLookup {
    /// Look up a UTXO by outpoint.
    fn get(&self, outpoint: &OutPoint) -> Option<&UTXO>;
    
    /// Check if a UTXO exists.
    fn contains_key(&self, outpoint: &OutPoint) -> bool {
        self.get(outpoint).is_some()
    }
    
    /// Get the number of UTXOs (approximate for overlays).
    fn len(&self) -> usize;
    
    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implementation for standard UtxoSet (HashMap / FxHashMap).
#[cfg(feature = "production")]
impl UtxoLookup for UtxoSet {
    #[inline]
    fn get(&self, outpoint: &OutPoint) -> Option<&UTXO> {
        FxHashMap::get(self, outpoint).map(|a| a.as_ref())
    }
    #[inline]
    fn contains_key(&self, outpoint: &OutPoint) -> bool {
        FxHashMap::contains_key(self, outpoint)
    }
    #[inline]
    fn len(&self) -> usize { FxHashMap::len(self) }
    #[inline]
    fn is_empty(&self) -> bool { FxHashMap::is_empty(self) }
}
#[cfg(not(feature = "production"))]
impl UtxoLookup for UtxoSet {
    #[inline]
    fn get(&self, outpoint: &OutPoint) -> Option<&UTXO> {
        std::collections::HashMap::get(self, outpoint).map(|a| a.as_ref())
    }
    #[inline]
    fn contains_key(&self, outpoint: &OutPoint) -> bool {
        std::collections::HashMap::contains_key(self, outpoint)
    }
    #[inline]
    fn len(&self) -> usize { std::collections::HashMap::len(self) }
    #[inline]
    fn is_empty(&self) -> bool { std::collections::HashMap::is_empty(self) }
}

/// Zero-copy overlay for UTXO set modifications during block validation.
///
/// Provides a view over the base UTXO set that captures additions and deletions
/// without copying the underlying data.
#[derive(Debug)]
pub struct UtxoOverlay<'a> {
    /// Reference to the base UTXO set (read-only)
    base: &'a UtxoSet,
    /// UTXOs added in this overlay (new outputs from current block)
    additions: FxHashMap<OutPoint, std::sync::Arc<UTXO>>,
    /// UTXOs marked as spent in this overlay (OutPointKey to avoid clone in remove hot path)
    deletions: FxHashSet<OutPointKey>,
    /// OPTIMIZATION #3: Cached flag to skip deletions.contains() check when empty
    has_deletions: bool,
}

impl<'a> UtxoOverlay<'a> {
    /// Create a new overlay over the given UTXO set.
    /// 
    /// This is O(1) - no copying of the base set.
    #[inline]
    pub fn new(base: &'a UtxoSet) -> Self {
        Self {
            base,
            // Pre-allocate for typical block size (~3000 transactions)
            additions: FxHashMap::with_capacity_and_hasher(6000, Default::default()),
            deletions: FxHashSet::with_capacity_and_hasher(6000, Default::default()),
            has_deletions: false, // OPTIMIZATION #3: Start with no deletions
        }
    }

    /// Create a new overlay with custom capacity hints.
    #[inline]
    pub fn with_capacity(base: &'a UtxoSet, additions_cap: usize, deletions_cap: usize) -> Self {
        Self {
            base,
            additions: FxHashMap::with_capacity_and_hasher(additions_cap, Default::default()),
            deletions: FxHashSet::with_capacity_and_hasher(deletions_cap, Default::default()),
            has_deletions: false, // OPTIMIZATION #3: Start with no deletions
        }
    }

    /// Look up a UTXO by outpoint.
    /// 
    /// First checks deletions, then additions, then base set.
    #[inline]
    pub fn get(&self, outpoint: &OutPoint) -> Option<&UTXO> {
        if self.has_deletions && self.deletions.contains(&outpoint_to_key(outpoint)) {
            return None;
        }
        
        // Check additions first (for intra-block spending)
        if let Some(arc) = self.additions.get(outpoint) {
            return Some(arc.as_ref());
        }
        
        // Fall back to base set (UtxoSet holds Arc<UTXO>; deref to &UTXO)
        self.base.get(outpoint).map(|a| a.as_ref())
    }

    /// Check if a UTXO exists.
    #[inline]
    pub fn contains_key(&self, outpoint: &OutPoint) -> bool {
        if self.has_deletions && self.deletions.contains(&outpoint_to_key(outpoint)) {
            return false;
        }
        self.additions.contains_key(outpoint) || self.base.contains_key(outpoint)
    }

    /// Add a new UTXO (created by a transaction in this block).
    #[inline]
    pub fn insert(&mut self, outpoint: OutPoint, utxo: UTXO) {
        self.insert_arc(outpoint, std::sync::Arc::new(utxo));
    }

    /// Add a new UTXO from Arc. Use when sharing with undo log.
    #[inline]
    pub fn insert_arc(&mut self, outpoint: OutPoint, utxo: std::sync::Arc<UTXO>) {
        if self.deletions.remove(&outpoint_to_key(&outpoint)) {
            // Update flag if deletions set becomes empty
            if self.deletions.is_empty() {
                self.has_deletions = false;
            }
        }
        self.additions.insert(outpoint, utxo);
    }

    /// Mark a UTXO as spent without returning it. Use when the return value is discarded (e.g. IBD path).
    /// Avoids cloning the UTXO (~50 bytes per input).
    #[inline]
    pub fn mark_spent(&mut self, outpoint: &OutPoint) {
        // Check additions first (intra-block spend)
        if self.additions.remove(outpoint).is_some() {
            return;
        }
        // Mark as deleted from base set (no clone — caller doesn't need the UTXO)
        if self.base.contains_key(outpoint) {
            self.deletions.insert(outpoint_to_key(outpoint));
            self.has_deletions = true;
        }
    }

    /// Mark a UTXO as spent (consumed by a transaction in this block).
    /// Returns the UTXO for undo-log path. Use `mark_spent` when return value is discarded.
    #[inline]
    pub fn remove(&mut self, outpoint: &OutPoint) -> Option<std::sync::Arc<UTXO>> {
        // Check additions first (intra-block spend)
        if let Some(arc) = self.additions.remove(outpoint) {
            return Some(arc);
        }
        // Mark as deleted from base set (base holds Arc<UTXO>; share Arc for undo)
        if let Some(arc) = self.base.get(outpoint) {
            self.deletions.insert(outpoint_to_key(outpoint));
            self.has_deletions = true;
            return Some(std::sync::Arc::clone(arc));
        }
        
        None
    }

    /// Get the number of additions in this overlay.
    #[inline]
    pub fn additions_len(&self) -> usize {
        self.additions.len()
    }

    /// Get the number of deletions in this overlay.
    #[inline]
    pub fn deletions_len(&self) -> usize {
        self.deletions.len()
    }

    /// Get the size of the base UTXO set.
    #[inline]
    pub fn base_len(&self) -> usize {
        self.base.len()
    }

    /// Apply the overlay changes to produce a new UTXO set.
    ///
    /// This is called at the end of successful block validation.
    /// Base clone is cheap (Arc refcount only); additions wrapped in Arc.
    pub fn apply_to_base(self) -> UtxoSet {
        let mut result = self.base.clone();
        for key in self.deletions {
            result.remove(&key_to_outpoint(&key));
        }
        for (outpoint, arc) in self.additions {
            result.insert(outpoint, arc);
        }
        result
    }

    /// Get immutable access to additions (for undo log generation).
    #[inline]
    pub fn additions(&self) -> &FxHashMap<OutPoint, std::sync::Arc<UTXO>> {
        &self.additions
    }

    /// Get immutable access to deletion keys (for undo log generation).
    #[inline]
    pub fn deletions(&self) -> &FxHashSet<OutPointKey> {
        &self.deletions
    }

    /// Consume the overlay and return its changes as owned data.
    ///
    /// This releases the borrow on the base UTXO set, allowing the caller
    /// to apply the changes directly to the mutable utxo_set without
    /// re-iterating all transactions.
    ///
    /// Returns (additions, deletions) — the net UTXO changes from this block.
    #[inline]
    pub fn into_changes(self) -> (FxHashMap<OutPoint, std::sync::Arc<UTXO>>, FxHashSet<OutPoint>) {
        let deletions = self.deletions.into_iter().map(|k| key_to_outpoint(&k)).collect();
        (self.additions, deletions)
    }
}

/// Implementation of UtxoLookup for UtxoOverlay.
impl<'a> UtxoLookup for UtxoOverlay<'a> {
    #[inline]
    fn get(&self, outpoint: &OutPoint) -> Option<&UTXO> {
        if self.has_deletions && self.deletions.contains(&outpoint_to_key(outpoint)) {
            return None;
        }
        if let Some(arc) = self.additions.get(outpoint) {
            return Some(arc.as_ref());
        }
        self.base.get(outpoint).map(|a| a.as_ref())
    }

    #[inline]
    fn contains_key(&self, outpoint: &OutPoint) -> bool {
        if self.has_deletions && self.deletions.contains(&outpoint_to_key(outpoint)) {
            return false;
        }
        self.additions.contains_key(outpoint) || self.base.contains_key(outpoint)
    }
    
    #[inline]
    fn len(&self) -> usize {
        // Approximate: base size + additions - deletions
        // Note: This may not be exact if additions shadow base entries
        self.base.len() + self.additions.len() - self.deletions.len()
    }
}

/// Fast UTXO set using FxHash for better performance.
/// 
/// FxHashMap uses a faster hash function than the default SipHash.
/// For fixed-size keys like OutPoint (36 bytes), this is 2-3x faster.
#[cfg(feature = "production")]
pub type FastUtxoSet = rustc_hash::FxHashMap<OutPoint, UTXO>;

#[cfg(not(feature = "production"))]
pub type FastUtxoSet = std::collections::HashMap<OutPoint, UTXO>;

/// Convert standard UtxoSet to FastUtxoSet (clones UTXOs).
#[cfg(feature = "production")]
#[inline]
pub fn to_fast_utxo_set(utxo_set: &UtxoSet) -> FastUtxoSet {
    utxo_set.iter()
        .map(|(k, v)| (k.clone(), (**v).clone()))
        .collect()
}

#[cfg(not(feature = "production"))]
#[inline]
pub fn to_fast_utxo_set(utxo_set: &UtxoSet) -> FastUtxoSet {
    utxo_set.iter()
        .map(|(k, v)| (k.clone(), (**v).clone()))
        .collect()
}

/// Apply a transaction to the overlay (for validation phase).
///
/// This is the overlay-compatible version of `apply_transaction_with_id`.
/// It modifies the overlay in place rather than returning a new UTXO set.
///
/// Returns undo entries for use in block undo log (optional in validation phase).
#[inline]
pub fn apply_transaction_to_overlay(
    overlay: &mut UtxoOverlay<'_>,
    tx: &crate::types::Transaction,
    tx_id: crate::types::Hash,
    height: crate::types::Natural,
) -> Vec<crate::reorganization::UndoEntry> {
    use crate::reorganization::UndoEntry;
    use crate::transaction::is_coinbase;
    
    let mut undo_entries = Vec::with_capacity(tx.inputs.len() + tx.outputs.len());
    
    // Remove spent inputs (except for coinbase)
    if !is_coinbase(tx) {
        for input in &tx.inputs {
            // Record the UTXO that existed before (for restoration during disconnect)
            if let Some(previous_utxo) = overlay.remove(&input.prevout) {
                undo_entries.push(UndoEntry {
                    outpoint: input.prevout,
                    previous_utxo: Some(previous_utxo),
                    new_utxo: None, // This UTXO is being spent
                });
            }
        }
    }
    
    // Add new outputs
    for (i, output) in tx.outputs.iter().enumerate() {
        let outpoint = OutPoint {
            hash: tx_id,
            index: i as u32,
        };
        
        let utxo = UTXO {
            value: output.value,
            script_pubkey: output.script_pubkey.as_slice().into(),
            height,
            is_coinbase: is_coinbase(tx),
        };
        
        let utxo_arc = std::sync::Arc::new(utxo);
        // Record new UTXO for undo log
        undo_entries.push(UndoEntry {
            outpoint,
            previous_utxo: None, // Newly created
            new_utxo: Some(std::sync::Arc::clone(&utxo_arc)),
        });
        
        overlay.insert_arc(outpoint, utxo_arc);
    }
    
    undo_entries
}

/// Apply a transaction to the overlay WITHOUT building undo entries.
///
/// This is the fast path for IBD where undo logs are discarded.
/// Avoids cloning outpoints and UTXOs for undo entries, saving
/// significant allocation overhead (2 clones per input, 2 per output).
#[inline]
pub fn apply_transaction_to_overlay_no_undo(
    overlay: &mut UtxoOverlay<'_>,
    tx: &crate::types::Transaction,
    tx_id: crate::types::Hash,
    height: crate::types::Natural,
) {
    use crate::transaction::is_coinbase;
    
    // Remove spent inputs (except for coinbase). Use mark_spent to avoid cloning UTXO.
    if !is_coinbase(tx) {
        for input in &tx.inputs {
            overlay.mark_spent(&input.prevout);
        }
    }
    
    // Add new outputs
    let is_cb = is_coinbase(tx);
    for (i, output) in tx.outputs.iter().enumerate() {
        let outpoint = OutPoint {
            hash: tx_id,
            index: i as u32,
        };
        
        let utxo = UTXO {
            value: output.value,
            script_pubkey: output.script_pubkey.as_slice().into(),
            height,
            is_coinbase: is_cb,
        };
        
        overlay.insert(outpoint, utxo);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outpoint(idx: u8) -> OutPoint {
        OutPoint {
            hash: [idx; 32],
            index: idx as u32,
        }
    }

    fn make_utxo(value: i64) -> UTXO {
        UTXO {
            value,
            script_pubkey: vec![crate::opcodes::OP_DUP, crate::opcodes::OP_HASH160].into(), // P2PKH prefix
            height: 1,
            is_coinbase: false,
        }
    }

    #[test]
    fn test_overlay_lookup_from_base() {
        let mut base = UtxoSet::default();
        utxo_set_insert(&mut base, make_outpoint(1), make_utxo(1000));
        utxo_set_insert(&mut base, make_outpoint(2), make_utxo(2000));
        
        let overlay = UtxoOverlay::new(&base);
        
        assert_eq!(overlay.get(&make_outpoint(1)).unwrap().value, 1000);
        assert_eq!(overlay.get(&make_outpoint(2)).unwrap().value, 2000);
        assert!(overlay.get(&make_outpoint(3)).is_none());
    }

    #[test]
    fn test_overlay_additions() {
        let base = UtxoSet::default();
        let mut overlay = UtxoOverlay::new(&base);
        
        overlay.insert(make_outpoint(1), make_utxo(1000));
        
        assert_eq!(overlay.get(&make_outpoint(1)).unwrap().value, 1000);
        assert_eq!(overlay.additions_len(), 1);
    }

    #[test]
    fn test_overlay_deletions() {
        let mut base = UtxoSet::default();
        utxo_set_insert(&mut base, make_outpoint(1), make_utxo(1000));
        
        let mut overlay = UtxoOverlay::new(&base);
        
        let removed = overlay.remove(&make_outpoint(1));
        assert_eq!(removed.unwrap().value, 1000);
        assert!(overlay.get(&make_outpoint(1)).is_none());
        assert_eq!(overlay.deletions_len(), 1);
    }

    #[test]
    fn test_overlay_intra_block_spend() {
        // Simulate spending an output created in the same block
        let base = UtxoSet::default();
        let mut overlay = UtxoOverlay::new(&base);
        
        // Add output
        overlay.insert(make_outpoint(1), make_utxo(1000));
        assert!(overlay.get(&make_outpoint(1)).is_some());
        
        // Spend it in same block
        let removed = overlay.remove(&make_outpoint(1));
        assert_eq!(removed.unwrap().value, 1000);
        assert!(overlay.get(&make_outpoint(1)).is_none());
        
        // Should not be in deletions (was only in additions)
        assert_eq!(overlay.deletions_len(), 0);
        assert_eq!(overlay.additions_len(), 0);
    }

    #[test]
    fn test_overlay_apply() {
        let mut base = UtxoSet::default();
        utxo_set_insert(&mut base, make_outpoint(1), make_utxo(1000));
        utxo_set_insert(&mut base, make_outpoint(2), make_utxo(2000));
        
        let mut overlay = UtxoOverlay::new(&base);
        
        // Remove one, add one
        overlay.remove(&make_outpoint(1));
        overlay.insert(make_outpoint(3), make_utxo(3000));
        
        let result = overlay.apply_to_base();
        
        assert!(result.get(&make_outpoint(1)).is_none()); // Removed
        assert_eq!(result.get(&make_outpoint(2)).unwrap().value, 2000); // Unchanged
        assert_eq!(result.get(&make_outpoint(3)).unwrap().value, 3000); // Added
    }

    #[test]
    fn test_overlay_no_clone_on_creation() {
        // This test verifies the design - overlay creation is O(1)
        let mut base = UtxoSet::default();
        for i in 0..10000 {
            utxo_set_insert(&mut base, make_outpoint(i as u8), make_utxo(i as i64));
        }
        
        // Creating overlay should be instant (no clone)
        let start = std::time::Instant::now();
        let _overlay = UtxoOverlay::new(&base);
        let elapsed = start.elapsed();
        
        // Should be sub-microsecond (just pointer + empty hashmap allocation)
        assert!(elapsed.as_micros() < 100, "Overlay creation took {:?}", elapsed);
    }
}

