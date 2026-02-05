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

use crate::types::{OutPoint, UTXO, UtxoSet};
use std::collections::HashMap;

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

/// Implementation for standard UtxoSet (HashMap).
impl UtxoLookup for UtxoSet {
    #[inline]
    fn get(&self, outpoint: &OutPoint) -> Option<&UTXO> {
        HashMap::get(self, outpoint)
    }
    
    #[inline]
    fn contains_key(&self, outpoint: &OutPoint) -> bool {
        HashMap::contains_key(self, outpoint)
    }
    
    #[inline]
    fn len(&self) -> usize {
        HashMap::len(self)
    }
    
    #[inline]
    fn is_empty(&self) -> bool {
        HashMap::is_empty(self)
    }
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
    additions: HashMap<OutPoint, UTXO>,
    /// UTXOs marked as spent in this overlay
    deletions: std::collections::HashSet<OutPoint>,
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
            additions: HashMap::with_capacity(6000),
            deletions: std::collections::HashSet::with_capacity(6000),
        }
    }

    /// Create a new overlay with custom capacity hints.
    #[inline]
    pub fn with_capacity(base: &'a UtxoSet, additions_cap: usize, deletions_cap: usize) -> Self {
        Self {
            base,
            additions: HashMap::with_capacity(additions_cap),
            deletions: std::collections::HashSet::with_capacity(deletions_cap),
        }
    }

    /// Look up a UTXO by outpoint.
    /// 
    /// First checks deletions, then additions, then base set.
    #[inline]
    pub fn get(&self, outpoint: &OutPoint) -> Option<&UTXO> {
        // Check if deleted in this overlay
        if self.deletions.contains(outpoint) {
            return None;
        }
        
        // Check additions first (for intra-block spending)
        if let Some(utxo) = self.additions.get(outpoint) {
            return Some(utxo);
        }
        
        // Fall back to base set
        self.base.get(outpoint)
    }

    /// Check if a UTXO exists.
    #[inline]
    pub fn contains_key(&self, outpoint: &OutPoint) -> bool {
        if self.deletions.contains(outpoint) {
            return false;
        }
        self.additions.contains_key(outpoint) || self.base.contains_key(outpoint)
    }

    /// Add a new UTXO (created by a transaction in this block).
    #[inline]
    pub fn insert(&mut self, outpoint: OutPoint, utxo: UTXO) {
        // Remove from deletions if re-adding
        self.deletions.remove(&outpoint);
        self.additions.insert(outpoint, utxo);
    }

    /// Mark a UTXO as spent (consumed by a transaction in this block).
    #[inline]
    pub fn remove(&mut self, outpoint: &OutPoint) -> Option<UTXO> {
        // Check additions first (intra-block spend)
        if let Some(utxo) = self.additions.remove(outpoint) {
            return Some(utxo);
        }
        
        // Mark as deleted from base set
        if let Some(utxo) = self.base.get(outpoint) {
            self.deletions.insert(outpoint.clone());
            return Some(utxo.clone());
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
    /// Note: This does clone, but only once per block instead of 3 times.
    pub fn apply_to_base(self) -> UtxoSet {
        let mut result = self.base.clone();
        
        // Remove spent UTXOs
        for outpoint in self.deletions {
            result.remove(&outpoint);
        }
        
        // Add new UTXOs
        for (outpoint, utxo) in self.additions {
            result.insert(outpoint, utxo);
        }
        
        result
    }

    /// Get immutable access to additions (for undo log generation).
    #[inline]
    pub fn additions(&self) -> &HashMap<OutPoint, UTXO> {
        &self.additions
    }

    /// Get immutable access to deletions (for undo log generation).
    #[inline]
    pub fn deletions(&self) -> &std::collections::HashSet<OutPoint> {
        &self.deletions
    }
}

/// Implementation of UtxoLookup for UtxoOverlay.
impl<'a> UtxoLookup for UtxoOverlay<'a> {
    #[inline]
    fn get(&self, outpoint: &OutPoint) -> Option<&UTXO> {
        // Check if deleted in this overlay
        if self.deletions.contains(outpoint) {
            return None;
        }
        
        // Check additions first (for intra-block spending)
        if let Some(utxo) = self.additions.get(outpoint) {
            return Some(utxo);
        }
        
        // Fall back to base set
        self.base.get(outpoint)
    }
    
    #[inline]
    fn contains_key(&self, outpoint: &OutPoint) -> bool {
        if self.deletions.contains(outpoint) {
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
pub type FastUtxoSet = HashMap<OutPoint, UTXO>;

/// Convert standard UtxoSet to FastUtxoSet.
#[cfg(feature = "production")]
#[inline]
pub fn to_fast_utxo_set(utxo_set: &UtxoSet) -> FastUtxoSet {
    utxo_set.iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[cfg(not(feature = "production"))]
#[inline]
pub fn to_fast_utxo_set(utxo_set: &UtxoSet) -> FastUtxoSet {
    utxo_set.clone()
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
    
    let mut undo_entries = Vec::new();
    
    // Remove spent inputs (except for coinbase)
    if !is_coinbase(tx) {
        for input in &tx.inputs {
            // Record the UTXO that existed before (for restoration during disconnect)
            if let Some(previous_utxo) = overlay.remove(&input.prevout) {
                undo_entries.push(UndoEntry {
                    outpoint: input.prevout.clone(),
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
            index: i as crate::types::Natural,
        };
        
        let utxo = UTXO {
            value: output.value,
            script_pubkey: output.script_pubkey.clone(),
            height,
            is_coinbase: is_coinbase(tx),
        };
        
        // Record new UTXO for undo log
        undo_entries.push(UndoEntry {
            outpoint: outpoint.clone(),
            previous_utxo: None, // Newly created
            new_utxo: Some(utxo.clone()),
        });
        
        overlay.insert(outpoint, utxo);
    }
    
    undo_entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_outpoint(idx: u8) -> OutPoint {
        OutPoint {
            hash: [idx; 32],
            index: idx as u64,
        }
    }

    fn make_utxo(value: i64) -> UTXO {
        UTXO {
            value,
            script_pubkey: vec![0x76, 0xa9], // P2PKH prefix
            height: 1,
            is_coinbase: false,
        }
    }

    #[test]
    fn test_overlay_lookup_from_base() {
        let mut base = UtxoSet::new();
        base.insert(make_outpoint(1), make_utxo(1000));
        base.insert(make_outpoint(2), make_utxo(2000));
        
        let overlay = UtxoOverlay::new(&base);
        
        assert_eq!(overlay.get(&make_outpoint(1)).unwrap().value, 1000);
        assert_eq!(overlay.get(&make_outpoint(2)).unwrap().value, 2000);
        assert!(overlay.get(&make_outpoint(3)).is_none());
    }

    #[test]
    fn test_overlay_additions() {
        let base = UtxoSet::new();
        let mut overlay = UtxoOverlay::new(&base);
        
        overlay.insert(make_outpoint(1), make_utxo(1000));
        
        assert_eq!(overlay.get(&make_outpoint(1)).unwrap().value, 1000);
        assert_eq!(overlay.additions_len(), 1);
    }

    #[test]
    fn test_overlay_deletions() {
        let mut base = UtxoSet::new();
        base.insert(make_outpoint(1), make_utxo(1000));
        
        let mut overlay = UtxoOverlay::new(&base);
        
        let removed = overlay.remove(&make_outpoint(1));
        assert_eq!(removed.unwrap().value, 1000);
        assert!(overlay.get(&make_outpoint(1)).is_none());
        assert_eq!(overlay.deletions_len(), 1);
    }

    #[test]
    fn test_overlay_intra_block_spend() {
        // Simulate spending an output created in the same block
        let base = UtxoSet::new();
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
        let mut base = UtxoSet::new();
        base.insert(make_outpoint(1), make_utxo(1000));
        base.insert(make_outpoint(2), make_utxo(2000));
        
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
        let mut base = UtxoSet::new();
        for i in 0..10000 {
            base.insert(make_outpoint(i as u8), make_utxo(i as i64));
        }
        
        // Creating overlay should be instant (no clone)
        let start = std::time::Instant::now();
        let _overlay = UtxoOverlay::new(&base);
        let elapsed = start.elapsed();
        
        // Should be sub-microsecond (just pointer + empty hashmap allocation)
        assert!(elapsed.as_micros() < 100, "Overlay creation took {:?}", elapsed);
    }
}

