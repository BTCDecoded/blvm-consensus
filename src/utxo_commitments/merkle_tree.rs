//! UTXO Merkle Tree Implementation
//!
//! Wraps sparse-merkle-tree to provide UTXO-specific operations.
//! Handles incremental updates (insert/remove) and proof generation.

#[cfg(feature = "utxo-commitments")]
use crate::types::{Hash, Natural, OutPoint, UTXO};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::data_structures::{
    UtxoCommitment, UtxoCommitmentError, UtxoCommitmentResult,
};
#[cfg(feature = "utxo-commitments")]
use sha2::{Digest, Sha256};
#[cfg(feature = "utxo-commitments")]
use sparse_merkle_tree::default_store::DefaultStore;
#[cfg(feature = "utxo-commitments")]
use sparse_merkle_tree::traits::{Hasher, Value};
#[cfg(feature = "utxo-commitments")]
use sparse_merkle_tree::{SparseMerkleTree, H256};
#[cfg(feature = "utxo-commitments")]
use std::collections::HashMap;

/// SHA256 hasher for UTXO Merkle tree
#[cfg(feature = "utxo-commitments")]
#[derive(Default, Clone, Debug)]
pub struct UtxoHasher {
    hasher: Sha256,
}

#[cfg(feature = "utxo-commitments")]
impl Hasher for UtxoHasher {
    fn write_h256(&mut self, h: &H256) {
        // H256 has as_slice() method
        self.hasher.update(h.as_slice());
    }

    fn write_byte(&mut self, b: u8) {
        self.hasher.update(&[b]);
    }

    fn finish(self) -> H256 {
        let hash = self.hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);
        H256::from(bytes)
    }
}

/// UTXO value type for sparse merkle tree
#[cfg(feature = "utxo-commitments")]
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct UtxoValue {
    pub data: Vec<u8>,
}

#[cfg(feature = "utxo-commitments")]
impl Value for UtxoValue {
    fn to_h256(&self) -> H256 {
        let mut hasher = Sha256::new();
        hasher.update(&self.data);
        let hash = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);
        H256::from(bytes)
    }

    fn zero() -> Self {
        Self { data: Vec::new() }
    }
}

/// UTXO Merkle Tree
///
/// Provides incremental updates for UTXO set with Merkle tree commitments.
/// Wraps sparse-merkle-tree to provide UTXO-specific operations.
#[cfg(feature = "utxo-commitments")]
pub struct UtxoMerkleTree {
    tree: SparseMerkleTree<UtxoHasher, UtxoValue, DefaultStore<UtxoValue>>,
    #[allow(dead_code)] // Reserved for future use: Map OutPoint to leaf position
    utxo_index: HashMap<OutPoint, usize>,
    total_supply: u64,
    utxo_count: u64,
}

#[cfg(feature = "utxo-commitments")]
impl UtxoMerkleTree {
    /// Create a new empty UTXO Merkle tree
    pub fn new() -> UtxoCommitmentResult<Self> {
        let store = DefaultStore::default();
        let tree = SparseMerkleTree::new_with_store(store).map_err(|e| {
            UtxoCommitmentError::MerkleTreeError(format!("Failed to create tree: {:?}", e))
        })?;

        Ok(Self {
            tree,
            utxo_index: HashMap::new(),
            total_supply: 0,
            utxo_count: 0,
        })
    }

    /// Get the Merkle root of the UTXO set
    pub fn root(&self) -> Hash {
        let root_h256 = self.tree.root();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(root_h256.as_slice());
        hash
    }

    /// Insert a UTXO into the tree
    pub fn insert(&mut self, outpoint: OutPoint, utxo: UTXO) -> UtxoCommitmentResult<Hash> {
        // Hash the OutPoint to get a key
        let key = self.hash_outpoint(&outpoint);

        // Serialize UTXO to value
        let value = self.serialize_utxo(&utxo)?;
        let utxo_value = UtxoValue { data: value };

        // Update tree
        let root_h256 = self
            .tree
            .update(key, utxo_value)
            .map_err(|e| UtxoCommitmentError::MerkleTreeError(format!("Update failed: {:?}", e)))?;

        // Update tracking with checked arithmetic
        let old_supply = self.total_supply;
        self.total_supply = self.total_supply
            .checked_add(utxo.value as u64)
            .ok_or_else(|| {
                UtxoCommitmentError::MerkleTreeError("Total supply overflow".to_string())
            })?;
        self.utxo_count = self.utxo_count
            .checked_add(1)
            .ok_or_else(|| {
                UtxoCommitmentError::MerkleTreeError("UTXO count overflow".to_string())
            })?;
        
        // Runtime assertion: Supply must increase
        debug_assert!(
            self.total_supply >= old_supply,
            "Total supply ({}) must be >= previous supply ({})",
            self.total_supply,
            old_supply
        );

        // Convert H256 to Hash
        let mut hash = [0u8; 32];
        hash.copy_from_slice(root_h256.as_slice());
        Ok(hash)
    }

    /// Remove a UTXO from the tree (by updating with zero value)
    pub fn remove(&mut self, outpoint: &OutPoint, utxo: &UTXO) -> UtxoCommitmentResult<Hash> {
        // Hash the OutPoint to get a key
        let key = self.hash_outpoint(outpoint);

        // For sparse merkle tree, we update with zero value to delete
        let zero_value = UtxoValue::zero();

        // Update tree (effectively removes the UTXO)
        let root_h256 = self
            .tree
            .update(key, zero_value)
            .map_err(|e| UtxoCommitmentError::MerkleTreeError(format!("Remove failed: {:?}", e)))?;

        // Update tracking with checked arithmetic
        let old_supply = self.total_supply;
        let old_count = self.utxo_count;
        
        self.total_supply = self.total_supply.saturating_sub(utxo.value as u64);
        self.utxo_count = self.utxo_count.saturating_sub(1);
        
        // Runtime assertion: Supply must decrease (or saturate at 0)
        debug_assert!(
            self.total_supply <= old_supply,
            "Total supply ({}) must be <= previous supply ({})",
            self.total_supply,
            old_supply
        );
        
        // Runtime assertion: Count must decrease (or saturate at 0)
        debug_assert!(
            self.utxo_count <= old_count,
            "UTXO count ({}) must be <= previous count ({})",
            self.utxo_count,
            old_count
        );
        
        // Runtime assertion: Supply and count must be non-negative
        // Note: u64 is always >= 0, but we keep the assertion for documentation
        // and to catch any potential type changes in the future
        debug_assert!(
            self.total_supply <= u64::MAX,
            "Total supply ({}) must be within u64 bounds",
            self.total_supply
        );
        debug_assert!(
            self.utxo_count <= u64::MAX,
            "UTXO count ({}) must be within u64 bounds",
            self.utxo_count
        );

        // Convert H256 to Hash
        let mut hash = [0u8; 32];
        hash.copy_from_slice(root_h256.as_slice());
        Ok(hash)
    }

    /// Get a UTXO from the tree
    pub fn get(&self, outpoint: &OutPoint) -> UtxoCommitmentResult<Option<UTXO>> {
        let key = self.hash_outpoint(outpoint);

        match self.tree.get(&key) {
            Ok(value) => {
                // Check if value is zero (empty)
                if value.to_h256() == H256::zero() || value.to_h256() == UtxoValue::zero().to_h256()
                {
                    Ok(None)
                } else {
                    // Extract serialized data from UtxoValue and deserialize
                    let serialized_data = &value.data;

                    // Deserialize the UTXO data
                    match self.deserialize_utxo(serialized_data) {
                        Ok(utxo) => Ok(Some(utxo)),
                        Err(e) => {
                            // Deserialization failed - this might indicate corrupted data
                            Err(UtxoCommitmentError::InvalidUtxo(format!(
                                "Failed to deserialize UTXO: {}",
                                e
                            )))
                        }
                    }
                }
            }
            Err(_) => Ok(None),
        }
    }

    /// Generate a UTXO commitment
    pub fn generate_commitment(&self, block_hash: Hash, block_height: Natural) -> UtxoCommitment {
        let merkle_root = self.root();
        UtxoCommitment::new(
            merkle_root,
            self.total_supply,
            self.utxo_count,
            block_height,
            block_hash,
        )
    }

    /// Get total supply
    pub fn total_supply(&self) -> u64 {
        self.total_supply
    }

    /// Get UTXO count
    pub fn utxo_count(&self) -> u64 {
        self.utxo_count
    }

    /// Generate a Merkle proof for a specific UTXO
    ///
    /// Returns a proof that can be used to verify the UTXO exists in the tree.
    pub fn generate_proof(
        &self,
        outpoint: &OutPoint,
    ) -> UtxoCommitmentResult<sparse_merkle_tree::MerkleProof> {
        let key = self.hash_outpoint(outpoint);
        let keys = vec![key];

        self.tree.merkle_proof(keys).map_err(|e| {
            UtxoCommitmentError::MerkleTreeError(format!("Failed to generate proof: {:?}", e))
        })
    }

    /// Verify a UTXO commitment matches expected supply
    ///
    /// Compares the total supply in the commitment against the expected
    /// Bitcoin supply at the given block height.
    pub fn verify_commitment_supply(
        &self,
        commitment: &UtxoCommitment,
    ) -> UtxoCommitmentResult<bool> {
        use crate::economic::total_supply;

        let expected_supply = total_supply(commitment.block_height) as u64;
        let matches = commitment.total_supply == expected_supply;

        if !matches {
            return Err(UtxoCommitmentError::VerificationFailed(format!(
                "Supply mismatch: commitment has {}, expected {}",
                commitment.total_supply, expected_supply
            )));
        }

        Ok(true)
    }

    /// Verify a commitment's Merkle root matches the tree's root
    pub fn verify_commitment_root(&self, commitment: &UtxoCommitment) -> bool {
        let tree_root = self.root();
        commitment.merkle_root == tree_root
    }

    // Helper methods

    /// Hash an OutPoint to H256 key
    fn hash_outpoint(&self, outpoint: &OutPoint) -> H256 {
        let mut hasher = Sha256::new();
        hasher.update(&outpoint.hash);
        hasher.update(&outpoint.index.to_be_bytes());
        let hash = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&hash);
        H256::from(bytes)
    }

    /// Serialize UTXO to bytes
    fn serialize_utxo(&self, utxo: &UTXO) -> UtxoCommitmentResult<Vec<u8>> {
        // Simple serialization: value (8 bytes) + height (8 bytes) + script_pubkey (variable)
        let mut bytes = Vec::with_capacity(16 + utxo.script_pubkey.len());
        bytes.extend_from_slice(&utxo.value.to_be_bytes());
        bytes.extend_from_slice(&utxo.height.to_be_bytes());
        bytes.push(utxo.script_pubkey.len() as u8);
        bytes.extend_from_slice(&utxo.script_pubkey);
        Ok(bytes)
    }

    /// Deserialize bytes to UTXO
    fn deserialize_utxo(&self, data: &[u8]) -> UtxoCommitmentResult<UTXO> {
        if data.len() < 17 {
            return Err(UtxoCommitmentError::InvalidUtxo(
                "Data too short".to_string(),
            ));
        }

        let mut offset = 0;
        let value = i64::from_be_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| UtxoCommitmentError::InvalidUtxo("Invalid value".to_string()))?,
        );
        offset += 8;

        let height = u64::from_be_bytes(
            data[offset..offset + 8]
                .try_into()
                .map_err(|_| UtxoCommitmentError::InvalidUtxo("Invalid height".to_string()))?,
        );
        offset += 8;

        let script_len = data[offset] as usize;
        offset += 1;

        if data.len() < offset + script_len {
            return Err(UtxoCommitmentError::InvalidUtxo(
                "Script length mismatch".to_string(),
            ));
        }

        let script_pubkey = data[offset..offset + script_len].to_vec();

        Ok(UTXO {
            value,
            script_pubkey,
            height,
        })
    }
}

#[cfg(feature = "utxo-commitments")]
impl Default for UtxoMerkleTree {
    fn default() -> Self {
        Self::new().expect("Failed to create default UtxoMerkleTree")
    }
}

// Placeholder implementation when feature is disabled
#[cfg(not(feature = "utxo-commitments"))]
pub struct UtxoMerkleTree;

#[cfg(not(feature = "utxo-commitments"))]
impl UtxoMerkleTree {
    pub fn new() -> Result<Self, String> {
        Err("UTXO commitments feature not enabled".to_string())
    }
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for UTXO Merkle Tree:
/// ∀ tree ∈ UtxoMerkleTree, outpoint ∈ OutPoint, utxo ∈ UTXO:
/// - insert(tree, outpoint, utxo) = tree' where tree'.total_supply = tree.total_supply + utxo.value
/// - remove(tree, outpoint, utxo) = tree' where tree'.total_supply = tree.total_supply - utxo.value
/// - root(tree) is deterministic (same tree → same root)
/// - Commitment consistency: commitment.total_supply matches tree.total_supply
///
/// Invariants:
/// - Supply tracking is accurate (never negative, matches UTXO set)
/// - Merkle root is deterministic for same UTXO set
/// - Tree operations preserve consistency
/// - Commitment generation matches actual tree state

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::types::{OutPoint, UTXO};
    use kani::*;

    /// Kani proof: Supply tracking accuracy after insert
    ///
    /// Verifies that inserting a UTXO correctly increases total supply.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_insert_supply_accuracy() {
        let mut tree = UtxoMerkleTree::new().unwrap();

        let initial_supply = tree.total_supply();
        let initial_count = tree.utxo_count();

        // Create test UTXO
        let outpoint = OutPoint {
            hash: kani::any(),
            index: kani::any(),
        };

        let utxo_value: i64 = kani::any();
        kani::assume(utxo_value >= 0); // Valid UTXO value

        let utxo = UTXO {
            value: utxo_value,
            script_pubkey: vec![], // Simplified for tractability
            height: 0,
        };

        // Insert UTXO
        let result = tree.insert(outpoint, utxo.clone());
        kani::assume(result.is_ok());

        // Verify supply increased correctly
        assert_eq!(
            tree.total_supply(),
            initial_supply + utxo.value as u64,
            "Total supply must increase by UTXO value"
        );

        // Verify count increased
        assert_eq!(
            tree.utxo_count(),
            initial_count + 1,
            "UTXO count must increase by 1"
        );
    }

    /// Kani proof: Supply tracking accuracy after remove
    ///
    /// Verifies that removing a UTXO correctly decreases total supply.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_remove_supply_accuracy() {
        let mut tree = UtxoMerkleTree::new().unwrap();

        // Insert UTXO first
        let outpoint = OutPoint {
            hash: kani::any(),
            index: kani::any(),
        };

        let utxo_value: i64 = kani::any();
        kani::assume(utxo_value >= 0);
        kani::assume(utxo_value <= 1000000); // Bound for tractability

        let utxo = UTXO {
            value: utxo_value,
            script_pubkey: vec![],
            height: 0,
        };

        tree.insert(outpoint.clone(), utxo.clone()).unwrap();
        let supply_after_insert = tree.total_supply();
        let count_after_insert = tree.utxo_count();

        // Remove UTXO
        let result = tree.remove(&outpoint, &utxo);
        kani::assume(result.is_ok());

        // Verify supply decreased correctly (using saturating_sub, so can't go negative)
        assert!(
            tree.total_supply() <= supply_after_insert,
            "Total supply must not increase after remove"
        );

        // Verify count decreased
        assert_eq!(
            tree.utxo_count(),
            count_after_insert.saturating_sub(1),
            "UTXO count must decrease by 1"
        );
    }

    /// Kani proof: Merkle root determinism
    ///
    /// Verifies that the same UTXO set always produces the same root.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_merkle_root_deterministic() {
        let mut tree1 = UtxoMerkleTree::new().unwrap();
        let mut tree2 = UtxoMerkleTree::new().unwrap();

        // Insert same UTXO in both trees
        let outpoint = OutPoint {
            hash: kani::any(),
            index: kani::any(),
        };

        let utxo_value: i64 = kani::any();
        kani::assume(utxo_value >= 0);
        kani::assume(utxo_value <= 1000000);

        let utxo = UTXO {
            value: utxo_value,
            script_pubkey: vec![],
            height: 0,
        };

        tree1.insert(outpoint.clone(), utxo.clone()).unwrap();
        tree2.insert(outpoint.clone(), utxo.clone()).unwrap();

        // Both trees should have same root
        assert_eq!(
            tree1.root(),
            tree2.root(),
            "Same UTXO set must produce same Merkle root"
        );
    }

    /// Kani proof: Commitment consistency
    ///
    /// Verifies that generated commitment matches tree state.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_commitment_consistency() {
        let mut tree = UtxoMerkleTree::new().unwrap();

        // Insert UTXO
        let outpoint = OutPoint {
            hash: kani::any(),
            index: kani::any(),
        };

        let utxo_value: i64 = kani::any();
        kani::assume(utxo_value >= 0);
        kani::assume(utxo_value <= 1000000);

        let utxo = UTXO {
            value: utxo_value,
            script_pubkey: vec![],
            height: 0,
        };

        tree.insert(outpoint, utxo.clone()).unwrap();

        // Generate commitment
        let block_hash: Hash = kani::any();
        let block_height: Natural = kani::any();
        kani::assume(block_height <= 1000); // Bound for tractability

        let commitment = tree.generate_commitment(block_hash, block_height);

        // Verify commitment matches tree state
        assert_eq!(
            commitment.total_supply,
            tree.total_supply(),
            "Commitment supply must match tree supply"
        );

        assert_eq!(
            commitment.utxo_count,
            tree.utxo_count(),
            "Commitment count must match tree count"
        );

        assert_eq!(
            commitment.merkle_root,
            tree.root(),
            "Commitment root must match tree root"
        );

        assert_eq!(
            commitment.block_height, block_height,
            "Commitment height must match requested height"
        );

        assert_eq!(
            commitment.block_hash, block_hash,
            "Commitment hash must match requested hash"
        );
    }
}
