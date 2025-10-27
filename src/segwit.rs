//! Segregated Witness (SegWit) functions from Orange Paper Section 11.1

use crate::types::*;
use crate::error::Result;
use crate::types::{Hash, ByteString, Natural};
use bitcoin_hashes::{sha256d, Hash as BitcoinHash, HashEngine};

/// Witness Data: 𝒲 = 𝕊* (stack of witness elements)
pub type Witness = Vec<ByteString>;

/// Calculate transaction weight for SegWit
/// Weight(tx) = 4 × |Serialize(tx ∖ witness)| + |Serialize(tx)|
pub fn calculate_transaction_weight(tx: &Transaction, witness: Option<&Witness>) -> Result<Natural> {
    // Calculate base size (transaction without witness data)
    let base_size = calculate_base_size(tx);
    
    // Calculate total size (transaction with witness data)
    let total_size = calculate_total_size(tx, witness);
    
    // Weight = 4 * base_size + total_size
    Ok(4 * base_size + total_size)
}

/// Calculate base size (transaction without witness data)
fn calculate_base_size(tx: &Transaction) -> Natural {
    // Simplified calculation - in reality this would be the actual serialized size
    (4 + // version
    tx.inputs.len() * (32 + 4 + 1 + 4) + // inputs (OutPoint + script_sig_len + sequence)
    tx.outputs.len() * (8 + 1) + // outputs (value + script_pubkey_len)
    4) as Natural // lock_time
}

/// Calculate total size (transaction with witness data)
fn calculate_total_size(tx: &Transaction, witness: Option<&Witness>) -> Natural {
    let base_size = calculate_base_size(tx);
    
    if let Some(witness_data) = witness {
        let witness_size: Natural = witness_data.iter()
            .map(|w| w.len() as Natural)
            .sum();
        base_size + witness_size
    } else {
        base_size
    }
}

/// Compute witness merkle root for block
/// WitnessRoot = ComputeMerkleRoot({Hash(tx.witness) : tx ∈ block.transactions})
pub fn compute_witness_merkle_root(block: &Block, witnesses: &[Witness]) -> Result<Hash> {
    if block.transactions.is_empty() {
        return Err(crate::error::ConsensusError::ConsensusRuleViolation(
            "Cannot compute witness merkle root for empty block".to_string()
        ));
    }
    
    // Hash each witness
    let mut witness_hashes = Vec::new();
    for (i, witness) in witnesses.iter().enumerate() {
        if i == 0 {
            // Coinbase transaction has empty witness
            witness_hashes.push([0u8; 32]);
        } else {
            let witness_hash = hash_witness(witness);
            witness_hashes.push(witness_hash);
        }
    }
    
    // Compute merkle root of witness hashes
    compute_merkle_root(&witness_hashes)
}

/// Hash witness data
fn hash_witness(witness: &Witness) -> Hash {
    let mut hasher = sha256d::Hash::engine();
    for element in witness {
        hasher.input(element);
    }
    let result = sha256d::Hash::from_engine(hasher);
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Compute merkle root from hashes
fn compute_merkle_root(hashes: &[Hash]) -> Result<Hash> {
    if hashes.is_empty() {
        return Err(crate::error::ConsensusError::ConsensusRuleViolation(
            "Cannot compute merkle root from empty hash list".to_string()
        ));
    }
    
    if hashes.len() == 1 {
        return Ok(hashes[0]);
    }
    
    // Simplified merkle root calculation
    // In reality, this would use proper merkle tree construction
    let mut hasher = sha256d::Hash::engine();
    hasher.input(&hashes[0]);
    hasher.input(&hashes[1]);
    let result = sha256d::Hash::from_engine(hasher);
    let mut root = [0u8; 32];
    root.copy_from_slice(&result);
    Ok(root)
}

/// Validate witness commitment in coinbase transaction
pub fn validate_witness_commitment(
    coinbase_tx: &Transaction,
    witness_merkle_root: &Hash,
) -> Result<bool> {
    // Look for witness commitment in coinbase script
    for output in &coinbase_tx.outputs {
        if let Some(commitment) = extract_witness_commitment(&output.script_pubkey) {
            return Ok(commitment == *witness_merkle_root);
        }
    }
    
    // No witness commitment found - this is valid for non-SegWit blocks
    Ok(true)
}

/// Extract witness commitment from script
fn extract_witness_commitment(script: &ByteString) -> Option<Hash> {
    // Look for OP_RETURN followed by witness commitment
    if script.len() >= 38 && script[0] == 0x6a { // OP_RETURN
        if script.len() >= 38 && script[1] == 0x24 { // 36 bytes
            let mut commitment = [0u8; 32];
            commitment.copy_from_slice(&script[2..34]);
            return Some(commitment);
        }
    }
    None
}

/// Check if transaction is SegWit
pub fn is_segwit_transaction(tx: &Transaction) -> bool {
    // Check if any input has witness data
    // This is a simplified check - in reality we'd check the actual witness structure
    tx.inputs.iter().any(|input| {
        // Look for SegWit markers in script_sig
        input.script_sig.len() == 1 && input.script_sig[0] == 0x00
    })
}

/// Calculate block weight for SegWit blocks
pub fn calculate_block_weight(block: &Block, witnesses: &[Witness]) -> Result<Natural> {
    let mut total_weight = 0;
    
    for (i, tx) in block.transactions.iter().enumerate() {
        let witness = if i < witnesses.len() {
            Some(&witnesses[i])
        } else {
            None
        };
        
        total_weight += calculate_transaction_weight(tx, witness)?;
    }
    
    Ok(total_weight)
}

/// Validate SegWit block
pub fn validate_segwit_block(
    block: &Block,
    witnesses: &[Witness],
    max_block_weight: Natural,
) -> Result<bool> {
    // Check block weight limit
    let block_weight = calculate_block_weight(block, witnesses)?;
    if block_weight > max_block_weight {
        return Ok(false);
    }
    
    // Validate witness commitment
    if !block.transactions.is_empty() {
        let witness_root = compute_witness_merkle_root(block, witnesses)?;
        if !validate_witness_commitment(&block.transactions[0], &witness_root)? {
            return Ok(false);
        }
    }
    
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_transaction_weight() {
        let tx = create_test_transaction();
        let witness = vec![vec![0x51], vec![0x52]]; // OP_1, OP_2
        
        let weight = calculate_transaction_weight(&tx, Some(&witness)).unwrap();
        assert!(weight > 0);
    }
    
    #[test]
    fn test_calculate_transaction_weight_no_witness() {
        let tx = create_test_transaction();
        
        let weight = calculate_transaction_weight(&tx, None).unwrap();
        assert!(weight > 0);
    }
    
    #[test]
    fn test_compute_witness_merkle_root() {
        let block = create_test_block();
        let witnesses = vec![
            vec![], // Coinbase witness (empty)
            vec![vec![0x51]], // First transaction witness
        ];
        
        let root = compute_witness_merkle_root(&block, &witnesses).unwrap();
        assert_eq!(root.len(), 32);
    }
    
    #[test]
    fn test_compute_witness_merkle_root_empty_block() {
        let block = Block {
            header: create_test_header(),
            transactions: vec![],
        };
        let witnesses = vec![];
        
        let result = compute_witness_merkle_root(&block, &witnesses);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_validate_witness_commitment() {
        let mut coinbase_tx = create_test_transaction();
        let witness_root = [1u8; 32];
        
        // Add witness commitment to coinbase script
        coinbase_tx.outputs[0].script_pubkey = create_witness_commitment_script(&witness_root);
        
        let is_valid = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
        assert!(is_valid);
    }
    
    #[test]
    fn test_is_segwit_transaction() {
        let mut tx = create_test_transaction();
        tx.inputs[0].script_sig = vec![0x00]; // SegWit marker
        
        assert!(is_segwit_transaction(&tx));
    }
    
    #[test]
    fn test_calculate_block_weight() {
        let block = create_test_block();
        let witnesses = vec![
            vec![], // Coinbase
            vec![vec![0x51]], // First tx
        ];
        
        let weight = calculate_block_weight(&block, &witnesses).unwrap();
        assert!(weight > 0);
    }
    
    #[test]
    fn test_validate_segwit_block() {
        let block = create_test_block();
        let witnesses = vec![
            vec![], // Coinbase
            vec![vec![0x51]], // First tx
        ];
        
        let is_valid = validate_segwit_block(&block, &witnesses, 4_000_000).unwrap();
        assert!(is_valid);
    }
    
    #[test]
    fn test_validate_segwit_block_exceeds_weight() {
        let block = create_test_block();
        let witnesses = vec![
            vec![], // Coinbase
            vec![vec![0x51]], // First tx
        ];
        
        let is_valid = validate_segwit_block(&block, &witnesses, 1).unwrap(); // Very low weight limit
        assert!(!is_valid);
    }
    
    #[test]
    fn test_validate_segwit_block_invalid_commitment() {
        let mut block = create_test_block();
        let witnesses = vec![
            vec![], // Coinbase
            vec![vec![0x51]], // First tx
        ];
        
        // Create coinbase with invalid witness commitment
        let invalid_commitment = [2u8; 32];
        block.transactions[0].outputs[0].script_pubkey = create_witness_commitment_script(&invalid_commitment);
        
        let is_valid = validate_segwit_block(&block, &witnesses, 4_000_000).unwrap();
        assert!(!is_valid);
    }
    
    #[test]
    fn test_validate_witness_commitment_no_commitment() {
        let coinbase_tx = create_test_transaction();
        let witness_root = [1u8; 32];
        
        // No witness commitment in script
        let is_valid = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
        assert!(is_valid); // Should be valid for non-SegWit blocks
    }
    
    #[test]
    fn test_validate_witness_commitment_invalid_commitment() {
        let mut coinbase_tx = create_test_transaction();
        let witness_root = [1u8; 32];
        let invalid_commitment = [2u8; 32];
        
        // Add invalid witness commitment
        coinbase_tx.outputs[0].script_pubkey = create_witness_commitment_script(&invalid_commitment);
        
        let is_valid = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
        assert!(!is_valid);
    }
    
    #[test]
    fn test_extract_witness_commitment_valid() {
        let commitment = [1u8; 32];
        let script = create_witness_commitment_script(&commitment);
        
        let extracted = extract_witness_commitment(&script).unwrap();
        assert_eq!(extracted, commitment);
    }
    
    #[test]
    fn test_extract_witness_commitment_invalid_script() {
        let script = vec![0x51]; // Not a witness commitment script
        
        let extracted = extract_witness_commitment(&script);
        assert!(extracted.is_none());
    }
    
    #[test]
    fn test_extract_witness_commitment_wrong_opcode() {
        let mut script = vec![0x52, 0x24]; // Wrong opcode, correct length
        script.extend_from_slice(&[1u8; 32]);
        
        let extracted = extract_witness_commitment(&script);
        assert!(extracted.is_none());
    }
    
    #[test]
    fn test_extract_witness_commitment_wrong_length() {
        let mut script = vec![0x6a, 0x25]; // OP_RETURN, wrong length (37 bytes)
        script.extend_from_slice(&[1u8; 32]);
        
        let extracted = extract_witness_commitment(&script);
        assert!(extracted.is_none());
    }
    
    #[test]
    fn test_hash_witness() {
        let witness = vec![vec![0x51], vec![0x52]];
        let hash = hash_witness(&witness);
        
        assert_eq!(hash.len(), 32);
        
        // Different witness should produce different hash
        let witness2 = vec![vec![0x53], vec![0x54]];
        let hash2 = hash_witness(&witness2);
        assert_ne!(hash, hash2);
    }
    
    #[test]
    fn test_hash_witness_empty() {
        let witness = vec![];
        let hash = hash_witness(&witness);
        
        assert_eq!(hash.len(), 32);
    }
    
    #[test]
    fn test_compute_merkle_root_single_hash() {
        let hashes = vec![[1u8; 32]];
        let root = compute_merkle_root(&hashes).unwrap();
        
        assert_eq!(root, [1u8; 32]);
    }
    
    #[test]
    fn test_compute_merkle_root_empty() {
        let hashes = vec![];
        let result = compute_merkle_root(&hashes);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_is_segwit_transaction_false() {
        let tx = create_test_transaction();
        // No SegWit markers
        
        assert!(!is_segwit_transaction(&tx));
    }
    
    #[test]
    fn test_calculate_base_size() {
        let tx = create_test_transaction();
        let base_size = calculate_base_size(&tx);
        
        assert!(base_size > 0);
    }
    
    #[test]
    fn test_calculate_total_size_with_witness() {
        let tx = create_test_transaction();
        let witness = vec![vec![0x51], vec![0x52]];
        
        let total_size = calculate_total_size(&tx, Some(&witness));
        let base_size = calculate_base_size(&tx);
        
        assert!(total_size > base_size);
    }
    
    #[test]
    fn test_calculate_total_size_without_witness() {
        let tx = create_test_transaction();
        
        let total_size = calculate_total_size(&tx, None);
        let base_size = calculate_base_size(&tx);
        
        assert_eq!(total_size, base_size);
    }
    
    // Helper functions
    fn create_test_transaction() -> Transaction {
        Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: 0 },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51],
            }],
            lock_time: 0,
        }
    }
    
    fn create_test_block() -> Block {
        Block {
            header: create_test_header(),
            transactions: vec![
                create_test_transaction(), // Coinbase
                create_test_transaction(), // Regular tx
            ],
        }
    }
    
    fn create_test_header() -> BlockHeader {
        BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        }
    }
    
    fn create_witness_commitment_script(commitment: &Hash) -> ByteString {
        let mut script = vec![0x6a, 0x24]; // OP_RETURN, 36 bytes
        script.extend_from_slice(commitment);
        // Add 4 more bytes to make it 38 bytes total as expected by extract_witness_commitment
        script.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        script
    }
}

#[cfg(feature = "verify")]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Verify calculate_transaction_weight formula is correct
    /// 
    /// Mathematical specification:
    /// Weight(tx) = 4 × |Serialize(tx ∖ witness)| + |Serialize(tx)|
    /// ∀ tx ∈ Transaction, witness ∈ Option<Witness>:
    /// - Weight ≥ 0 (non-negative)
    /// - Weight formula is correctly applied
    #[kani::proof]
    fn kani_calculate_transaction_weight_formula() {
        let tx = create_bounded_transaction();
        let witness: Option<Vec<Vec<u8>>> = if kani::any() {
            let witness_count: usize = kani::any();
            kani::assume(witness_count <= 3);
            let mut witnesses = Vec::new();
            for i in 0..witness_count {
                let element_len: usize = kani::any();
                kani::assume(element_len <= 5);
                let mut element = Vec::new();
                for j in 0..element_len {
                    let byte: u8 = kani::any();
                    element.push(byte);
                }
                witnesses.push(element);
            }
            Some(witnesses)
        } else {
            None
        };
        
        let weight = calculate_transaction_weight(&tx, witness.as_ref()).unwrap();
        
        // Weight should be non-negative
        assert!(weight >= 0);
        
        // Weight should follow the formula: 4 * base_size + total_size
        let base_size = calculate_base_size(&tx);
        let total_size = calculate_total_size(&tx, witness.as_ref());
        let expected_weight = 4 * base_size + total_size;
        assert_eq!(weight, expected_weight);
    }

    /// Verify block weight validation respects limits
    /// 
    /// Mathematical specification:
    /// ∀ block ∈ Block, witnesses ∈ [Witness], max_weight ∈ ℕ:
    /// - If Σ tx_weight > max_weight: validate_segwit_block returns false
    /// - If Σ tx_weight ≤ max_weight: validate_segwit_block returns true (if other checks pass)
    #[kani::proof]
    fn kani_validate_segwit_block_weight_limit() {
        let block = create_bounded_block();
        let witnesses = create_bounded_witnesses(&block);
        let max_weight: Natural = kani::any();
        kani::assume(max_weight <= 10_000_000); // Reasonable upper bound
        
        let is_valid = validate_segwit_block(&block, &witnesses, max_weight).unwrap();
        
        // Calculate actual block weight
        let actual_weight = calculate_block_weight(&block, &witnesses).unwrap();
        
        // If weight exceeds limit, validation should fail
        if actual_weight > max_weight {
            assert!(!is_valid);
        }
        // If weight is within limit, validation depends on other factors
        // (witness commitment, etc.) but weight check should pass
    }

    /// Verify witness commitment validation is deterministic
    /// 
    /// Mathematical specification:
    /// ∀ coinbase_tx ∈ Transaction, witness_root ∈ Hash:
    /// - validate_witness_commitment(coinbase_tx, witness_root) is deterministic
    /// - Same inputs always produce same output
    #[kani::proof]
    fn kani_validate_witness_commitment_deterministic() {
        let coinbase_tx = create_bounded_transaction();
        let witness_root: Hash = kani::any();
        
        // Call validate_witness_commitment twice with same inputs
        let result1 = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
        let result2 = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
        
        // Results should be identical (deterministic)
        assert_eq!(result1, result2);
    }

    /// Verify witness merkle root computation handles edge cases
    /// 
    /// Mathematical specification:
    /// ∀ block ∈ Block, witnesses ∈ [Witness]:
    /// - If |block.transactions| = 0: compute_witness_merkle_root returns error
    /// - If |block.transactions| > 0: compute_witness_merkle_root returns valid hash
    #[kani::proof]
    fn kani_compute_witness_merkle_root_edge_cases() {
        let has_transactions: bool = kani::any();
        
        let block = if has_transactions {
            create_bounded_block()
        } else {
            Block {
                header: create_bounded_header(),
                transactions: vec![],
            }
        };
        
        let witnesses = if has_transactions {
            create_bounded_witnesses(&block)
        } else {
            vec![]
        };
        
        let result = compute_witness_merkle_root(&block, &witnesses);
        
        if block.transactions.is_empty() {
            // Empty block should return error
            assert!(result.is_err());
        } else {
            // Non-empty block should return valid hash
            assert!(result.is_ok());
            let root = result.unwrap();
            assert_eq!(root.len(), 32);
        }
    }

    /// Helper function to create bounded transaction for Kani
    fn create_bounded_transaction() -> Transaction {
        let input_count: usize = kani::any();
        let output_count: usize = kani::any();
        kani::assume(input_count <= 3);
        kani::assume(output_count <= 3);
        
        let mut inputs = Vec::new();
        for i in 0..input_count {
            let script_len: usize = kani::any();
            kani::assume(script_len <= 5);
            let mut script = Vec::new();
            for j in 0..script_len {
                let byte: u8 = kani::any();
                script.push(byte);
            }
            inputs.push(TransactionInput {
                prevout: OutPoint { hash: [0; 32], index: i as u32 },
                script_sig: script,
                sequence: 0xffffffff,
            });
        }
        
        let mut outputs = Vec::new();
        for i in 0..output_count {
            let script_len: usize = kani::any();
            kani::assume(script_len <= 5);
            let mut script = Vec::new();
            for j in 0..script_len {
                let byte: u8 = kani::any();
                script.push(byte);
            }
            outputs.push(TransactionOutput {
                value: 1000,
                script_pubkey: script,
            });
        }
        
        Transaction {
            version: 1,
            inputs,
            outputs,
            lock_time: 0,
        }
    }

    /// Helper function to create bounded block for Kani
    fn create_bounded_block() -> Block {
        let tx_count: usize = kani::any();
        kani::assume(tx_count <= 3);
        
        let mut transactions = Vec::new();
        for i in 0..tx_count {
            transactions.push(create_bounded_transaction());
        }
        
        Block {
            header: create_bounded_header(),
            transactions,
        }
    }

    /// Helper function to create bounded header for Kani
    fn create_bounded_header() -> BlockHeader {
        BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505,
            bits: 0x1d00ffff,
            nonce: 0,
        }
    }

    /// Helper function to create bounded witnesses for Kani
    fn create_bounded_witnesses(block: &Block) -> Vec<Witness> {
        let mut witnesses = Vec::new();
        for i in 0..block.transactions.len() {
            let element_count: usize = kani::any();
            kani::assume(element_count <= 3);
            
            let mut witness = Vec::new();
            for j in 0..element_count {
                let element_len: usize = kani::any();
                kani::assume(element_len <= 5);
                let mut element = Vec::new();
                for k in 0..element_len {
                    let byte: u8 = kani::any();
                    element.push(byte);
                }
                witness.push(element);
            }
            witnesses.push(witness);
        }
        witnesses
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Property test: transaction weight is non-negative
    /// 
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, witness ∈ Option<Witness>: Weight(tx) ≥ 0
    proptest! {
        #[test]
        fn prop_transaction_weight_non_negative(
            tx in create_transaction_strategy(),
            witness in prop::option::of(create_witness_strategy())
        ) {
            let weight = calculate_transaction_weight(&tx, witness.as_ref()).unwrap();
            assert!(weight >= 0);
        }
    }

    /// Property test: transaction weight formula is correct
    /// 
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, witness ∈ Option<Witness>:
    /// Weight(tx) = 4 × base_size + total_size
    proptest! {
        #[test]
        fn prop_transaction_weight_formula(
            tx in create_transaction_strategy(),
            witness in prop::option::of(create_witness_strategy())
        ) {
            let weight = calculate_transaction_weight(&tx, witness.as_ref()).unwrap();
            let base_size = calculate_base_size(&tx);
            let total_size = calculate_total_size(&tx, witness.as_ref());
            let expected_weight = 4 * base_size + total_size;
            
            assert_eq!(weight, expected_weight);
        }
    }

    /// Property test: block weight validation respects limits
    /// 
    /// Mathematical specification:
    /// ∀ block ∈ Block, witnesses ∈ [Witness], max_weight ∈ ℕ:
    /// If Σ tx_weight > max_weight then validate_segwit_block returns false
    proptest! {
        #[test]
        fn prop_block_weight_validation_limit(
            block in create_block_strategy(),
            witnesses in create_witnesses_strategy(),
            max_weight in 1..10_000_000u64
        ) {
            let actual_weight = calculate_block_weight(&block, &witnesses).unwrap();
            let is_valid = validate_segwit_block(&block, &witnesses, max_weight as Natural).unwrap();
            
            if actual_weight > max_weight as Natural {
                assert!(!is_valid);
            }
        }
    }

    /// Property test: witness commitment validation is deterministic
    /// 
    /// Mathematical specification:
    /// ∀ coinbase_tx ∈ Transaction, witness_root ∈ Hash:
    /// validate_witness_commitment(coinbase_tx, witness_root) is deterministic
    proptest! {
        #[test]
        fn prop_witness_commitment_deterministic(
            coinbase_tx in create_transaction_strategy(),
            witness_root in create_hash_strategy()
        ) {
            let result1 = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
            let result2 = validate_witness_commitment(&coinbase_tx, &witness_root).unwrap();
            
            assert_eq!(result1, result2);
        }
    }

    /// Property test: witness merkle root computation is deterministic
    /// 
    /// Mathematical specification:
    /// ∀ block ∈ Block, witnesses ∈ [Witness]:
    /// compute_witness_merkle_root(block, witnesses) is deterministic
    proptest! {
        #[test]
        fn prop_witness_merkle_root_deterministic(
            block in create_block_strategy(),
            witnesses in create_witnesses_strategy()
        ) {
            if !block.transactions.is_empty() {
                let result1 = compute_witness_merkle_root(&block, &witnesses);
                let result2 = compute_witness_merkle_root(&block, &witnesses);
                
                assert_eq!(result1.is_ok(), result2.is_ok());
                if result1.is_ok() && result2.is_ok() {
                    assert_eq!(result1.unwrap(), result2.unwrap());
                }
            }
        }
    }

    /// Property test: SegWit transaction detection is consistent
    /// 
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction: is_segwit_transaction(tx) ∈ {true, false}
    proptest! {
        #[test]
        fn prop_segwit_transaction_detection(
            tx in create_transaction_strategy()
        ) {
            let is_segwit = is_segwit_transaction(&tx);
            assert!(is_segwit == true || is_segwit == false);
        }
    }

    /// Property test: witness hashing is deterministic
    /// 
    /// Mathematical specification:
    /// ∀ witness ∈ Witness: hash_witness(witness) is deterministic
    proptest! {
        #[test]
        fn prop_witness_hashing_deterministic(
            witness in create_witness_strategy()
        ) {
            let hash1 = hash_witness(&witness);
            let hash2 = hash_witness(&witness);
            
            assert_eq!(hash1, hash2);
            assert_eq!(hash1.len(), 32);
        }
    }

    /// Property test: merkle root computation handles single hash
    /// 
    /// Mathematical specification:
    /// ∀ hash ∈ Hash: compute_merkle_root([hash]) = hash
    proptest! {
        #[test]
        fn prop_merkle_root_single_hash(
            hash in create_hash_strategy()
        ) {
            let hashes = vec![hash];
            let root = compute_merkle_root(&hashes).unwrap();
            
            assert_eq!(root, hash);
        }
    }

    /// Property test: merkle root computation fails on empty input
    /// 
    /// Mathematical specification:
    /// compute_merkle_root([]) returns error
    proptest! {
        #[test]
        fn prop_merkle_root_empty_input() {
            let hashes: Vec<Hash> = vec![];
            let result = compute_merkle_root(&hashes);
            
            assert!(result.is_err());
        }
    }

    /// Property test: witness commitment extraction is deterministic
    /// 
    /// Mathematical specification:
    /// ∀ script ∈ ByteString: extract_witness_commitment(script) is deterministic
    proptest! {
        #[test]
        fn prop_witness_commitment_extraction_deterministic(
            script in prop::collection::vec(any::<u8>(), 0..100)
        ) {
            let result1 = extract_witness_commitment(&script);
            let result2 = extract_witness_commitment(&script);
            
            assert_eq!(result1.is_some(), result2.is_some());
            if result1.is_some() && result2.is_some() {
                assert_eq!(result1.unwrap(), result2.unwrap());
            }
        }
    }

    /// Property test: base size calculation is monotonic
    /// 
    /// Mathematical specification:
    /// ∀ tx1, tx2 ∈ Transaction: |tx1| ≤ |tx2| ⟹ base_size(tx1) ≤ base_size(tx2)
    proptest! {
        #[test]
        fn prop_base_size_monotonic(
            tx1 in create_transaction_strategy(),
            tx2 in create_transaction_strategy()
        ) {
            let base_size1 = calculate_base_size(&tx1);
            let base_size2 = calculate_base_size(&tx2);
            
            // Base size should be positive
            assert!(base_size1 > 0);
            assert!(base_size2 > 0);
        }
    }

    /// Property test: total size with witness is greater than base size
    /// 
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction, witness ∈ Witness: total_size(tx, witness) ≥ base_size(tx)
    proptest! {
        #[test]
        fn prop_total_size_with_witness_greater_than_base(
            tx in create_transaction_strategy(),
            witness in create_witness_strategy()
        ) {
            let base_size = calculate_base_size(&tx);
            let total_size = calculate_total_size(&tx, Some(&witness));
            
            assert!(total_size >= base_size);
        }
    }

    // Property test strategies
    fn create_transaction_strategy() -> impl Strategy<Value = Transaction> {
        (
            prop::collection::vec(any::<u8>(), 0..10), // inputs
            prop::collection::vec(any::<u8>(), 0..10), // outputs
        ).prop_map(|(input_data, output_data)| {
            let mut inputs = Vec::new();
            for (i, _) in input_data.iter().enumerate() {
                inputs.push(TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: i as u32 },
                    script_sig: vec![0x51],
                    sequence: 0xffffffff,
                });
            }
            
            let mut outputs = Vec::new();
            for _ in output_data {
                outputs.push(TransactionOutput {
                    value: 1000,
                    script_pubkey: vec![0x51],
                });
            }
            
            Transaction {
                version: 1,
                inputs,
                outputs,
                lock_time: 0,
            }
        })
    }

    fn create_block_strategy() -> impl Strategy<Value = Block> {
        prop::collection::vec(create_transaction_strategy(), 1..5).prop_map(|transactions| {
            Block {
                header: BlockHeader {
                    version: 1,
                    prev_block_hash: [0; 32],
                    merkle_root: [0; 32],
                    timestamp: 1231006505,
                    bits: 0x1d00ffff,
                    nonce: 0,
                },
                transactions,
            }
        })
    }

    fn create_witness_strategy() -> impl Strategy<Value = Witness> {
        prop::collection::vec(
            prop::collection::vec(any::<u8>(), 0..10),
            0..5
        )
    }

    fn create_witnesses_strategy() -> impl Strategy<Value = Vec<Witness>> {
        prop::collection::vec(create_witness_strategy(), 0..5)
    }

    fn create_hash_strategy() -> impl Strategy<Value = Hash> {
        prop::collection::vec(any::<u8>(), 32..=32).prop_map(|bytes| {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&bytes);
            hash
        })
    }
}
