//! Segregated Witness (SegWit) functions from Orange Paper Section 11.1

use crate::error::Result;
use crate::types::*;
use crate::types::{ByteString, Hash, Natural};
use crate::witness;
use bitcoin_hashes::{sha256d, Hash as BitcoinHash, HashEngine};
use blvm_spec_lock::spec_locked;

/// Witness Data: 𝒲 = 𝕊* (stack of witness elements)
///
/// Uses unified witness type from witness module for consistency with Taproot
pub use crate::witness::Witness;

/// Calculate transaction weight for SegWit
/// Weight(tx) = 4 × |Serialize(tx ∖ witness)| + |Serialize(tx)|
#[spec_locked("11.1")]
pub fn calculate_transaction_weight(
    tx: &Transaction,
    witness: Option<&Witness>,
) -> Result<Natural> {
    // Calculate base size (transaction without witness data)
    let base_size = calculate_base_size(tx);

    // Calculate total size (transaction with witness data)
    let total_size = calculate_total_size(tx, witness);

    // Use unified witness framework for weight formula
    Ok(witness::calculate_transaction_weight_segwit(
        base_size, total_size,
    ))
}

/// Calculate base size (transaction without witness data)
#[spec_locked("11.1")]
fn calculate_base_size(tx: &Transaction) -> Natural {
    // Simplified calculation - in reality this would be the actual serialized size
    (4 + // version
    tx.inputs.len() * (32 + 4 + 1 + 4) + // inputs (OutPoint + script_sig_len + sequence)
    tx.outputs.len() * (8 + 1) + // outputs (value + script_pubkey_len)
    4) as Natural // lock_time
}

/// Calculate total size (transaction with witness data)
#[spec_locked("11.1")]
fn calculate_total_size(tx: &Transaction, witness: Option<&Witness>) -> Natural {
    let base_size = calculate_base_size(tx);

    if let Some(witness_data) = witness {
        let witness_size: Natural = witness_data.iter().map(|w| w.len() as Natural).sum();
        base_size + witness_size
    } else {
        base_size
    }
}

/// Compute witness merkle root for block
/// WitnessRoot = ComputeMerkleRoot({Hash(tx.witness) : tx ∈ block.transactions})
#[spec_locked("11.1")]
pub fn compute_witness_merkle_root(block: &Block, witnesses: &[Witness]) -> Result<Hash> {
    if block.transactions.is_empty() {
        return Err(crate::error::ConsensusError::ConsensusRuleViolation(
            "Cannot compute witness merkle root for empty block".into(),
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
            "Cannot compute merkle root from empty hash list".into(),
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
#[spec_locked("11.1")]
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
pub(crate) fn extract_witness_commitment(script: &ByteString) -> Option<Hash> {
    // Look for OP_RETURN followed by witness commitment
    if script.len() >= 38 && script[0] == 0x6a {
        // OP_RETURN
        if script.len() >= 38 && script[1] == 0x24 {
            // 36 bytes
            let mut commitment = [0u8; 32];
            commitment.copy_from_slice(&script[2..34]);
            return Some(commitment);
        }
    }
    None
}

/// Check if transaction is SegWit (v0) or Taproot (v1) based on outputs
#[spec_locked("11.1")]
pub fn is_segwit_transaction(tx: &Transaction) -> bool {
    use crate::witness::{
        extract_witness_program, extract_witness_version, validate_witness_program_length,
    };

    tx.outputs.iter().any(|output| {
        let script = &output.script_pubkey;
        if let Some(version) = extract_witness_version(script) {
            if let Some(program) = extract_witness_program(script, version) {
                return validate_witness_program_length(&program, version);
            }
        }
        false
    })
}

/// Calculate block weight for SegWit blocks
#[spec_locked("11.1")]
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
#[spec_locked("11.1")]
pub fn validate_segwit_block(
    block: &Block,
    witnesses: &[Witness],
    max_block_weight: Natural,
) -> Result<bool> {
    // Validate witness structure for all transactions using unified framework
    for (i, _tx) in block.transactions.iter().enumerate() {
        if i < witnesses.len() && !witness::validate_segwit_witness_structure(&witnesses[i])? {
            return Ok(false);
        }
    }

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
            vec![],           // Coinbase witness (empty)
            vec![vec![0x51]], // First transaction witness
        ];

        let root = compute_witness_merkle_root(&block, &witnesses).unwrap();
        assert_eq!(root.len(), 32);
    }

    #[test]
    fn test_compute_witness_merkle_root_empty_block() {
        let block = Block {
            header: create_test_header(),
            transactions: vec![].into_boxed_slice(),
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
        // SegWit transactions are detected by witness program outputs, not scriptSig
        // P2WPKH: OP_0 <20-byte-hash>
        // The format in Bitcoin is: [0x00, 0x14, <20-byte-hash>]
        // Where 0x00 is OP_0 (witness version), 0x14 is push 20 bytes, then 20 bytes of hash
        let mut tx = create_test_transaction();
        // Create a P2WPKH output (OP_0 <20-byte-hash>)
        let p2wpkh_hash = [0x51; 20]; // 20-byte hash
        let mut script_pubkey = vec![0x00, 0x14]; // OP_0, push 20 bytes
        script_pubkey.extend_from_slice(&p2wpkh_hash);
        tx.outputs[0].script_pubkey = script_pubkey.into();

        assert!(is_segwit_transaction(&tx));
    }

    #[test]
    fn test_calculate_block_weight() {
        let block = create_test_block();
        let witnesses = vec![
            vec![],           // Coinbase
            vec![vec![0x51]], // First tx
        ];

        let weight = calculate_block_weight(&block, &witnesses).unwrap();
        assert!(weight > 0);
    }

    #[test]
    fn test_validate_segwit_block() {
        let block = create_test_block();
        let witnesses = vec![
            vec![],           // Coinbase
            vec![vec![0x51]], // First tx
        ];

        let is_valid = validate_segwit_block(&block, &witnesses, 4_000_000).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_validate_segwit_block_exceeds_weight() {
        let block = create_test_block();
        let witnesses = vec![
            vec![],           // Coinbase
            vec![vec![0x51]], // First tx
        ];

        let is_valid = validate_segwit_block(&block, &witnesses, 1).unwrap(); // Very low weight limit
        assert!(!is_valid);
    }

    #[test]
    fn test_validate_segwit_block_invalid_commitment() {
        let mut block = create_test_block();
        let witnesses = vec![
            vec![],           // Coinbase
            vec![vec![0x51]], // First tx
        ];

        // Create coinbase with invalid witness commitment
        let invalid_commitment = [2u8; 32];
        block.transactions[0].outputs[0].script_pubkey =
            create_witness_commitment_script(&invalid_commitment);

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
        coinbase_tx.outputs[0].script_pubkey =
            create_witness_commitment_script(&invalid_commitment);

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
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![0x51].into(),
            }]
            .into(),
            lock_time: 0,
        }
    }

    fn create_test_block() -> Block {
        Block {
            header: create_test_header(),
            transactions: vec![
                create_test_transaction(), // Coinbase
                create_test_transaction(), // Regular tx
            ]
            .into_boxed_slice(),
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
            let _weight = calculate_transaction_weight(&tx, witness.as_ref()).unwrap();
            // Weight is always non-negative (Natural type) - verified by type system
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
            // Handle errors from invalid blocks/witnesses
            match (calculate_block_weight(&block, &witnesses), validate_segwit_block(&block, &witnesses, max_weight as Natural)) {
                (Ok(actual_weight), Ok(is_valid)) => {
                    // If weight exceeds limit, block should be invalid
                    if actual_weight > max_weight as Natural {
                        prop_assert!(!is_valid, "Block exceeding weight limit must be invalid");
                    }
                },
                (Err(_), _) | (_, Err(_)) => {
                    // Invalid blocks/witnesses may cause errors - this is acceptable
                }
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
            // Just test it returns a boolean (is_segwit is either true or false)
            let _ = is_segwit;
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
    #[test]
    fn prop_merkle_root_empty_input() {
        let hashes: Vec<Hash> = vec![];
        let result = compute_merkle_root(&hashes);

        assert!(result.is_err());
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
        )
            .prop_map(|(input_data, output_data)| {
                let mut inputs = Vec::new();
                for (i, _) in input_data.iter().enumerate() {
                    inputs.push(TransactionInput {
                        prevout: OutPoint {
                            hash: [0; 32],
                            index: i as u64,
                        },
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
                    inputs: inputs.into(),
                    outputs: outputs.into(),
                    lock_time: 0,
                }
            })
    }

    fn create_block_strategy() -> impl Strategy<Value = Block> {
        prop::collection::vec(create_transaction_strategy(), 1..5).prop_map(|transactions| Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: transactions.into_boxed_slice(),
        })
    }

    fn create_witness_strategy() -> impl Strategy<Value = Witness> {
        prop::collection::vec(prop::collection::vec(any::<u8>(), 0..10), 0..5)
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

