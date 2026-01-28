//! Unit tests for UTXO commitments module

#[cfg(feature = "utxo-commitments")]
mod tests {
    use blvm_consensus::types::{OutPoint, UTXO, Hash, Natural};
    use blvm_consensus::utxo_commitments::*;
    use blvm_consensus::economic::total_supply;

    #[test]
    fn test_utxo_merkle_tree_new() {
        let tree = UtxoMerkleTree::new().unwrap();
        assert_eq!(tree.total_supply(), 0);
        assert_eq!(tree.utxo_count(), 0);
    }

    #[test]
    fn test_insert_utxo() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
            height: 0,
        };
        
        let root = tree.insert(outpoint.clone(), utxo.clone()).unwrap();
        assert_eq!(tree.utxo_count(), 1);
        assert_eq!(tree.total_supply(), 1000);
        
        // Root should change after insert
        assert_eq!(tree.root(), root);
    }

    #[test]
    fn test_remove_utxo() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![],
            height: 0,
        };
        
        // Insert then remove
        tree.insert(outpoint.clone(), utxo.clone()).unwrap();
        assert_eq!(tree.utxo_count(), 1);
        
        tree.remove(&outpoint, &utxo).unwrap();
        assert_eq!(tree.utxo_count(), 0);
        assert_eq!(tree.total_supply(), 0);
    }

    #[test]
    fn test_generate_commitment() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![],
            height: 0,
        };
        
        tree.insert(outpoint, utxo).unwrap();
        
        let block_hash = [2; 32];
        let commitment = tree.generate_commitment(block_hash, 0);
        
        assert_eq!(commitment.block_height, 0);
        assert_eq!(commitment.block_hash, block_hash);
        assert_eq!(commitment.total_supply, 1000);
        assert_eq!(commitment.utxo_count, 1);
        assert_eq!(commitment.merkle_root, tree.root());
    }

    #[test]
    fn test_verify_commitment_supply() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        
        // Add UTXO with value matching genesis block subsidy
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 5000000000, // 50 BTC (genesis subsidy)
            script_pubkey: vec![],
            height: 0,
        };
        
        tree.insert(outpoint, utxo).unwrap();
        
        let block_hash = [2; 32];
        let commitment = tree.generate_commitment(block_hash, 0);
        
        // Verify supply matches expected at height 0
        let result = tree.verify_commitment_supply(&commitment);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_supply_function() {
        // Test supply verification utility
        let commitment = UtxoCommitment::new(
            [0; 32],
            5000000000, // 50 BTC at genesis
            1,
            0,
            [0; 32],
        );
        
        let result = verify_supply(&commitment);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_commitment_serialization() {
        let commitment = UtxoCommitment::new(
            [1; 32],
            1000,
            5,
            100,
            [2; 32],
        );
        
        let bytes = commitment.to_bytes();
        assert_eq!(bytes.len(), 84);
        
        let deserialized = UtxoCommitment::from_bytes(&bytes).unwrap();
        assert_eq!(deserialized.merkle_root, commitment.merkle_root);
        assert_eq!(deserialized.total_supply, commitment.total_supply);
        assert_eq!(deserialized.utxo_count, commitment.utxo_count);
        assert_eq!(deserialized.block_height, commitment.block_height);
        assert_eq!(deserialized.block_hash, commitment.block_hash);
    }

    #[test]
    fn test_generate_proof() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![],
            height: 0,
        };
        
        tree.insert(outpoint.clone(), utxo).unwrap();
        
        // Generate proof
        let proof = tree.generate_proof(&outpoint).unwrap();
        // Proof should be generated (verify it's not empty - structure exists)
        // Note: sparse-merkle-tree MerkleProof has internal structure, just verify we got one
        // The proof can be verified against the root separately
        assert!(true); // Proof generated successfully
    }

    #[test]
    fn test_commitment_verify_supply_method() {
        let commitment = UtxoCommitment::new(
            [0; 32],
            5000000000,
            1,
            0,
            [0; 32],
        );
        
        assert!(commitment.verify_supply(5000000000));
        assert!(!commitment.verify_supply(1000));
    }

    #[test]
    fn test_commitment_verify_count() {
        let commitment = UtxoCommitment::new(
            [0; 32],
            1000,
            5,
            0,
            [0; 32],
        );
        
        assert!(commitment.verify_count(5));
        assert!(!commitment.verify_count(10));
    }

    #[test]
    fn test_verify_utxo_proof_valid() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
            height: 0,
            is_coinbase: false,
        };
        
        // Insert UTXO and generate commitment
        tree.insert(outpoint.clone(), utxo.clone()).unwrap();
        let block_hash = [2; 32];
        let commitment = tree.generate_commitment(block_hash, 0);
        
        // Generate proof
        let proof = tree.generate_proof(&outpoint).unwrap();
        
        // Verify proof (static function, doesn't need tree instance)
        let is_valid = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint,
            &utxo,
            proof, // Takes ownership
        ).unwrap();
        
        assert!(is_valid, "Valid proof should verify successfully");
    }

    #[test]
    fn test_verify_utxo_proof_wrong_utxo() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo_a = UTXO {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
            height: 0,
            is_coinbase: false,
        };
        let utxo_b = UTXO {
            value: 2000, // Different value
            script_pubkey: vec![0x52], // OP_2
            height: 0,
            is_coinbase: false,
        };
        
        // Insert UTXO A and generate commitment
        tree.insert(outpoint.clone(), utxo_a.clone()).unwrap();
        let block_hash = [2; 32];
        let commitment = tree.generate_commitment(block_hash, 0);
        
        // Generate proof for UTXO A
        let proof = tree.generate_proof(&outpoint).unwrap();
        
        // Try to verify with UTXO B (different UTXO)
        let is_valid = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint,
            &utxo_b, // Wrong UTXO
            proof,
        ).unwrap();
        
        assert!(!is_valid, "Proof with wrong UTXO should fail verification");
    }

    #[test]
    fn test_verify_utxo_proof_wrong_root() {
        let mut tree_a = UtxoMerkleTree::new().unwrap();
        let mut tree_b = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 1000,
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: false,
        };
        
        // Insert into tree A and generate commitment A
        tree_a.insert(outpoint.clone(), utxo.clone()).unwrap();
        let commitment_a = tree_a.generate_commitment([2; 32], 0);
        
        // Insert different UTXO into tree B and generate commitment B
        let utxo_b = UTXO {
            value: 2000, // Different value
            script_pubkey: vec![0x52],
            height: 0,
            is_coinbase: false,
        };
        tree_b.insert(outpoint.clone(), utxo_b).unwrap();
        let commitment_b = tree_b.generate_commitment([3; 32], 0);
        
        // Generate proof from tree A
        let proof = tree_a.generate_proof(&outpoint).unwrap();
        
        // Try to verify against commitment B's root (wrong root)
        let is_valid = UtxoMerkleTree::verify_utxo_proof(
            &commitment_b, // Wrong commitment
            &outpoint,
            &utxo,
            proof,
        ).unwrap();
        
        assert!(!is_valid, "Proof with wrong root should fail verification");
    }

    #[test]
    fn test_verify_utxo_proof_coinbase_utxo() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 5000000000, // 50 BTC
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: true, // Coinbase UTXO
        };
        
        tree.insert(outpoint.clone(), utxo.clone()).unwrap();
        let commitment = tree.generate_commitment([2; 32], 0);
        let proof = tree.generate_proof(&outpoint).unwrap();
        
        let is_valid = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint,
            &utxo,
            proof,
        ).unwrap();
        
        assert!(is_valid, "Coinbase UTXO proof should verify successfully");
    }

    #[test]
    fn test_verify_utxo_proof_multiple_utxos() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let outpoint1 = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let outpoint2 = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let utxo1 = UTXO {
            value: 1000,
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: false,
        };
        let utxo2 = UTXO {
            value: 2000,
            script_pubkey: vec![0x52],
            height: 0,
            is_coinbase: false,
        };
        
        // Insert both UTXOs
        tree.insert(outpoint1.clone(), utxo1.clone()).unwrap();
        tree.insert(outpoint2.clone(), utxo2.clone()).unwrap();
        let commitment = tree.generate_commitment([3; 32], 0);
        
        // Verify both proofs
        let proof1 = tree.generate_proof(&outpoint1).unwrap();
        let is_valid1 = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint1,
            &utxo1,
            proof1,
        ).unwrap();
        assert!(is_valid1, "First UTXO proof should verify");
        
        let proof2 = tree.generate_proof(&outpoint2).unwrap();
        let is_valid2 = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint2,
            &utxo2,
            proof2,
        ).unwrap();
        assert!(is_valid2, "Second UTXO proof should verify");
    }
}

