//! Integration tests for UTXO proof verification
//!
//! Tests the complete flow:
//! 1. Generate UTXO set and commitment
//! 2. Generate proofs for UTXOs
//! 3. Verify proofs against commitment
//! 4. Test attack scenarios
//! 5. Test batch verification

#[cfg(feature = "utxo-commitments")]
#[cfg(test)]
mod tests {
    use blvm_consensus::types::{OutPoint, UTXO, Hash, Natural};
    use blvm_consensus::utxo_commitments::*;
    use blvm_consensus::utxo_commitments::peer_consensus::{ConsensusConfig, ConsensusResult, PeerConsensus};

    #[test]
    fn test_integration_proof_verification_flow() {
        // Create UTXO tree and add UTXOs
        let mut tree = UtxoMerkleTree::new().unwrap();
        
        let outpoint1 = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo1 = UTXO {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
            height: 0,
            is_coinbase: false,
        };
        
        let outpoint2 = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let utxo2 = UTXO {
            value: 2000,
            script_pubkey: vec![0x52], // OP_2
            height: 0,
            is_coinbase: false,
        };
        
        // Insert UTXOs
        tree.insert(outpoint1.clone(), utxo1.clone()).unwrap();
        tree.insert(outpoint2.clone(), utxo2.clone()).unwrap();
        
        // Generate commitment
        let block_hash = [3; 32];
        let commitment = tree.generate_commitment(block_hash, 0);
        
        // Generate proofs
        let proof1 = tree.generate_proof(&outpoint1).unwrap();
        let proof2 = tree.generate_proof(&outpoint2).unwrap();
        
        // Verify proofs
        let is_valid1 = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint1,
            &utxo1,
            proof1,
        ).unwrap();
        assert!(is_valid1, "Proof 1 should be valid");
        
        let is_valid2 = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint2,
            &utxo2,
            proof2,
        ).unwrap();
        assert!(is_valid2, "Proof 2 should be valid");
    }

    #[test]
    fn test_integration_batch_verification() {
        // Create UTXO tree with multiple UTXOs
        let mut tree = UtxoMerkleTree::new().unwrap();
        let mut utxos_to_verify = Vec::new();
        
        // Add 10 UTXOs
        for i in 0..10 {
            let outpoint = OutPoint {
                hash: [i as u8; 32],
                index: 0,
            };
            let utxo = UTXO {
                value: (i + 1) * 1000,
                script_pubkey: vec![0x51 + (i as u8)],
                height: 0,
                is_coinbase: false,
            };
            
            tree.insert(outpoint.clone(), utxo.clone()).unwrap();
            
            // Generate proof
            let proof = tree.generate_proof(&outpoint).unwrap();
            utxos_to_verify.push((outpoint, utxo, proof));
        }
        
        // Generate commitment
        let block_hash = [99; 32];
        let commitment = tree.generate_commitment(block_hash, 0);
        
        // Create consensus result
        let consensus = ConsensusResult {
            commitment,
            agreement_count: 8,
            total_peers: 10,
            agreement_ratio: 0.8,
        };
        
        // Test batch verification
        let peer_consensus = PeerConsensus::new(ConsensusConfig::default());
        let result = peer_consensus.verify_utxo_proofs(&consensus, utxos_to_verify);
        
        assert!(result.is_ok(), "Batch verification should succeed");
        assert!(result.unwrap(), "All proofs should be valid");
    }

    #[test]
    fn test_integration_attack_scenario_missing_utxo() {
        // Simulate attack: commitment has correct supply but missing UTXO
        let mut tree_a = UtxoMerkleTree::new().unwrap();
        let mut tree_b = UtxoMerkleTree::new().unwrap();
        
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo = UTXO {
            value: 5000,
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: false,
        };
        
        // Tree A has the UTXO
        tree_a.insert(outpoint.clone(), utxo.clone()).unwrap();
        let commitment_a = tree_a.generate_commitment([2; 32], 0);
        
        // Tree B doesn't have the UTXO (attack scenario)
        // But has same total supply (different UTXOs)
        let outpoint_b = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let utxo_b = UTXO {
            value: 5000, // Same value, different UTXO
            script_pubkey: vec![0x52],
            height: 0,
            is_coinbase: false,
        };
        tree_b.insert(outpoint_b, utxo_b).unwrap();
        let commitment_b = tree_b.generate_commitment([3; 32], 0);
        
        // Generate proof from tree A
        let proof = tree_a.generate_proof(&outpoint).unwrap();
        
        // Try to verify against commitment B (should fail)
        let is_valid = UtxoMerkleTree::verify_utxo_proof(
            &commitment_b, // Wrong commitment (missing UTXO)
            &outpoint,
            &utxo,
            proof,
        ).unwrap();
        
        assert!(!is_valid, "Proof verification should fail for missing UTXO");
    }

    #[test]
    fn test_integration_attack_scenario_modified_utxo() {
        // Simulate attack: commitment has correct supply but modified UTXO
        let mut tree = UtxoMerkleTree::new().unwrap();
        
        let outpoint = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo_correct = UTXO {
            value: 5000,
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: false,
        };
        
        // Insert correct UTXO
        tree.insert(outpoint.clone(), utxo_correct.clone()).unwrap();
        let commitment = tree.generate_commitment([2; 32], 0);
        
        // Generate proof
        let proof = tree.generate_proof(&outpoint).unwrap();
        
        // Try to verify with modified UTXO (different value)
        let utxo_modified = UTXO {
            value: 1000, // Modified value
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: false,
        };
        
        let is_valid = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint,
            &utxo_modified, // Wrong UTXO data
            proof,
        ).unwrap();
        
        assert!(!is_valid, "Proof verification should fail for modified UTXO");
    }

    #[test]
    fn test_integration_peer_consensus_with_proof_verification() {
        // Test complete peer consensus flow with proof verification
        let mut tree = UtxoMerkleTree::new().unwrap();
        
        // Add UTXOs
        let outpoint1 = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo1 = UTXO {
            value: 1000,
            script_pubkey: vec![0x51],
            height: 0,
            is_coinbase: false,
        };
        
        let outpoint2 = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let utxo2 = UTXO {
            value: 2000,
            script_pubkey: vec![0x52],
            height: 0,
            is_coinbase: false,
        };
        
        tree.insert(outpoint1.clone(), utxo1.clone()).unwrap();
        tree.insert(outpoint2.clone(), utxo2.clone()).unwrap();
        
        let commitment = tree.generate_commitment([3; 32], 0);
        
        // Create consensus result
        let consensus = ConsensusResult {
            commitment,
            agreement_count: 8,
            total_peers: 10,
            agreement_ratio: 0.8,
        };
        
        // Generate proofs
        let proof1 = tree.generate_proof(&outpoint1).unwrap();
        let proof2 = tree.generate_proof(&outpoint2).unwrap();
        
        // Verify proofs via peer consensus
        let peer_consensus = PeerConsensus::new(ConsensusConfig::default());
        let utxos_to_verify = vec![
            (outpoint1, utxo1, proof1),
            (outpoint2, utxo2, proof2),
        ];
        
        let result = peer_consensus.verify_utxo_proofs(&consensus, utxos_to_verify);
        
        assert!(result.is_ok(), "Proof verification should succeed");
        assert!(result.unwrap(), "All proofs should be valid");
    }

    #[test]
    fn test_integration_proof_serialization() {
        // Test that proofs can be serialized and deserialized
        let mut tree = UtxoMerkleTree::new().unwrap();
        
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
        
        tree.insert(outpoint.clone(), utxo.clone()).unwrap();
        let commitment = tree.generate_commitment([2; 32], 0);
        
        // Generate proof
        let proof = tree.generate_proof(&outpoint).unwrap();
        
        // Serialize proof
        let proof_bytes = bincode::serialize(&proof).unwrap();
        
        // Deserialize proof
        let proof_deserialized: sparse_merkle_tree::MerkleProof = bincode::deserialize(&proof_bytes).unwrap();
        
        // Verify deserialized proof still works
        let is_valid = UtxoMerkleTree::verify_utxo_proof(
            &commitment,
            &outpoint,
            &utxo,
            proof_deserialized,
        ).unwrap();
        
        assert!(is_valid, "Deserialized proof should still be valid");
    }

    #[test]
    fn test_integration_large_utxo_set() {
        // Test with larger UTXO set to ensure scalability
        let mut tree = UtxoMerkleTree::new().unwrap();
        let mut utxos_to_verify = Vec::new();
        
        // Add 100 UTXOs
        for i in 0..100 {
            let outpoint = OutPoint {
                hash: {
                    let mut hash = [0u8; 32];
                    hash[0] = (i % 256) as u8;
                    hash[1] = ((i / 256) % 256) as u8;
                    hash
                },
                index: (i % 10) as u32,
            };
            let utxo = UTXO {
                value: (i + 1) * 100,
                script_pubkey: vec![0x51; (i % 20 + 1)],
                height: 0,
                is_coinbase: false,
            };
            
            tree.insert(outpoint.clone(), utxo.clone()).unwrap();
            
            // Generate proof for every 10th UTXO
            if i % 10 == 0 {
                let proof = tree.generate_proof(&outpoint).unwrap();
                utxos_to_verify.push((outpoint, utxo, proof));
            }
        }
        
        let commitment = tree.generate_commitment([99; 32], 0);
        
        // Verify all proofs
        for (outpoint, utxo, proof) in &utxos_to_verify {
            let is_valid = UtxoMerkleTree::verify_utxo_proof(
                &commitment,
                outpoint,
                utxo,
                proof.clone(),
            ).unwrap();
            assert!(is_valid, "Proof should be valid for UTXO at index");
        }
    }
}

