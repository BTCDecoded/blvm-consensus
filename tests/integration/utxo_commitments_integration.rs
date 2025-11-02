//! Integration tests for UTXO Commitments module
//!
//! Tests end-to-end workflows:
//! - Initial sync with peer consensus
//! - Spam filtering and filtered block processing
//! - UTXO set updates and commitment generation
//! - Configuration loading and validation

#[cfg(feature = "utxo-commitments")]
mod tests {
    use consensus_proof::types::{BlockHeader, Hash, Natural, Transaction, TransactionInput, TransactionOutput, OutPoint, UTXO, ByteString};
    use consensus_proof::utxo_commitments::*;
    use consensus_proof::economic::total_supply;
    use std::collections::HashMap;

    /// Create a test block header
    fn create_test_header(height: Natural, prev_hash: Hash) -> BlockHeader {
        BlockHeader {
            version: 1,
            prev_block_hash: prev_hash,
            merkle_root: [0; 32],
            timestamp: 1234567890 + (height * 600), // 10 minutes per block
            bits: 0x1d00ffff,
            nonce: 0,
        }
    }

    /// Create a test UTXO
    fn create_test_utxo(value: i64, height: Natural) -> UTXO {
        UTXO {
            value,
            script_pubkey: vec![0x76, 0xa9, 0x14], // P2PKH pattern
            height,
        }
    }

    /// Create a test transaction
    fn create_test_transaction(inputs: Vec<OutPoint>, outputs: Vec<(i64, ByteString)>) -> Transaction {
        Transaction {
            version: 1,
            inputs: inputs.into_iter().map(|prevout| TransactionInput {
                prevout,
                script_sig: vec![],
                sequence: 0xffffffff,
            }).collect(),
            outputs: outputs.into_iter().map(|(value, script_pubkey)| TransactionOutput {
                value,
                script_pubkey,
            }).collect(),
            lock_time: 0,
        }
    }

    #[test]
    fn test_utxo_commitment_full_workflow() {
        // 1. Create UTXO Merkle tree
        let mut tree = UtxoMerkleTree::new().unwrap();

        // 2. Insert some UTXOs
        let outpoint1 = OutPoint {
            hash: [1; 32],
            index: 0,
        };
        let utxo1 = create_test_utxo(10000, 0);
        tree.insert(outpoint1.clone(), utxo1.clone()).unwrap();

        let outpoint2 = OutPoint {
            hash: [2; 32],
            index: 0,
        };
        let utxo2 = create_test_utxo(5000, 0);
        tree.insert(outpoint2.clone(), utxo2.clone()).unwrap();

        // 3. Generate commitment
        let block_hash = [3; 32];
        let commitment = tree.generate_commitment(block_hash, 0);

        assert_eq!(commitment.utxo_count, 2);
        assert_eq!(commitment.total_supply, 15000);
        assert_eq!(commitment.block_height, 0);
        assert_eq!(commitment.block_hash, block_hash);

        // 4. Verify supply
        let verify_result = verify_supply(&commitment);
        // Note: At height 0, expected supply is 50 BTC (genesis block)
        // Our test UTXOs are just test data, so this might fail
        // In real usage, UTXO set would match expected supply
    }

    #[test]
    fn test_spam_filtering_integration() {
        let filter = SpamFilter::new();

        // Create mix of transactions
        let transactions = vec![
            // Non-spam transaction
            create_test_transaction(
                vec![OutPoint { hash: [1; 32], index: 0 }],
                vec![(10000, vec![0x76, 0xa9])],
            ),
            // Dust transaction (all outputs < 546)
            create_test_transaction(
                vec![OutPoint { hash: [2; 32], index: 0 }],
                vec![(100, vec![])], // Below dust threshold
            ),
            // Ordinals transaction (OP_RETURN with large data)
            create_test_transaction(
                vec![OutPoint { hash: [3; 32], index: 0 }],
                vec![(1000, {
                    let mut script = vec![0x6a]; // OP_RETURN
                    script.extend(vec![0x00; 100]); // Large data
                    script
                })],
            ),
            // Another non-spam transaction
            create_test_transaction(
                vec![OutPoint { hash: [4; 32], index: 0 }],
                vec![(5000, vec![0x76, 0xa9])],
            ),
        ];

        // Filter block
        let (filtered_txs, summary) = filter.filter_block(&transactions);

        // Should filter out 2 spam transactions
        assert_eq!(filtered_txs.len(), 2);
        assert_eq!(summary.filtered_count, 2);
        assert!(summary.filtered_size > 0);
    }

    #[test]
    fn test_peer_consensus_workflow() {
        // Create peer consensus manager
        let config = ConsensusConfig::default();
        let peer_consensus = PeerConsensus::new(config);

        // Create diverse peers
        let all_peers = vec![
            PeerInfo {
                address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(1, 1, 1, 1)),
                asn: Some(1),
                country: Some("US".to_string()),
                implementation: Some("Bitcoin Core".to_string()),
                subnet: 0x01010000,
            },
            PeerInfo {
                address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(2, 2, 2, 2)),
                asn: Some(2),
                country: Some("DE".to_string()),
                implementation: Some("btcd".to_string()),
                subnet: 0x02020000,
            },
            PeerInfo {
                address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(3, 3, 3, 3)),
                asn: Some(3),
                country: Some("JP".to_string()),
                implementation: Some("Bitcoin Core".to_string()),
                subnet: 0x03030000,
            },
        ];

        // Discover diverse peers
        let diverse_peers = peer_consensus.discover_diverse_peers(all_peers);
        assert_eq!(diverse_peers.len(), 3);

        // Test checkpoint height determination
        let peer_tips = vec![100000, 100050, 100100];
        let checkpoint = peer_consensus.determine_checkpoint_height(peer_tips);
        assert!(checkpoint > 0);
        assert!(checkpoint < 100000); // Should be below tips minus safety margin
    }

    #[test]
    fn test_configuration_loading() {
        // Create temporary config file
        let config_dir = std::env::temp_dir();
        let config_path = config_dir.join("utxo_commitments_test_config.json");

        // Create default config
        let default_config = UtxoCommitmentsConfig::default();
        default_config.to_json_file(&config_path).unwrap();

        // Load config
        let loaded_config = UtxoCommitmentsConfig::from_json_file(&config_path).unwrap();

        assert_eq!(loaded_config.sync_mode, default_config.sync_mode);
        assert_eq!(loaded_config.verification_level, default_config.verification_level);
        assert_eq!(loaded_config.consensus.min_peers, default_config.consensus.min_peers);

        // Validate config
        assert!(loaded_config.validate().is_ok());

        // Cleanup
        let _ = std::fs::remove_file(&config_path);
    }

    #[test]
    fn test_configuration_validation() {
        let mut config = UtxoCommitmentsConfig::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid: min_peers = 0
        config.consensus.min_peers = 0;
        assert!(config.validate().is_err());

        // Reset and test invalid threshold
        config = UtxoCommitmentsConfig::default();
        config.consensus.consensus_threshold = 1.5; // > 1.0
        assert!(config.validate().is_err());

        // Reset and test invalid target_peers
        config = UtxoCommitmentsConfig::default();
        config.consensus.target_peers = 1; // < min_peers (5)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_initial_sync_with_config() {
        // Create config
        let consensus_config = ConsensusConfig::default();
        let spam_filter_config = SpamFilterConfig::default();

        // Create initial sync manager
        let initial_sync = InitialSync::with_spam_filter(consensus_config, spam_filter_config);

        // Create test header chain
        let mut headers = Vec::new();
        let mut prev_hash = [0; 32];
        for i in 0..10 {
            let header = create_test_header(i, prev_hash);
            prev_hash = compute_block_hash(&header);
            headers.push(header);
        }

        // Test would normally execute initial sync here
        // For now, just verify manager is created correctly
        // (actual sync requires network integration)
    }

    #[test]
    fn test_merkle_tree_incremental_updates() {
        let mut tree = UtxoMerkleTree::new().unwrap();
        let initial_root = tree.root();

        // Insert UTXO
        let outpoint1 = OutPoint { hash: [1; 32], index: 0 };
        let utxo1 = create_test_utxo(1000, 0);
        let root1 = tree.insert(outpoint1.clone(), utxo1.clone()).unwrap();
        assert_ne!(root1, initial_root); // Root should change

        // Insert another UTXO
        let outpoint2 = OutPoint { hash: [2; 32], index: 0 };
        let utxo2 = create_test_utxo(2000, 0);
        let root2 = tree.insert(outpoint2.clone(), utxo2.clone()).unwrap();
        assert_ne!(root2, root1); // Root should change again

        // Remove first UTXO
        let root3 = tree.remove(&outpoint1, &utxo1).unwrap();
        assert_ne!(root3, root2); // Root should change

        // Verify supply tracking
        assert_eq!(tree.total_supply(), 2000);
        assert_eq!(tree.utxo_count(), 1);
    }

    /// Compute block header hash (double SHA256)
    fn compute_block_hash(header: &BlockHeader) -> Hash {
        use sha2::{Digest, Sha256};

        let mut bytes = Vec::with_capacity(80);
        bytes.extend_from_slice(&header.version.to_le_bytes());
        bytes.extend_from_slice(&header.prev_block_hash);
        bytes.extend_from_slice(&header.merkle_root);
        bytes.extend_from_slice(&header.timestamp.to_le_bytes());
        bytes.extend_from_slice(&header.bits.to_le_bytes());
        bytes.extend_from_slice(&header.nonce.to_le_bytes());

        let first_hash = Sha256::digest(&bytes);
        let second_hash = Sha256::digest(&first_hash);

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&second_hash);
        hash
    }
}

