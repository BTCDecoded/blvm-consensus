//! Unit tests for spam filter

#[cfg(feature = "utxo-commitments")]
mod tests {
    use consensus_proof::types::{Transaction, TransactionInput, TransactionOutput, OutPoint, ByteString};
    use consensus_proof::utxo_commitments::spam_filter::*;

    fn create_test_transaction(script_pubkey: ByteString) -> Transaction {
        Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey,
            }],
            lock_time: 0,
        }
    }

    #[test]
    fn test_ordinals_detection() {
        let filter = SpamFilter::new();
        
        // OP_RETURN with large data (typical Ordinals)
        let ordinal_script = {
            let mut script = vec![0x6a]; // OP_RETURN
            script.extend(vec![0x00; 100]); // Large data push
            script
        };
        
        let tx = create_test_transaction(ordinal_script);
        let result = filter.is_spam(&tx);
        
        assert!(result.is_spam);
        assert!(result.detected_types.contains(&SpamType::Ordinals));
    }

    #[test]
    fn test_dust_detection() {
        let filter = SpamFilter::new();
        
        // Transaction with all outputs below dust threshold
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 100, // Below 546 satoshi threshold
                script_pubkey: vec![],
            }],
            lock_time: 0,
        };
        
        let result = filter.is_spam(&tx);
        assert!(result.is_spam);
        assert!(result.detected_types.contains(&SpamType::Dust));
    }

    #[test]
    fn test_non_spam_transaction() {
        let filter = SpamFilter::new();
        
        // Normal transaction with sufficient value
        let normal_script = vec![0x76, 0xa9, 0x14]; // P2PKH pattern
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0,
                },
                script_sig: vec![],
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 10000, // Above dust threshold
                script_pubkey: normal_script,
            }],
            lock_time: 0,
        };
        
        let result = filter.is_spam(&tx);
        assert!(!result.is_spam);
        assert_eq!(result.spam_type, SpamType::NotSpam);
    }

    #[test]
    fn test_brc20_detection() {
        let filter = SpamFilter::new();
        
        // BRC-20 transaction with JSON pattern
        let brc20_script = {
            let mut script = vec![0x6a]; // OP_RETURN
            let json_data = b"{\"p\":\"brc-20\",\"op\":\"mint\"}";
            script.extend(json_data);
            script
        };
        
        let tx = create_test_transaction(brc20_script);
        let result = filter.is_spam(&tx);
        
        assert!(result.is_spam);
        assert!(result.detected_types.contains(&SpamType::BRC20));
    }

    #[test]
    fn test_filter_block() {
        let filter = SpamFilter::new();
        
        // Create mix of spam and non-spam transactions
        let transactions = vec![
            create_test_transaction(vec![0x76, 0xa9]), // Non-spam
            create_test_transaction({
                let mut script = vec![0x6a];
                script.extend(vec![0x00; 100]);
                script
            }), // Ordinals spam
            create_test_transaction(vec![0x76, 0xa9]), // Non-spam
            Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: OutPoint { hash: [0; 32], index: 0 },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                }],
                outputs: vec![TransactionOutput {
                    value: 100, // Dust
                    script_pubkey: vec![],
                }],
                lock_time: 0,
            }, // Dust spam
        ];
        
        let (filtered_txs, summary) = filter.filter_block(&transactions);
        
        // Should filter out 2 spam transactions
        assert_eq!(filtered_txs.len(), 2);
        assert_eq!(summary.filtered_count, 2);
        assert!(summary.filtered_size > 0);
        assert!(summary.by_type.ordinals > 0 || summary.by_type.dust > 0);
    }

    #[test]
    fn test_custom_config() {
        // Disable Ordinals filtering
        let config = SpamFilterConfig {
            filter_ordinals: false,
            filter_dust: true,
            filter_brc20: true,
            dust_threshold: 546,
            min_output_value: 546,
        };
        
        let filter = SpamFilter::with_config(config);
        
        // Ordinals should not be detected
        let ordinal_script = {
            let mut script = vec![0x6a];
            script.extend(vec![0x00; 100]);
            script
        };
        
        let tx = create_test_transaction(ordinal_script);
        let result = filter.is_spam(&tx);
        
        // Should not detect as spam (Ordinals filtering disabled)
        assert!(!result.detected_types.contains(&SpamType::Ordinals));
    }
}

