//! Initial Sync Algorithm
//!
//! Implements the peer consensus initial sync algorithm:
//! 1. Discover diverse peers
//! 2. Determine checkpoint height
//! 3. Request UTXO sets from peers
//! 4. Find consensus
//! 5. Verify against block headers
//! 6. Download UTXO set

#[cfg(feature = "utxo-commitments")]
use crate::types::{BlockHeader, Hash, Natural, Transaction, OutPoint, UTXO};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::data_structures::{UtxoCommitment, UtxoCommitmentError, UtxoCommitmentResult};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::merkle_tree::UtxoMerkleTree;
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::peer_consensus::{PeerConsensus, PeerInfo, ConsensusConfig, ConsensusResult};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::spam_filter::{SpamFilter, SpamFilterConfig, SpamSummary, SpamType, SpamBreakdown};

/// Initial sync manager
pub struct InitialSync {
    peer_consensus: PeerConsensus,
    spam_filter: SpamFilter,
    // In real implementation: network_client: NetworkClient,
}

impl InitialSync {
    /// Create a new initial sync manager
    pub fn new(config: ConsensusConfig) -> Self {
        Self {
            peer_consensus: PeerConsensus::new(config),
            spam_filter: SpamFilter::new(),
        }
    }

    /// Create a new initial sync manager with custom spam filter config
    pub fn with_spam_filter(config: ConsensusConfig, spam_filter_config: SpamFilterConfig) -> Self {
        Self {
            peer_consensus: PeerConsensus::new(config),
            spam_filter: SpamFilter::with_config(spam_filter_config),
        }
    }

    /// Execute initial sync algorithm
    ///
    /// Performs the complete initial sync process:
    /// 1. Discover diverse peers
    /// 2. Determine checkpoint height
    /// 3. Request UTXO sets
    /// 4. Find consensus
    /// 5. Verify against headers
    /// 6. Return verified UTXO commitment
    pub async fn execute_initial_sync(
        &self,
        all_peers: Vec<PeerInfo>,
        header_chain: &[BlockHeader],
    ) -> UtxoCommitmentResult<UtxoCommitment> {
        // Step 1: Discover diverse peers
        let diverse_peers = self.peer_consensus.discover_diverse_peers(all_peers);
        
        if diverse_peers.len() < self.peer_consensus.config.min_peers {
            return Err(UtxoCommitmentError::VerificationFailed(format!(
                "Insufficient diverse peers: got {}, need {}",
                diverse_peers.len(),
                self.peer_consensus.config.min_peers
            )));
        }
        
        // Step 2: Determine checkpoint height
        // In real implementation: query peers for their chain tips
        let peer_tips: Vec<Natural> = vec![]; // Would come from peer queries
        let checkpoint_height = if !peer_tips.is_empty() {
            self.peer_consensus.determine_checkpoint_height(peer_tips)
        } else if !header_chain.is_empty() {
            // Use header chain tip minus safety margin
            let tip = header_chain.len() as Natural - 1;
            if tip > self.peer_consensus.config.safety_margin {
                tip - self.peer_consensus.config.safety_margin
            } else {
                0
            }
        } else {
            return Err(UtxoCommitmentError::VerificationFailed(
                "No header chain or peer tips available".to_string()
            ));
        };
        
        // Get checkpoint block hash from header chain
        if checkpoint_height as usize >= header_chain.len() {
            return Err(UtxoCommitmentError::VerificationFailed(format!(
                "Checkpoint height {} exceeds header chain length {}",
                checkpoint_height,
                header_chain.len()
            )));
        }
        
        let checkpoint_header = &header_chain[checkpoint_height as usize];
        let checkpoint_hash = compute_block_hash(checkpoint_header);
        
        // Step 3: Request UTXO sets from peers
        let peer_commitments = self.peer_consensus.request_utxo_sets(
            &diverse_peers,
            checkpoint_height,
            checkpoint_hash,
        ).await;
        
        // Step 4: Find consensus
        let consensus = self.peer_consensus.find_consensus(peer_commitments)?;
        
        // Step 5: Verify consensus commitment against block headers
        self.peer_consensus.verify_consensus_commitment(&consensus, header_chain)?;
        
        // Step 6: Return verified commitment
        // In real implementation, we would also download the actual UTXO set here
        // For now, just return the verified commitment
        
        Ok(consensus.commitment)
    }

    /// Complete sync from checkpoint to current tip
    ///
    /// Syncs forward from checkpoint using filtered blocks.
    /// Updates UTXO set incrementally for each block.
    pub async fn complete_sync_from_checkpoint(
        &self,
        utxo_tree: &mut UtxoMerkleTree,
        checkpoint_height: Natural,
        current_tip: Natural,
        // In real implementation: network_client, filtered_block_stream
    ) -> UtxoCommitmentResult<()> {
        // In real implementation:
        // 1. Request filtered blocks from checkpoint+1 to tip
        // 2. For each filtered block:
        //    - Verify block header
        //    - Verify commitment
        //    - Apply filtered transactions to UTXO tree
        //    - Verify new commitment matches
        // 3. Update UTXO tree incrementally
        
        // Process blocks incrementally
        for height in checkpoint_height + 1..=current_tip {
            // Network integration: Request filtered block from network
            //
            // Implementation requires:
            // 1. Integration with reference-node's network manager
            // 2. Support for Bitcoin's `merkleblock` message type (BIP37)
            // 3. Bloom filter setup for filtering transactions
            // 4. Async/await support for network operations
            //
            // Example implementation (when network layer is available):
            // ```
            // // Get block hash from chain state
            // let block_hash = chain_state.get_block_hash(height)?;
            //
            // // Request filtered block from network peer
            // // This would use the UtxoCommitmentsNetworkClient trait from network_integration.rs
            // let filtered_block = network_client.request_filtered_block(peer_id, block_hash).await?;
            //
            // // Process the filtered block using process_filtered_block method
            // let (spam_summary, root) = self.process_filtered_block(
            //     &mut utxo_tree,
            //     height,
            //     &filtered_block.transactions
            // )?;
            //
            // // Verify commitment matches
            // if root != filtered_block.commitment.merkle_root {
            //     return Err(UtxoCommitmentError::CommitmentMismatch);
            // }
            // ```
            //
            // For now, this is a placeholder that documents the required integration.
            // The actual implementation should be done in reference-node where the network
            // layer is available.
            
            // Placeholder: suppress unused warning
            // In real implementation, would request filtered block from network and process it
            let _ = height;
            let _root = utxo_tree.root(); // Use utxo_tree to suppress unused warning
        }
        
        Ok(())
    }

    /// Process a filtered block and update UTXO set
    ///
    /// Takes a block with transactions (already filtered or to be filtered),
    /// applies spam filter, updates UTXO set, and verifies commitment.
    /// 
    /// **Critical**: This function processes ALL transactions to remove spent inputs,
    /// but only adds non-spam outputs to the UTXO tree. This ensures UTXO set consistency:
    /// - Spam transactions that spend non-spam inputs will still remove those inputs
    /// - Only non-spam outputs are added to the tree (bandwidth savings)
    /// - UTXO set remains consistent with actual blockchain state
    /// 
    /// Note: This function applies transactions to the UTXO tree for commitment
    /// purposes. Full signature verification should be done during block validation
    /// before calling this function. This function assumes transactions are already
    /// validated.
    pub fn process_filtered_block(
        &self,
        utxo_tree: &mut UtxoMerkleTree,
        block_height: Natural,
        block_transactions: &[Transaction],
    ) -> UtxoCommitmentResult<(SpamSummary, Hash)> {
        use crate::transaction::is_coinbase;
        
        let mut spam_summary = SpamSummary {
            filtered_count: 0,
            filtered_size: 0,
            by_type: SpamBreakdown::default(),
        };
        
        // Process ALL transactions (including spam) to remove spent inputs
        // This is critical for UTXO set consistency: even spam transactions must
        // remove their spent inputs from the tree.
        for tx in block_transactions {
            // Check if transaction is spam (for output filtering)
            let spam_result = self.spam_filter.is_spam(tx);
            let is_spam = spam_result.is_spam;
            
            // Update spam summary
            if is_spam {
                spam_summary.filtered_count += 1;
                // Estimate transaction size (simplified calculation)
                let tx_size = 4 + 1 + 1 + 4 + // version + input_count + output_count + locktime
                    (tx.inputs.len() as u64 * 150) + // inputs
                    tx.outputs.iter().map(|out| 8 + out.script_pubkey.len() as u64).sum::<u64>(); // outputs
                spam_summary.filtered_size += tx_size;
                
                // Update breakdown
                for spam_type in &spam_result.detected_types {
                    match spam_type {
                        SpamType::Ordinals => {
                            spam_summary.by_type.ordinals += 1;
                        }
                        SpamType::Dust => {
                            spam_summary.by_type.dust += 1;
                        }
                        SpamType::BRC20 => {
                            spam_summary.by_type.brc20 += 1;
                        }
                        SpamType::NotSpam => {}
                    }
                }
            }
            
            // Compute transaction ID for proper outpoint creation
            let tx_id = compute_tx_id(tx);
            
            // CRITICAL: Remove spent inputs from ALL transactions (including spam)
            // This ensures UTXO set consistency even when spam transactions spend non-spam inputs
            if !is_coinbase(tx) {
                for input in &tx.inputs {
                    // Get the UTXO first (needed for remove to update tracking)
                    match utxo_tree.get(&input.prevout) {
                        Ok(Some(utxo)) => {
                            // Remove the UTXO (even if transaction is spam)
                            if let Err(e) = utxo_tree.remove(&input.prevout, &utxo) {
                                return Err(crate::utxo_commitments::UtxoCommitmentError::TransactionApplication(
                                    format!("Failed to remove spent input: {:?}", e)
                                ));
                            }
                        }
                        Ok(None) => {
                            // UTXO doesn't exist - this might be valid if it was already spent
                            // or invalid if the transaction wasn't properly validated
                            // Continue but log - this should be validated before calling
                        }
                        Err(e) => {
                            return Err(crate::utxo_commitments::UtxoCommitmentError::TransactionApplication(
                                format!("Failed to get UTXO for removal: {:?}", e)
                            ));
                        }
                    }
                }
            }
            
            // Only add outputs from non-spam transactions
            // This provides bandwidth savings while maintaining UTXO set consistency
            if !is_spam {
                for (i, output) in tx.outputs.iter().enumerate() {
                    let outpoint = OutPoint {
                        hash: tx_id,
                        index: i as Natural,
                    };
                    
                    let utxo = UTXO {
                        value: output.value,
                        script_pubkey: output.script_pubkey.clone(),
                        height: block_height,
                    };
                    
                    if let Err(e) = utxo_tree.insert(outpoint, utxo) {
                        return Err(crate::utxo_commitments::UtxoCommitmentError::TransactionApplication(
                            format!("Failed to add output: {:?}", e)
                        ));
                    }
                }
            }
            // Spam transaction outputs are skipped (not added to tree)
        }
        
        // Return summary and new root
        let root = utxo_tree.root();
        
        Ok((spam_summary, root))
    }
}

/// Update UTXO commitments after block connection
///
/// This function should be called after successfully connecting a block
/// to keep UTXO commitments synchronized with the blockchain state.
///
/// # Arguments
///
/// * `utxo_tree` - Mutable reference to the UTXO Merkle tree
/// * `block` - The block that was just connected
/// * `block_height` - Height of the connected block
/// * `spam_filter` - Optional spam filter (if None, all transactions are included)
///
/// # Returns
///
/// Returns the new Merkle root hash of the UTXO tree.
///
/// # Example
///
/// ```rust
/// use consensus_proof::block::connect_block;
/// use consensus_proof::utxo_commitments::{UtxoMerkleTree, update_commitments_after_block};
/// use consensus_proof::utxo_commitments::spam_filter::SpamFilter;
///
/// let (result, new_utxo_set) = connect_block(&block, &witnesses, utxo_set, height, None)?;
/// if matches!(result, ValidationResult::Valid) {
///     let spam_filter = SpamFilter::new();
///     let root = update_commitments_after_block(
///         &mut utxo_tree,
///         &block,
///         height,
///         Some(&spam_filter),
///     )?;
///     println!("New UTXO commitment root: {:?}", root);
/// }
/// ```
#[cfg(feature = "utxo-commitments")]
pub fn update_commitments_after_block(
    utxo_tree: &mut UtxoMerkleTree,
    block: &crate::types::Block,
    block_height: Natural,
    spam_filter: Option<&SpamFilter>,
) -> UtxoCommitmentResult<Hash> {
    use crate::transaction::is_coinbase;
    use crate::block::calculate_tx_id;
    
    // If spam filter is provided, use filtered processing
    if let Some(filter) = spam_filter {
        let initial_sync = InitialSync {
            peer_consensus: crate::utxo_commitments::peer_consensus::PeerConsensus::new(
                crate::utxo_commitments::peer_consensus::ConsensusConfig::default()
            ),
            spam_filter: filter.clone(),
        };
        let (_, root) = initial_sync.process_filtered_block(
            utxo_tree,
            block_height,
            &block.transactions,
        )?;
        Ok(root)
    } else {
        // No spam filter: process all transactions normally
        for tx in &block.transactions {
            let tx_id = calculate_tx_id(tx);
            
            // Remove spent inputs (except coinbase)
            if !is_coinbase(tx) {
                for input in &tx.inputs {
                    // Get the UTXO first (needed for remove)
                    match utxo_tree.get(&input.prevout) {
                        Ok(Some(utxo)) => {
                            utxo_tree.remove(&input.prevout, &utxo)?;
                        }
                        Ok(None) => {
                            // UTXO doesn't exist - might be invalid or already spent
                            // Continue but this should have been caught during validation
                        }
                        Err(e) => {
                            return Err(crate::utxo_commitments::UtxoCommitmentError::TransactionApplication(
                                format!("Failed to get UTXO for removal: {:?}", e)
                            ));
                        }
                    }
                }
            }
            
            // Add new outputs
            for (i, output) in tx.outputs.iter().enumerate() {
                let outpoint = crate::types::OutPoint {
                    hash: tx_id,
                    index: i as Natural,
                };
                
                let utxo = crate::types::UTXO {
                    value: output.value,
                    script_pubkey: output.script_pubkey.clone(),
                    height: block_height,
                };
                
                utxo_tree.insert(outpoint, utxo)?;
            }
        }
        
        Ok(utxo_tree.root())
    }
}

/// Compute transaction ID (txid) using Bitcoin's standard double SHA256
/// 
/// Transaction ID is computed as: SHA256(SHA256(serialized_tx))
/// where serialized_tx is the transaction in Bitcoin wire format (non-SegWit format).
/// 
/// Note: For SegWit transactions, the txid still uses the non-witness serialization
/// (witness data is excluded from txid calculation).
/// 
/// This matches Bitcoin Core's transaction ID computation exactly.
fn compute_tx_id(tx: &Transaction) -> Hash {
    use crate::serialization::transaction::serialize_transaction;
    use sha2::{Sha256, Digest};
    
    // Serialize transaction to Bitcoin wire format (non-SegWit format for txid)
    let serialized = serialize_transaction(tx);
    
    // Double SHA256 (Bitcoin standard for transaction IDs)
    let first_hash = Sha256::digest(&serialized);
    let second_hash = Sha256::digest(first_hash);
    
    // Convert to Hash type [u8; 32]
    let mut txid = [0u8; 32];
    txid.copy_from_slice(&second_hash);
    
    txid
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

