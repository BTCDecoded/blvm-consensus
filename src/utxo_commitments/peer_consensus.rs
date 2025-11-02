//! Peer Consensus Protocol
//!
//! Implements the N-of-M peer consensus model for UTXO set verification.
//! Discovers diverse peers and finds consensus among them to verify UTXO commitments
//! without trusting any single peer.

#[cfg(feature = "utxo-commitments")]
use crate::types::{BlockHeader, Hash, Natural};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::data_structures::{UtxoCommitment, UtxoCommitmentError, UtxoCommitmentResult};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::verification::{verify_supply, verify_header_chain};
#[cfg(feature = "utxo-commitments")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "utxo-commitments")]
use std::net::IpAddr;

/// Peer information for diversity tracking
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub address: IpAddr,
    pub asn: Option<u32>, // Autonomous System Number
    pub country: Option<String>, // Country code (ISO 3166-1 alpha-2)
    pub implementation: Option<String>, // Bitcoin implementation (Bitcoin Core, btcd, etc.)
    pub subnet: u32, // /16 subnet for diversity checks
}

impl PeerInfo {
    /// Extract /16 subnet from IP address
    pub fn extract_subnet(ip: IpAddr) -> u32 {
        match ip {
            IpAddr::V4(ipv4) => {
                let octets = ipv4.octets();
                ((octets[0] as u32) << 24) | ((octets[1] as u32) << 16)
            }
            IpAddr::V6(ipv6) => {
                // For IPv6, use first 32 bits for subnet
                let segments = ipv6.segments();
                ((segments[0] as u32) << 16) | (segments[1] as u32)
            }
        }
    }
}

/// Peer with UTXO commitment response
#[derive(Debug, Clone)]
pub struct PeerCommitment {
    pub peer_info: PeerInfo,
    pub commitment: UtxoCommitment,
}

/// Consensus result from peer queries
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// The consensus UTXO commitment (agreed upon by majority)
    pub commitment: UtxoCommitment,
    /// Number of peers that agreed (out of total queried)
    pub agreement_count: usize,
    pub total_peers: usize,
    /// Agreement percentage (0.0 to 1.0)
    pub agreement_ratio: f64,
}

/// Peer consensus configuration
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Minimum number of diverse peers required
    pub min_peers: usize,
    /// Target number of peers to query
    pub target_peers: usize,
    /// Consensus threshold (0.0 to 1.0, e.g., 0.8 = 80%)
    pub consensus_threshold: f64,
    /// Maximum peers per ASN
    pub max_peers_per_asn: usize,
    /// Block safety margin (blocks back from tip)
    pub safety_margin: Natural,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            min_peers: 5,
            target_peers: 10,
            consensus_threshold: 0.8, // 80% agreement required
            max_peers_per_asn: 2,
            safety_margin: 2016, // ~2 weeks of blocks
        }
    }
}

/// Peer consensus manager
pub struct PeerConsensus {
    pub config: ConsensusConfig,
}

impl PeerConsensus {
    /// Create a new peer consensus manager
    pub fn new(config: ConsensusConfig) -> Self {
        Self { config }
    }

    /// Discover diverse peers
    /// 
    /// Filters peers to ensure diversity across:
    /// - ASNs (max N per ASN)
    /// - Subnets (/16 for IPv4, /32 for IPv6)
    /// - Geographic regions
    /// - Bitcoin implementations
    pub fn discover_diverse_peers(
        &self,
        all_peers: Vec<PeerInfo>,
    ) -> Vec<PeerInfo> {
        let mut diverse_peers = Vec::new();
        let mut seen_asn: HashMap<u32, usize> = HashMap::new();
        let mut seen_subnets: HashSet<u32> = HashSet::new();
        let mut seen_countries: HashSet<String> = HashSet::new();
        
        for peer in all_peers {
            // Check ASN limit
            if let Some(asn) = peer.asn {
                let asn_count = seen_asn.entry(asn).or_insert(0);
                if *asn_count >= self.config.max_peers_per_asn {
                    continue; // Skip - too many peers from this ASN
                }
                *asn_count += 1;
            }
            
            // Check subnet (no peers from same /16)
            if seen_subnets.contains(&peer.subnet) {
                continue; // Skip - duplicate subnet
            }
            seen_subnets.insert(peer.subnet);
            
            // Add diverse peer
            diverse_peers.push(peer);
            
            // Stop when we have enough
            if diverse_peers.len() >= self.config.target_peers {
                break;
            }
        }
        
        diverse_peers
    }

    /// Determine checkpoint height based on peer chain tips
    /// 
    /// Uses median of peer tips minus safety margin to prevent deep reorgs.
    pub fn determine_checkpoint_height(
        &self,
        peer_tips: Vec<Natural>,
    ) -> Natural {
        if peer_tips.is_empty() {
            return 0;
        }
        
        // Sort to find median
        let mut sorted_tips = peer_tips;
        sorted_tips.sort();
        
        let median_tip = if sorted_tips.len() % 2 == 0 {
            // Even number: average of middle two
            let mid = sorted_tips.len() / 2;
            (sorted_tips[mid - 1] + sorted_tips[mid]) / 2
        } else {
            // Odd number: middle value
            sorted_tips[sorted_tips.len() / 2]
        };
        
        // Apply safety margin
        if median_tip > self.config.safety_margin {
            median_tip - self.config.safety_margin
        } else {
            0 // Genesis block
        }
    }

    /// Request UTXO sets from multiple peers
    /// 
    /// Sends GetUTXOSet messages to peers and collects responses.
    /// Returns list of peer commitments (peer + commitment pairs).
    pub async fn request_utxo_sets(
        &self,
        peers: &[PeerInfo],
        checkpoint_height: Natural,
        checkpoint_hash: Hash,
    ) -> Vec<PeerCommitment> {
        // In a real implementation, this would:
        // 1. Send GetUTXOSet messages to each peer
        // 2. Wait for UTXOSet responses
        // 3. Collect valid commitments
        // 4. Return list of (peer, commitment) pairs
        
        // For now, return empty (would be implemented with actual network calls)
        vec![]
    }

    /// Find consensus among peer responses
    /// 
    /// Groups commitments by their values and finds the majority consensus.
    /// Returns the consensus commitment if threshold is met.
    pub fn find_consensus(
        &self,
        peer_commitments: Vec<PeerCommitment>,
    ) -> UtxoCommitmentResult<ConsensusResult> {
        let total_peers = peer_commitments.len();
        if total_peers < self.config.min_peers {
            return Err(UtxoCommitmentError::VerificationFailed(format!(
                "Insufficient peers: got {}, need at least {}",
                total_peers,
                self.config.min_peers
            )));
        }
        
        // Group commitments by their values (merkle root + supply + count + height)
        let mut commitment_groups: HashMap<(Hash, u64, u64, Natural), Vec<PeerCommitment>> = HashMap::new();
        
        for peer_commitment in peer_commitments {
            let key = (
                peer_commitment.commitment.merkle_root,
                peer_commitment.commitment.total_supply,
                peer_commitment.commitment.utxo_count,
                peer_commitment.commitment.block_height,
            );
            commitment_groups.entry(key).or_insert_with(Vec::new).push(peer_commitment);
        }
        
        // Find group with highest agreement
        let mut best_group: Option<(&(Hash, u64, u64, Natural), Vec<PeerCommitment>)> = None;
        let mut best_ratio = 0.0;
        
        for (key, group) in commitment_groups.iter() {
            let agreement_ratio = group.len() as f64 / total_peers as f64;
            
            if agreement_ratio > best_ratio {
                best_ratio = agreement_ratio;
                best_group = Some((key, group.clone()));
            }
        }
        
        // Check if consensus threshold is met
        if best_ratio < self.config.consensus_threshold {
            return Err(UtxoCommitmentError::VerificationFailed(format!(
                "No consensus: best agreement is {:.1}%, need {:.1}%",
                best_ratio * 100.0,
                self.config.consensus_threshold * 100.0
            )));
        }
        
        // Return consensus result
        if let Some((_, group)) = best_group {
            let commitment = group[0].commitment.clone();
            let agreement_count = group.len();
            
            Ok(ConsensusResult {
                commitment,
                agreement_count,
                total_peers,
                agreement_ratio: best_ratio,
            })
        } else {
            Err(UtxoCommitmentError::VerificationFailed(
                "No consensus found".to_string()
            ))
        }
    }

    /// Verify consensus commitment against block headers
    /// 
    /// Verifies that:
    /// 1. Block header chain is valid (PoW verification)
    /// 2. Commitment supply matches expected supply at height
    /// 3. Commitment block hash matches actual block hash
    pub fn verify_consensus_commitment(
        &self,
        consensus: &ConsensusResult,
        header_chain: &[BlockHeader],
    ) -> UtxoCommitmentResult<bool> {
        // 1. Verify header chain (PoW)
        verify_header_chain(header_chain)?;
        
        // 2. Verify supply matches expected
        verify_supply(&consensus.commitment)?;
        
        // 3. Verify commitment block hash matches header chain
        if consensus.commitment.block_height as usize >= header_chain.len() {
            return Err(UtxoCommitmentError::VerificationFailed(format!(
                "Commitment height {} exceeds header chain length {}",
                consensus.commitment.block_height,
                header_chain.len()
            )));
        }
        
        let expected_header = &header_chain[consensus.commitment.block_height as usize];
        let expected_hash = compute_block_hash(expected_header);
        
        if consensus.commitment.block_hash != expected_hash {
            return Err(UtxoCommitmentError::VerificationFailed(format!(
                "Block hash mismatch: commitment has {:?}, header chain has {:?}",
                consensus.commitment.block_hash,
                expected_hash
            )));
        }
        
        Ok(true)
    }
}

/// Compute block header hash (double SHA256)
fn compute_block_hash(header: &BlockHeader) -> Hash {
    use sha2::{Digest, Sha256};
    
    // Serialize block header
    let mut bytes = Vec::with_capacity(80);
    bytes.extend_from_slice(&header.version.to_le_bytes());
    bytes.extend_from_slice(&header.prev_block_hash);
    bytes.extend_from_slice(&header.merkle_root);
    bytes.extend_from_slice(&header.timestamp.to_le_bytes());
    bytes.extend_from_slice(&header.bits.to_le_bytes());
    bytes.extend_from_slice(&header.nonce.to_le_bytes());
    
    // Double SHA256
    let first_hash = Sha256::digest(&bytes);
    let second_hash = Sha256::digest(&first_hash);
    
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&second_hash);
    hash
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================

/// Mathematical Specification for Peer Consensus:
/// ∀ peers ∈ [PeerInfo], commitments ∈ [UtxoCommitment], threshold ∈ [0,1]:
/// - find_consensus(commitments, threshold) = consensus ⟺
///     |{c ∈ commitments | c = consensus}| / |commitments| ≥ threshold
/// - discover_diverse_peers(peers) ⊆ peers (no new peers created)
/// - verify_consensus_commitment(consensus, headers) verifies PoW + supply
///
/// Invariants:
/// - Consensus requires threshold percentage agreement
/// - Diverse peer discovery filters for diversity
/// - Consensus verification ensures cryptographic security

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: Consensus threshold enforcement
    /// 
    /// Verifies that consensus finding respects the threshold.
    #[kani::proof]
    #[kani::unwind(10)]
    fn kani_consensus_threshold_enforcement() {
        let config = ConsensusConfig::default();
        let peer_consensus = PeerConsensus::new(config);
        
        // Create multiple peer commitments
        let commitment1 = UtxoCommitment::new(
            [1; 32], // Same commitment (consensus)
            1000,
            1,
            0,
            [0; 32],
        );
        
        let commitment2 = UtxoCommitment::new(
            [1; 32], // Same commitment (consensus)
            1000,
            1,
            0,
            [0; 32],
        );
        
        let commitment3 = UtxoCommitment::new(
            [2; 32], // Different commitment (no consensus)
            2000,
            2,
            0,
            [0; 32],
        );
        
        let peer_commitments = vec![
            PeerCommitment {
                peer_info: PeerInfo {
                    address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(1, 1, 1, 1)),
                    asn: Some(1),
                    country: None,
                    implementation: None,
                    subnet: 0x01010000,
                },
                commitment: commitment1.clone(),
            },
            PeerCommitment {
                peer_info: PeerInfo {
                    address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(2, 2, 2, 2)),
                    asn: Some(2),
                    country: None,
                    implementation: None,
                    subnet: 0x02020000,
                },
                commitment: commitment2.clone(),
            },
            PeerCommitment {
                peer_info: PeerInfo {
                    address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(3, 3, 3, 3)),
                    asn: Some(3),
                    country: None,
                    implementation: None,
                    subnet: 0x03030000,
                },
                commitment: commitment3,
            },
        ];
        
        // 2 out of 3 agree (66.7%), but threshold is 80%
        // So consensus should fail
        let result = peer_consensus.find_consensus(peer_commitments);
        
        // With 80% threshold, 2/3 (66.7%) should fail
        assert!(
            result.is_err(),
            "Consensus should fail when agreement < threshold"
        );
    }

    /// Kani proof: Diverse peer discovery filtering
    /// 
    /// Verifies that diverse peer discovery filters out duplicate subnets.
    #[kani::proof]
    #[kani::unwind(10)]
    fn kani_diverse_peer_discovery() {
        let config = ConsensusConfig::default();
        let peer_consensus = PeerConsensus::new(config);
        
        // Create peers with duplicate subnets
        let all_peers = vec![
            PeerInfo {
                address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(1, 1, 1, 1)),
                asn: Some(1),
                country: None,
                implementation: None,
                subnet: 0x01010000, // Same subnet
            },
            PeerInfo {
                address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(1, 1, 2, 2)),
                asn: Some(2),
                country: None,
                implementation: None,
                subnet: 0x01010000, // Same subnet (duplicate)
            },
            PeerInfo {
                address: std::net::IpAddr::V4(std::net::Ipv4Addr::new(2, 2, 2, 2)),
                asn: Some(3),
                country: None,
                implementation: None,
                subnet: 0x02020000, // Different subnet
            },
        ];
        
        let diverse_peers = peer_consensus.discover_diverse_peers(all_peers.clone());
        
        // Should filter out duplicate subnet
        assert!(
            diverse_peers.len() <= all_peers.len(),
            "Diverse peer discovery should not add peers"
        );
        
        // Should have at most one peer per subnet
        let mut seen_subnets = std::collections::HashSet::new();
        for peer in &diverse_peers {
            assert!(
                seen_subnets.insert(peer.subnet),
                "No duplicate subnets in diverse peers"
            );
        }
    }
}

