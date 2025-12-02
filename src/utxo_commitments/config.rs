//! Configuration for UTXO Commitments Module
//!
//! Provides configuration management for:
//! - Peer consensus thresholds
//! - Spam filter settings
//! - Sync mode selection
//! - Verification levels

#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::peer_consensus::ConsensusConfig;
use crate::spam_filter::{SpamFilterConfig, SpamFilterConfigSerializable};
use serde::{Deserialize, Serialize};

/// Sync mode selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncMode {
    /// Use peer consensus for initial sync (fast, trusts N of M peers)
    PeerConsensus,
    /// Sync from genesis (slow, but no trust required)
    Genesis,
    /// Hybrid: Use peer consensus but verify from genesis in background
    Hybrid,
}

/// Verification level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationLevel {
    /// Minimal verification (peer consensus only)
    Minimal,
    /// Standard verification (peer consensus + PoW + supply checks)
    Standard,
    /// Paranoid verification (all checks + background genesis verification)
    Paranoid,
}

/// Storage preferences for UTXO commitments
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Keep filtered blocks (for debugging/analysis)
    pub keep_filtered_blocks: bool,
    /// Keep spam summary statistics
    pub keep_spam_summary: bool,
    /// Keep full UTXO set history (for verification)
    pub keep_utxo_history: bool,
    /// Maximum age for filtered blocks (days)
    pub filtered_blocks_max_age_days: u32,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            keep_filtered_blocks: false,
            keep_spam_summary: true,
            keep_utxo_history: false,
            filtered_blocks_max_age_days: 30,
        }
    }
}

/// Complete configuration for UTXO commitments module
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtxoCommitmentsConfig {
    /// Sync mode
    pub sync_mode: SyncMode,
    /// Verification level
    pub verification_level: VerificationLevel,
    /// Peer consensus configuration
    pub consensus: ConsensusConfigSerializable,
    /// Spam filter configuration
    pub spam_filter: crate::spam_filter::SpamFilterConfigSerializable,
    /// Storage preferences
    pub storage: StorageConfig,
}

/// Serializable version of ConsensusConfig
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsensusConfigSerializable {
    pub min_peers: usize,
    pub target_peers: usize,
    pub consensus_threshold: f64,
    pub max_peers_per_asn: usize,
    pub safety_margin: u64,
}

impl From<ConsensusConfigSerializable> for ConsensusConfig {
    fn from(serializable: ConsensusConfigSerializable) -> Self {
        ConsensusConfig {
            min_peers: serializable.min_peers,
            target_peers: serializable.target_peers,
            consensus_threshold: serializable.consensus_threshold,
            max_peers_per_asn: serializable.max_peers_per_asn,
            safety_margin: serializable.safety_margin,
        }
    }
}

impl From<ConsensusConfig> for ConsensusConfigSerializable {
    fn from(config: ConsensusConfig) -> Self {
        ConsensusConfigSerializable {
            min_peers: config.min_peers,
            target_peers: config.target_peers,
            consensus_threshold: config.consensus_threshold,
            max_peers_per_asn: config.max_peers_per_asn,
            safety_margin: config.safety_margin,
        }
    }
}

/// Serializable version of SpamFilterConfig
// SpamFilterConfigSerializable moved to spam_filter module

impl Default for UtxoCommitmentsConfig {
    fn default() -> Self {
        Self {
            sync_mode: SyncMode::PeerConsensus,
            verification_level: VerificationLevel::Standard,
            consensus: ConsensusConfigSerializable {
                min_peers: 5,
                target_peers: 10,
                consensus_threshold: 0.8,
                max_peers_per_asn: 2,
                safety_margin: 2016,
            },
            spam_filter: crate::spam_filter::SpamFilterConfigSerializable {
                filter_ordinals: true,
                filter_dust: true,
                filter_brc20: true,
                filter_large_witness: true,
                filter_low_fee_rate: false,
                filter_high_size_value_ratio: true,
                filter_many_small_outputs: true,
                dust_threshold: 546,
                min_output_value: 546,
                min_fee_rate: 1,
                max_witness_size: 1000,
                max_size_value_ratio: 1000.0,
                max_small_outputs: 10,
            },
            storage: StorageConfig::default(),
        }
    }
}

impl UtxoCommitmentsConfig {
    /// Load configuration from JSON file
    pub fn from_json_file(path: &std::path::Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        serde_json::from_str(&content).map_err(|e| format!("Failed to parse config JSON: {}", e))
    }

    /// Save configuration to JSON file
    pub fn to_json_file(&self, path: &std::path::Path) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        std::fs::write(path, content).map_err(|e| format!("Failed to write config file: {}", e))
    }

    /// Create default configuration file template
    pub fn create_default_config_file(path: &std::path::Path) -> Result<(), String> {
        let default_config = Self::default();
        default_config.to_json_file(path)
    }

    /// Convert to ConsensusConfig
    pub fn to_consensus_config(&self) -> ConsensusConfig {
        self.consensus.clone().into()
    }

    /// Convert to SpamFilterConfig
    pub fn to_spam_filter_config(&self) -> SpamFilterConfig {
        self.spam_filter.clone().into()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate consensus config
        if self.consensus.min_peers == 0 {
            return Err("min_peers must be > 0".to_string());
        }
        if self.consensus.target_peers < self.consensus.min_peers {
            return Err("target_peers must be >= min_peers".to_string());
        }
        if self.consensus.consensus_threshold < 0.0 || self.consensus.consensus_threshold > 1.0 {
            return Err("consensus_threshold must be between 0.0 and 1.0".to_string());
        }
        if self.consensus.max_peers_per_asn == 0 {
            return Err("max_peers_per_asn must be > 0".to_string());
        }

        // Validate spam filter config
        if self.spam_filter.dust_threshold < 0 {
            return Err("dust_threshold must be >= 0".to_string());
        }
        if self.spam_filter.min_output_value < 0 {
            return Err("min_output_value must be >= 0".to_string());
        }

        Ok(())
    }
}
