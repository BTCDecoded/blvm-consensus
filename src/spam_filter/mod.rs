//! Spam Filtering for UTXO Commitments
//!
//! Implements spam detection and filtering for Bitcoin transactions:
//! - Ordinals/Inscriptions detection
//! - Dust output filtering
//! - BRC-20 pattern detection
//! - Adaptive witness size thresholds based on script type
//!
//! This filter enables 40-60% bandwidth savings by skipping spam transactions
//! during ongoing sync while maintaining consensus correctness.
//!
//! **Critical Design Note**: Spam filtering applies to OUTPUTS only, not entire transactions.
//! When a spam transaction is processed:
//! - Its spent INPUTS are still removed from the UTXO tree (maintains consistency)
//! - Its OUTPUTS are filtered out (bandwidth savings)
//!
//! This ensures the UTXO tree remains consistent even when spam transactions spend
//! non-spam inputs. The `process_filtered_block` function in `initial_sync.rs` implements
//! this correctly by processing all transactions but only adding non-spam outputs.

mod script_analyzer;

use crate::opcodes::*;
use crate::types::{ByteString, Transaction, OutPoint, UTXO, UtxoSet};
use crate::witness::Witness;
use serde::{Deserialize, Serialize};
use script_analyzer::{ScriptType, TransactionType};

/// Default dust threshold (546 satoshis = 0.00000546 BTC)
pub const DEFAULT_DUST_THRESHOLD: i64 = 546;

/// Default minimum fee rate threshold (satoshis per vbyte)
/// Transactions with fee rate below this are suspicious
pub const DEFAULT_MIN_FEE_RATE: u64 = 1;

/// Default maximum witness size (bytes) - larger witness stacks suggest data embedding
pub const DEFAULT_MAX_WITNESS_SIZE: usize = 1000;

/// Default maximum transaction size to value ratio
/// Non-monetary transactions often have very large size relative to value transferred
pub const DEFAULT_MAX_SIZE_VALUE_RATIO: f64 = 1000.0; // bytes per satoshi

/// Spam filter preset configurations
///
/// Presets provide easy-to-use configurations for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpamFilterPreset {
    /// No spam filtering (all transactions pass)
    Disabled,
    /// Lenient filtering, minimal false positives
    /// - Higher thresholds
    /// - Only obvious spam patterns
    Conservative,
    /// Balanced filtering (default)
    /// - Standard thresholds
    /// - Comprehensive detection
    Moderate,
    /// Strong filtering, may have false positives
    /// - Lower thresholds
    /// - Aggressive detection
    Aggressive,
}

impl SpamFilterPreset {
    /// Convert preset to configuration
    pub fn to_config(&self) -> SpamFilterConfig {
        match self {
            Self::Disabled => SpamFilterConfig {
                filter_ordinals: false,
                filter_dust: false,
                filter_brc20: false,
                filter_large_witness: false,
                filter_low_fee_rate: false,
                filter_high_size_value_ratio: false,
                filter_many_small_outputs: false,
                ..SpamFilterConfig::default()
            },
            Self::Conservative => SpamFilterConfig {
                filter_ordinals: true,
                filter_dust: true,
                filter_brc20: true,
                filter_large_witness: true,
                filter_low_fee_rate: false,
                filter_high_size_value_ratio: true,
                filter_many_small_outputs: true,
                max_witness_size: 2000,  // Higher threshold
                max_size_value_ratio: 2000.0,  // Higher ratio
                max_small_outputs: 20,  // More lenient
                ..SpamFilterConfig::default()
            },
            Self::Moderate => SpamFilterConfig::default(),
            Self::Aggressive => SpamFilterConfig {
                filter_ordinals: true,
                filter_dust: true,
                filter_brc20: true,
                filter_large_witness: true,
                filter_low_fee_rate: true,  // Enable fee rate filtering
                filter_high_size_value_ratio: true,
                filter_many_small_outputs: true,
                max_witness_size: 500,  // Lower threshold
                max_size_value_ratio: 500.0,  // Lower ratio
                max_small_outputs: 5,  // More strict
                min_fee_rate: 2,  // Higher fee rate requirement
                ..SpamFilterConfig::default()
            },
        }
    }
}

/// Spam classification for a transaction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpamType {
    /// Ordinals/Inscriptions (data embedded in witness or script)
    Ordinals,
    /// Dust outputs (< threshold satoshis)
    Dust,
    /// BRC-20 token transactions
    BRC20,
    /// Large witness data (suggests data embedding in witness)
    LargeWitness,
    /// Low fee rate (suggests non-monetary use)
    LowFeeRate,
    /// High size-to-value ratio (large transaction, small value transfer)
    HighSizeValueRatio,
    /// Many small outputs (common in token/ordinal distribution)
    ManySmallOutputs,
    /// Not spam (valid transaction)
    NotSpam,
}

/// Adaptive witness size thresholds based on script type
///
/// These thresholds will be refined with real-world data collection.
/// For now, they use conservative estimates based on typical transaction patterns.
#[derive(Debug, Clone)]
pub struct WitnessSizeThresholds {
    /// Normal single-sig witness size (95th percentile)
    pub normal_single_sig: usize,
    /// Normal multi-sig witness size (95th percentile for 2-of-3)
    pub normal_multi_sig: usize,
    /// Normal P2WSH witness size (95th percentile)
    pub normal_p2wsh: usize,
    /// Suspicious threshold (current default)
    pub suspicious_threshold: usize,
    /// Definitely spam threshold (99.9th percentile)
    pub definitely_spam: usize,
}

impl Default for WitnessSizeThresholds {
    fn default() -> Self {
        // These will be populated from real-world data collection
        // For now, use conservative estimates
        Self {
            normal_single_sig: 200,
            normal_multi_sig: 500,
            normal_p2wsh: 800,
            suspicious_threshold: 1000,
            definitely_spam: 2000,
        }
    }
}

/// Witness element analysis result
#[derive(Debug, Clone)]
pub struct WitnessElementAnalysis {
    /// Total witness size (including varint overhead)
    pub total_size: usize,
    /// Number of witness elements
    pub element_count: usize,
    /// Number of large elements (> 200 bytes)
    pub large_elements: usize,
    /// Number of medium elements (100-200 bytes)
    pub medium_elements: usize,
    /// Number of small elements (< 100 bytes)
    pub small_elements: usize,
    /// Whether pattern suggests data splitting (many medium elements)
    pub suspicious_pattern: bool,
}

/// Spam filter configuration
#[derive(Debug, Clone)]
pub struct SpamFilterConfig {
    /// Filter Ordinals/Inscriptions
    pub filter_ordinals: bool,
    /// Filter dust outputs
    pub filter_dust: bool,
    /// Filter BRC-20 patterns
    pub filter_brc20: bool,
    /// Filter transactions with large witness data
    pub filter_large_witness: bool,
    /// Filter transactions with low fee rate
    pub filter_low_fee_rate: bool,
    /// Filter transactions with high size-to-value ratio
    pub filter_high_size_value_ratio: bool,
    /// Filter transactions with many small outputs
    pub filter_many_small_outputs: bool,
    /// Minimum output value to consider non-dust (satoshis)
    pub dust_threshold: i64,
    /// Minimum output value to include in filtered blocks (satoshis)
    pub min_output_value: i64,
    /// Minimum fee rate threshold (satoshis per vbyte)
    pub min_fee_rate: u64,
    /// Maximum witness size before flagging (bytes)
    /// Note: This is now adaptive based on script type when `use_adaptive_thresholds` is enabled
    pub max_witness_size: usize,
    /// Maximum size-to-value ratio (bytes per satoshi)
    pub max_size_value_ratio: f64,
    /// Maximum number of small outputs before flagging
    pub max_small_outputs: usize,
    
    // NEW: Adaptive thresholds
    /// Use adaptive witness size thresholds based on script type
    /// Default: true (enables data-driven thresholds)
    pub use_adaptive_thresholds: bool,
    /// Adaptive threshold configuration
    pub adaptive_thresholds: WitnessSizeThresholds,
    
    // NEW: Taproot-specific options
    /// Filter Taproot-specific spam patterns (control blocks, annexes)
    /// Default: true
    pub filter_taproot_spam: bool,
    /// Maximum Taproot control block size (bytes)
    /// Control blocks: 33 bytes base + 32 bytes per tree level
    /// BIP-110 limits to 257 bytes (depth 7), we use 289 bytes (depth 8) for policy
    /// Default: 289 bytes (allows depth 8, more lenient than BIP-110)
    pub max_taproot_control_size: usize,
    /// Reject Taproot annexes (last witness element starting with OP_RESERVED)
    /// Default: true
    pub reject_taproot_annexes: bool,
    
    // NEW: Total witness size check
    /// Filter transactions with large total witness size across all inputs
    /// Default: false (disabled by default, can be aggressive)
    pub filter_large_total_witness: bool,
    /// Maximum total witness size across all inputs (bytes)
    /// Default: 5000 bytes
    pub max_total_witness_size: usize,
    
    // NEW: Enhanced detection options
    /// Use improved envelope protocol detection (checks for OP_ENDIF)
    /// Default: true
    pub use_improved_envelope_detection: bool,
    /// Use JSON validation for BRC-20 detection (requires serde_json)
    /// Default: true (if serde_json available)
    pub use_json_validation_brc20: bool,
    
    // NEW: Fee rate calculation options
    /// Require UTXO set for fee rate calculation (reject if unavailable)
    /// If false, falls back to heuristic when UTXO set unavailable
    /// Default: false (use heuristic fallback)
    pub require_utxo_for_fee_rate: bool,
    /// Minimum fee rate for large transactions (satoshis per vbyte)
    /// Transactions larger than large_tx_threshold_bytes require this fee rate
    /// Default: 2 sat/vB (higher than standard 1 sat/vB)
    pub min_fee_rate_large_tx: u64,
    /// Large transaction threshold (bytes)
    /// Transactions larger than this require min_fee_rate_large_tx
    /// Default: 1000 bytes
    pub large_tx_threshold_bytes: usize,
}

impl Default for SpamFilterConfig {
    fn default() -> Self {
        Self {
            filter_ordinals: true,
            filter_dust: true,
            filter_brc20: true,
            filter_large_witness: true,
            filter_low_fee_rate: false, // Disabled by default (too aggressive)
            filter_high_size_value_ratio: true,
            filter_many_small_outputs: true,
            dust_threshold: DEFAULT_DUST_THRESHOLD,
            min_output_value: DEFAULT_DUST_THRESHOLD,
            min_fee_rate: DEFAULT_MIN_FEE_RATE,
            max_witness_size: DEFAULT_MAX_WITNESS_SIZE,
            max_size_value_ratio: DEFAULT_MAX_SIZE_VALUE_RATIO,
            max_small_outputs: 10, // Flag if more than 10 small outputs
            
            // NEW: Adaptive thresholds
            use_adaptive_thresholds: true, // Enable by default
            adaptive_thresholds: WitnessSizeThresholds::default(),
            
            // NEW defaults
            filter_taproot_spam: true,
            max_taproot_control_size: 289, // 33 + 32*8 (depth 8)
            reject_taproot_annexes: true,
            filter_large_total_witness: false, // Disabled by default (can be aggressive)
            max_total_witness_size: 5000,
            use_improved_envelope_detection: true,
            use_json_validation_brc20: true,
            require_utxo_for_fee_rate: false, // Use heuristic fallback
            min_fee_rate_large_tx: 2, // 2 sat/vB
            large_tx_threshold_bytes: 1000, // 1 KB
        }
    }
}

/// Spam filter result
#[derive(Debug, Clone)]
pub struct SpamFilterResult {
    /// Whether transaction is spam
    pub is_spam: bool,
    /// Primary spam type detected
    pub spam_type: SpamType,
    /// All detected spam types (transaction may match multiple)
    pub detected_types: Vec<SpamType>,
}

/// Spam filter implementation
#[derive(Clone)]
pub struct SpamFilter {
    config: SpamFilterConfig,
    #[cfg(feature = "production")]
    pub(crate) script_type_cache: std::sync::Arc<std::sync::RwLock<lru::LruCache<u64, bool>>>,
}

impl SpamFilter {
    /// Create a new spam filter with default configuration
    pub fn new() -> Self {
        Self {
            config: SpamFilterConfig::default(),
            #[cfg(feature = "production")]
            script_type_cache: std::sync::Arc::new(std::sync::RwLock::new(
                lru::LruCache::new(std::num::NonZeroUsize::new(10_000).unwrap())
            )),
        }
    }

    /// Create a new spam filter with custom configuration
    pub fn with_config(config: SpamFilterConfig) -> Self {
        Self { config }
    }
    
    /// Create a new spam filter with a preset configuration
    ///
    /// Presets provide easy-to-use configurations for common use cases:
    /// - `Disabled`: No spam filtering
    /// - `Conservative`: Lenient filtering, minimal false positives
    /// - `Moderate`: Balanced filtering (default)
    /// - `Aggressive`: Strong filtering, may have false positives
    pub fn with_preset(preset: SpamFilterPreset) -> Self {
        Self::with_config(preset.to_config())
    }

    /// Check if a transaction is spam (without witness data)
    ///
    /// This is the backward-compatible method. For better detection, use `is_spam_with_witness`.
    pub fn is_spam(&self, tx: &Transaction) -> SpamFilterResult {
        self.is_spam_with_witness(tx, None, None)
    }

    /// Check if a transaction is spam (with optional witness data and UTXO set)
    ///
    /// Witness data is required for detecting Taproot/SegWit-based Ordinals.
    /// UTXO set is optional but improves fee rate calculation accuracy.
    /// If witness data is not provided, detection will be less accurate.
    pub fn is_spam_with_witness(
        &self,
        tx: &Transaction,
        witnesses: Option<&[Witness]>,
        utxo_set: Option<&UtxoSet>,
    ) -> SpamFilterResult {
        let mut detected_types = Vec::new();

        // Check for Ordinals/Inscriptions (now with witness data support)
        if self.config.filter_ordinals && self.detect_ordinals(tx, witnesses) {
            detected_types.push(SpamType::Ordinals);
        }

        // Check for dust outputs
        if self.config.filter_dust && self.detect_dust(tx) {
            detected_types.push(SpamType::Dust);
        }

        // Check for BRC-20 patterns
        if self.config.filter_brc20 && self.detect_brc20(tx) {
            detected_types.push(SpamType::BRC20);
        }

        // Check for large witness data (now with adaptive thresholds)
        if self.config.filter_large_witness && self.detect_large_witness(tx, witnesses) {
            detected_types.push(SpamType::LargeWitness);
        }

        // Check for large total witness size (across all inputs)
        if self.config.filter_large_total_witness {
            if self.detect_large_total_witness(witnesses) {
                detected_types.push(SpamType::LargeWitness);
            }
        }

        // Check for low fee rate (requires fee calculation)
        if self.config.filter_low_fee_rate {
            if self.detect_low_fee_rate(tx, witnesses, utxo_set) {
                detected_types.push(SpamType::LowFeeRate);
            }
        }

        // Check for high size-to-value ratio
        if self.config.filter_high_size_value_ratio
            && self.detect_high_size_value_ratio(tx, witnesses)
        {
            detected_types.push(SpamType::HighSizeValueRatio);
        }

        // Check for many small outputs
        if self.config.filter_many_small_outputs && self.detect_many_small_outputs(tx) {
            detected_types.push(SpamType::ManySmallOutputs);
        }

        let is_spam = !detected_types.is_empty();
        let spam_type = detected_types.first().cloned().unwrap_or(SpamType::NotSpam);

        SpamFilterResult {
            is_spam,
            spam_type,
            detected_types,
        }
    }

    /// Filter a transaction based on spam detection
    ///
    /// Returns `Some(tx)` if transaction should be included (not spam),
    /// or `None` if transaction should be filtered (spam).
    pub fn filter_transaction(&self, tx: &Transaction) -> Option<Transaction> {
        let result = self.is_spam(tx);
        if result.is_spam {
            None // Filter out spam
        } else {
            Some(tx.clone()) // Include non-spam
        }
    }
    /// Detect Ordinals/Inscriptions in transaction
    ///
    /// Ordinals typically embed data in:
    /// - Witness scripts (SegWit v0 or Taproot) - PRIMARY METHOD
    /// - Script pubkey (OP_RETURN or data push)
    /// - Envelope protocol patterns
    fn detect_ordinals(&self, tx: &Transaction, witnesses: Option<&[Witness]>) -> bool {
        // Check outputs for OP_RETURN or data pushes (common Ordinals pattern)
        for output in &tx.outputs {
            if self.has_ordinal_pattern(&output.script_pubkey) {
                return true;
            }
        }

        // Check inputs for envelope protocol in scriptSig
        for input in &tx.inputs {
            if self.has_envelope_pattern(&input.script_sig) {
                return true;
            }
        }

        // Check witness data (PRIMARY METHOD for Taproot/SegWit Ordinals)
        if let Some(witnesses) = witnesses {
            for (i, witness) in witnesses.iter().enumerate() {
                if i >= tx.inputs.len() {
                    break;
                }

                // Check for Taproot-specific spam patterns
                if self.config.filter_taproot_spam {
                    // Check if any output in transaction is Taproot
                    // Note: This is simplified - full implementation would track which output is spent
                    for output in &tx.outputs {
                        if self.is_taproot_output(&output.script_pubkey) {
                            if self.detect_taproot_spam(output, witness) {
                                return true;
                            }
                        }
                    }
                }

                // Check for large witness stacks (common in Ordinals)
                // Use adaptive thresholds if enabled
                if self.config.use_adaptive_thresholds {
                    if self.has_large_witness_stack_adaptive(witness, tx, i) {
                        return true;
                    }
                } else {
                    if self.has_large_witness_stack(witness) {
                        return true;
                    }
                }

                // Check for data patterns in witness elements
                if self.has_witness_data_pattern(witness) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if output is Taproot (P2TR)
    ///
    /// P2TR format: OP_1 + PUSH_32_BYTES + 32-byte x-only pubkey = 34 bytes
    fn is_taproot_output(&self, script_pubkey: &ByteString) -> bool {
        // P2TR: OP_1 (0x51) + 0x20 + 32-byte x-only pubkey = 34 bytes
        script_pubkey.len() == 34 
            && script_pubkey[0] == crate::opcodes::OP_1 
            && script_pubkey[1] == 0x20
    }

    /// Detect Taproot-specific spam patterns
    ///
    /// Checks for:
    /// - Taproot annexes (last witness element starting with OP_RESERVED)
    /// - Large control blocks (script path spends with deep trees)
    fn detect_taproot_spam(
        &self,
        output: &crate::types::TransactionOutput,
        witness: &Witness,
    ) -> bool {
        if !self.is_taproot_output(&output.script_pubkey) {
            return false;
        }
        
        // Check for annex (last witness element starting with 0x50)
        // BIP-341: Annex is the last witness element if it starts with 0x50
        if self.config.reject_taproot_annexes {
            if let Some(last) = witness.last() {
                if !last.is_empty() && last[0] == 0x50 {
                    // Annex detected - BIP-110 invalidates these
                    return true;
                }
            }
        }
        
        // Check for large control blocks (script path spends)
        // Control blocks are typically the last element in Taproot script path spends
        // Format: 33 + 32*n bytes (where n is tree depth)
        // Large control blocks suggest deep trees (potential data embedding)
        if witness.len() >= 2 {
            // Script path spend: script + control block + witness items
            // Control block is typically the last element
            if let Some(control_block) = witness.last() {
                // Control block: 33 bytes base + 32 bytes per tree level
                // TAPROOT_CONTROL_BASE_SIZE = 33, TAPROOT_CONTROL_NODE_SIZE = 32
                // BIP-110 limits to TAPROOT_CONTROL_MAX_SIZE_REDUCED (257 bytes, depth 7)
                // For policy, we use a configurable threshold (289 bytes, depth 8)
                if control_block.len() > self.config.max_taproot_control_size {
                    return true;
                }
            }
        }
        
        false
    }

    /// Check if witness stack is suspiciously large (suggests data embedding)
    /// 
    /// Uses adaptive thresholds based on script type if enabled.
    fn has_large_witness_stack(&self, witness: &Witness) -> bool {
        let total_size = self.calculate_witness_size(witness);
        total_size > self.config.max_witness_size
    }

    /// Check if witness stack is suspiciously large using adaptive thresholds
    ///
    /// This method uses script type detection to apply appropriate thresholds.
    /// Falls back to fixed threshold if adaptive thresholds are disabled or script type cannot be determined.
    fn has_large_witness_stack_adaptive(
        &self,
        witness: &Witness,
        tx: &Transaction,
        _input_index: usize,
    ) -> bool {
        let total_size = self.calculate_witness_size(witness);
        
        // If adaptive thresholds disabled, use fixed threshold
        if !self.config.use_adaptive_thresholds {
            return total_size > self.config.max_witness_size;
        }

        // Try to detect script type from the output being spent
        // Note: This is simplified - full implementation would track which output is spent
        // For now, check all outputs in the transaction
        let mut detected_script_type: Option<ScriptType> = None;
        
        // Try to find the script type of the output being spent
        // In a real implementation, we'd track prevout -> output mapping
        for output in &tx.outputs {
            let script_type = ScriptType::detect(&output.script_pubkey);
            if script_type != ScriptType::Unknown {
                detected_script_type = Some(script_type);
                break; // Use first detected script type
            }
        }

        // Get threshold based on script type
        let threshold = if let Some(script_type) = detected_script_type {
            script_type.recommended_threshold()
        } else {
            // Fallback to fixed threshold if script type unknown
            self.config.max_witness_size
        };

        total_size > threshold
    }

    /// Analyze witness elements for suspicious patterns
    ///
    /// Detects data splitting patterns (many medium-sized elements).
    fn analyze_witness_elements(&self, witness: &Witness) -> WitnessElementAnalysis {
        let total_size = self.calculate_witness_size(witness);
        let element_count = witness.len();
        
        let mut large_elements = 0;
        let mut medium_elements = 0;
        let mut small_elements = 0;
        
        for element in witness {
            if element.len() > 200 {
                large_elements += 1;
            } else if element.len() >= 100 {
                medium_elements += 1;
            } else {
                small_elements += 1;
            }
        }
        
        // Suspicious pattern: many medium elements (suggests data splitting)
        let suspicious_pattern = medium_elements >= 10;
        
        WitnessElementAnalysis {
            total_size,
            element_count,
            large_elements,
            medium_elements,
            small_elements,
            suspicious_pattern,
        }
    }

    /// Calculate accurate witness size including varint overhead
    ///
    /// Witness size includes:
    /// - Stack count varint (1 byte typically for small stacks)
    /// - For each element: length varint (1-9 bytes) + element data
    ///
    /// This matches the actual serialized size of witness data in Bitcoin transactions.
    fn calculate_witness_size(&self, witness: &Witness) -> usize {
        // Stack count varint (typically 1 byte for small stacks)
        let mut size = 1;
        
        // Each element: length varint + element data
        for element in witness {
            // Varint encoding: 1 byte for <128, 2 for <16384, etc.
            // Bitcoin varint encoding: values < 0xfd use 1 byte, larger values use prefix + data
            // For witness element lengths, we use compact size encoding:
            // - < 0xfd: 1 byte
            // - 0xfd-0xffff: 0xfd prefix (1 byte) + 2 bytes data
            // - 0x10000-0xffffffff: 0xfe prefix (1 byte) + 4 bytes data
            // - > 0xffffffff: 0xff prefix (1 byte) + 8 bytes data
            size += if element.len() <= VARINT_1BYTE_MAX as usize {
                1
            } else if element.len() <= 0xffff {
                3 // VARINT_2BYTE_PREFIX + 2 bytes
            } else if element.len() <= 0xffffffff {
                5 // VARINT_4BYTE_PREFIX + 4 bytes
            } else {
                9 // VARINT_8BYTE_PREFIX + 8 bytes
            };
            size += element.len();
        }
        
        size
    }

    /// Check if witness contains data patterns (non-signature data)
    fn has_witness_data_pattern(&self, witness: &Witness) -> bool {
        if witness.is_empty() {
            return false;
        }

        // Check for very large witness elements (>520 bytes is max for signatures)
        // Elements larger than typical signature size suggest data embedding
        for element in witness {
            // Typical signatures are 71-73 bytes (DER-encoded) or 64 bytes (Schnorr)
            // Witness elements >200 bytes are suspicious for data embedding
            if element.len() > 200 {
                // Check if it looks like data (not a signature)
                // Signatures typically start with 0x30 (DER) or are exactly 64 bytes (Schnorr)
                if element.len() != 64 && (element.is_empty() || element[0] != DER_SIGNATURE_PREFIX) {
                    // Likely data embedding
                    return true;
                }
            }
        }

        // Check for multiple large elements (suggests data chunks)
        let large_elements = witness.iter().filter(|elem| elem.len() > 100).count();
        if large_elements >= 3 {
            return true;
        }

        // Check for suspicious pattern (many medium elements - data splitting)
        let analysis = self.analyze_witness_elements(witness);
        if analysis.suspicious_pattern {
            return true;
        }

        false
    }

    /// Check if script has Ordinals pattern
    ///
    /// Ordinals typically use:
    /// - OP_RETURN followed by data
    /// - Large data pushes
    /// - Envelope protocol markers
    fn has_ordinal_pattern(&self, script: &ByteString) -> bool {
        if script.is_empty() {
            return false;
        }

        // Check for OP_RETURN - common in Ordinals
        if script[0] == OP_RETURN {
            // OP_RETURN followed by data suggests Ordinals
            if script.len() > 80 {
                // Large data pushes are suspicious
                return true;
            }
        }

        // Check for envelope protocol pattern
        // Envelope protocol: OP_FALSE OP_IF ... OP_ENDIF
        // This is a simplified check - full implementation would parse script
        if script.len() > 100 {
            // Large scripts are often Ordinals
            // More sophisticated check would parse opcodes
            return true;
        }

        false
    }

    /// Check if script has envelope protocol pattern
    fn has_envelope_pattern(&self, script: &ByteString) -> bool {
        // Envelope protocol: OP_FALSE OP_IF ... OP_ENDIF
        if script.len() < 4 {
            return false;
        }

        // Check for OP_FALSE OP_IF pattern (common in inscriptions)
        if script[0] == OP_0 && script[1] == OP_IF {
            if self.config.use_improved_envelope_detection {
                // Improved: Verify OP_ENDIF exists later in script
                // Envelope protocol: OP_FALSE OP_IF ... OP_ENDIF
                if script.iter().skip(2).any(|&b| b == OP_ENDIF) {
                    return true;
                }
            } else {
                // Original simple check (backward compatibility)
                return true;
            }
        }

        false
    }

    /// Detect dust outputs
    ///
    /// Dust outputs are outputs with value below threshold (default: 546 satoshis).
    fn detect_dust(&self, tx: &Transaction) -> bool {
        // Check if all outputs are below threshold
        let mut all_dust = true;

        for output in &tx.outputs {
            if output.value >= self.config.dust_threshold {
                all_dust = false;
                break;
            }
        }

        all_dust && !tx.outputs.is_empty()
    }

    /// Detect transactions with large witness data
    ///
    /// Large witness stacks often indicate data embedding (Ordinals, inscriptions).
    /// Now uses adaptive thresholds based on script type.
    fn detect_large_witness(&self, tx: &Transaction, witnesses: Option<&[Witness]>) -> bool {
        if let Some(witnesses) = witnesses {
            for (i, witness) in witnesses.iter().enumerate() {
                // Use adaptive thresholds if enabled
                if self.config.use_adaptive_thresholds {
                    if self.has_large_witness_stack_adaptive(witness, tx, i) {
                        return true;
                    }
                } else {
                    if self.has_large_witness_stack(witness) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Detect transactions with low fee rate
    ///
    /// Non-monetary transactions often pay minimal fees relative to size.
    /// Now accepts optional UTXO set for accurate fee calculation.
    fn detect_low_fee_rate(
        &self,
        tx: &Transaction,
        witnesses: Option<&[Witness]>,
        utxo_set: Option<&UtxoSet>,
    ) -> bool {
        let tx_size = self.estimate_transaction_size_with_witness(tx, witnesses);
        
        // If require_utxo_for_fee_rate is true and UTXO set unavailable, reject
        if self.config.require_utxo_for_fee_rate && utxo_set.is_none() {
            // Cannot calculate accurate fee rate, reject if strict mode enabled
            return true; // Reject as spam (conservative)
        }
        
        // Calculate fee rate
        let fee_rate = if let Some(utxo_set) = utxo_set {
            // Accurate calculation with UTXO set
            self.calculate_fee_rate_accurate(tx, utxo_set, tx_size)
        } else {
            // Fallback to heuristic when UTXO set unavailable
            self.calculate_fee_rate_heuristic(tx, tx_size)
        };
        
        // Check against threshold (use large tx threshold if applicable)
        let threshold = if tx_size > self.config.large_tx_threshold_bytes {
            self.config.min_fee_rate_large_tx
        } else {
            self.config.min_fee_rate
        };
        
        fee_rate < threshold
    }

    /// Calculate fee rate accurately using UTXO set
    fn calculate_fee_rate_accurate(
        &self,
        tx: &Transaction,
        utxo_set: &UtxoSet,
        tx_size: usize,
    ) -> u64 {
        if tx_size == 0 {
            return 0;
        }
        
        // Calculate actual fee
        let mut input_total = 0u64;
        for input in &tx.inputs {
            if let Some(utxo) = utxo_set.get(&input.prevout) {
                input_total += utxo.value as u64;
            }
        }
        
        let output_total: u64 = tx.outputs.iter().map(|out| out.value as u64).sum();
        let fee = input_total.saturating_sub(output_total);
        
        // Fee rate in satoshis per vbyte
        if tx_size > 0 {
            fee / tx_size as u64
        } else {
            0
        }
    }

    /// Calculate fee rate using heuristics (fallback)
    fn calculate_fee_rate_heuristic(&self, tx: &Transaction, tx_size: usize) -> u64 {
        if tx_size == 0 {
            return 0;
        }
        
        let total_output_value: i64 = tx.outputs.iter().map(|out| out.value).sum();
        
        // Heuristic: large transactions with small output value likely have low fee rate
        if tx_size > 1000 && total_output_value < 10000 {
            // Assume minimal fee (1000 sats) for large transactions
            1000u64.saturating_div(tx_size as u64)
        } else {
            // For other transactions, assume reasonable fee rate
            // This is conservative - may have false negatives
            self.config.min_fee_rate
        }
    }

    /// Detect transactions with large total witness size across all inputs
    fn detect_large_total_witness(&self, witnesses: Option<&[Witness]>) -> bool {
        if !self.config.filter_large_total_witness {
            return false; // Feature disabled
        }
        
        if let Some(witnesses) = witnesses {
            let total_size: usize = witnesses
                .iter()
                .map(|w| self.calculate_witness_size(w))
                .sum();
            
            total_size > self.config.max_total_witness_size
        } else {
            false
        }
    }

    /// Detect transactions with high size-to-value ratio
    ///
    /// Non-monetary transactions often have very large size relative to value transferred.
    /// Now uses transaction type detection to adjust thresholds for legitimate transactions
    /// (consolidations, CoinJoins) that legitimately have high ratios.
    fn detect_high_size_value_ratio(
        &self,
        tx: &Transaction,
        witnesses: Option<&[Witness]>,
    ) -> bool {
        let tx_size = self.estimate_transaction_size_with_witness(tx, witnesses) as f64;
        let total_output_value: f64 = tx.outputs.iter().map(|out| out.value as f64).sum();

        // Avoid division by zero
        if total_output_value <= 0.0 {
            // Transaction with zero outputs is suspicious
            return tx_size > 1000.0;
        }

        let ratio = tx_size / total_output_value;
        
        // Use transaction type to adjust threshold
        let threshold = if self.config.use_adaptive_thresholds {
            let tx_type = TransactionType::detect(tx);
            tx_type.recommended_size_value_ratio()
        } else {
            self.config.max_size_value_ratio
        };
        
        ratio > threshold
    }

    /// Detect transactions with many small outputs
    ///
    /// Token distributions and Ordinal transfers often create many small outputs.
    fn detect_many_small_outputs(&self, tx: &Transaction) -> bool {
        let small_output_count = tx
            .outputs
            .iter()
            .filter(|out| out.value < self.config.dust_threshold)
            .count();

        small_output_count > self.config.max_small_outputs
    }

    /// Estimate transaction size including witness data
    fn estimate_transaction_size_with_witness(
        &self,
        tx: &Transaction,
        witnesses: Option<&[Witness]>,
    ) -> usize {
        // Base transaction size (non-witness)
        let base_size = estimate_transaction_size(tx) as usize;

        // Add witness size if available
        if let Some(witnesses) = witnesses {
            let witness_size: usize = witnesses
                .iter()
                .map(|witness| {
                    // Witness stack count (varint, ~1 byte)
                    let mut size = 1;
                    // Each witness element: length (varint, ~1 byte) + element data
                    for element in witness {
                        size += 1; // varint for length
                        size += element.len();
                    }
                    size
                })
                .sum();

            // SegWit marker and flag (2 bytes)
            let has_witness = witness_size > 0;
            if has_witness {
                base_size + 2 + witness_size
            } else {
                base_size
            }
        } else {
            base_size
        }
    }

    /// Detect BRC-20 token transactions
    ///
    /// BRC-20 transactions typically have:
    /// - OP_RETURN outputs with JSON data
    /// - Specific JSON patterns (mint, transfer, deploy)
    fn detect_brc20(&self, tx: &Transaction) -> bool {
        // Check outputs for OP_RETURN with JSON-like data
        for output in &tx.outputs {
            if self.has_brc20_pattern(&output.script_pubkey) {
                return true;
            }
        }

        false
    }

    /// Check if script has BRC-20 pattern
    ///
    /// BRC-20 transactions use OP_RETURN with JSON:
    /// - {"p":"brc-20","op":"mint",...}
    /// - {"p":"brc-20","op":"transfer",...}
    /// - {"p":"brc-20","op":"deploy",...}
    fn has_brc20_pattern(&self, script: &ByteString) -> bool {
        if script.len() < 20 {
            return false;
        }

        // Check for OP_RETURN
        if script[0] != OP_RETURN {
            return false;
        }

        // Extract data after OP_RETURN
        let data = &script[1..];
        
        // Try to decode as UTF-8
        let script_str = match String::from_utf8(data.to_vec()) {
            Ok(s) => s,
            Err(_) => {
                // Not valid UTF-8, use simple pattern matching
                return self.has_brc20_pattern_simple(data);
            }
        };

        // Use JSON validation if enabled
        if self.config.use_json_validation_brc20 {
            self.has_brc20_pattern_json(&script_str)
        } else {
            // Fallback to simple string matching
            self.has_brc20_pattern_simple(data)
        }
    }

    /// Check for BRC-20 pattern using JSON validation
    fn has_brc20_pattern_json(&self, json_str: &str) -> bool {
        // Remove whitespace for more robust matching
        let cleaned: String = json_str.chars().filter(|c| !c.is_whitespace()).collect();
        
        // Try to parse as JSON
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&cleaned) {
            // Check if it's a valid BRC-20 transaction
            if let Some(obj) = json_value.as_object() {
                // Check for protocol field: "p": "brc-20"
                if let Some(protocol) = obj.get("p") {
                    if protocol.as_str() == Some("brc-20") {
                        // Check for operation field: "op": "mint" | "transfer" | "deploy"
                        if let Some(op) = obj.get("op") {
                            if let Some(op_str) = op.as_str() {
                                return matches!(op_str, "mint" | "transfer" | "deploy");
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback: try parsing original string (with whitespace)
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(obj) = json_value.as_object() {
                if let Some(protocol) = obj.get("p") {
                    if protocol.as_str() == Some("brc-20") {
                        if let Some(op) = obj.get("op") {
                            if let Some(op_str) = op.as_str() {
                                return matches!(op_str, "mint" | "transfer" | "deploy");
                            }
                        }
                    }
                }
            }
        }
        
        false
    }

    /// Check for BRC-20 pattern using simple string matching (fallback)
    fn has_brc20_pattern_simple(&self, data: &[u8]) -> bool {
        // Convert to string for pattern matching
        if let Ok(script_str) = String::from_utf8(data.to_vec()) {
            // Check for BRC-20 markers (case-insensitive)
            let lower = script_str.to_lowercase();
            lower.contains("brc-20")
                || lower.contains("\"p\":\"brc-20\"")
                || lower.contains("op\":\"mint")
                || lower.contains("op\":\"transfer")
                || lower.contains("op\":\"deploy")
        } else {
            // Not valid UTF-8, try byte pattern matching
            // Look for "brc-20" in bytes (case-insensitive)
            let pattern = b"brc-20";
            let pattern_lower = b"BRC-20";
            data.windows(pattern.len()).any(|window| {
                window == pattern || window == pattern_lower
            })
        }
    }

    /// Filter transactions from a block (without witness data)
    ///
    /// Returns filtered transactions (non-spam only) and summary of filtered spam.
    ///
    /// **Important**: This function filters entire transactions. For UTXO commitment processing,
    /// use `process_filtered_block` in `initial_sync.rs` which correctly handles spam
    /// transactions by removing spent inputs while filtering outputs.
    ///
    /// This function is primarily used for:
    /// - Bandwidth estimation (calculating filtered size)
    /// - Statistics and reporting
    /// - Network message filtering (where entire transactions can be dropped)
    ///
    /// **Do not use this for UTXO tree updates** - it will cause UTXO set inconsistency
    /// when spam transactions spend non-spam inputs.
    pub fn filter_block(&self, transactions: &[Transaction]) -> (Vec<Transaction>, SpamSummary) {
        self.filter_block_with_witness(transactions, None)
    }

    /// Filter transactions from a block (with optional witness data)
    ///
    /// Returns filtered transactions (non-spam only) and summary of filtered spam.
    /// Witness data improves detection accuracy for SegWit/Taproot-based spam.
    ///
    /// **Important**: This function filters entire transactions. For UTXO commitment processing,
    /// use `process_filtered_block` in `initial_sync.rs` which correctly handles spam
    /// transactions by removing spent inputs while filtering outputs.
    ///
    /// This function is primarily used for:
    /// - Bandwidth estimation (calculating filtered size)
    /// - Statistics and reporting
    /// - Network message filtering (where entire transactions can be dropped)
    ///
    /// **Do not use this for UTXO tree updates** - it will cause UTXO set inconsistency
    /// when spam transactions spend non-spam inputs.
    pub fn filter_block_with_witness(
        &self,
        transactions: &[Transaction],
        witnesses: Option<&[Vec<Witness>]>,
    ) -> (Vec<Transaction>, SpamSummary) {
        let mut filtered_txs = Vec::new();
        let mut filtered_count = 0u32;
        let mut filtered_size = 0u64;
        let mut spam_breakdown = SpamBreakdown::default();

        for (i, tx) in transactions.iter().enumerate() {
            // Get witness data for this transaction if available
            let tx_witnesses = witnesses.and_then(|w| w.get(i));

            let result = if let Some(tx_witnesses) = tx_witnesses {
                self.is_spam_with_witness(tx, Some(tx_witnesses), None)
            } else {
                self.is_spam(tx)
            };

            if result.is_spam {
                filtered_count += 1;
                let tx_size = if let Some(tx_witnesses) = tx_witnesses {
                    self.estimate_transaction_size_with_witness(tx, Some(tx_witnesses)) as u64
                } else {
                    estimate_transaction_size(tx)
                };
                filtered_size += tx_size;

                // Update breakdown
                for spam_type in &result.detected_types {
                    match spam_type {
                        SpamType::Ordinals => spam_breakdown.ordinals += 1,
                        SpamType::Dust => spam_breakdown.dust += 1,
                        SpamType::BRC20 => spam_breakdown.brc20 += 1,
                        SpamType::LargeWitness => spam_breakdown.ordinals += 1, // Count as Ordinals
                        SpamType::LowFeeRate => spam_breakdown.dust += 1, // Count as suspicious
                        SpamType::HighSizeValueRatio => spam_breakdown.ordinals += 1, // Count as Ordinals
                        SpamType::ManySmallOutputs => spam_breakdown.dust += 1, // Count as dust-like
                        SpamType::NotSpam => {}
                    }
                }
            } else {
                filtered_txs.push(tx.clone());
            }
        }

        let summary = SpamSummary {
            filtered_count,
            filtered_size,
            by_type: spam_breakdown,
        };

        (filtered_txs, summary)
    }
}

impl Default for SpamFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of filtered spam
#[derive(Debug, Clone, Default)]
pub struct SpamSummary {
    /// Number of transactions filtered
    pub filtered_count: u32,
    /// Total size of filtered transactions (bytes, estimated)
    pub filtered_size: u64,
    /// Breakdown by spam type
    pub by_type: SpamBreakdown,
}

/// Breakdown of spam by category
#[derive(Debug, Clone, Default)]
pub struct SpamBreakdown {
    pub ordinals: u32,
    pub inscriptions: u32,
    pub dust: u32,
    pub brc20: u32,
}

/// Estimate transaction size in bytes
fn estimate_transaction_size(tx: &Transaction) -> u64 {
    // Simplified estimation:
    // - Version: 4 bytes
    // - Input count: varint (1-9 bytes, estimate 1)
    // - Per input: ~150 bytes (prevout + script + sequence)
    // - Output count: varint (1-9 bytes, estimate 1)
    // - Per output: ~35 bytes (value + script)
    // - Locktime: 4 bytes

    let base_size: u64 = 4 + 1 + 1 + 4; // Version + input count + output count + locktime
    let input_size = tx.inputs.len() as u64 * 150;
    let output_size = tx
        .outputs
        .iter()
        .map(|out| 8 + out.script_pubkey.len() as u64)
        .sum::<u64>();

    let total_size = base_size
        .checked_add(input_size)
        .and_then(|sum| sum.checked_add(output_size))
        .unwrap_or(u64::MAX); // Overflow protection

    // Runtime assertion: Estimated size must be reasonable
    debug_assert!(
        total_size <= 1_000_000,
        "Transaction size estimate ({total_size}) must not exceed MAX_TX_SIZE (1MB)"
    );

    total_size
}

/// Serializable adaptive thresholds
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WitnessSizeThresholdsSerializable {
    #[serde(default = "default_normal_single_sig")]
    pub normal_single_sig: usize,
    #[serde(default = "default_normal_multi_sig")]
    pub normal_multi_sig: usize,
    #[serde(default = "default_normal_p2wsh")]
    pub normal_p2wsh: usize,
    #[serde(default = "default_suspicious_threshold")]
    pub suspicious_threshold: usize,
    #[serde(default = "default_definitely_spam")]
    pub definitely_spam: usize,
}

impl From<WitnessSizeThresholdsSerializable> for WitnessSizeThresholds {
    fn from(serializable: WitnessSizeThresholdsSerializable) -> Self {
        WitnessSizeThresholds {
            normal_single_sig: serializable.normal_single_sig,
            normal_multi_sig: serializable.normal_multi_sig,
            normal_p2wsh: serializable.normal_p2wsh,
            suspicious_threshold: serializable.suspicious_threshold,
            definitely_spam: serializable.definitely_spam,
        }
    }
}

impl From<WitnessSizeThresholds> for WitnessSizeThresholdsSerializable {
    fn from(thresholds: WitnessSizeThresholds) -> Self {
        WitnessSizeThresholdsSerializable {
            normal_single_sig: thresholds.normal_single_sig,
            normal_multi_sig: thresholds.normal_multi_sig,
            normal_p2wsh: thresholds.normal_p2wsh,
            suspicious_threshold: thresholds.suspicious_threshold,
            definitely_spam: thresholds.definitely_spam,
        }
    }
}

/// Serializable spam filter configuration (for config files)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpamFilterConfigSerializable {
    #[serde(default = "default_true")]
    pub filter_ordinals: bool,
    #[serde(default = "default_true")]
    pub filter_dust: bool,
    #[serde(default = "default_true")]
    pub filter_brc20: bool,
    #[serde(default = "default_true")]
    pub filter_large_witness: bool,
    #[serde(default = "default_false")]
    pub filter_low_fee_rate: bool,
    #[serde(default = "default_true")]
    pub filter_high_size_value_ratio: bool,
    #[serde(default = "default_true")]
    pub filter_many_small_outputs: bool,
    #[serde(default = "default_dust_threshold")]
    pub dust_threshold: i64,
    #[serde(default = "default_dust_threshold")]
    pub min_output_value: i64,
    #[serde(default = "default_min_fee_rate")]
    pub min_fee_rate: u64,
    #[serde(default = "default_max_witness_size")]
    pub max_witness_size: usize,
    #[serde(default = "default_max_size_value_ratio")]
    pub max_size_value_ratio: f64,
    #[serde(default = "default_max_small_outputs")]
    pub max_small_outputs: usize,
    
    // NEW: Adaptive thresholds
    #[serde(default = "default_true")]
    pub use_adaptive_thresholds: bool,
    #[serde(default = "default_adaptive_thresholds")]
    pub adaptive_thresholds: WitnessSizeThresholdsSerializable,
    
    // NEW: Taproot-specific options
    #[serde(default = "default_true")]
    pub filter_taproot_spam: bool,
    #[serde(default = "default_max_taproot_control_size")]
    pub max_taproot_control_size: usize,
    #[serde(default = "default_true")]
    pub reject_taproot_annexes: bool,
    
    // NEW: Total witness size check
    #[serde(default = "default_false")]
    pub filter_large_total_witness: bool,
    #[serde(default = "default_max_total_witness_size")]
    pub max_total_witness_size: usize,
    
    // NEW: Enhanced detection options
    #[serde(default = "default_true")]
    pub use_improved_envelope_detection: bool,
    #[serde(default = "default_true")]
    pub use_json_validation_brc20: bool,
    
    // NEW: Fee rate calculation options
    #[serde(default = "default_false")]
    pub require_utxo_for_fee_rate: bool,
    #[serde(default = "default_min_fee_rate_large_tx")]
    pub min_fee_rate_large_tx: u64,
    #[serde(default = "default_large_tx_threshold_bytes")]
    pub large_tx_threshold_bytes: usize,
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_dust_threshold() -> i64 {
    546
}

fn default_min_fee_rate() -> u64 {
    1
}

fn default_max_witness_size() -> usize {
    1000
}

fn default_max_size_value_ratio() -> f64 {
    1000.0
}

fn default_max_small_outputs() -> usize {
    10
}

fn default_max_taproot_control_size() -> usize {
    289
}

fn default_max_total_witness_size() -> usize {
    5000
}

fn default_min_fee_rate_large_tx() -> u64 {
    2
}

fn default_large_tx_threshold_bytes() -> usize {
    1000
}

fn default_normal_single_sig() -> usize {
    200
}

fn default_normal_multi_sig() -> usize {
    500
}

fn default_normal_p2wsh() -> usize {
    800
}

fn default_suspicious_threshold() -> usize {
    1000
}

fn default_definitely_spam() -> usize {
    2000
}

fn default_adaptive_thresholds() -> WitnessSizeThresholdsSerializable {
    WitnessSizeThresholdsSerializable {
        normal_single_sig: 200,
        normal_multi_sig: 500,
        normal_p2wsh: 800,
        suspicious_threshold: 1000,
        definitely_spam: 2000,
    }
}

impl From<SpamFilterConfigSerializable> for SpamFilterConfig {
    fn from(serializable: SpamFilterConfigSerializable) -> Self {
        SpamFilterConfig {
            filter_ordinals: serializable.filter_ordinals,
            filter_dust: serializable.filter_dust,
            filter_brc20: serializable.filter_brc20,
            filter_large_witness: serializable.filter_large_witness,
            filter_low_fee_rate: serializable.filter_low_fee_rate,
            filter_high_size_value_ratio: serializable.filter_high_size_value_ratio,
            filter_many_small_outputs: serializable.filter_many_small_outputs,
            dust_threshold: serializable.dust_threshold,
            min_output_value: serializable.min_output_value,
            min_fee_rate: serializable.min_fee_rate,
            max_witness_size: serializable.max_witness_size,
            max_size_value_ratio: serializable.max_size_value_ratio,
            max_small_outputs: serializable.max_small_outputs,
            // NEW: Adaptive thresholds
            use_adaptive_thresholds: serializable.use_adaptive_thresholds,
            adaptive_thresholds: serializable.adaptive_thresholds.into(),
            // NEW fields
            filter_taproot_spam: serializable.filter_taproot_spam,
            max_taproot_control_size: serializable.max_taproot_control_size,
            reject_taproot_annexes: serializable.reject_taproot_annexes,
            filter_large_total_witness: serializable.filter_large_total_witness,
            max_total_witness_size: serializable.max_total_witness_size,
            use_improved_envelope_detection: serializable.use_improved_envelope_detection,
            use_json_validation_brc20: serializable.use_json_validation_brc20,
            require_utxo_for_fee_rate: serializable.require_utxo_for_fee_rate,
            min_fee_rate_large_tx: serializable.min_fee_rate_large_tx,
            large_tx_threshold_bytes: serializable.large_tx_threshold_bytes,
        }
    }
}

impl From<SpamFilterConfig> for SpamFilterConfigSerializable {
    fn from(config: SpamFilterConfig) -> Self {
        SpamFilterConfigSerializable {
            filter_ordinals: config.filter_ordinals,
            filter_dust: config.filter_dust,
            filter_brc20: config.filter_brc20,
            filter_large_witness: config.filter_large_witness,
            filter_low_fee_rate: config.filter_low_fee_rate,
            filter_high_size_value_ratio: config.filter_high_size_value_ratio,
            filter_many_small_outputs: config.filter_many_small_outputs,
            dust_threshold: config.dust_threshold,
            min_output_value: config.min_output_value,
            min_fee_rate: config.min_fee_rate,
            max_witness_size: config.max_witness_size,
            max_size_value_ratio: config.max_size_value_ratio,
            max_small_outputs: config.max_small_outputs,
            // NEW: Adaptive thresholds
            use_adaptive_thresholds: config.use_adaptive_thresholds,
            adaptive_thresholds: config.adaptive_thresholds.into(),
            // NEW fields
            filter_taproot_spam: config.filter_taproot_spam,
            max_taproot_control_size: config.max_taproot_control_size,
            reject_taproot_annexes: config.reject_taproot_annexes,
            filter_large_total_witness: config.filter_large_total_witness,
            max_total_witness_size: config.max_total_witness_size,
            use_improved_envelope_detection: config.use_improved_envelope_detection,
            use_json_validation_brc20: config.use_json_validation_brc20,
            require_utxo_for_fee_rate: config.require_utxo_for_fee_rate,
            min_fee_rate_large_tx: config.min_fee_rate_large_tx,
            large_tx_threshold_bytes: config.large_tx_threshold_bytes,
        }
    }
}

