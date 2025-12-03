//! Spam Filtering for UTXO Commitments
//!
//! Implements spam detection and filtering for Bitcoin transactions:
//! - Ordinals/Inscriptions detection
//! - Dust output filtering
//! - BRC-20 pattern detection
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

use crate::types::{ByteString, Transaction};
use crate::witness::Witness;
use serde::{Deserialize, Serialize};

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
    pub max_witness_size: usize,
    /// Maximum size-to-value ratio (bytes per satoshi)
    pub max_size_value_ratio: f64,
    /// Maximum number of small outputs before flagging
    pub max_small_outputs: usize,
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
}

impl SpamFilter {
    /// Create a new spam filter with default configuration
    pub fn new() -> Self {
        Self {
            config: SpamFilterConfig::default(),
        }
    }

    /// Create a new spam filter with custom configuration
    pub fn with_config(config: SpamFilterConfig) -> Self {
        Self { config }
    }

    /// Check if a transaction is spam (without witness data)
    ///
    /// This is the backward-compatible method. For better detection, use `is_spam_with_witness`.
    pub fn is_spam(&self, tx: &Transaction) -> SpamFilterResult {
        self.is_spam_with_witness(tx, None)
    }

    /// Check if a transaction is spam (with optional witness data)
    ///
    /// Witness data is required for detecting Taproot/SegWit-based Ordinals.
    /// If witness data is not provided, detection will be less accurate.
    pub fn is_spam_with_witness(
        &self,
        tx: &Transaction,
        witnesses: Option<&[Witness]>,
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

        // Check for large witness data
        if self.config.filter_large_witness && self.detect_large_witness(witnesses) {
            detected_types.push(SpamType::LargeWitness);
        }

        // Check for low fee rate (requires fee calculation)
        if self.config.filter_low_fee_rate && self.detect_low_fee_rate(tx, witnesses) {
            detected_types.push(SpamType::LowFeeRate);
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

                // Check for large witness stacks (common in Ordinals)
                if self.has_large_witness_stack(witness) {
                    return true;
                }

                // Check for data patterns in witness elements
                if self.has_witness_data_pattern(witness) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if witness stack is suspiciously large (suggests data embedding)
    fn has_large_witness_stack(&self, witness: &Witness) -> bool {
        // Calculate total witness size
        let total_size: usize = witness.iter().map(|elem| elem.len()).sum();

        // Large witness stacks (>1000 bytes) are suspicious
        total_size > self.config.max_witness_size
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
                if element.len() != 64 && (element.is_empty() || element[0] != 0x30) {
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

        // Check for OP_RETURN (0x6a) - common in Ordinals
        if script[0] == 0x6a {
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
        // Envelope protocol: OP_FALSE (0x00) OP_IF (0x63) ... OP_ENDIF (0x68)
        // This is a simplified check
        if script.len() < 4 {
            return false;
        }

        // Check for OP_FALSE OP_IF pattern (common in inscriptions)
        if script[0] == 0x00 && script[1] == 0x63 {
            // Likely envelope protocol
            return true;
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
    fn detect_large_witness(&self, witnesses: Option<&[Witness]>) -> bool {
        if let Some(witnesses) = witnesses {
            for witness in witnesses {
                if self.has_large_witness_stack(witness) {
                    return true;
                }
            }
        }
        false
    }

    /// Detect transactions with low fee rate
    ///
    /// Non-monetary transactions often pay minimal fees relative to size.
    /// This requires calculating the transaction fee, which we estimate from size.
    fn detect_low_fee_rate(&self, tx: &Transaction, witnesses: Option<&[Witness]>) -> bool {
        // Estimate transaction size (including witness)
        let tx_size = self.estimate_transaction_size_with_witness(tx, witnesses);

        // For this check, we need to estimate the fee
        // Since we don't have UTXO set here, we use heuristics:
        // - If transaction has many inputs and small outputs, likely low fee
        // - If transaction is very large but outputs are small, likely low fee

        // Simplified heuristic: very large transactions with small total output value
        // are likely non-monetary
        let total_output_value: i64 = tx.outputs.iter().map(|out| out.value).sum();

        // If transaction is large but total value is small, fee rate is likely low
        if tx_size > 1000 && total_output_value < 10000 {
            // Estimate fee rate (assuming minimal fee)
            // This is a heuristic - actual fee calculation requires UTXO set
            let estimated_fee_rate = if tx_size > 0 {
                // Assume minimal fee (1000 sats) for large transactions
                1000u64.saturating_div(tx_size as u64)
            } else {
                0
            };

            if estimated_fee_rate < self.config.min_fee_rate {
                return true;
            }
        }

        false
    }

    /// Detect transactions with high size-to-value ratio
    ///
    /// Non-monetary transactions often have very large size relative to value transferred.
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
        ratio > self.config.max_size_value_ratio
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
        if script[0] != 0x6a {
            return false;
        }

        // Convert to string and check for BRC-20 JSON pattern
        // BRC-20 JSON typically contains "p":"brc-20"
        if let Ok(script_str) = String::from_utf8(script[1..].to_vec()) {
            // Check for BRC-20 markers
            if script_str.contains("brc-20")
                || script_str.contains("\"p\":\"brc-20\"")
                || script_str.contains("op\":\"mint")
                || script_str.contains("op\":\"transfer")
                || script_str.contains("op\":\"deploy")
            {
                return true;
            }
        }

        false
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
                self.is_spam_with_witness(tx, Some(tx_witnesses))
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
        }
    }
}
