//! Script Type Detection for Adaptive Spam Filtering
//!
//! This module provides script type detection to enable adaptive witness size thresholds
//! based on the script type (P2PKH, P2SH, P2WPKH, P2WSH, P2TR, MultiSig, etc.).
//!
//! Reference: BIP-XXX Technical Improvement Plan - Category 1.1

use crate::types::ByteString;
use crate::opcodes::*;

/// Script type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScriptType {
    /// Pay-to-Public-Key-Hash (legacy)
    P2PKH,
    /// Pay-to-Script-Hash
    P2SH,
    /// Pay-to-Witness-Public-Key-Hash (SegWit v0)
    P2WPKH,
    /// Pay-to-Witness-Script-Hash (SegWit v0)
    P2WSH,
    /// Pay-to-Taproot (SegWit v1)
    P2TR,
    /// Multi-signature script (n-of-m)
    MultiSig { n: u8, m: u8 },
    /// Payment channel (HTLC patterns)
    PaymentChannel,
    /// Unknown or complex script
    Unknown,
}

impl ScriptType {
    /// Detect script type from scriptPubKey
    ///
    /// This analyzes the output script to determine its type.
    /// For input scripts (scriptSig), use `detect_input_script_type` instead.
    pub fn detect(script_pubkey: &ByteString) -> Self {
        if script_pubkey.is_empty() {
            return Self::Unknown;
        }

        // P2TR: OP_1 (0x51) + 0x20 + 32-byte x-only pubkey = 34 bytes
        if script_pubkey.len() == 34
            && script_pubkey[0] == OP_1
            && script_pubkey[1] == 0x20
        {
            return Self::P2TR;
        }

        // P2WPKH: OP_0 (0x00) + 0x14 + 20-byte hash = 22 bytes
        if script_pubkey.len() == 22
            && script_pubkey[0] == OP_0
            && script_pubkey[1] == 0x14
        {
            return Self::P2WPKH;
        }

        // P2WSH: OP_0 (0x00) + 0x20 + 32-byte hash = 34 bytes
        if script_pubkey.len() == 34
            && script_pubkey[0] == OP_0
            && script_pubkey[1] == 0x20
        {
            return Self::P2WSH;
        }

        // P2PKH: OP_DUP OP_HASH160 <20-byte-hash> OP_EQUALVERIFY OP_CHECKSIG
        // Format: [0x76, 0xa9, 0x14, <20 bytes>, 0x88, 0xac]
        if script_pubkey.len() == 25
            && script_pubkey[0] == 0x76  // OP_DUP
            && script_pubkey[1] == 0xa9  // OP_HASH160
            && script_pubkey[2] == 0x14  // push 20 bytes
            && script_pubkey[23] == 0x88 // OP_EQUALVERIFY
            && script_pubkey[24] == 0xac // OP_CHECKSIG
        {
            return Self::P2PKH;
        }

        // P2SH: OP_HASH160 <20-byte-hash> OP_EQUAL
        // Format: [0xa9, 0x14, <20 bytes>, 0x87]
        if script_pubkey.len() == 23
            && script_pubkey[0] == OP_HASH160  // OP_HASH160
            && script_pubkey[1] == PUSH_20_BYTES  // push 20 bytes
            && script_pubkey[22] == OP_EQUAL // OP_EQUAL
        {
            return Self::P2SH;
        }

        // Multi-sig detection: Check for OP_CHECKMULTISIG pattern
        // Format: OP_n <pubkey1> <pubkey2> ... <pubkeym> OP_m OP_CHECKMULTISIG
        if let Some(multisig) = Self::detect_multisig(script_pubkey) {
            return multisig;
        }

        // Payment channel detection: Check for HTLC patterns
        // This is a simplified check - full implementation would parse script
        if Self::has_htlc_pattern(script_pubkey) {
            return Self::PaymentChannel;
        }

        Self::Unknown
    }

    /// Detect multi-signature script type
    ///
    /// Returns `Some(MultiSig { n, m })` if script is a multi-sig, `None` otherwise.
    fn detect_multisig(script: &ByteString) -> Option<Self> {
        if script.len() < 3 {
            return None;
        }

        // Multi-sig format: OP_n <pubkey1> ... <pubkeym> OP_m OP_CHECKMULTISIG
        // OP_CHECKMULTISIG = 0xae
        if script[script.len() - 1] != OP_CHECKMULTISIG {
            return None;
        }

        // Parse m (number of required signatures)
        let m_opcode = script[script.len() - 2];
        let m = if m_opcode >= OP_1 && m_opcode <= OP_16 {
            (m_opcode - OP_1 + 1) as u8
        } else {
            return None;
        };

        // Parse n (number of public keys) - first opcode
        let n_opcode = script[0];
        let n = if n_opcode >= OP_1 && n_opcode <= OP_16 {
            (n_opcode - OP_1 + 1) as u8
        } else {
            return None;
        };

        // Validate: n >= m (can't require more signatures than available keys)
        if n < m {
            return None;
        }

        // Validate: typical pubkey size is 33 or 65 bytes
        // Rough check: script should be approximately n * 33 + overhead
        let expected_min_size = (n as usize) * 33 + 3; // n pubkeys + n opcode + m opcode + OP_CHECKMULTISIG
        if script.len() < expected_min_size {
            return None;
        }

        Some(Self::MultiSig { n, m })
    }

    /// Check if script has HTLC (Hash Time Lock Contract) pattern
    ///
    /// HTLCs are used in payment channels (Lightning Network).
    /// This is a simplified check - full implementation would parse script.
    fn has_htlc_pattern(script: &ByteString) -> bool {
        // HTLC patterns typically include:
        // - OP_IF / OP_NOTIF (conditional branches)
        // - OP_CHECKLOCKTIMEVERIFY or OP_CHECKSEQUENCEVERIFY
        // - OP_HASH160 or OP_SHA256
        // - OP_EQUALVERIFY
        // This is a heuristic - not all scripts with these patterns are HTLCs

        if script.len() < 20 {
            return false;
        }

        // Check for common HTLC opcodes
        let has_conditional = script.contains(&OP_IF) || script.contains(&OP_NOTIF);
        let has_time_lock = script.contains(&OP_CHECKLOCKTIMEVERIFY) || script.contains(&OP_CHECKSEQUENCEVERIFY);
        let has_hash = script.contains(&OP_HASH160) || script.contains(&OP_SHA256);
        let has_equal = script.contains(&OP_EQUALVERIFY);

        // HTLCs typically have all of these
        has_conditional && has_time_lock && has_hash && has_equal
    }

    /// Get expected witness size range for this script type
    ///
    /// Returns (min_bytes, typical_bytes, max_bytes) for normal usage.
    /// These values are based on typical transaction patterns and will be
    /// refined with real-world data collection.
    pub fn expected_witness_size_range(&self) -> (usize, usize, usize) {
        match self {
            Self::P2PKH => (0, 0, 0), // No witness
            Self::P2SH => (0, 0, 0),  // No witness (unless nested SegWit)
            Self::P2WPKH => (107, 107, 107), // 1 signature (71-73 bytes) + 1 pubkey (33 bytes) + varints
            Self::P2WSH => (200, 300, 500),  // Variable: depends on redeem script
            Self::P2TR => (64, 64, 64),      // 1 Schnorr signature (64 bytes) + varint
            Self::MultiSig { n, m } => {
                // n signatures + m pubkeys + redeem script + varints
                let sig_size = (*n as usize) * 73; // Max signature size
                let pubkey_size = (*m as usize) * 33; // Compressed pubkeys
                let script_size = 3 + (*m as usize) * 33; // OP_n + m pubkeys + OP_m + OP_CHECKMULTISIG
                let total = sig_size + pubkey_size + script_size + 10; // +10 for varints
                (total / 2, total, total * 2) // Conservative range
            }
            Self::PaymentChannel => (300, 500, 1000), // HTLC scripts can be complex
            Self::Unknown => (0, 200, 1000), // Unknown - use conservative default
        }
    }

    /// Get recommended threshold for this script type
    ///
    /// Returns the recommended maximum witness size threshold based on script type.
    /// These values will be refined with real-world data collection.
    pub fn recommended_threshold(&self) -> usize {
        match self {
            Self::P2PKH | Self::P2SH => 0, // No witness
            Self::P2WPKH => 150,            // 107 typical + 50% buffer
            Self::P2WSH => 800,            // 300 typical + buffer for complex scripts
            Self::P2TR => 100,             // 64 typical + buffer
            Self::MultiSig { n, m } => {
                // Calculate based on n-of-m
                let base = (*n as usize) * 73 + (*m as usize) * 33 + 50; // Signatures + pubkeys + overhead
                base + (base / 2) // Add 50% buffer
            }
            Self::PaymentChannel => 1500, // HTLC scripts can be large
            Self::Unknown => 1000,         // Conservative default
        }
    }
}

/// Detect script type from input script (scriptSig)
///
/// This analyzes the input script to infer the script type of the output being spent.
/// This is less reliable than analyzing scriptPubKey directly, but useful when
/// scriptPubKey is not available.
pub fn detect_input_script_type(script_sig: &ByteString) -> Option<ScriptType> {
    if script_sig.is_empty() {
        // Empty scriptSig suggests SegWit (P2WPKH, P2WSH, or P2TR)
        return Some(ScriptType::P2WPKH); // Most common SegWit type
    }

    // Check for multi-sig pattern in scriptSig
    if let Some(multisig) = ScriptType::detect_multisig(script_sig) {
        return Some(multisig);
    }

    // P2PKH scriptSig: <signature> <pubkey>
    // Typical sizes: signature ~71-73 bytes, pubkey 33 or 65 bytes
    if script_sig.len() >= 100 && script_sig.len() <= 150 {
        // Could be P2PKH
        return Some(ScriptType::P2PKH);
    }

    None
}

/// Transaction type classification for edge case handling
///
/// This helps reduce false positives by identifying legitimate transaction patterns
/// that might otherwise be flagged as spam (e.g., consolidations, CoinJoins).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionType {
    /// Normal payment transaction
    Payment,
    /// Consolidation transaction (many inputs, few outputs)
    Consolidation,
    /// CoinJoin transaction (many inputs, many outputs, similar values)
    CoinJoin,
    /// Payment channel (Lightning Network)
    PaymentChannel,
    /// Unknown or complex transaction
    Unknown,
}

impl TransactionType {
    /// Detect transaction type from transaction structure
    ///
    /// This analyzes the transaction to determine its likely purpose.
    /// Used to adjust spam detection thresholds for legitimate use cases.
    pub fn detect(tx: &crate::types::Transaction) -> Self {
        let input_count = tx.inputs.len();
        let output_count = tx.outputs.len();
        
        // Consolidation: many inputs, few outputs
        // Typical pattern: 10+ inputs, 1-3 outputs
        if input_count >= 10 && output_count <= 3 {
            return Self::Consolidation;
        }
        
        // CoinJoin: many inputs, many outputs, similar output values
        // Typical pattern: 5+ inputs, 5+ outputs, output values within 10% of each other
        if input_count >= 5 && output_count >= 5 {
            let output_values: Vec<i64> = tx.outputs.iter().map(|out| out.value).collect();
            if Self::has_similar_values(&output_values, 0.1) {
                return Self::CoinJoin;
            }
        }
        
        // Payment channel: complex scripts, specific patterns
        // This is a simplified check - full implementation would parse scripts
        if Self::has_payment_channel_pattern(tx) {
            return Self::PaymentChannel;
        }
        
        // Default to payment for normal transactions
        if input_count <= 5 && output_count <= 5 {
            return Self::Payment;
        }
        
        Self::Unknown
    }
    
    /// Check if output values are similar (within percentage threshold)
    fn has_similar_values(values: &[i64], threshold: f64) -> bool {
        if values.is_empty() {
            return false;
        }
        
        let avg: f64 = values.iter().sum::<i64>() as f64 / values.len() as f64;
        
        for &value in values {
            let diff = (value as f64 - avg).abs() / avg;
            if diff > threshold {
                return false;
            }
        }
        
        true
    }
    
    /// Check if transaction has payment channel patterns
    fn has_payment_channel_pattern(tx: &crate::types::Transaction) -> bool {
        // Payment channels typically have:
        // - Complex scripts (HTLC patterns)
        // - Time locks
        // - Multiple outputs with similar values
        
        // Check for complex scripts in outputs
        let mut has_complex_script = false;
        for output in &tx.outputs {
            if output.script_pubkey.len() > 100 {
                has_complex_script = true;
                break;
            }
        }
        
        has_complex_script && tx.outputs.len() >= 2
    }
    
    /// Get recommended size-to-value ratio threshold for this transaction type
    ///
    /// Consolidations and CoinJoins legitimately have high size-to-value ratios,
    /// so we use higher thresholds for these transaction types.
    pub fn recommended_size_value_ratio(&self) -> f64 {
        match self {
            Self::Payment => 1000.0,        // Standard threshold
            Self::Consolidation => 5000.0,  // Consolidations can have high ratios
            Self::CoinJoin => 3000.0,       // CoinJoins have moderate ratios
            Self::PaymentChannel => 2000.0, // Payment channels can be complex
            Self::Unknown => 1000.0,        // Conservative default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_p2pkh() {
        // P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
        let script = ByteString::from(vec![
            OP_DUP, OP_HASH160, PUSH_20_BYTES, // OP_DUP OP_HASH160 push 20 bytes
            OP_0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
            OP_EQUALVERIFY, OP_CHECKSIG, // OP_EQUALVERIFY OP_CHECKSIG
        ]);
        assert_eq!(ScriptType::detect(&script), ScriptType::P2PKH);
    }

    #[test]
    fn test_detect_p2sh() {
        // P2SH: OP_HASH160 <20 bytes> OP_EQUAL
        let script = ByteString::from(vec![
            OP_HASH160, PUSH_20_BYTES, // OP_HASH160 push 20 bytes
            OP_0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
            OP_EQUAL, // OP_EQUAL
        ]);
        assert_eq!(ScriptType::detect(&script), ScriptType::P2SH);
    }

    #[test]
    fn test_detect_p2wpkh() {
        // P2WPKH: OP_0 <20-byte-hash>
        let script = ByteString::from(vec![
            OP_0, PUSH_20_BYTES, // OP_0 push 20 bytes
            OP_0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
        ]);
        assert_eq!(ScriptType::detect(&script), ScriptType::P2WPKH);
    }

    #[test]
    fn test_detect_p2wsh() {
        // P2WSH: OP_0 <32-byte-hash>
        let script = ByteString::from(vec![
            OP_0, PUSH_32_BYTES, // OP_0 push 32 bytes
            OP_0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
            PUSH_20_BYTES, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);
        assert_eq!(ScriptType::detect(&script), ScriptType::P2WSH);
    }

    #[test]
    fn test_detect_p2tr() {
        // P2TR: OP_1 <32-byte-hash>
        let script = ByteString::from(vec![
            OP_1, PUSH_32_BYTES, // OP_1 push 32 bytes
            OP_0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
            PUSH_20_BYTES, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
            0x1e, 0x1f,
        ]);
        assert_eq!(ScriptType::detect(&script), ScriptType::P2TR);
    }

    #[test]
    fn test_detect_multisig() {
        // 2-of-3 multi-sig: OP_2 <pubkey1> <pubkey2> <pubkey3> OP_3 OP_CHECKMULTISIG
        let mut script = vec![OP_2];
        // Add 3 pubkeys (33 bytes each)
        for _ in 0..3 {
            script.push(0x21); // push 33 bytes
            script.extend(vec![0u8; 33]);
        }
        script.push(OP_3);
        script.push(OP_CHECKMULTISIG);

        let script_type = ScriptType::detect(&ByteString::from(script));
        match script_type {
            ScriptType::MultiSig { n, m } => {
                assert_eq!(n, 3);
                assert_eq!(m, 2);
            }
            _ => panic!("Expected MultiSig, got {:?}", script_type),
        }
    }

    #[test]
    fn test_expected_witness_size_range() {
        let p2wpkh = ScriptType::P2WPKH;
        let (min, typical, max) = p2wpkh.expected_witness_size_range();
        assert!(min <= typical && typical <= max);
        assert!(typical > 0);

        let p2tr = ScriptType::P2TR;
        let (min, typical, max) = p2tr.expected_witness_size_range();
        assert!(min <= typical && typical <= max);
        assert_eq!(typical, 64); // Schnorr signature is always 64 bytes
    }

    #[test]
    fn test_recommended_threshold() {
        let p2wpkh = ScriptType::P2WPKH;
        assert!(p2wpkh.recommended_threshold() > 0);

        let p2tr = ScriptType::P2TR;
        assert!(p2tr.recommended_threshold() > 0);

        let multisig = ScriptType::MultiSig { n: 2, m: 3 };
        assert!(multisig.recommended_threshold() > 0);
    }
}


