//! Transaction hash calculation for signature verification
//!
//! Implements Bitcoin's transaction sighash algorithm for ECDSA signature verification.
//! This is critical for proper signature validation in script execution.
//!
//! Performance optimizations (Phase 6.2):
//! - Precomputed sighash templates for common transaction patterns

use crate::error::Result;
use crate::types::*;
use sha2::{Digest, Sha256};
use blvm_spec_lock::spec_locked;

// OPTIMIZATION: Inline varint encoding helper to avoid Vec allocations in hot path
#[inline]
fn write_varint_to_vec(vec: &mut Vec<u8>, value: u64) {
    if value < 0xfd {
        vec.push(value as u8);
    } else if value <= 0xffff {
        vec.push(0xfd);
        vec.extend_from_slice(&(value as u16).to_le_bytes());
    } else if value <= 0xffffffff {
        vec.push(0xfe);
        vec.extend_from_slice(&(value as u32).to_le_bytes());
    } else {
        vec.push(0xff);
        vec.extend_from_slice(&value.to_le_bytes());
    }
}

#[cfg(feature = "production")]
use std::collections::HashMap;
#[cfg(feature = "production")]
use std::sync::OnceLock;

/// SIGHASH types for transaction signature verification
/// 
/// IMPORTANT: The enum values match the canonical sighash bytes used in sighash computation.
/// Early Bitcoin allowed sighash type 0x00 (treated as SIGHASH_ALL behavior), which we
/// represent as `AllLegacy` to preserve the correct byte value for sighash calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SighashType {
    /// Sign all inputs and outputs (sighash byte 0x00 - early Bitcoin legacy)
    AllLegacy = 0x00,
    /// Sign all inputs and outputs (sighash byte 0x01 - standard)
    All = 0x01,
    /// Sign no outputs (anyone can spend)
    None = 0x02,
    /// Sign output at same index as input
    Single = 0x03,
    /// Sign only this input (anyone can spend other inputs)
    AnyoneCanPay = 0x80,
}

impl SighashType {
    /// Parse sighash type from byte
    /// 
    /// Note: In early Bitcoin (pre-BIP66), sighash type 0x00 was accepted and treated
    /// as SIGHASH_ALL. We represent this as `AllLegacy` to preserve the correct byte
    /// value for sighash computation.
    pub fn from_byte(byte: u8) -> Result<Self> {
        match byte {
            // 0x00 was accepted in early Bitcoin - preserve as AllLegacy for correct sighash
            0x00 => Ok(SighashType::AllLegacy),
            0x01 => Ok(SighashType::All),
            0x02 => Ok(SighashType::None),
            0x03 => Ok(SighashType::Single),
            0x81 => Ok(SighashType::All | SighashType::AnyoneCanPay),
            0x82 => Ok(SighashType::None | SighashType::AnyoneCanPay),
            0x83 => Ok(SighashType::Single | SighashType::AnyoneCanPay),
            _ => Err(crate::error::ConsensusError::InvalidSighashType(byte)),
        }
    }
    
    /// Check if this sighash type has SIGHASH_ALL behavior
    pub fn is_all(&self) -> bool {
        matches!(self, SighashType::All | SighashType::AllLegacy)
    }
}

impl std::ops::BitOr for SighashType {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SighashType::All, SighashType::AnyoneCanPay) => SighashType::All,
            (SighashType::None, SighashType::AnyoneCanPay) => SighashType::None,
            (SighashType::Single, SighashType::AnyoneCanPay) => SighashType::Single,
            _ => self,
        }
    }
}

/// Phase 6.2: Transaction structure pattern for template matching
#[cfg(feature = "production")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SighashPattern {
    version: u64,
    input_count: usize,
    output_count: usize,
    lock_time: u64,
    sighash_type: SighashType,
}

/// Phase 6.2: Precomputed sighash template cache
///
/// Caches common sighash preimage patterns to avoid redundant computation.
/// Only caches for standard transaction patterns (single input, single output, etc.)
#[cfg(feature = "production")]
static SIGHASH_TEMPLATES: OnceLock<HashMap<SighashPattern, Vec<u8>>> = OnceLock::new();

/// Phase 6.2: Check if transaction matches a common pattern suitable for templating
#[cfg(feature = "production")]
#[inline]
fn matches_template_pattern(
    tx: &Transaction,
    input_index: usize,
    sighash_type: SighashType,
) -> bool {
    // Only cache simple patterns to avoid complexity
    // Pattern: Single input, single output, standard P2PKH structure
    tx.inputs.len() == 1
        && tx.outputs.len() == 1
        && input_index == 0
        && matches!(sighash_type, SighashType::All)
        && tx.version == 1
        && tx.lock_time == 0
}

/// Phase 6.2: Get or create sighash template for common patterns
#[cfg(feature = "production")]
fn get_sighash_template(
    tx: &Transaction,
    input_index: usize,
    _prevouts: &[TransactionOutput],
    sighash_type: SighashType,
) -> Option<Vec<u8>> {
    if !matches_template_pattern(tx, input_index, sighash_type) {
        return None;
    }

    let pattern = SighashPattern {
        version: tx.version,
        input_count: tx.inputs.len(),
        output_count: tx.outputs.len(),
        lock_time: tx.lock_time,
        sighash_type,
    };

    let templates = SIGHASH_TEMPLATES.get_or_init(|| HashMap::new());

    // For now, return None (templates would need to be pre-populated)
    // Future: Pre-compute templates for standard patterns at startup
    templates.get(&pattern).cloned()
}

/// Calculate transaction sighash for signature verification
///
/// This implements the Bitcoin transaction hash algorithm used for ECDSA signatures.
/// The sighash determines which parts of the transaction are signed.
///
/// Performance optimization (Phase 6.2): Checks for precomputed templates
/// before computing sighash from scratch.
///
/// # Arguments
/// * `tx` - The transaction being signed
/// * `input_index` - Index of the input being signed
/// * `prevouts` - Previous transaction outputs (for input validation)
/// * `sighash_type` - Type of sighash to calculate
/// * `script_code` - Optional script code to use instead of scriptPubKey (for P2SH redeem script)
///
/// # Returns
/// 32-byte hash to be signed with ECDSA
#[spec_locked("5.1")]
pub fn calculate_transaction_sighash(
    tx: &Transaction,
    input_index: usize,
    prevouts: &[TransactionOutput],
    sighash_type: SighashType,
) -> Result<Hash> {
    calculate_transaction_sighash_with_script_code(tx, input_index, prevouts, sighash_type, None)
}

/// Calculate transaction sighash with optional script code override
/// 
/// For P2SH transactions, script_code should be the redeem script (not the scriptPubKey).
/// For non-P2SH, script_code should be None (uses scriptPubKey from prevout).
#[spec_locked("5.1")]
pub fn calculate_transaction_sighash_with_script_code(
    tx: &Transaction,
    input_index: usize,
    prevouts: &[TransactionOutput],
    sighash_type: SighashType,
    script_code: Option<&[u8]>,
) -> Result<Hash> {
    // Validate input index
    if input_index >= tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidInputIndex(input_index));
    }

    // Validate prevouts match inputs
    if prevouts.len() != tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidPrevoutsCount(
            prevouts.len(),
            tx.inputs.len(),
        ));
    }

    // Phase 6.2: Check for template cache (only for common patterns)
    #[cfg(feature = "production")]
    if let Some(template) = get_sighash_template(tx, input_index, prevouts, sighash_type) {
        // Template found - hash it directly
        let first_hash = Sha256::digest(&template);
        let second_hash = Sha256::digest(first_hash);
        let mut result = [0u8; 32];
        result.copy_from_slice(&second_hash);
        return Ok(result);
    }

    // Create sighash preimage (standard computation)
    // OPTIMIZATION: Pre-allocate with estimated capacity to avoid reallocations
    // Typical transaction: ~100-400 bytes (version: 4, inputs: ~50-200 each, outputs: ~30-150 each, locktime: 4, sighash: 4)
    let estimated_size = 4 + 2 + (tx.inputs.len() * 50) + 2 + (tx.outputs.len() * 30) + 4 + 4;
    let mut preimage = Vec::with_capacity(estimated_size.min(4096)); // Cap at 4KB to avoid huge allocations

    // 1. Transaction version (4 bytes, little endian)
    preimage.extend_from_slice(&(tx.version as u32).to_le_bytes());

    // 2. Number of inputs (varint)
    // OPTIMIZATION: Write varint directly to avoid Vec allocation
    write_varint_to_vec(&mut preimage, tx.inputs.len() as u64);

    // 3. Inputs (depending on sighash type)
    // For sighash calculation, the signed input's scriptSig is replaced with the
    // scriptPubKey from the UTXO being spent. Other inputs have empty scriptSigs.
    for (i, input) in tx.inputs.iter().enumerate() {
        // Prevout hash and index (always included)
        preimage.extend_from_slice(&input.prevout.hash);
        preimage.extend_from_slice(&(input.prevout.index as u32).to_le_bytes());
        
        if i == input_index {
            // For the input being signed, use script_code if provided (P2SH redeem script),
            // otherwise use the scriptPubKey from the UTXO being spent
            // OPTIMIZATION: Use match instead of unwrap_or to avoid Option overhead
            let script_code_bytes = match script_code {
                Some(s) => s,
                None => &prevouts[i].script_pubkey,
            };
            // OPTIMIZATION: Write varint directly to avoid Vec allocation
            write_varint_to_vec(&mut preimage, script_code_bytes.len() as u64);
            preimage.extend_from_slice(script_code_bytes);
        } else {
            // Other inputs have empty scriptSigs
            preimage.push(0); // empty script
        }
        
        // Sequence (4 bytes, little endian - Bitcoin serializes as u32)
        preimage.extend_from_slice(&(input.sequence as u32).to_le_bytes());
    }

    // 4. Number of outputs (varint)
    // OPTIMIZATION: Write varint directly to avoid Vec allocation
    write_varint_to_vec(&mut preimage, tx.outputs.len() as u64);

    // 5. Outputs (depending on sighash type)
    // Note: AllLegacy (0x00) has the same behavior as All (0x01) for output inclusion
    match sighash_type {
        SighashType::All | SighashType::AllLegacy => {
            // Include all outputs
            for output in &tx.outputs {
                preimage.extend_from_slice(&output.value.to_le_bytes());
                // OPTIMIZATION: Write varint directly to avoid Vec allocation
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
        SighashType::None => {
            // No outputs
        }
        SighashType::Single => {
            // Include output at same index as input
            if input_index < tx.outputs.len() {
                let output = &tx.outputs[input_index];
                preimage.extend_from_slice(&output.value.to_le_bytes());
                // OPTIMIZATION: Write varint directly to avoid Vec allocation
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
        _ => {
            // AnyoneCanPay combinations
            for output in &tx.outputs {
                preimage.extend_from_slice(&output.value.to_le_bytes());
                // OPTIMIZATION: Write varint directly to avoid Vec allocation
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
    }

    // 6. Lock time (4 bytes, little endian)
    preimage.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());

    // 7. SIGHASH type (4 bytes, little endian)
    preimage.extend_from_slice(&(sighash_type as u32).to_le_bytes());

    // Calculate double SHA256 hash
    let first_hash = Sha256::digest(&preimage);
    let second_hash = Sha256::digest(first_hash);

    let mut result = [0u8; 32];
    result.copy_from_slice(&second_hash);

    Ok(result)
}

/// Batch compute sighashes for all inputs of a transaction
///
/// This function computes sighashes for all inputs at once, which is useful when
/// validating transactions with many inputs. The sighashes are computed in parallel
/// when the production feature is enabled.
///
/// # Arguments
/// * `tx` - The transaction being signed
/// * `prevouts` - Previous transaction outputs (for input validation)
/// * `sighash_type` - Type of sighash to calculate (must be the same for all inputs)
///
/// # Returns
/// Vector of 32-byte hashes, one per input (in same order)
#[spec_locked("5.1.1")]
pub fn batch_compute_sighashes(
    tx: &Transaction,
    prevouts: &[TransactionOutput],
    sighash_type: SighashType,
) -> Result<Vec<Hash>> {
    // Validate prevouts match inputs
    if prevouts.len() != tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidPrevoutsCount(
            prevouts.len(),
            tx.inputs.len(),
        ));
    }

    #[cfg(feature = "production")]
    {
        use crate::optimizations::simd_vectorization;

        // Serialize all sighash preimages in parallel
        let preimages: Vec<Vec<u8>> = {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                (0..tx.inputs.len())
                    .into_par_iter()
                    .map(|input_index| {
                        serialize_sighash_preimage(tx, input_index, prevouts, sighash_type)
                    })
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                (0..tx.inputs.len())
                    .map(|input_index| {
                        serialize_sighash_preimage(tx, input_index, prevouts, sighash_type)
                    })
                    .collect()
            }
        };

        // Batch hash all preimages using double SHA256
        let preimage_refs: Vec<&[u8]> = preimages.iter().map(|v| v.as_slice()).collect();
        Ok(simd_vectorization::batch_double_sha256(&preimage_refs))
    }

    #[cfg(not(feature = "production"))]
    {
        // Sequential fallback for non-production builds
        let mut results = Vec::with_capacity(tx.inputs.len());
        for i in 0..tx.inputs.len() {
            results.push(calculate_transaction_sighash(
                tx,
                i,
                prevouts,
                sighash_type,
            )?);
        }
        Ok(results)
    }
}

/// Serialize sighash preimage (helper for batch computation)
///
/// This extracts the serialization logic from calculate_transaction_sighash
/// to allow batch processing.
#[allow(dead_code)]
fn serialize_sighash_preimage(
    tx: &Transaction,
    input_index: usize,
    _prevouts: &[TransactionOutput],
    sighash_type: SighashType,
) -> Vec<u8> {
    let mut preimage = Vec::new();

    // 1. Transaction version (4 bytes, little endian)
    preimage.extend_from_slice(&(tx.version as u32).to_le_bytes());

    // 2. Number of inputs (varint)
    // OPTIMIZATION: Write varint directly to avoid Vec allocation
    write_varint_to_vec(&mut preimage, tx.inputs.len() as u64);

    // 3. Inputs (depending on sighash type)
    for (i, input) in tx.inputs.iter().enumerate() {
        if matches!(sighash_type, SighashType::AnyoneCanPay) || i == input_index {
            // Include this input
            preimage.extend_from_slice(&input.prevout.hash);
            preimage.extend_from_slice(&(input.prevout.index as u32).to_le_bytes());
            // OPTIMIZATION: Write varint directly to avoid Vec allocation
            write_varint_to_vec(&mut preimage, input.script_sig.len() as u64);
            preimage.extend_from_slice(&input.script_sig);
            preimage.extend_from_slice(&(input.sequence as u32).to_le_bytes());
        } else {
            // Skip this input (use dummy values)
            preimage.extend_from_slice(&[0u8; 32]); // prevout hash
            preimage.extend_from_slice(&[0u8; 4]); // prevout index
            preimage.push(0); // empty script_sig
            preimage.extend_from_slice(&[0u8; 4]); // sequence
        }
    }

    // 4. Number of outputs (varint)
    // OPTIMIZATION: Write varint directly to avoid Vec allocation
    write_varint_to_vec(&mut preimage, tx.outputs.len() as u64);

    // 5. Outputs (depending on sighash type)
    // Note: AllLegacy (0x00) has the same behavior as All (0x01) for output inclusion
    match sighash_type {
        SighashType::All | SighashType::AllLegacy => {
            // Include all outputs
            for output in &tx.outputs {
                preimage.extend_from_slice(&output.value.to_le_bytes());
                // OPTIMIZATION: Write varint directly to avoid Vec allocation
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
        SighashType::None => {
            // No outputs
        }
        SighashType::Single => {
            // Include output at same index as input
            if input_index < tx.outputs.len() {
                let output = &tx.outputs[input_index];
                preimage.extend_from_slice(&output.value.to_le_bytes());
                // OPTIMIZATION: Write varint directly to avoid Vec allocation
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
        _ => {
            // AnyoneCanPay combinations
            for output in &tx.outputs {
                preimage.extend_from_slice(&output.value.to_le_bytes());
                // OPTIMIZATION: Write varint directly to avoid Vec allocation
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
    }

    // 6. Lock time (4 bytes, little endian)
    preimage.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());

    // 7. SIGHASH type (4 bytes, little endian)
    preimage.extend_from_slice(&(sighash_type as u32).to_le_bytes());

    preimage
}

/// Encode integer as Bitcoin varint
/// Clear sighash templates cache
///
/// Useful for benchmarking to ensure consistent results without cache state
/// pollution between runs.
///
/// # Example
///
/// ```rust
/// use blvm_consensus::transaction_hash::clear_sighash_templates;
///
/// // Clear cache before benchmark run
/// clear_sighash_templates();
/// ```
#[cfg(all(feature = "production", feature = "benchmarking"))]
pub fn clear_sighash_templates() {
    // SIGHASH_TEMPLATES is a OnceLock<HashMap>, not wrapped in RwLock
    // OnceLock doesn't allow mutation after initialization, so we can't clear it directly.
    // This cache is currently not populated (see get_sighash_template which returns None),
    // so clearing is a no-op, but we provide the function for API consistency and future use
    // when templates are actually populated.
    // Note: If templates need to be clearable, SIGHASH_TEMPLATES should be changed to
    // RwLock<HashMap> similar to SCRIPT_CACHE and HASH_CACHE.
    let _ = SIGHASH_TEMPLATES.get();
}

fn encode_varint(value: u64) -> Vec<u8> {
    if value < 0xfd {
        vec![value as u8]
    } else if value <= 0xffff {
        let mut result = vec![0xfd];
        result.extend_from_slice(&(value as u16).to_le_bytes());
        result
    } else if value <= 0xffffffff {
        let mut result = vec![0xfe];
        result.extend_from_slice(&(value as u32).to_le_bytes());
        result
    } else {
        let mut result = vec![0xff];
        result.extend_from_slice(&value.to_le_bytes());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sighash_type_parsing() {
        assert_eq!(SighashType::from_byte(0x01).unwrap(), SighashType::All);
        assert_eq!(SighashType::from_byte(0x02).unwrap(), SighashType::None);
        assert_eq!(SighashType::from_byte(0x03).unwrap(), SighashType::Single);
        // 0x00 is accepted as AllLegacy for historical compatibility (pre-BIP66)
        // It behaves like All but preserves the 0x00 byte for sighash computation
        assert_eq!(SighashType::from_byte(0x00).unwrap(), SighashType::AllLegacy);
    }

    #[test]
    fn test_varint_encoding() {
        assert_eq!(encode_varint(0), vec![0]);
        assert_eq!(encode_varint(252), vec![252]);
        assert_eq!(encode_varint(253), vec![0xfd, 253, 0]);
        assert_eq!(encode_varint(65535), vec![0xfd, 255, 255]);
        assert_eq!(encode_varint(65536), vec![0xfe, 0, 0, 1, 0]);
    }

    #[test]
    fn test_sighash_calculation() {
        // Create a simple transaction for testing
        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [1u8; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51], // OP_1
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![
                    0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
                    0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
                ]
                .into(), // P2PKH
            }]
            .into(),
            lock_time: 0,
        };

        let prevouts = vec![TransactionOutput {
            value: 10000000000,
            script_pubkey: vec![
                0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde,
                0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
            ],
        }];

        // Test SIGHASH_ALL
        let sighash = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::All).unwrap();
        assert_eq!(sighash.len(), 32);

        // Test SIGHASH_NONE
        let sighash_none =
            calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::None).unwrap();
        assert_ne!(sighash, sighash_none);

        // Test SIGHASH_SINGLE
        let sighash_single =
            calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::Single).unwrap();
        assert_ne!(sighash, sighash_single);
    }

    #[test]
    fn test_sighash_invalid_input_index() {
        let tx = Transaction {
            version: 1,
            inputs: vec![].into(),
            outputs: vec![].into(),
            lock_time: 0,
        };

        let result = calculate_transaction_sighash(&tx, 0, &[], SighashType::All);
        assert!(result.is_err());
    }
}

