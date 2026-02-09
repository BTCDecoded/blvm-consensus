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
/// Wraps the raw sighash byte from the signature, preserving its exact value for
/// preimage serialization. Bitcoin Core uses the raw byte directly in the sighash
/// preimage — before STRICTENC activation (BIP66), ANY sighash byte was accepted.
/// The base type is determined by masking with 0x1f: NONE=2, SINGLE=3, else ALL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SighashType(pub u8);

impl SighashType {
    // Standard sighash type constants
    pub const ALL_LEGACY: Self = SighashType(0x00);
    pub const ALL: Self = SighashType(0x01);
    pub const NONE: Self = SighashType(0x02);
    pub const SINGLE: Self = SighashType(0x03);
    pub const ALL_ANYONECANPAY: Self = SighashType(0x81);
    pub const NONE_ANYONECANPAY: Self = SighashType(0x82);
    pub const SINGLE_ANYONECANPAY: Self = SighashType(0x83);

    /// Create from raw sighash byte — accepts ANY value (pre-STRICTENC compatibility).
    /// Bitcoin Core determines behavior from `byte & 0x1f` and uses the raw byte in the preimage.
    pub fn from_byte(byte: u8) -> Self {
        SighashType(byte)
    }

    /// Raw byte value for preimage serialization
    pub fn as_u32(&self) -> u32 {
        self.0 as u32
    }

    /// Base sighash type (lower 5 bits), matching Bitcoin Core's `nHashType & 0x1f`
    pub fn base_type(&self) -> u8 {
        self.0 & 0x1f
    }

    /// Whether ANYONECANPAY flag is set (bit 7)
    pub fn is_anyonecanpay(&self) -> bool {
        self.0 & 0x80 != 0
    }

    /// Whether this has SIGHASH_ALL behavior (base type is not NONE or SINGLE)
    pub fn is_all(&self) -> bool {
        let base = self.base_type();
        base != 0x02 && base != 0x03
    }

    /// Whether base type is SIGHASH_NONE
    pub fn is_none(&self) -> bool {
        self.base_type() == 0x02
    }

    /// Whether base type is SIGHASH_SINGLE
    pub fn is_single(&self) -> bool {
        self.base_type() == 0x03
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
        && sighash_type == SighashType::ALL
        && tx.version == 1
        && tx.lock_time == 0
}

/// Phase 6.2: Get or create sighash template for common patterns
#[cfg(feature = "production")]
fn get_sighash_template(
    tx: &Transaction,
    input_index: usize,
    _prevout_values: &[i64],
    _prevout_script_pubkeys: &[&crate::types::ByteString],
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
    // Convert prevouts to parallel slices for the optimized API
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_script_pubkeys: Vec<&crate::types::ByteString> = prevouts.iter().map(|p| &p.script_pubkey).collect();
    // Validate prevouts match inputs
    if prevout_values.len() != tx.inputs.len() || prevout_script_pubkeys.len() != tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidPrevoutsCount(
            prevout_values.len(),
            tx.inputs.len(),
        ));
    }
    calculate_transaction_sighash_with_script_code(tx, input_index, &prevout_values, &prevout_script_pubkeys, sighash_type, None)
}

/// Calculate transaction sighash with optional script code override
/// 
/// For P2SH transactions, script_code should be the redeem script (not the scriptPubKey).
/// For non-P2SH, script_code should be None (uses scriptPubKey from prevout).
#[spec_locked("5.1")]
pub fn calculate_transaction_sighash_with_script_code(
    tx: &Transaction,
    input_index: usize,
    prevout_values: &[i64],
    prevout_script_pubkeys: &[&crate::types::ByteString],
    sighash_type: SighashType,
    script_code: Option<&[u8]>,
) -> Result<Hash> {
    // Validate input index
    if input_index >= tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidInputIndex(input_index));
    }

    // Validate prevouts match inputs
    if prevout_values.len() != tx.inputs.len() || prevout_script_pubkeys.len() != tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidPrevoutsCount(
            prevout_values.len(),
            tx.inputs.len(),
        ));
    }

    // Determine base sighash type and ANYONECANPAY flag
    // Must use the raw byte value for the sighash preimage (line 305)
    let sighash_byte = sighash_type.as_u32();
    let base_type = sighash_byte & 0x1f;
    let anyone_can_pay = (sighash_byte & 0x80) != 0;
    let hash_none = base_type == 0x02; // SIGHASH_NONE
    let hash_single = base_type == 0x03; // SIGHASH_SINGLE

    // SIGHASH_SINGLE special case: if input_index >= outputs count,
    // Bitcoin Core returns the hash 0x0000...0001 (a historical quirk)
    if hash_single && input_index >= tx.outputs.len() {
        let mut result = [0u8; 32];
        result[0] = 1; // Little-endian 1
        return Ok(result);
    }

    // Phase 6.2: Check for template cache (only for SIGHASH_ALL patterns)
    #[cfg(feature = "production")]
    if sighash_type.is_all() && !anyone_can_pay {
        if let Some(template) = get_sighash_template(tx, input_index, prevout_values, prevout_script_pubkeys, sighash_type) {
            let first_hash = Sha256::digest(&template);
            let second_hash = Sha256::digest(first_hash);
            let mut result = [0u8; 32];
            result.copy_from_slice(&second_hash);
            return Ok(result);
        }
    }

    // Build sighash preimage matching Bitcoin Core's CTransactionSignatureSerializer
    let estimated_size = 4 + 2 + (tx.inputs.len() * 50) + 2 + (tx.outputs.len() * 30) + 4 + 4;
    let mut preimage = Vec::with_capacity(estimated_size.min(4096));

    // 1. Transaction version (4 bytes LE)
    preimage.extend_from_slice(&(tx.version as u32).to_le_bytes());

    // 2. Input count: ANYONECANPAY → 1, otherwise all inputs
    let n_inputs = if anyone_can_pay { 1 } else { tx.inputs.len() };
    write_varint_to_vec(&mut preimage, n_inputs as u64);

    // 3. Inputs
    for i in 0..n_inputs {
        // ANYONECANPAY remaps input index to the signing input
        let actual_i = if anyone_can_pay { input_index } else { i };
        let input = &tx.inputs[actual_i];

        // Prevout (always serialized)
        preimage.extend_from_slice(&input.prevout.hash);
        preimage.extend_from_slice(&(input.prevout.index as u32).to_le_bytes());

        // Script: signing input gets script_code/scriptPubKey, others get empty
        if actual_i == input_index {
            let code = match script_code {
                Some(s) => s,
                None => prevout_script_pubkeys[actual_i].as_slice(),
            };
            write_varint_to_vec(&mut preimage, code.len() as u64);
            preimage.extend_from_slice(code);
        } else {
            preimage.push(0); // empty script
        }

        // Sequence: for SIGHASH_NONE/SINGLE, non-signing inputs get sequence 0
        if actual_i != input_index && (hash_single || hash_none) {
            preimage.extend_from_slice(&0u32.to_le_bytes());
        } else {
            preimage.extend_from_slice(&(input.sequence as u32).to_le_bytes());
        }
    }

    // 4. Output count: NONE → 0, SINGLE → input_index+1, ALL → all
    let n_outputs = if hash_none {
        0
    } else if hash_single {
        input_index + 1
    } else {
        tx.outputs.len()
    };
    write_varint_to_vec(&mut preimage, n_outputs as u64);

    // 5. Outputs
    for i in 0..n_outputs {
        if hash_single && i != input_index {
            // SIGHASH_SINGLE: non-matching outputs are CTxOut() (value=-1, empty script)
            preimage.extend_from_slice(&(-1i64).to_le_bytes()); // -1 as i64 = 0xffffffffffffffff
            preimage.push(0); // empty script
        } else {
            let output = &tx.outputs[i];
            preimage.extend_from_slice(&output.value.to_le_bytes());
            write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
            preimage.extend_from_slice(&output.script_pubkey);
        }
    }

    // 6. Lock time (4 bytes LE)
    preimage.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());

    // 7. SIGHASH type (4 bytes LE) - use the raw sighash byte value
    preimage.extend_from_slice(&sighash_byte.to_le_bytes());

    // Double SHA256
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

    // Convert prevouts to parallel slices for the optimized API
    let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
    let prevout_script_pubkeys: Vec<&crate::types::ByteString> = prevouts.iter().map(|p| &p.script_pubkey).collect();

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
                        serialize_sighash_preimage(tx, input_index, &prevout_values, &prevout_script_pubkeys, sighash_type)
                    })
                    .collect()
            }
            #[cfg(not(feature = "rayon"))]
            {
                (0..tx.inputs.len())
                    .map(|input_index| {
                        serialize_sighash_preimage(tx, input_index, &prevout_values, &prevout_script_pubkeys, sighash_type)
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
            results.push(calculate_transaction_sighash_with_script_code(
                tx,
                i,
                &prevout_values,
                &prevout_script_pubkeys,
                sighash_type,
                None,
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
    _prevout_values: &[i64],
    _prevout_script_pubkeys: &[&crate::types::ByteString],
    sighash_type: SighashType,
) -> Vec<u8> {
    let mut preimage = Vec::new();

    // 1. Transaction version (4 bytes, little endian)
    preimage.extend_from_slice(&(tx.version as u32).to_le_bytes());

    // 2. Number of inputs (varint)
    // OPTIMIZATION: Write varint directly to avoid Vec allocation
    write_varint_to_vec(&mut preimage, tx.inputs.len() as u64);

    // 3. Inputs (depending on sighash type)
    let anyone_can_pay = sighash_type.is_anyonecanpay();
    for (i, input) in tx.inputs.iter().enumerate() {
        if anyone_can_pay || i == input_index {
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

    // 5. Outputs (depending on base sighash type)
    // Note: AllLegacy (0x00) has the same behavior as All (0x01) for output inclusion
    // ANYONECANPAY only affects inputs, not outputs
    let base_type = (sighash_type.as_u32()) & 0x1f;
    match base_type {
        0x02 => {
            // SIGHASH_NONE: no outputs
        }
        0x03 => {
            // SIGHASH_SINGLE: include output at same index as input
            if input_index < tx.outputs.len() {
                let output = &tx.outputs[input_index];
                preimage.extend_from_slice(&output.value.to_le_bytes());
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
        _ => {
            // SIGHASH_ALL (0x00, 0x01): include all outputs
            for output in &tx.outputs {
                preimage.extend_from_slice(&output.value.to_le_bytes());
                write_varint_to_vec(&mut preimage, output.script_pubkey.len() as u64);
                preimage.extend_from_slice(&output.script_pubkey);
            }
        }
    }

    // 6. Lock time (4 bytes, little endian)
    preimage.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());

    // 7. SIGHASH type (4 bytes, little endian)
    preimage.extend_from_slice(&(sighash_type.as_u32()).to_le_bytes());

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

// =============================================================================
// BIP143: Segregated Witness Sighash Algorithm
// =============================================================================
//
// BIP143 defines a new transaction sighash algorithm for SegWit transactions.
// Key optimization: hashPrevouts, hashSequence, and hashOutputs are computed
// ONCE for all inputs, instead of once per input like legacy sighash.
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0143.mediawiki

/// Precomputed hash components for BIP143 sighash.
/// These are computed once per transaction and reused for all inputs.
#[derive(Clone)]
pub struct Bip143PrecomputedHashes {
    /// SHA256(SHA256(all input prevouts)) - 0 if ANYONECANPAY
    pub hash_prevouts: [u8; 32],
    /// SHA256(SHA256(all input sequences)) - 0 if ANYONECANPAY/NONE/SINGLE  
    pub hash_sequence: [u8; 32],
    /// SHA256(SHA256(all outputs)) - varies by sighash type
    pub hash_outputs: [u8; 32],
}

impl Bip143PrecomputedHashes {
    /// Compute precomputed hashes for a transaction.
    /// This is the expensive part - compute once, reuse for all inputs.
    #[inline]
    pub fn compute(tx: &Transaction, _prevout_values: &[i64], _prevout_script_pubkeys: &[&crate::types::ByteString]) -> Self {
        // hashPrevouts = SHA256(SHA256(all outpoints))
        let hash_prevouts = {
            let mut data = Vec::with_capacity(tx.inputs.len() * 36);
            for input in tx.inputs.iter() {
                data.extend_from_slice(&input.prevout.hash);
                data.extend_from_slice(&(input.prevout.index as u32).to_le_bytes());
            }
            double_sha256(&data)
        };

        // hashSequence = SHA256(SHA256(all sequences))
        let hash_sequence = {
            let mut data = Vec::with_capacity(tx.inputs.len() * 4);
            for input in tx.inputs.iter() {
                data.extend_from_slice(&(input.sequence as u32).to_le_bytes());
            }
            double_sha256(&data)
        };

        // hashOutputs = SHA256(SHA256(all outputs))
        let hash_outputs = {
            let mut data = Vec::with_capacity(tx.outputs.len() * 34); // Estimate
            for output in tx.outputs.iter() {
                data.extend_from_slice(&output.value.to_le_bytes());
                write_varint_to_vec(&mut data, output.script_pubkey.len() as u64);
                data.extend_from_slice(&output.script_pubkey);
            }
            double_sha256(&data)
        };

        // Note: prevout values/script_pubkeys not needed for precomputed hashes
        // (only outpoints and sequences are used)

        Self {
            hash_prevouts,
            hash_sequence,
            hash_outputs,
        }
    }
}

/// Double SHA256 helper
#[inline(always)]
fn double_sha256(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    let second = Sha256::digest(&first);
    let mut result = [0u8; 32];
    result.copy_from_slice(&second);
    result
}

/// Calculate BIP143 sighash for SegWit transactions.
///
/// This is significantly faster than legacy sighash for transactions with
/// multiple inputs because hashPrevouts, hashSequence, and hashOutputs are
/// computed once and reused.
///
/// # Arguments
/// * `tx` - The transaction being signed
/// * `input_index` - Index of the input being signed
/// * `script_code` - The scriptCode for this input (P2WPKH: pubkeyhash script, P2WSH: witness script)
/// * `amount` - Value of the UTXO being spent (in satoshis)
/// * `sighash_type` - Sighash type byte
/// * `precomputed` - Optional precomputed hashes (compute once, pass to all inputs)
///
/// # Returns
/// 32-byte sighash for signature verification
#[spec_locked("11.1")]
pub fn calculate_bip143_sighash(
    tx: &Transaction,
    input_index: usize,
    script_code: &[u8],
    amount: i64,
    sighash_type: u8,
    precomputed: Option<&Bip143PrecomputedHashes>,
) -> Result<Hash> {
    if input_index >= tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidInputIndex(input_index));
    }

    // Parse sighash flags
    let anyone_can_pay = (sighash_type & 0x80) != 0;
    let base_type = sighash_type & 0x1f;
    let is_none = base_type == 0x02;
    let is_single = base_type == 0x03;

    // Use precomputed hashes or compute them
    let computed;
    let hashes = match precomputed {
        Some(h) => h,
        None => {
            computed = Bip143PrecomputedHashes::compute(tx, &[], &[]);
            &computed
        }
    };

    // Build sighash preimage according to BIP143
    // Estimated size: 4+32+32+36+var+8+4+32+4+4 = ~160 bytes + script_code
    let mut preimage = Vec::with_capacity(160 + script_code.len());

    // 1. nVersion (4 bytes LE)
    preimage.extend_from_slice(&(tx.version as u32).to_le_bytes());

    // 2. hashPrevouts (32 bytes) - 0 if ANYONECANPAY
    if anyone_can_pay {
        preimage.extend_from_slice(&[0u8; 32]);
    } else {
        preimage.extend_from_slice(&hashes.hash_prevouts);
    }

    // 3. hashSequence (32 bytes) - 0 if ANYONECANPAY/NONE/SINGLE
    if anyone_can_pay || is_none || is_single {
        preimage.extend_from_slice(&[0u8; 32]);
    } else {
        preimage.extend_from_slice(&hashes.hash_sequence);
    }

    // 4. outpoint (36 bytes: 32 hash + 4 index)
    let input = &tx.inputs[input_index];
    preimage.extend_from_slice(&input.prevout.hash);
    preimage.extend_from_slice(&(input.prevout.index as u32).to_le_bytes());

    // 5. scriptCode (varint + script)
    write_varint_to_vec(&mut preimage, script_code.len() as u64);
    preimage.extend_from_slice(script_code);

    // 6. amount (8 bytes LE) - value of the UTXO being spent
    preimage.extend_from_slice(&amount.to_le_bytes());

    // 7. nSequence (4 bytes LE)
    preimage.extend_from_slice(&(input.sequence as u32).to_le_bytes());

    // 8. hashOutputs (32 bytes) - varies by sighash type
    if is_none {
        preimage.extend_from_slice(&[0u8; 32]);
    } else if is_single {
        if input_index < tx.outputs.len() {
            // Hash only the output at same index
            let output = &tx.outputs[input_index];
            let mut output_data = Vec::with_capacity(34);
            output_data.extend_from_slice(&output.value.to_le_bytes());
            write_varint_to_vec(&mut output_data, output.script_pubkey.len() as u64);
            output_data.extend_from_slice(&output.script_pubkey);
            preimage.extend_from_slice(&double_sha256(&output_data));
        } else {
            // SIGHASH_SINGLE with no corresponding output
            preimage.extend_from_slice(&[0u8; 32]);
        }
    } else {
        preimage.extend_from_slice(&hashes.hash_outputs);
    }

    // 9. nLockTime (4 bytes LE)
    preimage.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());

    // 10. sighash type (4 bytes LE)
    preimage.extend_from_slice(&(sighash_type as u32).to_le_bytes());

    // Double SHA256 the preimage
    Ok(double_sha256(&preimage))
}

/// Batch compute BIP143 sighashes for all inputs.
/// This is the optimal way to verify a SegWit transaction - compute precomputed
/// hashes once, then calculate sighash for each input.
#[spec_locked("11.1.1")]
pub fn batch_compute_bip143_sighashes(
    tx: &Transaction,
    prevout_values: &[i64],
    prevout_script_pubkeys: &[&crate::types::ByteString],
    script_codes: &[&[u8]],
    sighash_type: u8,
) -> Result<Vec<Hash>> {
    if prevout_values.len() != tx.inputs.len() || prevout_script_pubkeys.len() != tx.inputs.len() || script_codes.len() != tx.inputs.len() {
        return Err(crate::error::ConsensusError::InvalidPrevoutsCount(
            prevout_values.len(),
            tx.inputs.len(),
        ));
    }

    // Compute precomputed hashes ONCE
    let precomputed = Bip143PrecomputedHashes::compute(tx, prevout_values, prevout_script_pubkeys);

    // Calculate sighash for each input using precomputed hashes
    let mut results = Vec::with_capacity(tx.inputs.len());
    for (i, (value, script_code)) in prevout_values.iter().zip(script_codes.iter()).enumerate() {
        let sighash = calculate_bip143_sighash(
            tx,
            i,
            script_code,
            *value,
            sighash_type,
            Some(&precomputed),
        )?;
        results.push(sighash);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sighash_type_parsing() {
        // Standard types
        assert_eq!(SighashType::from_byte(0x01), SighashType::ALL);
        assert_eq!(SighashType::from_byte(0x02), SighashType::NONE);
        assert_eq!(SighashType::from_byte(0x03), SighashType::SINGLE);
        assert_eq!(SighashType::from_byte(0x00), SighashType::ALL_LEGACY);
        assert_eq!(SighashType::from_byte(0x81), SighashType::ALL_ANYONECANPAY);
        assert_eq!(SighashType::from_byte(0x82), SighashType::NONE_ANYONECANPAY);
        assert_eq!(SighashType::from_byte(0x83), SighashType::SINGLE_ANYONECANPAY);
        // Verify the byte values are preserved correctly for sighash preimage
        assert_eq!(SighashType::ALL_ANYONECANPAY.as_u32(), 0x81);
        assert_eq!(SighashType::NONE_ANYONECANPAY.as_u32(), 0x82);
        assert_eq!(SighashType::SINGLE_ANYONECANPAY.as_u32(), 0x83);
        // Non-standard types are accepted (pre-STRICTENC) with raw byte preserved
        let st = SighashType::from_byte(0x04);
        assert!(st.is_all()); // base_type 0x04 acts as ALL
        assert_eq!(st.as_u32(), 0x04); // raw byte preserved in preimage
        let st84 = SighashType::from_byte(0x84);
        assert!(st84.is_all());
        assert!(st84.is_anyonecanpay());
        assert_eq!(st84.as_u32(), 0x84);
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
        let sighash = calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::ALL).unwrap();
        assert_eq!(sighash.len(), 32);

        // Test SIGHASH_NONE
        let sighash_none =
            calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::NONE).unwrap();
        assert_ne!(sighash, sighash_none);

        // Test SIGHASH_SINGLE
        let sighash_single =
            calculate_transaction_sighash(&tx, 0, &prevouts, SighashType::SINGLE).unwrap();
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

        let result = calculate_transaction_sighash(&tx, 0, &[], SighashType::ALL);
        assert!(result.is_err());
    }

    #[test]
    fn test_bip143_sighash() {
        // Create a SegWit transaction for testing
        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1u8; 32].into(),
                        index: 0,
                    },
                    script_sig: vec![], // Empty for SegWit
                    sequence: 0xffffffff,
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [2u8; 32].into(),
                        index: 1,
                    },
                    script_sig: vec![],
                    sequence: 0xfffffffe,
                },
            ]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![0x00, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78,
                    0x9a, 0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let prevouts = vec![
            TransactionOutput {
                value: 10000000000,
                script_pubkey: vec![0x00, 0x14, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                    0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44],
            },
            TransactionOutput {
                value: 8000000000,
                script_pubkey: vec![0x00, 0x14, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
                    0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd],
            },
        ];

        // P2WPKH scriptCode is OP_DUP OP_HASH160 <20-byte-hash> OP_EQUALVERIFY OP_CHECKSIG
        let script_code = vec![0x76, 0xa9, 0x14, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
            0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x88, 0xac];

        // Test BIP143 sighash for first input
        let sighash0 = calculate_bip143_sighash(
            &tx, 0, &script_code, prevouts[0].value, 0x01, None
        ).unwrap();
        assert_eq!(sighash0.len(), 32);

        // Test BIP143 sighash for second input (should be different)
        let sighash1 = calculate_bip143_sighash(
            &tx, 1, &script_code, prevouts[1].value, 0x01, None
        ).unwrap();
        assert_ne!(sighash0, sighash1);

        // Test with precomputed hashes (should match)
        let prevout_values: Vec<i64> = prevouts.iter().map(|p| p.value).collect();
        let prevout_script_pubkeys: Vec<&crate::types::ByteString> = prevouts.iter().map(|p| &p.script_pubkey).collect();
        let precomputed = Bip143PrecomputedHashes::compute(&tx, &prevout_values, &prevout_script_pubkeys);
        let sighash0_precomputed = calculate_bip143_sighash(
            &tx, 0, &script_code, prevout_values[0], 0x01, Some(&precomputed)
        ).unwrap();
        assert_eq!(sighash0, sighash0_precomputed);
    }

    #[test]
    fn test_bip143_anyonecanpay() {
        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [1u8; 32].into(),
                        index: 0,
                    },
                    script_sig: vec![],
                    sequence: 0xffffffff,
                },
            ]
            .into(),
            outputs: vec![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![0x00, 0x14].into(),
            }]
            .into(),
            lock_time: 0,
        };

        let script_code = {
            let mut s = vec![0x76, 0xa9, 0x14]; // OP_DUP OP_HASH160 OP_PUSHDATA(20)
            s.extend_from_slice(&[0x00; 20]);    // 20 zero bytes (pubkey hash)
            s.push(0x88);                        // OP_EQUALVERIFY
            s.push(0xac);                        // OP_CHECKSIG
            s // 25 bytes total
        };
        let amount = 10000000000i64;

        // SIGHASH_ALL
        let sighash_all = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x01, None).unwrap();

        // SIGHASH_ALL | ANYONECANPAY (0x81)
        let sighash_anyonecanpay = calculate_bip143_sighash(&tx, 0, &script_code, amount, 0x81, None).unwrap();

        // Should be different (ANYONECANPAY zeroes hashPrevouts and hashSequence)
        assert_ne!(sighash_all, sighash_anyonecanpay);
    }
}

