//! Transaction wire format serialization/deserialization
//!
//! Bitcoin transaction wire format specification.
//! Must match Bitcoin protocol serialization exactly for consensus compatibility.

use super::varint::{decode_varint, encode_varint};
use crate::error::{ConsensusError, Result};
use crate::types::*;
use blvm_spec_lock::spec_locked;
use std::borrow::Cow;

#[cfg(feature = "production")]
use smallvec::SmallVec;

/// Error type for transaction parsing failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionParseError {
    InsufficientBytes,
    InvalidVersion,
    InvalidInputCount,
    InvalidOutputCount,
    InvalidScriptLength,
    InvalidLockTime,
}

impl std::fmt::Display for TransactionParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransactionParseError::InsufficientBytes => {
                write!(f, "Insufficient bytes to parse transaction")
            }
            TransactionParseError::InvalidVersion => write!(f, "Invalid transaction version"),
            TransactionParseError::InvalidInputCount => write!(f, "Invalid input count"),
            TransactionParseError::InvalidOutputCount => write!(f, "Invalid output count"),
            TransactionParseError::InvalidScriptLength => write!(f, "Invalid script length"),
            TransactionParseError::InvalidLockTime => write!(f, "Invalid lock time"),
        }
    }
}

impl std::error::Error for TransactionParseError {}

/// Serialize a transaction to Bitcoin wire format
///
/// Format (non-SegWit):
/// - Version (4 bytes, little-endian)
/// - Input count (VarInt)
/// - For each input:
///   - Previous output hash (32 bytes)
///   - Previous output index (4 bytes, little-endian)
///   - Script length (VarInt)
///   - Script bytes
///   - Sequence (4 bytes, little-endian)
/// - Output count (VarInt)
/// - For each output:
///   - Value (8 bytes, little-endian)
///   - Script length (VarInt)
///   - Script bytes
/// - Lock time (4 bytes, little-endian)
#[inline(always)]
#[spec_locked("3.2")]
pub fn serialize_transaction(tx: &Transaction) -> Vec<u8> {
    serialize_transaction_inner(tx)
}

/// Append serialized transaction to buffer (shared logic for into/inner).
#[inline(always)]
fn serialize_transaction_append(result: &mut Vec<u8>, tx: &Transaction) {
    result.extend_from_slice(&(tx.version as i32).to_le_bytes());

    // Input count (VarInt)
    result.extend_from_slice(&encode_varint(tx.inputs.len() as u64));

    // Inputs
    for input in &tx.inputs {
        // Previous output hash (32 bytes)
        result.extend_from_slice(&input.prevout.hash);

        // Previous output index (4 bytes, little-endian) - Bitcoin uses u32 in wire format
        result.extend_from_slice(&input.prevout.index.to_le_bytes());

        // Script length (VarInt)
        result.extend_from_slice(&encode_varint(input.script_sig.len() as u64));

        // Script bytes
        result.extend_from_slice(&input.script_sig);

        // Sequence (4 bytes, little-endian) - Bitcoin uses u32 in wire format
        result.extend_from_slice(&(input.sequence as u32).to_le_bytes());
    }

    // Output count (VarInt)
    result.extend_from_slice(&encode_varint(tx.outputs.len() as u64));

    // Outputs
    for output in &tx.outputs {
        // Value (8 bytes, little-endian)
        result.extend_from_slice(&(output.value as u64).to_le_bytes());

        // Script length (VarInt)
        result.extend_from_slice(&encode_varint(output.script_pubkey.len() as u64));

        // Script bytes
        result.extend_from_slice(&output.script_pubkey);
    }

    // Lock time (4 bytes, little-endian) - Bitcoin uses u32 in wire format
    result.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());
}

/// Serialize transaction into an existing buffer (for thread-local reuse in IBD).
///
/// Clears `dst`, writes serialized bytes, returns the length.
#[inline(always)]
#[spec_locked("3.2")]
pub fn serialize_transaction_into(dst: &mut Vec<u8>, tx: &Transaction) -> usize {
    dst.clear();
    serialize_transaction_append(dst, tx);
    dst.len()
}

/// Inner serialization function (actual implementation)
#[inline(always)]
fn serialize_transaction_inner(tx: &Transaction) -> Vec<u8> {
    #[cfg(feature = "production")]
    let mut result = {
        use crate::optimizations::prealloc_tx_buffer;
        prealloc_tx_buffer()
    };

    #[cfg(not(feature = "production"))]
    let mut result = Vec::new();

    serialize_transaction_append(&mut result, tx);
    result
}

/// Serialize a transaction in SegWit wire format (version + 0x00 0x01 + inputs + outputs + witness + locktime).
///
/// Used by BIP152 cmpctblock prefilled txs and blocktxn, which use TX_WITH_WITNESS per Bitcoin Core.
/// `witnesses` must have one `Witness` (stack) per input; empty stacks are valid.
#[spec_locked("8.2.2")]
pub fn serialize_transaction_with_witness(
    tx: &Transaction,
    witnesses: &[crate::witness::Witness],
) -> Vec<u8> {
    assert_eq!(
        witnesses.len(),
        tx.inputs.len(),
        "witness count must match input count"
    );
    let mut result = Vec::new();
    result.extend_from_slice(&(tx.version as i32).to_le_bytes());
    result.push(0x00);
    result.push(0x01);
    result.extend_from_slice(&encode_varint(tx.inputs.len() as u64));
    for input in &tx.inputs {
        result.extend_from_slice(&input.prevout.hash);
        result.extend_from_slice(&input.prevout.index.to_le_bytes());
        result.extend_from_slice(&encode_varint(input.script_sig.len() as u64));
        result.extend_from_slice(&input.script_sig);
        result.extend_from_slice(&(input.sequence as u32).to_le_bytes());
    }
    result.extend_from_slice(&encode_varint(tx.outputs.len() as u64));
    for output in &tx.outputs {
        result.extend_from_slice(&(output.value as u64).to_le_bytes());
        result.extend_from_slice(&encode_varint(output.script_pubkey.len() as u64));
        result.extend_from_slice(&output.script_pubkey);
    }
    for witness in witnesses {
        result.extend_from_slice(&encode_varint(witness.len() as u64));
        for element in witness {
            result.extend_from_slice(&encode_varint(element.len() as u64));
            result.extend_from_slice(element);
        }
    }
    result.extend_from_slice(&(tx.lock_time as u32).to_le_bytes());
    result
}

/// Deserialize a transaction from Bitcoin wire format
#[spec_locked("3.2")]
pub fn deserialize_transaction(data: &[u8]) -> Result<Transaction> {
    let mut offset = 0;

    // Version (4 bytes) - Bitcoin uses signed 32-bit in wire format, but we store as u64
    if data.len() < offset + 4 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InsufficientBytes.to_string(),
        )));
    }
    let version = i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;
    offset += 4;

    // Check for SegWit marker (0x00 0x01) after version
    // In SegWit transactions: version (4) + marker (1) + flag (1) + inputs...
    let is_segwit = if data.len() >= offset + 2 {
        data[offset] == 0x00 && data[offset + 1] == 0x01
    } else {
        false
    };

    if is_segwit {
        offset += 2; // Skip marker (0x00) and flag (0x01)
    }

    // Input count (VarInt)
    let (input_count, varint_len) = decode_varint(&data[offset..])?;
    offset += varint_len;

    if input_count > 1000000 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InvalidInputCount.to_string(),
        )));
    }

    #[cfg(feature = "production")]
    let mut inputs = SmallVec::<[TransactionInput; 2]>::new();
    #[cfg(not(feature = "production"))]
    let mut inputs = Vec::new();
    for _ in 0..input_count {
        // Previous output hash (32 bytes)
        if data.len() < offset + 32 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&data[offset..offset + 32]);
        offset += 32;

        // Previous output index (4 bytes)
        if data.len() < offset + 4 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let index = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Script length (VarInt)
        let (script_len, varint_len) = decode_varint(&data[offset..])?;
        offset += varint_len;

        // Script bytes
        if data.len() < offset + script_len as usize {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let script_sig = data[offset..offset + script_len as usize].to_vec();
        offset += script_len as usize;

        // Sequence (4 bytes) - Bitcoin uses u32 in wire format, but we store as u64
        if data.len() < offset + 4 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let sequence = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as u64;
        offset += 4;

        inputs.push(TransactionInput {
            prevout: OutPoint { hash, index },
            script_sig,
            sequence,
        });
    }

    // Output count (VarInt)
    let (output_count, varint_len) = decode_varint(&data[offset..])?;
    offset += varint_len;

    if output_count > 1000000 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InvalidOutputCount.to_string(),
        )));
    }

    #[cfg(feature = "production")]
    let mut outputs = SmallVec::<[TransactionOutput; 2]>::new();
    #[cfg(not(feature = "production"))]
    let mut outputs = Vec::new();
    for _ in 0..output_count {
        // Value (8 bytes)
        if data.len() < offset + 8 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let value = i64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        // Script length (VarInt)
        let (script_len, varint_len) = decode_varint(&data[offset..])?;
        offset += varint_len;

        // Script bytes
        if data.len() < offset + script_len as usize {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let script_pubkey = data[offset..offset + script_len as usize].to_vec();
        offset += script_len as usize;

        outputs.push(TransactionOutput {
            value,
            script_pubkey,
        });
    }

    // For SegWit transactions, skip witness data (one stack per input)
    // Witness data comes after outputs but before locktime
    if is_segwit {
        for _ in 0..input_count {
            // Number of stack items for this input
            let (stack_count, varint_len) = decode_varint(&data[offset..])?;
            offset += varint_len;
            
            // Skip each stack item
            for _ in 0..stack_count {
                let (item_len, varint_len) = decode_varint(&data[offset..])?;
                offset += varint_len;
                
                if data.len() < offset + item_len as usize {
                    return Err(ConsensusError::Serialization(Cow::Owned(
                        TransactionParseError::InsufficientBytes.to_string(),
                    )));
                }
                offset += item_len as usize;
            }
        }
    }

    // Lock time (4 bytes) - Bitcoin uses u32 in wire format, but we store as u64
    if data.len() < offset + 4 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InsufficientBytes.to_string(),
        )));
    }
    let lock_time = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;

    Ok(Transaction {
        version,
        inputs,
        outputs,
        lock_time,
    })
}

/// Deserialize a transaction from Bitcoin wire format, returning transaction, witness, and bytes consumed
/// 
/// This is used by block deserialization to track how many bytes each transaction uses.
#[spec_locked("8.2.2")]
pub fn deserialize_transaction_with_witness(data: &[u8]) -> Result<(Transaction, Vec<crate::witness::Witness>, usize)> {
    let mut offset = 0;

    // Version (4 bytes)
    if data.len() < offset + 4 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InsufficientBytes.to_string(),
        )));
    }
    let version = i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;
    offset += 4;

    // Check for SegWit marker (0x00 0x01) after version
    let is_segwit = if data.len() >= offset + 2 {
        data[offset] == 0x00 && data[offset + 1] == 0x01
    } else {
        false
    };

    if is_segwit {
        offset += 2; // Skip marker (0x00) and flag (0x01)
    }

    // Input count (VarInt)
    let (input_count, varint_len) = decode_varint(&data[offset..])?;
    offset += varint_len;

    if input_count > 1000000 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InvalidInputCount.to_string(),
        )));
    }

    #[cfg(feature = "production")]
    let mut inputs = SmallVec::<[TransactionInput; 2]>::new();
    #[cfg(not(feature = "production"))]
    let mut inputs = Vec::new();
    
    for _ in 0..input_count {
        // Previous output hash (32 bytes)
        if data.len() < offset + 32 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&data[offset..offset + 32]);
        offset += 32;

        // Previous output index (4 bytes)
        if data.len() < offset + 4 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let index = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        offset += 4;

        // Script length (VarInt)
        let (script_len, varint_len) = decode_varint(&data[offset..])?;
        offset += varint_len;

        // Script bytes
        if data.len() < offset + script_len as usize {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let script_sig = data[offset..offset + script_len as usize].to_vec();
        offset += script_len as usize;

        // Sequence (4 bytes)
        if data.len() < offset + 4 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let sequence = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as u64;
        offset += 4;

        inputs.push(TransactionInput {
            prevout: OutPoint { hash, index },
            script_sig,
            sequence,
        });
    }

    // Output count (VarInt)
    let (output_count, varint_len) = decode_varint(&data[offset..])?;
    offset += varint_len;

    if output_count > 1000000 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InvalidOutputCount.to_string(),
        )));
    }

    #[cfg(feature = "production")]
    let mut outputs = SmallVec::<[TransactionOutput; 2]>::new();
    #[cfg(not(feature = "production"))]
    let mut outputs = Vec::new();
    
    for _ in 0..output_count {
        // Value (8 bytes)
        if data.len() < offset + 8 {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let value = i64::from_le_bytes([
            data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
        ]);
        offset += 8;

        // Script length (VarInt)
        let (script_len, varint_len) = decode_varint(&data[offset..])?;
        offset += varint_len;

        // Script bytes
        if data.len() < offset + script_len as usize {
            return Err(ConsensusError::Serialization(Cow::Owned(
                TransactionParseError::InsufficientBytes.to_string(),
            )));
        }
        let script_pubkey = data[offset..offset + script_len as usize].to_vec();
        offset += script_len as usize;

        outputs.push(TransactionOutput {
            value,
            script_pubkey,
        });
    }

    // Parse witness data for SegWit transactions
    // CRITICAL FIX: Return Vec<Witness> (one Witness per input) instead of flattening
    // Each Witness is Vec<ByteString> representing the witness stack for that input
    use crate::witness::Witness;
    let mut all_witnesses: Vec<Witness> = Vec::new();
    if is_segwit {
        for _ in 0..input_count {
            // Number of stack items for this input
            let (stack_count, varint_len) = decode_varint(&data[offset..])?;
            offset += varint_len;
            
            // Create witness stack for this input
            let mut witness_stack: Witness = Vec::new();
            
            // Parse each stack item
            for _ in 0..stack_count {
                let (item_len, varint_len) = decode_varint(&data[offset..])?;
                offset += varint_len;
                
                if data.len() < offset + item_len as usize {
                    return Err(ConsensusError::Serialization(Cow::Owned(
                        TransactionParseError::InsufficientBytes.to_string(),
                    )));
                }
                witness_stack.push(data[offset..offset + item_len as usize].to_vec());
                offset += item_len as usize;
            }
            
            all_witnesses.push(witness_stack);
        }
    } else {
        // Non-SegWit: create empty witness stacks for all inputs
        for _ in 0..input_count {
            all_witnesses.push(Witness::new());
        }
    }

    // Lock time (4 bytes)
    if data.len() < offset + 4 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            TransactionParseError::InsufficientBytes.to_string(),
        )));
    }
    let lock_time = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;
    offset += 4;

    let tx = Transaction {
        version,
        inputs,
        outputs,
        lock_time,
    };

    Ok((tx, all_witnesses, offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_round_trip() {
        let tx = Transaction {
            version: 1,
            inputs: crate::tx_inputs![TransactionInput {
                prevout: OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![0x51], // OP_1
                sequence: 0xffffffff,
            }],
            outputs: crate::tx_outputs![TransactionOutput {
                value: 5000000000,
                script_pubkey: vec![0x51], // OP_1
            }],
            lock_time: 0,
        };

        let serialized = serialize_transaction(&tx);
        let deserialized = deserialize_transaction(&serialized).unwrap();

        assert_eq!(deserialized.version, tx.version);
        assert_eq!(deserialized.inputs.len(), tx.inputs.len());
        assert_eq!(
            deserialized.inputs[0].prevout.hash,
            tx.inputs[0].prevout.hash
        );
        assert_eq!(
            deserialized.inputs[0].prevout.index,
            tx.inputs[0].prevout.index
        );
        assert_eq!(deserialized.inputs[0].script_sig, tx.inputs[0].script_sig);
        assert_eq!(deserialized.inputs[0].sequence, tx.inputs[0].sequence);
        assert_eq!(deserialized.outputs.len(), tx.outputs.len());
        assert_eq!(deserialized.outputs[0].value, tx.outputs[0].value);
        assert_eq!(
            deserialized.outputs[0].script_pubkey,
            tx.outputs[0].script_pubkey
        );
        assert_eq!(deserialized.lock_time, tx.lock_time);
    }

    #[test]
    fn test_deserialize_insufficient_bytes() {
        assert!(deserialize_transaction(&[]).is_err());
        assert!(deserialize_transaction(&[0, 0, 0, 0]).is_err()); // Only version
        assert!(deserialize_transaction(&[0, 0, 0, 0, 1]).is_err()); // Version + input count
    }
}

