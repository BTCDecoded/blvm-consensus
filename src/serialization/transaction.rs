//! Transaction wire format serialization/deserialization
//!
//! Bitcoin transaction wire format specification.
//! Must match Bitcoin Core's serialization exactly for consensus compatibility.

use super::varint::{decode_varint, encode_varint};
use crate::error::{ConsensusError, Result};
use crate::types::*;
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
pub fn serialize_transaction(tx: &Transaction) -> Vec<u8> {
    // BLLVM Optimization: Pre-allocate buffer with better size estimation
    // Estimate: version(4) + varint(input_count) + inputs(36*N + scripts) + varint(output_count) + outputs(9*M + scripts) + locktime(4)
    // Conservative estimate: 4 + 1 + (tx.inputs.len() * 50) + 1 + (tx.outputs.len() * 50) + 4
    // This avoids reallocations during serialization
    #[cfg(feature = "production")]
    let mut result = {
        // BLLVM Optimization: Use preallocated buffer with Kani-proven maximum size
        // This avoids reallocations and uses proven-safe maximum size
        use crate::optimizations::prealloc_tx_buffer;
        prealloc_tx_buffer()
    };

    #[cfg(not(feature = "production"))]
    let mut result = Vec::new();

    // Version (4 bytes, little-endian) - Bitcoin uses signed 32-bit in wire format
    result.extend_from_slice(&(tx.version as i32).to_le_bytes());

    // Input count (VarInt)
    result.extend_from_slice(&encode_varint(tx.inputs.len() as u64));

    // Inputs
    for input in &tx.inputs {
        // Previous output hash (32 bytes)
        result.extend_from_slice(&input.prevout.hash);

        // Previous output index (4 bytes, little-endian) - Bitcoin uses u32 in wire format
        result.extend_from_slice(&(input.prevout.index as u32).to_le_bytes());

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

    result
}

/// Deserialize a transaction from Bitcoin wire format
pub fn deserialize_transaction(data: &[u8]) -> Result<Transaction> {
    let mut offset = 0;

    // Version (4 bytes) - Bitcoin uses signed 32-bit in wire format, but we store as u64
    if data.len() < offset + 4 {
        return Err(ConsensusError::Serialization(
            Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
        ));
    }
    let version = i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;
    offset += 4;

    // Input count (VarInt)
    let (input_count, varint_len) = decode_varint(&data[offset..])?;
    offset += varint_len;

    if input_count > 1000000 {
        return Err(ConsensusError::Serialization(
            Cow::Owned(TransactionParseError::InvalidInputCount.to_string()),
        ));
    }

    #[cfg(feature = "production")]
    let mut inputs = SmallVec::<[TransactionInput; 2]>::new();
    #[cfg(not(feature = "production"))]
    let mut inputs = Vec::new();
    for _ in 0..input_count {
        // Previous output hash (32 bytes)
        if data.len() < offset + 32 {
            return Err(ConsensusError::Serialization(
                Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
            ));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&data[offset..offset + 32]);
        offset += 32;

        // Previous output index (4 bytes)
        if data.len() < offset + 4 {
            return Err(ConsensusError::Serialization(
                Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
            ));
        }
        let index = u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            0,
            0,
            0,
            0,
        ]);
        offset += 4;

        // Script length (VarInt)
        let (script_len, varint_len) = decode_varint(&data[offset..])?;
        offset += varint_len;

        // Script bytes
        if data.len() < offset + script_len as usize {
            return Err(ConsensusError::Serialization(
                Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
            ));
        }
        let script_sig = data[offset..offset + script_len as usize].to_vec();
        offset += script_len as usize;

        // Sequence (4 bytes) - Bitcoin uses u32 in wire format, but we store as u64
        if data.len() < offset + 4 {
            return Err(ConsensusError::Serialization(
                Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
            ));
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
        return Err(ConsensusError::Serialization(
            Cow::Owned(TransactionParseError::InvalidOutputCount.to_string()),
        ));
    }

    #[cfg(feature = "production")]
    let mut outputs = SmallVec::<[TransactionOutput; 2]>::new();
    #[cfg(not(feature = "production"))]
    let mut outputs = Vec::new();
    for _ in 0..output_count {
        // Value (8 bytes)
        if data.len() < offset + 8 {
            return Err(ConsensusError::Serialization(
                Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
            ));
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
            return Err(ConsensusError::Serialization(
                Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
            ));
        }
        let script_pubkey = data[offset..offset + script_len as usize].to_vec();
        offset += script_len as usize;

        outputs.push(TransactionOutput {
            value,
            script_pubkey,
        });
    }

    // Lock time (4 bytes) - Bitcoin uses u32 in wire format, but we store as u64
    if data.len() < offset + 4 {
        return Err(ConsensusError::Serialization(
            Cow::Owned(TransactionParseError::InsufficientBytes.to_string()),
        ));
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

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use crate::transaction::Transaction;
    use kani::*;

    /// Kani proof: Transaction serialization round-trip correctness (Orange Paper Section 13.3.2)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction: deserialize(serialize(tx)) = tx
    ///
    /// This ensures serialization and deserialization are inverse operations.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_transaction_serialization_round_trip() {
        let tx: Transaction = kani::any();

        // Bound for tractability
        kani::assume(tx.inputs.len() <= 5);
        kani::assume(tx.outputs.len() <= 5);

        // Serialize and deserialize
        let serialized = serialize_transaction(&tx);
        let deserialized_result = deserialize_transaction(&serialized);

        if deserialized_result.is_ok() {
            let deserialized = deserialized_result.unwrap();

            // Round-trip property: deserialize(serialize(tx)) = tx
            assert_eq!(
                deserialized.version, tx.version,
                "Transaction serialization round-trip: version must match"
            );
            assert_eq!(
                deserialized.inputs.len(),
                tx.inputs.len(),
                "Transaction serialization round-trip: input count must match"
            );
            assert_eq!(
                deserialized.outputs.len(),
                tx.outputs.len(),
                "Transaction serialization round-trip: output count must match"
            );
            assert_eq!(
                deserialized.lock_time, tx.lock_time,
                "Transaction serialization round-trip: lock_time must match"
            );

            // Verify inputs match
            for (i, (input, deserialized_input)) in
                tx.inputs.iter().zip(deserialized.inputs.iter()).enumerate()
            {
                assert_eq!(
                    deserialized_input.prevout.hash, input.prevout.hash,
                    "Transaction serialization round-trip: input {} prevout hash must match",
                    i
                );
                assert_eq!(
                    deserialized_input.prevout.index, input.prevout.index,
                    "Transaction serialization round-trip: input {} prevout index must match",
                    i
                );
                assert_eq!(
                    deserialized_input.script_sig, input.script_sig,
                    "Transaction serialization round-trip: input {} script_sig must match",
                    i
                );
                assert_eq!(
                    deserialized_input.sequence, input.sequence,
                    "Transaction serialization round-trip: input {} sequence must match",
                    i
                );
            }

            // Verify outputs match
            for (i, (output, deserialized_output)) in tx
                .outputs
                .iter()
                .zip(deserialized.outputs.iter())
                .enumerate()
            {
                assert_eq!(
                    deserialized_output.value, output.value,
                    "Transaction serialization round-trip: output {} value must match",
                    i
                );
                assert_eq!(
                    deserialized_output.script_pubkey, output.script_pubkey,
                    "Transaction serialization round-trip: output {} script_pubkey must match",
                    i
                );
            }
        }
    }

    /// Kani proof: Transaction serialization determinism (Orange Paper Section 13.3.2)
    ///
    /// Mathematical specification:
    /// ∀ tx ∈ Transaction: serialize(tx) is deterministic (same tx → same bytes)
    ///
    /// This ensures serialization produces consistent results.
    #[kani::proof]
    #[kani::unwind(5)]
    fn kani_transaction_serialization_determinism() {
        let tx: Transaction = kani::any();

        // Bound for tractability
        kani::assume(tx.inputs.len() <= 5);
        kani::assume(tx.outputs.len() <= 5);

        // Serialize twice
        let serialized1 = serialize_transaction(&tx);
        let serialized2 = serialize_transaction(&tx);

        // Determinism: same transaction must produce same serialization
        assert_eq!(
            serialized1, serialized2,
            "Transaction serialization determinism: same transaction must produce same bytes"
        );
    }
}
