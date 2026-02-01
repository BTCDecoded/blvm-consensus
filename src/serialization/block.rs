//! Block header wire format serialization/deserialization
//!
//! Bitcoin block header wire format specification.
//! Must match Bitcoin Core's serialization exactly for consensus compatibility.

use super::transaction::{deserialize_transaction, deserialize_transaction_with_witness, serialize_transaction};
use super::varint::{decode_varint, encode_varint};
use crate::error::{ConsensusError, Result};
use crate::segwit::Witness;
use crate::types::*;
use blvm_spec_lock::spec_locked;
use std::borrow::Cow;

/// Error type for block parsing failures
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockParseError {
    InsufficientBytes,
    InvalidVersion,
    InvalidTimestamp,
    InvalidBits,
    InvalidNonce,
    InvalidTransactionCount,
    InvalidWitnessMarker,
}

impl std::fmt::Display for BlockParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockParseError::InsufficientBytes => {
                write!(f, "Insufficient bytes to parse block header")
            }
            BlockParseError::InvalidVersion => write!(f, "Invalid block version"),
            BlockParseError::InvalidTimestamp => write!(f, "Invalid block timestamp"),
            BlockParseError::InvalidBits => write!(f, "Invalid block bits"),
            BlockParseError::InvalidNonce => write!(f, "Invalid block nonce"),
            BlockParseError::InvalidTransactionCount => write!(f, "Invalid transaction count"),
            BlockParseError::InvalidWitnessMarker => write!(f, "Invalid witness marker"),
        }
    }
}

impl std::error::Error for BlockParseError {}

/// Serialize a block header to Bitcoin wire format
///
/// Block header is exactly 80 bytes:
/// - Version (4 bytes, little-endian)
/// - Previous block hash (32 bytes)
/// - Merkle root (32 bytes)
/// - Timestamp (4 bytes, little-endian)
/// - Bits (4 bytes, little-endian)
/// - Nonce (4 bytes, little-endian)
#[spec_locked("3.2")]
pub fn serialize_block_header(header: &BlockHeader) -> Vec<u8> {
    let mut result = Vec::with_capacity(80);
    result.extend_from_slice(&(header.version as i32).to_le_bytes());
    result.extend_from_slice(&header.prev_block_hash);
    result.extend_from_slice(&header.merkle_root);
    result.extend_from_slice(&(header.timestamp as u32).to_le_bytes());
    result.extend_from_slice(&(header.bits as u32).to_le_bytes());
    result.extend_from_slice(&(header.nonce as u32).to_le_bytes());
    assert_eq!(result.len(), 80);
    result
}

/// Deserialize a block header from Bitcoin wire format
#[spec_locked("3.2")]
pub fn deserialize_block_header(data: &[u8]) -> Result<BlockHeader> {
    // Block header must be exactly 80 bytes
    if data.len() < 80 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            BlockParseError::InsufficientBytes.to_string(),
        )));
    }

    let mut offset = 0;

    // Version (4 bytes, little-endian) - Bitcoin uses signed 32-bit in wire format
    let version = i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as i64;
    offset += 4;

    // Previous block hash (32 bytes)
    let mut prev_block_hash = [0u8; 32];
    prev_block_hash.copy_from_slice(&data[offset..offset + 32]);
    offset += 32;

    // Merkle root (32 bytes)
    let mut merkle_root = [0u8; 32];
    merkle_root.copy_from_slice(&data[offset..offset + 32]);
    offset += 32;

    // Timestamp (4 bytes, little-endian) - Bitcoin uses u32 in wire format, but we store as u64
    let timestamp = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;
    offset += 4;

    // Bits (4 bytes, little-endian) - Bitcoin uses u32 in wire format, but we store as u64
    let bits = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;
    offset += 4;

    // Nonce (4 bytes, little-endian) - Bitcoin uses u32 in wire format, but we store as u64
    let nonce = u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;

    Ok(BlockHeader {
        version,
        prev_block_hash,
        merkle_root,
        timestamp,
        bits,
        nonce,
    })
}

/// Parse witness data from Bitcoin wire format
///
/// Witness format for each transaction:
/// - VarInt: witness stack count
/// - For each witness element:
///   - VarInt: element length
///   - Element bytes
fn parse_witness(data: &[u8], mut offset: usize) -> Result<(Witness, usize)> {
    // Witness stack count (VarInt)
    let (stack_count, varint_len) = decode_varint(&data[offset..])?;
    offset += varint_len;

    let mut witness = Witness::new();

    for _ in 0..stack_count {
        // Witness element length (VarInt)
        let (element_len, varint_len) = decode_varint(&data[offset..])?;
        offset += varint_len;

        if data.len() < offset + element_len as usize {
            return Err(ConsensusError::Serialization(Cow::Owned(
                BlockParseError::InsufficientBytes.to_string(),
            )));
        }

        // Witness element bytes
        let element = data[offset..offset + element_len as usize].to_vec();
        witness.push(element);
        offset += element_len as usize;
    }

    Ok((witness, offset))
}

/// Deserialize a complete block from Bitcoin wire format (including witness data)
///
/// Format:
/// - Block header (80 bytes)
/// - VarInt: transaction count
/// - For each transaction:
///   - Transaction (non-witness serialization)
/// - If SegWit block (marker 0x0001 found):
///   - Witness data for each transaction
// CRITICAL FIX: Return Vec<Vec<Witness>> (one Vec per transaction, each containing one Witness per input)
// This allows proper P2WSH-in-P2SH handling where we need the full witness stack per input
#[spec_locked("3.2")]
pub fn deserialize_block_with_witnesses(data: &[u8]) -> Result<(Block, Vec<Vec<Witness>>)> {
    if data.len() < 80 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            BlockParseError::InsufficientBytes.to_string(),
        )));
    }

    let mut offset = 0;

    // Parse block header (80 bytes)
    let header = deserialize_block_header(&data[offset..offset + 80])?;
    offset += 80;

    // Transaction count (VarInt)
    let (tx_count, varint_len) = decode_varint(&data[offset..])?;
    offset += varint_len;

    if tx_count == 0 {
        return Err(ConsensusError::Serialization(Cow::Owned(
            BlockParseError::InvalidTransactionCount.to_string(),
        )));
    }

    let mut transactions = Vec::new();
    // CRITICAL FIX: Store Vec<Witness> per transaction (one Witness per input)
    // This allows proper P2WSH-in-P2SH handling where we need the full witness stack per input
    let mut all_witnesses: Vec<Vec<Witness>> = Vec::new();

    // Parse transactions - each transaction handles its own SegWit format internally
    // The deserialize_transaction_with_witness function returns both the tx and its witness stacks
    // (one Witness per input)
    for _ in 0..tx_count {
        let (tx, input_witnesses, bytes_consumed) = deserialize_transaction_with_witness(&data[offset..])?;
        offset += bytes_consumed;
        transactions.push(tx);
        all_witnesses.push(input_witnesses);
    }

    // Ensure we have witnesses for all transactions
    while all_witnesses.len() < transactions.len() {
        all_witnesses.push(Vec::new());
    }
    
    // Return Vec<Vec<Witness>> - one Vec per transaction, each containing one Witness per input
    // This preserves the per-input witness structure needed for P2WSH-in-P2SH
    Ok((
        Block {
            header,
            transactions: transactions.into_boxed_slice(),
        },
        all_witnesses,
    ))
}

/// Serialize a complete block to Bitcoin wire format (including witness data).
///
/// Format:
/// - Block header (80 bytes)
/// - VarInt: transaction count
/// - If SegWit block and `include_witness` is true:
///   - Marker (0x00) and flag (0x01)
/// - For each transaction:
///   - Transaction (non-witness serialization)
/// - If `include_witness` and any witness data present:
///   - Witness data for each transaction
#[spec_locked("3.2")]
pub fn serialize_block_with_witnesses(
    block: &Block,
    witnesses: &[Witness],
    include_witness: bool,
) -> Vec<u8> {
    let mut result = Vec::new();

    // Serialize block header
    result.extend_from_slice(&serialize_block_header(&block.header));

    // Transaction count (VarInt)
    let tx_count = block.transactions.len() as u64;
    result.extend_from_slice(&encode_varint(tx_count));

    let has_witness = include_witness && witnesses.iter().any(|w| !w.is_empty());

    // SegWit marker and flag (0x00 0x01) if including witness data
    if has_witness {
        result.push(0x00);
        result.push(0x01);
    }

    // Serialize transactions (non-witness serialization)
    for tx in block.transactions.iter() {
        let tx_bytes = serialize_transaction(tx);
        result.extend_from_slice(&tx_bytes);
    }

    // Serialize witness data if requested
    if has_witness {
        for witness in witnesses.iter().take(block.transactions.len()) {
            // Witness stack count (VarInt)
            result.extend_from_slice(&encode_varint(witness.len() as u64));

            for element in witness {
                // Element length (VarInt)
                result.extend_from_slice(&encode_varint(element.len() as u64));
                // Element bytes
                result.extend_from_slice(element);
            }
        }
    }

    result
}

/// Validate that a serialized block size matches the size implied by the Block + Witness data.
///
/// This is intended for wire-level validation where a pre-serialized block is provided
/// alongside its parsed representation. It re-serializes the block (with or without
/// witness data) and compares the length against the provided size.
#[spec_locked("3.2")]
pub fn validate_block_serialized_size(
    block: &Block,
    witnesses: &[Vec<Witness>],
    include_witness: bool,
    provided_size: usize,
) -> bool {
    // Flatten Vec<Vec<Witness>> to Vec<Witness> for serialize_block_with_witnesses
    // TODO: Update serialize_block_with_witnesses to accept Vec<Vec<Witness>>
    let flattened: Vec<Witness> = witnesses.iter()
        .map(|input_witnesses| {
            let mut flattened: Witness = Vec::new();
            for witness_stack in input_witnesses {
                flattened.extend(witness_stack.clone());
            }
            flattened
        })
        .collect();
    let serialized = serialize_block_with_witnesses(block, &flattened, include_witness);
    serialized.len() == provided_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_block_header() {
        let header = BlockHeader {
            version: 1,
            prev_block_hash: [1; 32],
            merkle_root: [2; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0x12345678,
        };

        let serialized = serialize_block_header(&header);
        assert_eq!(serialized.len(), 80);

        let deserialized = deserialize_block_header(&serialized).unwrap();
        assert_eq!(deserialized.version, header.version);
        assert_eq!(deserialized.prev_block_hash, header.prev_block_hash);
        assert_eq!(deserialized.merkle_root, header.merkle_root);
        assert_eq!(deserialized.timestamp, header.timestamp);
        assert_eq!(deserialized.bits, header.bits);
        assert_eq!(deserialized.nonce, header.nonce);
    }

    #[test]
    fn test_deserialize_block_header_insufficient_bytes() {
        let data = vec![0u8; 79];
        let result = deserialize_block_header(&data);
        assert!(result.is_err());
    }
}

