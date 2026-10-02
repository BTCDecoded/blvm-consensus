//! Shared helpers for mainnet golden vector tests.

use blvm_consensus::block::calculate_tx_id;
use blvm_consensus::crypto::OptimizedSha256;
use blvm_consensus::serialization::block::{
    deserialize_block_header, deserialize_block_with_witnesses, serialize_block_header,
    serialize_block_with_witnesses,
};
use blvm_consensus::serialization::transaction::{
    deserialize_transaction_with_witness, serialize_transaction, serialize_transaction_with_witness,
};
use blvm_consensus::types::{Block, BlockHeader, Hash, Transaction, Witness};

pub fn hex_decode(hex: &str) -> Vec<u8> {
    hex::decode(hex.replace(['\n', ' ', '\t'], "")).expect("valid hex fixture")
}

/// Parse display-order hash hex (explorer convention) into internal wire byte order.
pub fn hash_from_display(hex: &str) -> Hash {
    let mut bytes = hex_decode(hex);
    bytes.reverse();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

/// FIPS / NIST SHA-256 test vectors (digest bytes in standard big-endian hex order).
pub fn hash_from_fips_hex(hex: &str) -> Hash {
    let bytes = hex_decode(hex);
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    out
}

pub fn block_hash(header: &BlockHeader) -> Hash {
    OptimizedSha256::new().hash256(&serialize_block_header(header))
}

pub fn calculate_wtxid(tx: &Transaction, witnesses: &[Witness]) -> Hash {
    let has_witness = witnesses.iter().any(|stack| !stack.is_empty());
    if !has_witness {
        return calculate_tx_id(tx);
    }
    let bytes = serialize_transaction_with_witness(tx, witnesses);
    OptimizedSha256::new().hash256(&bytes)
}

fn tx_uses_segwit_framing(bytes: &[u8]) -> bool {
    bytes.len() >= 6 && bytes[4] == 0x00 && bytes[5] == 0x01
}

pub fn tx_roundtrip_bytes(original: &[u8], tx: &Transaction, witnesses: &[Witness]) -> Vec<u8> {
    if tx_uses_segwit_framing(original) {
        serialize_transaction_with_witness(tx, witnesses)
    } else {
        serialize_transaction(tx)
    }
}

pub struct DecodedTx {
    pub tx: Transaction,
    pub witnesses: Vec<Witness>,
}

pub fn decode_tx_full(bytes: &[u8]) -> DecodedTx {
    let (tx, witnesses, bytes_consumed) =
        deserialize_transaction_with_witness(bytes).expect("tx decode");
    assert_eq!(
        bytes_consumed,
        bytes.len(),
        "tx decoder must consume entire buffer"
    );
    DecodedTx { tx, witnesses }
}

pub fn assert_tx_roundtrip(hex: &str) -> DecodedTx {
    let bytes = hex_decode(hex);
    let decoded = decode_tx_full(&bytes);
    let roundtrip = tx_roundtrip_bytes(&bytes, &decoded.tx, &decoded.witnesses);
    assert_eq!(roundtrip, bytes, "tx re-encode must match mainnet bytes");
    decoded
}

pub struct DecodedBlock {
    pub block: Block,
    pub witnesses: Vec<Vec<Witness>>,
}

pub fn decode_block_full(bytes: &[u8]) -> DecodedBlock {
    let (block, witnesses) = deserialize_block_with_witnesses(bytes).expect("block decode");
    DecodedBlock { block, witnesses }
}

pub fn block_roundtrip_bytes(
    block: &Block,
    witnesses: &[Vec<Witness>],
    include_witness: bool,
) -> Vec<u8> {
    serialize_block_with_witnesses(block, witnesses, include_witness)
}

pub fn block_has_witness_data(witnesses: &[Vec<Witness>]) -> bool {
    witnesses
        .iter()
        .any(|tx_witnesses| tx_witnesses.iter().any(|stack| !stack.is_empty()))
}

pub fn assert_block_roundtrip(hex: &str) -> DecodedBlock {
    let bytes = hex_decode(hex);
    let decoded = decode_block_full(&bytes);
    let include_witness = block_has_witness_data(&decoded.witnesses);
    let roundtrip = block_roundtrip_bytes(&decoded.block, &decoded.witnesses, include_witness);
    assert_eq!(roundtrip, bytes, "block re-encode must match mainnet bytes");
    decoded
}

pub fn assert_header_roundtrip(hex: &str) -> BlockHeader {
    let bytes = hex_decode(hex);
    assert_eq!(bytes.len(), 80, "header fixture must be 80 bytes");
    let header = deserialize_block_header(&bytes).expect("header decode");
    assert_eq!(serialize_block_header(&header), bytes);
    header
}
