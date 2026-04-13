//! AUTO-GENERATED from Orange Paper - DO NOT EDIT
//! Run: cargo spec-lock extract-property-tests --spec-path ... --output ...

#![cfg(test)]
#![cfg(feature = "property-tests")]
use proptest::prelude::*;

/// Property (Transaction Serialization Round-Trip) - Orange Paper 8.2.2
#[test]
fn prop_transaction_serialization_round_trip() {
    proptest!(|(tx in blvm_consensus::test_utils::transaction_strategy())| {
        let bytes = blvm_consensus::serialization::serialize_transaction(&tx);
        let tx2 = blvm_consensus::serialization::deserialize_transaction(&bytes).unwrap();
        prop_assert_eq!(tx, tx2);
    });
}

/// Property (SegWit Transaction Serialization Round-Trip) - Orange Paper 8.2.2
#[test]
fn prop_segwit_transaction_serialization_round_trip() {
    proptest!(|((tx, w) in blvm_consensus::test_utils::transaction_with_witness_strategy())| {
        let bytes = blvm_consensus::serialization::serialize_transaction_with_witness(&tx, &w);
        let (tx2, w2, _) = blvm_consensus::serialization::deserialize_transaction_with_witness(&bytes).unwrap();
        prop_assert_eq!(tx, tx2);
        prop_assert_eq!(w, w2);
    });
}

/// Property (Block Header Serialization Round-Trip) - Orange Paper 8.2.2
#[test]
fn prop_block_header_serialization_round_trip() {
    use blvm_consensus::types::BlockHeader;
    // Header wire format uses i32 version and u32 timestamp/bits/nonce (see serialize_block_header).
    proptest!(|(v in any::<i32>().prop_map(i64::from), prev in prop::array::uniform32(any::<u8>()), mr in prop::array::uniform32(any::<u8>()), ts in 0u64..=u32::MAX as u64, bits in 0u64..=u32::MAX as u64, nonce in 0u64..=u32::MAX as u64)| {
        let header = BlockHeader { version: v, prev_block_hash: prev, merkle_root: mr, timestamp: ts, bits, nonce };
        let bytes = blvm_consensus::serialization::serialize_block_header(&header);
        let header2 = blvm_consensus::serialization::deserialize_block_header(&bytes).unwrap();
        prop_assert_eq!(header.version, header2.version);
        prop_assert_eq!(header.prev_block_hash, header2.prev_block_hash);
        prop_assert_eq!(header.merkle_root, header2.merkle_root);
        prop_assert_eq!(header.timestamp, header2.timestamp);
        prop_assert_eq!(header.bits, header2.bits);
        prop_assert_eq!(header.nonce, header2.nonce);
    });
}

// Serialization determinism: block header round-trip covered above. Locktime encoding
// round-trip and BIP65/BIP112 consistency require EncodeLocktime/DecodeLocktime bindings.
