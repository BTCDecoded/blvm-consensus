//! Bitcoin wire format serialization/deserialization
//!
//! This module provides consensus-critical serialization functions that must match
//! consensus wire format exactly to ensure consensus compatibility.
//!
//! All serialization uses little-endian byte order (Bitcoin standard).

pub mod block;
pub mod transaction;
pub mod varint;

pub use block::{
    deserialize_block_header, deserialize_block_with_witnesses, serialize_block_header,
    serialize_block_with_witnesses,
};
pub use transaction::{
    deserialize_transaction, deserialize_transaction_with_witness, serialize_transaction,
    serialize_transaction_with_witness,
};
pub use varint::{decode_varint, encode_varint, VarIntError};
