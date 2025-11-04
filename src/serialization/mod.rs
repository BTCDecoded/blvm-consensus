//! Bitcoin wire format serialization/deserialization
//!
//! This module provides consensus-critical serialization functions that must match
//! Bitcoin Core's wire format exactly to ensure consensus compatibility.
//!
//! All serialization uses little-endian byte order (Bitcoin standard).

pub mod block;
pub mod transaction;
pub mod varint;

pub use block::{
    deserialize_block_header, deserialize_block_with_witnesses, serialize_block_header,
};
pub use transaction::{deserialize_transaction, serialize_transaction};
pub use varint::{decode_varint, encode_varint, VarIntError};
