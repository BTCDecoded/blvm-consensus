//! Core Bitcoin types for consensus validation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "production")]
use smallvec::SmallVec;

// Re-export smallvec for macro use in other crates
#[cfg(feature = "production")]
pub use smallvec;

/// Helper macro to create Transaction inputs/outputs that works with both Vec and SmallVec
#[cfg(feature = "production")]
#[macro_export]
macro_rules! tx_inputs {
    ($($item:expr),* $(,)?) => {
        {
            $crate::smallvec::SmallVec::from_vec(vec![$($item),*])
        }
    };
}

#[cfg(not(feature = "production"))]
#[macro_export]
macro_rules! tx_inputs {
    ($($item:expr),* $(,)?) => {
        vec![$($item),*]
    };
}

#[cfg(feature = "production")]
#[macro_export]
macro_rules! tx_outputs {
    ($($item:expr),* $(,)?) => {
        {
            $crate::smallvec::SmallVec::from_vec(vec![$($item),*])
        }
    };
}

#[cfg(not(feature = "production"))]
#[macro_export]
macro_rules! tx_outputs {
    ($($item:expr),* $(,)?) => {
        vec![$($item),*]
    };
}

/// Hash type: 256-bit hash
pub type Hash = [u8; 32];

/// Byte string type
pub type ByteString = Vec<u8>;

/// Natural number type
pub type Natural = u64;

/// Integer type  
pub type Integer = i64;

/// Block height: newtype wrapper for type safety
///
/// Prevents mixing up block heights with other u64 values (e.g., timestamps, counts).
/// Uses `#[repr(transparent)]` for zero-cost abstraction - same memory layout as u64.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockHeight(pub u64);

impl BlockHeight {
    /// Create a new BlockHeight from a u64
    #[inline]
    pub fn new(height: u64) -> Self {
        BlockHeight(height)
    }

    /// Get the inner u64 value
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for BlockHeight {
    #[inline]
    fn from(height: u64) -> Self {
        BlockHeight(height)
    }
}

impl From<BlockHeight> for u64 {
    #[inline]
    fn from(height: BlockHeight) -> Self {
        height.0
    }
}

impl std::ops::Deref for BlockHeight {
    type Target = u64;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Block hash: newtype wrapper for type safety
///
/// Prevents mixing up block hashes with other Hash values (e.g., transaction hashes, merkle roots).
/// Uses `#[repr(transparent)]` for zero-cost abstraction - same memory layout as Hash.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockHash(pub Hash);

impl BlockHash {
    /// Create a new BlockHash from a Hash
    #[inline]
    pub fn new(hash: Hash) -> Self {
        BlockHash(hash)
    }

    /// Get the inner Hash value
    #[inline]
    pub fn as_hash(self) -> Hash {
        self.0
    }

    /// Get a reference to the inner Hash
    #[inline]
    pub fn as_hash_ref(&self) -> &Hash {
        &self.0
    }
}

impl From<Hash> for BlockHash {
    #[inline]
    fn from(hash: Hash) -> Self {
        BlockHash(hash)
    }
}

impl From<BlockHash> for Hash {
    #[inline]
    fn from(hash: BlockHash) -> Self {
        hash.0
    }
}

impl std::ops::Deref for BlockHash {
    type Target = Hash;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// OutPoint: 𝒪 = ℍ × ℕ
///
/// Performance optimization: Cache-line aligned for better memory access patterns
#[repr(align(64))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutPoint {
    pub hash: Hash,
    pub index: Natural,
}

/// Transaction Input: ℐ = 𝒪 × 𝕊 × ℕ
///
/// Performance optimization: Hot fields (prevout, sequence) grouped together
/// for better cache locality. script_sig is accessed less frequently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionInput {
    pub prevout: OutPoint,      // Hot: 40 bytes (frequently accessed)
    pub sequence: Natural,      // Hot: 8 bytes (frequently accessed)
    pub script_sig: ByteString, // Cold: Vec (pointer, less frequently accessed)
}

/// Transaction Output: 𝒯 = ℤ × 𝕊
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionOutput {
    pub value: Integer,
    pub script_pubkey: ByteString,
}

/// Transaction: 𝒯𝒳 = ℕ × ℐ* × 𝒯* × ℕ
///
/// Performance optimization: Uses SmallVec for inputs/outputs to eliminate
/// heap allocations for the common case of 1-2 inputs/outputs (80%+ of transactions).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transaction {
    pub version: Natural,
    #[cfg(feature = "production")]
    pub inputs: SmallVec<[TransactionInput; 2]>,
    #[cfg(not(feature = "production"))]
    pub inputs: Vec<TransactionInput>,
    #[cfg(feature = "production")]
    pub outputs: SmallVec<[TransactionOutput; 2]>,
    #[cfg(not(feature = "production"))]
    pub outputs: Vec<TransactionOutput>,
    pub lock_time: Natural,
}

/// Block Header: ℋ = ℤ × ℍ × ℍ × ℕ × ℕ × ℕ
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHeader {
    pub version: Integer,
    pub prev_block_hash: Hash,
    pub merkle_root: Hash,
    pub timestamp: Natural,
    pub bits: Natural,
    pub nonce: Natural,
}

/// Block: ℬ = ℋ × 𝒯𝒳*
///
/// Performance optimization: Uses Box<[Transaction]> instead of Vec<Transaction>
/// since transactions are never modified after block creation. This saves 8 bytes
/// (no capacity field) and provides better cache usage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Box<[Transaction]>,
}

/// UTXO: 𝒰 = ℤ × 𝕊 × ℕ
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UTXO {
    pub value: Integer,
    pub script_pubkey: ByteString,
    pub height: Natural,
}

/// UTXO Set: 𝒰𝒮 = 𝒪 → 𝒰
pub type UtxoSet = HashMap<OutPoint, UTXO>;

/// Validation result
///
/// Important: This result must be checked - ignoring validation results
/// may cause consensus violations or security issues.
#[must_use = "Validation result must be checked - ignoring may cause consensus violations"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    Valid,
    Invalid(String),
}

/// Script execution context
#[derive(Debug, Clone)]
pub struct ScriptContext {
    pub script_sig: ByteString,
    pub script_pubkey: ByteString,
    pub witness: Option<ByteString>,
    pub flags: u32,
}

/// Block validation context
#[derive(Debug, Clone)]
pub struct BlockContext {
    pub height: Natural,
    pub prev_headers: Vec<BlockHeader>,
    pub utxo_set: UtxoSet,
}
