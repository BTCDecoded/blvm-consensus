//! Essential Bitcoin types for consensus validation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "production")]
use smallvec::SmallVec;

#[cfg(feature = "production")]
use rustc_hash::FxHashMap;

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

/// Shareable script_pubkey for UTXO: Arc<[u8]> to avoid clones in IBD apply path.
/// Clone is O(1); serialization format identical to ByteString for on-disk compatibility.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SharedByteString(std::sync::Arc<[u8]>);

impl std::fmt::Debug for SharedByteString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SharedByteString").field(&self.0.as_ref()).finish()
    }
}

impl std::ops::Deref for SharedByteString {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl Serialize for SharedByteString {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.as_ref().serialize(s)
    }
}

impl<'de> Deserialize<'de> for SharedByteString {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v: Vec<u8> = Deserialize::deserialize(d)?;
        Ok(Self(std::sync::Arc::from(v.into_boxed_slice())))
    }
}

impl From<ByteString> for SharedByteString {
    #[inline]
    fn from(v: ByteString) -> Self {
        Self(std::sync::Arc::from(v.into_boxed_slice()))
    }
}

impl From<&[u8]> for SharedByteString {
    #[inline]
    fn from(v: &[u8]) -> Self {
        Self(std::sync::Arc::from(v))
    }
}

impl Default for SharedByteString {
    #[inline]
    fn default() -> Self {
        Self(std::sync::Arc::from([]))
    }
}

impl AsRef<[u8]> for SharedByteString {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl SharedByteString {
    /// Get the underlying Arc<[u8]> for storage in script context (avoids clone of bytes).
    #[inline]
    pub fn as_arc(&self) -> std::sync::Arc<[u8]> {
        std::sync::Arc::clone(&self.0)
    }
}

/// Natural number type
pub type Natural = u64;

/// Integer type  
pub type Integer = i64;

/// Network type for consensus validation
///
/// Used to determine activation heights for various BIPs and consensus rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Network {
    /// Bitcoin mainnet
    Mainnet,
    /// Bitcoin testnet
    Testnet,
    /// Bitcoin regtest (local testing)
    Regtest,
}

/// Time context for consensus validation
///
/// Provides network time and median time-past for timestamp validation.
/// Required for proper block header timestamp validation (BIP113).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeContext {
    /// Current network time (Unix timestamp)
    /// Used to reject blocks with timestamps too far in the future
    pub network_time: u64,
    /// Median time-past of previous 11 blocks (BIP113)
    /// Used to reject blocks with timestamps before median time-past
    pub median_time_past: u64,
}

impl Network {
    /// Get network from environment variable or default to mainnet
    ///
    /// Checks `BITCOIN_NETWORK` environment variable:
    /// - "testnet" -> Network::Testnet
    /// - "regtest" -> Network::Regtest
    /// - otherwise -> Network::Mainnet
    pub fn from_env() -> Self {
        match std::env::var("BITCOIN_NETWORK").as_deref() {
            Ok("testnet") => Network::Testnet,
            Ok("regtest") => Network::Regtest,
            _ => Network::Mainnet,
        }
    }

    /// Get human-readable part (HRP) for Bech32 encoding
    ///
    /// Used by bllvm-protocol for address encoding (BIP173/350/351)
    pub fn hrp(&self) -> &'static str {
        match self {
            Network::Mainnet => "bc",
            Network::Testnet => "tb",
            Network::Regtest => "bcrt",
        }
    }
}

/// Block height: newtype wrapper for type safety
///
/// Prevents mixing up block heights with other u64 values (e.g., timestamps, counts).
/// Uses `#[repr(transparent)]` for zero-cost abstraction - same memory layout as u64.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockHeight(pub u64);

impl BlockHeight {
    /// Create a new BlockHeight from a u64
    #[inline(always)]
    pub fn new(height: u64) -> Self {
        BlockHeight(height)
    }

    /// Get the inner u64 value
    #[inline(always)]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for BlockHeight {
    #[inline(always)]
    fn from(height: u64) -> Self {
        BlockHeight(height)
    }
}

impl From<BlockHeight> for u64 {
    #[inline(always)]
    fn from(height: BlockHeight) -> Self {
        height.0
    }
}

impl std::ops::Deref for BlockHeight {
    type Target = u64;

    #[inline(always)]
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
    #[inline(always)]
    pub fn new(hash: Hash) -> Self {
        BlockHash(hash)
    }

    /// Get the inner Hash value
    #[inline(always)]
    pub fn as_hash(self) -> Hash {
        self.0
    }

    /// Get a reference to the inner Hash
    #[inline(always)]
    pub fn as_hash_ref(&self) -> &Hash {
        &self.0
    }
}

impl From<Hash> for BlockHash {
    #[inline(always)]
    fn from(hash: Hash) -> Self {
        BlockHash(hash)
    }
}

impl From<BlockHash> for Hash {
    #[inline(always)]
    fn from(hash: BlockHash) -> Self {
        hash.0
    }
}

impl std::ops::Deref for BlockHash {
    type Target = Hash;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// OutPoint: 𝒪 = ℍ × ℕ
///
/// Uses u32 for index (Bitcoin wire format); saves 24 bytes vs repr(align(64)) padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutPoint {
    pub hash: Hash,
    pub index: u32,
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
///
/// `Default` is only used as a placeholder when extracting the header for BIP113
/// tracking; the default value is never used for consensus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BlockHeader {
    pub version: Integer,
    pub prev_block_hash: Hash,
    pub merkle_root: Hash,
    pub timestamp: Natural,
    pub bits: Natural,
    pub nonce: Natural,
}

impl std::convert::AsRef<BlockHeader> for BlockHeader {
    #[inline]
    fn as_ref(&self) -> &BlockHeader {
        self
    }
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct UTXO {
    pub value: Integer,
    pub script_pubkey: SharedByteString,
    pub height: Natural,
    /// Whether this UTXO is from a coinbase transaction
    /// Coinbase outputs require maturity (COINBASE_MATURITY blocks) before they can be spent
    pub is_coinbase: bool,
}

/// UTXO Set: 𝒰𝒮 = 𝒪 → 𝒰
///
/// Arc<UTXO> avoids ~1500+ clones/block during IBD supplement_utxo_map and apply_sync_batch.
/// In production builds, uses FxHashMap for 2-3x faster lookups in large UTXO sets.
#[cfg(feature = "production")]
pub type UtxoSet = FxHashMap<OutPoint, std::sync::Arc<UTXO>>;

#[cfg(not(feature = "production"))]
pub type UtxoSet = HashMap<OutPoint, std::sync::Arc<UTXO>>;

/// Insert owned UTXO into UtxoSet (wraps in Arc). Convenience for tests and one-off inserts.
#[inline]
pub fn utxo_set_insert(set: &mut UtxoSet, op: OutPoint, u: UTXO) {
    use std::sync::Arc;
    set.insert(op, Arc::new(u));
}

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
