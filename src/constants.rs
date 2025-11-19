//! Bitcoin consensus constants from Orange Paper

/// Maximum money supply: 21,000,000 BTC in satoshis
pub const MAX_MONEY: i64 = 21_000_000 * 100_000_000;

/// Maximum transaction size: 1MB
pub const MAX_TX_SIZE: usize = 1_000_000;

/// Maximum block serialized size in bytes (network rule)
/// This is the maximum size of a block when serialized without witness data
pub const MAX_BLOCK_SERIALIZED_SIZE: usize = 4_000_000;

/// Maximum block weight in weight units (network rule, BIP141)
/// Weight = (stripped_size × 4) + witness_size
/// This is the primary limit for SegWit blocks
pub const MAX_BLOCK_WEIGHT: usize = 4_000_000;

/// Maximum block size (deprecated - use MAX_BLOCK_WEIGHT for SegWit blocks)
/// Kept for backward compatibility
#[deprecated(note = "Use MAX_BLOCK_WEIGHT for SegWit blocks")]
pub const MAX_BLOCK_SIZE: usize = MAX_BLOCK_WEIGHT;

/// Maximum number of inputs per transaction
pub const MAX_INPUTS: usize = 1000;

/// Maximum number of outputs per transaction
pub const MAX_OUTPUTS: usize = 1000;

/// Maximum script length
pub const MAX_SCRIPT_SIZE: usize = 10_000;

/// Maximum stack size during script execution
pub const MAX_STACK_SIZE: usize = 1000;

/// Maximum number of operations in script
pub const MAX_SCRIPT_OPS: usize = 201;

/// Maximum script element size (BIP141: witness elements can be up to 520 bytes)
pub const MAX_SCRIPT_ELEMENT_SIZE: usize = 520;

/// Halving interval: 210,000 blocks
pub const HALVING_INTERVAL: u64 = 210_000;

/// Initial block subsidy: 50 BTC
pub const INITIAL_SUBSIDY: i64 = 50 * 100_000_000;

/// Satoshis per BTC
pub const SATOSHIS_PER_BTC: i64 = 100_000_000;

/// Difficulty adjustment interval: 2016 blocks
pub const DIFFICULTY_ADJUSTMENT_INTERVAL: u64 = 2016;

/// Target time per block: 10 minutes
pub const TARGET_TIME_PER_BLOCK: u64 = 600;

/// Maximum target (minimum difficulty)
pub const MAX_TARGET: u32 = 0x1d00ffff;

/// Minimum target (maximum difficulty) - Bitcoin's genesis target
pub const MIN_TARGET: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Lock time threshold: transactions with lock time < this are block height
pub const LOCKTIME_THRESHOLD: u32 = 500_000_000;

/// Sequence number for final transaction
pub const SEQUENCE_FINAL: u32 = 0xffffffff;

/// Sequence number for RBF
pub const SEQUENCE_RBF: u32 = 0xfffffffe;

/// Minimum relay fee for RBF replacement (BIP125)
///
/// A replacement transaction must pay at least this much more in fees
/// than the transaction it replaces. This prevents spam replacements
/// with minimal fee increases.
pub const MIN_RELAY_FEE: i64 = 1000; // 1000 satoshis

/// Coinbase maturity requirement: 100 blocks
///
/// Coinbase outputs cannot be spent until 100 blocks deep.
/// This prevents miners from spending coinbase immediately and helps
/// secure the network against deep reorgs.
pub const COINBASE_MATURITY: u64 = 100;

/// Maximum block sigop cost (network rule)
///
/// Total sigop cost for a block must not exceed this value.
/// Sigop cost = (legacy sigops × 4) + (P2SH sigops × 4) + witness sigops
///
/// Reference: Bitcoin Core `consensus.h` MAX_BLOCK_SIGOPS_COST = 80000
pub const MAX_BLOCK_SIGOPS_COST: u64 = 80_000;

/// Witness commitment hash length (BIP141)
///
/// The witness commitment in the coinbase transaction contains:
/// - OP_RETURN (0x6a): 1 byte
/// - Push opcode (0x24): 1 byte
/// - Commitment hash: 32 bytes
/// Total: 34 bytes
///
/// Reference: BIP141 - Witness commitment format
pub const WITNESS_COMMITMENT_HASH_LENGTH: usize = 32;

/// Witness commitment script length (BIP141)
///
/// Total length of witness commitment script:
/// - OP_RETURN (0x6a): 1 byte
/// - Push opcode (0x24): 1 byte  
/// - Commitment hash: 32 bytes
/// Total: 34 bytes
pub const WITNESS_COMMITMENT_SCRIPT_LENGTH: usize = 34;

/// Taproot script length (BIP341)
///
/// Taproot P2TR script format: OP_1 <32-byte-program>
/// - OP_1 (0x51): 1 byte
/// - Push opcode (0x20): 1 byte
/// - Program hash: 32 bytes
/// Total: 34 bytes
pub const TAPROOT_SCRIPT_LENGTH: usize = 34;

/// Taproot program hash length (BIP341)
///
/// Taproot witness program (P2TR) is 32 bytes
pub const TAPROOT_PROGRAM_LENGTH: usize = 32;

/// SegWit witness program lengths (BIP141)
///
/// SegWit v0 programs:
/// - P2WPKH: 20 bytes
/// - P2WSH: 32 bytes
pub const SEGWIT_P2WPKH_LENGTH: usize = 20;
pub const SEGWIT_P2WSH_LENGTH: usize = 32;
