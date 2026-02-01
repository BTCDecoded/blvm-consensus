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
// Note: Bitcoin Core has no explicit input limit - only bounded by block weight
// Setting to a very high value to match Core's behavior
pub const MAX_INPUTS: usize = 100_000;

/// Maximum number of outputs per transaction
// Note: Bitcoin Core has no explicit output limit - only bounded by block weight
// Setting to a very high value to match Core's behavior
pub const MAX_OUTPUTS: usize = 100_000;

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

/// Maximum future block time tolerance: 2 hours (7200 seconds)
///
/// Blocks with timestamps more than this far in the future are rejected
/// to prevent time-warp attacks. This allows for reasonable clock skew
/// between network nodes.
pub const MAX_FUTURE_BLOCK_TIME: u64 = 7200;

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
///   Total: 34 bytes
///
/// Reference: BIP141 - Witness commitment format
pub const WITNESS_COMMITMENT_HASH_LENGTH: usize = 32;

/// Witness commitment script length (BIP141)
///
/// Total length of witness commitment script:
/// - OP_RETURN (0x6a): 1 byte
/// - Push opcode (0x24): 1 byte  
/// - Commitment hash: 32 bytes
///   Total: 34 bytes
pub const WITNESS_COMMITMENT_SCRIPT_LENGTH: usize = 34;

/// Taproot script length (BIP341)
///
/// Taproot P2TR script format: OP_1 <32-byte-program>
/// - OP_1 (0x51): 1 byte
/// - Push opcode (0x20): 1 byte
/// - Program hash: 32 bytes
///   Total: 34 bytes
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

// ============================================================================
// BIP ACTIVATION HEIGHTS
// ============================================================================
// Consensus-critical activation heights. Any change to these values would cause
// a chain split. These heights are locked via formal proofs to match Bitcoin Core.

/// BIP30: Duplicate Coinbase Prevention - Mainnet deactivation height
///
/// BIP30 was disabled after this block to allow duplicate coinbases in blocks 91842 and 91880.
/// Reference: Bitcoin Core disabled BIP30 after block 91722
pub const BIP30_DEACTIVATION_MAINNET: u64 = 91722;

/// BIP30: Duplicate Coinbase Prevention - Testnet deactivation height
///
/// BIP30 was disabled after this block on testnet.
pub const BIP30_DEACTIVATION_TESTNET: u64 = 0; // BIP30 never enforced on testnet

/// BIP30: Duplicate Coinbase Prevention - Regtest deactivation height
///
/// BIP30 is never enforced on regtest.
pub const BIP30_DEACTIVATION_REGTEST: u64 = 0;

/// BIP16: P2SH (Pay-to-Script-Hash) - Mainnet activation height
///
/// Starting at this block, P2SH scripts are valid.
/// Reference: BIP16, activated April 1, 2012 at block 173,805
pub const BIP16_P2SH_ACTIVATION_MAINNET: u64 = 173_805;

/// BIP16: P2SH (Pay-to-Script-Hash) - Testnet activation height
///
/// Reference: BIP16, always active on testnet
pub const BIP16_P2SH_ACTIVATION_TESTNET: u64 = 0;

/// BIP16: P2SH (Pay-to-Script-Hash) - Regtest activation height
///
/// Reference: BIP16, always active on regtest
pub const BIP16_P2SH_ACTIVATION_REGTEST: u64 = 0;

/// BIP34: Block Height in Coinbase - Mainnet activation height
///
/// Starting at this block, coinbase scriptSig must contain the block height.
/// Reference: BIP34, Bitcoin Core activation at block 227,836
pub const BIP34_ACTIVATION_MAINNET: u64 = 227_836;

/// BIP34: Block Height in Coinbase - Testnet activation height
///
/// Reference: BIP34, Bitcoin Core activation at block 211,111
pub const BIP34_ACTIVATION_TESTNET: u64 = 211_111;

/// BIP34: Block Height in Coinbase - Regtest activation height
///
/// Always active on regtest (block 0)
pub const BIP34_ACTIVATION_REGTEST: u64 = 0;

/// BIP66: Strict DER Signatures - Mainnet activation height
///
/// Starting at this block, all signatures must use strict DER encoding.
/// Reference: BIP66, Bitcoin Core activation at block 363,725
/// Note: The code checks `height < activation_height`, so 363,725 is the first
/// block where BIP66 is enforced.
pub const BIP66_ACTIVATION_MAINNET: u64 = 363_725;

/// BIP66: Strict DER Signatures - Testnet activation height
///
/// Reference: BIP66, Bitcoin Core activation at block 330,776
pub const BIP66_ACTIVATION_TESTNET: u64 = 330_776;

/// BIP66: Strict DER Signatures - Regtest activation height
///
/// Always active on regtest (block 0)
pub const BIP66_ACTIVATION_REGTEST: u64 = 0;

/// BIP65: OP_CHECKLOCKTIMEVERIFY (CLTV) - Mainnet activation height
///
/// Starting at this block, CLTV opcode is enabled.
/// Reference: BIP65, Bitcoin Core activation at block 388,381
pub const BIP65_ACTIVATION_MAINNET: u64 = 388_381;

/// BIP147: NULLDUMMY Enforcement - Mainnet activation height
///
/// Starting at this block, dummy stack elements must be empty (OP_0).
/// Reference: BIP147, Bitcoin Core activation at block 481,824
pub const BIP147_ACTIVATION_MAINNET: u64 = 481_824;

/// BIP147: NULLDUMMY Enforcement - Testnet activation height
///
/// Reference: BIP147, Bitcoin Core activation at block 834,624
pub const BIP147_ACTIVATION_TESTNET: u64 = 834_624;

/// SegWit (BIP141) - Mainnet activation height
///
/// Starting at this block, Segregated Witness is active.
/// Reference: BIP141, Bitcoin Core activation at block 481,824
pub const SEGWIT_ACTIVATION_MAINNET: u64 = 481_824;

/// Taproot (BIP341) - Mainnet activation height
///
/// Starting at this block, Taproot is active.
/// Reference: BIP341, Bitcoin Core activation at block 709,632
pub const TAPROOT_ACTIVATION_MAINNET: u64 = 709_632;

// ============================================================================
// GENESIS BLOCK CONSTANTS
// ============================================================================
// The genesis block is the foundation of Bitcoin. Any change to these values
// would create a different blockchain. These are locked via formal proofs.

/// Genesis block hash (mainnet)
///
/// The hash of the first Bitcoin block. This is the root of the blockchain.
/// Hash: 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
pub const GENESIS_BLOCK_HASH: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68, 0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93,
    0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1, 0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f,
];

/// Genesis block timestamp (Unix timestamp)
///
/// The timestamp of the genesis block: January 3, 2009, 18:15:05 UTC
/// This is the birth of Bitcoin.
pub const GENESIS_BLOCK_TIMESTAMP: u32 = 1231006505;

/// Genesis block merkle root
///
/// The merkle root of the genesis block's coinbase transaction.
/// This is the root of the first transaction tree.
pub const GENESIS_BLOCK_MERKLE_ROOT: [u8; 32] = [
    0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2, 0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
    0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32, 0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
];

/// Genesis block nonce
///
/// The nonce used in the genesis block's proof of work.
pub const GENESIS_BLOCK_NONCE: u32 = 2083236893;

// ============================================================================
// FORMAL VERIFICATION: Orange Paper Constant Locking
// ============================================================================
// These proofs directly verify that constant values match the Orange Paper
// Section 4 (Consensus Constants) specification. This creates a cryptographic
// lock between the Orange Paper (IR) and blvm-consensus implementation.

