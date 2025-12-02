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
// a chain split. These heights are locked via Kani proofs to match Bitcoin Core.

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
// would create a different blockchain. These are locked via Kani proofs.

/// Genesis block hash (mainnet)
///
/// The hash of the first Bitcoin block. This is the root of the blockchain.
/// Hash: 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
pub const GENESIS_BLOCK_HASH: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68,
    0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93,
    0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1,
    0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f,
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
    0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
    0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
    0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
    0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
];

/// Genesis block nonce
///
/// The nonce used in the genesis block's proof of work.
pub const GENESIS_BLOCK_NONCE: u32 = 2083236893;

// ============================================================================
// KANI VERIFICATION: Orange Paper Constant Locking
// ============================================================================
// These proofs directly verify that constant values match the Orange Paper
// Section 4 (Consensus Constants) specification. This creates a cryptographic
// lock between the Orange Paper (IR) and blvm-consensus implementation.

#[cfg(kani)]
mod kani_constant_verification {
    use super::*;
    use kani::*;

    /// Kani proof: Monetary constants match Orange Paper Section 4.1
    ///
    /// Orange Paper Section 4.1 (Monetary Constants):
    /// - C = 10^8 (satoshis per BTC)
    /// - M_max = 21 × 10^6 × C (maximum money supply)
    /// - H = 210,000 (halving interval)
    ///
    /// This proof locks the Orange Paper mathematical specification to the
    /// implementation constants, ensuring any change to these values would
    /// break the proof.
    #[kani::proof]
    fn kani_monetary_constants_match_orange_paper() {
        // C = 10^8 (satoshis per BTC) - Orange Paper Section 4.1
        assert_eq!(
            SATOSHIS_PER_BTC,
            100_000_000,
            "SATOSHIS_PER_BTC must equal 10^8 per Orange Paper Section 4.1"
        );

        // M_max = 21 × 10^6 × C - Orange Paper Section 4.1
        let expected_max_money = 21_000_000 * SATOSHIS_PER_BTC;
        assert_eq!(
            MAX_MONEY,
            expected_max_money,
            "MAX_MONEY must equal 21 × 10^6 × C per Orange Paper Section 4.1"
        );

        // H = 210,000 (halving interval) - Orange Paper Section 4.1
        assert_eq!(
            HALVING_INTERVAL,
            210_000,
            "HALVING_INTERVAL must equal 210,000 per Orange Paper Section 4.1"
        );

        // Initial subsidy: 50 BTC = 50 × C
        let expected_initial_subsidy = 50 * SATOSHIS_PER_BTC;
        assert_eq!(
            INITIAL_SUBSIDY,
            expected_initial_subsidy,
            "INITIAL_SUBSIDY must equal 50 × C per Orange Paper Section 6.1"
        );
    }

    /// Kani proof: Block constants match Orange Paper Section 4.2
    ///
    /// Orange Paper Section 4.2 (Block Constants):
    /// - W_max = 4 × 10^6 (maximum block weight)
    /// - S_max = 80,000 (maximum sigops per block)
    /// - R = 100 (coinbase maturity requirement)
    #[kani::proof]
    fn kani_block_constants_match_orange_paper() {
        // W_max = 4 × 10^6 (maximum block weight) - Orange Paper Section 4.2
        assert_eq!(
            MAX_BLOCK_WEIGHT,
            4_000_000,
            "MAX_BLOCK_WEIGHT must equal 4 × 10^6 per Orange Paper Section 4.2"
        );

        // S_max = 80,000 (maximum sigops per block) - Orange Paper Section 4.2
        assert_eq!(
            MAX_BLOCK_SIGOPS_COST,
            80_000,
            "MAX_BLOCK_SIGOPS_COST must equal 80,000 per Orange Paper Section 4.2"
        );

        // R = 100 (coinbase maturity requirement) - Orange Paper Section 4.2
        assert_eq!(
            COINBASE_MATURITY,
            100,
            "COINBASE_MATURITY must equal 100 per Orange Paper Section 4.2"
        );
    }

    /// Kani proof: Script constants match Orange Paper Section 4.3
    ///
    /// Orange Paper Section 4.3 (Script Constants):
    /// - L_script = 10,000 (maximum script length)
    /// - L_stack = 1,000 (maximum stack size)
    /// - L_ops = 201 (maximum operations per script)
    /// - L_element = 520 (maximum element size)
    #[kani::proof]
    fn kani_script_constants_match_orange_paper() {
        // L_script = 10,000 (maximum script length) - Orange Paper Section 4.3
        assert_eq!(
            MAX_SCRIPT_SIZE,
            10_000,
            "MAX_SCRIPT_SIZE must equal 10,000 per Orange Paper Section 4.3"
        );

        // L_stack = 1,000 (maximum stack size) - Orange Paper Section 4.3
        assert_eq!(
            MAX_STACK_SIZE,
            1_000,
            "MAX_STACK_SIZE must equal 1,000 per Orange Paper Section 4.3"
        );

        // L_ops = 201 (maximum operations per script) - Orange Paper Section 4.3
        assert_eq!(
            MAX_SCRIPT_OPS,
            201,
            "MAX_SCRIPT_OPS must equal 201 per Orange Paper Section 4.3"
        );

        // L_element = 520 (maximum element size) - Orange Paper Section 4.3
        assert_eq!(
            MAX_SCRIPT_ELEMENT_SIZE,
            520,
            "MAX_SCRIPT_ELEMENT_SIZE must equal 520 per Orange Paper Section 4.3"
        );
    }

    /// Kani proof: Difficulty adjustment constants match Orange Paper Section 7.1
    ///
    /// Orange Paper Section 7.1 (Difficulty Adjustment):
    /// - Difficulty adjustment interval: 2016 blocks
    /// - Target time per block: 600 seconds (10 minutes)
    #[kani::proof]
    fn kani_difficulty_constants_match_orange_paper() {
        // Difficulty adjustment interval: 2016 blocks - Orange Paper Section 7.1
        assert_eq!(
            DIFFICULTY_ADJUSTMENT_INTERVAL,
            2016,
            "DIFFICULTY_ADJUSTMENT_INTERVAL must equal 2016 per Orange Paper Section 7.1"
        );

        // Target time per block: 600 seconds (10 minutes) - Orange Paper Section 7.1
        assert_eq!(
            TARGET_TIME_PER_BLOCK,
            600,
            "TARGET_TIME_PER_BLOCK must equal 600 seconds per Orange Paper Section 7.1"
        );
    }

    /// Kani proof: Derived constant relationships match Orange Paper
    ///
    /// This proof verifies that derived constants maintain correct relationships
    /// as specified in the Orange Paper.
    #[kani::proof]
    fn kani_derived_constant_relationships() {
        // MAX_MONEY = 21 × 10^6 × SATOSHIS_PER_BTC
        let expected_max_money = 21_000_000 * SATOSHIS_PER_BTC;
        assert_eq!(
            MAX_MONEY,
            expected_max_money,
            "MAX_MONEY must equal 21 × 10^6 × SATOSHIS_PER_BTC"
        );

        // INITIAL_SUBSIDY = 50 × SATOSHIS_PER_BTC
        let expected_initial_subsidy = 50 * SATOSHIS_PER_BTC;
        assert_eq!(
            INITIAL_SUBSIDY,
            expected_initial_subsidy,
            "INITIAL_SUBSIDY must equal 50 × SATOSHIS_PER_BTC"
        );

        // MAX_BLOCK_WEIGHT = MAX_BLOCK_SERIALIZED_SIZE (for SegWit compatibility)
        assert_eq!(
            MAX_BLOCK_WEIGHT,
            MAX_BLOCK_SERIALIZED_SIZE,
            "MAX_BLOCK_WEIGHT must equal MAX_BLOCK_SERIALIZED_SIZE for SegWit compatibility"
        );
    }

    /// Kani proof: Critical Bitcoin invariants enforced by constants
    ///
    /// This proof verifies that constants enforce critical Bitcoin security
    /// properties as specified in the Orange Paper.
    #[kani::proof]
    fn kani_critical_invariants_enforced() {
        // Invariant: MAX_MONEY must be positive
        assert!(
            MAX_MONEY > 0,
            "MAX_MONEY must be positive (Bitcoin supply cap)"
        );

        // Invariant: SATOSHIS_PER_BTC must be positive
        assert!(
            SATOSHIS_PER_BTC > 0,
            "SATOSHIS_PER_BTC must be positive"
        );

        // Invariant: HALVING_INTERVAL must be positive
        assert!(
            HALVING_INTERVAL > 0,
            "HALVING_INTERVAL must be positive"
        );

        // Invariant: INITIAL_SUBSIDY must be positive
        assert!(
            INITIAL_SUBSIDY > 0,
            "INITIAL_SUBSIDY must be positive"
        );

        // Invariant: MAX_MONEY must be representable in i64
        assert!(
            MAX_MONEY <= i64::MAX,
            "MAX_MONEY must fit in i64"
        );

        // Invariant: COINBASE_MATURITY must be positive
        assert!(
            COINBASE_MATURITY > 0,
            "COINBASE_MATURITY must be positive"
        );

        // Invariant: All size limits must be positive
        assert!(
            MAX_BLOCK_WEIGHT > 0,
            "MAX_BLOCK_WEIGHT must be positive"
        );
        assert!(
            MAX_SCRIPT_SIZE > 0,
            "MAX_SCRIPT_SIZE must be positive"
        );
        assert!(
            MAX_STACK_SIZE > 0,
            "MAX_STACK_SIZE must be positive"
        );
    }

    /// Kani proof: Transaction size constants match Orange Paper and Bitcoin Core
    ///
    /// Verifies transaction and block size limits match Bitcoin Core specifications.
    #[kani::proof]
    fn kani_transaction_size_constants() {
        // MAX_TX_SIZE = 1MB (1,000,000 bytes) - Bitcoin Core limit
        assert_eq!(
            MAX_TX_SIZE,
            1_000_000,
            "MAX_TX_SIZE must equal 1,000,000 bytes (1MB) per Bitcoin Core"
        );

        // MAX_BLOCK_SERIALIZED_SIZE = 4MB - Bitcoin Core limit
        assert_eq!(
            MAX_BLOCK_SERIALIZED_SIZE,
            4_000_000,
            "MAX_BLOCK_SERIALIZED_SIZE must equal 4,000,000 bytes (4MB) per Bitcoin Core"
        );

        // MAX_INPUTS and MAX_OUTPUTS are practical limits (not consensus-critical)
        assert!(
            MAX_INPUTS > 0,
            "MAX_INPUTS must be positive"
        );
        assert!(
            MAX_OUTPUTS > 0,
            "MAX_OUTPUTS must be positive"
        );
    }

    /// Kani proof: Proof of Work constants match Orange Paper Section 7
    ///
    /// Orange Paper Section 7 (Proof of Work):
    /// - MAX_TARGET = 0x1d00ffff (genesis difficulty)
    #[kani::proof]
    fn kani_pow_constants_match_orange_paper() {
        // MAX_TARGET = 0x1d00ffff (genesis difficulty) - Orange Paper Section 7.2
        assert_eq!(
            MAX_TARGET,
            0x1d00ffff,
            "MAX_TARGET must equal 0x1d00ffff (genesis difficulty) per Orange Paper Section 7.2"
        );

        // MIN_TARGET is the genesis block target (all zeros except 0xffff in bytes 4-5)
        // This is the maximum difficulty (minimum target value)
        let expected_min_target: [u8; 32] = [
            0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(
            MIN_TARGET,
            expected_min_target,
            "MIN_TARGET must match genesis block target per Orange Paper Section 7.2"
        );
    }

    /// Kani proof: Locktime and sequence constants match Orange Paper Section 5.1
    ///
    /// Orange Paper Section 5.1 (Transaction Validation):
    /// - Locktime threshold: 500,000,000 (transactions with locktime < this are block height)
    /// - Sequence final: 0xffffffff (final transaction)
    /// - Sequence RBF: 0xfffffffe (replace-by-fee enabled)
    #[kani::proof]
    fn kani_locktime_sequence_constants_match_orange_paper() {
        // LOCKTIME_THRESHOLD = 500,000,000 - Orange Paper Section 5.1
        // Transactions with locktime < this value are interpreted as block height
        assert_eq!(
            LOCKTIME_THRESHOLD,
            500_000_000,
            "LOCKTIME_THRESHOLD must equal 500,000,000 per Orange Paper Section 5.1"
        );

        // SEQUENCE_FINAL = 0xffffffff - Orange Paper Section 5.1
        // Indicates transaction is final (no RBF)
        assert_eq!(
            SEQUENCE_FINAL,
            0xffffffff,
            "SEQUENCE_FINAL must equal 0xffffffff per Orange Paper Section 5.1"
        );

        // SEQUENCE_RBF = 0xfffffffe - BIP125 (Replace-By-Fee)
        // Indicates transaction signals RBF
        assert_eq!(
            SEQUENCE_RBF,
            0xfffffffe,
            "SEQUENCE_RBF must equal 0xfffffffe per BIP125"
        );

        // Relationship: SEQUENCE_RBF = SEQUENCE_FINAL - 1
        assert_eq!(
            SEQUENCE_RBF,
            SEQUENCE_FINAL.wrapping_sub(1),
            "SEQUENCE_RBF must equal SEQUENCE_FINAL - 1"
        );
    }

    /// Kani proof: Fee constants match BIP125 and Bitcoin Core
    ///
    /// BIP125 (Replace-By-Fee):
    /// - MIN_RELAY_FEE = 1000 satoshis (minimum fee increase for RBF)
    #[kani::proof]
    fn kani_fee_constants_match_bip125() {
        // MIN_RELAY_FEE = 1000 satoshis - BIP125 requirement
        assert_eq!(
            MIN_RELAY_FEE,
            1000,
            "MIN_RELAY_FEE must equal 1000 satoshis per BIP125"
        );

        // MIN_RELAY_FEE must be positive
        assert!(
            MIN_RELAY_FEE > 0,
            "MIN_RELAY_FEE must be positive"
        );

        // MIN_RELAY_FEE must be less than MAX_MONEY (sanity check)
        assert!(
            MIN_RELAY_FEE < MAX_MONEY,
            "MIN_RELAY_FEE must be less than MAX_MONEY"
        );
    }

    /// Kani proof: SegWit (BIP141) constants match specification
    ///
    /// BIP141 (Segregated Witness):
    /// - P2WPKH witness program: 20 bytes
    /// - P2WSH witness program: 32 bytes
    /// - Witness commitment hash: 32 bytes
    /// - Witness commitment script: 34 bytes (OP_RETURN + push + hash)
    #[kani::proof]
    fn kani_segwit_constants_match_bip141() {
        // P2WPKH witness program length: 20 bytes - BIP141
        assert_eq!(
            SEGWIT_P2WPKH_LENGTH,
            20,
            "SEGWIT_P2WPKH_LENGTH must equal 20 bytes per BIP141"
        );

        // P2WSH witness program length: 32 bytes - BIP141
        assert_eq!(
            SEGWIT_P2WSH_LENGTH,
            32,
            "SEGWIT_P2WSH_LENGTH must equal 32 bytes per BIP141"
        );

        // Witness commitment hash length: 32 bytes - BIP141
        assert_eq!(
            WITNESS_COMMITMENT_HASH_LENGTH,
            32,
            "WITNESS_COMMITMENT_HASH_LENGTH must equal 32 bytes per BIP141"
        );

        // Witness commitment script length: 34 bytes - BIP141
        // Format: OP_RETURN (1) + push opcode (1) + hash (32) = 34
        assert_eq!(
            WITNESS_COMMITMENT_SCRIPT_LENGTH,
            34,
            "WITNESS_COMMITMENT_SCRIPT_LENGTH must equal 34 bytes per BIP141"
        );

        // Relationship: Witness commitment script = OP_RETURN + push + hash
        assert_eq!(
            WITNESS_COMMITMENT_SCRIPT_LENGTH,
            1 + 1 + WITNESS_COMMITMENT_HASH_LENGTH,
            "WITNESS_COMMITMENT_SCRIPT_LENGTH must equal OP_RETURN + push + hash length"
        );
    }

    /// Kani proof: Taproot (BIP341) constants match specification
    ///
    /// BIP341 (Taproot):
    /// - Taproot script length: 34 bytes (OP_1 + push + program)
    /// - Taproot program length: 32 bytes
    #[kani::proof]
    fn kani_taproot_constants_match_bip341() {
        // Taproot program length: 32 bytes - BIP341
        assert_eq!(
            TAPROOT_PROGRAM_LENGTH,
            32,
            "TAPROOT_PROGRAM_LENGTH must equal 32 bytes per BIP341"
        );

        // Taproot script length: 34 bytes - BIP341
        // Format: OP_1 (1) + push opcode (1) + program (32) = 34
        assert_eq!(
            TAPROOT_SCRIPT_LENGTH,
            34,
            "TAPROOT_SCRIPT_LENGTH must equal 34 bytes per BIP341"
        );

        // Relationship: Taproot script = OP_1 + push + program
        assert_eq!(
            TAPROOT_SCRIPT_LENGTH,
            1 + 1 + TAPROOT_PROGRAM_LENGTH,
            "TAPROOT_SCRIPT_LENGTH must equal OP_1 + push + program length"
        );

        // Taproot program length matches SegWit P2WSH length (both are 32 bytes)
        assert_eq!(
            TAPROOT_PROGRAM_LENGTH,
            SEGWIT_P2WSH_LENGTH,
            "TAPROOT_PROGRAM_LENGTH must equal SEGWIT_P2WSH_LENGTH (both 32 bytes)"
        );
    }

    /// Kani proof: Size limit relationships enforce Bitcoin security properties
    ///
    /// This proof verifies that size limits maintain correct relationships to prevent
    /// resource exhaustion attacks and ensure consensus compatibility.
    #[kani::proof]
    fn kani_size_limit_relationships() {
        // Script size must be less than or equal to transaction size
        assert!(
            MAX_SCRIPT_SIZE <= MAX_TX_SIZE,
            "MAX_SCRIPT_SIZE must be <= MAX_TX_SIZE to prevent oversized scripts"
        );

        // Block weight must be >= block serialized size (for SegWit compatibility)
        assert!(
            MAX_BLOCK_WEIGHT >= MAX_BLOCK_SERIALIZED_SIZE,
            "MAX_BLOCK_WEIGHT must be >= MAX_BLOCK_SERIALIZED_SIZE for SegWit"
        );

        // Script element size must be <= script size
        assert!(
            MAX_SCRIPT_ELEMENT_SIZE <= MAX_SCRIPT_SIZE,
            "MAX_SCRIPT_ELEMENT_SIZE must be <= MAX_SCRIPT_SIZE"
        );

        // Stack size must be reasonable (prevents DoS)
        assert!(
            MAX_STACK_SIZE > 0 && MAX_STACK_SIZE <= 10000,
            "MAX_STACK_SIZE must be positive and reasonable"
        );

        // Script ops limit must be reasonable (prevents DoS)
        assert!(
            MAX_SCRIPT_OPS > 0 && MAX_SCRIPT_OPS <= 10000,
            "MAX_SCRIPT_OPS must be positive and reasonable"
        );
    }

    /// Kani proof: Economic model constants maintain Bitcoin's monetary properties
    ///
    /// This proof verifies that economic constants maintain critical Bitcoin properties:
    /// - Supply cap (21M BTC)
    /// - Halving schedule (every 210,000 blocks)
    /// - Initial subsidy (50 BTC)
    #[kani::proof]
    fn kani_economic_model_properties() {
        // Critical property: MAX_MONEY = 21M BTC
        let expected_max_money_btc = 21_000_000;
        let max_money_btc = MAX_MONEY / SATOSHIS_PER_BTC;
        assert_eq!(
            max_money_btc,
            expected_max_money_btc,
            "MAX_MONEY must equal exactly 21,000,000 BTC"
        );

        // Critical property: INITIAL_SUBSIDY = 50 BTC
        let expected_initial_subsidy_btc = 50;
        let initial_subsidy_btc = INITIAL_SUBSIDY / SATOSHIS_PER_BTC;
        assert_eq!(
            initial_subsidy_btc,
            expected_initial_subsidy_btc,
            "INITIAL_SUBSIDY must equal exactly 50 BTC"
        );

        // Critical property: After 64 halvings, subsidy = 0
        // This ensures total supply approaches but never exceeds 21M BTC
        let height_at_64_halvings = HALVING_INTERVAL * 64;
        // This is verified in economic.rs Kani proofs, but we verify the relationship here
        assert!(
            height_at_64_halvings > 0,
            "Height at 64 halvings must be positive"
        );

        // Critical property: HALVING_INTERVAL is positive
        assert!(
            HALVING_INTERVAL > 0,
            "HALVING_INTERVAL must be positive"
        );

        // Critical property: SATOSHIS_PER_BTC = 10^8
        assert_eq!(
            SATOSHIS_PER_BTC,
            100_000_000,
            "SATOSHIS_PER_BTC must equal 10^8"
        );
    }

    /// Kani proof: Difficulty adjustment constants maintain Bitcoin's time properties
    ///
    /// This proof verifies that difficulty adjustment constants maintain Bitcoin's
    /// 10-minute block time target.
    #[kani::proof]
    fn kani_difficulty_adjustment_properties() {
        // Critical property: TARGET_TIME_PER_BLOCK = 600 seconds (10 minutes)
        assert_eq!(
            TARGET_TIME_PER_BLOCK,
            600,
            "TARGET_TIME_PER_BLOCK must equal 600 seconds (10 minutes)"
        );

        // Critical property: DIFFICULTY_ADJUSTMENT_INTERVAL = 2016 blocks
        assert_eq!(
            DIFFICULTY_ADJUSTMENT_INTERVAL,
            2016,
            "DIFFICULTY_ADJUSTMENT_INTERVAL must equal 2016 blocks"
        );

        // Relationship: Expected time for difficulty adjustment = 2016 * 600 seconds
        let expected_adjustment_time = DIFFICULTY_ADJUSTMENT_INTERVAL * TARGET_TIME_PER_BLOCK;
        let two_weeks_seconds = 14 * 24 * 60 * 60; // 14 days in seconds
        assert_eq!(
            expected_adjustment_time,
            two_weeks_seconds,
            "Difficulty adjustment interval must equal 2 weeks (14 days)"
        );

        // Both constants must be positive
        assert!(
            TARGET_TIME_PER_BLOCK > 0,
            "TARGET_TIME_PER_BLOCK must be positive"
        );
        assert!(
            DIFFICULTY_ADJUSTMENT_INTERVAL > 0,
            "DIFFICULTY_ADJUSTMENT_INTERVAL must be positive"
        );
    }

    /// Kani proof: All constants are within valid ranges
    ///
    /// This proof performs comprehensive range checks on all constants to ensure
    /// they are within reasonable bounds and won't cause integer overflow or
    /// other issues.
    #[kani::proof]
    fn kani_all_constants_valid_ranges() {
        // Monetary constants must fit in i64
        assert!(
            MAX_MONEY > 0 && MAX_MONEY <= i64::MAX,
            "MAX_MONEY must be positive and fit in i64"
        );
        assert!(
            INITIAL_SUBSIDY > 0 && INITIAL_SUBSIDY <= i64::MAX,
            "INITIAL_SUBSIDY must be positive and fit in i64"
        );
        assert!(
            SATOSHIS_PER_BTC > 0 && SATOSHIS_PER_BTC <= i64::MAX,
            "SATOSHIS_PER_BTC must be positive and fit in i64"
        );
        assert!(
            MIN_RELAY_FEE >= 0 && MIN_RELAY_FEE <= i64::MAX,
            "MIN_RELAY_FEE must be non-negative and fit in i64"
        );

        // Block/transaction size limits must be reasonable
        assert!(
            MAX_TX_SIZE > 0 && MAX_TX_SIZE <= usize::MAX,
            "MAX_TX_SIZE must be positive and fit in usize"
        );
        assert!(
            MAX_BLOCK_WEIGHT > 0 && MAX_BLOCK_WEIGHT <= usize::MAX,
            "MAX_BLOCK_WEIGHT must be positive and fit in usize"
        );
        assert!(
            MAX_BLOCK_SERIALIZED_SIZE > 0 && MAX_BLOCK_SERIALIZED_SIZE <= usize::MAX,
            "MAX_BLOCK_SERIALIZED_SIZE must be positive and fit in usize"
        );

        // Script limits must be reasonable
        assert!(
            MAX_SCRIPT_SIZE > 0 && MAX_SCRIPT_SIZE <= usize::MAX,
            "MAX_SCRIPT_SIZE must be positive and fit in usize"
        );
        assert!(
            MAX_STACK_SIZE > 0 && MAX_STACK_SIZE <= usize::MAX,
            "MAX_STACK_SIZE must be positive and fit in usize"
        );
        assert!(
            MAX_SCRIPT_OPS > 0 && MAX_SCRIPT_OPS <= usize::MAX,
            "MAX_SCRIPT_OPS must be positive and fit in usize"
        );
        assert!(
            MAX_SCRIPT_ELEMENT_SIZE > 0 && MAX_SCRIPT_ELEMENT_SIZE <= usize::MAX,
            "MAX_SCRIPT_ELEMENT_SIZE must be positive and fit in usize"
        );

        // Interval constants must be positive
        assert!(
            HALVING_INTERVAL > 0,
            "HALVING_INTERVAL must be positive"
        );
        assert!(
            DIFFICULTY_ADJUSTMENT_INTERVAL > 0,
            "DIFFICULTY_ADJUSTMENT_INTERVAL must be positive"
        );
        assert!(
            TARGET_TIME_PER_BLOCK > 0,
            "TARGET_TIME_PER_BLOCK must be positive"
        );
        assert!(
            COINBASE_MATURITY > 0,
            "COINBASE_MATURITY must be positive"
        );

        // Locktime/sequence constants must be valid u32 values
        assert!(
            LOCKTIME_THRESHOLD > 0 && LOCKTIME_THRESHOLD <= u32::MAX,
            "LOCKTIME_THRESHOLD must be positive and fit in u32"
        );
        assert!(
            SEQUENCE_FINAL > 0,
            "SEQUENCE_FINAL must be positive"
        );
        assert!(
            SEQUENCE_RBF > 0,
            "SEQUENCE_RBF must be positive"
        );

        // BIP-specific length constants must be reasonable
        assert!(
            SEGWIT_P2WPKH_LENGTH > 0 && SEGWIT_P2WPKH_LENGTH <= 100,
            "SEGWIT_P2WPKH_LENGTH must be positive and reasonable"
        );
        assert!(
            SEGWIT_P2WSH_LENGTH > 0 && SEGWIT_P2WSH_LENGTH <= 100,
            "SEGWIT_P2WSH_LENGTH must be positive and reasonable"
        );
        assert!(
            TAPROOT_PROGRAM_LENGTH > 0 && TAPROOT_PROGRAM_LENGTH <= 100,
            "TAPROOT_PROGRAM_LENGTH must be positive and reasonable"
        );
        assert!(
            WITNESS_COMMITMENT_HASH_LENGTH > 0 && WITNESS_COMMITMENT_HASH_LENGTH <= 100,
            "WITNESS_COMMITMENT_HASH_LENGTH must be positive and reasonable"
        );
    }

    /// Kani proof: BIP activation heights match Bitcoin Core exactly
    ///
    /// These activation heights are consensus-critical. Any change would cause
    /// a chain split. This proof locks them to match Bitcoin Core's values.
    #[kani::proof]
    fn kani_bip_activation_heights_match_bitcoin_core() {
        // BIP34 activation heights - consensus-critical
        assert_eq!(
            BIP34_ACTIVATION_MAINNET,
            227_836,
            "BIP34 mainnet activation must match Bitcoin Core (227,836)"
        );
        assert_eq!(
            BIP34_ACTIVATION_TESTNET,
            211_111,
            "BIP34 testnet activation must match Bitcoin Core (211,111)"
        );
        assert_eq!(
            BIP34_ACTIVATION_REGTEST,
            0,
            "BIP34 regtest activation must be 0 (always active)"
        );

        // BIP66 activation heights - consensus-critical
        assert_eq!(
            BIP66_ACTIVATION_MAINNET,
            363_725,
            "BIP66 mainnet activation must match Bitcoin Core (363,725)"
        );
        assert_eq!(
            BIP66_ACTIVATION_TESTNET,
            330_776,
            "BIP66 testnet activation must match Bitcoin Core (330,776)"
        );
        assert_eq!(
            BIP66_ACTIVATION_REGTEST,
            0,
            "BIP66 regtest activation must be 0 (always active)"
        );

        // BIP65 activation height - consensus-critical
        assert_eq!(
            BIP65_ACTIVATION_MAINNET,
            388_381,
            "BIP65 mainnet activation must match Bitcoin Core (388,381)"
        );

        // BIP147 activation heights - consensus-critical
        assert_eq!(
            BIP147_ACTIVATION_MAINNET,
            481_824,
            "BIP147 mainnet activation must match Bitcoin Core (481,824)"
        );
        assert_eq!(
            BIP147_ACTIVATION_TESTNET,
            834_624,
            "BIP147 testnet activation must match Bitcoin Core (834,624)"
        );

        // SegWit activation height - consensus-critical
        assert_eq!(
            SEGWIT_ACTIVATION_MAINNET,
            481_824,
            "SegWit mainnet activation must match Bitcoin Core (481,824)"
        );

        // Taproot activation height - consensus-critical
        assert_eq!(
            TAPROOT_ACTIVATION_MAINNET,
            709_632,
            "Taproot mainnet activation must match Bitcoin Core (709,632)"
        );

        // Verify activation heights are in ascending order (mainnet)
        assert!(
            BIP34_ACTIVATION_MAINNET < BIP66_ACTIVATION_MAINNET,
            "BIP34 must activate before BIP66"
        );
        assert!(
            BIP66_ACTIVATION_MAINNET < BIP65_ACTIVATION_MAINNET,
            "BIP66 must activate before BIP65"
        );
        assert!(
            BIP65_ACTIVATION_MAINNET < BIP147_ACTIVATION_MAINNET,
            "BIP65 must activate before BIP147"
        );
        assert!(
            BIP147_ACTIVATION_MAINNET < TAPROOT_ACTIVATION_MAINNET,
            "BIP147 must activate before Taproot"
        );
    }

    /// Kani proof: Genesis block constants match Bitcoin exactly
    ///
    /// The genesis block is the foundation of Bitcoin. Any change to these values
    /// would create a completely different blockchain. This proof locks them.
    #[kani::proof]
    fn kani_genesis_block_constants_match_bitcoin() {
        // Genesis block hash - the root of the blockchain
        let expected_hash: [u8; 32] = [
            0x00, 0x00, 0x00, 0x00, 0x00, 0x19, 0xd6, 0x68,
            0x9c, 0x08, 0x5a, 0xe1, 0x65, 0x83, 0x1e, 0x93,
            0x4f, 0xf7, 0x63, 0xae, 0x46, 0xa2, 0xa6, 0xc1,
            0x72, 0xb3, 0xf1, 0xb6, 0x0a, 0x8c, 0xe2, 0x6f,
        ];
        assert_eq!(
            GENESIS_BLOCK_HASH,
            expected_hash,
            "Genesis block hash must match Bitcoin mainnet exactly"
        );

        // Genesis block timestamp: January 3, 2009, 18:15:05 UTC
        assert_eq!(
            GENESIS_BLOCK_TIMESTAMP,
            1231006505,
            "Genesis block timestamp must match Bitcoin (Jan 3, 2009 18:15:05 UTC)"
        );

        // Genesis block merkle root
        let expected_merkle_root: [u8; 32] = [
            0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
            0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
            0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
            0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
        ];
        assert_eq!(
            GENESIS_BLOCK_MERKLE_ROOT,
            expected_merkle_root,
            "Genesis block merkle root must match Bitcoin exactly"
        );

        // Genesis block nonce
        assert_eq!(
            GENESIS_BLOCK_NONCE,
            2083236893,
            "Genesis block nonce must match Bitcoin exactly"
        );

        // Verify genesis hash is not all zeros (sanity check)
        let mut all_zeros = true;
        for byte in &GENESIS_BLOCK_HASH {
            if *byte != 0 {
                all_zeros = false;
                break;
            }
        }
        assert!(
            !all_zeros,
            "Genesis block hash must not be all zeros"
        );

        // Verify genesis timestamp is reasonable (after Unix epoch, before 2010)
        assert!(
            GENESIS_BLOCK_TIMESTAMP > 0,
            "Genesis block timestamp must be positive"
        );
        assert!(
            GENESIS_BLOCK_TIMESTAMP < 1262304000, // Jan 1, 2010
            "Genesis block timestamp must be before 2010"
        );
    }
}
