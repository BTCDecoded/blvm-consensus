//! BIP Validation Rules
//!
//! Implementation of critical Bitcoin Improvement Proposals (BIPs) that enforce
//! consensus rules for block and transaction validation.
//!
//! Mathematical specifications from Orange Paper Section 5.4.

use crate::activation::IsForkActive;
use crate::block::calculate_tx_id;
use crate::error::Result;
use crate::opcodes::{
    OP_0, OP_CHECKMULTISIG, OP_CHECKMULTISIGVERIFY, OP_PUSHDATA1, OP_PUSHDATA2, OP_PUSHDATA4,
};
use crate::transaction::is_coinbase;
use crate::types::*;
use blvm_spec_lock::spec_locked;

/// BIP30 index: maps coinbase txid → count of unspent outputs.
/// When count > 0, a coinbase with that txid has unspent outputs (BIP30 would reject duplicate).
/// Uses FxHashMap for faster lookups on integer-like keys (#17).
#[cfg(feature = "production")]
pub type Bip30Index = rustc_hash::FxHashMap<crate::types::Hash, usize>;
#[cfg(not(feature = "production"))]
pub type Bip30Index = std::collections::HashMap<crate::types::Hash, usize>;

/// Build Bip30Index from an existing UTXO set (for IBD resume).
/// Scans coinbase UTXOs and counts outputs per txid. O(n) over utxo_set.
pub fn build_bip30_index(utxo_set: &UtxoSet) -> Bip30Index {
    let mut index = Bip30Index::default();
    for (outpoint, utxo) in utxo_set.iter() {
        if utxo.is_coinbase {
            *index.entry(outpoint.hash).or_insert(0) += 1;
        }
    }
    index
}

/// Mainnet block 91842. Digest order, matching `block_header_hash`.
const BIP30_REPEAT_91842: Hash = [
    0xec, 0xca, 0xe0, 0x00, 0xe3, 0xc8, 0xe4, 0xe0, 0x93, 0x93, 0x63, 0x60, 0x43, 0x1f, 0x3b, 0x76,
    0x03, 0xc5, 0x63, 0xc1, 0xff, 0x61, 0x81, 0x39, 0x0a, 0x4d, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00,
];
/// Mainnet block 91880. Digest order, matching `block_header_hash`.
const BIP30_REPEAT_91880: Hash = [
    0x21, 0xd7, 0x7c, 0xcb, 0x4c, 0x08, 0x38, 0x6a, 0x04, 0xac, 0x01, 0x96, 0xae, 0x10, 0xf6, 0xa1,
    0xd2, 0xc2, 0xa3, 0x77, 0x55, 0x8c, 0xa1, 0x90, 0xf1, 0x43, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
];
/// Known mainnet block at the height-in-coinbase activation. Digest order.
const BIP34_MAINNET_HASH: Hash = [
    0xb8, 0x08, 0x08, 0x9c, 0x75, 0x6a, 0xdd, 0x15, 0x91, 0xb1, 0xd1, 0x7b, 0xab, 0x44, 0xbb, 0xa3,
    0xfe, 0xd9, 0xe0, 0x2f, 0x94, 0x2a, 0xb4, 0x89, 0x4b, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];
/// Known testnet block at the height-in-coinbase activation. Digest order.
const BIP34_TESTNET_HASH: Hash = [
    0xf8, 0x8e, 0xcd, 0x99, 0x12, 0xd0, 0x0d, 0x3f, 0x5c, 0x2a, 0x8e, 0x0f, 0x50, 0x41, 0x7d, 0x3e,
    0x41, 0x5c, 0x75, 0xb3, 0xab, 0xe5, 0x84, 0x34, 0x6d, 0xa9, 0xb3, 0x23, 0x00, 0x00, 0x00, 0x00,
];

/// From this height the duplicate-coinbase lookup is mandatory again, even on a chain
/// whose height-in-coinbase activation block is the known one. A coinbase from before
/// that rule can carry an indicated height this large.
pub const BIP34_IMPLIES_BIP30_LIMIT: u64 = 1_983_702;

/// True only for the two mainnet blocks whose coinbase repeats an earlier unspent one.
pub fn is_bip30_repeat_block(height: u64, block_hash: &Hash) -> bool {
    match height {
        91_842 => block_hash == &BIP30_REPEAT_91842,
        91_880 => block_hash == &BIP30_REPEAT_91880,
        _ => false,
    }
}

/// True when this header is the known block at the height-in-coinbase activation.
/// The header is hashed only at that height.
pub fn is_known_bip34_header(network: Network, height: u64, header: &BlockHeader) -> bool {
    let expected = match network {
        Network::Mainnet if height == crate::constants::BIP34_ACTIVATION_MAINNET => {
            &BIP34_MAINNET_HASH
        }
        Network::Testnet if height == crate::constants::BIP34_ACTIVATION_TESTNET => {
            &BIP34_TESTNET_HASH
        }
        _ => return false,
    };
    &crate::block::block_header_hash(header) == expected
}

/// True when `block_hash` is the known block at this network's height-in-coinbase activation.
pub fn is_known_bip34_block(network: Network, height: u64, block_hash: &Hash) -> bool {
    match network {
        Network::Mainnet => {
            height == crate::constants::BIP34_ACTIVATION_MAINNET
                && block_hash == &BIP34_MAINNET_HASH
        }
        Network::Testnet => {
            height == crate::constants::BIP34_ACTIVATION_TESTNET
                && block_hash == &BIP34_TESTNET_HASH
        }
        Network::Regtest | Network::Signet | Network::Testnet4 => false,
    }
}

/// The duplicate lookup may be skipped only after the known height-in-coinbase block,
/// and only before [`BIP34_IMPLIES_BIP30_LIMIT`].
pub fn bip30_lookup_skippable(network: Network, height: u64, bip34_hash_matches: bool) -> bool {
    if !bip34_hash_matches || height >= BIP34_IMPLIES_BIP30_LIMIT {
        return false;
    }
    let activation = match network {
        Network::Mainnet => crate::constants::BIP34_ACTIVATION_MAINNET,
        Network::Testnet => crate::constants::BIP34_ACTIVATION_TESTNET,
        Network::Regtest | Network::Signet | Network::Testnet4 => return false,
    };
    height > activation
}

/// BIP30: Duplicate Coinbase Prevention
///
/// A coinbase whose transaction id already has an unspent output is invalid.
/// Mathematical specification: Orange Paper Section 5.4.1
///
/// **BIP30Check**: ℬ × 𝒰𝒮 × ℕ × Network → {valid, invalid}
///
/// Two mainnet blocks are exempt, and only when the block hash matches:
/// height 91842 and height 91880. Every other duplicate is invalid, on every
/// network, at every height. There is no deactivation height.
///
/// **Optimization**: When `bip30_index` is `Some`, uses O(1) lookup instead of O(n) iteration
/// over the UTXO set. Caller must maintain the index in sync with UTXO changes.
/// **#2**: When `coinbase_txid` is `Some`, skips `calculate_tx_id(coinbase)` — caller precomputed.
#[spec_locked("5.4.1", "BIP30Check")]
pub fn check_bip30(
    block: &Block,
    utxo_set: &UtxoSet,
    bip30_index: Option<&Bip30Index>,
    height: Natural,
    activation: &impl IsForkActive,
    coinbase_txid: Option<&Hash>,
) -> Result<bool> {
    if !activation.is_fork_active(ForkId::Bip30, height) {
        return Ok(true);
    }
    // The two historical duplicates are exempt by hash. Any other block at
    // those heights is still checked. The header is hashed only then.
    if height == 91_842 || height == 91_880 {
        let hash = crate::block::block_header_hash(&block.header);
        if is_bip30_repeat_block(height, &hash) {
            return Ok(true);
        }
    }
    // Find coinbase transaction
    let coinbase = block.transactions.first();

    if let Some(tx) = coinbase {
        if !is_coinbase(tx) {
            // Not a coinbase transaction - BIP30 doesn't apply
            return Ok(true);
        }

        let txid = coinbase_txid
            .copied()
            .unwrap_or_else(|| calculate_tx_id(tx));

        // Fast path: O(1) lookup when index is provided
        if let Some(index) = bip30_index {
            if index.get(&txid).is_some_and(|&c| c > 0) {
                return Ok(false);
            }
            return Ok(true);
        }

        // Fallback: O(n) iteration when index not available (tests, sync path)
        // BIP30: Check if ANY UTXO exists with this txid
        for (outpoint, _utxo) in utxo_set.iter() {
            if outpoint.hash == txid {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// BIP34: Block Height in Coinbase
///
/// Starting at the mainnet height in `BIP34_ACTIVATION_MAINNET`,
/// coinbase scriptSig must contain the block height.
/// Mathematical specification: Orange Paper Section 5.4.2
///
/// **BIP34Check**: ℬ × ℕ → {valid, invalid}
///
/// Activation Heights:
/// - Mainnet: `BIP34_ACTIVATION_MAINNET` (227,931)
/// - Testnet: Block 211,111
/// - Regtest: Block 0 (always active)
#[spec_locked("5.4.2", "BIP34Check")]
pub fn check_bip34(block: &Block, height: Natural, activation: &impl IsForkActive) -> Result<bool> {
    if !activation.is_fork_active(ForkId::Bip34, height) {
        return Ok(true);
    }

    // Find coinbase transaction
    let coinbase = block.transactions.first();

    if let Some(tx) = coinbase {
        if !is_coinbase(tx) {
            return Ok(true);
        }

        // The script must begin with the minimal push of this height.
        // A longer encoding of the same integer is a different script.
        if !script_sig_starts_with_height(&tx.inputs[0].script_sig, height) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Little-endian magnitude of `height`, plus a trailing `0x00` when the high bit is set.
fn bip34_magnitude(height: u64) -> ([u8; 9], usize) {
    let mut raw = [0u8; 9];
    let mut n = height;
    let mut len = 0usize;
    while n > 0 {
        raw[len] = (n & 0xff) as u8;
        n >>= 8;
        len += 1;
    }
    if len > 0 && raw[len - 1] & 0x80 != 0 {
        raw[len] = 0x00;
        len += 1;
    }
    (raw, len)
}

/// Coinbase `scriptSig` whose prefix is the BIP34 height encoding.
///
/// Height 0 is `OP_0`. Heights 1..=16 are `OP_1`..=`OP_16`. Taller heights are
/// one length byte plus the minimal little-endian magnitude. A trailing `0xff`
/// keeps the script at least 2 bytes, the coinbase minimum.
pub fn encode_bip34_coinbase_script(height: u64) -> Vec<u8> {
    if height == 0 {
        return vec![0x00, 0xff];
    }
    if (1..=16).contains(&height) {
        return vec![0x50 + height as u8, 0xff];
    }
    let (raw, len) = bip34_magnitude(height);
    let mut script = Vec::with_capacity(len + 2);
    script.push(len as u8);
    script.extend_from_slice(&raw[..len]);
    if script.len() < 2 {
        script.push(0xff);
    }
    script
}

/// True when `script_sig` begins with the BIP34 height prefix.
///
/// Height 0 is `OP_0`. Heights 1 through 16 are the single opcodes `OP_1`..=`OP_16`
/// (`0x51`..=`0x60`), not a length-prefixed push of the same integer. Testnet4 and
/// signet activate BIP34 at height 1, and testnet4 block 1's coinbase starts with
/// `OP_1`. A longer height is one length byte plus the little-endian magnitude.
/// When the high bit of the last magnitude byte is set, one extra `0x00` follows so
/// the number stays non-negative. Bytes after that prefix are ignored.
fn script_sig_starts_with_height(script_sig: &[u8], height: u64) -> bool {
    if height == 0 {
        return script_sig.first().copied() == Some(0x00);
    }
    if (1..=16).contains(&height) {
        // OP_1 = 0x51. Heights 1..=16 are the single opcode 0x50 + height.
        return script_sig.first().copied() == Some(0x50 + height as u8);
    }
    let (raw, len) = bip34_magnitude(height);
    script_sig.len() > len && script_sig[0] == len as u8 && script_sig[1..1 + len] == raw[..len]
}

/// BIP54: Consensus Cleanup activation (with optional override).
///
/// Orange Paper §5.4.9. When `activation_override` is `Some(h)`, returns true iff `height >= h`
/// (caller-derived activation, e.g. from BIP9 version bits). When `None`, uses per-network constants
/// (`BIP54_ACTIVATION_*`). This allows the node to run BIP54 when miners are signalling
/// without configuring a fixed activation height.
#[spec_locked("5.4.9", "IsBip54ActiveAt")]
pub fn is_bip54_active_at(
    height: Natural,
    network: crate::types::Network,
    activation_override: Option<u64>,
) -> bool {
    let activation = match activation_override {
        Some(h) => h,
        None => match network {
            crate::types::Network::Mainnet => crate::constants::BIP54_ACTIVATION_MAINNET,
            crate::types::Network::Testnet => crate::constants::BIP54_ACTIVATION_TESTNET,
            crate::types::Network::Regtest
            | crate::types::Network::Signet
            | crate::types::Network::Testnet4 => crate::constants::BIP54_ACTIVATION_REGTEST,
        },
    };
    height >= activation
}

/// BIP54: Consensus Cleanup activation (constant-only).
///
/// Orange Paper §5.4.9. Returns true if block at `height` on `network` is at or past the configured
/// BIP54 activation height. For activation derived from miner signalling (version bits),
/// use `connect_block_ibd` with `bip54_activation_override` set from
/// `blvm_consensus::version_bits::activation_height_from_headers` (e.g. with `version_bits::bip54_deployment_mainnet()`).
#[spec_locked("5.4.9", "IsBip54Active")]
pub fn is_bip54_active(height: Natural, network: crate::types::Network) -> bool {
    is_bip54_active_at(height, network, None)
}

/// BIP54 timewarp mitigation at difficulty period boundaries (Orange Paper §5.4.9).
#[spec_locked("5.4.9", "BIP54TimewarpCheck")]
pub fn check_bip54_timewarp(
    header: &BlockHeader,
    height: Natural,
    boundary: Option<&Bip54BoundaryTimestamps>,
    bip54_active: bool,
) -> bool {
    if !bip54_active {
        return true;
    }
    let rem = height % 2016;
    if rem == 2015 {
        let Some(b) = boundary else {
            return false;
        };
        header.timestamp >= b.timestamp_n_minus_2015
    } else if rem == 0 {
        let Some(b) = boundary else {
            return false;
        };
        const TWOHOURS: u64 = 7200;
        let min_ts = b.timestamp_n_minus_1.saturating_sub(TWOHOURS);
        header.timestamp >= min_ts
    } else {
        true
    }
}

/// BIP54: reject non-coinbase txs with witness-stripped size exactly 64 bytes.
#[spec_locked("5.4.9", "CheckBip54TxStrippedSize")]
pub fn check_bip54_tx_stripped_size(tx: &Transaction) -> bool {
    is_coinbase(tx) || crate::transaction::calculate_transaction_size(tx) != 64
}

/// BIP54 per-transaction sigop cap (≤ 2,500 for non-coinbase when active).
#[spec_locked("5.4.9", "CheckBip54SigOpLimit")]
pub fn check_bip54_sigop_limit<U: crate::utxo_overlay::UtxoLookup>(
    bip54_active: bool,
    tx: &Transaction,
    utxo_lookup: &U,
    wits: Option<&[Witness]>,
    tx_flags: u32,
) -> Result<Option<&'static str>> {
    if !bip54_active || is_coinbase(tx) {
        return Ok(None);
    }
    let sigop_count =
        crate::sigop::get_transaction_sigop_count_for_bip54(tx, utxo_lookup, wits, tx_flags)?;
    if sigop_count > crate::constants::BIP54_MAX_SIGOPS_PER_TX {
        return Ok(Some("BIP54: Transaction sigop count exceeds 2500"));
    }
    Ok(None)
}

/// BIP54: Coinbase nLockTime and nSequence (Consensus Cleanup).
///
/// Orange Paper §5.4.9. After BIP54 activation, coinbase must have lock_time == height - 13 and sequence != 0xffff_ffff.
#[spec_locked("5.4.9", "CheckBip54Coinbase")]
pub fn check_bip54_coinbase(coinbase: &Transaction, height: Natural) -> bool {
    let required_lock_time = height.saturating_sub(13);
    if coinbase.lock_time != required_lock_time {
        return false;
    }
    if coinbase.inputs.is_empty() {
        return false;
    }
    if coinbase.inputs[0].sequence == 0xffff_ffff {
        return false;
    }
    true
}

// Network type is now in crate::types::Network

/// BIP66: Strict DER Signature Validation
///
/// Enforces strict DER encoding for ECDSA signatures.
/// Mathematical specification: Orange Paper Section 5.4.3
///
/// **BIP66Check**: 𝕊 × ℕ → {valid, invalid}
///
/// Activation Heights:
/// - Mainnet: `BIP66_ACTIVATION_MAINNET` (363,725)
/// - Testnet: Block 330,776
/// - Regtest: Block 0 (always active)
#[spec_locked("5.4.3", "BIP66Check")]
pub fn check_bip66(
    signature: &[u8],
    height: Natural,
    activation: &impl IsForkActive,
) -> Result<bool> {
    if !activation.is_fork_active(ForkId::Bip66, height) {
        return Ok(true);
    }

    // Check if signature is strictly DER-encoded
    // The secp256k1 library's from_der() method should enforce strict DER
    // We verify by attempting to parse and checking for strict compliance
    is_strict_der(signature)
}

/// Check if signature is strictly DER-encoded
///
/// Implements IsValidSignatureEncoding (BIP66 strict DER) exactly.
/// BIP66 requires strict DER encoding with specific rules:
/// - Format: `0x30 [total-length] 0x02 [R-length] [R] 0x02 [S-length] [S] [sighash]`
/// - No leading zeros in R or S (unless needed to prevent negative interpretation)
/// - Minimal length encoding
fn is_strict_der(signature: &[u8]) -> Result<bool> {
    // Format: 0x30 [total-length] 0x02 [R-length] [R] 0x02 [S-length] [S] [sighash]
    // * total-length: 1-byte length descriptor of everything that follows,
    //   excluding the sighash byte.
    // * R-length: 1-byte length descriptor of the R value that follows.
    // * R: arbitrary-length big-endian encoded R value. It must use the shortest
    //   possible encoding for a positive integer (which means no null bytes at
    //   the start, except a single one when the next byte has its highest bit set).
    // * S-length: 1-byte length descriptor of the S value that follows.
    // * S: arbitrary-length big-endian encoded S value. The same rules apply.
    // * sighash: 1-byte value indicating what data is hashed (not part of the DER
    //   signature)

    // Minimum and maximum size constraints.
    if signature.len() < 9 {
        return Ok(false);
    }
    if signature.len() > 73 {
        return Ok(false);
    }

    // A signature is of type 0x30 (compound).
    if signature[0] != 0x30 {
        return Ok(false);
    }

    // Make sure the length covers the entire signature.
    if signature[1] != (signature.len() - 3) as u8 {
        return Ok(false);
    }

    // Extract the length of the R element.
    let len_r = signature[3] as usize;

    // Make sure the length of the S element is still inside the signature.
    if 5 + len_r >= signature.len() {
        return Ok(false);
    }

    // Extract the length of the S element.
    let len_s = signature[5 + len_r] as usize;

    // Verify that the length of the signature matches the sum of the length
    // of the elements.
    if (len_r + len_s + 7) != signature.len() {
        return Ok(false);
    }

    // Check whether the R element is an integer.
    if signature[2] != 0x02 {
        return Ok(false);
    }

    // Zero-length integers are not allowed for R.
    if len_r == 0 {
        return Ok(false);
    }

    // Negative numbers are not allowed for R.
    if (signature[4] & 0x80) != 0 {
        return Ok(false);
    }

    // Null bytes at the start of R are not allowed, unless R would
    // otherwise be interpreted as a negative number.
    if len_r > 1 && signature[4] == 0x00 && (signature[5] & 0x80) == 0 {
        return Ok(false);
    }

    // Check whether the S element is an integer.
    if signature[len_r + 4] != 0x02 {
        return Ok(false);
    }

    // Zero-length integers are not allowed for S.
    if len_s == 0 {
        return Ok(false);
    }

    // Negative numbers are not allowed for S.
    if (signature[len_r + 6] & 0x80) != 0 {
        return Ok(false);
    }

    // Null bytes at the start of S are not allowed, unless S would otherwise be
    // interpreted as a negative number.
    if len_s > 1 && signature[len_r + 6] == 0x00 && (signature[len_r + 7] & 0x80) == 0 {
        return Ok(false);
    }

    Ok(true)
}

// Network type is now in crate::types::Network

/// BIP90: Block Version Enforcement
///
/// Enforces minimum block versions based on activation heights.
/// Mathematical specification: Orange Paper Section 5.4.4
///
/// **BIP90Check**: ℋ × ℕ → {valid, invalid}
///
/// Activation Heights:
/// - BIP34: Mainnet 227,931 (requires version >= 2)
/// - BIP66: Mainnet 363,725 (requires version >= 3)
/// - BIP65: Mainnet 388,381 (requires version >= 4)
#[spec_locked("5.4.4", "BIP90Check")]
pub fn check_bip90(
    block_version: i64,
    height: Natural,
    activation: &impl IsForkActive,
) -> Result<bool> {
    if activation.is_fork_active(ForkId::Bip34, height) && block_version < 2 {
        return Ok(false);
    }
    if activation.is_fork_active(ForkId::Bip66, height) && block_version < 3 {
        return Ok(false);
    }
    if activation.is_fork_active(ForkId::Bip65, height) && block_version < 4 {
        return Ok(false);
    }

    Ok(true)
}

/// Convenience: BIP30 check using network (builds activation table).
pub fn check_bip30_network(
    block: &Block,
    utxo_set: &UtxoSet,
    bip30_index: Option<&Bip30Index>,
    height: Natural,
    network: crate::types::Network,
    coinbase_txid: Option<&Hash>,
) -> Result<bool> {
    let table = crate::activation::ForkActivationTable::from_network(network);
    check_bip30(block, utxo_set, bip30_index, height, &table, coinbase_txid)
}

/// Convenience: BIP34 check using network.
pub fn check_bip34_network(
    block: &Block,
    height: Natural,
    network: crate::types::Network,
) -> Result<bool> {
    let table = crate::activation::ForkActivationTable::from_network(network);
    check_bip34(block, height, &table)
}

/// Convenience: BIP66 check using network (for script/signature callers).
pub fn check_bip66_network(
    signature: &[u8],
    height: Natural,
    network: crate::types::Network,
) -> Result<bool> {
    let table = crate::activation::ForkActivationTable::from_network(network);
    check_bip66(signature, height, &table)
}

/// Convenience: BIP90 check using network.
pub fn check_bip90_network(
    block_version: i64,
    height: Natural,
    network: crate::types::Network,
) -> Result<bool> {
    let table = crate::activation::ForkActivationTable::from_network(network);
    check_bip90(block_version, height, &table)
}

/// Convenience: BIP147 check using network (Bip147Network for backward compatibility).
pub fn check_bip147_network(
    script_sig: &[u8],
    script_pubkey: &[u8],
    height: Natural,
    network: Bip147Network,
) -> Result<bool> {
    let table = match network {
        Bip147Network::Mainnet => {
            crate::activation::ForkActivationTable::from_network(crate::types::Network::Mainnet)
        }
        Bip147Network::Testnet => {
            crate::activation::ForkActivationTable::from_network(crate::types::Network::Testnet)
        }
        Bip147Network::Regtest => {
            crate::activation::ForkActivationTable::from_network(crate::types::Network::Regtest)
        }
    };
    check_bip147(script_sig, script_pubkey, height, &table)
}

/// Returns true when `opcode` appears as an executable instruction (not inside push data).
pub(crate) fn script_contains_executable_opcode(script: &[u8], opcode: u8) -> bool {
    let mut i = 0;
    while i < script.len() {
        let op = script[i];
        if op > 0 && op < OP_PUSHDATA1 {
            i += 1 + op as usize;
            continue;
        }
        match op {
            OP_PUSHDATA1 => {
                if i + 1 >= script.len() {
                    break;
                }
                let len = script[i + 1] as usize;
                i += 2 + len;
            }
            OP_PUSHDATA2 => {
                if i + 2 >= script.len() {
                    break;
                }
                let len = u16::from_le_bytes([script[i + 1], script[i + 2]]) as usize;
                i += 3 + len;
            }
            OP_PUSHDATA4 => {
                if i + 4 >= script.len() {
                    break;
                }
                let len = u32::from_le_bytes([
                    script[i + 1],
                    script[i + 2],
                    script[i + 3],
                    script[i + 4],
                ]) as usize;
                i += 5 + len;
            }
            _ => {
                if op == opcode {
                    return true;
                }
                i += 1;
            }
        }
    }
    false
}

fn script_has_checkmultisig(script_pubkey: &[u8]) -> bool {
    script_contains_executable_opcode(script_pubkey, OP_CHECKMULTISIG)
        || script_contains_executable_opcode(script_pubkey, OP_CHECKMULTISIGVERIFY)
}

/// BIP147: NULLDUMMY Enforcement
///
/// Enforces that OP_CHECKMULTISIG dummy elements are empty.
/// Mathematical specification: Orange Paper Section 5.4.5
///
/// **BIP147Check**: 𝕊 × 𝕊 × ℕ → {valid, invalid}
///
/// Activation Heights:
/// - Mainnet: Block 481,824 (SegWit activation)
/// - Testnet: Block 834,624
/// - Regtest: Block 0 (always active)
#[spec_locked("5.4.5", "BIP147Check")]
pub fn check_bip147(
    script_sig: &[u8],
    script_pubkey: &[u8],
    height: Natural,
    activation: &impl IsForkActive,
) -> Result<bool> {
    if !activation.is_fork_active(ForkId::Bip147, height) {
        return Ok(true);
    }

    // BIP147 applies only when scriptPubKey executes OP_CHECKMULTISIG(VERIFY).
    if !script_has_checkmultisig(script_pubkey) {
        return Ok(true);
    }

    // BIP147: first stack item consumed by OP_CHECKMULTISIG (first push in scriptSig) must be empty.
    is_null_dummy(script_sig)
}

/// BIP147: The dummy element is the first element consumed by OP_CHECKMULTISIG.
/// It must be empty (OP_0) after activation.
fn is_null_dummy(script_sig: &[u8]) -> Result<bool> {
    if script_sig.is_empty() {
        return Ok(false);
    }

    let mut pc = 0usize;
    let mut first_push_empty = false;
    let mut saw_push = false;

    while pc < script_sig.len() {
        let opcode = script_sig[pc];
        pc += 1;
        match opcode {
            OP_0 => {
                if !saw_push {
                    first_push_empty = true;
                    saw_push = true;
                }
            }
            0x01..=0x4b => {
                let len = opcode as usize;
                if pc + len > script_sig.len() {
                    return Ok(false);
                }
                if !saw_push {
                    first_push_empty = len == 0;
                    saw_push = true;
                }
                pc += len;
            }
            OP_PUSHDATA1 => {
                if pc >= script_sig.len() {
                    return Ok(false);
                }
                let len = script_sig[pc] as usize;
                pc += 1;
                if pc + len > script_sig.len() {
                    return Ok(false);
                }
                if !saw_push {
                    first_push_empty = len == 0;
                    saw_push = true;
                }
                pc += len;
            }
            OP_PUSHDATA2 => {
                if pc + 2 > script_sig.len() {
                    return Ok(false);
                }
                let len = u16::from_le_bytes([script_sig[pc], script_sig[pc + 1]]) as usize;
                pc += 2;
                if pc + len > script_sig.len() {
                    return Ok(false);
                }
                if !saw_push {
                    first_push_empty = len == 0;
                    saw_push = true;
                }
                pc += len;
            }
            OP_PUSHDATA4 => {
                if pc + 4 > script_sig.len() {
                    return Ok(false);
                }
                let len = u32::from_le_bytes([
                    script_sig[pc],
                    script_sig[pc + 1],
                    script_sig[pc + 2],
                    script_sig[pc + 3],
                ]) as usize;
                pc += 4;
                if pc + len > script_sig.len() {
                    return Ok(false);
                }
                if !saw_push {
                    first_push_empty = len == 0;
                    saw_push = true;
                }
                pc += len;
            }
            _ => return Ok(false),
        }
    }

    Ok(first_push_empty)
}

/// Network type for BIP147 activation heights
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bip147Network {
    Mainnet,
    Testnet,
    Regtest,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::BIP147_ACTIVATION_MAINNET;

    use crate::opcodes::{OP_0, OP_1, OP_2, OP_CHECKMULTISIG, OP_CHECKSIG};

    #[test]
    fn bip34_height_1_through_16_is_a_single_opcode() {
        // Testnet4 block 1 coinbase scriptSig (mempool.space): OP_1, OP_0, then a 6-byte push.
        let block1: &[u8] = &[0x51, 0x00, 0x06, 0x2f, 0x40, 0x77, 0x69, 0x7a, 0x2f];
        assert!(script_sig_starts_with_height(block1, 1));
        assert!(!script_sig_starts_with_height(&[0x01, 0x01], 1));
        assert!(script_sig_starts_with_height(&[0x52, 0x00], 2));
        assert!(script_sig_starts_with_height(&[0x60], 16));
        // Height 17 is a one-byte push, not OP_17.
        assert!(script_sig_starts_with_height(&[0x01, 0x11, 0xff], 17));
        assert!(!script_sig_starts_with_height(&[0x51], 17));
        assert!(script_sig_starts_with_height(&[0x00], 0));
        assert_eq!(OP_1, 0x51);
    }

    #[test]
    fn encode_bip34_coinbase_script_satisfies_the_prefix_check() {
        for height in [0u64, 1, 16, 17, 127, 128, 150, 255, 256, 210_000] {
            let script = encode_bip34_coinbase_script(height);
            assert!(
                script_sig_starts_with_height(&script, height),
                "height {height} script {script:?}"
            );
            assert!(
                (2..=100).contains(&script.len()),
                "height {height} len {}",
                script.len()
            );
        }
        assert_eq!(encode_bip34_coinbase_script(0), vec![0x00, 0xff]);
        assert_eq!(encode_bip34_coinbase_script(1), vec![0x51, 0xff]);
        assert_eq!(encode_bip34_coinbase_script(16), vec![0x60, 0xff]);
        assert_eq!(encode_bip34_coinbase_script(17), vec![0x01, 0x11]);
        assert_eq!(encode_bip34_coinbase_script(127), vec![0x01, 0x7f]);
        assert_eq!(encode_bip34_coinbase_script(128), vec![0x02, 0x80, 0x00]);
        assert_eq!(encode_bip34_coinbase_script(150), vec![0x02, 0x96, 0x00]);
        assert_eq!(encode_bip34_coinbase_script(255), vec![0x02, 0xff, 0x00]);
        assert_eq!(encode_bip34_coinbase_script(256), vec![0x02, 0x00, 0x01]);
        assert_eq!(
            encode_bip34_coinbase_script(210_000),
            vec![0x03, 0x50, 0x34, 0x03]
        );
    }

    #[test]
    fn test_bip30_repeat_hashes_are_display_ids_reversed() {
        fn digest(display: &str) -> Hash {
            let mut bytes = hex::decode(display).unwrap();
            bytes.reverse();
            bytes.try_into().unwrap()
        }
        assert!(is_bip30_repeat_block(
            91_842,
            &digest("00000000000a4d0a398161ffc163c503763b1f4360639393e0e4c8e300e0caec")
        ));
        assert!(is_bip30_repeat_block(
            91_880,
            &digest("00000000000743f190a18c5577a3c2d2a1f610ae9601ac046a38084ccb7cd721")
        ));
        assert!(!is_bip30_repeat_block(91_842, &[0; 32]));
        assert!(!is_bip30_repeat_block(
            91_722,
            &digest("00000000000a4d0a398161ffc163c503763b1f4360639393e0e4c8e300e0caec")
        ));
        assert!(is_known_bip34_block(
            Network::Mainnet,
            crate::constants::BIP34_ACTIVATION_MAINNET,
            &digest("000000000000024b89b42a942fe0d9fea3bb44ab7bd1b19115dd6a759c0808b8")
        ));
        assert!(!is_known_bip34_block(
            Network::Mainnet,
            crate::constants::BIP34_ACTIVATION_MAINNET,
            &[0; 32]
        ));
    }

    #[test]
    fn test_bip30_basic() {
        // Test that BIP30 check passes for new coinbase
        let transactions: Vec<Transaction> = vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![],
            }]
            .into(),
            lock_time: 0,
        }];
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: transactions.into_boxed_slice(),
        };

        let utxo_set = UtxoSet::default();
        let result = check_bip30_network(
            &block,
            &utxo_set,
            None,
            0,
            crate::types::Network::Mainnet,
            None,
        )
        .unwrap();
        assert!(result, "BIP30 should pass for new coinbase");
    }

    #[test]
    fn test_bip34_before_activation() {
        let transactions: Vec<Transaction> = vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![],
            }]
            .into(),
            lock_time: 0,
        }];
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: transactions.into_boxed_slice(),
        };

        // Before activation, BIP34 should pass
        let result = check_bip34_network(&block, 100_000, crate::types::Network::Mainnet).unwrap();
        assert!(result, "BIP34 should pass before activation");
    }

    #[test]
    fn test_bip34_after_activation() {
        let height = crate::constants::BIP34_ACTIVATION_MAINNET;
        let transactions: Vec<Transaction> = vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                // Height encoded as CScriptNum: 0x03 (push 3 bytes) + height in little-endian
                script_sig: vec![
                    0x03,
                    (height & 0xff) as u8,
                    ((height >> 8) & 0xff) as u8,
                    ((height >> 16) & 0xff) as u8,
                ],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![],
            }]
            .into(),
            lock_time: 0,
        }];
        let block = Block {
            header: BlockHeader {
                version: 2, // BIP34 requires version >= 2
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: transactions.into_boxed_slice(),
        };

        let result = check_bip34_network(&block, height, crate::types::Network::Mainnet).unwrap();
        assert!(result, "BIP34 should pass with correct height encoding");
    }

    #[test]
    fn test_bip90_version_enforcement() {
        // Test version 1 before BIP34 activation
        let result = check_bip90_network(1, 100_000, crate::types::Network::Mainnet).unwrap();
        assert!(result, "Version 1 should be valid before BIP34");

        // Test version 1 after BIP34 activation (should fail)
        let result = check_bip90_network(
            1,
            crate::constants::BIP34_ACTIVATION_MAINNET,
            crate::types::Network::Mainnet,
        )
        .unwrap();
        assert!(
            !result,
            "Version 1 should be invalid after BIP34 activation"
        );

        // Test version 2 after BIP34 activation (should pass)
        let result = check_bip90_network(
            2,
            crate::constants::BIP34_ACTIVATION_MAINNET,
            crate::types::Network::Mainnet,
        )
        .unwrap();
        assert!(result, "Version 2 should be valid after BIP34 activation");

        // Test version 2 after BIP66 activation (should fail)
        // BIP66 activates at block 363,725, so we test at that height
        let result = check_bip90_network(2, 363_725, crate::types::Network::Mainnet).unwrap();
        assert!(
            !result,
            "Version 2 should be invalid after BIP66 activation"
        );

        // Test version 3 after BIP66 activation (should pass)
        // BIP66 activates at block 363,725, so we test at that height
        let result = check_bip90_network(3, 363_725, crate::types::Network::Mainnet).unwrap();
        assert!(result, "Version 3 should be valid after BIP66 activation");
    }

    #[test]
    fn test_bip30_duplicate_coinbase() {
        use crate::block::calculate_tx_id;

        // Create a coinbase transaction
        let coinbase_tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![],
            }]
            .into(),
            lock_time: 0,
        };

        let txid = calculate_tx_id(&coinbase_tx);

        // Create UTXO set with a UTXO from this coinbase
        let mut utxo_set = UtxoSet::default();
        utxo_set.insert(
            OutPoint {
                hash: txid,
                index: 0,
            },
            std::sync::Arc::new(UTXO {
                value: 50_0000_0000,
                script_pubkey: vec![].into(),
                height: 0,
                is_coinbase: false,
            }),
        );

        // Create block with same coinbase (duplicate)
        let transactions: Vec<Transaction> = vec![coinbase_tx];
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: transactions.into_boxed_slice(),
        };

        // BIP30 should fail for duplicate coinbase
        let result = check_bip30_network(
            &block,
            &utxo_set,
            None,
            0,
            crate::types::Network::Mainnet,
            None,
        )
        .unwrap();
        assert!(!result, "BIP30 should fail for duplicate coinbase");
    }

    #[test]
    fn test_bip34_invalid_height() {
        let height = crate::constants::BIP34_ACTIVATION_MAINNET;
        let transactions: Vec<Transaction> = vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32],
                    index: 0xffffffff,
                },
                // Wrong height encoding
                script_sig: vec![0x03, 0x00, 0x00, 0x00], // Height 0 instead of activation height
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![],
            }]
            .into(),
            lock_time: 0,
        }];
        let block = Block {
            header: BlockHeader {
                version: 2,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 1231006505,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: transactions.into_boxed_slice(),
        };

        // BIP34 should fail with wrong height
        let result = check_bip34_network(&block, height, crate::types::Network::Mainnet).unwrap();
        assert!(!result, "BIP34 should fail with incorrect height encoding");
    }

    #[test]
    fn test_bip66_strict_der() {
        use crate::constants::BIP66_ACTIVATION_MAINNET;
        use crate::types::Network;

        // 30 06 02 01 01 02 01 01 01 — tag, length 6, R = 1, S = 1, sighash 01.
        let accept: Vec<u8> = vec![0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01];
        assert_eq!(
            check_bip66_network(&accept, BIP66_ACTIVATION_MAINNET, Network::Mainnet).unwrap(),
            true,
            "9-byte strict DER must pass at activation"
        );

        let mut high_bit_r = accept.clone();
        high_bit_r[4] = 0x81;
        // R length 2, leading 0x00, next byte high bit clear.
        let leading_zero_r: Vec<u8> =
            vec![0x30, 0x07, 0x02, 0x02, 0x00, 0x01, 0x02, 0x01, 0x01, 0x01];
        let mut bad_length = accept.clone();
        bad_length[1] = 0x05;
        let rejects = [
            vec![0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01],
            vec![0u8; 74],
            high_bit_r.clone(),
            leading_zero_r,
            bad_length,
        ];
        for sig in &rejects {
            assert_eq!(
                check_bip66_network(sig, BIP66_ACTIVATION_MAINNET, Network::Mainnet).unwrap(),
                false,
                "non-strict encoding must fail at activation: {sig:?}"
            );
        }

        assert_eq!(
            check_bip66_network(&high_bit_r, BIP66_ACTIVATION_MAINNET - 1, Network::Mainnet)
                .unwrap(),
            true,
            "pre-activation must accept a non-strict signature"
        );
    }

    #[test]
    fn test_bip147_skips_checkmultisig_byte_in_push_data() {
        // Push data contains 0xae; executable path is OP_CHECKSIG only.
        let script_pubkey = vec![0x01, OP_CHECKMULTISIG, OP_CHECKSIG];
        let script_sig = vec![1, OP_1];
        let result = check_bip147_network(
            &script_sig,
            &script_pubkey,
            BIP147_ACTIVATION_MAINNET,
            Bip147Network::Mainnet,
        )
        .unwrap();
        assert!(
            result,
            "BIP147 must not apply when CHECKMULTISIG is only in push data"
        );
    }

    #[test]
    fn test_bip147_null_dummy() {
        // Executable OP_CHECKMULTISIG (matches bip_validation_suite fixture).
        let script_pubkey = vec![OP_CHECKMULTISIG];
        let script_sig_valid = vec![OP_0];
        let result = check_bip147_network(
            &script_sig_valid,
            &script_pubkey,
            BIP147_ACTIVATION_MAINNET,
            Bip147Network::Mainnet,
        )
        .unwrap();
        assert!(result, "BIP147 should pass with NULLDUMMY");

        let script_sig_invalid = vec![1, OP_1];
        let result = check_bip147_network(
            &script_sig_invalid,
            &script_pubkey,
            BIP147_ACTIVATION_MAINNET,
            Bip147Network::Mainnet,
        )
        .unwrap();
        assert!(!result, "BIP147 should fail without NULLDUMMY");

        // Before activation, should always pass
        let result = check_bip147_network(
            &script_sig_invalid,
            &script_pubkey,
            100_000,
            Bip147Network::Mainnet,
        )
        .unwrap();
        assert!(result, "BIP147 should pass before activation");
    }
}
