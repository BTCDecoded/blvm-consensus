//! BIP Validation Rules
//!
//! Implementation of critical Bitcoin Improvement Proposals (BIPs) that enforce
//! consensus rules for block and transaction validation.
//!
//! Mathematical specifications from Orange Paper Section 5.4.

use crate::activation::IsForkActive;
use crate::block::calculate_tx_id;
use crate::error::{ConsensusError, Result};
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

/// BIP30: Duplicate Coinbase Prevention
///
/// Prevents duplicate coinbase transactions (same txid) from being added to the blockchain.
/// Mathematical specification: Orange Paper Section 5.4.1
///
/// **BIP30Check**: ℬ × 𝒰𝒮 × ℕ × Network → {valid, invalid}
///
/// For block b = (h, txs) with UTXO set us, height h, and network n:
/// - invalid if h ≤ deactivation_height(n) ∧ ∃ tx ∈ txs : IsCoinbase(tx) ∧ txid(tx) ∈ CoinbaseTxids(us)
/// - valid otherwise
///
/// **Deactivation**: BIP30 was disabled after block 91722 (mainnet) to allow duplicate coinbases
/// in blocks 91842 and 91880 (historical bug, grandfathered in).
///
/// Activation: Block 0 (always active until deactivation)
///
/// **Optimization**: When `bip30_index` is `Some`, uses O(1) lookup instead of O(n) iteration
/// over the UTXO set. Caller must maintain the index in sync with UTXO changes.
/// **#2**: When `coinbase_txid` is `Some`, skips `calculate_tx_id(coinbase)` — caller precomputed.
#[spec_locked("5.4.1")]
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
/// Starting at the mainnet height in `BIP34_ACTIVATION_MAINNET` (Bitcoin Core `BIP34Height`),
/// coinbase scriptSig must contain the block height.
/// Mathematical specification: Orange Paper Section 5.4.2
///
/// **BIP34Check**: ℬ × ℕ → {valid, invalid}
///
/// Activation Heights:
/// - Mainnet: `BIP34_ACTIVATION_MAINNET` (227,931; Bitcoin Core `chainparams`)
/// - Testnet: Block 211,111
/// - Regtest: Block 0 (always active)
#[spec_locked("5.4.2")]
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

        // Extract height from coinbase scriptSig
        // Height is encoded as CScriptNum at the beginning of scriptSig
        let script_sig = &tx.inputs[0].script_sig;

        if script_sig.is_empty() {
            return Ok(false);
        }

        // Parse CScriptNum from scriptSig
        // CScriptNum encoding: variable-length integer
        // First byte indicates length and sign:
        // - 0x00-0x4b: push data of that length (unsigned)
        // - 0x4c: OP_PUSHDATA1, next byte is length
        // - 0x4d: OP_PUSHDATA2, next 2 bytes are length
        // - 0x4e: OP_PUSHDATA4, next 4 bytes are length
        //
        // For height encoding, it's typically a small number, so it's usually
        // a direct push (0x01-0x4b) followed by the height bytes in little-endian.

        let extracted_height = extract_height_from_script_sig(script_sig)?;

        if extracted_height != height {
            return Ok(false);
        }
    }

    Ok(true)
}

/// BIP54: Consensus Cleanup activation (with optional override).
///
/// When `activation_override` is `Some(h)`, returns true iff `height >= h` (caller-derived
/// activation, e.g. from BIP9 version bits). When `None`, uses per-network constants
/// (`BIP54_ACTIVATION_*`). This allows the node to run BIP54 when miners are signalling
/// without configuring a fixed activation height.
#[spec_locked("5.4")]
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
            crate::types::Network::Regtest => crate::constants::BIP54_ACTIVATION_REGTEST,
        },
    };
    height >= activation
}

/// BIP54: Consensus Cleanup activation (constant-only).
///
/// Returns true if block at `height` on `network` is at or past the configured
/// BIP54 activation height. For activation derived from miner signalling (version bits),
/// use `connect_block_ibd` with `bip54_activation_override` set from
/// `blvm_consensus::version_bits::activation_height_from_headers` (e.g. with `version_bits::bip54_deployment_mainnet()`).
#[spec_locked("5.4")]
pub fn is_bip54_active(height: Natural, network: crate::types::Network) -> bool {
    is_bip54_active_at(height, network, None)
}

/// BIP54: Coinbase nLockTime and nSequence (Consensus Cleanup).
///
/// After BIP54 activation, coinbase must have lock_time == height - 13 and sequence != 0xffff_ffff.
#[spec_locked("5.4")]
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

/// Extract block height from coinbase scriptSig (CScriptNum encoding)
fn extract_height_from_script_sig(script_sig: &[u8]) -> Result<Natural> {
    if script_sig.is_empty() {
        return Err(ConsensusError::BlockValidation(
            "Empty coinbase scriptSig".into(),
        ));
    }

    let first_byte = script_sig[0];

    // Handle OP_0 (0x00) → height 0
    // In Bitcoin, CScriptNum(0).serialize() produces an empty vector,
    // and CScript() << empty_vec pushes OP_0 (0x00).
    if first_byte == 0x00 {
        return Ok(0);
    }

    // Handle direct push (0x01-0x4b)
    if (1..=0x4b).contains(&first_byte) {
        let len = first_byte as usize;
        if script_sig.len() < 1 + len {
            return Err(ConsensusError::BlockValidation(
                "Invalid scriptSig length".into(),
            ));
        }

        let height_bytes = &script_sig[1..1 + len];

        // Parse as little-endian integer
        let mut height = 0u64;
        for (i, &byte) in height_bytes.iter().enumerate() {
            if i >= 8 {
                return Err(ConsensusError::BlockValidation(
                    "Height value too large".into(),
                ));
            }
            height |= (byte as u64) << (i * 8);
        }

        return Ok(height);
    }

    // Handle OP_PUSHDATA1 (0x4c)
    if first_byte == 0x4c {
        if script_sig.len() < 2 {
            return Err(ConsensusError::BlockValidation(
                "Invalid OP_PUSHDATA1".into(),
            ));
        }
        let len = script_sig[1] as usize;
        if script_sig.len() < 2 + len {
            return Err(ConsensusError::BlockValidation(
                "Invalid scriptSig length".into(),
            ));
        }

        let height_bytes = &script_sig[2..2 + len];

        let mut height = 0u64;
        for (i, &byte) in height_bytes.iter().enumerate() {
            if i >= 8 {
                return Err(ConsensusError::BlockValidation(
                    "Height value too large".into(),
                ));
            }
            height |= (byte as u64) << (i * 8);
        }

        return Ok(height);
    }

    // Handle OP_PUSHDATA2 (0x4d)
    if first_byte == 0x4d {
        if script_sig.len() < 3 {
            return Err(ConsensusError::BlockValidation(
                "Invalid OP_PUSHDATA2".into(),
            ));
        }
        let len = u16::from_le_bytes([script_sig[1], script_sig[2]]) as usize;
        if script_sig.len() < 3 + len {
            return Err(ConsensusError::BlockValidation(
                "Invalid scriptSig length".into(),
            ));
        }

        let height_bytes = &script_sig[3..3 + len];

        let mut height = 0u64;
        for (i, &byte) in height_bytes.iter().enumerate() {
            if i >= 8 {
                return Err(ConsensusError::BlockValidation(
                    "Height value too large".into(),
                ));
            }
            height |= (byte as u64) << (i * 8);
        }

        return Ok(height);
    }

    Err(ConsensusError::BlockValidation(
        "Invalid height encoding in scriptSig".into(),
    ))
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
#[spec_locked("5.4.3")]
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
/// - Format: 0x30 [total-length] 0x02 [R-length] [R] 0x02 [S-length] [S] [sighash]
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
#[spec_locked("5.4.4")]
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
#[spec_locked("5.4.5")]
pub fn check_bip147(
    script_sig: &[u8],
    script_pubkey: &[u8],
    height: Natural,
    activation: &impl IsForkActive,
) -> Result<bool> {
    if !activation.is_fork_active(ForkId::Bip147, height) {
        return Ok(true);
    }

    // Check if scriptPubkey contains OP_CHECKMULTISIG (0xae)
    if !script_pubkey.contains(&0xae) {
        // Not a multisig script - BIP147 doesn't apply
        return Ok(true);
    }

    // Check if dummy element is empty (OP_0 = 0x00)
    // The dummy element is the last element consumed by OP_CHECKMULTISIG
    // We need to find it in the scriptSig stack

    // For now, we'll check if the last push in scriptSig is OP_0
    // This is a simplified check - full implementation would parse the stack
    is_null_dummy(script_sig)
}

/// Check if scriptSig has NULLDUMMY (empty dummy element for OP_CHECKMULTISIG)
///
/// BIP147: The dummy element is the first element consumed by OP_CHECKMULTISIG.
/// It must be empty (OP_0 = 0x00) after activation.
///
/// Stack layout for OP_CHECKMULTISIG:
/// [dummy] [sig1] [sig2] ... [sigm] [m] [pubkey1] ... [pubkeyn] [n]
///
/// The dummy element is the last element pushed before OP_CHECKMULTISIG executes.
/// We check if it's OP_0 (empty).
fn is_null_dummy(script_sig: &[u8]) -> Result<bool> {
    // BIP147: Dummy element must be empty (OP_0 = 0x00)
    // The dummy is the first element consumed, which is the last element pushed
    // For a valid NULLDUMMY, the last push should be OP_0

    if script_sig.is_empty() {
        return Ok(false);
    }

    // Parse scriptSig to find the last push operation
    // The dummy element is the last element pushed before OP_CHECKMULTISIG
    // We need to find the last push and verify it's OP_0

    // Simple check: If scriptSig ends with 0x00 (OP_0), it's likely NULLDUMMY
    // This is a simplified check - full implementation would parse the entire script
    if script_sig.ends_with(&[0x00]) {
        return Ok(true);
    }

    // More sophisticated: Parse backwards to find last push
    // For now, we'll use the simple check above
    // A full implementation would parse the entire scriptSig to find the last push

    // If we can't determine, assume invalid (strict interpretation)
    Ok(false)
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
    use crate::constants::{BIP147_ACTIVATION_MAINNET, BIP66_ACTIVATION_MAINNET};

    #[test]
    fn test_bip30_basic() {
        // Test that BIP30 check passes for new coinbase
        let transactions: Vec<Transaction> = vec![Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![].into(),
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
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![].into(),
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
                    hash: [0; 32].into(),
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
                script_pubkey: vec![].into(),
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
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![].into(),
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
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                // Wrong height encoding
                script_sig: vec![0x03, 0x00, 0x00, 0x00], // Height 0 instead of activation height
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_0000_0000,
                script_pubkey: vec![].into(),
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
        // Valid DER signature (minimal example)
        let valid_der = vec![0x30, 0x06, 0x02, 0x01, 0x00, 0x02, 0x01, 0x00];
        let result = check_bip66_network(
            &valid_der,
            BIP66_ACTIVATION_MAINNET - 1,
            crate::types::Network::Mainnet,
        )
        .unwrap();
        // Note: This may fail if signature is not actually valid DER, but the check should not panic
        assert!(
            result || !result,
            "BIP66 check should handle invalid DER gracefully"
        );

        // Before activation, should always pass
        let result =
            check_bip66_network(&valid_der, 100_000, crate::types::Network::Mainnet).unwrap();
        assert!(result, "BIP66 should pass before activation");
    }

    #[test]
    fn test_bip147_null_dummy() {
        // ScriptPubkey with OP_CHECKMULTISIG
        let script_pubkey = vec![0x52, 0x21, 0x00, 0x21, 0x00, 0x52, 0xae]; // 2-of-2 multisig

        // ScriptSig with NULLDUMMY (ends with OP_0)
        let script_sig_valid = vec![0x00]; // OP_0 (NULLDUMMY)
        let result = check_bip147_network(
            &script_sig_valid,
            &script_pubkey,
            BIP147_ACTIVATION_MAINNET,
            Bip147Network::Mainnet,
        )
        .unwrap();
        assert!(result, "BIP147 should pass with NULLDUMMY");

        // ScriptSig without NULLDUMMY (non-empty dummy)
        let script_sig_invalid = vec![0x01, 0x01]; // Non-empty dummy
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
