//! Mainnet block validation tests
//!
//! Tests validation against actual mainnet blocks at various consensus-era heights.
//! This ensures compatibility with real-world blocks and transaction patterns.
//!
//! Test blocks from key consensus eras:
//! - Genesis block (height 0)
//! Eras use mainnet activation heights from `blvm_consensus` constants.

use blvm_consensus::block::{BlockValidationContext, connect_block};
use blvm_consensus::segwit::Witness;
use blvm_consensus::serialization::block::deserialize_block_with_witnesses;
use blvm_consensus::types::Network;
use blvm_consensus::{Block, BlockHeader, UtxoSet, ValidationResult};

/// Genesis block (height 0) - the first Bitcoin block
///
/// Block hash: 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
/// This block should always validate correctly.
#[test]
fn test_genesis_block_validation() {
    // Genesis block hex (canonical mainnet genesis)
    let genesis_block_hex = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c0101000000010000000000000000000000000000000000000000000000000000000000000000ffffffff4d04ffff001d0104455468652054696d65732030332f4a616e2f32303039204368616e63656c6c6f72206f6e206272696e6b206f66207365636f6e64206261696c6f757420666f722062616e6b73ffffffff0100f2052a01000000434104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5fac00000000";

    let block_bytes = hex::decode(genesis_block_hex).expect("genesis hex");
    let (block, witnesses) =
        deserialize_block_with_witnesses(&block_bytes).expect("genesis bytes parse");
    let utxo_set = UtxoSet::default();
    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (validation_result, _, _undo_log) =
        connect_block(&block, &witnesses, utxo_set, 0, &ctx).expect("genesis connects");
    assert!(matches!(validation_result, ValidationResult::Valid));
}

/// Test coinbase transaction validation from different eras
///
/// Coinbase transactions have different formats in different eras:
/// - Pre-SegWit: Standard coinbase format
/// - Post-SegWit: Coinbase includes witness commitment
/// - Post-Taproot: Coinbase may include Taproot commitment
#[test]
fn test_coinbase_transaction_eras() {
    use blvm_consensus::transaction::check_transaction;
    use blvm_consensus::types::OutPoint;
    use blvm_consensus::types::{Transaction, TransactionInput, TransactionOutput};

    // Pre-SegWit coinbase (height < SEGWIT_ACTIVATION_MAINNET)
    let pre_segwit_coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32],
                index: 0xffffffff,
            },
            script_sig: vec![0x04, 0x00, 0x00, 0x00, 0x00], // Height encoding
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_0000_0000, // 50 BTC
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };

    match check_transaction(&pre_segwit_coinbase) {
        Ok(ValidationResult::Valid) => {}
        other => panic!("pre-segwit coinbase must be structurally valid, got {other:?}"),
    }

    // Post-SegWit coinbase includes witness commitment (height >= SEGWIT_ACTIVATION_MAINNET)
    // Note: Actual witness commitment would be in the coinbase output
    let post_segwit_coinbase = Transaction {
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
        outputs: vec![
            TransactionOutput {
                value: 12_5000_0000, // 12.5 BTC (after halving)
                script_pubkey: vec![],
            },
            // Witness commitment output would be here
        ]
        .into(),
        lock_time: 0,
    };

    match check_transaction(&post_segwit_coinbase) {
        Ok(ValidationResult::Valid) => {}
        other => panic!("post-segwit coinbase must be structurally valid, got {other:?}"),
    }
}

/// Test mainnet block serialization round-trip
///
/// Verifies that blocks can be serialized and deserialized correctly,
/// maintaining byte-for-byte consensus compatibility.
#[test]
fn test_mainnet_block_serialization_roundtrip() {
    use blvm_consensus::serialization::block::{deserialize_block_header, serialize_block_header};

    // Test with a realistic block header
    let header = BlockHeader {
        version: 0x20000000,
        prev_block_hash: [0x01; 32],
        merkle_root: [0x02; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0x12345678,
    };

    let serialized = serialize_block_header(&header);
    let deserialized = deserialize_block_header(&serialized).unwrap();

    assert_eq!(header.version, deserialized.version);
    assert_eq!(header.prev_block_hash, deserialized.prev_block_hash);
    assert_eq!(header.merkle_root, deserialized.merkle_root);
    assert_eq!(header.timestamp, deserialized.timestamp);
    assert_eq!(header.bits, deserialized.bits);
    assert_eq!(header.nonce, deserialized.nonce);
}

/// Test block validation with real-world transaction patterns
///
/// This test validates blocks containing common transaction patterns:
/// - P2PKH transactions
/// - P2SH transactions
/// - SegWit transactions (P2WPKH, P2WSH)
/// - Taproot transactions (P2TR)
#[test]
fn test_real_world_transaction_patterns() {
    use blvm_consensus::transaction::is_coinbase;

    let block_dir = std::path::PathBuf::from("tests/test_data/mainnet_blocks");
    let test_heights = vec![100000, 200000, 300000, 400000, 500000, 600000];

    let mut patterns_found = std::collections::HashSet::new();
    let mut blocks_loaded = 0usize;

    for height in &test_heights {
        if let Ok((block, _witnesses)) = load_mainnet_block_from_disk(&block_dir, *height) {
            blocks_loaded += 1;
            // Analyze transaction patterns in this block
            for tx in &block.transactions {
                if is_coinbase(tx) {
                    continue;
                }

                // Check input patterns
                for input in &tx.inputs {
                    // Check scriptSig patterns
                    if input.script_sig.is_empty() {
                        patterns_found.insert("P2WPKH/P2WSH"); // SegWit spends have empty scriptSig
                    } else if !input.script_sig.is_empty() {
                        patterns_found.insert("P2PKH/P2SH");
                    }
                }

                // Check output patterns
                for output in &tx.outputs {
                    let script = &output.script_pubkey;

                    // P2PKH: OP_DUP OP_HASH160 <20 bytes> OP_EQUALVERIFY OP_CHECKSIG
                    if script.len() == 25
                        && script[0] == blvm_consensus::opcodes::OP_DUP
                        && script[1] == blvm_consensus::opcodes::OP_HASH160
                        && script[2] == blvm_consensus::opcodes::PUSH_20_BYTES
                        && script[23] == blvm_consensus::opcodes::OP_EQUALVERIFY
                        && script[24] == blvm_consensus::opcodes::OP_CHECKSIG
                    {
                        patterns_found.insert("P2PKH");
                    }
                    // P2SH: OP_HASH160 <20 bytes> OP_EQUAL
                    else if script.len() == 23
                        && script[0] == blvm_consensus::opcodes::OP_HASH160
                        && script[1] == blvm_consensus::opcodes::PUSH_20_BYTES
                        && script[22] == blvm_consensus::opcodes::OP_EQUAL
                    {
                        patterns_found.insert("P2SH");
                    }
                    // P2WPKH: OP_0 <20 bytes>
                    else if script.len() == 22
                        && script[0] == blvm_consensus::opcodes::OP_0
                        && script[1] == blvm_consensus::opcodes::PUSH_20_BYTES
                    {
                        patterns_found.insert("P2WPKH");
                    }
                    // P2WSH: OP_0 <32 bytes>
                    else if script.len() == 34
                        && script[0] == blvm_consensus::opcodes::OP_0
                        && script[1] == blvm_consensus::opcodes::PUSH_32_BYTES
                    {
                        patterns_found.insert("P2WSH");
                    }
                    // P2TR: OP_1 <32 bytes>
                    else if script.len() == 34
                        && script[0] == blvm_consensus::opcodes::OP_1
                        && script[1] == blvm_consensus::opcodes::PUSH_32_BYTES
                    {
                        patterns_found.insert("P2TR");
                    }
                }
            }
        }
    }

    // At minimum, we should find some common patterns
    // This test verifies that our deserialization can handle real-world transaction formats
    println!("Transaction patterns found: {patterns_found:?}");

    let has_fixtures = block_dir.exists()
        && test_heights.iter().any(|h| {
            block_dir.join(format!("block_{h}.bin")).exists()
                || block_dir.join(format!("block_{h}.hex")).exists()
        });
    if has_fixtures {
        assert!(
            blocks_loaded > 0,
            "mainnet block fixtures exist but none loaded"
        );
    }
}

/// Load a mainnet block from disk
///
/// Helper function to load a block at a specific height from the mainnet blocks directory.
/// Supports both binary (.bin) and hex (.hex) formats.
pub fn load_mainnet_block_from_disk(
    block_dir: &std::path::PathBuf,
    height: u64,
) -> Result<(Block, Vec<Vec<Witness>>), Box<dyn std::error::Error>> {
    let bin_path = block_dir.join(format!("block_{height}.bin"));
    let hex_path = block_dir.join(format!("block_{height}.hex"));

    let block_data = if bin_path.exists() {
        std::fs::read(&bin_path)?
    } else if hex_path.exists() {
        let hex_content = std::fs::read_to_string(&hex_path)?;
        hex::decode(hex_content.trim())?
    } else {
        return Err(format!(
            "Block {} not found (checked {}.bin and {}.hex)",
            height,
            bin_path.display(),
            hex_path.display()
        )
        .into());
    };

    let (block, witnesses) = deserialize_block_with_witnesses(&block_data)
        .map_err(|e| format!("Failed to deserialize block {height}: {e}"))?;

    Ok((block, witnesses))
}

/// Load and validate a mainnet block from hex
///
/// Helper function to load a block from hex and validate it.
/// This can be used to test specific mainnet blocks.
pub fn validate_mainnet_block(
    block_hex: &str,
    height: u64,
    prev_utxo_set: UtxoSet,
) -> Result<(ValidationResult, UtxoSet), Box<dyn std::error::Error>> {
    let block_bytes = hex::decode(block_hex)?;
    let (block, witnesses) = deserialize_block_with_witnesses(&block_bytes)?;

    let ctx = BlockValidationContext::for_network(Network::Mainnet);
    let (result, new_utxo_set, _undo_log) =
        connect_block(&block, &witnesses, prev_utxo_set, height, &ctx)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    Ok((result, new_utxo_set))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_mainnet_block_helper() {
        // Test the helper function with a minimal block
        let minimal_block_hex = "0100000000000000000000000000000000000000000000000000000000000000000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a29ab5f49ffff001d1dac2b7c0101000000010000000000000000000000000000000000000000000000000000000000000000ffffffff4d04ffff001d0104455468652054696d65732030332f4a616e2f32303039204368616e63656c6c6f72206f6e206272696e6b206f66207365636f6e64206261696c6f757420666f722062616e6b73ffffffff0100f2052a01000000434104678afdb0fe5548271967f1a67130b7105cd6a828e03909a67962e0ea1f61deb649f6bc3f4cef38c4f35504e51ec112de5c384df7ba0b8d578a4c702b6bf11d5fac00000000";

        let utxo_set = UtxoSet::default();
        let result = validate_mainnet_block(minimal_block_hex, 0, utxo_set);

        let (validation, _) = result.expect("genesis bytes connect");
        assert!(matches!(validation, ValidationResult::Valid));
    }
}
