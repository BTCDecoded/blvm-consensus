//! Integration tests for consensus validation

use blvm_consensus::mining::calculate_merkle_root;
use blvm_consensus::opcodes::OP_1;
use blvm_consensus::types::*;
use blvm_consensus::*;

#[test]
fn test_consensus_proof_basic_functionality() {
    let consensus = ConsensusProof::new();

    // Test basic transaction validation
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&tx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_consensus_proof_coinbase_validation() {
    let consensus = ConsensusProof::new();

    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x01, 0x00], // BIP34 height 0 — valid coinbase script length
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&coinbase_tx).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_consensus_proof_utxo_validation() {
    let consensus = ConsensusProof::new();

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let mut utxo_set = UtxoSet::default();
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 2000,
        script_pubkey: vec![OP_1].into(),
        height: 100,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    let (result, _total_value) = consensus.validate_tx_inputs(&tx, &utxo_set, 100).unwrap();
    assert!(matches!(result, ValidationResult::Valid));
}

#[test]
fn test_consensus_proof_insufficient_funds() {
    let consensus = ConsensusProof::new();

    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32].into(),
                index: 0,
            },
            script_sig: vec![OP_1],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 2000, // More than available
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let mut utxo_set = UtxoSet::default();
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = UTXO {
        value: 1000, // Less than needed
        script_pubkey: vec![OP_1].into(),
        height: 100,
        is_coinbase: false,
    };
    utxo_set.insert(outpoint, std::sync::Arc::new(utxo));

    let (result, _total_value) = consensus.validate_tx_inputs(&tx, &utxo_set, 100).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_consensus_proof_invalid_transaction() {
    let consensus = ConsensusProof::new();

    let invalid_tx = Transaction {
        version: 1,
        inputs: vec![].into(), // Empty inputs
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };

    let result = consensus.validate_transaction(&invalid_tx).unwrap();
    assert!(matches!(result, ValidationResult::Invalid(_)));
}

#[test]
fn test_consensus_proof_block_validation() {
    let consensus = ConsensusProof::new();

    let coinbase = Transaction {
        version: 2,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff,
            },
            script_sig: vec![0x01, 0x00],
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 5_000_000_000,
            script_pubkey: vec![OP_1].into(),
        }]
        .into(),
        lock_time: 0,
    };
    let merkle_root = calculate_merkle_root(std::slice::from_ref(&coinbase)).expect("merkle root");
    let block = Block {
        header: BlockHeader {
            version: 4,
            prev_block_hash: [0; 32],
            merkle_root,
            timestamp: 1_231_006_505,
            bits: 0x0300ffff,
            nonce: 0,
        },
        transactions: vec![coinbase].into(),
    };

    let utxo_set = UtxoSet::default();
    let witnesses: Vec<Vec<blvm_consensus::segwit::Witness>> = block
        .transactions
        .iter()
        .map(|tx| tx.inputs.iter().map(|_| Vec::new()).collect())
        .collect();
    let (result, new_utxo_set) = consensus
        .validate_block_with_time_context(&block, &witnesses, utxo_set, 0, None, Network::Regtest)
        .unwrap();
    assert_eq!(result, ValidationResult::Valid);
    assert!(!new_utxo_set.is_empty(), "valid coinbase must create UTXOs");
}

#[test]
fn test_consensus_proof_script_verification() {
    let consensus = ConsensusProof::new();

    let script_sig = vec![OP_1]; // OP_1
    let script_pubkey = vec![OP_1]; // OP_1

    let result = consensus
        .verify_script(&script_sig, &script_pubkey, None, 0)
        .unwrap();
    assert!(result, "OP_1/OP_1 must verify via ConsensusProof");
}

#[test]
fn test_consensus_proof_proof_of_work() {
    let consensus = ConsensusProof::new();

    let header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x0300ffff,
        nonce: 0,
    };

    let result = consensus.check_proof_of_work(&header).unwrap();
    let again = consensus.check_proof_of_work(&header).unwrap();
    assert_eq!(
        result, again,
        "PoW check must be deterministic for a fixed header"
    );
}

#[test]
fn test_consensus_proof_economic_functions() {
    let consensus = ConsensusProof::new();

    // Test block subsidy
    // Using Orange Paper constant: initial subsidy = 50 * C where C = 10^8
    use blvm_consensus::orange_paper_constants::C;
    let initial_subsidy = 50_i64 * C as i64;
    let subsidy = consensus.get_block_subsidy(0);
    assert_eq!(subsidy, initial_subsidy);

    // Test total supply
    // Using Orange Paper constant H (halving interval = 210,000)
    use blvm_consensus::orange_paper_constants::H;
    let supply = consensus.total_supply(H);
    assert!(supply > 0);

    // Test difficulty adjustment
    let current_header = BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    };

    let mut prev_headers = Vec::new();
    for i in 0..2016 {
        prev_headers.push(BlockHeader {
            version: 1,
            prev_block_hash: [i as u8; 32],
            merkle_root: [0; 32],
            timestamp: 1231006505 + (i * 600),
            bits: 0x1d00ffff,
            nonce: 0,
        });
    }

    let next_work = consensus
        .get_next_work_required(&current_header, &prev_headers)
        .unwrap();
    assert!(
        next_work > 0,
        "difficulty retarget must return positive bits"
    );
}
