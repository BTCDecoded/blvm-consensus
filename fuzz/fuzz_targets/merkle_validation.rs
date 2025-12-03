#![no_main]
use bllvm_consensus::mining::calculate_merkle_root;
use bllvm_consensus::segwit::compute_witness_merkle_root;
use bllvm_consensus::types::{Block, Hash, Transaction, TransactionInput, TransactionOutput};
use bllvm_consensus::witness::Witness;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Merkle validation fuzzing
    // Tests Merkle root computation for blocks, witness commitments, and script paths

    if data.is_empty() {
        return;
    }

    // Test 1: Block Merkle root computation
    // Build transactions from fuzzed data
    let tx_count = if data.len() >= 1 {
        (data[0] as usize).min(100) // Limit to 100 transactions
    } else {
        1
    };

    let mut offset = 1;
    let mut transactions = Vec::new();

    for i in 0..tx_count {
        if offset >= data.len() {
            break;
        }

        // Parse version (4 bytes)
        let version = if offset + 4 <= data.len() {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as u64
        } else {
            1
        };
        offset += 4;

        // Parse input count (1 byte, limited)
        let input_count = if offset < data.len() {
            (data[offset] as usize).min(10) // Limit to 10 inputs
        } else {
            0
        };
        offset += 1;

        let mut inputs = Vec::new();
        for _ in 0..input_count {
            if offset + 36 <= data.len() {
                let hash: Hash = data[offset..offset + 32].try_into().unwrap_or([0; 32]);
                offset += 32;
                let index = if offset + 4 <= data.len() {
                    u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]) as u64
                } else {
                    0
                };
                offset += 4;

                inputs.push(TransactionInput {
                    prevout: bllvm_consensus::types::OutPoint { hash, index },
                    script_sig: vec![].into(),
                    sequence: 0xffffffff,
                });
            } else {
                break;
            }
        }

        // Parse output count (1 byte, limited)
        let output_count = if offset < data.len() {
            (data[offset] as usize).min(10) // Limit to 10 outputs
        } else {
            0
        };
        offset += 1;

        let mut outputs = Vec::new();
        for _ in 0..output_count {
            if offset + 8 <= data.len() {
                let value = u64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ]);
                offset += 8;

                // Script pubkey (variable length, limit to 100 bytes)
                let script_len = if offset < data.len() {
                    (data[offset] as usize).min(100)
                } else {
                    0
                };
                offset += 1;

                let script_pubkey = if offset + script_len <= data.len() {
                    data[offset..offset + script_len].to_vec()
                } else {
                    vec![]
                };
                offset += script_len;

                outputs.push(TransactionOutput {
                    value,
                    script_pubkey: script_pubkey.into(),
                });
            } else {
                break;
            }
        }

        // Locktime (4 bytes)
        let lock_time = if offset + 4 <= data.len() {
            u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as u64
        } else {
            0
        };
        offset += 4;

        transactions.push(Transaction {
            version,
            inputs,
            outputs,
            lock_time,
        });
    }

    // Test Merkle root computation
    if !transactions.is_empty() {
        let _merkle_root_result = calculate_merkle_root(&transactions);
    }

    // Test 2: Witness Merkle root computation
    // Build witnesses for SegWit transactions
    if !transactions.is_empty() {
        let mut witnesses = Vec::new();
        let mut witness_offset = offset;

        for _ in 0..transactions.len().min(100) {
            if witness_offset >= data.len() {
                break;
            }

            // Witness stack element count (1 byte, limited)
            let elem_count = if witness_offset < data.len() {
                (data[witness_offset] as usize).min(10) // Limit to 10 elements
            } else {
                0
            };
            witness_offset += 1;

            let mut witness_elements = Vec::new();
            for _ in 0..elem_count {
                if witness_offset >= data.len() {
                    break;
                }

                // Element length (1 byte, limited)
                let elem_len = if witness_offset < data.len() {
                    (data[witness_offset] as usize).min(100) // Limit to 100 bytes per element
                } else {
                    0
                };
                witness_offset += 1;

                let elem = if witness_offset + elem_len <= data.len() {
                    data[witness_offset..witness_offset + elem_len].to_vec()
                } else {
                    vec![]
                };
                witness_offset += elem_len;

                witness_elements.push(elem);
            }

            witnesses.push(Witness::from(witness_elements));
        }

        // Build block
        let block = Block {
            header: bllvm_consensus::types::BlockHeader {
                version: 1,
                prev_block_hash: [0; 32],
                merkle_root: [0; 32],
                timestamp: 0,
                bits: 0,
                nonce: 0,
            },
            transactions: transactions.clone(),
        };

        // Test witness merkle root computation
        if witnesses.len() == transactions.len() {
            let _witness_merkle_result = compute_witness_merkle_root(&block, &witnesses);
        }
    }

    // Test 3: Edge cases
    // Empty transaction list
    let _empty_result = calculate_merkle_root(&[]);

    // Single transaction
    if !transactions.is_empty() {
        let single_tx = vec![transactions[0].clone()];
        let _single_result = calculate_merkle_root(&single_tx);
    }

    // Large number of transactions (if we have enough data)
    if transactions.len() > 1 {
        // Test with subset
        let subset: Vec<Transaction> = transactions.iter().take(50).cloned().collect();
        let _subset_result = calculate_merkle_root(&subset);
    }
});

