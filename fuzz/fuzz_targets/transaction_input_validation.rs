#![no_main]
use bllvm_consensus::transaction::{check_transaction, check_tx_inputs};
use bllvm_consensus::types::{Hash, OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Transaction input validation fuzzing
    // Tests OutPoint validation, ScriptSig validation, and sequence number validation

    if data.is_empty() {
        return;
    }

    // Test 1: OutPoint validation
    // Build OutPoint from fuzzed data
    if data.len() >= 36 {
        let hash: Hash = data[0..32].try_into().unwrap_or([0; 32]);
        let index = if data.len() >= 36 {
            u32::from_le_bytes([data[32], data[33], data[34], data[35]]) as u64
        } else {
            0
        };

        let outpoint = OutPoint { hash, index };

        // Test OutPoint in transaction input
        let script_sig_len = if data.len() > 36 {
            (data[36] as usize).min(1000) // Limit scriptSig size
        } else {
            0
        };

        let script_sig = if data.len() > 37 && script_sig_len > 0 {
            let start = 37;
            let end = (start + script_sig_len).min(data.len());
            data[start..end].to_vec()
        } else {
            vec![]
        };

        let sequence = if data.len() >= 41 {
            u32::from_le_bytes([
                data.get(37).copied().unwrap_or(0),
                data.get(38).copied().unwrap_or(0),
                data.get(39).copied().unwrap_or(0),
                data.get(40).copied().unwrap_or(0),
            ])
        } else {
            0xffffffff
        };

        let input = TransactionInput {
            prevout: outpoint,
            script_sig: script_sig.into(),
            sequence,
        };

        // Test 2: Transaction with single input
        let tx = Transaction {
            version: 1,
            inputs: vec![input],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }],
            lock_time: 0,
        };

        let _result = check_transaction(&tx);
    }

    // Test 3: Multiple inputs
    if data.len() >= 100 {
        let input_count = (data[0] as usize).min(10); // Limit to 10 inputs
        let mut offset = 1;
        let mut inputs = Vec::new();

        for _ in 0..input_count {
            if offset + 36 > data.len() {
                break;
            }

            let hash: Hash = data[offset..offset + 32].try_into().unwrap_or([0; 32]);
            offset += 32;
            let index = if offset + 4 <= data.len() {
                u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as u64
            } else {
                0
            };
            offset += 4;

            let script_sig_len = if offset < data.len() {
                (data[offset] as usize).min(100)
            } else {
                0
            };
            offset += 1;

            let script_sig = if offset + script_sig_len <= data.len() {
                data[offset..offset + script_sig_len].to_vec()
            } else {
                vec![]
            };
            offset += script_sig_len;

            let sequence = if offset + 4 <= data.len() {
                u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ])
            } else {
                0xffffffff
            };
            offset += 4;

            inputs.push(TransactionInput {
                prevout: OutPoint { hash, index },
                script_sig: script_sig.into(),
                sequence,
            });
        }

        let tx = Transaction {
            version: 1,
            inputs,
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }],
            lock_time: 0,
        };

        let _result = check_transaction(&tx);
    }

    // Test 4: check_tx_inputs validation
    if data.len() >= 50 {
        // Build transaction
        let hash: Hash = data[0..32].try_into().unwrap_or([0; 32]);
        let input = TransactionInput {
            prevout: OutPoint {
                hash,
                index: if data.len() >= 36 {
                    u32::from_le_bytes([data[32], data[33], data[34], data[35]]) as u64
                } else {
                    0
                },
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        };

        let tx = Transaction {
            version: 1,
            inputs: vec![input],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: vec![].into(),
            }],
            lock_time: 0,
        };

        // Build UTXO set
        let mut utxo_set = UtxoSet::new();
        utxo_set.insert(
            OutPoint {
                hash,
                index: 0,
            },
            TransactionOutput {
                value: 2000,
                script_pubkey: vec![].into(),
            },
        );

        // Test check_tx_inputs
        let _result = check_tx_inputs(&tx, &utxo_set, 0);
    }

    // Test 5: Edge cases
    // Empty inputs
    let tx_empty_inputs = Transaction {
        version: 1,
        inputs: vec![],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }],
        lock_time: 0,
    };
    let _result_empty = check_transaction(&tx_empty_inputs);

    // Coinbase input (special case)
    let coinbase_input = TransactionInput {
        prevout: OutPoint {
            hash: [0; 32],
            index: 0xffffffff,
        },
        script_sig: vec![0x00, 0x00].into(),
        sequence: 0xffffffff,
    };
    let coinbase_tx = Transaction {
        version: 1,
        inputs: vec![coinbase_input],
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![].into(),
        }],
        lock_time: 0,
    };
    let _result_coinbase = check_transaction(&coinbase_tx);

    // Very large sequence number
    let large_sequence_input = TransactionInput {
        prevout: OutPoint {
            hash: [1; 32],
            index: 0,
        },
        script_sig: vec![].into(),
        sequence: 0xffffffff,
    };
    let tx_large_sequence = Transaction {
        version: 1,
        inputs: vec![large_sequence_input],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![].into(),
        }],
        lock_time: 0,
    };
    let _result_large_sequence = check_transaction(&tx_large_sequence);
});

