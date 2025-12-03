#![no_main]
use bllvm_consensus::constants::MAX_MONEY;
use bllvm_consensus::transaction::check_transaction;
use bllvm_consensus::types::{Transaction, TransactionInput, TransactionOutput};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Transaction output validation fuzzing
    // Tests value validation (MAX_MONEY checks), ScriptPubKey validation, and dust output detection

    if data.is_empty() {
        return;
    }

    // Test 1: Value validation
    // Parse value from fuzzed data
    if data.len() >= 8 {
        let value = u64::from_le_bytes([
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
        ]);

        // Test with different values
        let test_values = [
            value,
            0,
            1,
            1000,
            1000000,
            21000000 * 100000000, // 21M BTC in satoshis
            MAX_MONEY as u64,
            MAX_MONEY as u64 + 1, // Should fail
            u64::MAX, // Should fail
        ];

        for &test_value in &test_values {
            let tx = Transaction {
                version: 1,
                inputs: vec![TransactionInput {
                    prevout: bllvm_consensus::types::OutPoint {
                        hash: [1; 32],
                        index: 0,
                    },
                    script_sig: vec![].into(),
                    sequence: 0xffffffff,
                }],
                outputs: vec![TransactionOutput {
                    value: test_value as i64,
                    script_pubkey: vec![].into(),
                }],
                lock_time: 0,
            };

            let _result = check_transaction(&tx);
        }
    }

    // Test 2: ScriptPubKey validation
    if data.len() >= 10 {
        let script_len = (data[0] as usize).min(10000); // Limit script size
        let script_pubkey = if script_len > 0 && script_len <= data.len() - 1 {
            data[1..=script_len].to_vec()
        } else {
            vec![]
        };

        let value = if data.len() >= script_len + 9 {
            u64::from_le_bytes([
                data[script_len + 1],
                data[script_len + 2],
                data[script_len + 3],
                data[script_len + 4],
                data[script_len + 5],
                data[script_len + 6],
                data[script_len + 7],
                data[script_len + 8],
            ]) as i64
        } else {
            1000
        };

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: bllvm_consensus::types::OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value,
                script_pubkey: script_pubkey.into(),
            }],
            lock_time: 0,
        };

        let _result = check_transaction(&tx);
    }

    // Test 3: Multiple outputs
    if data.len() >= 20 {
        let output_count = (data[0] as usize).min(10); // Limit to 10 outputs
        let mut offset = 1;
        let mut outputs = Vec::new();

        for _ in 0..output_count {
            if offset + 8 > data.len() {
                break;
            }

            let value = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as i64;
            offset += 8;

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
        }

        let tx = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: bllvm_consensus::types::OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            }],
            outputs,
            lock_time: 0,
        };

        let _result = check_transaction(&tx);
    }

    // Test 4: Edge cases
    // Empty outputs
    let tx_empty_outputs = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: bllvm_consensus::types::OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }],
        outputs: vec![],
        lock_time: 0,
    };
    let _result_empty = check_transaction(&tx_empty_outputs);

    // Zero value output
    let tx_zero_value = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: bllvm_consensus::types::OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![].into(),
        }],
        lock_time: 0,
    };
    let _result_zero = check_transaction(&tx_zero_value);

    // Negative value (should fail)
    let tx_negative_value = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: bllvm_consensus::types::OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: -1,
            script_pubkey: vec![].into(),
        }],
        lock_time: 0,
    };
    let _result_negative = check_transaction(&tx_negative_value);

    // Very large scriptPubKey
    if data.len() > 100 {
        let large_script: Vec<u8> = data.iter().take(10000).copied().collect();
        let tx_large_script = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: bllvm_consensus::types::OutPoint {
                    hash: [1; 32],
                    index: 0,
                },
                script_sig: vec![].into(),
                sequence: 0xffffffff,
            }],
            outputs: vec![TransactionOutput {
                value: 1000,
                script_pubkey: large_script.into(),
            }],
            lock_time: 0,
        };
        let _result_large_script = check_transaction(&tx_large_script);
    }

    // Output sum overflow test
    let tx_overflow = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: bllvm_consensus::types::OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![].into(),
            sequence: 0xffffffff,
        }],
        outputs: vec![
            TransactionOutput {
                value: MAX_MONEY / 2,
                script_pubkey: vec![].into(),
            },
            TransactionOutput {
                value: MAX_MONEY / 2 + 1, // Should cause overflow
                script_pubkey: vec![].into(),
            },
        ],
        lock_time: 0,
    };
    let _result_overflow = check_transaction(&tx_overflow);
});

