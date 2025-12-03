#![no_main]
use bllvm_consensus::taproot::{
    compute_taproot_signature_hash, compute_taproot_tweak, extract_taproot_output_key,
    is_taproot_output, validate_taproot_script, validate_taproot_script_path,
    validate_taproot_transaction, validate_taproot_key_aggregation,
};
use bllvm_consensus::types::{Hash, Transaction, TransactionOutput};
use bllvm_consensus::witness::Witness;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Taproot validation fuzzing
    // Tests Taproot script validation, key aggregation, signature hash computation, and transaction validation

    if data.is_empty() {
        return;
    }

    // Test 1: Taproot script validation
    // Minimum size for Taproot script: OP_1 (1 byte) + 32-byte hash = 33 bytes
    if data.len() >= 33 {
        let script: Vec<u8> = data.iter().take(33).copied().collect();
        let _result = validate_taproot_script(&script);
        
        // Test extraction of output key
        if let Ok(true) = validate_taproot_script(&script) {
            let _output_key = extract_taproot_output_key(&script);
        }
    }

    // Test 2: Taproot key aggregation
    // Need: internal_pubkey (32 bytes) + merkle_root (32 bytes) + output_key (32 bytes) = 96 bytes
    if data.len() >= 96 {
        let internal_pubkey: [u8; 32] = data[0..32].try_into().unwrap_or([0; 32]);
        let merkle_root: Hash = data[32..64].try_into().unwrap_or([0; 32]);
        let output_key: [u8; 32] = data[64..96].try_into().unwrap_or([0; 32]);

        // Test tweak computation
        let _tweak_result = compute_taproot_tweak(&internal_pubkey, &merkle_root);

        // Test key aggregation validation
        if let Ok(expected_key) = compute_taproot_tweak(&internal_pubkey, &merkle_root) {
            let _validation_result = validate_taproot_key_aggregation(
                &internal_pubkey,
                &merkle_root,
                &output_key,
            );
        }
    }

    // Test 3: Taproot script path validation
    // Need: script (variable) + merkle_proof (variable) + merkle_root (32 bytes)
    if data.len() >= 50 {
        let script_size = (data[0] as usize).min(1000).max(1);
        let script: Vec<u8> = if script_size <= data.len() - 1 {
            data[1..=script_size].to_vec()
        } else {
            data[1..].to_vec()
        };

        let merkle_root: Hash = if data.len() >= script.len() + 33 {
            let root_start = script.len() + 1;
            data[root_start..root_start + 32].try_into().unwrap_or([0; 32])
        } else {
            [0; 32]
        };

        // Build merkle proof from remaining data
        let proof_start = script.len() + 33;
        let proof_count = if proof_start < data.len() {
            ((data.len() - proof_start) / 32).min(16) // Limit to 16 proof elements
        } else {
            0
        };

        let mut merkle_proof = Vec::new();
        for i in 0..proof_count {
            let proof_start_idx = proof_start + (i * 32);
            if proof_start_idx + 32 <= data.len() {
                let proof_hash: Hash = data[proof_start_idx..proof_start_idx + 32]
                    .try_into()
                    .unwrap_or([0; 32]);
                merkle_proof.push(proof_hash);
            }
        }

        let _script_path_result = validate_taproot_script_path(&script, &merkle_proof, &merkle_root);
    }

    // Test 4: Taproot transaction validation
    // Build minimal transaction from fuzzed data
    if data.len() >= 50 {
        // Parse version (4 bytes)
        let version = if data.len() >= 4 {
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64
        } else {
            1
        };

        // Build outputs (at least one Taproot output)
        let mut outputs = Vec::new();
        let mut offset = 4;

        // Try to create at least one Taproot output
        if offset + 33 <= data.len() {
            let script_pubkey: Vec<u8> = data[offset..offset + 33].to_vec();
            offset += 33;

            // Value (8 bytes)
            let value = if offset + 8 <= data.len() {
                u64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ])
            } else {
                0
            };
            offset += 8;

            outputs.push(TransactionOutput {
                value,
                script_pubkey: script_pubkey.into(),
            });
        }

        // Build minimal transaction
        let tx = Transaction {
            version,
            inputs: vec![], // Empty inputs for simplicity
            outputs,
            lock_time: 0,
        };

        // Test transaction validation without witness
        let _result = validate_taproot_transaction(&tx, None);

        // Test with witness if we have enough data
        if offset < data.len() {
            let witness_data = &data[offset..];
            // Build witness structure (simplified: single stack element)
            if !witness_data.is_empty() {
                let witness_elem: Vec<u8> = witness_data.iter().take(1000).copied().collect();
                let witness = Witness::from(vec![witness_elem]);
                let _result_with_witness = validate_taproot_transaction(&tx, Some(&witness));
            }
        }

        // Test is_taproot_output
        for output in &tx.outputs {
            let _is_taproot = is_taproot_output(output);
        }
    }

    // Test 5: Taproot signature hash computation
    // Need: transaction + input_index + prevouts + sighash_type
    if data.len() >= 100 {
        // Build minimal transaction
        let version = if data.len() >= 4 {
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64
        } else {
            1
        };

        let input_count = if data.len() >= 5 {
            (data[4] as usize).min(10) // Limit to 10 inputs
        } else {
            1
        };

        let mut offset = 5;
        let mut inputs = Vec::new();

        // Build inputs
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

                inputs.push(bllvm_consensus::types::TransactionInput {
                    prevout: bllvm_consensus::types::OutPoint { hash, index },
                    script_sig: vec![].into(),
                    sequence: 0xffffffff,
                });
            } else {
                break;
            }
        }

        // Build prevouts (at least one)
        let mut prevouts = Vec::new();
        if offset + 41 <= data.len() {
            let value = if offset + 8 <= data.len() {
                u64::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ])
            } else {
                0
            };
            offset += 8;

            let script_pubkey: Vec<u8> = if offset + 33 <= data.len() {
                data[offset..offset + 33].to_vec()
            } else {
                vec![0; 33]
            };
            offset += 33;

            prevouts.push(TransactionOutput {
                value,
                script_pubkey: script_pubkey.into(),
            });
        }

        let tx = Transaction {
            version,
            inputs,
            outputs: vec![],
            lock_time: 0,
        };

        let input_index = if !tx.inputs.is_empty() {
            (data[0] as usize) % tx.inputs.len()
        } else {
            0
        };

        let sighash_type = if data.len() > 0 {
            data[0]
        } else {
            0
        };

        // Test signature hash computation
        if !prevouts.is_empty() && input_index < tx.inputs.len() {
            let _sig_hash_result = compute_taproot_signature_hash(
                &tx,
                input_index,
                &prevouts,
                sighash_type,
            );
        }
    }
});

