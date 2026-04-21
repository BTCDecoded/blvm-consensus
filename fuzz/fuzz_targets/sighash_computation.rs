#![no_main]

use blvm_consensus::transaction_hash::{
    calculate_bip143_sighash, calculate_transaction_sighash_single_input,
    Bip143PrecomputedHashes, SighashType,
};
use blvm_consensus::types::{OutPoint, Transaction, TransactionInput, TransactionOutput};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Sighash computation fuzzing
    // Exercises legacy and BIP143 sighash paths with non-circular spec-model invariants:
    //   - Determinism (same inputs -> same hash)
    //   - Script perturbation sensitivity (different script -> different hash)
    //   - SIGHASH_SINGLE quirk ([1, 0, ..., 0] when input_index >= outputs.len())
    //   - BIP143 precomputed equivalence (None vs Some -> identical hash)
    //   - Bounds errors (input_index >= inputs.len() -> Err)

    // Header: 23 bytes minimum
    // [0] scenario, [1] sighash_byte, [2] input_index_raw, [3] n_inputs, [4] n_outputs
    // [5..9] version, [9..13] lock_time, [13..21] prevout_value, [21] seq_seed, [22] val_seed
    // [23..] script data (split into script_a / script_b)
    if data.len() < 23 {
        return;
    }

    let scenario = data[0] % 6;
    let sighash_type_byte = data[1];
    let input_index_raw = data[2] as usize;
    let n_inputs = ((data[3] & 0x03) as usize).max(1); // 1..4
    let n_outputs = ((data[4] & 0x07) as usize).min(4); // 0..4
    let tx_version =
        u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as u64;
    let tx_lock_time =
        u32::from_le_bytes([data[9], data[10], data[11], data[12]]) as u64;
    let prevout_value = i64::from_le_bytes([
        data[13], data[14], data[15], data[16],
        data[17], data[18], data[19], data[20],
    ]);
    let seq_seed = data[21];
    let val_seed = data[22];

    // Script data: split at midpoint for perturbation invariant
    let script_data = &data[23..];
    let mid = script_data.len() / 2;
    let script_a = &script_data[..mid.min(520)];
    let script_b = &script_data[mid..script_data.len().min(mid + 520)];

    // Build transaction from fuzz bytes — all fields derived from input
    let base_sequence = u32::from_le_bytes([seq_seed, seq_seed, seq_seed, seq_seed]);
    let mut inputs = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let hash_byte = script_data.get(i).copied().unwrap_or(i as u8);
        inputs.push(TransactionInput {
            prevout: OutPoint {
                hash: [hash_byte; 32],
                index: i as u32,
            },
            script_sig: vec![].into(),
            sequence: (base_sequence ^ (i as u32)) as u64,
        });
    }

    let mut outputs = Vec::with_capacity(n_outputs);
    for i in 0..n_outputs {
        let out_val = script_data.get(n_inputs + i).copied().unwrap_or(val_seed);
        outputs.push(TransactionOutput {
            value: i64::from(out_val),
            script_pubkey: vec![0x76, 0xa9, 0x14].into(),
        });
    }

    let tx = Transaction {
        version: tx_version,
        inputs: inputs.into(),
        outputs: outputs.into(),
        lock_time: tx_lock_time,
    };

    match scenario {
        // Scenario 0: Legacy determinism + script perturbation
        0 => {
            let input_index = input_index_raw % tx.inputs.len();
            let sighash_type = SighashType::from_byte(sighash_type_byte);

            let hash_a1 = calculate_transaction_sighash_single_input(
                &tx, input_index, script_a, prevout_value, sighash_type, None,
            )
            .expect("valid input_index must not fail");

            let hash_a2 = calculate_transaction_sighash_single_input(
                &tx, input_index, script_a, prevout_value, sighash_type, None,
            )
            .expect("valid input_index must not fail");

            assert_eq!(hash_a1, hash_a2, "legacy sighash must be deterministic");

            if script_a != script_b {
                let hash_b = calculate_transaction_sighash_single_input(
                    &tx, input_index, script_b, prevout_value, sighash_type, None,
                )
                .expect("valid input_index must not fail");

                // EXPLANATION OF THE SIGHASH_SINGLE BUG:
                // In legacy Bitcoin consensus, SIGHASH_SINGLE has a well-known bug. 
                // If the input index being signed is greater than or equal to the total 
                // number of transaction outputs, the protocol does not fail. Instead, 
                // it immediately returns a literal `1` (formatted as a 32-byte 
                // little-endian array starting with 1 followed by 31 zeros).
                //
                // Because this hardcoded error hash is returned before the script data 
                // is ever processed, passing in two different scripts (script_a and script_b) 
                // will produce the exact same `[1, 0, 0...]` hash. We must check for this 
                // specific hash and skip our script perturbation assertion if we hit it.
                let mut single_bug_hash = [0u8; 32];
                single_bug_hash[0] = 1;
                
                // Only assert they are different if we didn't hit the bug
                if hash_a1 != single_bug_hash {
                    assert_ne!(
                        hash_a1, hash_b,
                        "different script_for_signing must produce different hash (unless SIGHASH_SINGLE bug)"
                    );
                }
            }
        }

        // Scenario 1: SIGHASH_SINGLE data-driven quirk
        1 => {
            // Force SIGHASH_SINGLE, preserve ANYONECANPAY flag
            let forced_byte = (sighash_type_byte & 0x80) | 0x03;
            let sighash_type = SighashType::from_byte(forced_byte);
            let input_index = input_index_raw % tx.inputs.len();

            let result = calculate_transaction_sighash_single_input(
                &tx, input_index, script_a, prevout_value, sighash_type, None,
            )
            .expect("valid input_index must not fail");

            if input_index >= tx.outputs.len() {
                // SIGHASH_SINGLE quirk: must return exactly [1, 0, 0, ..., 0]
                let mut expected = [0u8; 32];
                expected[0] = 1;
                assert_eq!(
                    result, expected,
                    "SIGHASH_SINGLE quirk: must be [1, 0..0]"
                );
            } else {
                // Normal SINGLE — verify determinism
                let result2 = calculate_transaction_sighash_single_input(
                    &tx, input_index, script_a, prevout_value, sighash_type, None,
                )
                .expect("valid input_index must not fail");
                assert_eq!(result, result2);
            }
        }

        // Scenario 2: Legacy bounds error
        2 => {
            let input_index = tx.inputs.len() + (input_index_raw % 256);
            let result = calculate_transaction_sighash_single_input(
                &tx,
                input_index,
                script_a,
                prevout_value,
                SighashType::from_byte(sighash_type_byte),
                None,
            );
            assert!(
                result.is_err(),
                "out-of-bounds input_index must return Err"
            );
        }

        // Scenario 3: BIP143 determinism + script perturbation
        3 => {
            let input_index = input_index_raw % tx.inputs.len();

            let hash_a1 = calculate_bip143_sighash(
                &tx, input_index, script_a, prevout_value, sighash_type_byte, None,
            )
            .expect("valid input_index must not fail");

            let hash_a2 = calculate_bip143_sighash(
                &tx, input_index, script_a, prevout_value, sighash_type_byte, None,
            )
            .expect("valid input_index must not fail");

            assert_eq!(hash_a1, hash_a2, "BIP143 sighash must be deterministic");

            if script_a != script_b {
                let hash_b = calculate_bip143_sighash(
                    &tx, input_index, script_b, prevout_value, sighash_type_byte,
                    None,
                )
                .expect("valid input_index must not fail");
                assert_ne!(
                    hash_a1, hash_b,
                    "different script_code must produce different BIP143 hash"
                );
            }
        }

        // Scenario 4: BIP143 precomputed equivalence
        4 => {
            let input_index = input_index_raw % tx.inputs.len();

            let hash_none = calculate_bip143_sighash(
                &tx, input_index, script_a, prevout_value, sighash_type_byte, None,
            )
            .expect("valid input_index must not fail");

            let precomputed = Bip143PrecomputedHashes::compute(&tx, &[], &[]);
            let hash_pre = calculate_bip143_sighash(
                &tx,
                input_index,
                script_a,
                prevout_value,
                sighash_type_byte,
                Some(&precomputed),
            )
            .expect("valid input_index must not fail");

            assert_eq!(
                hash_none, hash_pre,
                "BIP143 with None vs Some(precomputed) must produce identical hash"
            );
        }

        // Scenario 5: BIP143 bounds error
        5 => {
            let input_index = tx.inputs.len() + (input_index_raw % 256);
            let result = calculate_bip143_sighash(
                &tx, input_index, script_a, prevout_value, sighash_type_byte, None,
            );
            assert!(
                result.is_err(),
                "out-of-bounds input_index must return Err for BIP143"
            );
        }

        _ => unreachable!(),
    }
});
