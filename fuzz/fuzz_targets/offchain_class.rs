#![no_main]
//! Rules the chain differential does not score.
//!
//! The chain oracle only accepts canonical blocks. These checks panic when a
//! generated input of that class gets the wrong verdict. Witness-program and
//! opcode verdicts are also compared with the script library in blvm-bench.
//! A truncated tapscript push is covered by the consensus unit sweep: reaching
//! that pre-scan from here would require a valid taproot witness.

use blvm_consensus::constants::MAX_MONEY;
use blvm_consensus::pow::{check_proof_of_work, next_required_bits};
use blvm_consensus::script::{SigVersion, cast_to_bool, verify_script_with_context_full};
use blvm_consensus::transaction::check_tx_inputs_with_owned_data;
use blvm_consensus::types::{
    BlockHeader, Network, OutPoint, Transaction, TransactionInput, TransactionOutput,
    ValidationResult,
};
use libfuzzer_sys::fuzz_target;

fn spend(script_sig: Vec<u8>, script_pubkey: Vec<u8>, height: u64) -> Option<bool> {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1u8; 32],
                index: 0,
            },
            sequence: 0xffff_ffff,
            script_sig,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 0,
            script_pubkey: vec![],
        }]
        .into(),
        lock_time: 0,
    };
    let values = vec![0i64];
    let scripts: Vec<&[u8]> = vec![script_pubkey.as_slice()];
    verify_script_with_context_full(
        &tx.inputs[0].script_sig,
        &script_pubkey,
        None,
        0x801,
        &tx,
        0,
        &values,
        &scripts,
        Some(height),
        None,
        Network::Mainnet,
        SigVersion::Base,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .ok()
}

fn header_with_bits(bits: u64) -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: [0u8; 32],
        merkle_root: [0u8; 32],
        timestamp: 1_300_000_000,
        bits,
        nonce: 0,
    }
}

fuzz_target!(|data: &[u8]| {
    // Future witness programs succeed only when the program bytes are true.
    // Before Taproot, a 32-byte version-1 program is in that set too.
    if data.len() >= 3 {
        let version = 0x51 + (data[0] % 16);
        let len = 2 + (data[1] as usize % 39);
        let mut payload = vec![0u8; len];
        let copy = (data.len() - 2).min(len);
        payload[..copy].copy_from_slice(&data[2..2 + copy]);
        let mut script = vec![version, len as u8];
        script.extend_from_slice(&payload);
        let truthy = cast_to_bool(&payload);
        let got = spend(vec![], script, 400_000).expect("witness program must not abort");
        assert_eq!(
            got, truthy,
            "witness program verdict payload={payload:02x?}"
        );
    }

    if data.len() >= 4 {
        let mut bits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64;
        bits |= 0x0080_0000;
        let negative = check_proof_of_work(&header_with_bits(bits)).unwrap();
        assert!(!negative, "negative compact target accepted: {bits:#x}");
        // Exponents outside 3..=32 are a decode error, not a zero target.
        let exponent = 3 + (data[0] as u64 % 30);
        let zero_mantissa = exponent << 24;
        let zero = check_proof_of_work(&header_with_bits(zero_mantissa)).unwrap();
        assert!(!zero, "zero mantissa accepted: {zero_mantissa:#x}");
    }

    if data.len() >= 8 {
        let parent_bits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64;
        let height = 1 + (u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as u64 % 2015);
        let ancestor = move |_h: u64| Some((parent_bits, 1_000u64));
        let required = next_required_bits(Network::Mainnet, height, 1_100, &ancestor).unwrap();
        assert_eq!(required, Some(parent_bits));

        let limit_ancestor = |_h: u64| Some((0x1b00_ffffu64, 1_000u64));
        let late = next_required_bits(Network::Testnet, 1, 1_000 + 1_201, &limit_ancestor).unwrap();
        assert_eq!(late, Some(0x1d00_ffff));
        let on_time =
            next_required_bits(Network::Testnet, 1, 1_000 + 1_200, &limit_ancestor).unwrap();
        assert_eq!(on_time, Some(0x1b00_ffff));
        let regtest_boundary =
            next_required_bits(Network::Regtest, 144, 1_100, &limit_ancestor).unwrap();
        assert_eq!(regtest_boundary, Some(0x1b00_ffff));
        let regtest_late =
            next_required_bits(Network::Regtest, 1, 1_000 + 1_201, &limit_ancestor).unwrap();
        assert_eq!(regtest_late, Some(0x207f_ffff));
    }

    if !data.is_empty() {
        let bump = 1 + (data[0] as i64 % 1_000);
        let half = MAX_MONEY / 2 + bump;
        let tx = Transaction {
            version: 1,
            inputs: vec![
                TransactionInput {
                    prevout: OutPoint {
                        hash: [2u8; 32],
                        index: 0,
                    },
                    sequence: 0xffff_ffff,
                    script_sig: vec![],
                },
                TransactionInput {
                    prevout: OutPoint {
                        hash: [3u8; 32],
                        index: 1,
                    },
                    sequence: 0xffff_ffff,
                    script_sig: vec![],
                },
            ]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: vec![],
            }]
            .into(),
            lock_time: 0,
        };
        let over = check_tx_inputs_with_owned_data(
            &tx,
            1_000,
            &[Some((half, false, 0)), Some((half, false, 0))],
        )
        .unwrap();
        assert!(
            matches!(over.0, ValidationResult::Invalid(_)),
            "input sum {half}+{half} accepted"
        );

        let one = Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [2u8; 32],
                    index: 0,
                },
                sequence: 0xffff_ffff,
                script_sig: vec![],
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 0,
                script_pubkey: vec![],
            }]
            .into(),
            lock_time: 0,
        };
        let at_cap =
            check_tx_inputs_with_owned_data(&one, 1_000, &[Some((MAX_MONEY, false, 0))]).unwrap();
        assert!(
            matches!(at_cap.0, ValidationResult::Valid),
            "a single max-money input was rejected"
        );
    }
});
