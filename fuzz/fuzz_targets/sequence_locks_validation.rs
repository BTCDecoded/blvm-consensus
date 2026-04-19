#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use blvm_consensus::{
    locktime::{
        check_bip65, extract_sequence_locktime_value, extract_sequence_type_flag,
        get_locktime_type, is_sequence_disabled, locktime_types_match,
    },
    sequence_locks::{calculate_sequence_locks, evaluate_sequence_locks, sequence_locks},
    types::{BlockHeader, OutPoint, Transaction, TransactionInput},
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUTS: usize = 8;
const MAX_HEADERS: usize = 11;
const LOCKTIME_VERIFY_SEQUENCE: u32 = 0x01;
const LOCKTIME_THRESHOLD: u32 = 500_000_000;

#[derive(Debug)]
struct FuzzInput {
    tx_version: u64,
    flags: u32,
    block_height: u64,
    block_time: u64,
    lock_time_probe: u32,
    stack_locktime_probe: u32,
    inputs: Vec<FuzzInputEntry>,
    header_timestamps: Vec<u64>,
}

#[derive(Debug)]
struct FuzzInputEntry {
    sequence: u32,
    prev_height: u64,
}

impl<'a> Arbitrary<'a> for FuzzInputEntry {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            sequence: u.arbitrary()?,
            prev_height: u.arbitrary::<u64>()? & (i64::MAX as u64),
        })
    }
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let input_len = usize::from(u.int_in_range(0..=MAX_INPUTS as u8)?);
        let header_len = usize::from(u.int_in_range(0..=MAX_HEADERS as u8)?);

        let mut inputs = Vec::with_capacity(input_len);
        for _ in 0..input_len {
            inputs.push(u.arbitrary()?);
        }

        let mut header_timestamps = Vec::with_capacity(header_len);
        for _ in 0..header_len {
            // FIX: explicit type
            header_timestamps.push(u.arbitrary::<u32>()? as u64);
        }

        Ok(Self {
            tx_version: u.arbitrary()?,
            flags: u.arbitrary()?,
            // FIX: removed masking to allow overflow exploration
            block_height: u.arbitrary()?,
            block_time: u.arbitrary()?,
            lock_time_probe: u.arbitrary()?,
            stack_locktime_probe: u.arbitrary()?,
            inputs,
            header_timestamps,
        })
    }
}

fn build_headers(timestamps: &[u64]) -> Vec<BlockHeader> {
    timestamps
        .iter()
        .enumerate()
        .map(|(index, &timestamp)| BlockHeader {
            version: 0,
            prev_block_hash: [index as u8; 32],
            merkle_root: [timestamp as u8; 32],
            timestamp,
            bits: 0,
            nonce: index as u64,
        })
        .collect()
}

fn build_transaction(case: &FuzzInput) -> (Transaction, Vec<u64>) {
    let inputs: Vec<TransactionInput> = case
        .inputs
        .iter()
        .enumerate()
        .map(|(index, entry)| TransactionInput {
            prevout: OutPoint {
                hash: [index as u8; 32],
                index: index as u32,
            },
            sequence: u64::from(entry.sequence),
            script_sig: Vec::new(),
        })
        .collect();

    let prev_heights = case.inputs.iter().map(|entry| entry.prev_height).collect();

    (
        Transaction {
            version: case.tx_version,
            inputs: inputs.into(),
            outputs: Vec::new().into(),
            lock_time: u64::from(case.lock_time_probe),
        },
        prev_heights,
    )
}

fn bip68_enforced(tx_version: u64, flags: u32) -> bool {
    tx_version >= 2 && (flags & LOCKTIME_VERIFY_SEQUENCE) != 0
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(case) = FuzzInput::arbitrary(&mut u) else {
        return;
    };

    // Basic probes
    let _ = get_locktime_type(case.lock_time_probe);
    let _ = get_locktime_type(case.stack_locktime_probe);
    let _ = locktime_types_match(case.lock_time_probe, case.stack_locktime_probe);
    let _ = check_bip65(case.lock_time_probe, case.stack_locktime_probe);

    // Boundary invariant
    assert!(matches!(
        get_locktime_type(LOCKTIME_THRESHOLD - 1),
        blvm_consensus::locktime::LocktimeType::BlockHeight
    ));
    assert!(matches!(
        get_locktime_type(LOCKTIME_THRESHOLD),
        blvm_consensus::locktime::LocktimeType::Timestamp
    ));

    let headers = build_headers(&case.header_timestamps);
    let recent_headers = headers.as_slice();

    // FIX: randomize Some / None path
    let headers_opt = if case.header_timestamps.is_empty() {
        None
    } else {
        Some(recent_headers)
    };

    let (tx, prev_heights) = build_transaction(&case);

    let all_sequences_disabled = tx
        .inputs
        .iter()
        .all(|input| is_sequence_disabled(input.sequence as u32));

    // Bitfield invariants
    for entry in &case.inputs {
        let expected_type = (entry.sequence & 0x0040_0000) != 0;
        let expected_value = (entry.sequence & 0x0000_ffff) as u16;
        let expected_disabled = (entry.sequence & 0x8000_0000) != 0;

        assert_eq!(extract_sequence_type_flag(entry.sequence), expected_type);
        assert_eq!(extract_sequence_locktime_value(entry.sequence), expected_value);
        assert_eq!(is_sequence_disabled(entry.sequence), expected_disabled);
    }

    // FIX: no unwrap
    let Ok(calculated) =
        calculate_sequence_locks(&tx, case.flags, &prev_heights, headers_opt)
    else {
        return;
    };

    let evaluated = evaluate_sequence_locks(case.block_height, case.block_time, calculated);

    let Ok(combined) = sequence_locks(
        &tx,
        case.flags,
        &prev_heights,
        case.block_height,
        case.block_time,
        headers_opt,
    ) else {
        return;
    };

    // Core invariant
    assert_eq!(combined, evaluated);

    // BIP68 enforcement invariant
    if !bip68_enforced(tx.version, case.flags) {
        assert_eq!(calculated, (-1, -1));
    }

    // Disabled sequences invariant
    if all_sequences_disabled {
        assert_eq!(calculated, (-1, -1));
    }

    // Stronger evaluation invariants
    if calculated.0 >= 0 && case.block_height <= calculated.0 as u64 {
        assert!(!evaluated);
    }

    if calculated.1 >= 0 && case.block_time <= calculated.1 as u64 {
        assert!(!evaluated);
    }

    if evaluated {
        if calculated.0 >= 0 {
            assert!(case.block_height > calculated.0 as u64);
        }
        if calculated.1 >= 0 {
            assert!(case.block_time > calculated.1 as u64);
        }
    }
});