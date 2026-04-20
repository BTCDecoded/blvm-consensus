#![no_main]
//! BIP68 sequence locks: calculate then evaluate against a candidate height/time.
use blvm_consensus::bip113::get_median_time_past;
use blvm_consensus::sequence_locks::{calculate_sequence_locks, evaluate_sequence_locks};
use blvm_consensus::serialization::block::deserialize_block_header;
use blvm_consensus::serialization::transaction::deserialize_transaction;
use blvm_consensus::types::BlockHeader;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(tx) = deserialize_transaction(data) else {
        return;
    };
    let n = tx.inputs.len();
    if n == 0 {
        return;
    }
    let mut fb = [0u8; 4];
    fb.copy_from_slice(data.get(0..4).unwrap_or(&[1, 0, 0, 0]));
    let flags = u32::from_le_bytes(fb) | 0x01;
    let mut prev_heights = vec![0u64; n];
    for i in 0..n {
        let o = 4 + i * 8;
        if data.len() >= o + 8 {
            prev_heights[i] = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
        }
    }
    let mut headers: Vec<BlockHeader> = Vec::new();
    let mut off = 4 + n * 8;
    while data.len() >= off + 80 {
        if let Ok(h) = deserialize_block_header(&data[off..off + 80]) {
            headers.push(h);
            off += 80;
            if headers.len() >= 11 {
                break;
            }
        } else {
            break;
        }
    }
    let recent = if headers.is_empty() {
        None
    } else {
        Some(headers.as_slice())
    };
    if let Ok(pair) = calculate_sequence_locks(&tx, flags, &prev_heights, recent) {
        let bh = data
            .get(off)
            .map(|b| *b as u64)
            .unwrap_or(0)
            .saturating_add(500_000);
        let bt = get_median_time_past(&headers);
        let _ = evaluate_sequence_locks(bh, bt, pair);
    }
});
