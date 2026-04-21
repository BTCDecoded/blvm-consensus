#![no_main]
//! Batch SHA256 / double-SHA256 / HASH160 helpers (SIMD path when enabled).
use blvm_consensus::optimizations::simd_vectorization::{batch_double_sha256, batch_hash160, batch_sha256};
use libfuzzer_sys::fuzz_target;

fn split_chunks<'a>(data: &'a [u8], n: usize) -> Vec<&'a [u8]> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    let chunk = (data.len() / n).max(1);
    let mut i = 0usize;
    for _ in 0..n {
        if i >= data.len() {
            break;
        }
        let end = (i + chunk).min(data.len());
        out.push(&data[i..end]);
        i = end;
    }
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n = (data[0] as usize % 12) + 1;
    let parts = split_chunks(&data[1..], n);
    let refs: Vec<&[u8]> = parts.iter().copied().collect();
    let _ = batch_sha256(&refs);
    let _ = batch_double_sha256(&refs);
    let _ = batch_hash160(&refs);
});
