#![no_main]
//! Cache-aligned hash helper (memory layout / optimization surface).
use blvm_consensus::optimizations::CacheAlignedHash;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut h = [0u8; 32];
    let take = data.len().min(32);
    h[..take].copy_from_slice(&data[..take]);
    let c = CacheAlignedHash::new(h);
    let _ = c.as_bytes();
});
