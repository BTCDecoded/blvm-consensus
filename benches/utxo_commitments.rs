// UTXO commitments implementation moved to blvm-protocol.
// Bench with: cargo bench -p blvm-protocol (when protocol has utxo_commitments benches)
#[cfg(feature = "utxo-commitments")]
use blvm_consensus::{BlockHeader, types::Natural};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

#[cfg(feature = "utxo-commitments")]
fn create_test_commitment(_height: Natural) -> [u8; 32] {
    [0u8; 32] // Placeholder; real bench in blvm-protocol
}

#[cfg(feature = "utxo-commitments")]
fn create_test_header_chain(count: usize) -> Vec<BlockHeader> {
    (0..count)
        .map(|i| BlockHeader {
            version: 1,
            prev_block_hash: {
                let mut h = [0u8; 32];
                if i > 0 {
                    h[0] = (i - 1) as u8;
                }
                h
            },
            merkle_root: [0u8; 32],
            timestamp: i as u64,
            bits: 0x1d00ffff,
            nonce: 0,
        })
        .collect()
}

#[cfg(feature = "utxo-commitments")]
fn benchmark_verify_supply(c: &mut Criterion) {
    // Bench moved to blvm-protocol; this is a no-op placeholder
    let _ = create_test_commitment(100);
    c.bench_function("verify_supply_placeholder", |b| b.iter(|| black_box(0)));
}

#[cfg(feature = "utxo-commitments")]
fn benchmark_verify_header_chain(c: &mut Criterion) {
    let headers = create_test_header_chain(100);
    c.bench_function("verify_header_chain_placeholder", |b| {
        b.iter(|| black_box(headers.len()))
    });
}

#[cfg(not(feature = "utxo-commitments"))]
fn benchmark_verify_supply(_c: &mut Criterion) {}
#[cfg(not(feature = "utxo-commitments"))]
fn benchmark_verify_header_chain(_c: &mut Criterion) {}

criterion_group!(
    benches,
    benchmark_verify_supply,
    benchmark_verify_header_chain
);
criterion_main!(benches);
