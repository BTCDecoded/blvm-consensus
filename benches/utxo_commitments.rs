#[cfg(feature = "utxo-commitments")]
use consensus_proof::utxo_commitments::data_structures::UtxoCommitment;
#[cfg(feature = "utxo-commitments")]
use consensus_proof::utxo_commitments::verification::{verify_header_chain, verify_supply};
use consensus_proof::{BlockHeader, Hash, Natural};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "utxo-commitments")]
fn create_test_commitment(height: Natural) -> UtxoCommitment {
    UtxoCommitment {
        block_height: height,
        block_hash: [0u8; 32],
        total_supply: 50_0000_0000 * height as u64, // Simplified
        merkle_root: [0u8; 32],
        commitment_hash: [0u8; 32],
    }
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
    let commitment = create_test_commitment(100);

    c.bench_function("verify_supply", |b| {
        b.iter(|| {
            black_box(verify_supply(black_box(&commitment)));
        })
    });
}

#[cfg(feature = "utxo-commitments")]
fn benchmark_verify_header_chain(c: &mut Criterion) {
    let headers = create_test_header_chain(100);

    c.bench_function("verify_header_chain_100", |b| {
        b.iter(|| {
            black_box(verify_header_chain(black_box(&headers)));
        })
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
