//! Performance benchmarks for BLLVM optimizations
//!
//! These benchmarks measure the performance impact of BLLVM optimizations:
//! - Pre-allocation (Phase 1)
//! - Cache alignment (Phase 2)
//!
//! Run with: cargo bench --bench bllvm_optimizations --features production

#[cfg(feature = "production")]
use bllvm_consensus::optimizations;

use bllvm_consensus::{
    mining::calculate_merkle_root,
    serialization::{block::serialize_block_header, transaction::serialize_transaction},
    types::{BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput},
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Helper to create a test transaction
fn create_test_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [1; 32],
                index: 0,
            },
            script_sig: vec![0x51], // OP_1
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 1000,
            script_pubkey: vec![0x51], // OP_1
        }],
        lock_time: 0,
    }
}

/// Helper to create a test block header
fn create_test_block_header() -> BlockHeader {
    BlockHeader {
        version: 1,
        prev_block_hash: [0; 32],
        merkle_root: [0; 32],
        timestamp: 1231006505,
        bits: 0x1d00ffff,
        nonce: 0,
    }
}

/// Helper to create multiple test transactions
fn create_test_transactions(count: usize) -> Vec<Transaction> {
    (0..count)
        .map(|i| {
            let mut tx = create_test_transaction();
            tx.inputs[0].prevout.index = i as u64;
            tx
        })
        .collect()
}

/// Benchmark transaction serialization with/without production optimizations
fn bench_transaction_serialization(c: &mut Criterion) {
    let tx = create_test_transaction();

    let mut group = c.benchmark_group("transaction_serialization");

    group.bench_function("serialize_transaction", |b| {
        b.iter(|| {
            black_box(serialize_transaction(black_box(&tx)));
        });
    });

    group.finish();
}

/// Benchmark batch hash operations with different batch sizes
fn bench_batch_hashing(c: &mut Criterion) {
    let group = c.benchmark_group("batch_hashing");

    for size in [10, 100, 1000, 2000].iter() {
        let transactions = create_test_transactions(*size);
        let serialized: Vec<Vec<u8>> = transactions
            .iter()
            .map(serialize_transaction)
            .collect();
        let tx_refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();

        #[cfg(feature = "production")]
        {
            group.bench_with_input(
                BenchmarkId::new("batch_double_sha256_aligned", size),
                &tx_refs,
                |b, refs| {
                    b.iter(|| {
                        black_box(
                            optimizations::simd_vectorization::batch_double_sha256_aligned(
                                black_box(refs),
                            ),
                        );
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("batch_double_sha256", size),
                &tx_refs,
                |b, refs| {
                    b.iter(|| {
                        black_box(optimizations::simd_vectorization::batch_double_sha256(
                            black_box(refs),
                        ));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark merkle root calculation with different transaction counts
fn bench_merkle_root_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_root_calculation");

    for size in [10, 100, 500, 1000, 2000].iter() {
        let transactions = create_test_transactions(*size);

        group.bench_with_input(
            BenchmarkId::new("calculate_merkle_root", size),
            &transactions,
            |b, txs| {
                b.iter(|| {
                    black_box(calculate_merkle_root(black_box(txs)).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark block header serialization
fn bench_block_header_serialization(c: &mut Criterion) {
    let header = create_test_block_header();

    let mut group = c.benchmark_group("block_header_serialization");

    group.bench_function("serialize_block_header", |b| {
        b.iter(|| {
            black_box(serialize_block_header(black_box(&header)));
        });
    });

    group.finish();
}

/// Benchmark pre-allocation impact
#[cfg(feature = "production")]
fn bench_preallocation_impact(c: &mut Criterion) {
    use bllvm_consensus::optimizations::{prealloc_block_buffer, prealloc_tx_buffer};

    let mut group = c.benchmark_group("preallocation");

    group.bench_function("prealloc_tx_buffer", |b| {
        b.iter(|| {
            black_box(prealloc_tx_buffer());
        });
    });

    group.bench_function("prealloc_block_buffer", |b| {
        b.iter(|| {
            black_box(prealloc_block_buffer());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transaction_serialization,
    bench_batch_hashing,
    bench_merkle_root_calculation,
    bench_block_header_serialization,
);

#[cfg(feature = "production")]
criterion_group!(production_benches, bench_preallocation_impact,);

criterion_main!(benches);
