//! Focused Performance Benchmarks
//!
//! Quick-running benchmarks that measure the most performance-critical operations.
//! Designed to run in < 30 seconds for rapid iteration during optimization work.
//!
//! Focus areas:
//! - Cryptographic operations (SHA256, double SHA256) - Now with SHA-NI + AVX2
//! - Transaction validation
//! - Block validation (realistic workloads)
//! - UTXO operations
//!
//! Run with: cargo bench --bench performance_focused --features production

use bllvm_consensus::{
    block::connect_block, segwit::Witness, types::Network, Block, BlockHeader, OutPoint, Transaction,
    TransactionInput, TransactionOutput, UtxoSet, UTXO,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ============================================================================
// CRYPTOGRAPHIC OPERATIONS (SHA-NI + AVX2)
// ============================================================================

fn bench_hash_single(c: &mut Criterion) {
    let data_32b = vec![0u8; 32]; // Transaction hash size
    let data_64b = vec![0u8; 64]; // Block header size

    let mut group = c.benchmark_group("hash_single");

    // Single SHA256 - 32 bytes (typical transaction hash)
    group.bench_function("sha256_32b", |b| {
        use sha2::{Digest, Sha256};
        b.iter(|| {
            let hash = Sha256::digest(black_box(&data_32b));
            black_box(hash)
        })
    });

    // Double SHA256 - 32 bytes (Bitcoin standard)
    group.bench_function("double_sha256_32b", |b| {
        use sha2::{Digest, Sha256};
        b.iter(|| {
            let hash1 = Sha256::digest(black_box(&data_32b));
            let hash2 = Sha256::digest(hash1);
            black_box(hash2)
        })
    });

    // Block header hash - 64 bytes
    group.bench_function("double_sha256_64b", |b| {
        use sha2::{Digest, Sha256};
        b.iter(|| {
            let hash1 = Sha256::digest(black_box(&data_64b));
            let hash2 = Sha256::digest(hash1);
            black_box(hash2)
        })
    });

    group.finish();
}

#[cfg(feature = "production")]
fn bench_hash_batch(c: &mut Criterion) {
    use bllvm_consensus::optimizations::simd_vectorization;

    let data = vec![0u8; 64];
    let mut group = c.benchmark_group("hash_batch");

    // Batch sizes that matter in real blocks
    for size in [8, 16, 32, 64, 128].iter() {
        let inputs: Vec<&[u8]> = vec![data.as_slice(); *size];

        group.bench_with_input(
            BenchmarkId::new("avx2_double_sha256", size),
            size,
            |b, _| {
                b.iter(|| black_box(simd_vectorization::batch_double_sha256(black_box(&inputs))))
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "production"))]
fn bench_hash_batch(_c: &mut Criterion) {}

// ============================================================================
// TRANSACTION VALIDATION
// ============================================================================

fn create_simple_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0,
            },
            script_sig: vec![0x51], // OP_1
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 1_000_000,                 // 0.01 BTC
            script_pubkey: vec![0x51].into(), // OP_1
        }]
        .into(),
        lock_time: 0,
    }
}

fn bench_transaction_basics(c: &mut Criterion) {
    let tx = create_simple_transaction();

    let mut group = c.benchmark_group("transaction");

    // Transaction serialization (needed for tx ID calculation)
    group.bench_function("serialize", |b| {
        use bllvm_consensus::serialization::transaction::serialize_transaction;
        b.iter(|| black_box(serialize_transaction(black_box(&tx))))
    });

    // Transaction ID calculation (SHA256D of serialized tx)
    group.bench_function("calculate_id", |b| {
        use bllvm_consensus::block::calculate_tx_id;
        b.iter(|| black_box(calculate_tx_id(black_box(&tx))))
    });

    group.finish();
}

// ============================================================================
// BLOCK VALIDATION (REALISTIC)
// ============================================================================

fn create_realistic_block(num_txs: usize) -> Block {
    let mut transactions = vec![
        // Coinbase
        Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [0; 32].into(),
                    index: 0xffffffff,
                },
                script_sig: vec![0x51; 4],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![TransactionOutput {
                value: 50_000_000_000,
                script_pubkey: vec![0x51].into(),
            }]
            .into(),
            lock_time: 0,
        },
    ];

    // Regular transactions
    for i in 0..num_txs {
        transactions.push(Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32].into(),
                    index: 0,
                },
                script_sig: vec![0x51; 20],
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![
                TransactionOutput {
                    value: 10_000_000,
                    script_pubkey: vec![0x51; 25].into(),
                },
                TransactionOutput {
                    value: 5_000_000,
                    script_pubkey: vec![0x51; 25],
                },
            ]
            .into(),
            lock_time: 0,
        });
    }

    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into(),
    }
}

fn bench_block_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_validation");
    group.sample_size(10); // Fewer samples for longer benchmarks

    // Small block (10 txs) - typical for quick blocks
    let block_10 = create_realistic_block(10);
    let witnesses_10: Vec<Witness> = block_10.transactions.iter().map(|_| Vec::new()).collect();

    group.bench_function("10_txs", |b| {
        b.iter(|| {
            let utxo_set = UtxoSet::new();
            let _result = connect_block(
                black_box(&block_10),
                black_box(&witnesses_10),
                black_box(utxo_set),
                black_box(0),
                black_box(None),
                black_box(Network::Mainnet),
            );
        })
    });

    // Medium block (100 txs) - typical average block
    let block_100 = create_realistic_block(100);
    let witnesses_100: Vec<Witness> = block_100.transactions.iter().map(|_| Vec::new()).collect();

    group.bench_function("100_txs", |b| {
        b.iter(|| {
            let utxo_set = UtxoSet::new();
            let _result = connect_block(
                black_box(&block_100),
                black_box(&witnesses_100),
                black_box(utxo_set),
                black_box(0),
                black_box(None),
                black_box(Network::Mainnet),
            );
        })
    });

    group.finish();
}

// ============================================================================
// UTXO OPERATIONS
// ============================================================================

fn bench_utxo_operations(c: &mut Criterion) {
    use std::collections::HashMap;

    let mut utxo_set: UtxoSet = HashMap::new();
    let outpoint = OutPoint {
        hash: [1; 32],
        index: 0,
    };
    let utxo = crate::UTXO {
        value: 1_000_000,
        script_pubkey: vec![0x51],
        height: 0,
        is_coinbase: false,
    };

    let mut group = c.benchmark_group("utxo");

    // Insert UTXO
    group.bench_function("insert", |b| {
        b.iter(|| {
            let mut set = utxo_set.clone();
            set.insert(black_box(outpoint.clone()), black_box(utxo.clone()));
            black_box(set)
        })
    });

    // Lookup UTXO
    utxo_set.insert(outpoint.clone(), utxo.clone());
    group.bench_function("get", |b| {
        b.iter(|| black_box(utxo_set.get(black_box(&outpoint))))
    });

    // Remove UTXO
    group.bench_function("remove", |b| {
        b.iter(|| {
            let mut set = utxo_set.clone();
            black_box(set.remove(black_box(&outpoint)))
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK GROUPS
// ============================================================================

criterion_group!(
    name = crypto_benches;
    config = Criterion::default();
    targets = bench_hash_single, bench_hash_batch
);

criterion_group!(
    name = transaction_benches;
    config = Criterion::default();
    targets = bench_transaction_basics
);

criterion_group!(
    name = block_benches;
    config = Criterion::default();
    targets = bench_block_validation
);

criterion_group!(
    name = utxo_benches;
    config = Criterion::default();
    targets = bench_utxo_operations
);

criterion_main!(
    crypto_benches,
    transaction_benches,
    block_benches,
    utxo_benches
);
