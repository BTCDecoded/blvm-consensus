use consensus_proof::mempool::{
    accept_to_memory_pool, is_standard_tx, replacement_checks, Mempool,
};
use consensus_proof::{OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::HashSet;

fn create_test_transaction() -> Transaction {
    Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            script_sig: vec![0x51], // OP_1
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![0x51], // OP_1
        }],
        lock_time: 0,
    }
}

fn create_complex_transaction(input_count: usize, output_count: usize) -> Transaction {
    Transaction {
        version: 1,
        inputs: (0..input_count)
            .map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: {
                        let mut h = [0u8; 32];
                        h[0] = i as u8;
                        h
                    },
                    index: i as u64,
                },
                script_sig: vec![0x51],
                sequence: 0xffffffff,
            })
            .collect(),
        outputs: (0..output_count)
            .map(|_| TransactionOutput {
                value: 1000000000,
                script_pubkey: vec![0x51],
            })
            .collect(),
        lock_time: 0,
    }
}

fn benchmark_mempool_acceptance(c: &mut Criterion) {
    let tx = create_test_transaction();
    let utxo_set = UtxoSet::new();
    let mempool: Mempool = HashSet::new();

    c.bench_function("accept_to_memory_pool_simple", |b| {
        b.iter(|| {
            black_box(accept_to_memory_pool(
                black_box(&tx),
                black_box(&utxo_set),
                black_box(&mempool),
                black_box(0),
            ))
        })
    });
}

fn benchmark_mempool_acceptance_complex(c: &mut Criterion) {
    let tx = create_complex_transaction(5, 3);
    let utxo_set = UtxoSet::new();
    let mempool: Mempool = HashSet::new();

    c.bench_function("accept_to_memory_pool_complex", |b| {
        b.iter(|| {
            black_box(accept_to_memory_pool(
                black_box(&tx),
                black_box(&utxo_set),
                black_box(&mempool),
                black_box(0),
            ))
        })
    });
}

fn benchmark_is_standard_tx(c: &mut Criterion) {
    let tx = create_test_transaction();

    c.bench_function("is_standard_tx", |b| {
        b.iter(|| black_box(is_standard_tx(black_box(&tx))))
    });
}

fn benchmark_replacement_checks(c: &mut Criterion) {
    let mut new_tx = create_test_transaction();
    new_tx.inputs[0].sequence = 0xfffffffe; // RBF

    let mut existing_tx = create_test_transaction();
    existing_tx.inputs[0].sequence = 0xfffffffe; // RBF

    let mempool: Mempool = HashSet::new();

    c.bench_function("replacement_checks", |b| {
        b.iter(|| {
            black_box(replacement_checks(
                black_box(&new_tx),
                black_box(&existing_tx),
                black_box(&mempool),
            ))
        })
    });
}

criterion_group!(
    benches,
    benchmark_mempool_acceptance,
    benchmark_mempool_acceptance_complex,
    benchmark_is_standard_tx,
    benchmark_replacement_checks
);
criterion_main!(benches);
