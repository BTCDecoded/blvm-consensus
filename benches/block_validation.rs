use consensus_proof::block::connect_block;
use consensus_proof::{Block, BlockHeader, Transaction, UtxoSet};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn create_test_block() -> Block {
    Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![Transaction {
            version: 1,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
        }],
    }
}

fn benchmark_connect_block(c: &mut Criterion) {
    let block = create_test_block();
    let utxo_set = UtxoSet::new();

    c.bench_function("connect_block", |b| {
        b.iter(|| {
            let _result =
                connect_block(black_box(&block), black_box(utxo_set.clone()), black_box(0));
            // Ignore errors for benchmarking (they're expected for invalid test data)
        })
    });
}

fn benchmark_connect_block_multi_tx(c: &mut Criterion) {
    let mut transactions = vec![Transaction {
        version: 1,
        inputs: vec![],
        outputs: vec![],
        lock_time: 0,
    }];

    // Add 10 regular transactions
    for _ in 0..10 {
        transactions.push(Transaction {
            version: 1,
            inputs: vec![],
            outputs: vec![],
            lock_time: 0,
        });
    }

    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0; 32],
            merkle_root: [0; 32],
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions,
    };

    let utxo_set = UtxoSet::new();

    c.bench_function("connect_block_multi_tx", |b| {
        b.iter(|| {
            let _result =
                connect_block(black_box(&block), black_box(utxo_set.clone()), black_box(0));
            // Ignore errors for benchmarking
        })
    });
}

criterion_group!(
    benches,
    benchmark_connect_block,
    benchmark_connect_block_multi_tx
);
criterion_main!(benches);
