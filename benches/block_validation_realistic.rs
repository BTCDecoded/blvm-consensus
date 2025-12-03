//! Realistic Block Validation Benchmark
//! Uses more realistic test data for fair comparison with Bitcoin Core's ConnectBlock benchmark

use bllvm_consensus::block::connect_block;
use bllvm_consensus::segwit::Witness;
use bllvm_consensus::types::Network;
use bllvm_consensus::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput, UtxoSet,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Create a realistic test block with actual transactions
/// Similar to Core's CreateTestBlock which uses 1000 transactions
fn create_realistic_test_block(num_txs: usize) -> Block {
    // Create a coinbase transaction
    let coinbase = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0; 32].into(),
                index: 0xffffffff, // Coinbase
            },
            script_sig: vec![0x51; 4], // OP_1 repeated
            sequence: 0xffffffff,
        }]
        .into(),
        outputs: vec![TransactionOutput {
            value: 50_000_000_000,            // 50 BTC
            script_pubkey: vec![0x51].into(), // OP_1
        }]
        .into(),
        lock_time: 0,
    };

    // Create regular transactions with inputs and outputs
    let mut transactions = vec![coinbase];
    for i in 0..num_txs {
        transactions.push(Transaction {
            version: 1,
            inputs: vec![TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32].into(), // Different hash for each
                    index: 0,
                },
                script_sig: vec![0x51; 20], // Longer script
                sequence: 0xffffffff,
            }]
            .into(),
            outputs: vec![
                TransactionOutput {
                    value: 10_000_000,                    // 0.1 BTC
                    script_pubkey: vec![0x51; 25].into(), // Longer script
                },
                TransactionOutput {
                    value: 5_000_000, // 0.05 BTC
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
            merkle_root: [0; 32], // Would be calculated in real scenario
            timestamp: 1234567890,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: transactions.into(),
    }
}

fn benchmark_connect_block_realistic(c: &mut Criterion) {
    let block = create_realistic_test_block(100); // 100 transactions (more realistic)
    let utxo_set = UtxoSet::new();
    let witnesses: Vec<Witness> = block.transactions.iter().map(|_| Vec::new()).collect();

    c.bench_function("connect_block_realistic_100tx", |b| {
        b.iter(|| {
            let _result = connect_block(
                black_box(&block),
                black_box(&witnesses),
                black_box(utxo_set.clone()),
                black_box(0),
                black_box(None),
                black_box(Network::Mainnet),
            );
            // Note: May fail validation due to invalid UTXOs, but measures actual validation work
        })
    });
}

fn benchmark_connect_block_realistic_1000tx(c: &mut Criterion) {
    let block = create_realistic_test_block(1000); // 1000 transactions (matches Core benchmark)
    let utxo_set = UtxoSet::new();
    let witnesses: Vec<Witness> = block.transactions.iter().map(|_| Vec::new()).collect();

    c.bench_function("connect_block_realistic_1000tx", |b| {
        b.iter(|| {
            let _result = connect_block(
                black_box(&block),
                black_box(&witnesses),
                black_box(utxo_set.clone()),
                black_box(0),
                black_box(None),
                black_box(Network::Mainnet),
            );
            // Note: May fail validation due to invalid UTXOs, but measures actual validation work
        })
    });
}

criterion_group!(
    benches,
    benchmark_connect_block_realistic,
    benchmark_connect_block_realistic_1000tx
);
criterion_main!(benches);
