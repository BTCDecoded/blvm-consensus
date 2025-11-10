use bllvm_consensus::transaction::check_transaction;
use bllvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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
            value: 100000000,
            script_pubkey: vec![0x51, 0x87], // OP_1 OP_EQUAL
        }],
        lock_time: 0,
    }
}

fn benchmark_transaction_validation(c: &mut Criterion) {
    let tx = create_test_transaction();

    c.bench_function("check_transaction", |b| {
        b.iter(|| black_box(check_transaction(black_box(&tx))))
    });
}

fn benchmark_transaction_validation_complex(c: &mut Criterion) {
    // Multi-input, multi-output transaction
    let tx = Transaction {
        version: 1,
        inputs: (0..10)
            .map(|i| TransactionInput {
                prevout: OutPoint {
                    hash: [i as u8; 32],
                    index: i,
                },
                script_sig: vec![0x51; 20], // Longer script
                sequence: 0xffffffff,
            })
            .collect(),
        outputs: (0..10)
            .map(|i| TransactionOutput {
                value: 10000000 + i as i64,
                script_pubkey: vec![0x51; 25],
            })
            .collect(),
        lock_time: 0,
    };

    c.bench_function("check_transaction_complex", |b| {
        b.iter(|| black_box(check_transaction(black_box(&tx))))
    });
}

criterion_group!(
    benches,
    benchmark_transaction_validation,
    benchmark_transaction_validation_complex
);
criterion_main!(benches);
