use consensus_proof::segwit::Witness;
use consensus_proof::segwit::{
    calculate_block_weight, calculate_transaction_weight, is_segwit_transaction,
};
use consensus_proof::{
    Block, BlockHeader, OutPoint, Transaction, TransactionInput, TransactionOutput,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn create_segwit_transaction() -> (Transaction, Witness) {
    let tx = Transaction {
        version: 1,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0,
            },
            script_sig: vec![], // Empty script_sig for SegWit
            sequence: 0xffffffff,
        }],
        outputs: vec![TransactionOutput {
            value: 5000000000,
            script_pubkey: vec![
                0x00, 0x14, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
                0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13,
            ], // P2WPKH
        }],
        lock_time: 0,
    };

    let witness: Witness = vec![
        vec![0x30, 0x44, 0x02, 0x20, 0x01, 0x02], // Signature
        vec![0x03, 0x04, 0x05],                   // Public key
    ];

    (tx, witness)
}

fn benchmark_is_segwit_transaction(c: &mut Criterion) {
    let tx = create_test_transaction();

    c.bench_function("is_segwit_transaction", |b| {
        b.iter(|| black_box(is_segwit_transaction(black_box(&tx))))
    });
}

fn benchmark_calculate_transaction_weight(c: &mut Criterion) {
    let tx = create_test_transaction();

    c.bench_function("calculate_transaction_weight_no_witness", |b| {
        b.iter(|| {
            black_box(calculate_transaction_weight(
                black_box(&tx),
                black_box(None),
            ))
        })
    });

    let (segwit_tx, witness) = create_segwit_transaction();

    c.bench_function("calculate_transaction_weight_with_witness", |b| {
        b.iter(|| {
            black_box(calculate_transaction_weight(
                black_box(&segwit_tx),
                black_box(Some(&witness)),
            ))
        })
    });
}

fn benchmark_calculate_block_weight(c: &mut Criterion) {
    let tx = create_test_transaction();
    let block = Block {
        header: BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [0u8; 32],
            timestamp: 0,
            bits: 0x1d00ffff,
            nonce: 0,
        },
        transactions: vec![tx],
    };

    let witnesses: Vec<Witness> = vec![vec![]];

    c.bench_function("calculate_block_weight", |b| {
        b.iter(|| {
            black_box(calculate_block_weight(
                black_box(&block),
                black_box(&witnesses),
            ))
        })
    });
}

criterion_group!(
    benches,
    benchmark_is_segwit_transaction,
    benchmark_calculate_transaction_weight,
    benchmark_calculate_block_weight
);
criterion_main!(benches);
