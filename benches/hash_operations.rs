use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sha2::{Digest, Sha256};

fn benchmark_sha256(c: &mut Criterion) {
    let data = vec![0u8; 1024];

    c.bench_function("sha256_1kb", |b| {
        b.iter(|| {
            let mut hasher = Sha256::new();
            hasher.update(black_box(&data));
            black_box(hasher.finalize())
        })
    });
}

fn benchmark_double_sha256(c: &mut Criterion) {
    let data = vec![0u8; 1024];

    c.bench_function("double_sha256_1kb", |b| {
        b.iter(|| {
            let hash1 = Sha256::digest(black_box(&data));
            black_box(Sha256::digest(hash1))
        })
    });
}

// SHA-NI benchmarks for single-hash performance
#[cfg(target_arch = "x86_64")]
fn benchmark_sha_ni_single(c: &mut Criterion) {
    use bllvm_consensus::crypto::sha_ni;

    if !sha_ni::is_sha_ni_available() {
        return; // Skip on CPUs without SHA-NI
    }

    let data_32b = vec![0u8; 32];
    let data_64b = vec![0u8; 64];
    let data_1kb = vec![0u8; 1024];

    // 32-byte input (typical transaction hash)
    c.bench_function("sha_ni_32b", |b| {
        b.iter(|| black_box(sha_ni::sha256(black_box(&data_32b))))
    });

    c.bench_function("sha2_crate_32b", |b| {
        b.iter(|| {
            let hash = Sha256::digest(black_box(&data_32b));
            black_box(hash)
        })
    });

    // 64-byte input (typical block header)
    c.bench_function("sha_ni_64b", |b| {
        b.iter(|| black_box(sha_ni::sha256(black_box(&data_64b))))
    });

    c.bench_function("sha2_crate_64b", |b| {
        b.iter(|| {
            let hash = Sha256::digest(black_box(&data_64b));
            black_box(hash)
        })
    });

    // 1KB input
    c.bench_function("sha_ni_1kb", |b| {
        b.iter(|| black_box(sha_ni::sha256(black_box(&data_1kb))))
    });

    // Double SHA256 comparison
    c.bench_function("sha_ni_double_32b", |b| {
        b.iter(|| black_box(sha_ni::hash256(black_box(&data_32b))))
    });

    c.bench_function("sha2_crate_double_32b", |b| {
        b.iter(|| {
            let hash1 = Sha256::digest(black_box(&data_32b));
            let hash2 = Sha256::digest(hash1);
            black_box(hash2)
        })
    });
}

#[cfg(not(target_arch = "x86_64"))]
fn benchmark_sha_ni_single(_c: &mut Criterion) {
    // SHA-NI not available on non-x86_64
}

#[cfg(feature = "production")]
fn benchmark_batch_sha256(c: &mut Criterion) {
    use bllvm_consensus::optimizations::simd_vectorization;

    let batch_sizes = vec![4, 8, 16, 32, 64, 128];
    let data_1kb = vec![0u8; 1024];

    for size in batch_sizes {
        let inputs: Vec<Vec<u8>> = (0..size).map(|_| data_1kb.clone()).collect();
        let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

        c.bench_function(&format!("batch_sha256_{}", size), |b| {
            b.iter(|| black_box(simd_vectorization::batch_sha256(black_box(&input_refs))))
        });

        // Compare with sequential
        c.bench_function(&format!("sequential_sha256_{}", size), |b| {
            b.iter(|| {
                let results: Vec<[u8; 32]> = inputs
                    .iter()
                    .map(|data| {
                        let hash = Sha256::digest(data);
                        let mut result = [0u8; 32];
                        result.copy_from_slice(&hash);
                        result
                    })
                    .collect();
                black_box(results)
            })
        });
    }
}

#[cfg(feature = "production")]
fn benchmark_batch_double_sha256(c: &mut Criterion) {
    use bllvm_consensus::optimizations::simd_vectorization;

    let batch_sizes = vec![4, 8, 16, 32, 64, 128];
    let data_1kb = vec![0u8; 1024];

    for size in batch_sizes {
        let inputs: Vec<Vec<u8>> = (0..size).map(|_| data_1kb.clone()).collect();
        let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

        c.bench_function(&format!("batch_double_sha256_{}", size), |b| {
            b.iter(|| {
                black_box(simd_vectorization::batch_double_sha256(black_box(
                    &input_refs,
                )))
            })
        });

        // Compare with sequential
        c.bench_function(&format!("sequential_double_sha256_{}", size), |b| {
            b.iter(|| {
                let results: Vec<[u8; 32]> = inputs
                    .iter()
                    .map(|data| {
                        let hash1 = Sha256::digest(data);
                        let hash2 = Sha256::digest(hash1);
                        let mut result = [0u8; 32];
                        result.copy_from_slice(&hash2);
                        result
                    })
                    .collect();
                black_box(results)
            })
        });
    }
}

#[cfg(feature = "production")]
fn benchmark_batch_hash160(c: &mut Criterion) {
    use bllvm_consensus::optimizations::simd_vectorization;
    use ripemd::Ripemd160;

    let batch_sizes = vec![4, 8, 16, 32, 64];
    let data_1kb = vec![0u8; 1024];

    for size in batch_sizes {
        let inputs: Vec<Vec<u8>> = (0..size).map(|_| data_1kb.clone()).collect();
        let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

        c.bench_function(&format!("batch_hash160_{}", size), |b| {
            b.iter(|| black_box(simd_vectorization::batch_hash160(black_box(&input_refs))))
        });

        // Compare with sequential
        c.bench_function(&format!("sequential_hash160_{}", size), |b| {
            b.iter(|| {
                let results: Vec<[u8; 20]> = inputs
                    .iter()
                    .map(|data| {
                        let sha256_hash = Sha256::digest(data);
                        let ripemd160_hash = Ripemd160::digest(sha256_hash);
                        let mut result = [0u8; 20];
                        result.copy_from_slice(&ripemd160_hash);
                        result
                    })
                    .collect();
                black_box(results)
            })
        });
    }
}

#[cfg(feature = "production")]
fn benchmark_merkle_root_batching(c: &mut Criterion) {
    use bllvm_consensus::mining::calculate_merkle_root;
    use bllvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    let tx_counts = vec![10, 100, 1000];

    for tx_count in tx_counts {
        // Create test transactions
        let transactions: Vec<Transaction> = (0..tx_count)
            .map(|i| Transaction {
                version: 1u64,
                inputs: vec![TransactionInput {
                    prevout: OutPoint {
                        hash: [i as u8; 32],
                        index: 0u64,
                    },
                    script_sig: vec![0x51], // OP_1
                    sequence: 0xffffffffu64,
                }],
                outputs: vec![TransactionOutput {
                    value: 5000000000i64,
                    script_pubkey: vec![
                        0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a,
                        0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
                    ],
                }],
                lock_time: 0u64,
            })
            .collect();

        c.bench_function(&format!("merkle_root_{}tx", tx_count), |b| {
            b.iter(|| black_box(calculate_merkle_root(black_box(&transactions)).unwrap()))
        });
    }
}

#[cfg(feature = "production")]
fn benchmark_block_validation_tx_ids(c: &mut Criterion) {
    use bllvm_consensus::optimizations::simd_vectorization;
    use bllvm_consensus::serialization::transaction::serialize_transaction;
    use bllvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    let tx_counts = vec![10, 100];

    for tx_count in tx_counts {
        let transactions: Vec<Transaction> = (0..tx_count)
            .map(|i| Transaction {
                version: 1u64,
                inputs: vec![TransactionInput {
                    prevout: OutPoint {
                        hash: [i as u8; 32],
                        index: 0u64,
                    },
                    script_sig: vec![0x51],
                    sequence: 0xffffffffu64,
                }],
                outputs: vec![TransactionOutput {
                    value: 5000000000i64,
                    script_pubkey: vec![
                        0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a,
                        0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
                    ],
                }],
                lock_time: 0u64,
            })
            .collect();

        c.bench_function(&format!("batch_tx_id_{}tx", tx_count), |b| {
            b.iter(|| {
                let serialized: Vec<Vec<u8>> = transactions
                    .iter()
                    .map(|tx| serialize_transaction(tx))
                    .collect();
                let refs: Vec<&[u8]> = serialized.iter().map(|v| v.as_slice()).collect();
                black_box(simd_vectorization::batch_double_sha256(black_box(&refs)))
            })
        });
    }
}

#[cfg(feature = "production")]
fn benchmark_sighash_batching(c: &mut Criterion) {
    use bllvm_consensus::transaction_hash::{batch_compute_sighashes, SighashType};
    use bllvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    let input_counts = vec![2, 5, 10, 20];

    for input_count in input_counts {
        let tx = Transaction {
            version: 1u64,
            inputs: (0..input_count)
                .map(|i| TransactionInput {
                    prevout: OutPoint {
                        hash: [i as u8; 32],
                        index: 0u64,
                    },
                    script_sig: vec![0x51],
                    sequence: 0xffffffffu64,
                })
                .collect(),
            outputs: vec![TransactionOutput {
                value: 5000000000i64,
                script_pubkey: vec![
                    0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
                    0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
                ],
            }],
            lock_time: 0u64,
        };

        let prevouts: Vec<TransactionOutput> = (0..input_count)
            .map(|_| TransactionOutput {
                value: 10000000000i64,
                script_pubkey: vec![
                    0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
                    0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
                ],
            })
            .collect();

        c.bench_function(&format!("batch_sighash_{}inputs", input_count), |b| {
            b.iter(|| {
                black_box(
                    batch_compute_sighashes(black_box(&tx), black_box(&prevouts), SighashType::All)
                        .unwrap(),
                )
            })
        });
    }
}

#[cfg(feature = "production")]
fn benchmark_pow_batching(c: &mut Criterion) {
    use bllvm_consensus::pow::batch_check_proof_of_work;
    use bllvm_consensus::BlockHeader;

    let header_counts = vec![8, 16, 32, 64, 128];

    for count in header_counts {
        let headers: Vec<BlockHeader> = (0..count)
            .map(|i| BlockHeader {
                version: 1i64,
                prev_block_hash: [i as u8; 32],
                merkle_root: [0u8; 32],
                timestamp: 1234567890u64,
                bits: 0x1d00ffffu64,
                nonce: i as u64,
            })
            .collect();

        c.bench_function(&format!("batch_pow_{}headers", count), |b| {
            b.iter(|| black_box(batch_check_proof_of_work(black_box(&headers)).unwrap()))
        });
    }
}

// Note: SipHash benchmarking is in reference-node/benches/compact_blocks.rs
// since siphasher is only used in reference-node

#[cfg(feature = "production")]
fn benchmark_batch_ecdsa_verification(c: &mut Criterion) {
    use bllvm_consensus::script::batch_verify_signatures;
    use secp256k1::{ecdsa::Signature, Message, Secp256k1};

    // Create test verification tasks with fixed test data
    // Using dummy but valid-format data for benchmarking (verification will fail but format is valid)
    let signature_counts = vec![2, 5, 10, 20, 50];

    // Pre-create verification tasks with fixed format data
    // Note: These signatures won't verify correctly, but the format is valid for benchmarking
    // Using valid DER signature format (minimum 64 bytes for ECDSA)
    static PUBKEY1: [u8; 33] = [0x02; 33];
    static PUBKEY2: [u8; 33] = [0x03; 33];
    static SIG1: [u8; 70] = {
        let mut sig = [0u8; 70];
        sig[0] = 0x30; // DER sequence
        sig[1] = 0x44; // Length
        sig[2] = 0x02; // INTEGER
        sig[3] = 0x20; // R length (32 bytes)
        sig
    };
    static SIG2: [u8; 70] = {
        let mut sig = [0u8; 70];
        sig[0] = 0x30;
        sig[1] = 0x44;
        sig[2] = 0x02;
        sig[3] = 0x20;
        sig[4] = 0x01; // Different first byte
        sig
    };
    static HASH1: [u8; 32] = [0u8; 32];
    static HASH2: [u8; 32] = [1u8; 32];

    let base_verification_tasks: Vec<(&[u8], &[u8], [u8; 32])> = vec![
        (&PUBKEY1[..], &SIG1[..], HASH1),
        (&PUBKEY2[..], &SIG2[..], HASH2),
    ];

    for count in signature_counts {
        // Replicate base tasks to reach desired count
        let mut verification_tasks = Vec::with_capacity(count);
        for i in 0..count {
            let base_idx = i % base_verification_tasks.len();
            let (pk, sig, hash) = base_verification_tasks[base_idx];
            verification_tasks.push((pk, sig, hash));
        }

        c.bench_function(&format!("batch_verify_signatures_{}", count), |b| {
            b.iter(|| black_box(batch_verify_signatures(black_box(&verification_tasks))))
        });

        // Compare with sequential
        c.bench_function(&format!("sequential_verify_signatures_{}", count), |b| {
            b.iter(|| {
                let secp = Secp256k1::new();
                let results: Vec<bool> = verification_tasks
                    .iter()
                    .map(|(pubkey, sig, hash)| {
                        match (
                            secp256k1::PublicKey::from_slice(pubkey),
                            Signature::from_der(sig),
                            Message::from_digest_slice(hash),
                        ) {
                            (Ok(pk), Ok(sig), Ok(msg)) => {
                                secp.verify_ecdsa(&msg, &sig, &pk).is_ok()
                            }
                            _ => false,
                        }
                    })
                    .collect();
                black_box(results)
            })
        });
    }
}

#[cfg(feature = "production")]
fn benchmark_sighash_templates(c: &mut Criterion) {
    use bllvm_consensus::transaction_hash::{calculate_transaction_sighash, SighashType};
    use bllvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    // Create standard transaction (1 input, 1 output) - most common pattern
    let tx = Transaction {
        version: 1u64,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0u64,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffffu64,
        }],
        outputs: vec![TransactionOutput {
            value: 5000000000i64,
            script_pubkey: vec![
                0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde,
                0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
            ],
        }],
        lock_time: 0u64,
    };

    let prevouts = vec![TransactionOutput {
        value: 10000000000i64,
        script_pubkey: vec![
            0x76, 0xa9, 0x14, 0x89, 0xab, 0xcd, 0xef, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde,
            0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0x88, 0xac,
        ],
    }];

    c.bench_function("sighash_with_template_check", |b| {
        b.iter(|| {
            black_box(
                calculate_transaction_sighash(
                    black_box(&tx),
                    0,
                    black_box(&prevouts),
                    SighashType::All,
                )
                .unwrap(),
            )
        })
    });
}

#[cfg(feature = "production")]
fn benchmark_early_exit_transaction(c: &mut Criterion) {
    use bllvm_consensus::transaction::check_transaction;
    use bllvm_consensus::{OutPoint, Transaction, TransactionInput, TransactionOutput};

    // Test with obviously invalid transaction (empty inputs)
    let invalid_tx = Transaction {
        version: 1u64,
        inputs: vec![],
        outputs: vec![TransactionOutput {
            value: 5000000000i64,
            script_pubkey: vec![0x76, 0xa9, 0x14],
        }],
        lock_time: 0u64,
    };

    // Test with valid transaction
    let valid_tx = Transaction {
        version: 1u64,
        inputs: vec![TransactionInput {
            prevout: OutPoint {
                hash: [0u8; 32],
                index: 0u64,
            },
            script_sig: vec![0x51],
            sequence: 0xffffffffu64,
        }],
        outputs: vec![TransactionOutput {
            value: 5000000000i64,
            script_pubkey: vec![0x76, 0xa9, 0x14],
        }],
        lock_time: 0u64,
    };

    c.bench_function("check_transaction_invalid_fast_path", |b| {
        b.iter(|| black_box(check_transaction(black_box(&invalid_tx)).unwrap()))
    });

    c.bench_function("check_transaction_valid_full", |b| {
        b.iter(|| black_box(check_transaction(black_box(&valid_tx)).unwrap()))
    });
}

#[cfg(feature = "production")]
criterion_group!(
    benches,
    benchmark_sha256,
    benchmark_double_sha256,
    benchmark_batch_sha256,
    benchmark_batch_double_sha256,
    benchmark_batch_hash160,
    benchmark_merkle_root_batching,
    benchmark_block_validation_tx_ids,
    benchmark_sighash_batching,
    benchmark_pow_batching,
    benchmark_batch_ecdsa_verification,
    benchmark_sighash_templates,
    benchmark_early_exit_transaction
);

#[cfg(not(feature = "production"))]
criterion_group!(
    benches,
    benchmark_sha256,
    benchmark_double_sha256,
    benchmark_sha_ni_single
);

criterion_main!(benches);
