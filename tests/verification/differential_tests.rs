//! Differential testing: Compare against multiple reference implementations
//!
//! Ensures our optimized implementation matches sha2 crate, Bitcoin Core patterns,
//! and NIST test vectors.

use bllvm_consensus::crypto::OptimizedSha256;
use sha2::{Digest, Sha256};

#[test]
fn compare_against_sha2_crate() {
    let test_vectors = vec![
        vec![],
        vec![0u8; 32],
        vec![0u8; 64],
        vec![0xffu8; 128],
        b"hello world".to_vec(),
        b"The quick brown fox jumps over the lazy dog".to_vec(),
        b"Bitcoin".to_vec(),
        b"".to_vec(),
        (0..256).map(|i| i as u8).collect(),
    ];

    for input in test_vectors {
        let reference = Sha256::digest(&input);
        let ours = OptimizedSha256::new().hash(&input);
        assert_eq!(
            &reference[..],
            &ours[..],
            "Failed for input length: {}",
            input.len()
        );
    }
}

#[test]
fn compare_double_sha256_against_sha2_crate() {
    let test_vectors = vec![
        vec![],
        vec![0u8; 32],
        vec![0u8; 64],
        b"hello world".to_vec(),
        b"Bitcoin transaction".to_vec(),
    ];

    for input in test_vectors {
        let reference = Sha256::digest(&Sha256::digest(&input));
        let ours = OptimizedSha256::new().hash256(&input);
        assert_eq!(
            &reference[..],
            &ours[..],
            "Double SHA256 failed for input length: {}",
            input.len()
        );
    }
}

#[test]
fn nist_test_vectors() {
    // NIST SHA256 test vectors (first few)
    let vectors = vec![
        (
            b"abc".to_vec(),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
            ],
        ),
        (
            b"".to_vec(),
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
            ],
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".to_vec(),
            [
                0x24, 0x8d, 0x6a, 0x61, 0xd2, 0x06, 0x38, 0xb8, 0xe5, 0xc0, 0x26, 0x93, 0x0c, 0x3e, 0x60, 0x39,
                0xa3, 0x3c, 0xe4, 0x59, 0x64, 0xff, 0x21, 0x67, 0xf6, 0xec, 0xed, 0xd4, 0x19, 0xdb, 0x06, 0xc1,
            ],
        ),
    ];

    for (input, expected) in vectors {
        let result = OptimizedSha256::new().hash(&input);
        assert_eq!(
            &result[..],
            &expected[..],
            "NIST vector failed for input: {:?}",
            String::from_utf8_lossy(&input)
        );
    }
}

#[test]
fn bitcoin_specific_vectors() {
    // Bitcoin-specific test: double SHA256 of empty string
    let empty = vec![];
    let hash1 = OptimizedSha256::new().hash(&empty);
    let hash2 = OptimizedSha256::new().hash(&hash1);
    
    // Known value: SHA256(SHA256("")) = 0x5df6e0e2...
    let expected: [u8; 32] = [
        0x5d, 0xf6, 0xe0, 0xe2, 0x76, 0x13, 0x59, 0xf3, 0x73, 0x9a, 0x1c, 0x6f, 0x87, 0x40, 0x64, 0x0a,
        0xf1, 0x2e, 0xc7, 0xc3, 0x72, 0x4a, 0x5c, 0x2c, 0xa5, 0xf3, 0x0f, 0x26, 0x60, 0x87, 0x7e, 0x6b,
    ];
    
    assert_eq!(&hash2[..], &expected[..]);
    
    // Also test hash256 convenience function
    let hash256_result = OptimizedSha256::new().hash256(&empty);
    assert_eq!(&hash256_result[..], &expected[..]);
}

