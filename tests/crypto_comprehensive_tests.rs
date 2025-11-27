//! Crypto module comprehensive tests
//!
//! Additional tests for cryptographic utilities.

#[cfg(target_arch = "x86_64")]
use bllvm_consensus::crypto::hash_compare::hash_eq;
use bllvm_consensus::crypto::int_ops::{safe_add, safe_sub};
#[cfg(not(target_arch = "x86_64"))]
use bllvm_consensus::types::Hash;

#[cfg(not(target_arch = "x86_64"))]
fn hash_eq(hash1: &Hash, hash2: &Hash) -> bool {
    hash1 == hash2
}

#[test]
fn test_safe_add_no_overflow() {
    let result = safe_add(100, 200);
    assert_eq!(result.unwrap(), 300);
}

#[test]
fn test_safe_add_overflow() {
    let result = safe_add(i64::MAX, 1);
    assert!(result.is_err());
}

#[test]
fn test_safe_add_underflow() {
    let result = safe_add(i64::MIN, -1);
    assert!(result.is_err());
}

#[test]
fn test_safe_add_boundary() {
    // Test at boundary values
    let result1 = safe_add(i64::MAX - 1, 1);
    assert_eq!(result1.unwrap(), i64::MAX);

    let result2 = safe_add(i64::MIN + 1, -1);
    assert_eq!(result2.unwrap(), i64::MIN);
}

#[test]
fn test_safe_sub_no_underflow() {
    let result = safe_sub(200, 100);
    assert_eq!(result.unwrap(), 100);
}

#[test]
fn test_safe_sub_underflow() {
    let result = safe_sub(i64::MIN, 1);
    assert!(result.is_err());
}

#[test]
fn test_safe_sub_overflow() {
    let result = safe_sub(i64::MAX, -1);
    assert!(result.is_err());
}

#[test]
fn test_safe_sub_boundary() {
    // Test at boundary values
    let result1 = safe_sub(i64::MIN + 1, 1);
    assert_eq!(result1.unwrap(), i64::MIN);

    let result2 = safe_sub(i64::MAX - 1, -1);
    assert_eq!(result2.unwrap(), i64::MAX);
}

#[test]
fn test_hash_eq_equal() {
    let hash1 = [0x42u8; 32];
    let hash2 = [0x42u8; 32];

    assert!(hash_eq(&hash1, &hash2));
}

#[test]
fn test_hash_eq_different() {
    let hash1 = [0x42u8; 32];
    let mut hash2 = [0x42u8; 32];
    hash2[0] = 0x43;

    assert!(!hash_eq(&hash1, &hash2));
}

#[test]
fn test_hash_eq_all_different() {
    let hash1 = [0x00u8; 32];
    let hash2 = [0xFFu8; 32];

    assert!(!hash_eq(&hash1, &hash2));
}

#[test]
fn test_safe_add_zero() {
    let result = safe_add(100, 0);
    assert_eq!(result.unwrap(), 100);
}

#[test]
fn test_safe_sub_zero() {
    let result = safe_sub(100, 0);
    assert_eq!(result.unwrap(), 100);
}

#[test]
fn test_safe_add_negative() {
    let result = safe_add(100, -50);
    assert_eq!(result.unwrap(), 50);
}

#[test]
fn test_safe_sub_negative() {
    let result = safe_sub(100, -50);
    assert_eq!(result.unwrap(), 150);
}
