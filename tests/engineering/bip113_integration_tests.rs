//! BIP113 (Median Time-Past) Integration Tests
//! 
//! Tests for median time-past calculation and integration with CLTV (BIP65)
//! timestamp validation.

use bllvm_consensus::*;
use bllvm_consensus::bip113::get_median_time_past;
use super::bip_test_helpers::*;

#[test]
fn test_median_time_past_single_block() {
    // Median time-past with single block
    let timestamps = vec![1000];
    let median = get_test_median_time_past(timestamps);
    
    assert_eq!(median, 1000);
}

#[test]
fn test_median_time_past_three_blocks() {
    // Median of 3 blocks
    let timestamps = vec![1000, 2000, 3000];
    let median = get_test_median_time_past(timestamps);
    
    // Median of [1000, 2000, 3000] = 2000
    assert_eq!(median, 2000);
}

#[test]
fn test_median_time_past_eleven_blocks() {
    // Exactly 11 blocks (BIP113 specification)
    let timestamps: Vec<u64> = (1..=11).map(|i| i * 100).collect();
    let median = get_test_median_time_past(timestamps);
    
    // Median of [100, 200, ..., 1100] = 600
    assert_eq!(median, 600);
}

#[test]
fn test_median_time_past_more_than_eleven() {
    // More than 11 blocks - should use last 11
    let timestamps: Vec<u64> = (1..=20).map(|i| i * 100).collect();
    let median = get_test_median_time_past(timestamps);
    
    // Should use last 11: [1000, 1100, ..., 2000]
    // Median = 1500
    assert_eq!(median, 1500);
}

#[test]
fn test_median_time_past_unsorted() {
    // Median calculation should handle unsorted input
    let timestamps = vec![3000, 1000, 2000];
    let median = get_test_median_time_past(timestamps);
    
    // Should sort internally: [1000, 2000, 3000], median = 2000
    assert_eq!(median, 2000);
}

#[test]
fn test_median_time_past_edge_case_two_blocks() {
    // Even number of blocks (median = average of two middle values)
    let timestamps = vec![1000, 2000];
    let median = get_test_median_time_past(timestamps);
    
    // Median of [1000, 2000] = (1000 + 2000) / 2 = 1500
    assert_eq!(median, 1500);
}

#[test]
fn test_median_time_past_cltv_integration_example() {
    // Example: CLTV timestamp validation with median time-past
    // Transaction locktime: 1609459200 (2021-01-01)
    // Required locktime: 1577836800 (2020-01-01)
    // Median time-past should be >= tx.lock_time for validation
    
    // Create headers with timestamps leading up to transaction
    let timestamps = vec![
        1577836800, // 2020-01-01
        1577840400, // 2020-01-01 01:00
        1577844000, // 2020-01-01 02:00
        1577847600, // 2020-01-01 03:00
        1577851200, // 2020-01-01 04:00
        1577854800, // 2020-01-01 05:00
        1577858400, // 2020-01-01 06:00
        1577862000, // 2020-01-01 07:00
        1577865600, // 2020-01-01 08:00
        1577869200, // 2020-01-01 09:00
        1577872800, // 2020-01-01 10:00
    ];
    
    let median = get_test_median_time_past(timestamps);
    
    // Median should be around 1577854800 (middle of range)
    assert_eq!(median, 1577854800);
    
    // Transaction with locktime 1609459200 (2021) would require median >= 1609459200
    // But median is only 1577854800, so validation would fail
    let tx_locktime: u32 = 1609459200;
    assert!(median < tx_locktime as u64);
}

#[test]
fn test_median_time_past_duplicate_timestamps() {
    // Test with duplicate timestamps
    let timestamps = vec![1000, 1000, 2000, 2000, 3000];
    let median = get_test_median_time_past(timestamps);
    
    // Sorted: [1000, 1000, 2000, 2000, 3000]
    // Median = 2000 (middle value)
    assert_eq!(median, 2000);
}

#[test]
fn test_median_time_past_empty() {
    // Empty headers should return 0
    let timestamps = vec![];
    let median = get_test_median_time_past(timestamps);
    
    assert_eq!(median, 0);
}

#[test]
fn test_median_time_past_cltv_validation_logic() {
    // Demonstrate how median time-past would be used in CLTV validation
    // For timestamp-based locktime: median time-past must be >= transaction locktime
    
    // Create 11 blocks with timestamps
    let base_time: u64 = 1609459200; // 2021-01-01
    let timestamps: Vec<u64> = (0..11).map(|i| base_time + i * 600).collect(); // 10-minute intervals
    let median = get_test_median_time_past(timestamps);
    
    // Transaction locktime requirement
    let required_locktime: u32 = (base_time + 3000) as u32; // base_time + 50 minutes
    
    // For CLTV validation:
    // 1. Check if tx.lock_time >= required_locktime (current implementation does this)
    // 2. Check if median_time_past >= tx.lock_time (would need block context)
    // 
    // Current test: median (base_time + 5*600) should be < required_locktime (base_time + 3000)
    assert!(median < required_locktime as u64);
}

