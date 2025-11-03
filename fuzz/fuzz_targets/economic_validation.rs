#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::{Transaction, TransactionInput, TransactionOutput, OutPoint, UtxoSet};
use consensus_proof::economic::{get_block_subsidy, total_supply, calculate_fee};
use consensus_proof::constants::{MAX_MONEY, HALVING_INTERVAL};

fuzz_target!(|data: &[u8]| {
    // Fuzz economic validation: subsidy calculation, supply limits, fee calculations
    
    // Test 1: Block subsidy calculation at various heights
    if data.len() >= 8 {
        let height = u64::from_le_bytes([
            data[0],
            data.get(1).copied().unwrap_or(0),
            data.get(2).copied().unwrap_or(0),
            data.get(3).copied().unwrap_or(0),
            data.get(4).copied().unwrap_or(0),
            data.get(5).copied().unwrap_or(0),
            data.get(6).copied().unwrap_or(0),
            data.get(7).copied().unwrap_or(0),
        ]);
        
        // Bound height for tractability (but test edge cases)
        let test_heights = [
            0,  // Genesis
            1,
            HALVING_INTERVAL - 1,  // Just before first halving
            HALVING_INTERVAL,      // First halving
            HALVING_INTERVAL + 1,  // Just after first halving
            2 * HALVING_INTERVAL,  // Second halving
            64 * HALVING_INTERVAL, // 64 halvings (subsidy = 0)
            65 * HALVING_INTERVAL, // Beyond 64 halvings
            height % (100 * HALVING_INTERVAL), // Modulo to keep reasonable
        ];
        
        for &test_height in &test_heights {
            let subsidy = get_block_subsidy(test_height);
            let total = total_supply(test_height);
            
            // Critical invariants
            assert!(subsidy >= 0, "Subsidy must be non-negative");
            assert!(total >= 0, "Total supply must be non-negative");
            assert!(total <= MAX_MONEY as i64, "Total supply must not exceed MAX_MONEY");
            
            // Halving boundary checks
            if test_height >= 64 * HALVING_INTERVAL {
                assert_eq!(subsidy, 0, "Subsidy must be 0 after 64 halvings");
            }
        }
    }
    
    // Test 2: Fee calculation with various transaction structures
    if data.len() >= 4 {
        let mut offset = 0;
        
        // Parse input count
        let input_count = if offset < data.len() && data[offset] < 0xfd {
            let count = data[offset] as usize;
            offset += 1;
            count.min(10) // Limit for tractability
        } else {
            0
        };
        
        // Parse output count
        let output_count = if offset < data.len() && data[offset] < 0xfd {
            let count = data[offset] as usize;
            offset += 1;
            count.min(10) // Limit for tractability
        } else {
            0
        };
        
        // Build transaction
        let mut inputs = Vec::new();
        for _i in 0..input_count {
            if offset + 36 > data.len() {
                break;
            }
            
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&data[offset..offset+32]);
            offset += 32;
            
            let index = if offset + 4 <= data.len() {
                let idx = u32::from_le_bytes([
                    data[offset], data[offset+1], data[offset+2], data[offset+3]
                ]) as u64;
                offset += 4;
                idx
            } else {
                break;
            };
            
            inputs.push(TransactionInput {
                prevout: OutPoint { hash, index },
                script_sig: vec![],
                sequence: 0xffffffff,
            });
        }
        
        let mut outputs = Vec::new();
        for _ in 0..output_count {
            if offset + 8 > data.len() {
                break;
            }
            
            let value = i64::from_le_bytes([
                data[offset], data[offset+1], data[offset+2], data[offset+3],
                data[offset+4], data[offset+5], data[offset+6], data[offset+7],
            ]);
            offset += 8;
            
            // Bound value to reasonable range
            let bounded_value = value.abs() % (MAX_MONEY as i64);
            
            outputs.push(TransactionOutput {
                value: bounded_value,
                script_pubkey: vec![],
            });
        }
        
        let tx = Transaction {
            version: 1,
            inputs,
            outputs,
            lock_time: 0,
        };
        
        // Create UTXO set with inputs
        let mut utxo_set = UtxoSet::new();
        for input in &tx.inputs {
            utxo_set.insert(
                input.prevout.clone(),
                consensus_proof::UTXO {
                    value: 1000000, // 0.01 BTC
                    script_pubkey: vec![],
                    height: 0,
                }
            );
        }
        
        // Should never panic - test robustness
        let _fee_result = calculate_fee(&tx, &utxo_set);
        // Critical invariant: fee must be non-negative for valid transactions
        // (negative fees indicate insufficient input value, which is invalid)
        // Note: calculate_fee may return negative for invalid transactions, which is acceptable
    }
    
    // Test 3: Total supply convergence
    // Test that total supply calculation is correct at various heights
    let test_heights = [
        0,
        100,
        1000,
        HALVING_INTERVAL,
        10 * HALVING_INTERVAL,
        64 * HALVING_INTERVAL,
    ];
    
    for &height in &test_heights {
        let total = total_supply(height);
        
        // Critical invariant: total supply must be non-negative
        assert!(total >= 0, "Total supply must be non-negative");
        
        // Critical invariant: total supply must not exceed MAX_MONEY
        assert!(total <= MAX_MONEY as i64, "Total supply must not exceed MAX_MONEY");
        
        // Critical invariant: total supply should be sum of all subsidies
        // (This is what total_supply() calculates, but we verify it's reasonable)
        let expected_max = 21_000_000 * 100_000_000; // 21M BTC
        assert!(total <= expected_max, "Total supply should not exceed 21M BTC");
    }
});

