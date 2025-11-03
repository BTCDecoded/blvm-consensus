#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::{Transaction, TransactionInput, TransactionOutput, OutPoint, UtxoSet};
use consensus_proof::mempool::{accept_to_memory_pool, replacement_checks, is_standard_tx, Mempool};
use std::collections::HashSet;

fuzz_target!(|data: &[u8]| {
    // Fuzz mempool operations: acceptance, RBF, standardness checks
    
    if data.len() < 4 {
        return;
    }
    
    // Parse transaction from fuzzed data (similar to transaction_validation fuzzer)
    let version = if data.len() >= 4 {
        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64
    } else {
        1
    };
    
    let mut offset = 4;
    let input_count = if offset < data.len() {
        let count_byte = data[offset];
        offset += 1;
        if count_byte < 0xfd {
            count_byte as usize
        } else if count_byte == 0xfd && offset + 2 <= data.len() {
            offset += 2;
            u16::from_le_bytes([data[offset-2], data[offset-1]]) as usize
        } else {
            0
        }
    } else {
        0
    };
    
    let input_count = input_count.min(10); // Limit for tractability
    
    let mut inputs = Vec::new();
    for _ in 0..input_count {
        if offset + 36 + 4 + 4 > data.len() {
            break;
        }
        
        let mut hash = [0u8; 32];
        if offset + 32 <= data.len() {
            hash.copy_from_slice(&data[offset..offset+32]);
            offset += 32;
        } else {
            break;
        }
        
        let index = if offset + 4 <= data.len() {
            let idx = u32::from_le_bytes([
                data[offset], data[offset+1], data[offset+2], data[offset+3]
            ]) as u64;
            offset += 4;
            idx
        } else {
            break;
        };
        
        let script_len = if offset < data.len() && data[offset] < 0xfd {
            let len = data[offset] as usize;
            offset += 1;
            len.min(520)
        } else {
            break;
        };
        
        if offset + script_len > data.len() {
            break;
        }
        let script_sig = data[offset..offset+script_len].to_vec();
        offset += script_len;
        
        if offset + 4 > data.len() {
            break;
        }
        let sequence = u32::from_le_bytes([
            data[offset], data[offset+1], data[offset+2], data[offset+3]
        ]) as u64;
        offset += 4;
        
        inputs.push(TransactionInput {
            prevout: OutPoint { hash, index },
            script_sig,
            sequence,
        });
    }
    
    let output_count = if offset < data.len() {
        let count_byte = data[offset];
        offset += 1;
        if count_byte < 0xfd {
            count_byte as usize
        } else if count_byte == 0xfd && offset + 2 <= data.len() {
            offset += 2;
            u16::from_le_bytes([data[offset-2], data[offset-1]]) as usize
        } else {
            0
        }
    } else {
        0
    };
    
    let output_count = output_count.min(10);
    
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
        
        let script_len = if offset < data.len() && data[offset] < 0xfd {
            let len = data[offset] as usize;
            offset += 1;
            len.min(520)
        } else {
            break;
        };
        
        if offset + script_len > data.len() {
            break;
        }
        let script_pubkey = data[offset..offset+script_len].to_vec();
        offset += script_len;
        
        outputs.push(TransactionOutput {
            value,
            script_pubkey,
        });
    }
    
    let lock_time = if offset + 4 <= data.len() {
        u32::from_le_bytes([
            data[offset], data[offset+1], data[offset+2], data[offset+3]
        ]) as u64
    } else {
        0
    };
    
    let tx = Transaction {
        version,
        inputs,
        outputs,
        lock_time,
    };
    
    // Test mempool operations - should never panic
    let utxo_set = UtxoSet::new();
    let mempool: Mempool = HashSet::new();
    
    // Test accept_to_memory_pool
    let _accept_result = accept_to_memory_pool(&tx, &utxo_set, &mempool, 0);
    
    // Test is_standard_tx
    let _standard_result = is_standard_tx(&tx);
    
    // Check RBF signaling manually (signals_rbf is private)
    // RBF is signaled when sequence < 0xffffffff - 1
    let _rbf_signal = tx.inputs.iter().any(|input| input.sequence < 0xffffffff - 1);
    
    // Test replacement_checks with two transactions
    // Create a second transaction with modified data
    if data.len() > 100 {
        let mut tx2 = tx.clone();
        // Modify sequence to signal RBF
        if !tx2.inputs.is_empty() {
            tx2.inputs[0].sequence = 0xfffffffe;
        }
        
        // Modify first tx to also signal RBF
        let mut tx1 = tx.clone();
        if !tx1.inputs.is_empty() {
            tx1.inputs[0].sequence = 0xfffffffe;
        }
        
        let _replacement_result = replacement_checks(&tx2, &tx1, &mempool);
    }
});

