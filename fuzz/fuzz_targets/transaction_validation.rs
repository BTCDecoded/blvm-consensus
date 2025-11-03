#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::{Transaction, TransactionInput, TransactionOutput, OutPoint};
use consensus_proof::transaction::check_transaction;

fuzz_target!(|data: &[u8]| {
    // Enhanced transaction validation fuzzing
    // Tests robustness with more realistic transaction structures
    
    if data.len() < 4 {
        return;
    }
    
    // Parse version (first 4 bytes)
    let version = if data.len() >= 4 {
        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as u64
    } else {
        1
    };
    
    // Parse input count (next 1-9 bytes for varint)
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
            0 // Skip if incomplete
        }
    } else {
        0
    };
    
    // Limit input count for tractability
    let input_count = input_count.min(50);
    
    // Parse inputs (OutPoint + script_sig + sequence)
    let mut inputs = Vec::new();
    for _ in 0..input_count {
        if offset + 36 + 4 + 4 > data.len() {
            break; // Not enough data
        }
        
        // OutPoint (32 bytes hash + 4 bytes index)
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&data[offset..offset+32]);
        offset += 32;
        
        let index = u32::from_le_bytes([
            data[offset], data[offset+1], data[offset+2], data[offset+3]
        ]) as u64;
        offset += 4;
        
        // Script length (varint)
        let script_len = if offset < data.len() && data[offset] < 0xfd {
            let len = data[offset] as usize;
            offset += 1;
            len.min(520) // MAX_SCRIPT_SIZE
        } else {
            break;
        };
        
        // Script
        if offset + script_len > data.len() {
            break;
        }
        let script_sig = data[offset..offset+script_len].to_vec();
        offset += script_len;
        
        // Sequence (4 bytes)
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
    
    // Parse output count (similar to input count)
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
    
    let output_count = output_count.min(50);
    
    // Parse outputs (value + script_pubkey)
    let mut outputs = Vec::new();
    for _ in 0..output_count {
        if offset + 8 > data.len() {
            break;
        }
        
        // Value (8 bytes)
        let value = i64::from_le_bytes([
            data[offset], data[offset+1], data[offset+2], data[offset+3],
            data[offset+4], data[offset+5], data[offset+6], data[offset+7],
        ]);
        offset += 8;
        
        // Script length
        let script_len = if offset < data.len() && data[offset] < 0xfd {
            let len = data[offset] as usize;
            offset += 1;
            len.min(520)
        } else {
            break;
        };
        
        // Script
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
    
    // Parse lock_time (last 4 bytes if available)
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
    
    // Should never panic - test robustness
    let _result = check_transaction(&tx);
});

