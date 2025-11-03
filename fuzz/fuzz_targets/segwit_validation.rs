#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::{Transaction, Block, BlockHeader, TransactionInput, TransactionOutput, OutPoint};
use consensus_proof::segwit::{calculate_transaction_weight, is_segwit_transaction, calculate_block_weight};

fuzz_target!(|data: &[u8]| {
    // Fuzz SegWit validation: witness data, weight calculations, SegWit transaction parsing
    
    if data.len() < 4 {
        return;
    }
    
    // Create transaction (similar to transaction_validation fuzzer)
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
        } else {
            0
        }
    } else {
        0
    };
    
    let input_count = input_count.min(10);
    
    let mut inputs = Vec::new();
    for _ in 0..input_count {
        if offset + 40 > data.len() {
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
            len.min(100)
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
            len.min(100)
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
    
    // Test SegWit functions - should never panic
    let _is_segwit = is_segwit_transaction(&tx);
    
    // Create witness data from remaining fuzzed data
    // Witness is Vec<ByteString> where ByteString = Vec<u8>
    let witness_data: Option<Vec<Vec<u8>>> = if offset < data.len() && data.len() - offset > 10 {
        // Create 1-3 witness entries
        let witness_count = ((data[offset] % 3) + 1) as usize;
        let mut witnesses = Vec::new();
        let mut w_offset = offset + 1;
        
        for _ in 0..witness_count {
            if w_offset >= data.len() {
                break;
            }
            let witness_len = if w_offset < data.len() {
                (data[w_offset] as usize).min(100).min(data.len() - w_offset - 1)
            } else {
                0
            };
            if w_offset + witness_len <= data.len() && w_offset + witness_len + 1 <= data.len() {
                witnesses.push(data[w_offset+1..w_offset+1+witness_len].to_vec());
                w_offset += witness_len + 1;
            } else {
                break;
            }
        }
        if !witnesses.is_empty() {
            Some(witnesses)
        } else {
            None
        }
    } else {
        None
    };
    
    // Test transaction weight calculation
    let _weight = calculate_transaction_weight(&tx, witness_data.as_ref());
    
    // Test block weight if we have enough data
    if !tx.inputs.is_empty() && !tx.outputs.is_empty() {
        let block = Block {
            header: BlockHeader {
                version: 1,
                prev_block_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                timestamp: 0,
                bits: 0x1d00ffff,
                nonce: 0,
            },
            transactions: vec![tx.clone()],
        };
        
        let witnesses: Vec<Vec<Vec<u8>>> = witness_data.as_ref().map(|w| vec![w.clone()]).unwrap_or_default();
        let _block_weight = calculate_block_weight(&block, &witnesses);
    }
});

