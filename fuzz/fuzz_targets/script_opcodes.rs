#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::ByteString;
use consensus_proof::script::eval_script;

fuzz_target!(|data: &[u8]| {
    // Fuzz individual opcode execution: edge cases, stack states, flags
    
    if data.len() < 2 {
        return;
    }
    
    // Extract opcode and flags from input
    let opcode = data[0];
    let flags = if data.len() >= 2 {
        u32::from_le_bytes([
            data.get(1).copied().unwrap_or(0),
            data.get(2).copied().unwrap_or(0),
            data.get(3).copied().unwrap_or(0),
            data.get(4).copied().unwrap_or(0),
        ])
    } else {
        0
    };
    
    // Build stack from remaining data
    let mut stack: Vec<ByteString> = Vec::new();
    let mut offset = 5;
    
    // Parse stack items (each item is length-prefixed)
    while offset < data.len() {
        if offset + 1 > data.len() {
            break;
        }
        
        let item_len = data[offset] as usize;
        offset += 1;
        
        // Limit item size to prevent excessive memory usage
        let item_len = item_len.min(520).min(data.len() - offset);
        
        if item_len == 0 {
            // Empty item
            stack.push(vec![]);
            continue;
        }
        
        if offset + item_len > data.len() {
            break;
        }
        
        let item = data[offset..offset + item_len].to_vec();
        stack.push(item);
        offset += item_len;
        
        // Limit stack size for tractability
        if stack.len() >= 100 {
            break;
        }
    }
    
    // Test script execution with various opcode patterns
    
    // Test 1: Execute script with single opcode
    let script = vec![opcode];
    let mut stack1 = stack.clone();
    let _result1 = eval_script(&script, &mut stack1, flags);
    // Should not panic regardless of opcode or stack state
    
    // Test 2: Execute script with empty stack
    let mut empty_stack = Vec::new();
    let _result2 = eval_script(&script, &mut empty_stack, flags);
    
    // Test 3: Execute with single-item stack
    if !stack.is_empty() {
        let mut single_stack = vec![stack[0].clone()];
        let _result3 = eval_script(&script, &mut single_stack, flags);
    }
    
    // Test 4: Execute with two-item stack (for binary operations)
    if stack.len() >= 2 {
        let mut binary_stack = vec![stack[0].clone(), stack[1].clone()];
        let _result4 = eval_script(&script, &mut binary_stack, flags);
    }
    
    // Test 5: Test all core opcodes explicitly
    let core_opcodes = [
        0x00, // OP_0
        0x51, // OP_1
        0x52, // OP_2
        0x60, // OP_16
        0x76, // OP_DUP
        0x87, // OP_EQUAL
        0x88, // OP_EQUALVERIFY
        0xa9, // OP_HASH160
        0xaa, // OP_HASH256
        0xac, // OP_CHECKSIG
        0xad, // OP_CHECKSIGVERIFY
        0x6a, // OP_RETURN
        0x69, // OP_VERIFY
    ];
    
    for &test_opcode in &core_opcodes {
        let test_script = vec![test_opcode];
        // Test with empty stack
        let mut test_stack = Vec::new();
        let _result = eval_script(&test_script, &mut test_stack, flags);
        
        // Test with non-empty stack
        if !stack.is_empty() {
            let mut test_stack = stack.clone();
            let _result = eval_script(&test_script, &mut test_stack, flags);
        }
    }
    
    // Test 6: Test with invalid/unknown opcodes
    // Should handle gracefully (return error, not panic)
    let invalid_opcodes = [0xff, 0xfe, 0xfd, 0xfc];
    for &invalid_opcode in &invalid_opcodes {
        let invalid_script = vec![invalid_opcode];
        let mut test_stack = stack.clone();
        let _result = eval_script(&invalid_script, &mut test_stack, flags);
        // Should return error for unknown opcodes
    }
    
    // Test 7: Test with various flag combinations
    let flag_combinations = [
        0u32,                          // No flags
        0x01,                          // P2SH
        0x02,                          // STRICTENC
        0x04,                          // DERSIG
        0x08,                          // LOW_S
        0x10,                          // NULLDUMMY
        0x20,                          // CHECKLOCKTIMEVERIFY
        0x40,                          // CHECKSEQUENCEVERIFY
        0x800,                         // SCRIPT_VERIFY_WITNESS
        0x2000,                        // SCRIPT_VERIFY_TAPROOT
        flags,                         // Fuzzed flags
    ];
    
    for &test_flags in &flag_combinations {
        if !stack.is_empty() {
            let mut test_stack = stack.clone();
            let _result = eval_script(&script, &mut test_stack, test_flags);
        }
    }
    
    // Test 8: Test stack size limits
    // Create large stack to test overflow handling
    let mut large_stack: Vec<ByteString> = Vec::new();
    for i in 0..100 {
        large_stack.push(vec![i as u8]);
    }
    let _result = eval_script(&script, &mut large_stack, flags);
    // Should handle large stacks gracefully (may fail due to limits, but shouldn't panic)
});

