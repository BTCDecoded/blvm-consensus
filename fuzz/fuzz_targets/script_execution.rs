#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::ByteString;
use consensus_proof::script::eval_script;

fuzz_target!(|data: &[u8]| {
    // Enhanced script execution fuzzing
    // Tests various script patterns and edge cases
    
    if data.is_empty() {
        return;
    }
    
    // Limit script size to avoid excessive memory usage
    let max_script_size = 10000;
    let script: ByteString = data.iter().take(max_script_size).copied().collect();
    
    // Test with different script flags combinations
    // This covers standard scripts, P2SH, SegWit scenarios
    // Add more flag combinations based on fuzzed data
    let base_flags = if data.len() > 1 {
        (data[0] as u32) << 8 | (data.get(1).copied().unwrap_or(0) as u32)
    } else {
        0
    };
    
    let flag_combinations = [
        0u32,                          // Standard
        0x01,                          // P2SH
        0x01 | 0x04,                   // P2SH + SCRIPT_VERIFY_DERSIG
        0x01 | 0x20,                   // P2SH + SCRIPT_VERIFY_NULLDUMMY
        0x02,                          // STRICTENC
        0x04,                          // DERSIG
        0x08,                          // LOW_S
        0x10,                          // NULLDUMMY
        0x20,                          // CHECKLOCKTIMEVERIFY
        0x40,                          // CHECKSEQUENCEVERIFY
        base_flags & 0xFF,             // Fuzzed flags (masked)
    ];
    
    for &flags in &flag_combinations {
        let mut stack = Vec::new();
        // Should never panic - test robustness
        let _result = eval_script(&script, &mut stack, flags);
        
        // Also test with non-empty initial stack state
        // Simulate script_sig + script_pubkey scenario
        if !script.is_empty() && data.len() > 1 {
            let split_point = (script.len() / 2).min(100);
            let script_sig = script[..split_point].to_vec();
            let script_pubkey = script[split_point..].to_vec();
            
            // Test eval_script with script_sig results on stack
            let mut stack_with_sig = vec![script_sig];
            let _result2 = eval_script(&script_pubkey, &mut stack_with_sig, flags);
        }
    }
});

