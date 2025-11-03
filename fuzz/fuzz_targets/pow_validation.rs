#![no_main]
use libfuzzer_sys::fuzz_target;
use consensus_proof::BlockHeader;
use consensus_proof::pow::check_proof_of_work;

fuzz_target!(|data: &[u8]| {
    // Fuzz Proof of Work validation: difficulty adjustment, target expansion/compression, PoW verification
    
    // Test 1: Block header proof of work validation
    if data.len() >= 88 {
        let header = BlockHeader {
            version: i64::from_le_bytes([
                data.get(0).copied().unwrap_or(1) as u8,
                data.get(1).copied().unwrap_or(0),
                data.get(2).copied().unwrap_or(0),
                data.get(3).copied().unwrap_or(0),
                data.get(4).copied().unwrap_or(0),
                data.get(5).copied().unwrap_or(0),
                data.get(6).copied().unwrap_or(0),
                data.get(7).copied().unwrap_or(0),
            ]),
            prev_block_hash: data.get(8..40)
                .unwrap_or(&[0; 32])
                .try_into()
                .unwrap_or([0; 32]),
            merkle_root: data.get(40..72)
                .unwrap_or(&[0; 32])
                .try_into()
                .unwrap_or([0; 32]),
            timestamp: u64::from_le_bytes([
                data.get(72).copied().unwrap_or(0),
                data.get(73).copied().unwrap_or(0),
                data.get(74).copied().unwrap_or(0),
                data.get(75).copied().unwrap_or(0),
                data.get(76).copied().unwrap_or(0),
                data.get(77).copied().unwrap_or(0),
                data.get(78).copied().unwrap_or(0),
                data.get(79).copied().unwrap_or(0),
            ]),
            bits: u32::from_le_bytes([
                data.get(80).copied().unwrap_or(0),
                data.get(81).copied().unwrap_or(0),
                data.get(82).copied().unwrap_or(0),
                data.get(83).copied().unwrap_or(0),
            ]) as u64,
            nonce: u32::from_le_bytes([
                data.get(84).copied().unwrap_or(0),
                data.get(85).copied().unwrap_or(0),
                data.get(86).copied().unwrap_or(0),
                data.get(87).copied().unwrap_or(0),
            ]) as u64,
        };
        
        // Should never panic - test robustness
        let _result = check_proof_of_work(&header);
    }
    
    // Test 2: Block header with various bits values for PoW checking
    // (Target expansion/compression is internal to pow module)
    
    // Test 3: Difficulty adjustment with multiple headers
    if data.len() >= 176 {
        // Need at least 2 headers (88 bytes each)
        let mut headers = Vec::new();
        
        for i in 0..2 {
            let offset = i * 88;
            if offset + 88 <= data.len() {
                let header = BlockHeader {
                    version: i64::from_le_bytes([
                        data[offset + 0],
                        data.get(offset + 1).copied().unwrap_or(0),
                        data.get(offset + 2).copied().unwrap_or(0),
                        data.get(offset + 3).copied().unwrap_or(0),
                        data.get(offset + 4).copied().unwrap_or(0),
                        data.get(offset + 5).copied().unwrap_or(0),
                        data.get(offset + 6).copied().unwrap_or(0),
                        data.get(offset + 7).copied().unwrap_or(0),
                    ]),
                    prev_block_hash: data.get(offset + 8..offset + 40)
                        .unwrap_or(&[0; 32])
                        .try_into()
                        .unwrap_or([0; 32]),
                    merkle_root: data.get(offset + 40..offset + 72)
                        .unwrap_or(&[0; 32])
                        .try_into()
                        .unwrap_or([0; 32]),
                    timestamp: u64::from_le_bytes([
                        data.get(offset + 72).copied().unwrap_or(0),
                        data.get(offset + 73).copied().unwrap_or(0),
                        data.get(offset + 74).copied().unwrap_or(0),
                        data.get(offset + 75).copied().unwrap_or(0),
                        data.get(offset + 76).copied().unwrap_or(0),
                        data.get(offset + 77).copied().unwrap_or(0),
                        data.get(offset + 78).copied().unwrap_or(0),
                        data.get(offset + 79).copied().unwrap_or(0),
                    ]),
                    bits: u32::from_le_bytes([
                        data.get(offset + 80).copied().unwrap_or(0),
                        data.get(offset + 81).copied().unwrap_or(0),
                        data.get(offset + 82).copied().unwrap_or(0),
                        data.get(offset + 83).copied().unwrap_or(0),
                    ]) as u64,
                    nonce: u32::from_le_bytes([
                        data.get(offset + 84).copied().unwrap_or(0),
                        data.get(offset + 85).copied().unwrap_or(0),
                        data.get(offset + 86).copied().unwrap_or(0),
                        data.get(offset + 87).copied().unwrap_or(0),
                    ]) as u64,
                };
                headers.push(header);
            }
        }
        
        if headers.len() >= 2 {
            // Use pow module directly for get_next_work_required
            use consensus_proof::pow::get_next_work_required;
            let current_header = &headers[0];
            // Should never panic - test robustness
            let _result = get_next_work_required(current_header, &headers);
        }
    }
});

