#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    #[cfg(feature = "utxo-commitments")]
    {
        use consensus_proof::{BlockHeader, Hash, Natural};
        use consensus_proof::utxo_commitments::verification::{verify_supply, verify_header_chain, verify_commitment_block_hash};
        use consensus_proof::utxo_commitments::data_structures::UtxoCommitment;
        
        // Fuzz UTXO commitment verification: merkle tree construction, commitment verification
        
        if data.len() < 88 {
            return; // Need at least block header
        }
        
        // Create block header from fuzzed data
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
        
        // Test header chain verification
        // Create a small chain from fuzzed data
        let mut headers = vec![header.clone()];
        if data.len() >= 176 {
            // Try to create a second header
            let header2 = BlockHeader {
                version: header.version,
                prev_block_hash: {
                    // Use hash of first header (simplified)
                    let mut h = [0u8; 32];
                    h[0] = 1;
                    h
                },
                merkle_root: data.get(88..120)
                    .unwrap_or(&[0; 32])
                    .try_into()
                    .unwrap_or([0; 32]),
                timestamp: header.timestamp + 1,
                bits: header.bits,
                nonce: header.nonce,
            };
            headers.push(header2);
        }
        
        // Test header chain verification - should never panic
        let _chain_result = verify_header_chain(&headers);
        
        // Test commitment block hash verification
        let commitment = UtxoCommitment {
            block_height: 100,
            block_hash: [0u8; 32], // Will likely fail verification, but should handle gracefully
            total_supply: 50_0000_0000 * 100,
            merkle_root: [0u8; 32],
            commitment_hash: [0u8; 32],
        };
        
        let _hash_result = verify_commitment_block_hash(&commitment, &header);
        
        // Test supply verification
        let _supply_result = verify_supply(&commitment);
    }
    // When feature is disabled, just return early
});
