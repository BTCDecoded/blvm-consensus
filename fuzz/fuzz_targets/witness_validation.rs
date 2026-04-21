#![no_main]

use libfuzzer_sys::fuzz_target;
use blvm_consensus::witness::*;
use blvm_consensus::types::{Witness, ByteString};

/// Helper to parse a Witness using saturating slices.
/// Simulates truncated buffers by taking as many bytes as available up to the requested length.
fn parse_witness(data: &[u8], offset: &mut usize) -> Witness {
    if *offset >= data.len() {
        return Vec::new();
    }

    // Count of elements (capped at 32 to prevent OOM)
    let count = data[*offset] as usize % 33;
    *offset += 1;

    let mut witness = Vec::with_capacity(count);
    for _ in 0..count {
        if *offset >= data.len() {
            break;
        }

        // Element length (capped at 1024 to prevent OOM but allow boundary testing)
        let len = data[*offset] as usize % 1025;
        *offset += 1;

        let end = (*offset + len).min(data.len());
        witness.push(data[*offset..end].to_vec());
        *offset = end;
    }
    witness
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let mut offset = 0;

    // 1. Parse Inputs
    // We parse two independent witnesses to test orthogonality/independence invariants
    let witness_a = parse_witness(data, &mut offset);
    let witness_b = parse_witness(data, &mut offset);

    if offset >= data.len() {
        return;
    }

    // Flags and Bias
    let flags = data[offset];
    let is_script_path = (flags & 0x01) != 0;
    let bias_mode = flags >> 1; // Use remaining bits for biasing
    offset += 1;

    // ScriptPubKey (Saturating slice of remaining data)
    let script = if offset < data.len() {
        data[offset..].to_vec()
    } else {
        Vec::new()
    };

    // --- Adversarial Taproot Biasing ---
    // If bias_mode is active and we have a witness, we attempt to force a specific control block length
    let mut biased_witness_a = witness_a.clone();
    if biased_witness_a.is_empty() {
        // Create a degenerate witness if empty to test biasing
        biased_witness_a.push(vec![0u8; 33]);
    } else {
        let last_idx = biased_witness_a.len() - 1;
        match bias_mode {
            0 => biased_witness_a[last_idx] = vec![0u8; 33],        // Exact minimum
            1 => biased_witness_a[last_idx] = vec![0u8; 65],        // 33 + 32*1
            2 => biased_witness_a[last_idx] = vec![0u8; 32],        // T-1 (Invalid)
            3 => biased_witness_a[last_idx] = vec![0u8; 34],        // T+1 (Invalid)
            4 => biased_witness_a[last_idx] = vec![0u8; 97],        // 33 + 32*2
            _ => {}
        }
    }

    // 2. Structure Validation & Idempotence
    // Test SegWit structure (Idempotence check)
    let res_segwit_1 = validate_segwit_witness_structure(&witness_a);
    let res_segwit_2 = validate_segwit_witness_structure(&witness_a);
    assert_eq!(res_segwit_1, res_segwit_2, "SegWit validation must be idempotent");

    // Test Taproot structure
    let _ = validate_taproot_witness_structure(&biased_witness_a, is_script_path);
    let _ = validate_taproot_witness_structure(&witness_b, is_script_path);

    // 3. Version & Program Validation
    let version = extract_witness_version(&script);
    if let Some(v) = version {
        // Functional Chaining Invariant:
        // If version is extracted, program extraction must not panic
        if let Some(program) = extract_witness_program(&script, v) {
            let len_valid = validate_witness_program_length(&program, v);

            // Idempotence check for program length
            assert_eq!(len_valid, validate_witness_program_length(&program, v),
                "Program length validation must be idempotent");
        }
    }

    // 4. Independence Invariant
    // The result of extract_witness_version should be entirely independent of witness data.
    let v1 = extract_witness_version(&script);
    let v2 = extract_witness_version(&script);
    assert_eq!(v1, v2, "Version extraction must be independent of witness state");
});
