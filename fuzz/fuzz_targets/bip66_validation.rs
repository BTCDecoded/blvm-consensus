#![no_main]
use blvm_consensus::bip_validation::check_bip66_network;
use blvm_consensus::types::Network;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // BIP66 strict DER signature validation fuzzing
    // Tests check_bip66_network → check_bip66 → is_strict_der directly,
    // bypassing the script VM gate that blocks most inputs in signature_verification.

    if data.is_empty() {
        return;
    }

    // First byte selects activation state; remaining bytes are the signature.
    // This keeps 1 call per input for maximum fuzzing throughput.
    let selector = data[0];
    let sig = &data[1..];

    let (network, height, fork_active) = match selector % 3 {
        0 => (Network::Mainnet, 0u64, false),       // pre-activation
        1 => (Network::Mainnet, 363_725u64, true),   // activation boundary
        _ => (Network::Regtest, 0u64, true),         // always active
    };

    let valid = check_bip66_network(sig, height, network)
        .expect("check_bip66_network must never return Err");

    // Invariant: pre-activation → always true regardless of content
    if !fork_active {
        assert!(
            valid,
            "BIP66 must return true before activation (h={}, net={:?})",
            height, network
        );
        return;
    }

    // Fork is active — validate spec-model invariants.
    // The DER structure excludes the trailing sighash byte:
    //   0x30 [total-len] 0x02 [R-len] [R] 0x02 [S-len] [S] [sighash]
    // is_strict_der checks the full buffer (including sighash byte in length math).

    if valid {
        // --- Bounds-safe spec-model oracle ---
        // Every index is guarded so the oracle never panics from malformed data.
        assert!(
            sig.len() >= 9 && sig.len() <= 73,
            "DER valid but len {} outside [9,73]",
            sig.len()
        );
        assert_eq!(sig[0], 0x30, "DER valid but missing SEQUENCE tag");
        assert_eq!(
            sig[1],
            (sig.len() - 3) as u8,
            "DER valid but total-length mismatch"
        );
        assert_eq!(sig[2], 0x02, "DER valid but R missing INTEGER tag");

        let len_r = sig[3] as usize;
        assert!(len_r > 0, "DER valid but R length is zero");
        assert!(
            5 + len_r < sig.len(),
            "DER valid but R overflows signature"
        );

        // R not negative
        assert!(sig[4] & 0x80 == 0, "DER valid but R is negative");
        // No unnecessary leading zero in R
        if len_r > 1 {
            assert!(
                !(sig[4] == 0x00 && sig[5] & 0x80 == 0),
                "DER valid but R has unnecessary leading zero"
            );
        }

        // S integer tag
        assert_eq!(
            sig[4 + len_r], 0x02,
            "DER valid but S missing INTEGER tag"
        );
        let len_s = sig[5 + len_r] as usize;
        assert!(len_s > 0, "DER valid but S length is zero");

        // S not negative
        assert!(
            sig[len_r + 6] & 0x80 == 0,
            "DER valid but S is negative"
        );
        // No unnecessary leading zero in S
        if len_s > 1 && len_r + 7 < sig.len() {
            assert!(
                !(sig[len_r + 6] == 0x00 && sig[len_r + 7] & 0x80 == 0),
                "DER valid but S has unnecessary leading zero"
            );
        }

        // Length consistency
        assert_eq!(
            len_r + len_s + 7,
            sig.len(),
            "DER valid but R+S lengths don't sum to signature length"
        );
    } else {
        // Weak negative invariant: rejected signatures must have at least one
        // obvious structural defect detectable from the first few bytes.
        let looks_valid = sig.len() >= 9
            && sig.len() <= 73
            && sig.get(0) == Some(&0x30)
            && sig.get(2) == Some(&0x02);
        if looks_valid {
            // If the outer structure looks fine, at least the length field
            // or R/S internals must be wrong. This catches false negatives.
            let total_len_ok = sig[1] == (sig.len() - 3) as u8;
            let len_r = sig[3] as usize;
            let r_tag_ok = len_r > 0 && 5 + len_r < sig.len();
            let s_tag_ok = r_tag_ok && sig.get(4 + len_r) == Some(&0x02);
            let len_s_present = r_tag_ok && 5 + len_r < sig.len();
            let sum_ok = if len_s_present {
                let len_s = sig[5 + len_r] as usize;
                len_r + len_s + 7 == sig.len()
            } else {
                false
            };

            // If structural checks pass, check if it's rejected due to BIP66 value constraints.
            // This prevents false positives when the implementation correctly rejects
            // a signature that is structurally correct but violates value rules.
            let value_ok = if total_len_ok && r_tag_ok && s_tag_ok && sum_ok {
                let len_r = sig[3] as usize;
                let len_s = sig[5 + len_r] as usize;

                let r_not_neg = sig[4] & 0x80 == 0;
                let r_no_lead_zero = if len_r > 1 {
                    !(sig[4] == 0x00 && sig[5] & 0x80 == 0)
                } else {
                    true
                };
                
                let s_len_ok = len_s > 0;
                let s_not_neg = sig[len_r + 6] & 0x80 == 0;
                let s_no_lead_zero = if len_s > 1 && len_r + 7 < sig.len() {
                    !(sig[len_r + 6] == 0x00 && sig[len_r + 7] & 0x80 == 0)
                } else {
                    true
                };

                r_not_neg && r_no_lead_zero && s_len_ok && s_not_neg && s_no_lead_zero
            } else {
                true // Structural failure already justifies rejection
            };

            // At least one check must fail for a valid rejection.
            assert!(
                !total_len_ok || !r_tag_ok || !s_tag_ok || !sum_ok || !value_ok,
                "Rejected signature looks structurally valid — possible false negative"
            );
        }
    }
});
