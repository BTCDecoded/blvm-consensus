//! Unified witness validation framework for SegWit (BIP141) and Taproot (BIP340/341/342)
//!
//! Provides shared functions for witness structure validation, weight calculation,
//! and witness data handling that are common to both SegWit and Taproot.

use crate::error::Result;
use crate::types::*;

/// Witness Data: 𝒲 = 𝕊* (stack of witness elements)
///
/// Shared witness type used by both SegWit and Taproot.
/// For SegWit: Vector of byte strings representing witness stack elements
/// For Taproot: Vector containing control block and script path data
pub type Witness = Vec<ByteString>;

/// Witness version for SegWit (v0) and Taproot (v1)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessVersion {
    /// SegWit version 0
    SegWitV0 = 0,
    /// Taproot version 1
    TaprootV1 = 1,
}

/// Validate witness structure for SegWit
///
/// BIP141: Witness must be a vector of byte strings (stack elements).
/// Each element can be up to MAX_SCRIPT_ELEMENT_SIZE bytes.
pub fn validate_segwit_witness_structure(witness: &Witness) -> Result<bool> {
    // Check each witness element size
    // BIP141: Each witness element can be up to 520 bytes (MAX_SCRIPT_ELEMENT_SIZE)
    // Using 520 as the limit per Bitcoin consensus rules
    const MAX_WITNESS_ELEMENT_SIZE: usize = 520;
    for element in witness {
        if element.len() > MAX_WITNESS_ELEMENT_SIZE {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Validate witness structure for Taproot
///
/// BIP341: Taproot witness structure depends on spending path:
/// - Key path: single signature (64 bytes)
/// - Script path: script, control block (33 + 32n bytes), and witness items
pub fn validate_taproot_witness_structure(witness: &Witness, is_script_path: bool) -> Result<bool> {
    if witness.is_empty() {
        return Ok(false);
    }

    if is_script_path {
        // Script path: at least script + control block
        if witness.len() < 2 {
            return Ok(false);
        }

        // Control block must be at least 33 bytes (internal key + leaf version + parity)
        let control_block = &witness[witness.len() - 1];
        if control_block.len() < 33 {
            return Ok(false);
        }

        // Control block size: 33 + 32n (where n is number of merkle proof levels)
        // Must be valid multiple
        if (control_block.len() - 33) % 32 != 0 {
            return Ok(false);
        }
    } else {
        // Key path: single Schnorr signature (64 bytes)
        if witness.len() != 1 {
            return Ok(false);
        }
        if witness[0].len() != 64 {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Calculate transaction weight using SegWit formula
///
/// BIP141: Weight(tx) = 4 × BaseSize(tx) + TotalSize(tx)
/// BaseSize: Transaction size without witness data
/// TotalSize: Transaction size with witness data
pub fn calculate_transaction_weight_segwit(base_size: Natural, total_size: Natural) -> Natural {
    4 * base_size + total_size
}

/// Calculate virtual size (vsize) from weight
///
/// BIP141: vsize = ceil(weight / 4)
/// Used for fee calculation in SegWit transactions
///
/// Mathematical specification:
/// - vsize = ⌈weight / 4⌉
/// - Implemented as: vsize = (weight + 3) / 4 (integer ceiling division)
pub fn weight_to_vsize(weight: Natural) -> Natural {
    #[allow(clippy::manual_div_ceil)]
    let result = (weight + 3) / 4; // Ceiling division
    
    // Runtime assertion: Verify ceiling division property
    // vsize must be >= weight / 4 (ceiling property)
    debug_assert!(
        (result as u64) >= (weight / 4),
        "Vsize ({}) must be >= weight / 4 ({})",
        result,
        weight / 4
    );
    
    // Runtime assertion: vsize must be <= (weight / 4) + 1 (ceiling property)
    // Note: When weight % 4 == 0, result == weight/4, otherwise result == (weight/4) + 1
    debug_assert!(
        (result as u64) <= ((weight / 4) + 1),
        "Vsize ({}) must be <= (weight / 4) + 1 ({})",
        result,
        (weight / 4) + 1
    );
    
    // Runtime assertion: Result must be non-negative
    // Note: Natural is unsigned, so this is always true, but documents the invariant
    debug_assert!(
        true, // Natural is always non-negative
        "Vsize ({}) must be non-negative (weight: {})",
        result,
        weight
    );
    
    result
}

/// Validate witness version in scriptPubKey
///
/// Shared function for extracting and validating witness version
/// from SegWit v0 (OP_0 <witness-program>) or Taproot v1 (OP_1 <witness-program>)
pub fn extract_witness_version(script: &ByteString) -> Option<WitnessVersion> {
    if script.is_empty() {
        return None;
    }

    match script[0] {
        0x51 => Some(WitnessVersion::TaprootV1), // OP_1
        0x00 => Some(WitnessVersion::SegWitV0),  // OP_0
        _ => None,
    }
}

/// Extract witness program from scriptPubKey
///
/// For SegWit v0: Returns bytes after OP_0
/// For Taproot v1: Returns bytes after OP_1
pub fn extract_witness_program(
    script: &ByteString,
    _version: WitnessVersion,
) -> Option<ByteString> {
    if script.len() < 2 {
        return None;
    }

    // Skip version opcode (1 byte) and return program
    Some(script[1..].to_vec())
}

/// Validate witness program length
///
/// BIP141: SegWit v0 programs are 20 or 32 bytes (P2WPKH or P2WSH)
/// BIP341: Taproot v1 programs are 32 bytes (P2TR)
pub fn validate_witness_program_length(program: &ByteString, version: WitnessVersion) -> bool {
    use crate::constants::{SEGWIT_P2WPKH_LENGTH, SEGWIT_P2WSH_LENGTH, TAPROOT_PROGRAM_LENGTH};
    
    match version {
        WitnessVersion::SegWitV0 => {
            // P2WPKH: 20 bytes, P2WSH: 32 bytes
            program.len() == SEGWIT_P2WPKH_LENGTH || program.len() == SEGWIT_P2WSH_LENGTH
        }
        WitnessVersion::TaprootV1 => {
            // P2TR: 32 bytes
            program.len() == TAPROOT_PROGRAM_LENGTH
        }
    }
}

/// Check if witness is empty (non-witness transaction)
pub fn is_witness_empty(witness: &Witness) -> bool {
    witness.is_empty() || witness.iter().all(|elem| elem.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_segwit_witness_structure() {
        let witness = vec![
            vec![0x01; 20], // P2WPKH witness
            vec![0x02; 72], // Signature
        ];
        assert!(validate_segwit_witness_structure(&witness).unwrap());

        // Too large element
        let invalid_witness = vec![vec![0x01; crate::constants::MAX_SCRIPT_ELEMENT_SIZE + 1]];
        assert!(!validate_segwit_witness_structure(&invalid_witness).unwrap());
    }

    #[test]
    fn test_validate_taproot_witness_structure_key_path() {
        // Key path: single 64-byte signature
        let witness = vec![vec![0x01; 64]];
        assert!(validate_taproot_witness_structure(&witness, false).unwrap());

        // Invalid: wrong length
        let invalid = vec![vec![0x01; 63]];
        assert!(!validate_taproot_witness_structure(&invalid, false).unwrap());

        // Invalid: multiple elements
        let invalid2 = vec![vec![0x01; 64], vec![0x02; 32]];
        assert!(!validate_taproot_witness_structure(&invalid2, false).unwrap());
    }

    #[test]
    fn test_validate_taproot_witness_structure_script_path() {
        // Script path: script + control block (33 bytes minimum)
        let witness = vec![
            vec![0x51],    // Script
            vec![0u8; 33], // Control block (internal key + leaf version + parity)
        ];
        assert!(validate_taproot_witness_structure(&witness, true).unwrap());

        // Invalid: control block too small
        let invalid = vec![vec![0x51], vec![0u8; 32]];
        assert!(!validate_taproot_witness_structure(&invalid, true).unwrap());

        // Invalid: only one element
        let invalid2 = vec![vec![0x51]];
        assert!(!validate_taproot_witness_structure(&invalid2, true).unwrap());
    }

    #[test]
    fn test_calculate_transaction_weight_segwit() {
        let base_size = 100;
        let total_size = 150;
        let weight = calculate_transaction_weight_segwit(base_size, total_size);
        assert_eq!(weight, 4 * 100 + 150); // 550
    }

    #[test]
    fn test_weight_to_vsize() {
        assert_eq!(weight_to_vsize(400), 100); // Exact division
        assert_eq!(weight_to_vsize(401), 101); // Ceiling
        assert_eq!(weight_to_vsize(403), 101); // Ceiling
        assert_eq!(weight_to_vsize(404), 101); // Ceiling
    }

    #[test]
    fn test_extract_witness_version() {
        let segwit_script = vec![0x00, 0x14, 0x01, 0x02, 0x03]; // OP_0 <20-byte-program>
        assert_eq!(
            extract_witness_version(&segwit_script),
            Some(WitnessVersion::SegWitV0)
        );

        let taproot_script = vec![0x51, 0x20]; // OP_1 <32-byte-program>
        assert_eq!(
            extract_witness_version(&taproot_script),
            Some(WitnessVersion::TaprootV1)
        );

        let non_witness_script = vec![0x76, 0xa9]; // OP_DUP OP_HASH160
        assert_eq!(extract_witness_version(&non_witness_script), None);
    }

    #[test]
    fn test_extract_witness_program() {
        let segwit_script = vec![0x00, 0x14, 0x01, 0x02, 0x03];
        let program = extract_witness_program(&segwit_script, WitnessVersion::SegWitV0);
        assert_eq!(program, Some(vec![0x14, 0x01, 0x02, 0x03]));
    }

    #[test]
    fn test_validate_witness_program_length() {
        let p2wpkh = vec![0u8; 20]; // 20 bytes
        assert!(validate_witness_program_length(
            &p2wpkh,
            WitnessVersion::SegWitV0
        ));

        let p2wsh = vec![0u8; 32]; // 32 bytes
        assert!(validate_witness_program_length(
            &p2wsh,
            WitnessVersion::SegWitV0
        ));

        let p2tr = vec![0u8; 32]; // 32 bytes
        assert!(validate_witness_program_length(
            &p2tr,
            WitnessVersion::TaprootV1
        ));

        let invalid = vec![0u8; 33];
        assert!(!validate_witness_program_length(
            &invalid,
            WitnessVersion::SegWitV0
        ));
        assert!(!validate_witness_program_length(
            &invalid,
            WitnessVersion::TaprootV1
        ));
    }

    #[test]
    fn test_is_witness_empty() {
        assert!(is_witness_empty(&vec![]));
        assert!(is_witness_empty(&vec![vec![]]));
        assert!(!is_witness_empty(&vec![vec![0x01]]));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use kani::*;

    /// Kani proof: SegWit witness structure validation correctness (BIP141)
    ///
    /// Mathematical specification:
    /// ∀ witness ∈ Witness:
    /// - validate_segwit_witness_structure(witness) = true ⟹
    ///   (∀ element ∈ witness: |element| ≤ MAX_WITNESS_ELEMENT_SIZE)
    #[kani::proof]
    fn kani_segwit_witness_structure_validation() {
        let witness: Witness = kani::any();

        // Bound for tractability
        kani::assume(witness.len() <= 10);
        for element in &witness {
            kani::assume(element.len() <= 600); // Allow testing boundary cases
        }

        const MAX_WITNESS_ELEMENT_SIZE: usize = 520;
        let result = validate_segwit_witness_structure(&witness);

        if result.is_ok() {
            let is_valid = result.unwrap();

            // Critical invariant: if valid, all elements must be <= MAX_WITNESS_ELEMENT_SIZE
            if is_valid {
                for element in &witness {
                    assert!(element.len() <= MAX_WITNESS_ELEMENT_SIZE,
                        "SegWit witness structure validation: valid witness must have all elements <= 520 bytes");
                }
            }
        }
    }

    /// Kani proof: Taproot witness structure validation correctness (BIP341)
    ///
    /// Mathematical specification:
    /// ∀ witness ∈ Witness, is_script_path ∈ bool:
    /// - validate_taproot_witness_structure(witness, is_script_path) = true ⟹
    ///   (if is_script_path: |witness| >= 2 ∧ |witness[-1]| >= 33 ∧ (|witness[-1]| - 33) % 32 == 0)
    ///   (if !is_script_path: |witness| == 1 ∧ |witness[0]| == 64)
    #[kani::proof]
    fn kani_taproot_witness_structure_validation() {
        let witness: Witness = kani::any();
        let is_script_path: bool = kani::any();

        // Bound for tractability
        kani::assume(witness.len() <= 10);
        for element in &witness {
            kani::assume(element.len() <= 200);
        }

        let result = validate_taproot_witness_structure(&witness, is_script_path);

        if result.is_ok() {
            let is_valid = result.unwrap();

            if is_valid {
                if is_script_path {
                    // Script path: at least 2 elements, control block >= 33 bytes, valid size
                    assert!(witness.len() >= 2,
                        "Taproot witness structure validation: script path must have at least 2 elements");
                    if !witness.is_empty() {
                        let control_block = &witness[witness.len() - 1];
                        assert!(control_block.len() >= 33,
                            "Taproot witness structure validation: control block must be >= 33 bytes");
                        assert!((control_block.len() - 33) % 32 == 0,
                            "Taproot witness structure validation: control block size must be 33 + 32n bytes");
                    }
                } else {
                    // Key path: exactly 1 element, 64 bytes
                    assert_eq!(witness.len(), 1,
                        "Taproot witness structure validation: key path must have exactly 1 element");
                    if !witness.is_empty() {
                        assert_eq!(witness[0].len(), 64,
                            "Taproot witness structure validation: key path signature must be 64 bytes");
                    }
                }
            }
        }
    }

    /// Kani proof: Witness program length validation correctness (BIP141/BIP341)
    ///
    /// Mathematical specification:
    /// ∀ program ∈ ByteString, version ∈ WitnessVersion:
    /// - validate_witness_program_length(program, version) = true ⟹
    ///   (if version = SegWitV0: |program| ∈ {20, 32})
    ///   (if version = TaprootV1: |program| == 32)
    #[kani::proof]
    fn kani_witness_program_length_validation() {
        let program: Vec<u8> = kani::any();
        let version: WitnessVersion = kani::any();

        // Bound for tractability
        kani::assume(program.len() <= 40);

        let result = validate_witness_program_length(&program, version);

        // Critical invariant: result must match specification
        use crate::constants::{SEGWIT_P2WPKH_LENGTH, SEGWIT_P2WSH_LENGTH, TAPROOT_PROGRAM_LENGTH};
        
        match version {
            WitnessVersion::SegWitV0 => {
                assert_eq!(
                    result,
                    program.len() == SEGWIT_P2WPKH_LENGTH || program.len() == SEGWIT_P2WSH_LENGTH,
                    "Witness program length validation: SegWit v0 must be {} or {} bytes",
                    SEGWIT_P2WPKH_LENGTH, SEGWIT_P2WSH_LENGTH
                );
            }
            WitnessVersion::TaprootV1 => {
                assert_eq!(
                    result,
                    program.len() == TAPROOT_PROGRAM_LENGTH,
                    "Witness program length validation: Taproot v1 must be {} bytes",
                    TAPROOT_PROGRAM_LENGTH
                );
            }
        }
    }
}
