//! UTXO Commitment Verification
//!
//! Provides utilities for verifying UTXO commitments against:
//! - Bitcoin supply calculations
//! - Block header chain (Proof of Work)
//! - Peer consensus consistency

#[cfg(feature = "utxo-commitments")]
use crate::economic::total_supply;
#[cfg(feature = "utxo-commitments")]
use crate::pow::check_proof_of_work;
#[cfg(feature = "utxo-commitments")]
use crate::types::{BlockHeader, Hash, Natural};
#[cfg(feature = "utxo-commitments")]
use crate::utxo_commitments::data_structures::{
    UtxoCommitment, UtxoCommitmentError, UtxoCommitmentResult,
};
#[cfg(feature = "utxo-commitments")]
use blvm_spec_lock::spec_locked;

/// Verify that a UTXO commitment's supply matches expected Bitcoin supply
///
/// Checks that the total supply in the commitment equals the sum of all
/// block subsidies up to the commitment's block height.
#[spec_locked("13.2")]
pub fn verify_supply(commitment: &UtxoCommitment) -> UtxoCommitmentResult<bool> {
    let expected_supply = total_supply(commitment.block_height) as u64;

    if commitment.total_supply != expected_supply {
        return Err(UtxoCommitmentError::VerificationFailed(format!(
            "Supply mismatch at height {}: commitment has {} satoshis, expected {} satoshis",
            commitment.block_height, commitment.total_supply, expected_supply
        )));
    }

    Ok(true)
}

/// Verify that a block header chain is valid (Proof of Work)
///
/// Verifies the chain of block headers from genesis to the commitment height,
/// checking that each header satisfies proof of work requirements.
#[spec_locked("13.2")]
pub fn verify_header_chain(headers: &[BlockHeader]) -> UtxoCommitmentResult<bool> {
    if headers.is_empty() {
        return Err(UtxoCommitmentError::VerificationFailed(
            "Empty header chain".to_string(),
        ));
    }

    // Verify each header's proof of work
    for (i, header) in headers.iter().enumerate() {
        match check_proof_of_work(header) {
            Ok(is_valid) => {
                if !is_valid {
                    return Err(UtxoCommitmentError::VerificationFailed(format!(
                        "Invalid proof of work at height {}",
                        i
                    )));
                }
            }
            Err(e) => {
                return Err(UtxoCommitmentError::VerificationFailed(format!(
                    "PoW check failed at height {}: {}",
                    i, e
                )));
            }
        }

        // Verify chain linkage (except for genesis)
        if i > 0 {
            let prev_header = &headers[i - 1];
            // Compute block hash using double SHA256
            let expected_prev_hash = compute_block_hash(prev_header);

            if header.prev_block_hash != expected_prev_hash {
                return Err(UtxoCommitmentError::VerificationFailed(format!(
                    "Chain linkage broken at height {}: expected prev_hash {:?}, got {:?}",
                    i, expected_prev_hash, header.prev_block_hash
                )));
            }
        }
    }

    Ok(true)
}

/// Verify commitment against block header
///
/// Verifies that the commitment's block_hash matches the actual block hash
/// at the given height.
#[spec_locked("13.2")]
pub fn verify_commitment_block_hash(
    commitment: &UtxoCommitment,
    header: &BlockHeader,
) -> UtxoCommitmentResult<bool> {
    let computed_hash = compute_block_hash(header);

    if commitment.block_hash != computed_hash {
        return Err(UtxoCommitmentError::VerificationFailed(format!(
            "Block hash mismatch: commitment has {:?}, header has {:?}",
            commitment.block_hash, computed_hash
        )));
    }

    Ok(true)
}

/// Compute block header hash (double SHA256)
fn compute_block_hash(header: &BlockHeader) -> Hash {
    use sha2::{Digest, Sha256};

    // Serialize block header (version, prev_block_hash, merkle_root, timestamp, bits, nonce)
    let mut bytes = Vec::with_capacity(80);
    bytes.extend_from_slice(&header.version.to_le_bytes());
    bytes.extend_from_slice(&header.prev_block_hash);
    bytes.extend_from_slice(&header.merkle_root);
    bytes.extend_from_slice(&header.timestamp.to_le_bytes());
    bytes.extend_from_slice(&header.bits.to_le_bytes());
    bytes.extend_from_slice(&header.nonce.to_le_bytes());

    // Double SHA256
    let first_hash = Sha256::digest(&bytes);
    let second_hash = Sha256::digest(&first_hash);

    let mut hash = [0u8; 32];
    hash.copy_from_slice(&second_hash);
    hash
}

/// Verify forward consistency
///
/// Verifies that applying a sequence of blocks to a commitment results in
/// a consistent new commitment. Used to ensure commitments remain valid
/// as the chain progresses.
#[spec_locked("13.2")]
pub fn verify_forward_consistency(
    initial_commitment: &UtxoCommitment,
    new_commitment: &UtxoCommitment,
    expected_height_increase: Natural,
) -> UtxoCommitmentResult<bool> {
    // Verify height progression
    if new_commitment.block_height != initial_commitment.block_height + expected_height_increase {
        return Err(UtxoCommitmentError::VerificationFailed(format!(
            "Height mismatch: initial {}, new {}, expected increase {}",
            initial_commitment.block_height, new_commitment.block_height, expected_height_increase
        )));
    }

    // Verify supply progression (should only increase or stay same, never decrease)
    if new_commitment.total_supply < initial_commitment.total_supply {
        return Err(UtxoCommitmentError::VerificationFailed(format!(
            "Supply decreased: initial {}, new {}",
            initial_commitment.total_supply, new_commitment.total_supply
        )));
    }

    // Note: We can't verify UTXO count changes without knowing transactions,
    // but we can verify supply changes match expected block subsidies
    let expected_new_supply = total_supply(new_commitment.block_height) as u64;
    if new_commitment.total_supply != expected_new_supply {
        return Err(UtxoCommitmentError::VerificationFailed(format!(
            "New supply mismatch: commitment has {}, expected {}",
            new_commitment.total_supply, expected_new_supply
        )));
    }

    Ok(true)
}

// ============================================================================
// FORMAL VERIFICATION
// ============================================================================
//
// Mathematical Specification for UTXO Commitment Verification:
// ∀ commitment ∈ UtxoCommitment, height ∈ ℕ:
// - verify_supply(commitment) = true ⟺ commitment.total_supply = total_supply(height)
// - verify_forward_consistency(c1, c2) = true ⟺ c2.height = c1.height + Δ ∧ c2.supply ≥ c1.supply
//
// Invariants:
// - Supply verification prevents inflation
// - Forward consistency ensures commitments progress correctly
// - Block hash verification ensures commitment matches actual block

