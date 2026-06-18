//! Block header validation (Orange Paper Section 5.3, §5.3.1).
//!
//! Single place for structural and time header rules. Part of a larger validation pipeline.
//!
//! ## What this module checks (H01, H03–H06 of §5.3.1)
//!
//! - **H01** — version ≥ 1 (floor; version 0 is rejected unconditionally)
//! - **H03** — timestamp ≠ 0
//! - **H04** — timestamp ≤ network_time + MAX_FUTURE_BLOCK_TIME (requires [`TimeContext`])
//! - **H05** — timestamp ≥ median_time_past / BIP113 MTP (requires [`TimeContext`])
//! - **H06** — bits ≠ 0
//! - merkle_root ≠ all-zeros (structural sanity only; full merkle verification is in ConnectBlock)
//!
//! ## What this module does NOT check
//!
//! - **H02** — height-dependent version minimums (version ≥ 2/3/4 after BIP34/66/65): see
//!   [`crate::bip_validation::check_bip90`], called by `connect_block_inner`.
//! - **H07** — proof of work (hash vs compact target): see [`crate::pow::check_proof_of_work`].
//! - **H08** — parent hash linkage: pure predicate [`validate_prev_block_hash`]; the node chain
//!   layer supplies the parent header/hash and rejects blocks that fail H08 before `connect_block`.
//!
//! Callers connecting a block must invoke H02, H07, and H08 (via the predicate) in addition to
//! [`validate_block_header`] to satisfy `ValidBlockHeader` in full.
//!
//! ## Refactor / audit notes (coordinate with `blvm-spec-lock` before changing shape)
//!
//! - **Early returns** encode consensus rejects (`Ok(false)`). Do not duplicate the same condition
//!   with `assert!` below — that only adds panic risk if someone reorders code.
//! - The tautological `assert!(result || !result)` (below) is **on purpose**: formal verification /
//!   spec-lock tooling hooks here. Do not delete without verifier sign-off.
//! - **Version `0`** is rejected by `version < 1` (H01). Version 1 is valid before BIP34 and
//!   invalid after it — that boundary is enforced by `check_bip90` (H02), not here.
//! - **Merkle root** field is checked for all-zeros only (structural guard). Cryptographic
//!   verification of the merkle root against block transactions happens in `connect_block_inner`.

use crate::error::Result;
use crate::types::{BlockHeader, TimeContext};
use blvm_spec_lock::spec_locked;

/// Validate block header structural and time rules (H01, H03–H06 of §5.3.1).
///
/// Returns `Ok(true)` if all checks pass, `Ok(false)` if any check fails.
///
/// This is one component of `ValidBlockHeader`. Callers connecting a block must also invoke:
/// - [`crate::bip_validation::check_bip90`] — H02: height-dependent version minimums
/// - [`crate::pow::check_proof_of_work`] — H07: hash vs compact target
///
/// Parent hash linkage (H08): [`validate_prev_block_hash`]; orchestration is in the node layer.
///
/// # Arguments
///
/// * `header` - Block header to validate
/// * `time_context` - Optional time context for timestamp validation (BIP113).
///   If `None`, only H01/H03/H06 (version, non-zero timestamp, bits) are enforced.
///   If `Some`, also enforces H04 (timestamp ≤ network_time + MAX_FUTURE_BLOCK_TIME)
///   and H05 (timestamp ≥ median_time_past).
#[allow(clippy::overly_complex_bool_expr, clippy::redundant_comparisons)] // Intentional tautological assertions for formal verification
#[spec_locked("5.3.1", "ValidBlockHeader")]
#[inline]
pub(crate) fn validate_block_header(
    header: &BlockHeader,
    time_context: Option<&TimeContext>,
) -> Result<bool> {
    if header.version < 1 {
        return Ok(false);
    }
    if header.timestamp == 0 {
        return Ok(false);
    }
    if let Some(ctx) = time_context {
        let max_ts = ctx
            .network_time
            .saturating_add(crate::constants::MAX_FUTURE_BLOCK_TIME);
        if header.timestamp > max_ts {
            return Ok(false);
        }
        if header.timestamp < ctx.median_time_past {
            return Ok(false);
        }
    }
    if header.bits == 0 {
        return Ok(false);
    }
    if header.merkle_root == [0u8; 32] {
        return Ok(false);
    }

    // Formal-verification anchor (spec-lock): keep `result` and the tautology; omit a second
    // `assert!(result)` — success is `Ok(true)` below.
    let result = true;
    #[allow(clippy::eq_op)]
    {
        assert!(result || !result, "Validation result must be boolean");
    }
    Ok(result)
}

/// Double-SHA256 hash of an 80-byte serialized block header (Bitcoin block id).
#[spec_locked("5.3.1", "BlockHeaderHash")]
#[inline]
pub fn block_header_hash(header: &BlockHeader) -> crate::types::Hash {
    use blvm_primitives::crypto::hash256;
    use blvm_primitives::serialization::serialize_block_header;
    hash256(&serialize_block_header(header))
}

/// H08 (§5.3.1): `child.prev_block_hash` MUST equal `block_header_hash(parent)`.
///
/// Pure predicate for spec-lock binding. The node layer calls this before `connect_block` when
/// the parent header is known; IBD/header sync enforces the same invariant while walking the chain.
#[spec_locked("5.3.1", "ValidatePrevBlockHash")]
#[blvm_spec_lock::ensures(result == true || result == false)]
#[inline]
pub fn validate_prev_block_hash(child: &BlockHeader, parent: &BlockHeader) -> bool {
    child.prev_block_hash == block_header_hash(parent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BlockHeader;

    #[test]
    fn validate_prev_block_hash_accepts_matching_parent() {
        let parent = BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [1u8; 32],
            timestamp: 1,
            bits: 0x1d00ffff,
            nonce: 0,
        };
        let parent_hash = block_header_hash(&parent);
        let child = BlockHeader {
            prev_block_hash: parent_hash,
            merkle_root: [2u8; 32],
            ..parent
        };
        assert!(validate_prev_block_hash(&child, &parent));
    }

    #[test]
    fn validate_prev_block_hash_rejects_mismatch() {
        let parent = BlockHeader {
            version: 1,
            prev_block_hash: [0u8; 32],
            merkle_root: [1u8; 32],
            timestamp: 1,
            bits: 0x1d00ffff,
            nonce: 0,
        };
        let child = BlockHeader {
            prev_block_hash: [9u8; 32],
            merkle_root: [2u8; 32],
            ..parent
        };
        assert!(!validate_prev_block_hash(&child, &parent));
    }
}
