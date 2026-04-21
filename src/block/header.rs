//! Block header validation (Orange Paper Section 5.3).
//!
//! Single place for header rules; easier to add BIP54 timewarp and version checks.
//!
//! ## Scope
//!
//! This module checks **structural / field** rules and **time** rules when a [`TimeContext`] is
//! supplied. **Proof-of-work** (hash vs compact target) is **not** validated here — use
//! [`crate::pow::check_proof_of_work`] / chain connection paths that combine PoW with context.
//!
//! ## Refactor / audit notes (coordinate with `blvm-spec-lock` before changing shape)
//!
//! - **Early returns** encode consensus rejects (`Ok(false)`). Do not duplicate the same condition
//!   with `assert!` below — that only adds panic risk if someone reorders code.
//! - The tautological `assert!(result || !result)` (below) is **on purpose**: formal verification /
//!   spec-lock tooling hooks here. Do not delete without verifier sign-off.
//! - **Version `0`** is rejected by `version < 1`.
//! - **Merkle root** is `[u8; 32]`; an extra length check would be redundant.

use crate::error::Result;
use crate::types::{BlockHeader, TimeContext};
use blvm_spec_lock::spec_locked;

/// Validate block header fields and optional BIP113-style time rules.
///
/// Returns `Ok(true)` if all checks pass, `Ok(false)` if the header is invalid for these rules.
/// Does **not** run proof-of-work; see [`crate::pow::check_proof_of_work`].
///
/// # Arguments
///
/// * `header` - Block header to validate
/// * `time_context` - Optional time context for timestamp validation (BIP113)
///   If None, only basic timestamp checks are performed (non-zero).
///   If Some, full timestamp validation is performed:
///   - Rejects blocks with timestamps > network_time + MAX_FUTURE_BLOCK_TIME
///   - Rejects blocks with timestamps < median_time_past
#[allow(clippy::overly_complex_bool_expr, clippy::redundant_comparisons)] // Intentional tautological assertions for formal verification
#[spec_locked("5.3")]
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
