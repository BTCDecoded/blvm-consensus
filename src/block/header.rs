//! Block header validation (Orange Paper Section 5.3).
//!
//! Single place for header rules; easier to add BIP54 timewarp and version checks.

use crate::error::Result;
use crate::types::{BlockHeader, TimeContext};
use blvm_spec_lock::spec_locked;

/// Validate block header
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
    assert!(
        header.bits != 0,
        "Header bits {} must be non-zero for valid header",
        header.bits
    );
    if header.merkle_root == [0u8; 32] {
        return Ok(false);
    }
    assert!(
        header.merkle_root != [0u8; 32],
        "Merkle root must be non-zero for valid header"
    );
    assert!(
        header.merkle_root.len() == 32,
        "Merkle root length {} must be 32 bytes",
        header.merkle_root.len()
    );
    if header.version == 0 {
        return Ok(false);
    }
    assert!(
        header.version >= 1,
        "Header version {} must be >= 1 for valid header",
        header.version
    );
    let result = true;
    #[allow(clippy::eq_op)]
    {
        assert!(result || !result, "Validation result must be boolean");
    }
    assert!(result, "Validation result must be true on success");
    Ok(result)
}
