//! Optimized integer arithmetic operations for hot paths
//!
//! Provides fast-path implementations for common arithmetic operations
//! that are used frequently in validation code. The fast path uses
//! pre-validation to avoid checked arithmetic overhead when values are
//! known to be safe.

use crate::constants::MAX_MONEY;
use crate::error::{ConsensusError, Result};

/// Safe maximum value for fast-path arithmetic
///
/// Values below this threshold are guaranteed to not overflow when
/// added together (even with many additions). This is set conservatively
/// to ensure safety.
#[allow(dead_code)] // Used in tests
const MAX_SAFE_VALUE: i64 = MAX_MONEY / 2;

/// Fast-path addition with overflow checking
///
/// Uses manual overflow detection for common cases (both positive values)
/// which is faster than `checked_add` for the hot path. Falls back to
/// `checked_add` for edge cases.
///
/// # Safety
///
/// This function maintains the same safety guarantees as `checked_add`,
/// but with better performance for common cases.
#[inline(always)]
#[cfg(feature = "production")]
pub fn safe_add(a: i64, b: i64) -> Result<i64> {
    // Fast path: both values are positive (common case in Bitcoin)
    // Manual overflow check: a + b > i64::MAX is equivalent to a > i64::MAX - b
    if a >= 0 && b >= 0 {
        if a > i64::MAX - b {
            return Err(ConsensusError::TransactionValidation(
                "Arithmetic overflow".to_string(),
            ));
        }
        Ok(a + b)
    } else if a < 0 && b < 0 {
        // Both negative: check for underflow (a + b < i64::MIN)
        // Equivalent to a < i64::MIN - b
        if a < i64::MIN - b {
            return Err(ConsensusError::TransactionValidation(
                "Arithmetic underflow".to_string(),
            ));
        }
        Ok(a + b)
    } else {
        // Mixed signs: use checked arithmetic (overflow not possible, but safer)
        a.checked_add(b).ok_or_else(|| {
            ConsensusError::TransactionValidation("Arithmetic overflow".to_string())
        })
    }
}

#[cfg(not(feature = "production"))]
pub fn safe_add(a: i64, b: i64) -> Result<i64> {
    // Always use checked arithmetic in non-production builds
    a.checked_add(b).ok_or_else(|| {
        ConsensusError::TransactionValidation("Arithmetic overflow".to_string())
    })
}

/// Fast-path subtraction with overflow checking
///
/// Uses manual overflow detection for common cases which is faster than
/// `checked_sub` for the hot path. Falls back to `checked_sub` for edge cases.
///
/// # Safety
///
/// This function maintains the same safety guarantees as `checked_sub`,
/// but with better performance for common cases.
#[inline(always)]
#[cfg(feature = "production")]
pub fn safe_sub(a: i64, b: i64) -> Result<i64> {
    // Fast path: a >= 0, b >= 0 (common case: subtracting output from input)
    // Manual underflow check: a - b < i64::MIN is equivalent to a < i64::MIN + b
    // But since a >= 0 and b >= 0, underflow only happens if result < 0, which is fine for i64
    // Actually, for a >= 0 and b >= 0, a - b can underflow if b > a, but that's fine (negative result)
    // The real issue is if a < i64::MIN + b (which can't happen if both are >= 0)
    if a >= 0 && b >= 0 {
        // No underflow possible (both positive)
        Ok(a - b)
    } else if a < 0 && b < 0 {
        // Both negative: check for overflow (a - b > i64::MAX)
        // Equivalent to a > i64::MAX + b, but since both are negative, this can't happen
        Ok(a - b)
    } else {
        // Mixed signs: use checked arithmetic
        a.checked_sub(b).ok_or_else(|| {
            ConsensusError::TransactionValidation("Arithmetic underflow".to_string())
        })
    }
}

#[cfg(not(feature = "production"))]
pub fn safe_sub(a: i64, b: i64) -> Result<i64> {
    // Always use checked arithmetic in non-production builds
    a.checked_sub(b).ok_or_else(|| {
        ConsensusError::TransactionValidation("Arithmetic underflow".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_add_positive() {
        assert_eq!(safe_add(100, 200).unwrap(), 300);
        assert_eq!(safe_add(MAX_SAFE_VALUE, 0).unwrap(), MAX_SAFE_VALUE);
    }

    #[test]
    fn test_safe_add_overflow() {
        assert!(safe_add(i64::MAX, 1).is_err());
    }

    #[test]
    fn test_safe_sub_positive() {
        assert_eq!(safe_sub(300, 200).unwrap(), 100);
        assert_eq!(safe_sub(MAX_SAFE_VALUE, 0).unwrap(), MAX_SAFE_VALUE);
    }

    #[test]
    fn test_safe_sub_underflow() {
        assert!(safe_sub(i64::MIN, 1).is_err());
    }
}

