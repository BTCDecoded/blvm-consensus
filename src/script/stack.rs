//! Stack types and operations for script execution.
//!
//! StackElement, to_stack_element, and cast_to_bool are used by both
//! production and non-production script execution paths.

use crate::types::*;

#[cfg(feature = "production")]
use smallvec::SmallVec;

/// Stack element: inline up to 80 bytes when production (sigs, pubkeys, hashes), else `Vec<u8>`.
#[cfg(feature = "production")]
pub type StackElement = SmallVec<[u8; 80]>;
#[cfg(not(feature = "production"))]
pub type StackElement = ByteString;

/// Convert bytes to StackElement (for tests and callers needing explicit conversion).
#[inline]
pub fn to_stack_element(data: &[u8]) -> StackElement {
    #[cfg(feature = "production")]
    return SmallVec::from_slice(data);
    #[cfg(not(feature = "production"))]
    return data.to_vec();
}

/// CastToBool: truthiness check for stack elements (BIP62/consensus).
/// Returns true if ANY byte is non-zero, except for "negative zero" (0x80 in last byte, rest zeros).
#[cfg(feature = "production")]
#[inline(always)]
pub fn cast_to_bool(v: &[u8]) -> bool {
    for i in 0..v.len() {
        if v[i] != 0 {
            // Negative zero: all zeros except 0x80 in the last byte
            if i == v.len() - 1 && v[i] == 0x80 {
                return false;
            }
            return true;
        }
    }
    false
}

#[cfg(not(feature = "production"))]
#[inline]
pub fn cast_to_bool(v: &[u8]) -> bool {
    for i in 0..v.len() {
        if v[i] != 0 {
            // Negative zero: all zeros except 0x80 in the last byte
            if i == v.len() - 1 && v[i] == 0x80 {
                return false;
            }
            return true;
        }
    }
    false
}
