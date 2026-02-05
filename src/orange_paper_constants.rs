//! Constants extracted from Orange Paper Section 4 (Consensus Constants)
//!
//! This file is AUTO-GENERATED from blvm-spec/THE_ORANGE_PAPER.md
//! DO NOT EDIT MANUALLY - changes should be made to Orange Paper
//!
//! To regenerate: cargo spec-lock extract-constants
//!
//! These constants are always available for use in property tests and code.
//! Each constant is linked to its Orange Paper section via documentation comments.

/// satoshis per BTC, see [Economic Model](#6-economic-model)
/// 
/// Source: Orange Paper Section 4.1
/// Formula: $C = 10^8$
pub const C: u64 = 100_000_000;

/// maximum money supply, see [Supply Limit](#supply-limit)
/// 
/// Source: Orange Paper Section 4.1
/// Formula: $M_MAX = 21 \times 10^6 \times C$
pub const M_MAX: i64 = ((21_000_000 * C) as i64) as i64;

/// halving interval, see [Block Subsidy](#61-block-subsidy)
/// 
/// Source: Orange Paper Section 4.1
/// Formula: $H = 210,000$
pub const H: u64 = 210_000;

/// maximum block weight, see [Block Validation](#53-block-validation)
/// 
/// Source: Orange Paper Section 4.2
/// Formula: $W_MAX = 4 \times 10^6$
pub const W_MAX: u64 = 4_000_000;

/// maximum sigops per block, see [Script Execution](#52-script-execution)
/// 
/// Source: Orange Paper Section 4.2
/// Formula: $S_MAX = 80,000$
pub const S_MAX: u64 = 80_000;

/// coinbase maturity requirement, see [Transaction Validation](#51-transaction-validation)
/// 
/// Source: Orange Paper Section 4.2
/// Formula: $R = 100$
pub const R: u64 = 100;

/// maximum script length, see [Script Security](#script-security)
/// 
/// Source: Orange Paper Section 4.3
/// Formula: $L_SCRIPT = 10,000$
pub const L_SCRIPT: u64 = 10_000;

/// maximum stack size, see [Script Execution Bounds](#theorem-84)
/// 
/// Source: Orange Paper Section 4.3
/// Formula: $L_STACK = 1,000$
pub const L_STACK: u64 = 1_000;

/// maximum operations per script, see [Script Execution Bounds](#theorem-84)
/// 
/// Source: Orange Paper Section 4.3
/// Formula: $L_OPS = 201$
pub const L_OPS: u64 = 201;

/// maximum element size, see [Script Execution](#52-script-execution)
/// 
/// Source: Orange Paper Section 4.3
/// Formula: $L_ELEMENT = 520$
pub const L_ELEMENT: u64 = 520;

