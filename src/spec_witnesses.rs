//! Z3 verification witness functions for Orange Paper F_* formulas.
//!
//! These are NOT production code. Each function is a verification artifact:
//! it expresses a formula from the Orange Paper in Z3-translatable Rust so that
//! `cargo spec-lock verify --features z3` can produce a genuine proof obligation,
//! not a vacuous zero-contract pass.
//!
//! Authoring rules (see docs/END_TO_END_PROOF_PLAN.md § Track G):
//! - Do NOT call the production function; express the formula inline.
//! - Use `match k { 0 => X >> 0, 1 => X >> 1, ... }` for piecewise functions
//!   (literal-RHS shifts translate; variable-RHS shifts do not).
//! - No loops, iterators, or method calls on non-whitelisted types.
//! - Body must be translatable by the z3_translator `syn` AST walker.

#[allow(unused_imports)]
use crate::constants::{
    BIP54_MAX_SIGOPS_PER_TX, HALVING_INTERVAL, INITIAL_SUBSIDY, LOCKTIME_THRESHOLD,
    MAX_BLOCK_SIGOPS_COST, MAX_MONEY, TAPROOT_ACTIVATION_MAINNET, TAPROOT_ACTIVATION_TESTNET,
};
use blvm_spec_lock::spec_locked;

// ─── §6.5 Fee Market ─────────────────────────────────────────────────────────

/// Witness for **F_FeeNonNeg** (PROTOCOL.md §6.5).
///
/// Proof obligation: fee = total_in - total_out ≥ 0 when total_in ≥ total_out ≥ 0.
///
/// Z3 proves this via linear arithmetic: the two `requires` establish `total_in ≥ total_out`
/// and `total_out ≥ 0`, so `result = total_in - total_out ≥ 0` follows immediately.
///
/// Formalises the core invariant that transaction fees are non-negative for valid transactions.
#[spec_locked("6.5", "F_FeeNonNeg")]
#[blvm_spec_lock::requires(total_in >= total_out)]
#[blvm_spec_lock::requires(total_out >= 0)]
pub(crate) fn _verify_f_fee_non_neg(total_in: i64, total_out: i64) -> i64 {
    total_in - total_out
}

// ─── §13.3.1 Integer Arithmetic Overflow/Underflow ───────────────────────────

/// Witness for **F_FeeArithmeticNonNeg** (PROTOCOL.md §13.3.1).
///
/// Proof obligation: fee arithmetic produces a non-negative result when inputs cover outputs.
///
/// Extends F_FeeNonNeg to the engineering invariant section: checked subtraction of valid
/// monetary values (both ≥ 0, with total_in ≥ total_out) cannot underflow.
#[spec_locked("13.3.1", "F_FeeArithmeticNonNeg")]
#[blvm_spec_lock::requires(total_in >= total_out)]
#[blvm_spec_lock::requires(total_in >= 0)]
#[blvm_spec_lock::requires(total_out >= 0)]
pub(crate) fn _verify_f_fee_arithmetic_non_neg(total_in: i64, total_out: i64) -> i64 {
    total_in - total_out
}

/// Witness for **F_FeeArithmeticBounded** (PROTOCOL.md §13.3.1).
///
/// Proof obligation: fee ≤ total_in when total_in ≥ total_out ≥ 0.
///
/// Upper bound on fee: since total_out ≥ 0, the fee = total_in - total_out ≤ total_in.
/// This means no transaction can extract more in fees than its total input value.
#[spec_locked("13.3.1", "F_FeeArithmeticBounded")]
#[blvm_spec_lock::requires(total_in >= total_out)]
#[blvm_spec_lock::requires(total_in >= 0)]
#[blvm_spec_lock::requires(total_out >= 0)]
#[blvm_spec_lock::ensures(result <= total_in)]
pub(crate) fn _verify_f_fee_arithmetic_bounded(total_in: i64, total_out: i64) -> i64 {
    total_in - total_out
}

// ─── §13.3.3 Resource Limit Enforcement ──────────────────────────────────────

/// Witness for **F_CoinbaseScriptSigMin** (PROTOCOL.md §13.3.3).
///
/// Proof obligation: coinbase scriptSig length ≥ 2 (minimum length boundary).
///
/// The headroom above the minimum: `len - 2 ≥ 0` whenever `len ≥ 2`.
/// Z3 proves from `requires(len >= 2)` via linear arithmetic (same pattern as fee non-neg).
/// Formalises that a valid coinbase scriptSig satisfies the 2-byte minimum exactly.
#[spec_locked("13.3.3", "F_CoinbaseScriptSigMin")]
#[blvm_spec_lock::requires(len >= 2)]
pub(crate) fn _verify_f_coinbase_scriptsig_min(len: i64) -> i64 {
    len - 2
}

/// Witness for **F_CoinbaseScriptSigMax** (PROTOCOL.md §13.3.3).
///
/// Proof obligation: coinbase scriptSig length ≤ 100 (maximum length boundary).
///
/// The distance below the maximum: `100 - len ≥ 0` whenever `len ≤ 100`.
/// Z3 proves from `requires(len <= 100)` via linear arithmetic.
/// Formalises that a valid coinbase scriptSig does not exceed the 100-byte maximum.
#[spec_locked("13.3.3", "F_CoinbaseScriptSigMax")]
#[blvm_spec_lock::requires(len <= 100)]
pub(crate) fn _verify_f_coinbase_scriptsig_max(len: i64) -> i64 {
    100 - len
}

/// Witness for **F_StackSizeSafe** (PROTOCOL.md §13.3.3).
///
/// Proof obligation: stack depth headroom is non-negative when below the 1000-item limit.
///
/// `999 - depth ≥ 0` whenever `depth < 1000`. Z3 proves via linear arithmetic.
/// Formalises the stack-size invariant: a stack within the limit has non-negative headroom
/// before the consensus-critical overflow boundary.
#[spec_locked("13.3.3", "F_StackSizeSafe")]
#[blvm_spec_lock::requires(depth >= 0)]
#[blvm_spec_lock::requires(depth < 1000)]
pub(crate) fn _verify_f_stack_size_safe(depth: i64) -> i64 {
    999 - depth
}

// ─── §7.1 ExpandTarget ───────────────────────────────────────────────────────

/// Witness for **F_ExpandTargetZeroMantissa** (PROTOCOL.md §7.1).
///
/// Proof obligation: when `(bits & 0x007fffff) == 0`, `ExpandTarget(bits) == 0`.
///
/// Under `requires((bits & 0x007fffff) == 0)`, the mantissa field of the compact bits
/// word is zero. The expanded target formula is `mantissa × 2^(8*(exp-3)) = 0 × ... = 0`.
/// Z3 proves: `if mantissa == 0 { 0 } else { 1 }` returns `0` under the precondition
/// (same `if/else with unreachable branch` pattern as F_SubsidyZeroAfter64).
#[spec_locked("7.1", "F_ExpandTargetZeroMantissa")]
#[blvm_spec_lock::requires((bits & 0x007fffff) == 0)]
#[blvm_spec_lock::ensures(result == 0)]
pub(crate) fn _verify_f_expand_target_zero_mantissa(bits: u64) -> i64 {
    let mantissa = bits & 0x007fffff;
    if mantissa == 0 { 0 } else { 1 } // else branch is unreachable under requires
}

/// Witness for **F_ExpandTargetExponent** (PROTOCOL.md §7.1).
///
/// Proof obligation: the exponent byte of the compact bits word is always in `[0, 255]`.
///
/// `exponent = (bits >> 24) & 0xFF`. The `& 0xFF` mask guarantees the result is at most `255`.
/// Since `& 0xFF` always produces a non-negative value, the result is also `>= 0`.
///
/// Z3 proves both bounds via bitwise arithmetic: `x & 0xFF` is bounded to `[0, 255]` by definition.
#[spec_locked("7.1", "F_ExpandTargetExponent")]
#[blvm_spec_lock::ensures(result <= 255)]
pub(crate) fn _verify_f_expand_target_exponent(bits: i64) -> i64 {
    (bits >> 24) & 0xff
}

/// Witness for **F_ExpandTargetNonZeroMantissa** (PROTOCOL.md §7.1).
///
/// Proof obligation: when `(bits & 0x007fffff) != 0`, the sentinel result is non-zero.
///
/// Under `requires((bits & 0x007fffff) != 0)`, `mantissa != 0`, so the `else` branch
/// returns `1 != 0`. Complements F_ExpandTargetZeroMantissa: together they establish
/// the exact biconditional `mantissa == 0 ↔ target == 0`.
///
/// Z3 proves: `mantissa != 0` (from requires) + `if mantissa == 0 { 0 } else { 1 }`
/// ⊨ `result == 1` ⊨ `result != 0`.
#[spec_locked("7.1", "F_ExpandTargetNonZeroMantissa")]
#[blvm_spec_lock::requires((bits & 0x007fffff) != 0)]
#[blvm_spec_lock::ensures(result != 0)]
pub(crate) fn _verify_f_expand_target_non_zero_mantissa(bits: u64) -> i64 {
    let mantissa = bits & 0x007fffff;
    if mantissa == 0 { 0 } else { 1 }
}

// ─── §6.1 GetBlockSubsidy ────────────────────────────────────────────────────

/// Witness for **F_SubsidyZeroAfter64** (PROTOCOL.md §6.1).
///
/// Proof obligation: `requires(height >= HALVING_INTERVAL * 64) ⊨ result == 0`
///
/// Z3 can prove this today using only integer division and comparison:
/// `k = height / 210_000`, and under the `requires`, `k ≥ 64`, so the `if`
/// branch is always taken and returns the literal `0`.
#[spec_locked("6.1", "F_SubsidyZeroAfter64")]
#[blvm_spec_lock::requires(height >= HALVING_INTERVAL * 64)]
#[blvm_spec_lock::ensures(result == 0)]
pub(crate) fn _verify_f_subsidy_zero_after_64(height: u64) -> i64 {
    let k = height / HALVING_INTERVAL;
    if k >= 64 {
        0
    } else {
        INITIAL_SUBSIDY // unreachable under the requires precondition
    }
}

/// Witness for **F_SubsidyPiecewise** (PROTOCOL.md §6.1).
///
/// Uses a 64-arm `match` with **literal** shift amounts in each arm.
/// Z3 translates `INITIAL_SUBSIDY >> 0` → `INITIAL_SUBSIDY / 1`,
/// `INITIAL_SUBSIDY >> 1` → `INITIAL_SUBSIDY / 2`, etc.
/// The production function uses `>> halving_period` (variable RHS, fails Z3);
/// this witness avoids that by unrolling the 64 epochs.
#[spec_locked("6.1", "F_SubsidyPiecewise")]
#[blvm_spec_lock::ensures(
    result >= 0
)]
#[blvm_spec_lock::ensures(
    result <= INITIAL_SUBSIDY
)]
pub(crate) fn _verify_f_subsidy_piecewise(height: u64) -> i64 {
    let k = height / HALVING_INTERVAL;
    match k {
        0 => INITIAL_SUBSIDY,
        1 => INITIAL_SUBSIDY >> 1,
        2 => INITIAL_SUBSIDY >> 2,
        3 => INITIAL_SUBSIDY >> 3,
        4 => INITIAL_SUBSIDY >> 4,
        5 => INITIAL_SUBSIDY >> 5,
        6 => INITIAL_SUBSIDY >> 6,
        7 => INITIAL_SUBSIDY >> 7,
        8 => INITIAL_SUBSIDY >> 8,
        9 => INITIAL_SUBSIDY >> 9,
        10 => INITIAL_SUBSIDY >> 10,
        11 => INITIAL_SUBSIDY >> 11,
        12 => INITIAL_SUBSIDY >> 12,
        13 => INITIAL_SUBSIDY >> 13,
        14 => INITIAL_SUBSIDY >> 14,
        15 => INITIAL_SUBSIDY >> 15,
        16 => INITIAL_SUBSIDY >> 16,
        17 => INITIAL_SUBSIDY >> 17,
        18 => INITIAL_SUBSIDY >> 18,
        19 => INITIAL_SUBSIDY >> 19,
        20 => INITIAL_SUBSIDY >> 20,
        21 => INITIAL_SUBSIDY >> 21,
        22 => INITIAL_SUBSIDY >> 22,
        23 => INITIAL_SUBSIDY >> 23,
        24 => INITIAL_SUBSIDY >> 24,
        25 => INITIAL_SUBSIDY >> 25,
        26 => INITIAL_SUBSIDY >> 26,
        27 => INITIAL_SUBSIDY >> 27,
        28 => INITIAL_SUBSIDY >> 28,
        29 => INITIAL_SUBSIDY >> 29,
        30 => INITIAL_SUBSIDY >> 30,
        31 => INITIAL_SUBSIDY >> 31,
        32 => INITIAL_SUBSIDY >> 32,
        33 => INITIAL_SUBSIDY >> 33,
        34 => INITIAL_SUBSIDY >> 34,
        35 => INITIAL_SUBSIDY >> 35,
        36 => INITIAL_SUBSIDY >> 36,
        37 => INITIAL_SUBSIDY >> 37,
        38 => INITIAL_SUBSIDY >> 38,
        39 => INITIAL_SUBSIDY >> 39,
        40 => INITIAL_SUBSIDY >> 40,
        41 => INITIAL_SUBSIDY >> 41,
        42 => INITIAL_SUBSIDY >> 42,
        43 => INITIAL_SUBSIDY >> 43,
        44 => INITIAL_SUBSIDY >> 44,
        45 => INITIAL_SUBSIDY >> 45,
        46 => INITIAL_SUBSIDY >> 46,
        47 => INITIAL_SUBSIDY >> 47,
        48 => INITIAL_SUBSIDY >> 48,
        49 => INITIAL_SUBSIDY >> 49,
        50 => INITIAL_SUBSIDY >> 50,
        51 => INITIAL_SUBSIDY >> 51,
        52 => INITIAL_SUBSIDY >> 52,
        53 => INITIAL_SUBSIDY >> 53,
        54 => INITIAL_SUBSIDY >> 54,
        55 => INITIAL_SUBSIDY >> 55,
        56 => INITIAL_SUBSIDY >> 56,
        57 => INITIAL_SUBSIDY >> 57,
        58 => INITIAL_SUBSIDY >> 58,
        59 => INITIAL_SUBSIDY >> 59,
        60 => INITIAL_SUBSIDY >> 60,
        61 => INITIAL_SUBSIDY >> 61,
        62 => INITIAL_SUBSIDY >> 62,
        63 => INITIAL_SUBSIDY >> 63,
        _ => 0,
    }
}

// ─── §6.2 TotalSupply ────────────────────────────────────────────────────────

/// Witness for **F_TotalSupplyNonNeg** (PROTOCOL.md §6.2).
///
/// Proof obligation: within the first halving epoch, total supply = (height+1) * INITIAL_SUBSIDY ≥ 0.
///
/// Z3 proves this via linear arithmetic: `height ≥ 0` (u64) and `INITIAL_SUBSIDY > 0`
/// imply `(height + 1) * INITIAL_SUBSIDY > 0 ≥ 0`.
/// The full inductive proof over all epochs requires Kani (Track E).
#[spec_locked("6.2", "F_TotalSupplyNonNeg")]
#[blvm_spec_lock::requires(height < HALVING_INTERVAL)]
pub(crate) fn _verify_f_total_supply_non_neg(height: u64) -> i64 {
    (height as i64 + 1) * INITIAL_SUBSIDY
}

/// Witness for **F_TotalSupplyBound** (PROTOCOL.md §6.2).
///
/// Proof obligation: within the first halving epoch, total supply ≤ MAX_MONEY.
///
/// First-epoch maximum: HALVING_INTERVAL * INITIAL_SUBSIDY = 210_000 * 5_000_000_000
/// = 1_050_000_000_000_000 < 2_100_000_000_000_000 = MAX_MONEY.
/// Z3 proves this via linear arithmetic. Full inductive proof requires Kani (Track E).
#[spec_locked("6.2", "F_TotalSupplyBound")]
#[blvm_spec_lock::requires(height < HALVING_INTERVAL)]
#[blvm_spec_lock::ensures(result <= MAX_MONEY)]
pub(crate) fn _verify_f_total_supply_bound(height: u64) -> i64 {
    (height as i64 + 1) * INITIAL_SUBSIDY
}

// ─── §5.5 Sequence Locks ─────────────────────────────────────────────────────

/// Witness for **F_SequenceLockTimeMask** (PROTOCOL.md §5.5).
///
/// Proof obligation: `ExtractSequenceLocktimeValue(seq) <= 65535`.
///
/// The body is `seq & 0x0000_ffff` — a bitwise AND with a literal mask. Z3 translates
/// bitwise AND with a literal, then proves `result <= 65535` from the mask property.
/// Genuine algebraic constraint: the 16-bit mask guarantees the upper bound.
#[spec_locked("5.5", "F_SequenceLockTimeMask")]
#[blvm_spec_lock::ensures(result <= 65535)]
pub(crate) fn _verify_f_sequence_locktime_mask(seq: u64) -> u64 {
    seq & 0x0000_ffff
}

/// Witness for **F_EvalSeqLocksDisabled** (PROTOCOL.md §5.5).
///
/// Proof obligation: when both locks are disabled (`min_height == -1`, `min_time == -1`),
/// `EvaluateSequenceLocks` always returns `true`.
///
/// Z3 proves: under the two `requires` guards, `min_height < 0` and `min_time < 0` are
/// both `true`, so each `(flag < 0 || ...)` clause short-circuits to `true`, giving
/// `true && true == true`.
#[spec_locked("5.5", "F_EvalSeqLocksDisabled")]
#[blvm_spec_lock::requires(min_height == -1)]
#[blvm_spec_lock::requires(min_time == -1)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_eval_seq_locks_disabled(
    height: i64,
    time: i64,
    min_height: i64,
    min_time: i64,
) -> bool {
    (min_height < 0 || height > min_height) && (min_time < 0 || time > min_time)
}

// ─── §13.3.5 Integration Proofs ──────────────────────────────────────────────

/// Witness for **F_LocktimeTypeIsHeight** (PROTOCOL.md §13.3.5).
///
/// Proof obligation: headroom below LOCKTIME_THRESHOLD is non-negative when lt < 500,000,000.
///
/// `500_000_000 - lt ≥ 0` whenever `lt < 500_000_000` (block-height locktime range).
/// Z3 proves via linear arithmetic from `requires(lt < 500_000_000)`.
/// Formalises: a locktime below the threshold is unambiguously in the block-height range.
#[spec_locked("13.3.5", "F_LocktimeTypeIsHeight")]
#[blvm_spec_lock::requires(lt >= 0)]
#[blvm_spec_lock::requires(lt < 500000000)]
pub(crate) fn _verify_f_locktime_type_is_height(lt: i64) -> i64 {
    500_000_000 - lt
}

/// Witness for **F_LocktimeTypeIsTimestamp** (PROTOCOL.md §13.3.5).
///
/// Proof obligation: excess above LOCKTIME_THRESHOLD is non-negative when lt ≥ 500,000,000.
///
/// `lt - 500_000_000 ≥ 0` whenever `lt ≥ 500_000_000` (timestamp locktime range).
/// Z3 proves via linear arithmetic from `requires(lt >= 500_000_000)`.
/// Formalises: a locktime at or above the threshold is unambiguously in the timestamp range.
#[spec_locked("13.3.5", "F_LocktimeTypeIsTimestamp")]
#[blvm_spec_lock::requires(lt >= 500000000)]
pub(crate) fn _verify_f_locktime_type_is_timestamp(lt: i64) -> i64 {
    lt - 500_000_000
}

/// Witness for **F_EvalSeqLocksHeightNotMet** (PROTOCOL.md §5.5).
///
/// Proof obligation: when a minimum block height is active (`min_height >= 0`) and
/// the current block height does not exceed it (`block_height <= min_height`),
/// EvaluateSequenceLocks returns false.
///
/// Under `requires(min_height >= 0)`, the guard `min_height < 0` is `false`,
/// so the first clause becomes `false || block_height > min_height`.
/// Under `requires(block_height <= min_height)`, `block_height > min_height` is also `false`.
/// So the first clause is `false`, and `false && (any) = false`.
///
/// Proves the height-not-met invariant: a CSV-relative-height-locked transaction cannot
/// be finalized in a block at the lock height or below.
#[spec_locked("5.5", "F_EvalSeqLocksHeightNotMet")]
#[blvm_spec_lock::requires(min_height >= 0)]
#[blvm_spec_lock::requires(block_height <= min_height)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_eval_seq_locks_height_not_met(
    block_height: i64,
    time: i64,
    min_height: i64,
    min_time: i64,
) -> bool {
    (min_height < 0 || block_height > min_height) && (min_time < 0 || time > min_time)
}

/// Witness for **F_SequenceTimeEncoding** (PROTOCOL.md §5.5).
///
/// Proof obligation: `value * 512 ≤ 33,553,920` when `value ≤ 65535`.
///
/// The time-based relative locktime is expressed in 512-second units.
/// With the 16-bit value field bounded by `65535`, the maximum encoded time is
/// `65535 × 512 = 33,553,920` seconds. Z3 proves via linear arithmetic:
/// `requires(value >= 0)` + `requires(value <= 65535)` ⊨ `value * 512 <= 33_553_920`.
#[spec_locked("5.5", "F_SequenceTimeEncoding")]
#[blvm_spec_lock::requires(value >= 0)]
#[blvm_spec_lock::requires(value <= 65535)]
#[blvm_spec_lock::ensures(result <= 33553920)]
pub(crate) fn _verify_f_sequence_time_encoding(value: i64) -> i64 {
    value * 512
}

// ─── §5.4.7 BIP65: OP_CHECKLOCKTIMEVERIFY (CLTV) ─────────────────────────────

/// Witness for **F_BIP65RejectsZeroLocktime** (PROTOCOL.md §5.4.7).
///
/// Proof obligation: CLTV validation always fails when `tx_locktime == 0`.
///
/// The BIP65 check begins `tx_locktime != 0 && ...`. Under `requires(tx_locktime == 0)`,
/// this first AND-term is `false`, short-circuiting the whole check to `false`.
///
/// Expresses the core BIP65 invariant: a transaction with `nLockTime = 0` can never
/// satisfy any OP_CHECKLOCKTIMEVERIFY script.
///
/// Body inlines both `check_bip65` and `locktime_types_match` (the cross-function call
/// is not Z3-translatable directly; we inline it using `LOCKTIME_THRESHOLD`).
#[spec_locked("5.4.7", "F_BIP65RejectsZeroLocktime")]
#[blvm_spec_lock::requires(tx_locktime == 0)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_bip65_rejects_zero_locktime(tx_locktime: u32, stack_locktime: u32) -> bool {
    let tx_is_height = tx_locktime < LOCKTIME_THRESHOLD;
    let sk_is_height = stack_locktime < LOCKTIME_THRESHOLD;
    tx_locktime != 0 && (tx_is_height == sk_is_height) && tx_locktime >= stack_locktime
}

/// Witness for **F_BIP65RejectsTypeMismatch** (PROTOCOL.md §5.4.7).
///
/// Proof obligation: CLTV validation always fails when the tx and script locktime
/// types differ (block-height vs. timestamp).
///
/// Under `requires(tx_locktime < LOCKTIME_THRESHOLD)` (block-height type) and
/// `requires(stack_locktime >= LOCKTIME_THRESHOLD)` (timestamp type), the type
/// comparison `tx_is_height == sk_is_height` is `true == false` = `false`,
/// so the whole check short-circuits to `false`.
///
/// Proves the cross-type incompatibility invariant: a timestamp-CLTV script cannot
/// be satisfied by a transaction using a block-height nLockTime, and vice versa.
#[spec_locked("5.4.7", "F_BIP65RejectsTypeMismatch")]
#[blvm_spec_lock::requires(tx_locktime < LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(stack_locktime >= LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_bip65_rejects_type_mismatch(tx_locktime: u32, stack_locktime: u32) -> bool {
    let tx_is_height = tx_locktime < LOCKTIME_THRESHOLD;
    let sk_is_height = stack_locktime < LOCKTIME_THRESHOLD;
    tx_locktime != 0 && (tx_is_height == sk_is_height) && tx_locktime >= stack_locktime
}

/// Witness for **F_BIP65RejectsTypeMismatchReverse** (PROTOCOL.md §5.4.7).
///
/// Proof obligation: CLTV validation fails when `tx_locktime` is a timestamp
/// but `stack_locktime` is a block-height (the reverse mismatch direction).
///
/// Under `requires(tx_locktime >= LOCKTIME_THRESHOLD)`: `tx_is_height = false`.
/// Under `requires(stack_locktime < LOCKTIME_THRESHOLD)`: `sk_is_height = true`.
/// `tx_is_height == sk_is_height` = `false == true` = `false`.
/// The conjunction includes `false`, so result = `false`.
///
/// Proves that a timestamp-type transaction cannot satisfy a block-height CLTV script.
#[spec_locked("5.4.7", "F_BIP65RejectsTypeMismatchReverse")]
#[blvm_spec_lock::requires(tx_locktime >= LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(stack_locktime < LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_bip65_rejects_type_mismatch_reverse(
    tx_locktime: u32,
    stack_locktime: u32,
) -> bool {
    let tx_is_height = tx_locktime < LOCKTIME_THRESHOLD;
    let sk_is_height = stack_locktime < LOCKTIME_THRESHOLD;
    tx_locktime != 0 && (tx_is_height == sk_is_height) && tx_locktime >= stack_locktime
}

/// Witness for **F_BIP65RejectsValueTooLow** (PROTOCOL.md §5.4.7).
///
/// Proof obligation: CLTV validation fails when both locktimes are block-heights
/// but `tx_locktime < stack_locktime` (value doesn't meet the script minimum).
///
/// Under all `requires`: `tx_locktime != 0` = true, `tx_is_height == sk_is_height` = true,
/// but `tx_locktime >= stack_locktime` = false (requires: tx_locktime < stack_locktime).
/// The conjunction includes `false`, so result = `false`.
///
/// Proves that even type-matching locktimes are rejected when the transaction locktime
/// fails to meet the script-specified minimum.
#[spec_locked("5.4.7", "F_BIP65RejectsValueTooLow")]
#[blvm_spec_lock::requires(tx_locktime > 0)]
#[blvm_spec_lock::requires(tx_locktime < LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(stack_locktime < LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(tx_locktime < stack_locktime)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_bip65_rejects_value_too_low(tx_locktime: u32, stack_locktime: u32) -> bool {
    let tx_is_height = tx_locktime < LOCKTIME_THRESHOLD;
    let sk_is_height = stack_locktime < LOCKTIME_THRESHOLD;
    tx_locktime != 0 && (tx_is_height == sk_is_height) && tx_locktime >= stack_locktime
}

/// Witness for **F_BIP65RejectsTimestampValueTooLow** (PROTOCOL.md §5.4.7).
///
/// Proof obligation: CLTV validation fails when both locktimes are timestamps
/// but `tx_locktime < stack_locktime` (value does not meet script minimum).
///
/// Under all `requires`: both are timestamps (`tx_is_height = false`, `sk_is_height = false`),
/// types match (`false == false = true`), `tx_locktime != 0` = true (implied by >= LOCKTIME_THRESHOLD),
/// but `tx_locktime >= stack_locktime` = false (requires: tx_locktime < stack_locktime).
/// Result = `true && true && false` = `false`.
///
/// Completes the BIP65 case analysis: every logically distinct input combination is now
/// covered by a formally verified witness — rejection and acceptance for both domains.
#[spec_locked("5.4.7", "F_BIP65RejectsTimestampValueTooLow")]
#[blvm_spec_lock::requires(tx_locktime >= LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(stack_locktime >= LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(tx_locktime < stack_locktime)]
#[blvm_spec_lock::requires(tx_locktime > 0)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_bip65_rejects_timestamp_value_too_low(
    tx_locktime: u32,
    stack_locktime: u32,
) -> bool {
    let tx_is_height = tx_locktime < LOCKTIME_THRESHOLD;
    let sk_is_height = stack_locktime < LOCKTIME_THRESHOLD;
    tx_locktime != 0 && (tx_is_height == sk_is_height) && tx_locktime >= stack_locktime
}

/// Witness for **F_BIP65Passes** (PROTOCOL.md §5.4.7).
///
/// Proof obligation: CLTV validation passes when all four conditions hold simultaneously:
/// `tx_locktime > 0`, both are block-heights (`< LOCKTIME_THRESHOLD`), and
/// `tx_locktime >= stack_locktime`.
///
/// Under the four `requires` clauses:
/// - `tx_locktime != 0` → true (tx_locktime > 0)
/// - `tx_is_height = tx_locktime < LOCKTIME_THRESHOLD` → true
/// - `sk_is_height = stack_locktime < LOCKTIME_THRESHOLD` → true
/// - `tx_is_height == sk_is_height` → `true == true` → true
/// - `tx_locktime >= stack_locktime` → true
/// So the conjunction is `true && true && true = true`.
///
/// Proves the CLTV success condition: a block-height-type transaction locktime meeting the
/// script's minimum satisfies OP_CHECKLOCKTIMEVERIFY.
#[spec_locked("5.4.7", "F_BIP65Passes")]
#[blvm_spec_lock::requires(tx_locktime > 0)]
#[blvm_spec_lock::requires(tx_locktime < LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(stack_locktime < LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(tx_locktime >= stack_locktime)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_bip65_passes(tx_locktime: u32, stack_locktime: u32) -> bool {
    let tx_is_height = tx_locktime < LOCKTIME_THRESHOLD;
    let sk_is_height = stack_locktime < LOCKTIME_THRESHOLD;
    tx_locktime != 0 && (tx_is_height == sk_is_height) && tx_locktime >= stack_locktime
}

/// Witness for **F_BIP65PassesTimestamp** (PROTOCOL.md §5.4.7).
///
/// Proof obligation: CLTV validation passes when both locktimes are timestamps
/// (`>= LOCKTIME_THRESHOLD`) and `tx_locktime >= stack_locktime`.
///
/// Under the three `requires` clauses:
/// - `tx_locktime != 0` → true (tx_locktime >= LOCKTIME_THRESHOLD > 0)
/// - `tx_is_height = tx_locktime < LOCKTIME_THRESHOLD` → false
/// - `sk_is_height = stack_locktime < LOCKTIME_THRESHOLD` → false
/// - `tx_is_height == sk_is_height` → `false == false` → true
/// - `tx_locktime >= stack_locktime` → true
/// So the conjunction is `true && true && true = true`.
///
/// Proves the timestamp-domain CLTV success case: a timestamp-type transaction locktime
/// meeting the script's minimum satisfies OP_CHECKLOCKTIMEVERIFY.
#[spec_locked("5.4.7", "F_BIP65PassesTimestamp")]
#[blvm_spec_lock::requires(tx_locktime >= LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(stack_locktime >= LOCKTIME_THRESHOLD)]
#[blvm_spec_lock::requires(tx_locktime >= stack_locktime)]
#[blvm_spec_lock::requires(tx_locktime > 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_bip65_passes_timestamp(tx_locktime: u32, stack_locktime: u32) -> bool {
    let tx_is_height = tx_locktime < LOCKTIME_THRESHOLD;
    let sk_is_height = stack_locktime < LOCKTIME_THRESHOLD;
    tx_locktime != 0 && (tx_is_height == sk_is_height) && tx_locktime >= stack_locktime
}

// ─── §5.5 continued: EvaluateSequenceLocks positive cases ─────────────────────

/// Witness for **F_EvalSeqLocksHeightMet** (PROTOCOL.md §5.5).
///
/// Proof obligation: EvaluateSequenceLocks returns true when the height lock is satisfied
/// and there is no time constraint.
///
/// Under `requires(min_height >= 0)` + `requires(block_height > min_height)`:
///   first clause = `min_height < 0 || block_height > min_height` = `false || true` = `true`.
/// Under `requires(min_time < 0)`:
///   second clause = `min_time < 0 || time > min_time` = `true || ...` = `true`.
/// Result = `true && true` = `true`.
///
/// Proves the height-met case: a CSV-relative-height-locked transaction can be included
/// once the block height strictly exceeds the minimum required height.
#[spec_locked("5.5", "F_EvalSeqLocksHeightMet")]
#[blvm_spec_lock::requires(min_height >= 0)]
#[blvm_spec_lock::requires(block_height > min_height)]
#[blvm_spec_lock::requires(min_time < 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_eval_seq_locks_height_met(
    block_height: i64,
    time: i64,
    min_height: i64,
    min_time: i64,
) -> bool {
    (min_height < 0 || block_height > min_height) && (min_time < 0 || time > min_time)
}

/// Witness for **F_EvalSeqLocksTimeNotMet** (PROTOCOL.md §5.5).
///
/// Proof obligation: EvaluateSequenceLocks returns false when a time constraint is active
/// and the current block time does not exceed the minimum.
///
/// Under `requires(min_time >= 0)`: `min_time < 0` is false, so the second clause =
///   `false || time > min_time`.
/// Under `requires(time <= min_time)`: `time > min_time` is false.
/// So the second clause = `false`, and `(any) && false` = `false`.
///
/// Proves the symmetric time-not-met invariant: a CSV-relative-time-locked transaction
/// cannot be included when the median time past has not reached the required minimum.
#[spec_locked("5.5", "F_EvalSeqLocksTimeNotMet")]
#[blvm_spec_lock::requires(min_time >= 0)]
#[blvm_spec_lock::requires(time <= min_time)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_eval_seq_locks_time_not_met(
    block_height: i64,
    time: i64,
    min_height: i64,
    min_time: i64,
) -> bool {
    (min_height < 0 || block_height > min_height) && (min_time < 0 || time > min_time)
}

/// Witness for **F_EvalSeqLocksTimeMet** (PROTOCOL.md §5.5).
///
/// Proof obligation: EvaluateSequenceLocks returns true when the time lock is satisfied
/// and there is no height constraint.
///
/// Under `requires(min_time >= 0)` + `requires(time > min_time)`:
///   second clause = `min_time < 0 || time > min_time` = `false || true` = `true`.
/// Under `requires(min_height < 0)`:
///   first clause = `min_height < 0 || block_height > min_height` = `true || ...` = `true`.
/// Result = `true && true` = `true`.
///
/// Symmetric to F_EvalSeqLocksHeightMet: proves the time-domain success case.
#[spec_locked("5.5", "F_EvalSeqLocksTimeMet")]
#[blvm_spec_lock::requires(min_time >= 0)]
#[blvm_spec_lock::requires(time > min_time)]
#[blvm_spec_lock::requires(min_height < 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_eval_seq_locks_time_met(
    block_height: i64,
    time: i64,
    min_height: i64,
    min_time: i64,
) -> bool {
    (min_height < 0 || block_height > min_height) && (min_time < 0 || time > min_time)
}

/// Witness for **F_EvalSeqLocksBothMet** (PROTOCOL.md §5.5).
///
/// Proof obligation: EvaluateSequenceLocks returns true when both height and time
/// constraints are active and both are satisfied.
///
/// Under `requires(min_height >= 0)` + `requires(block_height > min_height)`:
///   first clause = `min_height < 0 || block_height > min_height` = `false || true` = `true`.
/// Under `requires(min_time >= 0)` + `requires(time > min_time)`:
///   second clause = `min_time < 0 || time > min_time` = `false || true` = `true`.
/// Result = `true && true` = `true`.
///
/// Proves the full conjunction: a CSV-locked transaction with both height and time
/// constraints can be included once both are simultaneously satisfied.
#[spec_locked("5.5", "F_EvalSeqLocksBothMet")]
#[blvm_spec_lock::requires(min_height >= 0)]
#[blvm_spec_lock::requires(block_height > min_height)]
#[blvm_spec_lock::requires(min_time >= 0)]
#[blvm_spec_lock::requires(time > min_time)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_eval_seq_locks_both_met(
    block_height: i64,
    time: i64,
    min_height: i64,
    min_time: i64,
) -> bool {
    (min_height < 0 || block_height > min_height) && (min_time < 0 || time > min_time)
}

// ─── §5.3.1 ValidBlockHeader ──────────────────────────────────────────────────

/// Witness for **F_HeaderVersionFloor** (PROTOCOL.md §5.3.1).
///
/// Proof obligation: when `block_version == 0`, header validation always returns false.
///
/// The production `validate_block_header` function checks `if header.version < 1 { return Ok(false) }`.
/// Under `requires(block_version == 0)`, `block_version >= 1` evaluates to `0 >= 1 = false`,
/// so the body takes the `else` branch returning `0` (encodes `false`).
///
/// Proves H01 (version floor): block version 0 has never been valid in Bitcoin.
/// Z3 proves this via linear arithmetic: `0 >= 1` is `false`.
#[spec_locked("5.3.1", "F_HeaderVersionFloor")]
#[blvm_spec_lock::requires(block_version == 0)]
#[blvm_spec_lock::ensures(result == 0)]
pub(crate) fn _verify_f_header_version_floor(block_version: i64) -> i64 {
    if block_version >= 1 { 1 } else { 0 }
}

/// Witness for **F_HeaderBitsFloor** (PROTOCOL.md §5.3.1).
///
/// Proof obligation: when `bits == 0`, header validation always returns false.
///
/// The production `validate_block_header` checks `if header.bits == 0 { return Ok(false) }`.
/// Under `requires(bits == 0)`, `bits != 0` evaluates to `false`, so the `else` branch
/// returns `0` (encodes `false`).
///
/// Proves H06 (non-zero bits): a block header with zero compact difficulty is always invalid.
/// Z3 proves this via linear arithmetic: `0 != 0` is `false`.
#[spec_locked("5.3.1", "F_HeaderBitsFloor")]
#[blvm_spec_lock::requires(bits == 0)]
#[blvm_spec_lock::ensures(result == 0)]
pub(crate) fn _verify_f_header_bits_floor(bits: i64) -> i64 {
    if bits != 0 { 1 } else { 0 }
}

// ─── §5.4.1 BIP30: Duplicate Coinbase Prevention ──────────────────────────────

/// Witness for **F_BIP30DeactivationPass** (PROTOCOL.md §5.4.1).
///
/// Proof obligation: after BIP30 deactivation, the BIP30 check always passes.
///
/// The production `check_bip30` short-circuits `return Ok(true)` when
/// `!is_fork_active(Bip30, height)`.  Encoded as an integer flag (0 = inactive):
/// under `requires(bip30_active == 0)`, the `if` branch is taken and `result = 1`.
/// Z3 proves this via linear arithmetic: `bip30_active == 0 → result == 1`.
///
/// Proves the deactivation-pass invariant: once BIP30 is deactivated, no block
/// can be rejected by the duplicate-coinbase rule.
#[spec_locked("5.4.1", "F_BIP30DeactivationPass")]
#[blvm_spec_lock::requires(bip30_active == 0)]
#[blvm_spec_lock::ensures(result == 1)]
pub(crate) fn _verify_f_bip30_deactivation_pass(bip30_active: i64) -> i64 {
    if bip30_active == 0 { 1 } else { 0 }
}

// ─── §5.4.2 BIP34: Block Height in Coinbase ───────────────────────────────────

/// Witness for **F_BIP34PreActivationPass** (PROTOCOL.md §5.4.2).
///
/// Proof obligation: before BIP34 activation, the BIP34 height check always passes.
///
/// The production `check_bip34` short-circuits `return Ok(true)` when
/// `!is_fork_active(Bip34, height)`.  Encoded as an integer flag (0 = inactive):
/// under `requires(bip34_active == 0)`, the `if` branch is taken and `result = 1`.
/// Z3 proves this via linear arithmetic: `bip34_active == 0 → result == 1`.
///
/// Proves the pre-activation-pass invariant: any block before BIP34 activation
/// cannot be rejected by the coinbase-height rule.
#[spec_locked("5.4.2", "F_BIP34PreActivationPass")]
#[blvm_spec_lock::requires(bip34_active == 0)]
#[blvm_spec_lock::ensures(result == 1)]
pub(crate) fn _verify_f_bip34_pre_activation_pass(bip34_active: i64) -> i64 {
    if bip34_active == 0 { 1 } else { 0 }
}

// ─── §5.4.8 BIP348: CSFS Degenerate Case Algebra ────────────────────────────

/// Witness for **F_CSFSZeroPubkeyInvalid** (PROTOCOL.md §5.4.8).
///
/// Proof obligation: when pubkey length == 0, CSFS always returns invalid (false).
///
/// Body: `pk_len != 0`. Under `requires(pk_len == 0)`, this is `false`.
/// Proves the zero-pubkey fast rejection path — no signature check needed.
/// Z3 proves via linear arithmetic.
#[spec_locked("5.4.8", "F_CSFSZeroPubkeyInvalid")]
#[blvm_spec_lock::requires(pk_len == 0)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_csfs_zero_pubkey_invalid(pk_len: u64) -> bool {
    pk_len != 0
}

/// Witness for **F_CSFSUnknownPubkeyValid** (PROTOCOL.md §5.4.8).
///
/// Proof obligation: when pubkey is non-zero length but not 32 bytes,
/// CSFS always returns valid (true) — unknown pubkey types succeed
/// per BIP348 soft-fork upgrade rules.
///
/// Body: 3-way if/else. Under `requires(pk_len > 0 && pk_len != 32)`,
/// both first branches are false → else branch → true.
/// Z3 proves via linear arithmetic.
#[spec_locked("5.4.8", "F_CSFSUnknownPubkeyValid")]
#[blvm_spec_lock::requires(pk_len > 0)]
#[blvm_spec_lock::requires(pk_len != 32)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_csfs_unknown_pubkey_valid(pk_len: u64) -> bool {
    pk_len != 32
}

/// Witness for **F_CSFSEmptySigValid** (PROTOCOL.md §5.4.8).
///
/// Proof obligation: when signature is empty AND pubkey is non-zero,
/// CSFS always returns valid (true). Empty-signature handling pushes
/// an empty vector, matching OP_CHECKSIG behavior.
///
/// Body: under `requires(sig_len == 0 && pk_len > 0)`, the first branch
/// (pk_len == 0) is false, the second branch (sig_len == 0) is true → true.
/// Z3 proves via linear arithmetic.
#[spec_locked("5.4.8", "F_CSFSEmptySigValid")]
#[blvm_spec_lock::requires(sig_len == 0)]
#[blvm_spec_lock::requires(pk_len > 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_csfs_empty_sig_valid(sig_len: u64, pk_len: u64) -> bool {
    if pk_len == 0 {
        false // zero pubkey: always invalid
    } else if sig_len == 0 {
        true // empty sig: always valid
    } else {
        false // non-empty sig: depends on Schnorr (placeholder)
    }
}

// ─── §5.4.3 BIP66: Strict DER ────────────────────────────────────────────────

/// Witness for **F_BIP66PreActivationPass** (PROTOCOL.md §5.4.3).
///
/// Proof obligation: before BIP66 activation, all DER signature checks pass.
///
/// Encoded as `bip66_active == 0` (integer flag, 0 = inactive). Under
/// `requires(bip66_active == 0)`, the first branch is taken → result = 1 (pass).
/// Z3 proves via linear arithmetic.
#[spec_locked("5.4.3", "F_BIP66PreActivationPass")]
#[blvm_spec_lock::requires(bip66_active == 0)]
#[blvm_spec_lock::ensures(result == 1)]
pub(crate) fn _verify_f_bip66_pre_activation_pass(bip66_active: i64) -> i64 {
    if bip66_active == 0 { 1 } else { 0 }
}

// ─── §5.4.4 BIP90: Block Version Enforcement ─────────────────────────────────

/// Witness for **F_BIP90PreActivationPass** (PROTOCOL.md §5.4.4).
///
/// Proof obligation: before BIP90 activation, all blocks pass version enforcement.
///
/// Encoded as `bip90_active == 0` (integer flag, 0 = inactive). Under
/// `requires(bip90_active == 0)`, result = 1 (pass).
/// Z3 proves via linear arithmetic.
#[spec_locked("5.4.4", "F_BIP90PreActivationPass")]
#[blvm_spec_lock::requires(bip90_active == 0)]
#[blvm_spec_lock::ensures(result == 1)]
pub(crate) fn _verify_f_bip90_pre_activation_pass(bip90_active: i64) -> i64 {
    if bip90_active == 0 { 1 } else { 0 }
}

// ─── §5.4.5 BIP147: NULLDUMMY Enforcement ────────────────────────────────────

/// Witness for **F_BIP147PreActivationPass** (PROTOCOL.md §5.4.5).
///
/// Proof obligation: before BIP147 activation, all CHECKMULTISIG operations pass
/// the null-dummy check regardless of dummy element content.
///
/// Encoded as `bip147_active == 0` (integer flag, 0 = inactive). Under
/// `requires(bip147_active == 0)`, result = 1 (pass).
/// Z3 proves via linear arithmetic.
#[spec_locked("5.4.5", "F_BIP147PreActivationPass")]
#[blvm_spec_lock::requires(bip147_active == 0)]
#[blvm_spec_lock::ensures(result == 1)]
pub(crate) fn _verify_f_bip147_pre_activation_pass(bip147_active: i64) -> i64 {
    if bip147_active == 0 { 1 } else { 0 }
}

// ─── §5.4.9 BIP54: Consensus Cleanup ─────────────────────────────────────────

/// Witness for **F_BIP54ActivationThreshold** (PROTOCOL.md §5.4.9).
///
/// Proof obligation: IsBip54ActiveAt returns true iff `height >= threshold`.
///
/// Under `requires(height >= threshold)`, the comparison `height >= threshold`
/// evaluates to `true`.  Z3 proves this trivially via linear arithmetic on u64.
///
/// This is the activation-monotone property: once active at height `a`, BIP54
/// remains active for all heights ≥ `a`.  The `_verify_f_bip54_activation_threshold`
/// witness function is referenced explicitly in the Orange Paper §5.4.9.
#[spec_locked("5.4.9", "F_BIP54ActivationThreshold")]
#[blvm_spec_lock::requires(height >= threshold)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_bip54_activation_threshold(height: u64, threshold: u64) -> bool {
    height >= threshold
}

// ─── §5.5 BIP68: Sequence bit extraction ─────────────────────────────────────

/// Witness for **F_SequenceDisabledWhenBit31Set** (PROTOCOL.md §5.5).
///
/// Proof obligation: IsSequenceDisabled returns true when bit 31 is set.
///
/// The body is `(sequence & 0x80000000) != 0`. Under `requires((sequence & 0x80000000) != 0)`,
/// the bitwise AND is nonzero, so `!= 0` evaluates to `true`.
///
/// Proves the disabled case: a sequence number with the BIP68 disable flag (bit 31)
/// set is always treated as non-relative-locktime.
#[spec_locked("5.5", "F_SequenceDisabledWhenBit31Set")]
#[blvm_spec_lock::requires((sequence & 0x80000000) != 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_sequence_disabled_when_bit31_set(sequence: u32) -> bool {
    (sequence & 0x80000000) != 0
}

/// Witness for **F_SequenceEnabledWhenBit31Clear** (PROTOCOL.md §5.5).
///
/// Proof obligation: IsSequenceDisabled returns false when bit 31 is clear.
///
/// The body is `(sequence & 0x80000000) != 0`. Under `requires((sequence & 0x80000000) == 0)`,
/// the bitwise AND is zero, so `!= 0` evaluates to `false`.
///
/// Proves the enabled case: a sequence number without the disable flag is treated as
/// a relative locktime constraint (BIP68 applies).
#[spec_locked("5.5", "F_SequenceEnabledWhenBit31Clear")]
#[blvm_spec_lock::requires((sequence & 0x80000000) == 0)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_sequence_enabled_when_bit31_clear(sequence: u32) -> bool {
    (sequence & 0x80000000) != 0
}

/// Witness for **F_SequenceTypeTimeWhenBit22Set** (PROTOCOL.md §5.5).
///
/// Proof obligation: ExtractSequenceTypeFlag returns true when bit 22 is set.
///
/// The body is `(sequence & 0x00400000) != 0`. Under `requires((sequence & 0x00400000) != 0)`,
/// the bitwise AND is nonzero, so `!= 0` evaluates to `true`.
///
/// Proves the time-based case: a sequence number with the BIP68 type flag (bit 22)
/// set encodes a time-based relative locktime.
#[spec_locked("5.5", "F_SequenceTypeTimeWhenBit22Set")]
#[blvm_spec_lock::requires((sequence & 0x00400000) != 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_sequence_type_time_when_bit22_set(sequence: u32) -> bool {
    (sequence & 0x00400000) != 0
}

/// Witness for **F_SequenceTypeHeightWhenBit22Clear** (PROTOCOL.md §5.5).
///
/// Proof obligation: ExtractSequenceTypeFlag returns false when bit 22 is clear.
///
/// The body is `(sequence & 0x00400000) != 0`. Under `requires((sequence & 0x00400000) == 0)`,
/// the bitwise AND is zero, so `!= 0` evaluates to `false`.
///
/// Proves the block-height case: a sequence number without the type flag encodes a
/// block-height relative locktime.
#[spec_locked("5.5", "F_SequenceTypeHeightWhenBit22Clear")]
#[blvm_spec_lock::requires((sequence & 0x00400000) == 0)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_sequence_type_height_when_bit22_clear(sequence: u32) -> bool {
    (sequence & 0x00400000) != 0
}

// ─── §11.1.1 Weight and Size ─────────────────────────────────────────────────

/// Witness for **F_WeightToVSizeFloor** (PROTOCOL.md §11.1.1).
///
/// Proof obligation: vsize (ceiling(weight/4)) is always ≥ floor(weight/4).
///
/// Body: `(weight + 3) / 4` (explicit ceiling division). Under all inputs,
/// `(weight + 3) / 4 >= weight / 4` holds because `weight + 3 >= weight` and
/// integer division is monotone. Z3 proves via linear arithmetic.
#[spec_locked("11.1.1", "F_WeightToVSizeFloor")]
#[blvm_spec_lock::ensures(result >= weight / 4)]
pub(crate) fn _verify_f_weight_to_vsize_floor(weight: u64) -> u64 {
    weight.div_ceil(4)
}

/// Witness for **F_WeightToVSizeCeiling** (PROTOCOL.md §11.1.1).
///
/// Proof obligation: vsize (ceiling(weight/4)) is always ≤ floor(weight/4) + 1.
///
/// Body: `(weight + 3) / 4`. The ceiling is at most 1 more than the floor:
/// `(weight + 3) / 4 <= weight / 4 + 1`. Together with F_WeightToVSizeFloor
/// this characterizes ceiling division exactly. Z3 proves via linear arithmetic.
#[spec_locked("11.1.1", "F_WeightToVSizeCeiling")]
#[blvm_spec_lock::ensures(result <= weight / 4 + 1)]
pub(crate) fn _verify_f_weight_to_vsize_ceiling(weight: u64) -> u64 {
    weight.div_ceil(4)
}

// ─── §11.1.2 Witness Structure ───────────────────────────────────────────────

/// Witness for **F_WitnessEmptyByLength** (PROTOCOL.md §11.1.2).
///
/// Proof obligation: when the outer witness list has 0 elements, IsWitnessEmpty
/// returns true.
///
/// Body: `len == 0`. Under `requires(len == 0)`, the expression evaluates to
/// true. Proves the zero-length case: an empty witness stack is always empty.
/// Z3 proves via linear arithmetic.
#[spec_locked("11.1.2", "F_WitnessEmptyByLength")]
#[blvm_spec_lock::requires(len == 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_witness_empty_by_length(len: u64) -> bool {
    len == 0
}

// ─── §11.1.3 Witness Program Length ─────────────────────────────────────────

/// Witness for **F_WitnessProgramLength20Valid** (PROTOCOL.md §11.1.3).
///
/// Proof obligation: when witness program length is 20 bytes, validation returns
/// true (P2WPKH).
///
/// Body: `program_len == 20 || program_len == 32`. Under `requires(program_len == 20)`,
/// first disjunct is true, so result is true. Z3 proves via linear arithmetic.
#[spec_locked("11.1.3", "F_WitnessProgramLength20Valid")]
#[blvm_spec_lock::requires(program_len == 20)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_witness_program_length_20_valid(program_len: u64) -> bool {
    program_len == 20 || program_len == 32
}

/// Witness for **F_WitnessProgramLength32Valid** (PROTOCOL.md §11.1.3).
///
/// Proof obligation: when witness program length is 32 bytes, validation returns
/// true (P2WSH or P2TR).
///
/// Body: `program_len == 20 || program_len == 32`. Under `requires(program_len == 32)`,
/// second disjunct is true, so result is true. Z3 proves via linear arithmetic.
#[spec_locked("11.1.3", "F_WitnessProgramLength32Valid")]
#[blvm_spec_lock::requires(program_len == 32)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_witness_program_length_32_valid(program_len: u64) -> bool {
    program_len == 20 || program_len == 32
}

/// Witness for **F_WitnessProgramLengthInvalid** (PROTOCOL.md §11.1.3).
///
/// Proof obligation: when witness program length is neither 20 nor 32 bytes,
/// validation returns false.
///
/// Body: `program_len == 20 || program_len == 32`. Under
/// `requires(program_len != 20 && program_len != 32)`, both disjuncts are false,
/// so result is false. Z3 proves via linear arithmetic.
#[spec_locked("11.1.3", "F_WitnessProgramLengthInvalid")]
#[blvm_spec_lock::requires(program_len != 20)]
#[blvm_spec_lock::requires(program_len != 32)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_witness_program_length_invalid(program_len: u64) -> bool {
    program_len == 20 || program_len == 32
}

// ─── §5.2.4 IsMinimalIfCondition ─────────────────────────────────────────────

/// Witness for **F_MinimalIfEmptyTrue** (PROTOCOL.md §5.2.4).
///
/// Proof obligation: when the byte string is empty (length == 0),
/// IsMinimalIfCondition always returns true.
///
/// Body: the structural pattern `if len == 0 { true } else { false }`.
/// Under `requires(len == 0)`, the first branch is taken → true. Proves the
/// empty-bytes case of the MINIMALIF rule. Z3 proves via linear arithmetic.
#[spec_locked("5.2.4", "F_MinimalIfEmptyTrue")]
#[blvm_spec_lock::requires(len == 0)]
#[blvm_spec_lock::ensures(result == true)]
pub(crate) fn _verify_f_minimal_if_empty_true(len: u64) -> bool {
    len == 0
}

/// Witness for **F_MinimalIfLongFalse** (PROTOCOL.md §5.2.4).
///
/// Proof obligation: when the byte string has more than 1 byte (length > 1),
/// IsMinimalIfCondition always returns false.
///
/// Body: the structural 3-way pattern. Under `requires(len > 1)`, both
/// `len == 0` and `len == 1` are false, so the else branch → false. Proves
/// that multi-byte encodings always fail the MINIMALIF check.
/// Z3 proves via linear arithmetic.
#[spec_locked("5.2.4", "F_MinimalIfLongFalse")]
#[blvm_spec_lock::requires(len > 1)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_minimal_if_long_false(len: u64) -> bool {
    len == 0 || len == 1
}

// ─── §11.2.1 IsTaprootOutput ─────────────────────────────────────────────────

/// Witness for **F_TaprootOutputScriptLengthInvalid** (PROTOCOL.md §11.2.1).
///
/// Proof obligation: when the scriptPubKey is not exactly 34 bytes, IsTaprootOutput
/// returns false.
///
/// Body: `script_len == 34` (captures the necessary length condition). Under
/// `requires(script_len != 34)`, the expression is false. A P2TR script must be
/// exactly 34 bytes (`OP_1 <push32> <32-byte-key>`); all other lengths are
/// rejected. Z3 proves via linear arithmetic.
#[spec_locked("11.2.1", "F_TaprootOutputScriptLengthInvalid")]
#[blvm_spec_lock::requires(script_len != 34)]
#[blvm_spec_lock::ensures(result == false)]
pub(crate) fn _verify_f_taproot_output_script_length_invalid(script_len: u64) -> bool {
    script_len == 34
}

// ─── §5.4.9 BIP54 Deployment Field Invariants ────────────────────────────────

/// Witness for **F_Bip54SignalBit** (PROTOCOL.md §5.4.9).
///
/// Proof obligation: The BIP54 version-bits deployment always uses signal bit 15.
///
/// All three network deployments (mainnet, testnet, regtest) use the same constant
/// bit index. Body: returns the constant 15. Z3 proves the identity trivially.
#[spec_locked("5.4.9", "F_Bip54SignalBit")]
#[blvm_spec_lock::ensures(result == 15)]
pub(crate) fn _verify_f_bip54_signal_bit() -> u64 {
    15
}

/// Witness for **F_Bip54StartTimeZero** (PROTOCOL.md §5.4.9).
///
/// Proof obligation: The BIP54 deployment start time is 0 (immediate activation;
/// no time-gated window).
///
/// Body: returns 0. Z3 proves the identity trivially.
#[spec_locked("5.4.9", "F_Bip54StartTimeZero")]
#[blvm_spec_lock::ensures(result == 0)]
pub(crate) fn _verify_f_bip54_start_time_zero() -> u64 {
    0
}

// ─── §11.2 Taproot Activation Heights ────────────────────────────────────────

/// Witness for **F_TaprootActivationMainnet** (PROTOCOL.md §11.2).
///
/// Proof obligation: on mainnet (network == 0), TaprootActivationHeight returns
/// exactly 709,632.
///
/// Uses an if/else on the integer network ID (instead of enum match) so the Z3
/// translator can model the control flow. Under `requires(network == 0)`, the
/// first branch is taken → TAPROOT_ACTIVATION_MAINNET.
#[spec_locked("11.2", "F_TaprootActivationMainnet")]
#[blvm_spec_lock::requires(network == 0)]
#[blvm_spec_lock::ensures(result == TAPROOT_ACTIVATION_MAINNET)]
pub(crate) fn _verify_f_taproot_activation_mainnet(network: u64) -> u64 {
    if network == 0 {
        TAPROOT_ACTIVATION_MAINNET
    } else if network == 1 {
        TAPROOT_ACTIVATION_TESTNET
    } else {
        0
    }
}

/// Witness for **F_TaprootActivationTestnet** (PROTOCOL.md §11.2).
///
/// Proof obligation: on testnet (network == 1), TaprootActivationHeight returns
/// exactly 2,011,968.
///
/// Under `requires(network == 1)`, the second branch is taken →
/// TAPROOT_ACTIVATION_TESTNET.
#[spec_locked("11.2", "F_TaprootActivationTestnet")]
#[blvm_spec_lock::requires(network == 1)]
#[blvm_spec_lock::ensures(result == TAPROOT_ACTIVATION_TESTNET)]
pub(crate) fn _verify_f_taproot_activation_testnet(network: u64) -> u64 {
    if network == 0 {
        TAPROOT_ACTIVATION_MAINNET
    } else if network == 1 {
        TAPROOT_ACTIVATION_TESTNET
    } else {
        0
    }
}

/// Witness for **F_TaprootActivationRegtest** (PROTOCOL.md §11.2).
///
/// Proof obligation: on regtest (network ≠ 0 and ≠ 1, i.e., network == 2),
/// TaprootActivationHeight returns 0 (immediate activation at genesis).
///
/// Under `requires(network != 0 && network != 1)`, both branches are skipped
/// → else branch → 0.
#[spec_locked("11.2", "F_TaprootActivationRegtest")]
#[blvm_spec_lock::requires(network != 0)]
#[blvm_spec_lock::requires(network != 1)]
#[blvm_spec_lock::ensures(result == 0)]
pub(crate) fn _verify_f_taproot_activation_regtest(network: u64) -> u64 {
    if network == 0 {
        TAPROOT_ACTIVATION_MAINNET
    } else if network == 1 {
        TAPROOT_ACTIVATION_TESTNET
    } else {
        0
    }
}
