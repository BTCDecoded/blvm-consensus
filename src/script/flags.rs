//! Script verification flags matching Bitcoin Core `interpreter.h`.
//!
//! These constants map directly to Bitcoin Core's `SCRIPT_VERIFY_*` flags.
//! Divergence from Core's values is a consensus bug.
//!
//! Reference: bitcoin/bitcoin src/script/interpreter.h (Bitcoin Core 26+).

/// No flags — bare pubkey / standard evaluation.
pub const SCRIPT_VERIFY_NONE: u32 = 0;

/// Evaluate P2SH subscripts (BIP16).
pub const SCRIPT_VERIFY_P2SH: u32 = 1 << 0; // 0x0001

/// Require DER-encoded signatures (BIP66 stricter encoding).
pub const SCRIPT_VERIFY_STRICTENC: u32 = 1 << 1; // 0x0002

/// Enforce strict DER signature encoding (BIP66).
pub const SCRIPT_VERIFY_DERSIG: u32 = 1 << 2; // 0x0004

/// Enforce low-S signature requirement (BIP62).
pub const SCRIPT_VERIFY_LOW_S: u32 = 1 << 3; // 0x0008

/// OP_CHECKMULTISIG dummy element must be OP_0 (BIP147).
pub const SCRIPT_VERIFY_NULLDUMMY: u32 = 1 << 4; // 0x0010

/// Require only push opcodes in scriptSig.
pub const SCRIPT_VERIFY_SIGPUSHONLY: u32 = 1 << 5; // 0x0020

/// Require minimal encoding for pushdata (BIP62 rule 3 & 4).
pub const SCRIPT_VERIFY_MINIMALDATA: u32 = 1 << 6; // 0x0040

/// NOPs 1–10 are reserved; treat them as invalid if a future soft-fork hasn't defined them.
pub const SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_NOPS: u32 = 1 << 7; // 0x0080

/// Require clean stack after script evaluation (BIP62).
pub const SCRIPT_VERIFY_CLEANSTACK: u32 = 1 << 8; // 0x0100

/// Enable OP_CHECKLOCKTIMEVERIFY (BIP65).
pub const SCRIPT_VERIFY_CHECKLOCKTIMEVERIFY: u32 = 1 << 9; // 0x0200

/// Enable OP_CHECKSEQUENCEVERIFY (BIP112).
pub const SCRIPT_VERIFY_CHECKSEQUENCEVERIFY: u32 = 1 << 10; // 0x0400

/// Enable Segregated Witness evaluation (BIP141/143).
pub const SCRIPT_VERIFY_WITNESS: u32 = 1 << 11; // 0x0800

/// Reject unknown witness program versions (allows future soft-forks).
pub const SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_WITNESS_PROGRAM: u32 = 1 << 12; // 0x1000

/// Require minimal IF argument (BIP342 and pre-taproot cleanups).
pub const SCRIPT_VERIFY_MINIMALIF: u32 = 1 << 13; // 0x2000

/// Require empty sig on CHECKSIG failure (BIP342 null-fail rule).
pub const SCRIPT_VERIFY_NULLFAIL: u32 = 1 << 14; // 0x4000

/// Require compressed public keys in witness programs (BIP143).
pub const SCRIPT_VERIFY_WITNESS_PUBKEYTYPE: u32 = 1 << 15; // 0x8000

/// Signature hash must not hash the script code after a CODESEPARATOR (BIP143 §4).
pub const SCRIPT_VERIFY_CONST_SCRIPTCODE: u32 = 1 << 16; // 0x10000

/// Enable Taproot/Tapscript evaluation (BIP341/342).
///
/// WARNING: This is 0x20000 (bit 17), NOT 0x8000 (bit 15, which is WITNESS_PUBKEYTYPE).
/// Confusing the two disables Taproot validation entirely.
pub const SCRIPT_VERIFY_TAPROOT: u32 = 1 << 17; // 0x20000

/// Reject unknown Taproot leaf versions (allows future soft-forks).
pub const SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_TAPROOT_VERSION: u32 = 1 << 18; // 0x40000

/// Reject OP_SUCCESS opcodes that are not re-defined by a known Tapscript upgrade.
pub const SCRIPT_VERIFY_DISCOURAGE_OP_SUCCESS: u32 = 1 << 19; // 0x80000

/// Reject unknown pubkey types in Tapscript (allows future soft-forks).
pub const SCRIPT_VERIFY_DISCOURAGE_UPGRADABLE_PUBKEYTYPE: u32 = 1 << 20; // 0x100000

/// Standard mandatory flags for segwit-v0 transactions (pre-Taproot).
///
/// Equivalent to Bitcoin Core's `MANDATORY_SCRIPT_VERIFY_FLAGS` with witness enabled.
pub const SEGWIT_STANDARD_FLAGS: u32 =
    SCRIPT_VERIFY_P2SH | SCRIPT_VERIFY_WITNESS | SCRIPT_VERIFY_WITNESS_PUBKEYTYPE;

/// Full Taproot activation flag set (P2SH + SegWit + Taproot).
///
/// Use this as the baseline when verifying Taproot outputs.
pub const TAPROOT_STANDARD_FLAGS: u32 = SCRIPT_VERIFY_P2SH
    | SCRIPT_VERIFY_WITNESS
    | SCRIPT_VERIFY_WITNESS_PUBKEYTYPE
    | SCRIPT_VERIFY_TAPROOT;
