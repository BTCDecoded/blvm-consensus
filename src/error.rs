//! Error types for consensus validation

use std::borrow::Cow;
use thiserror::Error;

/// Detailed script error codes, modeled after libbitcoin-consensus `verify_result`.
///
/// NOTE: This enum intentionally mirrors libbitcoin's naming where possible so that
/// callers can perform precise compatibility checks. Not all variants are currently
/// used in production code yet – they provide a complete mapping target that script
/// validation can adopt incrementally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptErrorCode {
    EvalTrue,
    EvalFalse,
    ScriptSize,
    PushSize,
    OpCount,
    StackSize,
    SigCount,
    PubkeyCount,
    Verify,
    EqualVerify,
    CheckMultisigVerify,
    ChecksigVerify,
    NumEqualVerify,
    BadOpcode,
    DisabledOpcode,
    InvalidStackOperation,
    InvalidAltstackOperation,
    UnbalancedConditional,
    SigHashType,
    SigDer,
    MinimalData,
    SigPushOnly,
    SigHighS,
    SigNullDummy,
    PubkeyType,
    CleanStack,
    MinimalIf,
    SigNullFail,
    DiscourageUpgradableNops,
    DiscourageUpgradableWitnessProgram,
    WitnessProgramWrongLength,
    WitnessProgramEmptyWitness,
    WitnessProgramMismatch,
    WitnessMalleated,
    WitnessMalleatedP2SH,
    WitnessUnexpected,
    WitnessPubkeyType,
    TxInvalid,
    TxSizeInvalid,
    TxInputInvalid,
    NegativeLocktime,
    UnsatisfiedLocktime,
    ValueOverflow,
    UnknownError,
}

#[derive(Error, Debug, PartialEq, Clone)]
pub enum ConsensusError {
    #[error("Transaction validation failed: {0}")]
    TransactionValidation(Cow<'static, str>),

    #[error("Block validation failed: {0}")]
    BlockValidation(Cow<'static, str>),

    #[error("Script execution failed: {0}")]
    ScriptExecution(Cow<'static, str>),

    /// Script failed with a detailed error code.
    ///
    /// This variant is preferred for new code that needs precise, libbitcoin-compatible
    /// error reporting. Older callers can continue to use the `ScriptExecution` variant.
    #[error("Script execution failed with code {code:?}: {message}")]
    ScriptErrorWithCode {
        code: ScriptErrorCode,
        message: Cow<'static, str>,
    },

    #[error("UTXO not found: {0}")]
    UtxoNotFound(Cow<'static, str>),

    #[error("Invalid signature: {0}")]
    InvalidSignature(Cow<'static, str>),

    #[error("Invalid proof of work: {0}")]
    InvalidProofOfWork(Cow<'static, str>),

    #[error("Economic validation failed: {0}")]
    EconomicValidation(Cow<'static, str>),

    #[error("Serialization error: {0}")]
    Serialization(Cow<'static, str>),

    #[error("Consensus rule violation: {0}")]
    ConsensusRuleViolation(Cow<'static, str>),

    #[error("Invalid sighash type: {0}")]
    InvalidSighashType(u8),

    #[error("Invalid input index: {0}")]
    InvalidInputIndex(usize),

    #[error("Invalid prevouts count: expected {0}, got {1}")]
    InvalidPrevoutsCount(usize, usize),
}

pub type Result<T> = std::result::Result<T, ConsensusError>;
