//! Error types for consensus validation

use std::borrow::Cow;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Clone)]
pub enum ConsensusError {
    #[error("Transaction validation failed: {0}")]
    TransactionValidation(Cow<'static, str>),

    #[error("Block validation failed: {0}")]
    BlockValidation(Cow<'static, str>),

    #[error("Script execution failed: {0}")]
    ScriptExecution(Cow<'static, str>),

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
