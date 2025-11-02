//! Integration tests for consensus-proof

mod consensus_validation;
mod mempool_mining;

// Production optimization integration tests (only compiled with production feature)
#[cfg(feature = "production")]
mod production_integration_tests;

// UTXO commitments integration tests (only compiled with utxo-commitments feature)
#[cfg(feature = "utxo-commitments")]
mod utxo_commitments_integration;


























