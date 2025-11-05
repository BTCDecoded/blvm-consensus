//! Integration tests for consensus-proof

mod consensus_validation;
mod mempool_mining;

// Production optimization integration tests (only compiled with production feature)
#[cfg(feature = "production")]
mod production_integration_tests;
#[cfg(feature = "production")]
mod bllvm_integration_tests;

// UTXO commitments integration tests (only compiled with utxo-commitments feature)
#[cfg(feature = "utxo-commitments")]
mod utxo_commitments_integration;

// Bitcoin Core test vector integration (read-only, safe)
mod core_test_vectors;

// Differential fuzzing vs Bitcoin Core (requires Core RPC)
mod differential_tests;

// Reference-Node RPC integration tests (uses our own RPC infrastructure)
mod reference_node_rpc;

// Historical block replay (Phase 3.1)
mod historical_replay;

// BIP compliance tests
mod bip_compliance_tests;


























