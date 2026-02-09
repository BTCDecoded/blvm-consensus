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
#[cfg(feature = "utxo-commitments")]
mod utxo_proof_verification_tests;

// Bitcoin Core test vector integration (read-only, safe)
mod core_test_vectors;

// Differential testing integration (basic functionality, full implementation in blvm-bench)
mod differential_tests;

// blvm-node RPC integration tests (uses our own RPC infrastructure)
mod node_rpc;

// Historical block replay (Phase 3.1)
mod historical_replay;

// BIP compliance tests
mod bip_compliance_tests;

// BIP enforcement tests - verify BIP checks are called in connect_block
mod bip_enforcement_tests;

// BIP integration smoke tests - lightweight verification
mod bip_integration_smoke_tests;


























