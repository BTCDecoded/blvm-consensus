//! Integration tests for blvm-consensus
//!
//! BIP connect_block enforcement: `bip_enforcement_tests` + `bip_compliance_tests`.
//! Node RPC / historical replay / differential helpers live in sibling modules below.

mod bip_compliance_tests;
mod bip_enforcement_tests;
mod blvm_integration_tests;
mod consensus_validation;
mod core_test_vectors;
mod differential_tests;
mod helpers;
mod historical_replay;
mod mempool_mining;
mod node_rpc;

// Production feature-gated tests
#[cfg(feature = "production")]
mod production_integration_tests;
