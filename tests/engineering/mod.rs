//! Engineering-specific edge case tests
//!
//! These tests cover edge cases that are consensus-critical but not purely mathematical:
//! - Integer overflow/underflow protection
//! - Serialization correctness
//! - Resource limit enforcement
//! - Parser determinism

pub mod bip112_csv_integration_tests;
pub mod bip113_integration_tests;
pub mod bip65_cltv_integration_tests;
pub mod bip_interaction_tests;
pub mod bip_test_helpers;
pub mod integer_overflow_edge_cases;
pub mod parser_edge_cases;
pub mod resource_limits_edge_cases;
pub mod segwit_integration_tests;
pub mod serialization_edge_cases;
pub mod taproot_integration_tests;
pub mod transaction_size_edge_cases;
pub mod witness_validation_tests;
