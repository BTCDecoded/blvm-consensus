//! consensus Test Vector Integration
//!
//! Integrates consensus's test vectors to provide free verification coverage.
//! Test vectors are extracted from consensus's test suite and used to verify
//! consensus correctness.
//!
//! Source: consensus test data (`bitcoin/src/test/data/*.json`)

mod block_tests;
mod transaction_tests;
mod integration_test;

pub use block_tests::*;
pub use transaction_tests::*;

