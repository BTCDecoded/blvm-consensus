//! consensus Test Vector Integration
//!
//! Integrates consensus's test vectors to provide free verification coverage.
//! Test vectors are extracted from consensus's test suite and used to verify
//! consensus correctness.
//!
//! Source: consensus test data (`bitcoin/src/test/data/*.json`)

mod block_tests;
mod integration_test;
mod script_tests;
mod transaction_tests;

pub use block_tests::*;
pub use script_tests::{
    load_default_script_vectors, load_default_witness_script_vectors, load_script_test_vectors,
    parse_flag_string, run_core_script_tests, score_core_script_tests, score_witness_script_tests,
    WitnessScriptTestVector,
};
pub use transaction_tests::*;
