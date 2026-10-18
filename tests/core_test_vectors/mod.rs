//! consensus Test Vector Integration
//!
//! Integrates consensus's test vectors to provide free verification coverage.
//! Test vectors are extracted from consensus's test suite and used to verify
//! consensus correctness.
//!
//! Source: consensus test data (`bitcoin/src/test/data/*.json`)

mod script_tests;
mod transaction_tests;

pub use script_tests::{
    load_default_script_vectors, parse_core_script_asm, parse_flag_string, run_core_script_tests,
    score_core_script_tests,
};
pub use transaction_tests::*;
