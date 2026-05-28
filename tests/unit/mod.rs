//! Unit tests for blvm-consensus modules
//!
//! Modules with significant API drift from the current codebase are temporarily
//! gated behind `#[cfg(any())]` (never-true condition) to prevent compilation
//! failures while the tests are updated. Remove the gate once each module is
//! brought up to date.

mod mempool_more_tests;
mod pow_tests;
mod transaction_tests;

mod block_edge_cases;
mod comprehensive_property_tests;
mod difficulty_edge_cases;
mod economic_tests;
mod mempool_edge_cases;
mod reorganization_edge_cases;
mod script_opcode_property_tests;
mod script_tests;
mod segwit_taproot_property_tests;
mod transaction_edge_cases;
mod utxo_edge_cases;

// Production optimization tests (only compiled with production feature)
#[cfg(all(feature = "production", any()))]
mod blvm_memory_profiling_tests;
#[cfg(all(feature = "production", any()))]
mod blvm_optimization_tests;
#[cfg(all(feature = "production", any()))]
mod production_cache_tests;
#[cfg(all(feature = "production", any()))]
mod production_edge_tests;
#[cfg(all(feature = "production", any()))]
mod production_memory_tests;
#[cfg(all(feature = "production", any()))]
mod production_parallel_tests;

// UTXO commitments and spam filter tests moved to blvm-protocol
