//! Unit tests for consensus-proof modules

mod transaction_tests;
mod script_tests;
mod economic_tests;
mod pow_tests;

// Production optimization tests (only compiled with production feature)
#[cfg(feature = "production")]
mod production_correctness_tests;
#[cfg(feature = "production")]
mod production_context_tests;
#[cfg(feature = "production")]
mod production_parallel_tests;
#[cfg(feature = "production")]
mod production_memory_tests;
#[cfg(feature = "production")]
mod production_edge_tests;
#[cfg(feature = "production")]
mod production_cache_tests;

// UTXO commitments tests (only compiled with utxo-commitments feature)
#[cfg(feature = "utxo-commitments")]
mod utxo_commitments_tests;
#[cfg(feature = "utxo-commitments")]
mod spam_filter_tests;


























