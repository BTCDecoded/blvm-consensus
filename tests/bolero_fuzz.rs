//! Bolero property tests: `cargo test -p blvm-consensus --features bolero --test bolero_fuzz`.

#[cfg(feature = "bolero")]
#[path = "fuzzing/transaction_validation.rs"]
mod transaction_validation;

#[cfg(feature = "bolero")]
#[path = "fuzzing/block_validation.rs"]
mod block_validation;

#[cfg(not(feature = "bolero"))]
#[test]
fn bolero_feature_disabled() {
    // Integration test crate is always present; Bolero modules compile only with `--features bolero`.
}
