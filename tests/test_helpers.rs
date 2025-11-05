//! Test helper utilities for CI-aware testing
//!
//! Provides utilities to detect CI environment and adjust test behavior
//! without disabling tests.

/// Check if running in CI environment
///
/// Detects common CI environment variables:
/// - GITHUB_ACTIONS (GitHub Actions)
/// - CI (generic CI environment)
/// - CONTINUOUS_INTEGRATION (generic CI)
pub fn is_ci() -> bool {
    std::env::var("CI").is_ok()
        || std::env::var("GITHUB_ACTIONS").is_ok()
        || std::env::var("CONTINUOUS_INTEGRATION").is_ok()
}

/// Get performance threshold multiplier for CI
///
/// Returns a multiplier to adjust timing thresholds in CI environments.
/// CI environments may have slower resources or higher load, so we
/// allow more time for operations while still maintaining test validity.
pub fn performance_threshold_multiplier() -> u64 {
    if is_ci() {
        5 // Allow 5x more time in CI
    } else {
        1 // Normal threshold locally
    }
}

/// Get adjusted timeout for CI environments
///
/// Adjusts timeout values based on environment.
/// CI environments get more lenient timeouts.
pub fn adjusted_timeout(base_timeout_ms: u64) -> u64 {
    base_timeout_ms * performance_threshold_multiplier()
}
