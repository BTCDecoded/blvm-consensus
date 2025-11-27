//! Configuration module tests
//!
//! Tests for consensus configuration loading and validation.

use bllvm_consensus::config::{
    AdvancedConfig, BlockValidationConfig, ConsensusConfig, FeatureFlagsConfig, MempoolConfig,
    NetworkMessageLimits, PerformanceConfig,
};

#[test]
fn test_network_message_limits_default() {
    let limits = NetworkMessageLimits::default();

    assert_eq!(limits.max_addr_addresses, 1000);
    assert_eq!(limits.max_inv_items, 50000);
    assert_eq!(limits.max_headers, 2000);
    assert_eq!(limits.max_user_agent_length, 256);
}

#[test]
fn test_block_validation_config_default() {
    let config = BlockValidationConfig::default();

    assert_eq!(config.assume_valid_height, 0);
    assert_eq!(config.median_time_past_headers, 11);
    assert_eq!(config.coinbase_maturity_override, 0);
    assert_eq!(config.max_block_sigops_cost_override, 0);
}

#[test]
fn test_mempool_config_default() {
    let config = MempoolConfig::default();

    assert_eq!(config.max_mempool_mb, 300);
    assert!(config.min_relay_fee_rate > 0);
}

#[test]
fn test_consensus_config_default() {
    let config = ConsensusConfig::default();

    // Verify all sub-configs have defaults
    assert_eq!(config.network_limits.max_addr_addresses, 1000);
    assert_eq!(config.block_validation.assume_valid_height, 0);
}

#[test]
fn test_consensus_config_from_env() {
    // Test that from_env() doesn't panic
    // Note: Actual env var testing would require setting/unsetting env vars
    let _config = ConsensusConfig::from_env();
}

#[test]
fn test_performance_config_default() {
    let config = PerformanceConfig::default();

    // Verify performance config has reasonable defaults
    assert!(config.enable_cache_optimizations);
}

#[test]
fn test_feature_flags_config_default() {
    let config = FeatureFlagsConfig::default();

    // Verify feature flags have defaults
    assert!(!config.enable_experimental_optimizations);
}

#[test]
fn test_advanced_config_default() {
    let config = AdvancedConfig::default();

    // Verify advanced config has defaults
    assert!(!config.strict_mode);
}

#[test]
fn test_config_serialization() {
    let config = ConsensusConfig::default();

    // Test that config can be serialized
    let json = serde_json::to_string(&config);
    assert!(json.is_ok());
}

#[test]
fn test_config_deserialization() {
    let config = ConsensusConfig::default();
    let json = serde_json::to_string(&config).unwrap();

    // Test that config can be deserialized
    let deserialized: Result<ConsensusConfig, _> = serde_json::from_str(&json);
    assert!(deserialized.is_ok());
}
