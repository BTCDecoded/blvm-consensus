//! Configuration module tests
//!
//! Tests for consensus configuration loading and validation.

use blvm_consensus::config::{
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

    assert_eq!(config.assume_valid_height, 938343);
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
    assert_eq!(config.block_validation.assume_valid_height, 938343);
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

#[test]
fn test_get_consensus_config_returns_defaults() {
    use blvm_consensus::config::{
        get_assume_valid_height, get_consensus_config, get_consensus_config_ref,
        get_n_minimum_chain_work, use_overlay_delta,
    };

    let cfg = get_consensus_config();
    let cfg_ref = get_consensus_config_ref();
    assert_eq!(
        cfg_ref.network_limits.max_headers,
        cfg.network_limits.max_headers
    );
    assert_eq!(cfg_ref.mempool.max_mempool_mb, cfg.mempool.max_mempool_mb);
    assert!(get_assume_valid_height() > 0);
    let _min_work = get_n_minimum_chain_work();
    let _overlay = use_overlay_delta();
}

#[test]
fn test_consensus_config_from_env_parses_values() {
    let keys = [
        "BLVM_ASSUME_VALID_HEIGHT",
        "BLVM_MTP_HEADERS",
        "BLVM_MEMPOOL_MB",
        "BLVM_STRICT_MODE",
        "BLVM_MAX_REORG_DEPTH",
    ];
    for key in keys {
        std::env::remove_var(key);
    }

    std::env::set_var("BLVM_ASSUME_VALID_HEIGHT", "424242");
    std::env::set_var("BLVM_MTP_HEADERS", "15");
    std::env::set_var("BLVM_MEMPOOL_MB", "128");
    std::env::set_var("BLVM_STRICT_MODE", "true");
    std::env::set_var("BLVM_MAX_REORG_DEPTH", "144");

    let config = ConsensusConfig::from_env();
    assert_eq!(config.block_validation.assume_valid_height, 424_242);
    assert_eq!(config.block_validation.median_time_past_headers, 15);
    assert_eq!(config.mempool.max_mempool_mb, 128);
    assert!(config.advanced.strict_mode);
    assert_eq!(config.advanced.max_reorg_depth, 144);

    for key in keys {
        std::env::remove_var(key);
    }
}

#[test]
fn test_consensus_config_from_env_parses_debug_and_feature_flags() {
    std::env::remove_var("BLVM_CONSENSUS_DEBUG");
    std::env::remove_var("BLVM_CONSENSUS_FEATURES");

    std::env::set_var("BLVM_CONSENSUS_DEBUG", "assertions,rejections");
    std::env::set_var("BLVM_CONSENSUS_FEATURES", "batch_txid,simd_hash");

    let config = ConsensusConfig::from_env();
    assert!(config.debug.enable_runtime_assertions);
    assert!(config.debug.log_rejections);
    assert!(!config.debug.enable_verbose_logging);
    assert!(config.features.enable_batch_tx_id_computation);
    assert!(config.features.enable_simd_hash_operations);

    std::env::set_var("BLVM_CONSENSUS_DEBUG", "full");
    std::env::set_var("BLVM_CONSENSUS_FEATURES", "full");
    let full = ConsensusConfig::from_env();
    assert!(full.debug.enable_runtime_invariants);
    assert!(full.debug.enable_performance_profiling);
    assert!(full.features.enable_aggressive_caching);
    assert!(full.features.enable_reference_checks);

    std::env::remove_var("BLVM_CONSENSUS_DEBUG");
    std::env::remove_var("BLVM_CONSENSUS_FEATURES");
}

#[test]
fn test_consensus_config_from_env_parses_performance_and_mempool_tuning() {
    let keys = [
        "BLVM_MEMPOOL_TXS",
        "BLVM_MEMPOOL_EXPIRY_HOURS",
        "BLVM_SCRIPT_THREADS",
        "BLVM_CACHE_OPTIMIZATIONS",
    ];
    for key in keys {
        std::env::remove_var(key);
    }

    std::env::set_var("BLVM_MEMPOOL_TXS", "50000");
    std::env::set_var("BLVM_MEMPOOL_EXPIRY_HOURS", "48");
    std::env::set_var("BLVM_SCRIPT_THREADS", "4");
    std::env::set_var("BLVM_CACHE_OPTIMIZATIONS", "false");

    let config = ConsensusConfig::from_env();
    assert_eq!(config.mempool.max_mempool_txs, 50_000);
    assert_eq!(config.mempool.mempool_expiry_hours, 48);
    assert_eq!(config.performance.script_verification_threads, 4);
    assert!(!config.performance.enable_cache_optimizations);

    for key in keys {
        std::env::remove_var(key);
    }
}

#[test]
fn test_consensus_config_from_env_parses_advanced_overrides() {
    std::env::remove_var("BLVM_RBF");
    std::env::remove_var("BLVM_MAX_BLOCK_SIZE");

    std::env::set_var("BLVM_RBF", "false");
    std::env::set_var("BLVM_MAX_BLOCK_SIZE", "2000000");

    let config = ConsensusConfig::from_env();
    assert!(!config.advanced.enable_rbf);
    assert_eq!(config.advanced.max_block_size_override, 2_000_000);

    std::env::remove_var("BLVM_RBF");
    std::env::remove_var("BLVM_MAX_BLOCK_SIZE");
}

#[test]
fn test_consensus_config_from_env_parses_block_validation_overrides() {
    let keys = [
        "BLVM_COINBASE_MATURITY",
        "BLVM_MAX_SIGOPS_COST",
        "BLVM_PARALLEL_VALIDATION",
    ];
    for key in keys {
        std::env::remove_var(key);
    }

    std::env::set_var("BLVM_COINBASE_MATURITY", "50");
    std::env::set_var("BLVM_MAX_SIGOPS_COST", "12000");
    std::env::set_var("BLVM_PARALLEL_VALIDATION", "true");

    let config = ConsensusConfig::from_env();
    assert_eq!(config.block_validation.coinbase_maturity_override, 50);
    assert_eq!(
        config.block_validation.max_block_sigops_cost_override,
        12_000
    );
    assert!(config.block_validation.enable_parallel_validation);

    for key in keys {
        std::env::remove_var(key);
    }
}

#[test]
fn test_consensus_config_from_env_parses_network_limits() {
    std::env::remove_var("BLVM_MAX_ADDR_ADDRESSES");
    std::env::remove_var("BLVM_MAX_INV_ITEMS");
    std::env::remove_var("BLVM_MAX_HEADERS");

    std::env::set_var("BLVM_MAX_ADDR_ADDRESSES", "500");
    std::env::set_var("BLVM_MAX_INV_ITEMS", "1000");
    std::env::set_var("BLVM_MAX_HEADERS", "500");

    let config = ConsensusConfig::from_env();
    assert_eq!(config.network_limits.max_addr_addresses, 500);
    assert_eq!(config.network_limits.max_inv_items, 1000);
    assert_eq!(config.network_limits.max_headers, 500);

    std::env::remove_var("BLVM_MAX_ADDR_ADDRESSES");
    std::env::remove_var("BLVM_MAX_INV_ITEMS");
    std::env::remove_var("BLVM_MAX_HEADERS");
}

#[test]
fn test_consensus_config_from_env_parses_extended_mempool_and_network_limits() {
    let keys = [
        "BLVM_MAX_USER_AGENT_LENGTH",
        "BLVM_MEMPOOL_MIN_RELAY_FEE",
        "BLVM_MEMPOOL_MIN_TX_FEE",
        "BLVM_MEMPOOL_RBF_FEE_INCREMENT",
    ];
    for key in keys {
        std::env::remove_var(key);
    }

    std::env::set_var("BLVM_MAX_USER_AGENT_LENGTH", "512");
    std::env::set_var("BLVM_MEMPOOL_MIN_RELAY_FEE", "5");
    std::env::set_var("BLVM_MEMPOOL_MIN_TX_FEE", "1000");
    std::env::set_var("BLVM_MEMPOOL_RBF_FEE_INCREMENT", "500");

    let config = ConsensusConfig::from_env();
    assert_eq!(config.network_limits.max_user_agent_length, 512);
    assert_eq!(config.mempool.min_relay_fee_rate, 5);
    assert_eq!(config.mempool.min_tx_fee, 1000);
    assert_eq!(config.mempool.rbf_fee_increment, 500);

    for key in keys {
        std::env::remove_var(key);
    }
}

#[test]
fn test_consensus_config_from_env_parses_utxo_commitment_and_performance_tuning() {
    let keys = [
        "BLVM_UTXO_COMMITMENT_MAX_SET_MB",
        "BLVM_UTXO_COMMITMENT_MAX_UTXO_COUNT",
        "BLVM_UTXO_COMMITMENT_MAX_HISTORICAL",
        "BLVM_UTXO_COMMITMENT_INCREMENTAL",
        "BLVM_PARALLEL_BATCH_SIZE",
        "BLVM_SIMD",
        "BLVM_BATCH_UTXO_LOOKUPS",
    ];
    for key in keys {
        std::env::remove_var(key);
    }

    std::env::set_var("BLVM_UTXO_COMMITMENT_MAX_SET_MB", "64");
    std::env::set_var("BLVM_UTXO_COMMITMENT_MAX_UTXO_COUNT", "5000000");
    std::env::set_var("BLVM_UTXO_COMMITMENT_MAX_HISTORICAL", "12");
    std::env::set_var("BLVM_UTXO_COMMITMENT_INCREMENTAL", "true");
    std::env::set_var("BLVM_PARALLEL_BATCH_SIZE", "256");
    std::env::set_var("BLVM_SIMD", "true");
    std::env::set_var("BLVM_BATCH_UTXO_LOOKUPS", "true");

    let config = ConsensusConfig::from_env();
    assert_eq!(config.utxo_commitment.max_utxo_commitment_set_mb, 64);
    assert_eq!(config.utxo_commitment.max_utxo_count, 5_000_000);
    assert_eq!(config.utxo_commitment.max_historical_commitments, 12);
    assert!(config.utxo_commitment.enable_incremental_updates);
    assert_eq!(config.performance.parallel_batch_size, 256);
    assert!(config.performance.enable_simd_optimizations);
    assert!(config.performance.enable_batch_utxo_lookups);

    for key in keys {
        std::env::remove_var(key);
    }
}

#[test]
fn test_consensus_config_from_env_parses_ibd_chunk_settings() {
    let keys = ["BLVM_IBD_CHUNK_THRESHOLD", "BLVM_IBD_MIN_CHUNK_SIZE"];
    for key in keys {
        std::env::remove_var(key);
    }

    std::env::set_var("BLVM_IBD_CHUNK_THRESHOLD", "5000");
    std::env::set_var("BLVM_IBD_MIN_CHUNK_SIZE", "128");

    let config = ConsensusConfig::from_env();
    assert_eq!(config.performance.ibd_chunk_threshold, Some(5000));
    assert_eq!(config.performance.ibd_min_chunk_size, Some(128));

    for key in keys {
        std::env::remove_var(key);
    }
}

#[test]
fn test_consensus_config_from_env_parses_custom_checkpoints() {
    std::env::remove_var("BLVM_CUSTOM_CHECKPOINTS");
    std::env::set_var("BLVM_CUSTOM_CHECKPOINTS", "100000, 250000 ,500000");

    let config = ConsensusConfig::from_env();
    assert_eq!(
        config.advanced.custom_checkpoints,
        vec![100_000, 250_000, 500_000]
    );

    std::env::remove_var("BLVM_CUSTOM_CHECKPOINTS");
}
