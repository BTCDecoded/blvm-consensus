//! Configuration for blvm-consensus
//!
//! Provides configurable parameters for consensus validation, network message limits,
//! and performance optimizations. These settings can be loaded from config files,
//! environment variables, or passed programmatically.

use serde::{Deserialize, Serialize};

// Re-export foundational config types from blvm-primitives
pub use blvm_primitives::config::{
    AdvancedConfig, BlockValidationConfig, DebugConfig, FeatureFlagsConfig, MempoolConfig,
    NetworkMessageLimits, PerformanceConfig, UtxoCommitmentConfig,
};

/// Complete consensus configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ConsensusConfig {
    /// Network message size limits
    #[serde(default)]
    pub network_limits: NetworkMessageLimits,

    /// Block validation configuration
    #[serde(default)]
    pub block_validation: BlockValidationConfig,

    /// Mempool configuration
    #[serde(default)]
    pub mempool: MempoolConfig,

    /// UTXO commitment set configuration
    #[serde(default)]
    pub utxo_commitment: UtxoCommitmentConfig,

    /// Performance and optimization configuration
    #[serde(default)]
    pub performance: PerformanceConfig,

    /// Debug and development configuration
    #[serde(default)]
    pub debug: DebugConfig,

    /// Feature flags configuration
    #[serde(default)]
    pub features: FeatureFlagsConfig,

    /// Advanced configuration options
    #[serde(default)]
    pub advanced: AdvancedConfig,
}

impl ConsensusConfig {
    /// Load configuration from environment variables
    ///
    /// Environment variables. Short names (e.g. BLVM_ASSUME_VALID_HEIGHT) preferred.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Block validation
        if let Ok(val) = std::env::var("BLVM_ASSUME_VALID_HEIGHT") {
            if let Ok(height) = val.parse::<u64>() {
                config.block_validation.assume_valid_height = height;
            }
        }

        if let Ok(val) = std::env::var("BLVM_MTP_HEADERS") {
            if let Ok(count) = val.parse::<usize>() {
                config.block_validation.median_time_past_headers = count;
            }
        }
        if let Ok(val) = std::env::var("BLVM_PARALLEL_VALIDATION") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.block_validation.enable_parallel_validation = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLVM_COINBASE_MATURITY") {
            if let Ok(maturity) = val.parse::<u64>() {
                config.block_validation.coinbase_maturity_override = maturity;
            }
        }
        if let Ok(val) = std::env::var("BLVM_MAX_SIGOPS_COST") {
            if let Ok(cost) = val.parse::<u64>() {
                config.block_validation.max_block_sigops_cost_override = cost;
            }
        }

        if let Ok(val) = std::env::var("BLVM_MAX_ADDR_ADDRESSES") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_addr_addresses = limit;
            }
        }

        if let Ok(val) = std::env::var("BLVM_MAX_INV_ITEMS") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_inv_items = limit;
            }
        }

        if let Ok(val) = std::env::var("BLVM_MAX_HEADERS") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_headers = limit;
            }
        }

        if let Ok(val) = std::env::var("BLVM_MAX_USER_AGENT_LENGTH") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_user_agent_length = limit;
            }
        }

        // Load mempool configuration
        if let Ok(val) = std::env::var("BLVM_MEMPOOL_MB") {
            if let Ok(mb) = val.parse::<u64>() {
                config.mempool.max_mempool_mb = mb;
            }
        }
        if let Ok(val) = std::env::var("BLVM_MEMPOOL_TXS") {
            if let Ok(count) = val.parse::<usize>() {
                config.mempool.max_mempool_txs = count;
            }
        }
        if let Ok(val) = std::env::var("BLVM_MEMPOOL_EXPIRY_HOURS") {
            if let Ok(hours) = val.parse::<u64>() {
                config.mempool.mempool_expiry_hours = hours;
            }
        }
        if let Ok(val) = std::env::var("BLVM_MEMPOOL_MIN_RELAY_FEE") {
            if let Ok(rate) = val.parse::<u64>() {
                config.mempool.min_relay_fee_rate = rate;
            }
        }
        if let Ok(val) = std::env::var("BLVM_MEMPOOL_MIN_TX_FEE") {
            if let Ok(fee) = val.parse::<i64>() {
                config.mempool.min_tx_fee = fee;
            }
        }
        if let Ok(val) = std::env::var("BLVM_MEMPOOL_RBF_FEE_INCREMENT") {
            if let Ok(increment) = val.parse::<i64>() {
                config.mempool.rbf_fee_increment = increment;
            }
        }

        // Load UTXO commitment configuration
        if let Ok(val) = std::env::var("BLVM_UTXO_COMMITMENT_MAX_SET_MB") {
            if let Ok(mb) = val.parse::<u64>() {
                config.utxo_commitment.max_utxo_commitment_set_mb = mb;
            }
        }
        if let Ok(val) = std::env::var("BLVM_UTXO_COMMITMENT_MAX_UTXO_COUNT") {
            if let Ok(count) = val.parse::<u64>() {
                config.utxo_commitment.max_utxo_count = count;
            }
        }
        if let Ok(val) = std::env::var("BLVM_UTXO_COMMITMENT_MAX_HISTORICAL") {
            if let Ok(count) = val.parse::<usize>() {
                config.utxo_commitment.max_historical_commitments = count;
            }
        }
        if let Ok(val) = std::env::var("BLVM_UTXO_COMMITMENT_INCREMENTAL") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.utxo_commitment.enable_incremental_updates = enabled;
            }
        }

        // Load performance configuration
        if let Ok(val) = std::env::var("BLVM_SCRIPT_THREADS") {
            if let Ok(threads) = val.parse::<usize>() {
                config.performance.script_verification_threads = threads;
            }
        }
        if let Ok(val) = std::env::var("BLVM_PARALLEL_BATCH_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                config.performance.parallel_batch_size = size;
            }
        }
        if let Ok(val) = std::env::var("BLVM_SIMD") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.performance.enable_simd_optimizations = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLVM_CACHE_OPTIMIZATIONS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.performance.enable_cache_optimizations = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLVM_BATCH_UTXO_LOOKUPS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.performance.enable_batch_utxo_lookups = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLVM_IBD_CHUNK_THRESHOLD") {
            if let Ok(n) = val.parse::<usize>() {
                config.performance.ibd_chunk_threshold = Some(n);
            }
        }
        if let Ok(val) = std::env::var("BLVM_IBD_MIN_CHUNK_SIZE") {
            if let Ok(n) = val.parse::<usize>() {
                config.performance.ibd_min_chunk_size = Some(n);
            }
        }

        // Load debug configuration: BLVM_CONSENSUS_DEBUG=assertions,invariants,verbose,profile,rejections or =full
        if let Ok(val) = std::env::var("BLVM_CONSENSUS_DEBUG") {
            let parts: Vec<&str> = val.split(',').map(|s| s.trim()).collect();
            for p in &parts {
                match *p {
                    "full" => {
                        config.debug.enable_runtime_assertions = true;
                        config.debug.enable_runtime_invariants = true;
                        config.debug.enable_verbose_logging = true;
                        config.debug.enable_performance_profiling = true;
                        config.debug.log_rejections = true;
                    }
                    "assertions" => config.debug.enable_runtime_assertions = true,
                    "invariants" => config.debug.enable_runtime_invariants = true,
                    "verbose" => config.debug.enable_verbose_logging = true,
                    "profile" => config.debug.enable_performance_profiling = true,
                    "rejections" => config.debug.log_rejections = true,
                    _ => {}
                }
            }
        }

        // Load feature flags: BLVM_CONSENSUS_FEATURES=experimental,bounds_check,reference_checks,aggressive_cache,batch_txid,simd_hash or =full
        if let Ok(val) = std::env::var("BLVM_CONSENSUS_FEATURES") {
            let parts: Vec<&str> = val.split(',').map(|s| s.trim()).collect();
            for p in &parts {
                match *p {
                    "full" => {
                        config.features.enable_experimental_optimizations = true;
                        config.features.enable_bounds_check_optimizations = true;
                        config.features.enable_reference_checks = true;
                        config.features.enable_aggressive_caching = true;
                        config.features.enable_batch_tx_id_computation = true;
                        config.features.enable_simd_hash_operations = true;
                    }
                    "experimental" => config.features.enable_experimental_optimizations = true,
                    "bounds_check" => config.features.enable_bounds_check_optimizations = true,
                    "reference_checks" => config.features.enable_reference_checks = true,
                    "aggressive_cache" => config.features.enable_aggressive_caching = true,
                    "batch_txid" => config.features.enable_batch_tx_id_computation = true,
                    "simd_hash" => config.features.enable_simd_hash_operations = true,
                    _ => {}
                }
            }
        }

        // Load advanced configuration
        if let Ok(val) = std::env::var("BLVM_CUSTOM_CHECKPOINTS") {
            // Parse comma-separated list of heights
            config.advanced.custom_checkpoints = val
                .split(',')
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .collect();
        }
        if let Ok(val) = std::env::var("BLVM_MAX_REORG_DEPTH") {
            if let Ok(depth) = val.parse::<u64>() {
                config.advanced.max_reorg_depth = depth;
            }
        }
        if let Ok(val) = std::env::var("BLVM_STRICT_MODE") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.advanced.strict_mode = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLVM_MAX_BLOCK_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                config.advanced.max_block_size_override = size;
            }
        }
        if let Ok(val) = std::env::var("BLVM_RBF") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.advanced.enable_rbf = enabled;
            }
        }

        config
    }

    /// Get assume-valid height (respects benchmarking override if enabled)
    #[cfg(feature = "production")]
    pub fn get_assume_valid_height(&self) -> u64 {
        // Check for benchmarking override first
        #[cfg(feature = "benchmarking")]
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);
            let override_val = OVERRIDE.load(Ordering::Relaxed);
            if override_val != u64::MAX {
                return override_val;
            }
        }

        self.block_validation.assume_valid_height
    }

    /// Get assume-valid height (non-production version)
    #[cfg(not(feature = "production"))]
    pub fn get_assume_valid_height(&self) -> u64 {
        self.block_validation.assume_valid_height
    }
}

/// Global consensus configuration (cached at first use).
///
/// Uses a single OnceLock — from_env() runs once, then we clone. No init_consensus_config;
/// the node can extend from_env (e.g. config file path in env) later if needed.
/// CRITICAL: Was re-running 50+ std::env::var() per block before caching.
static GLOBAL_CONSENSUS_CONFIG: std::sync::OnceLock<ConsensusConfig> = std::sync::OnceLock::new();

/// Initialize global consensus configuration (optional, for tests or future node use).
///
/// If called before any get_consensus_config(), overrides the default from-env config.
#[allow(dead_code)] // Reserved for when node loads config from file
pub fn init_consensus_config(config: ConsensusConfig) {
    let _ = GLOBAL_CONSENSUS_CONFIG.set(config);
}

/// Get global consensus configuration by reference (cached; no clone).
///
/// Prefer this over [`get_consensus_config`] in hot paths to avoid cloning.
pub fn get_consensus_config_ref() -> &'static ConsensusConfig {
    GLOBAL_CONSENSUS_CONFIG.get_or_init(ConsensusConfig::from_env)
}

/// Get global consensus configuration (cached; clone for compatibility).
pub fn get_consensus_config() -> ConsensusConfig {
    get_consensus_config_ref().clone()
}

/// Assume-valid height (cached). Used by block and script hot paths.
/// Runtime override for assume-valid height, shared between get/set.
/// u64::MAX means "no override" (fall through to config).
#[cfg(all(feature = "production", feature = "benchmarking"))]
static ASSUME_VALID_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

/// Benchmarking: set_assume_valid_height() overrides when feature enabled.
pub fn get_assume_valid_height() -> u64 {
    #[cfg(all(feature = "production", feature = "benchmarking"))]
    {
        use std::sync::atomic::Ordering;
        let v = ASSUME_VALID_OVERRIDE.load(Ordering::Relaxed);
        if v != u64::MAX {
            return v;
        }
    }
    get_consensus_config_ref()
        .block_validation
        .assume_valid_height
}

/// Assume-valid block hash. When set, block at assume_valid_height must match.
pub fn get_assume_valid_hash() -> Option<[u8; 32]> {
    get_consensus_config_ref()
        .block_validation
        .assume_valid_hash
}

/// Minimum chain work. Skip only when best_header_chainwork >= this.
pub fn get_n_minimum_chain_work() -> u128 {
    get_consensus_config_ref()
        .block_validation
        .n_minimum_chain_work
}

/// Set assume-valid height for benchmarking (overrides config).
#[cfg(all(feature = "production", feature = "benchmarking"))]
pub fn set_assume_valid_height(height: u64) {
    use std::sync::atomic::Ordering;
    ASSUME_VALID_OVERRIDE.store(height, Ordering::Relaxed);
}

/// Reset assume-valid override (benchmarking).
#[cfg(all(feature = "production", feature = "benchmarking"))]
pub fn reset_assume_valid_height() {
    set_assume_valid_height(u64::MAX);
}

/// Use overlay delta for UTXO merge instead of sync_block_to_batch.
/// Always enabled. connect_block_ibd returns UtxoDelta for the node to apply to pending_writes
/// without re-walking the block.
pub fn use_overlay_delta() -> bool {
    true
}

/// Initialize Rayon thread pool for script verification.
///
/// Call this at node startup before any block validation.
/// - When `script_verification_threads` > 0: use that value explicitly.
/// - When 0: let Rayon use its default (respects RAYON_NUM_THREADS env; typically num_cpus).
///   IBD scripts set RAYON_NUM_THREADS=nproc-1 for par-1 workers.
///
/// Only takes effect once per process.
#[cfg(all(feature = "production", feature = "rayon"))]
pub fn init_rayon_for_script_verification() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let config = get_consensus_config_ref();
        let n = config.performance.script_verification_threads;
        if n > 0 {
            if let Err(e) = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
            {
                eprintln!(
                    "Warning: Failed to set Rayon script verification pool to {n} threads: {e}. Using default."
                );
            }
        }
        // n==0: Rayon uses default pool (reads RAYON_NUM_THREADS if set)
    });
}
