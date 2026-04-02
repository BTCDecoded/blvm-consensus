//! Configuration for blvm-consensus
//!
//! Provides configurable parameters for consensus validation, network message limits,
//! and performance optimizations. These settings can be loaded from config files,
//! environment variables, or passed programmatically.

use serde::{Deserialize, Serialize};

// Re-export foundational config types from blvm-primitives
pub use blvm_primitives::config::{BlockValidationConfig, NetworkMessageLimits};

/// Mempool configuration
///
/// Controls mempool size limits, fee rates, and transaction expiry.
/// These are operational parameters, not consensus-critical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MempoolConfig {
    /// Maximum mempool size in megabytes (default 300 MB)
    /// Default: 300 MB
    #[serde(default = "default_max_mempool_mb")]
    pub max_mempool_mb: u64,

    /// Maximum number of transactions in mempool (alternative to size-based limit)
    /// Default: 100000
    #[serde(default = "default_max_mempool_txs")]
    pub max_mempool_txs: usize,

    /// Mempool transaction expiry in hours (default 336 = 14 days)
    /// Transactions older than this are removed from mempool
    /// Default: 336 (14 days)
    #[serde(default = "default_mempool_expiry_hours")]
    pub mempool_expiry_hours: u64,

    /// Minimum relay fee rate in satoshis per virtual byte (default 1 sat/vB)
    /// Transactions with fee rate below this are not relayed
    /// Default: 1 sat/vB (1000 sat/kB)
    #[serde(default = "default_min_relay_fee_rate")]
    pub min_relay_fee_rate: u64,

    /// Minimum transaction fee in satoshis (absolute minimum, regardless of size)
    /// Default: 1000 satoshis
    #[serde(default = "default_min_tx_fee")]
    pub min_tx_fee: i64,

    /// RBF (Replace-By-Fee) minimum fee increment in satoshis (BIP125)
    /// Replacement transactions must pay at least this much more than the original
    /// Default: 1000 satoshis
    #[serde(default = "default_rbf_fee_increment")]
    pub rbf_fee_increment: i64,

    /// Maximum OP_RETURN data size in bytes (default 80)
    /// Default: 80 bytes
    #[serde(default = "default_max_op_return_size")]
    pub max_op_return_size: u32,

    /// Maximum number of OP_RETURN outputs allowed (default: 1)
    /// Transactions with more than this are rejected as non-standard
    #[serde(default = "default_max_op_return_outputs")]
    pub max_op_return_outputs: u32,

    /// Reject transactions with multiple OP_RETURN outputs
    /// Default: true
    #[serde(default = "default_reject_multiple_op_return")]
    pub reject_multiple_op_return: bool,

    /// Maximum standard script size in bytes
    /// Default: 200 bytes
    #[serde(default = "default_max_standard_script_size")]
    pub max_standard_script_size: u32,

    /// Reject envelope protocol (OP_FALSE OP_IF) scripts
    /// Default: true
    #[serde(default = "default_reject_envelope_protocol")]
    pub reject_envelope_protocol: bool,

    /// Reject spam transactions at mempool entry (opt-in)
    /// Default: false (spam filtering is opt-in for mempool)
    ///
    /// **Admission:** enforced in `blvm-node` by `MempoolPolicyConfig::reject_spam_in_mempool`
    /// in `MempoolManager::add_transaction` (uses `blvm-protocol` spam filter).
    #[serde(default = "default_reject_spam_in_mempool")]
    pub reject_spam_in_mempool: bool,

    /// Spam filter configuration (if reject_spam_in_mempool is enabled)
    /// Note: Prefer `blvm-node` `MempoolPolicyConfig::spam_filter` + `SpamFilterConfigSerializable`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spam_filter_config: Option<serde_json::Value>,

    /// Minimum fee rate for large transactions (satoshis per vbyte)
    /// Transactions larger than large_tx_threshold_bytes must pay at least this fee rate
    /// Default: 2 sat/vB (higher than standard min_relay_fee_rate)
    #[serde(default = "default_min_fee_rate_large_tx")]
    pub min_fee_rate_large_tx: u64,

    /// Large transaction threshold (bytes)
    /// Transactions larger than this require min_fee_rate_large_tx
    /// Default: 1000 bytes
    #[serde(default = "default_large_tx_threshold_bytes")]
    pub large_tx_threshold_bytes: u64,
}

fn default_rbf_fee_increment() -> i64 {
    1000
}

fn default_max_mempool_mb() -> u64 {
    300
}

fn default_max_mempool_txs() -> usize {
    100_000
}

fn default_mempool_expiry_hours() -> u64 {
    336 // 14 days
}

fn default_min_relay_fee_rate() -> u64 {
    1 // 1 sat/vB = 1000 sat/kB
}

fn default_min_tx_fee() -> i64 {
    1000
}

fn default_max_op_return_size() -> u32 {
    80
}

fn default_max_op_return_outputs() -> u32 {
    1
}

fn default_reject_multiple_op_return() -> bool {
    true
}

fn default_max_standard_script_size() -> u32 {
    200
}

fn default_reject_envelope_protocol() -> bool {
    true
}

fn default_reject_spam_in_mempool() -> bool {
    false
}

fn default_min_fee_rate_large_tx() -> u64 {
    2 // 2 sat/vB (higher than standard 1 sat/vB)
}

fn default_large_tx_threshold_bytes() -> u64 {
    1000 // 1 KB
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_mempool_mb: 300,
            max_mempool_txs: 100_000,
            mempool_expiry_hours: 336,
            min_relay_fee_rate: 1,
            min_tx_fee: 1000,
            rbf_fee_increment: 1000,
            max_op_return_size: 80,
            max_op_return_outputs: 1,
            reject_multiple_op_return: true,
            max_standard_script_size: 200,
            reject_envelope_protocol: true,
            reject_spam_in_mempool: false,
            spam_filter_config: None,
            min_fee_rate_large_tx: 2,
            large_tx_threshold_bytes: 1000,
        }
    }
}

/// UTXO Commitment configuration
///
/// Controls UTXO commitment set size, storage limits, and performance tuning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UtxoCommitmentConfig {
    /// Maximum UTXO commitment set size in megabytes
    /// This limits the in-memory size of the UTXO Merkle tree
    /// Default: 512 MB (sufficient for ~100M UTXOs)
    #[serde(default = "default_max_utxo_commitment_set_mb")]
    pub max_utxo_commitment_set_mb: u64,

    /// Maximum number of UTXOs in commitment set (alternative to size-based limit)
    /// Default: 100_000_000 (100 million UTXOs)
    #[serde(default = "default_max_utxo_count")]
    pub max_utxo_count: u64,

    /// Maximum number of historical commitments to keep in memory
    /// Older commitments are stored on disk
    /// Default: 1000 (keeps last ~7 days of commitments at 1 per block)
    #[serde(default = "default_max_historical_commitments")]
    pub max_historical_commitments: usize,

    /// Enable incremental commitment updates (recommended)
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_incremental_updates: bool,
}

fn default_max_utxo_commitment_set_mb() -> u64 {
    512
}

fn default_max_utxo_count() -> u64 {
    100_000_000
}

fn default_max_historical_commitments() -> usize {
    1000
}

impl Default for UtxoCommitmentConfig {
    fn default() -> Self {
        Self {
            max_utxo_commitment_set_mb: 512,
            max_utxo_count: 100_000_000,
            max_historical_commitments: 1000,
            enable_incremental_updates: true,
        }
    }
}

/// Performance and optimization configuration
///
/// Controls performance tuning, parallelization, and optimization features.
/// These are operational parameters that affect performance but not consensus correctness.
///
/// IBD batch tuning: When `ibd_chunk_threshold` / `ibd_min_chunk_size` are `None`,
/// hardware-derived values are used. When `Some(x)`, config overrides hardware.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads for script verification (default: number of CPU cores)
    /// Default: 0 (auto-detect from CPU count)
    #[serde(default)]
    pub script_verification_threads: usize,

    /// Batch size for parallel transaction validation
    /// Larger batches improve throughput but increase latency
    /// Default: 8 transactions per batch
    #[serde(default = "default_parallel_batch_size")]
    pub parallel_batch_size: usize,

    /// IBD batch: chunk threshold (parallelize when sig count exceeds this).
    /// None = use hardware-derived; Some(x) = override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ibd_chunk_threshold: Option<usize>,

    /// IBD batch: minimum chunk size for parallel batches.
    /// None = use hardware-derived; Some(x) = override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ibd_min_chunk_size: Option<usize>,

    /// Enable SIMD/vectorization optimizations (if available)
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_simd_optimizations: bool,

    /// Enable cache-friendly memory layouts
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_cache_optimizations: bool,

    /// Enable batch UTXO lookups (pre-fetch all UTXOs before validation)
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_batch_utxo_lookups: bool,
}

fn default_parallel_batch_size() -> usize {
    8
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            script_verification_threads: 0, // Auto-detect
            parallel_batch_size: 8,
            ibd_chunk_threshold: None,
            ibd_min_chunk_size: None,
            enable_simd_optimizations: true,
            enable_cache_optimizations: true,
            enable_batch_utxo_lookups: true,
        }
    }
}

/// Debug and development configuration
///
/// Controls debug assertions, runtime checks, and development features.
/// These options are safe to enable in production but may impact performance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct DebugConfig {
    /// Enable runtime assertions (debug_assert! statements)
    /// Default: false (enabled automatically in debug builds)
    #[serde(default = "default_false")]
    pub enable_runtime_assertions: bool,

    /// Enable runtime invariant checks (additional safety checks)
    /// Default: false
    #[serde(default = "default_false")]
    pub enable_runtime_invariants: bool,

    /// Enable verbose logging for consensus operations
    /// Default: false
    #[serde(default = "default_false")]
    pub enable_verbose_logging: bool,

    /// Enable performance profiling (timing measurements)
    /// Default: false
    #[serde(default = "default_false")]
    pub enable_performance_profiling: bool,

    /// Log all rejected transactions/blocks (for debugging)
    /// Default: false
    #[serde(default = "default_false")]
    pub log_rejections: bool,
}

fn default_false() -> bool {
    false
}

/// Feature flags configuration
///
/// Controls optional features and experimental functionality.
/// These are safe to enable/disable without affecting consensus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureFlagsConfig {
    /// Enable experimental optimizations (may be unstable)
    /// Default: false
    #[serde(default = "default_false")]
    pub enable_experimental_optimizations: bool,

    /// Enable bounds check optimizations (requires formal proofs)
    /// Default: true (if production feature enabled)
    #[serde(default = "default_true")]
    pub enable_bounds_check_optimizations: bool,

    /// Enable reference implementation checks (slower but safer)
    /// Default: false
    #[serde(default = "default_false")]
    pub enable_reference_checks: bool,

    /// Enable aggressive caching (may use more memory)
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_aggressive_caching: bool,

    /// Enable batch transaction ID computation (faster but uses more memory)
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_batch_tx_id_computation: bool,

    /// Enable SIMD hash operations (faster on supported CPUs)
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_simd_hash_operations: bool,
}

impl Default for FeatureFlagsConfig {
    fn default() -> Self {
        Self {
            enable_experimental_optimizations: false,
            enable_bounds_check_optimizations: true,
            enable_reference_checks: false,
            enable_aggressive_caching: true,
            enable_batch_tx_id_computation: true,
            enable_simd_hash_operations: true,
        }
    }
}

/// Advanced configuration options
///
/// Advanced settings for power users and specific use cases.
/// These options provide fine-grained control over behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdvancedConfig {
    /// Custom checkpoint heights (additional to assume-valid)
    /// Format: comma-separated list of block heights
    /// Example: "100000,200000,300000"
    /// Default: empty (no custom checkpoints)
    #[serde(default)]
    pub custom_checkpoints: Vec<u64>,

    /// Maximum depth for chain reorganization (safety limit)
    /// Prevents extremely deep reorganizations that could be DoS attacks
    /// Default: 100 blocks
    #[serde(default = "default_max_reorg_depth")]
    pub max_reorg_depth: u64,

    /// Enable strict mode (reject any non-standard transactions)
    /// Default: false (accept standard transactions)
    #[serde(default = "default_false")]
    pub strict_mode: bool,

    /// Maximum block size to accept (override consensus limit for testing)
    /// Default: 0 (use consensus limit)
    /// WARNING: Setting this may cause consensus divergence
    #[serde(default)]
    pub max_block_size_override: usize,

    /// Enable transaction replacement (RBF) by default
    /// Default: true
    #[serde(default = "default_true")]
    pub enable_rbf: bool,
}

fn default_max_reorg_depth() -> u64 {
    100
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            custom_checkpoints: Vec::new(),
            max_reorg_depth: 100,
            strict_mode: false,
            max_block_size_override: 0,
            enable_rbf: true,
        }
    }
}

fn default_true() -> bool {
    true
}

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
/// Benchmarking: set_assume_valid_height() overrides when feature enabled.
pub fn get_assume_valid_height() -> u64 {
    #[cfg(all(feature = "production", feature = "benchmarking"))]
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);
        let v = OVERRIDE.load(Ordering::Relaxed);
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
    use std::sync::atomic::{AtomicU64, Ordering};
    static OVERRIDE: AtomicU64 = AtomicU64::new(u64::MAX);
    OVERRIDE.store(height, Ordering::Relaxed);
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
