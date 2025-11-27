//! Configuration for bllvm-consensus
//!
//! Provides configurable parameters for consensus validation, network message limits,
//! and performance optimizations. These settings can be loaded from config files,
//! environment variables, or passed programmatically.

use serde::{Deserialize, Serialize};

/// Network message size limits configuration
///
/// These limits protect against DoS attacks by bounding the size of network messages.
/// All limits match Bitcoin Core's protocol limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkMessageLimits {
    /// Maximum addresses in an addr message (Bitcoin Core: 1000)
    #[serde(default = "default_max_addr_addresses")]
    pub max_addr_addresses: usize,

    /// Maximum inventory items in inv/getdata messages (Bitcoin Core: 50000)
    #[serde(default = "default_max_inv_items")]
    pub max_inv_items: usize,

    /// Maximum headers in a headers message (Bitcoin Core: 2000)
    #[serde(default = "default_max_headers")]
    pub max_headers: usize,

    /// Maximum user agent length in version message (Bitcoin Core: 256 bytes)
    #[serde(default = "default_max_user_agent_length")]
    pub max_user_agent_length: usize,
}

fn default_max_addr_addresses() -> usize {
    1000
}

fn default_max_inv_items() -> usize {
    50000
}

fn default_max_headers() -> usize {
    2000
}

fn default_max_user_agent_length() -> usize {
    256
}

impl Default for NetworkMessageLimits {
    fn default() -> Self {
        Self {
            max_addr_addresses: 1000,
            max_inv_items: 50000,
            max_headers: 2000,
            max_user_agent_length: 256,
        }
    }
}

/// Block validation configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockValidationConfig {
    /// Assume-valid height: blocks before this height skip signature verification
    /// (Bitcoin Core's -assumevalid parameter)
    /// Default: 0 (validate all blocks - safest option)
    #[serde(default)]
    pub assume_valid_height: u64,

    /// Number of recent headers required for median time-past calculation (BIP113)
    /// Default: 11 (Bitcoin Core standard)
    #[serde(default = "default_median_time_past_headers")]
    pub median_time_past_headers: usize,

    /// Enable parallel transaction validation (requires production feature)
    #[serde(default = "default_true")]
    pub enable_parallel_validation: bool,

    /// Coinbase maturity requirement override (for testing only)
    /// Default: 0 (use consensus constant: 100 blocks)
    /// WARNING: Changing this may cause consensus divergence
    #[serde(default)]
    pub coinbase_maturity_override: u64,

    /// Maximum block sigop cost override (for testing only)
    /// Default: 0 (use consensus constant: 80,000)
    /// WARNING: Changing this may cause consensus divergence
    #[serde(default)]
    pub max_block_sigops_cost_override: u64,
}

/// Mempool configuration (Bitcoin Core parity)
///
/// Controls mempool size limits, fee rates, and transaction expiry.
/// These are operational parameters, not consensus-critical.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MempoolConfig {
    /// Maximum mempool size in megabytes (Bitcoin Core: -maxmempool, default 300 MB)
    /// Default: 300 MB
    #[serde(default = "default_max_mempool_mb")]
    pub max_mempool_mb: u64,

    /// Maximum number of transactions in mempool (alternative to size-based limit)
    /// Default: 100000 (Bitcoin Core uses size-based limit primarily)
    #[serde(default = "default_max_mempool_txs")]
    pub max_mempool_txs: usize,

    /// Mempool transaction expiry in hours (Bitcoin Core: -mempool expiry, default 336 hours = 14 days)
    /// Transactions older than this are removed from mempool
    /// Default: 336 (14 days)
    #[serde(default = "default_mempool_expiry_hours")]
    pub mempool_expiry_hours: u64,

    /// Minimum relay fee rate in satoshis per virtual byte (Bitcoin Core: -minrelaytxfee, default 1000 sat/kB = 1 sat/vB)
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
    /// Default: 1000 satoshis (Bitcoin Core standard)
    #[serde(default = "default_rbf_fee_increment")]
    pub rbf_fee_increment: i64,
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

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            max_mempool_mb: 300,
            max_mempool_txs: 100_000,
            mempool_expiry_hours: 336,
            min_relay_fee_rate: 1,
            min_tx_fee: 1000,
            rbf_fee_increment: 1000,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads for script verification (Bitcoin Core: -par, default: number of CPU cores)
    /// Default: 0 (auto-detect from CPU count)
    #[serde(default)]
    pub script_verification_threads: usize,

    /// Batch size for parallel transaction validation
    /// Larger batches improve throughput but increase latency
    /// Default: 8 transactions per batch
    #[serde(default = "default_parallel_batch_size")]
    pub parallel_batch_size: usize,

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

    /// Enable bounds check optimizations (requires Kani proofs)
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

fn default_median_time_past_headers() -> usize {
    11
}

fn default_true() -> bool {
    true
}

impl Default for BlockValidationConfig {
    fn default() -> Self {
        Self {
            assume_valid_height: 0,
            median_time_past_headers: 11,
            enable_parallel_validation: true,
            coinbase_maturity_override: 0,
            max_block_sigops_cost_override: 0,
        }
    }
}

/// Complete consensus configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ConsensusConfig {
    /// Network message size limits
    #[serde(default)]
    pub network_limits: NetworkMessageLimits,

    /// Block validation configuration
    #[serde(default)]
    pub block_validation: BlockValidationConfig,

    /// Mempool configuration (Bitcoin Core parity)
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

    /// UTXO commitments module configuration (if feature enabled)
    /// This includes peer consensus, spam filtering, and sync mode
    #[cfg(feature = "utxo-commitments")]
    #[serde(default)]
    pub utxo_commitments: Option<crate::utxo_commitments::UtxoCommitmentsConfig>,
}

impl ConsensusConfig {
    /// Load configuration from environment variables
    ///
    /// Environment variables follow the pattern: `BLLVM_CONSENSUS_<SECTION>_<KEY>`
    ///
    /// Examples:
    /// - `BLLVM_CONSENSUS_BLOCK_VALIDATION_ASSUME_VALID_HEIGHT=700000`
    /// - `BLLVM_CONSENSUS_NETWORK_LIMITS_MAX_HEADERS=2000`
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Load assume-valid height from environment (backward compatibility)
        if let Ok(val) = std::env::var("ASSUME_VALID_HEIGHT") {
            if let Ok(height) = val.parse::<u64>() {
                config.block_validation.assume_valid_height = height;
            }
        }

        // Load from new environment variable format
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_BLOCK_VALIDATION_ASSUME_VALID_HEIGHT") {
            if let Ok(height) = val.parse::<u64>() {
                config.block_validation.assume_valid_height = height;
            }
        }

        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_BLOCK_VALIDATION_MEDIAN_TIME_PAST_HEADERS")
        {
            if let Ok(count) = val.parse::<usize>() {
                config.block_validation.median_time_past_headers = count;
            }
        }
        if let Ok(val) =
            std::env::var("BLLVM_CONSENSUS_BLOCK_VALIDATION_ENABLE_PARALLEL_VALIDATION")
        {
            if let Ok(enabled) = val.parse::<bool>() {
                config.block_validation.enable_parallel_validation = enabled;
            }
        }
        if let Ok(val) =
            std::env::var("BLLVM_CONSENSUS_BLOCK_VALIDATION_COINBASE_MATURITY_OVERRIDE")
        {
            if let Ok(maturity) = val.parse::<u64>() {
                config.block_validation.coinbase_maturity_override = maturity;
            }
        }
        if let Ok(val) =
            std::env::var("BLLVM_CONSENSUS_BLOCK_VALIDATION_MAX_BLOCK_SIGOPS_COST_OVERRIDE")
        {
            if let Ok(cost) = val.parse::<u64>() {
                config.block_validation.max_block_sigops_cost_override = cost;
            }
        }

        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_NETWORK_LIMITS_MAX_ADDR_ADDRESSES") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_addr_addresses = limit;
            }
        }

        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_NETWORK_LIMITS_MAX_INV_ITEMS") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_inv_items = limit;
            }
        }

        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_NETWORK_LIMITS_MAX_HEADERS") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_headers = limit;
            }
        }

        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_NETWORK_LIMITS_MAX_USER_AGENT_LENGTH") {
            if let Ok(limit) = val.parse::<usize>() {
                config.network_limits.max_user_agent_length = limit;
            }
        }

        // Load mempool configuration (Bitcoin Core parity)
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_MEMPOOL_MAX_MEMPOOL_MB") {
            if let Ok(mb) = val.parse::<u64>() {
                config.mempool.max_mempool_mb = mb;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_MEMPOOL_MAX_MEMPOOL_TXS") {
            if let Ok(count) = val.parse::<usize>() {
                config.mempool.max_mempool_txs = count;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_MEMPOOL_EXPIRY_HOURS") {
            if let Ok(hours) = val.parse::<u64>() {
                config.mempool.mempool_expiry_hours = hours;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_MEMPOOL_MIN_RELAY_FEE_RATE") {
            if let Ok(rate) = val.parse::<u64>() {
                config.mempool.min_relay_fee_rate = rate;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_MEMPOOL_MIN_TX_FEE") {
            if let Ok(fee) = val.parse::<i64>() {
                config.mempool.min_tx_fee = fee;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_MEMPOOL_RBF_FEE_INCREMENT") {
            if let Ok(increment) = val.parse::<i64>() {
                config.mempool.rbf_fee_increment = increment;
            }
        }

        // Load UTXO commitment configuration
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_UTXO_COMMITMENT_MAX_SET_MB") {
            if let Ok(mb) = val.parse::<u64>() {
                config.utxo_commitment.max_utxo_commitment_set_mb = mb;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_UTXO_COMMITMENT_MAX_UTXO_COUNT") {
            if let Ok(count) = val.parse::<u64>() {
                config.utxo_commitment.max_utxo_count = count;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_UTXO_COMMITMENT_MAX_HISTORICAL") {
            if let Ok(count) = val.parse::<usize>() {
                config.utxo_commitment.max_historical_commitments = count;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_UTXO_COMMITMENT_ENABLE_INCREMENTAL_UPDATES")
        {
            if let Ok(enabled) = val.parse::<bool>() {
                config.utxo_commitment.enable_incremental_updates = enabled;
            }
        }

        // Load performance configuration
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_PERFORMANCE_SCRIPT_VERIFICATION_THREADS") {
            if let Ok(threads) = val.parse::<usize>() {
                config.performance.script_verification_threads = threads;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_PERFORMANCE_PARALLEL_BATCH_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                config.performance.parallel_batch_size = size;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_PERFORMANCE_ENABLE_SIMD") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.performance.enable_simd_optimizations = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_PERFORMANCE_ENABLE_CACHE_OPTIMIZATIONS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.performance.enable_cache_optimizations = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_PERFORMANCE_ENABLE_BATCH_UTXO_LOOKUPS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.performance.enable_batch_utxo_lookups = enabled;
            }
        }

        // Load debug configuration
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_DEBUG_ENABLE_RUNTIME_ASSERTIONS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.debug.enable_runtime_assertions = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_DEBUG_ENABLE_RUNTIME_INVARIANTS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.debug.enable_runtime_invariants = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_DEBUG_ENABLE_VERBOSE_LOGGING") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.debug.enable_verbose_logging = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_DEBUG_ENABLE_PERFORMANCE_PROFILING") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.debug.enable_performance_profiling = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_DEBUG_LOG_REJECTIONS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.debug.log_rejections = enabled;
            }
        }

        // Load feature flags
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_FEATURES_ENABLE_EXPERIMENTAL_OPTIMIZATIONS")
        {
            if let Ok(enabled) = val.parse::<bool>() {
                config.features.enable_experimental_optimizations = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_FEATURES_ENABLE_BOUNDS_CHECK_OPTIMIZATIONS")
        {
            if let Ok(enabled) = val.parse::<bool>() {
                config.features.enable_bounds_check_optimizations = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_FEATURES_ENABLE_REFERENCE_CHECKS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.features.enable_reference_checks = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_FEATURES_ENABLE_AGGRESSIVE_CACHING") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.features.enable_aggressive_caching = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_FEATURES_ENABLE_BATCH_TX_ID_COMPUTATION") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.features.enable_batch_tx_id_computation = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_FEATURES_ENABLE_SIMD_HASH_OPERATIONS") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.features.enable_simd_hash_operations = enabled;
            }
        }

        // Load advanced configuration
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_ADVANCED_CUSTOM_CHECKPOINTS") {
            // Parse comma-separated list of heights
            config.advanced.custom_checkpoints = val
                .split(',')
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .collect();
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_ADVANCED_MAX_REORG_DEPTH") {
            if let Ok(depth) = val.parse::<u64>() {
                config.advanced.max_reorg_depth = depth;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_ADVANCED_STRICT_MODE") {
            if let Ok(enabled) = val.parse::<bool>() {
                config.advanced.strict_mode = enabled;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_ADVANCED_MAX_BLOCK_SIZE_OVERRIDE") {
            if let Ok(size) = val.parse::<usize>() {
                config.advanced.max_block_size_override = size;
            }
        }
        if let Ok(val) = std::env::var("BLLVM_CONSENSUS_ADVANCED_ENABLE_RBF") {
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

/// Global consensus configuration instance
///
/// This is initialized from environment variables or config file at startup.
/// All consensus functions should use this for configurable parameters.
static GLOBAL_CONSENSUS_CONFIG: std::sync::OnceLock<ConsensusConfig> = std::sync::OnceLock::new();

/// Initialize global consensus configuration
///
/// This should be called once at startup, before any consensus validation.
/// If not called, defaults will be used.
pub fn init_consensus_config(config: ConsensusConfig) {
    GLOBAL_CONSENSUS_CONFIG
        .set(config)
        .expect("Consensus config already initialized");
}

/// Get global consensus configuration
///
/// Returns the global config if initialized, otherwise returns defaults.
pub fn get_consensus_config() -> ConsensusConfig {
    GLOBAL_CONSENSUS_CONFIG
        .get()
        .cloned()
        .unwrap_or_else(ConsensusConfig::from_env)
}
