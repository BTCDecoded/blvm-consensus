//! IBD Hardware Tuning
//!
//! Derives batch verification and parallelization parameters from hardware
//! (CPU count, cache size). Used when config does not supply explicit overrides.
//!
//! Precedence: Config override (if set) > Hardware-derived > Hardcoded default

use std::sync::OnceLock;

/// Hardware profile detected at first use.
#[derive(Debug, Clone)]
pub struct IbdHardwareProfile {
    /// From std::thread::available_parallelism()
    pub num_threads: usize,
    /// L3 cache size in KB (None if unknown)
    pub l3_cache_kb: Option<u64>,
    /// Many-core system (16+ logical cores)
    pub is_many_core: bool,
}

static HARDWARE_PROFILE: OnceLock<IbdHardwareProfile> = OnceLock::new();

fn detect_hardware() -> IbdHardwareProfile {
    let num_threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .max(1);

    let l3_cache_kb = detect_l3_cache_kb();
    let is_many_core = num_threads >= 16;

    IbdHardwareProfile {
        num_threads,
        l3_cache_kb,
        is_many_core,
    }
}

/// Detect L3 cache size on Linux via /sys. Returns None on non-Linux or if unreadable.
#[cfg(target_os = "linux")]
fn detect_l3_cache_kb() -> Option<u64> {
    use std::fs;
    use std::path::Path;

    let path = Path::new("/sys/devices/system/cpu/cpu0/cache/index3/size");
    if !path.exists() {
        return None;
    }
    let s = fs::read_to_string(path).ok()?.trim().to_string();
    let (num, suffix) = s.split_at(s.len().saturating_sub(1));
    let num: u64 = num.trim().parse().ok()?;
    let mult = match suffix {
        "K" | "k" => 1u64,
        "M" | "m" => 1024,
        _ => 1,
    };
    Some(num * mult)
}

#[cfg(not(target_os = "linux"))]
fn detect_l3_cache_kb() -> Option<u64> {
    None
}

fn hardware_profile() -> &'static IbdHardwareProfile {
    HARDWARE_PROFILE.get_or_init(detect_hardware)
}

/// libsecp256k1 thresholds: n<64 uses ecmult_multi_simple_var (slow), n>=64 Strauss, n>=88 Pippenger.
/// Chunks of 64-128 use Strauss; chunks of 89+ use Pippenger.
pub const STRAUSS_MIN: usize = 64;
pub const PIPPENGER_MIN_CHUNK: usize = 88;

/// Chunk threshold: single batch when n <= this. Above = split for parallelism.
/// Precedence: config_override > env > hardware-derived > 128.
/// Hardware-derived: many-core (16+) → 96 (more parallelism); few-core → 128.
pub fn chunk_threshold_config_or_hardware(config_override: Option<usize>) -> usize {
    config_override
        .or_else(|| {
            std::env::var("BLVM_CONSENSUS_PERFORMANCE_IBD_CHUNK_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&n: &usize| n > 0 && n <= 1024)
        })
        .unwrap_or_else(|| {
            let p = hardware_profile();
            if p.is_many_core {
                96 // more parallelism for ECDSA split
            } else {
                128
            }
        })
}

/// Min chunk size when splitting for parallelism. 128+ uses Pippenger (2-3× faster).
/// Precedence: config_override > env > hardware-derived > 128.
/// Hardware-derived: many-core → 64 (Strauss threshold); few-core → 128.
pub fn min_chunk_size_config_or_hardware(config_override: Option<usize>) -> usize {
    config_override
        .or_else(|| {
            std::env::var("BLVM_CONSENSUS_PERFORMANCE_IBD_MIN_CHUNK_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&n: &usize| n > 0 && n <= 512)
        })
        .unwrap_or_else(|| {
            let p = hardware_profile();
            if p.is_many_core {
                64 // Strauss threshold; more chunks for parallelism
            } else {
                128
            }
        })
}

/// Compute ECDSA batch chunk ranges for parallel verification.
/// Single source of truth for verify_soa_batch and SegQueue verify_batch.
/// Returns vec![(0, n)] when n <= chunk_threshold (single batch).
#[cfg(feature = "production")]
pub fn compute_ecdsa_batch_chunk_ranges(
    n: usize,
    chunk_threshold: usize,
    min_chunk: usize,
    num_threads: usize,
) -> Vec<(usize, usize)> {
    if n <= chunk_threshold {
        return vec![(0, n)];
    }
    let min_chunk_pippenger = if n >= 256 { 128usize.max(min_chunk) } else { 64usize.max(min_chunk) };
    let max_chunks = n / min_chunk_pippenger;
    let target_chunks = if n > 1024 {
        (4 * num_threads).min(max_chunks)
    } else if n > 512 {
        (2 * num_threads).min(max_chunks)
    } else {
        num_threads.min(max_chunks)
    };
    let num_chunks = target_chunks.max(1);
    compute_chunk_ranges(n, num_chunks, min_chunk_pippenger)
}

/// Compute optimal chunk ranges for parallel batch verification.
/// Splits n sigs into num_chunks such that each chunk has >= min_chunk sigs.
/// min_chunk >= 1; smaller chunks use ecmult_multi_simple_var (n<64) but parallelism often wins.
pub fn compute_chunk_ranges(n: usize, num_chunks: usize, min_chunk: usize) -> Vec<(usize, usize)> {
    debug_assert!(num_chunks >= 1 && min_chunk >= 1);
    if num_chunks == 1 {
        return vec![(0, n)];
    }
    // Balanced split: base_size = n / num_chunks, first (n % num_chunks) chunks get +1
    let base_size = n / num_chunks;
    let remainder = n % num_chunks;
    let mut ranges = Vec::with_capacity(num_chunks);
    let mut start = 0;
    for i in 0..num_chunks {
        let chunk_len = base_size + if i < remainder { 1 } else { 0 };
        if chunk_len > 0 {
            ranges.push((start, start + chunk_len));
            start += chunk_len;
        }
    }
    debug_assert_eq!(start, n);
    ranges
}

/// Chunk size for batch hash operations (SHA256, HASH160). Cache-friendly, fits in L1.
/// Hardware-derived from L3 when known (L3/256 clamped 8–32); otherwise 16.
/// Used by simd_vectorization for batch hashing.
pub fn hash_batch_chunk_size() -> usize {
    let p = hardware_profile();
    let from_l3 = p.l3_cache_kb.map(|kb| (kb / 256) as usize);
    let derived = from_l3.unwrap_or(16);
    derived.clamp(8, 32)
}
