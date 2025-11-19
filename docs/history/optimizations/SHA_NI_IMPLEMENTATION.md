# Intel SHA-NI Implementation Plan

**Goal**: 10x faster single-hash performance without touching existing AVX2 batch code  
**Effort**: ~4-5 hours  
**Risk**: Low (additive only)

---

## The Problem

- Current single-hash: ~84ns (54x slower than Core's ~1.57ns)
- Current batch (128): 172µs with AVX2 ✅ (already optimal, 2.84x speedup)

## The Solution

Add Intel SHA Extensions (SHA-NI) for single hashes while keeping AVX2 for batches.

**Result**: Single hash ~15ns (3-4x slower than Core instead of 54x)

---

## Implementation

### 1. New File: `src/crypto/sha_ni.rs` (~100 lines)

```rust
//! Intel SHA Extensions (SHA-NI) for single-hash optimization

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn is_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    { std::arch::is_x86_feature_detected!("sha") }
    
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
unsafe fn sha256_impl(data: &[u8]) -> [u8; 32] {
    // Use Intel intrinsics:
    // - _mm_sha256rnds2_epu32()
    // - _mm_sha256msg1_epu32()
    // - _mm_sha256msg2_epu32()
    
    // Implementation based on Intel reference code
    // See: https://github.com/noloader/SHA-Intrinsics
    todo!("Implement using Intel's reference")
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_available() {
            unsafe { sha256_impl(data) }
        } else {
            fallback_sha256(data)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    { fallback_sha256(data) }
}

fn fallback_sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    let mut result = [0u8; 32];
    result.copy_from_slice(&hash);
    result
}

pub fn hash256(data: &[u8]) -> [u8; 32] {
    let first = sha256(data);
    sha256(&first)
}
```

### 2. Update: `src/crypto/mod.rs` (~10 lines)

```rust
// Add module declaration
#[cfg(target_arch = "x86_64")]
pub mod sha_ni;

// Update OptimizedSha256::hash() method
impl OptimizedSha256 {
    pub fn hash(&self, data: &[u8]) -> [u8; 32] {
        #[cfg(target_arch = "x86_64")]
        if sha_ni::is_available() {
            return sha_ni::sha256(data);
        }
        
        // Fallback unchanged
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash);
        result
    }
}
```

### 3. Tests (~30 lines)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_sha_ni_matches_reference() {
        if !sha_ni::is_available() { return; }
        
        let input = b"hello world";
        let result = sha_ni::sha256(input);
        
        // Must match sha2 crate
        use sha2::{Digest, Sha256};
        let expected = Sha256::digest(input);
        assert_eq!(&result[..], &expected[..]);
    }
}
```

---

## What's NOT Changing

- ✅ `src/crypto/sha256_avx2.rs` - Untouched (1233 lines)
- ✅ `src/crypto/avx2_batch.rs` - Untouched
- ✅ Batch performance - Unchanged (2.84x speedup maintained)

---

## Validation Steps

1. **Check CPU support**: `cat /proc/cpuinfo | grep sha_ni`
2. **Implement SHA-NI**: Use Intel's reference code
3. **Test correctness**: Compare against sha2 crate
4. **Benchmark**: Verify 10x improvement
5. **Regression test**: Ensure batch performance unchanged

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Single hash | 84ns | ~15ns | **5-6x faster** |
| Gap vs Core | 54x | 3-4x | **17x better** |
| Batch (128) | 172µs | 172µs | **Unchanged** ✅ |

---

## References

- [Intel SHA-NI White Paper](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sha-extensions.html)
- [Example Implementation](https://github.com/noloader/SHA-Intrinsics)
- [Bitcoin Core's approach](https://github.com/bitcoin/bitcoin/blob/master/src/crypto/sha256.cpp)

