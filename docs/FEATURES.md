# Crate features

This document lists what the main optional features enable. Use it when reading or editing code under `#[cfg(feature = "production")]` or `#[cfg(feature = "rayon")]`.

## `production`

**Default:** yes (enabled in `default`).

Enables runtime and allocation optimizations used for full-node and high-throughput builds:

- **Batch script queue** — Script verification runs in a batch queue (`checkqueue`) instead of inline, improving cache locality and throughput.
- **Script / sig caches** — LRU caches for script execution and signature verification to avoid repeated work.
- **Assume-valid** — Optional skip of script verification up to a configured height (benchmarking/IBD).
- **FxHash / rustc-hash** — `UtxoSet` and related maps use `FxHashMap`/`FxHashSet` for 2–3× faster lookups than `std::HashMap` on fixed-size keys.
- **SmallVec** — Transaction inputs/outputs use `SmallVec` to reduce allocations for typical 1–2 input/output txs.
- **blvm-secp256k1** — Pure Rust ECDSA/Schnorr/Taproot backend (default when `production` is on).
- **hashbrown / raw_entry** — Allocation-free sighash cache lookups where supported.

Without `production`, the crate uses `std::HashMap`/`HashSet`, no batch queue, and the same consensus logic with simpler allocation behavior (useful for tests and verification).

## `rayon`

**Default:** enabled indirectly via `production` (production pulls in `rayon`).

Enables parallelization where consensus allows:

- **Parallel structure checks** — In block validation, structure checks (e.g. duplicate txids, ordering) can run in parallel over transactions when both `production` and `rayon` are enabled.
- **Parallel UTXO proof verification** — When `parallel-verification` is also enabled (e.g. for UTXO commitments).

Without `rayon`, the same validation runs sequentially. Consensus results are identical; only throughput and CPU use differ.

## Other features

- **`benchmarking`** — Cache clearing and state reset helpers for reproducible benchmarks.
- **`property-tests`** — Proptest strategies (e.g. `transaction_with_witness_strategy`) for integration tests.
- **`utxo-commitments`** — Placeholder feature for downstream crates; UTXO commitment logic and integration tests live in `blvm-protocol` (which depends on `blvm-consensus`, not the reverse).

## When editing `#[cfg]` code

- Group related `#[cfg(feature = "production")]` (and `rayon`) blocks so production-only paths are easy to see.
- Prefer a single feature-gated module or re-export over repeating `#[cfg(all(feature = "production", feature = "rayon"))]` in many places.
