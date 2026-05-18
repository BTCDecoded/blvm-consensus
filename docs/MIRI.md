# Miri and production consensus code

**Miri** (`cargo miri test`) is enabled in **`.github/workflows/ci.yml`** for selected **`blvm-consensus`** targets with **`--all-features`**.

## Why Miri does not replace production fuzzing

- Miri executes a Rust **abstract machine**. It does **not** model every CPU-specific behavior used behind **`cfg(feature = "production")`** (e.g. SIMD intrinsics, prefetch hints, some `unsafe` fast paths).
- A passing Miri run means **Rust-level UB** and aliasing rules hold under Miri’s model for the exercised tests — not that every optimized assembly path has been validated.

## How Miri complements other gates

- **`cargo test --all-features`** and **`cargo fuzz`** exercise production-feature **code selection** and integration paths on native builds.
- Miri adds **extra confidence** on logic that Miri can execute (especially pure-Rust interpretations of script/consensus helpers under test configurations).

## Running locally

Use the nightly toolchain required by Miri and match CI flags, e.g.:

```bash
cargo +nightly miri test --all-features --test consensus_property_tests --lib -- --test-threads=1
```

Adjust target names to match your checkout; see **`ci.yml`** for the canonical job definition.
