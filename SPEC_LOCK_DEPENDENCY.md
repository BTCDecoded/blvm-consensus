# blvm-spec-lock Dependency Resolution

This document explains how `blvm-consensus` resolves the `blvm-spec-lock` dependency across different environments.

## Dependency Resolution Priority

1. **Default (Production / CI Rust build)**: Use the crate from [crates.io](https://crates.io/crates/blvm-spec-lock)
   - `blvm-spec-lock = ">=0.1.3, <1"` in `Cargo.toml` (proc-macro / `#[spec_locked]`)
   - CI strips `[patch.crates-io]` so resolution matches published crates

2. **Local development**: Optional path override
   - Create `.cargo/config.toml` (not committed) with `[patch.crates-io]` pointing at `../blvm-spec-lock`
   - See `.cargo/config.toml.example` if present

## `cargo-spec-lock` CLI (verification)

Formal verification runs **`cargo-spec-lock verify`** (Z3-backed when built with `--features z3`).

- **CI** (`.github/workflows/ci.yml`, Verify job): clones **blvm-spec** with `uses: BTCDecoded/rust-ci/setup-blvm-spec@main`, then installs the tool from crates.io:

  `cargo install blvm-spec-lock --locked --features z3`

  System packages for Z3/libclang are installed on the runner (apt/pacman) before `cargo install`.

- **Local**: Same command, or build from a `blvm-spec-lock` git checkout if you need an unreleased tool.

## Implementation Details

### Cargo.toml

The library dependency points to crates.io; optional `[patch.crates-io]` in a local config overrides for monorepo work.

### Notes

- The `local-spec-lock` feature in `Cargo.toml` is reserved for optional local path workflows
- To see what Cargo resolves: `cargo tree -i blvm-spec-lock`
