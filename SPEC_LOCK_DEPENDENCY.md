# blvm-spec-lock Dependency Resolution

This document explains how `blvm-consensus` resolves the `blvm-spec-lock` dependency across different environments.

## Dependency Resolution Priority

1. **Default (Production / CI Rust build)**: Use the crate from [crates.io](https://crates.io/crates/blvm-spec-lock)
   - `blvm-spec-lock = ">=0.1.3, <1"` in `Cargo.toml` (proc-macro / `#[spec_locked]`)
   - CI strips `[patch.crates-io]` so resolution matches published crates

2. **Local development**: Optional path override
   - Create `.cargo/config.toml` (not committed) with `[patch.crates-io]` pointing at a local clone of [blvm-spec-lock](https://github.com/BTCDecoded/blvm-spec-lock) (e.g. sibling directory `../blvm-spec-lock`)
   - See `.cargo/config.toml.example` if present

## `cargo-spec-lock` CLI (verification)

Formal verification runs **`cargo-spec-lock verify`** (Z3-backed when built with `--features z3`).

- **CI** (`.github/workflows/ci.yml`, Verify job): clones **[blvm-spec](https://github.com/BTCDecoded/blvm-spec)** next to the crate with `uses: BTCDecoded/rust-ci/setup-blvm-spec@main`, then installs the tool from crates.io:

  `cargo install blvm-spec-lock --version '>=0.1.3, <0.2' --locked --features z3`

  System packages for Z3/libclang are installed on the runner (apt/pacman) before `cargo install`.

- **Local**: Same command, or build from a `blvm-spec-lock` git checkout if you need an unreleased tool.

**CLI version:** `cargo-spec-lock --version` prints the crate version (useful in CI logs and attestations).

**`check-drift`:** **`--scoped-unparseables`** limits failure to **unparseable** Orange Paper **Properties** in sections that match a **`#[spec_locked]`** section prefix (and optional function name) in the scanned crate (dot-separated prefix, e.g. lock **`5.1`** includes **`5.1.2`**). **blvm-consensus** CI and the monorepo **`verify.yml`** workflow pass **`--scoped-unparseables`** so widening locked call-sites does not require fixing every unrelated unparseable bullet in one PR. Omit **`--scoped-unparseables`** for a **global** unparseable scan.

**`SPEC_LOCK_STRICT`:** When set to `1` / `true` / `yes`, verification fails on **Partial** or unsupported contracts (same effect as **`--strict`** on the CLI).

**`SPEC_LOCK_Z3_TIMEOUT_SECS`:** When set to a positive integer, overrides the per-function Z3 **`--timeout`** (seconds). Use when obligations are **richer** and the solver returns **Unknown** / **Partial** under the default timeout.

**Local verify (CLI only):** From the **`blvm-consensus`** crate root, with the Orange Paper on disk (sibling **`../blvm-spec`** or **`SPEC_LOCK_SPEC_PATH`**):

```bash
cargo spec-lock verify --verbose --crate-path . --spec-path "${SPEC_LOCK_SPEC_PATH:-../blvm-spec/PROTOCOL.md}"
```

CI equivalents live under **`.github/workflows`** (install **`cargo-spec-lock`** with Z3, then the same **`verify`** invocation).

## Implementation Details

### Cargo.toml

The library dependency points to crates.io; optional `[patch.crates-io]` in a local config overrides when you use path overrides during development.

### Where the Orange Paper markdown lives

Canonical spec repository: **[github.com/BTCDecoded/blvm-spec](https://github.com/BTCDecoded/blvm-spec)** (`PROTOCOL.md`, `ARCHITECTURE.md`, `THE_ORANGE_PAPER.md`).

**There is no copy of the spec inside `blvm-consensus`.** At `#[spec_locked]` expansion, `blvm-spec-lock` looks for **`../blvm-spec/PROTOCOL.md`** and **`../blvm-spec/ARCHITECTURE.md`** (or **`THE_ORANGE_PAPER.md`** there) relative to this crate’s `CARGO_MANIFEST_DIR`, or uses **`SPEC_LOCK_SPEC_PATH`** if set. Clone **blvm-spec** as a **sibling** of **blvm-consensus** (so `../blvm-spec` exists), or point **`SPEC_LOCK_SPEC_PATH`** at those files.

### Notes

- The `local-spec-lock` feature in `Cargo.toml` is reserved for optional local path workflows
- To see what Cargo resolves: `cargo tree -i blvm-spec-lock`

## Attestation and `verify` log shape

The **blvm-consensus** CI job records **`spec_lock_output.txt`** from **`cargo-spec-lock verify`** and, on **`main`**, uploads it with **`spec_lock_verify_meta.env`** for release attestation (**`cargo-spec-lock`** version, **`git` SHA** of **`../blvm-spec`**, optional **`SPEC_META.json`** **`version`** when **`jq`** is available). Build metadata steps count “how many things passed” with **`grep -c "Status: PASSED"`** on **`spec_lock_output.txt`**.

That count is **the number of lines containing `Status: PASSED`** in the tool’s text output, not a guaranteed “one line per `#[spec_locked]` function” or “total contracts proved.” A richer Orange Paper (more parseable **Properties** per function) can increase **contracts per function** without changing the same **PASSED**-line tally, depending on how **`cargo-spec-lock`** formats its report. Treat the hash of **`spec_lock_output.txt`** and the **verify** exit code as the primary integrity signals; treat **PASSED** counts as a **trending / auxiliary** metric unless the output format is versioned as a stable API.

## §5.2.6 vs `connect_block` (Orange Paper vs Bitcoin Core)

Orange Paper §5.2.6 defines a piecewise **`GetBlockScriptFlags`**: if the block hash is in **`ScriptFlagExceptions`**, use the table value; otherwise use per-transaction **`CalculateScriptFlags`**. **`#[spec_locked("5.2.6", "GetBlockScriptFlags")]`** on `get_block_script_flags` tracks that formulation.

**`connect_block`** instead uses **`get_block_script_verify_flags_core`**, which matches Bitcoin **`GetBlockScriptFlags`** in **`validation.cpp`**: default **`SCRIPT_VERIFY_P2SH | WITNESS | TAPROOT`**, replace on exception, then OR buried deployments (BIP66, BIP65, CSV, BIP147). The same block-level mask is passed to every non-coinbase script check, consistent with Core’s **`CheckInputScripts`**.
