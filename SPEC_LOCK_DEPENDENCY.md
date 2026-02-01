# blvm-spec-lock Dependency Resolution

This document explains how `blvm-consensus` resolves the `blvm-spec-lock` dependency across different environments.

## Dependency Resolution Priority

The system uses a three-tier dependency resolution strategy:

1. **Default (Production)**: Use crate from crates.io
   - `blvm-spec-lock = "^0.1.0"` in `Cargo.toml` (allows compatible updates: 0.1.x)
   - Used when the crate is published and available

2. **Fallback (CI)**: Clone GitHub repo and use path dependency
   - If crate not available on crates.io, CI clones the repo
   - Uses `[patch.crates-io]` section in `Cargo.toml` to override
   - Handled automatically by `scripts/setup-spec-lock.sh`

3. **Local Development**: Use local filesystem path
   - Create `.cargo/config.toml` (not committed to git)
   - Uses `[patch.crates-io]` to override with local path
   - See `.cargo/config.toml.example` for template

## Implementation Details

### Cargo.toml

```toml
# Default: Use crate from crates.io
# Version range allows compatible updates (^0.1.0 allows 0.1.x but not 0.2.x)
blvm-spec-lock = "^0.1.0"
```

The default dependency points to crates.io with a caret requirement, allowing compatible version updates within the same minor version (0.1.x).

### CI Workflow

All CI jobs run `scripts/setup-spec-lock.sh` which:

1. Checks if `blvm-spec-lock` exists on crates.io
2. If not found, clones the GitHub repo (`BTCDecoded/blvm-spec-lock`)
3. Patches `Cargo.toml` with `[patch.crates-io]` section to use local path

This ensures CI works even before the crate is published.

### Local Development

For local development:

1. Copy `.cargo/config.toml.example` to `.cargo/config.toml`
2. The config file uses `[patch.crates-io]` to override the crate dependency
3. `.cargo/config.toml` is in `.gitignore` (not committed)

Example `.cargo/config.toml`:
```toml
[patch.crates-io]
blvm-spec-lock = { path = "../blvm-spec-lock" }
```

## Verification

To verify which dependency source is being used:

```bash
# Check what Cargo resolves
cargo tree -i blvm-spec-lock

# Or check Cargo.toml for patch section
grep -A 2 "\[patch.crates-io\]" Cargo.toml
```

## Notes

- The `local-spec-lock` feature in `Cargo.toml` is reserved for future use
- CI automatically handles the fallback - no manual intervention needed
- Local development requires creating `.cargo/config.toml` manually
- The setup script is idempotent - safe to run multiple times

