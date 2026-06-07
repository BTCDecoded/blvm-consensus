# blvm-spec-lock Dependency Resolution

This document explains how `blvm-consensus` resolves the `blvm-spec-lock` dependency across different environments.

## Dependency Resolution Priority

1. **Default (Production / CI Rust build)**: Use the crate from [crates.io](https://crates.io/crates/blvm-spec-lock)
   - `blvm-spec-lock = ">=0.1, <1"` in `Cargo.toml` (proc-macro / `#[spec_locked]`)
   - CI strips `[patch.crates-io]` so resolution matches published crates

2. **Local development**: Optional path override
   - Create `.cargo/config.toml` (not committed) with `[patch.crates-io]` pointing at a local clone of [blvm-spec-lock](https://github.com/BTCDecoded/blvm-spec-lock) (e.g. sibling directory `../blvm-spec-lock`)
   - See `.cargo/config.toml.example` if present

## `cargo-spec-lock` CLI (verification)

Formal verification runs **`cargo-spec-lock verify`** (Z3-backed when built with `--features z3`).

Consumer CI runs **`check-drift`** then **`cargo-spec-lock verify`** with **`--spec-path`** — **merged `F_*` registry** (static + optional Z3 SAT smoke) **before** **`#[spec_locked]`** Rust rows; nested **`formula_registry`** appears in **`spec_lock_*_verify.json`** (**`report_format` 1**); see **`blvm-spec-lock`** **`docs/VERIFY_JSON.md`** and **`schemas/verify_report_v1.json`**. Each consumer repo defines the exact **`Verify`**/`verify` job in its own **`.github/workflows/ci.yml`**; supplemental monorepo-only **`workflow_dispatch`** mirrors may exist under **`BTCDecoded/.github`** / umbrella checkouts (**not** substitutes for **`ci.yml`**).

- **CI (authoritative)** — Formal verification gates live in **`blvm-consensus/.github/workflows/ci.yml`** (**Verify** job), **`blvm-node/.github/workflows/ci.yml`** (**verify** job: crates.io **blvm-consensus** tarball + workspace node), and **`blvm-protocol/.github/workflows/ci.yml`** (**Verify** job — **PROTOCOL.md** / **ARCHITECTURE.md** via **`setup-blvm-spec`**). Each repo configures **self-hosted** runners as in its workflow YAML. Each installs:

  `cargo install blvm-spec-lock --version '>=0.1, <1' --locked --features z3`

  System packages for Z3/libclang are installed on the runner (apt/pacman) before `cargo install`.

- **Umbrella (supplemental)** — In a **multi-repo workspace** that includes **`.github/workflows/verify.yml`** or **`verify-network.yml`**, those workflows re-run **check-drift** (where present) + **verify** (merged **`F_*`** + Rust rows), plus non-blocking **`coverage --rollup-from-verify-json`**, for convenience; they **do not** replace **blvm-consensus**, **blvm-node**, or **blvm-protocol** **`ci.yml`** gates. Uploads: **`verify-network-spec-lock-*`** vs **`monorepo-spec-lock-*`** / **`spec-lock-*`**; **`if-no-files-found: ignore`** where applicable.

- **Local**: Same command, or build from a `blvm-spec-lock` git checkout if you need an unreleased tool.

- **Verify jobs** (**`blvm-consensus`**, **`blvm-node`**, **`blvm-protocol`** **`ci.yml`**): after **`check-drift`**, **`cargo-spec-lock verify`** with **`--spec-path`** runs the **merged `F_*` registry** (static + Z3 SAT smoke when the binary is built with **`z3`**) **before** **`#[spec_locked]`** Rust rows; nested **`formula_registry`** in **`spec_lock_*_verify.json`**. [**blvm-node**](https://github.com/BTCDecoded/blvm-node) **`verify`** runs **`cargo-spec-lock verify`** on crates.io **blvm-consensus** then on workspace **blvm-node** (same **`Verify`** shell step). **`jq`** on **`.summary`** drives attestation counts on **consensus** and **node** (**no `grep Status: PASSED` fallback**); [**blvm-protocol**](https://github.com/BTCDecoded/blvm-protocol) gates **`verify`** on exit code (**no **`jq`** attestation counts**). Set **`SPEC_LOCK_VERIFY_FORMULAS_SKIP_Z3=1`** for static-only formula gate where applicable. On **`main`**, **blvm-consensus** **spec-lock-attestation-data** includes **`spec_lock_output.txt`**, **`spec_lock_verify_meta.env`**, **`spec_lock_drift.txt`**; **`spec-lock-verify-json`** (separate) uploads **`spec_lock_verify.json`**. **`spec-lock-drift-log`** uploads **`spec_lock_drift.txt`**.

  Umbrella **`verify.yml`** / **`verify-network.yml`** use the same **check-drift** + **verify** shape (**no** separate **`check-formulas`** / **`verify-formulas`** in CI). **`setup-blvm-spec`** clones **blvm-spec** when missing.

**CLI version:** `cargo-spec-lock --version` prints the crate version (useful in CI logs and attestations).

**`check-drift`:** **`--scoped-unparseables`** limits failure to **unparseable** Orange Paper **Properties** in sections that prefix-match a **`#[spec_locked]`** section string in the scanned crate (same dot-separated § prefix rule as widening contracts: lock **`5.1`** includes **`5.1.2`**). **`--scoped-formulas`** (when supported by the installed **`cargo-spec-lock`**) applies the **same § filter** to **`Formula` (`F_*`)** **`$$`** bodies that fail the **verify**/enrich parse gate (**`extract_parseable_condition`** + **`syn::Expr`**). Omit either flag for a **global** scan of that kind. CI may probe **`check-drift --help`** before appending **`--scoped-formulas`** so older toolchain installs keep working until publish.

**`coverage`:** **`cargo spec-lock coverage --format json`** reports **`F_*`** / **`formula_anchor_*`** metrics (**`formulas_defined`**, **`formulas_parseable_body`**, and related keys), plus **`constants_defined`** and **`constants_bound_to_rust`** (counts from the §4 consensus-constant registry; registry-derived **`0`** without **`--spec-path`**). **`--rollup-from-verify-json PATH`** reads the same **`report_format`** **1** document **`verify`** emits with **`--json-out`**, and fills **`formulas_verify_rollup`** / **`constants_verify_rollup`** (per-status totals over **`results[]`** rows that set **`formula_anchor`** vs **`constant_anchor`**; **`null`** when no such rows). Same **`F_*`** body parse gate as **`check-drift --scoped-formulas`**. **CI:** **blvm-consensus** **`ci.yml`** passes **`--rollup-from-verify-json spec_lock_verify.json`** after **`verify`** when emitting coverage JSON. See **`blvm-spec-lock`** **`docs/COVERAGE_JSON.md`**.

**`check-formulas`:** Registry-wide **static** check that every **`F_*`** **`$$`** body passes **`extract_parseable_condition`** + **`syn::Expr`**. Requires **`--spec-path`** / **`SPEC_LOCK_SPEC_PATH`**. By default this is **static only** (no solver). When **`cargo-spec-lock`** is built with **`--features z3`**, **`--z3-sat`** runs an optional **per-formula Z3 satisfiability smoke**: each enriched **`ensures`**-style condition is translated to Z3 and checked for **SAT** (exit non-zero on **UNSAT**). **`--timeout`** sets per-formula solver seconds for that pass (default **5**). This is **not** **`verify`** against Rust: it does not prove the formula against an implementation; it only catches **obvious contradictions** (e.g. `x < x`) after the static gate passes. **`SPEC_LOCK_Z3_TIMEOUT_SECS`** overrides the per-solver timeout when set (same semantics as **`verify`**).

**`verify-formulas`:** Standalone **`F_*`**-only **`report_format` 1** JSON (**`command`**: **`verify-formulas`**). The same document is nested as **`formula_registry`** inside **`cargo spec-lock verify`** output when **`--spec-path`** is set — **consumer CI** reads that from **`spec_lock_verify.json`**. For local dashboards, use **`verify-formulas --json-out`** or **`jq '.formula_registry'`** on **`verify`** JSON. **`docs/VERIFY_JSON.md`**, **`schemas/formula_verify_report_v1.json`**.

**`list-formulas`:** [experimental] **`F_*`** registry as **tab-separated**: **`id`**, **`section`**, **`parse_gate`** (`ok`/`fail`, same gate as **`check-formulas`/drift/`coverage`**), comma-separated **`depends_on`**, comma-separated **`missing_f_refs`**, comma-separated **`missing_c_refs`** (**`C_*`** under **Depends on** absent from merged §**4** **`$CONST = …$`** excerpts), condensed **`latex_body`**.

**Depends on resolutions (non-fatal):** When **`--spec-path`** merges shards, **`Depends on`** may cite **`F_*`** not in the merged registry or **`C_*`** not defined in any merged **`4.*`** **`$CONST = …$`** line. **`verify`**, **`check-formulas`**, **`check-drift`**, **`coverage`**, **`summary`**, **`extract-constants`**, **`extract-formulas`**, and **`extract-property-tests`** print stderr **`formula_id → missing_dep`** lines; **`list-formulas`** shows **`missing_f_refs`** and **`missing_c_refs`**. See **`blvm-spec-lock`** **`SPEC_WORDING.md`**.

**`SPEC_LOCK_STRICT`:** When set to `1` / `true` / `yes`, verification **also** fails on **Partial** (same effect as **`--strict`** on the CLI). **Failed** and **NoContracts** outcomes fail the process **regardless** of strict. Without strict, a run may exit **0** even when **`summary.partial > 0`** — use **`--strict`** or gate on **`jq '.summary.partial'`** if CI must not accept incomplete Z3 coverage.

**`SPEC_LOCK_FORMULAS`:** Set to **`0`**, **`false`**, **`no`**, or **`off`** so the Orange Paper parser skips **`Formula` (`F_*`)** blocks (**`cargo spec-lock list-formulas`** is empty; **`F_*`** anchors fail as missing). **Unset** (omit **`SPEC_LOCK_FORMULAS`**) keeps formulas **on** (**default**), which CI expects for **`F_SpecLockWitness`** in **`PROTOCOL.md`** §13.3.6. See **`blvm-spec-lock`** **`docs/VERIFY_JSON.md`**.

**JSON `detail`:** Each **`results[]`** row includes **`anchor_kind`**: **`function`**, **`formula`**, or **`constant`** (**additive** on **`report_format` 1**). Failed rows may include **`results[].detail.failure_kind`** (`counterexample`, `parse_error`, `solver_unknown`, …). When **`failure_kind`** is **`solver_unknown`**, optional **`detail.partial_reason`** is **`z3_timeout`** vs **`z3_unknown`** (message heuristic, including **`Determinism: Z3 unknown: …`**). **Partial** rows may include **`detail.partial_reason`** (`missing_z3_build`, `incomplete_coverage`, …). Rows may optionally include **`results[].formula_anchor`** (**`F_*`**) / **`constant_anchor`** (**`C_*`**) echoing **`#[spec_locked]`** when present (for dashboards / **`coverage`** rollups above). See **`blvm-spec-lock`** **`docs/VERIFY_JSON.md`** and **`schemas/verify_report_v1.json`**. **`coverage`** machine output: **`docs/COVERAGE_JSON.md`** and **`schemas/coverage_inventory_v1.json`** / **`coverage_spec_rollup_v1.json`**.

**`SPEC_LOCK_Z3_TIMEOUT_SECS`:** When set to a positive integer, overrides the per-function Z3 **`--timeout`** (seconds). Use when obligations are **richer** and the solver returns **Unknown** / **Partial** under the default timeout.

**Local verify (CLI only):** From the **`blvm-consensus`** crate root, with the Orange Paper on disk (sibling **`../blvm-spec`** or **`SPEC_LOCK_SPEC_PATH`**):

```bash
cargo spec-lock verify --verbose --crate-path . --spec-path "${SPEC_LOCK_SPEC_PATH:-../blvm-spec/PROTOCOL.md}"
```

The authoritative **`verify`** invocations live in **`blvm-consensus/.github/workflows/ci.yml`**, **`blvm-node/.github/workflows/ci.yml`**, and **`blvm-protocol/.github/workflows/ci.yml`** (see bullets above).

## Implementation Details

### Cargo.toml

The library dependency points to crates.io; optional `[patch.crates-io]` in a local config overrides when you use path overrides during development.

From **`blvm-spec-lock` ≥ 0.1.12**, crates.io also hosts **`blvm-spec-lock-core`** (shared parser/translator). You still depend only on **`blvm-spec-lock`** in this crate’s `Cargo.toml`; Cargo pulls **`blvm-spec-lock-core`** transitively.

### Where the Orange Paper markdown lives

Canonical spec repository: **[github.com/BTCDecoded/blvm-spec](https://github.com/BTCDecoded/blvm-spec)** (`PROTOCOL.md`, `ARCHITECTURE.md`, `THE_ORANGE_PAPER.md`).

**There is no copy of the spec inside `blvm-consensus`.** At `#[spec_locked]` expansion, `blvm-spec-lock` looks for **`../blvm-spec/PROTOCOL.md`** and **`../blvm-spec/ARCHITECTURE.md`** (or **`THE_ORANGE_PAPER.md`** there) relative to this crate’s `CARGO_MANIFEST_DIR`, or uses **`SPEC_LOCK_SPEC_PATH`** if set. Clone **blvm-spec** as a **sibling** of **blvm-consensus** (so `../blvm-spec` exists), or point **`SPEC_LOCK_SPEC_PATH`** at those files.

### Notes

- The `local-spec-lock` feature in `Cargo.toml` is reserved for optional local path workflows
- To see what Cargo resolves: `cargo tree -i blvm-spec-lock`
- **Governance tooling list:** **`governance/config/repos/blvm-consensus.yml`** lists **`cargo spec-lock verify`** with **`--spec-path`** (**merged `F_*` registry + Rust rows**; **`--json-out spec_lock_verify.json`**) and **`check-drift`**. If your spec checkout is **`THE_ORANGE_PAPER.md`** only, use that path in the YAML stub instead.
- **Named-formula anchor witness:** **`src/spec_lock_formula_witness.rs`** references **`PROTOCOL.md` §13.3.6 — `F_SpecLockWitness`** via **`#[spec_locked("13.3", "F_SpecLockWitness")]`** (§ **13.3** subsumes §**13.3.6**). Confirms **`cargo spec-lock verify --strict`** (or **`SPEC_LOCK_STRICT=1`**) derives obligations from **`Formula`** blocks alongside traditional **`Function`** headings.

## Attestation and `verify` machine output

The **blvm-consensus** CI job records **`spec_lock_output.txt`** from **`cargo-spec-lock verify`** and, on **`main`**, uploads it with **`spec_lock_verify_meta.env`** for release attestation (**`cargo-spec-lock`** version, **`git` SHA** of **`../blvm-spec`**, optional **`SPEC_META.json`** **`version`** when **`jq`** is available).

**Preferred signal for “how many functions passed”**: the versioned **`spec_lock_verify.json`** report (`report_format` **1**), written from **`verify`** with **`--json-out`** alongside **`--format human`**:

```bash
cargo-spec-lock verify $SPEC_ARGS --format human --json-out spec_lock_verify.json … | tee spec_lock_output.txt
```

Counts and dashboards should read **`jq -r '.summary.passed'`** / **`.summary.total`** rather than scraping human text — see **`blvm-spec-lock`** **`docs/VERIFY_JSON.md`** and **`schemas/verify_report_v1.json`**.

**Primary integrity:** keep treating the **exit code** of **`verify`** and **hashes** of archived logs as authoritative. **Authoritative CI** installs **`blvm-spec-lock`** with `cargo install --version '>=0.1, <1'` (same semver range as **`Cargo.toml`**) and passes **`--json-out`** so **`spec_lock_*_verify.json`** exists (**consensus** / **node**: **`jq`** **`.summary`** for attestations; **protocol**: exit code + uploaded artifacts).

**Hashes:** **`sha256(spec_lock_verify.json)`** is the preferred attestation fingerprint when the artifact exists (**`SPEC_LOCK_VERIFY_JSON`**-style tooling may follow). **`spec_lock_output_hash`** continues to fingerprint **`spec_lock_output.txt`** for backward compatibility.

**Legacy (human-only):** Counting **`grep -c "Status: PASSED"`** on **`spec_lock_output.txt`** counts **PASSED-line occurrences** in the formatter’s prose, **not** a guaranteed one-to-one with **`#[spec_locked]` rows** vs **`summary.passed`** in all future format versions. Do **not** use this path in **`blvm-consensus`** / **`blvm-node`** release metadata once **`spec_lock_verify.json`** is required (publish **`blvm-spec-lock`** before relying on merged **`formula_registry`**/`verify` JSON — see **`scripts/publish-crates-io.sh`** and **`Cargo.toml`** version sync in **`blvm-spec-lock`**). Prefer **`.summary`** from JSON.

## §5.2.6 vs `connect_block` (Orange Paper vs Bitcoin Core)

Orange Paper §5.2.6 defines a piecewise **`GetBlockScriptFlags`**: if the block hash is in **`ScriptFlagExceptions`**, use the table value; otherwise use per-transaction **`CalculateScriptFlags`**. **`#[spec_locked("5.2.6", "GetBlockScriptFlags")]`** on `get_block_script_flags` tracks that formulation.

**`connect_block`** instead uses **`get_block_script_verify_flags_core`**, which matches Bitcoin **`GetBlockScriptFlags`** in **`validation.cpp`**: default **`SCRIPT_VERIFY_P2SH | WITNESS | TAPROOT`**, replace on exception, then OR buried deployments (BIP66, BIP65, CSV, BIP147). The same block-level mask is passed to every non-coinbase script check, consistent with Core’s **`CheckInputScripts`**.
