# IBD Optimization Plan — Validation Report

**Date:** 2025-02-26  
**Scope:** Validate the 6-item optimization plan against the codebase and profile assumptions.

> **Note (2025-03):** ECDSA batch verification has been removed. References to `block_ecdsa_collector`, `collect_bulk`, etc. in this document are obsolete. See `docs/SECP256K1_FORK_REMOVAL_PLAN.md`.

---

## Summary

| # | Optimization | Verdict | Notes |
|---|--------------|---------|-------|
| 1 | Direct-collect | ✅ **Valid** | Highest impact; implementation path clear |
| 2 | Eliminate tx_refs Arc | ✅ **Valid** | use_embedded path already shows the pattern |
| 3 | Single-writer SoA | ⚠️ **Limited** | Only when 100% pre-extracted; rare in practice |
| 4 | Lock-free queue | ✅ **Valid** | SegQueue + atomics feasible; moderate complexity |
| 5 | Auto-tune parallelism | ✅ **Valid** | Config already has hooks |
| 6 | Overlap producer + batch verify | ⚠️ **Partial** | Drain threads already overlap; gains may be smaller |

**Combined estimate:** 3–4x realistic, 4–5x optimistic. Start with #1.

---

## 1. Direct-collect (1.8–2.5x) — ✅ VALID

**Claim:** Producer calls `collector.collect_bulk()` for pre-extracted P2PKH/P2WPKH instead of queueing ScriptChecks.

**Code verification:**

- **Producer has collector:** `block_ecdsa_collector` is in scope before the parallel block (block.rs ~1179). Session holds it; producer creates the session.
- **Pre-extracted data available:** `P2pkhPreExtracted` and `P2wpkhPreExtracted` contain `(pubkey, sig, sighash, flags)`. `collect_bulk` expects `(idx, pubkey, sig, sighash, flags)` — match.
- **Global index:** `ctx.ecdsa_index_base + input_idx` is available. `ecdsa_sub_counters` is on the session; producer has session via `tx_contexts_guard` and session creation.
- **Results handling:** Workers push `(tx_ctx_idx, valid)` to `session.results`. Producer can push `(tx_ctx_idx, true)` for direct-collected inputs. Aggregation in block.rs (lines 1209–1211) is per-input, so one result per check is correct.
- **n_todo:** Must not increment for direct-collected checks. Producer adds only non–pre-extracted checks to the queue; `n_todo` stays correct.

**Implementation path:**

1. In the producer loop (block.rs ~1012–1145), when building `tx_checks_reusable`:
   - For each input with `p2pkh_pre_extracted` or `p2wpkh_pre_extracted`: call `block_ecdsa_collector.collect_bulk()` with batched items (or per-tx batches), push `(tx_ctx_idx, true)` to `results_arc`, and skip adding that `ScriptCheck`.
2. Ensure `ecdsa_sub_counters` is accessible (session is created before the loop; producer has it).
3. Batch `collect_bulk` per tx to reduce calls (one call per tx with pre-extracted inputs is fine).

**Caveat:** Profile counters (`run_check_loop`, `p2pkh_collect`) are summed across workers (atomic add), not wall-clock. The 24.5 ms `p2pkh_collect` is from the per-sig `collect()` path (script.rs:1120), not `collect_bulk`. Pre-extracted path uses `collect_bulk` and does not add to `p2pkh_collect_ns`. So at high pre-extraction, `p2pkh_collect` may already be low; the gain comes from removing worker round-trip (queue, drain, `process_bulk_p2pkh`) for pre-extracted inputs. The 1.8–2.5x estimate is plausible if worker overhead dominates.

---

## 2. Eliminate tx_refs Arc (1.2–1.3x) — ✅ VALID

**Claim:** Use `script_pubkey_range` and `prevout_value` instead of `Arc<(Vec<Arc<[u8]>>, Vec<i64>)>` per tx.

**Code verification:**

- **Current flow:** `use_self_contained` when `batch.first().and_then(|c| c.tx_refs.as_ref()).is_some()` (checkqueue.rs:666). Workers use `tx_refs.0` and `tx_refs.1` for refs and prevout values.
- **Alternative:** `use_embedded` (lines 704–765) builds refs from `session.script_pubkey_buffer`, `script_pubkey_indices_buffer`, `prevout_values_buffer` using `script_pubkey_range` and `prevout_value` on each check. No `tx_refs` needed.
- **Producer already sets:** `script_pubkey_range` and `prevout_value` are set for every check (block.rs ~1013–1020). So we can always use the embedded-style path.
- **Change:** Stop setting `tx_refs`; ensure `use_embedded` (or equivalent) is used when all checks have `script_pubkey_range` and `prevout_value`. Drop the `use_self_contained` branch for this case.

**Caveat:** `use_embedded` builds `tx_refs_map` per batch from buffer ranges. That work remains but avoids per-tx `Arc` allocations and clones. The 1.2–1.3x is reasonable for allocation/cache pressure reduction.

---

## 3. Single-writer SoA (1.3–1.5x) — ⚠️ LIMITED

**Claim:** No shard locks when producer is the only writer.

**Code verification:**

- **Writers today:** Producer does not write to SoA. Workers write via `collect_bulk` (process_bulk_p2pkh) and `collect()` (run_check path for non–pre-extracted).
- **With direct-collect:** Producer writes via `collect_bulk` for pre-extracted. Workers still write via `collect()` for non–pre-extracted (P2SH, multisig, etc.).
- **Single-writer only when:** 100% of ECDSA inputs are pre-extracted. At 180k blocks, FAST_PATH logs suggest high P2PKH/P2WPKH share, but P2SH and others exist.
- **Conclusion:** Single-writer mode applies only when there are no non–pre-extracted checks. That is a narrow case. For typical blocks, we remain multi-writer; shard locks stay. Estimate should be downgraded or marked as conditional.

---

## 4. Lock-free queue (1.1–1.2x) — ✅ VALID

**Claim:** Replace `Mutex<QueueState>` with `SegQueue` + atomics.

**Code verification:**

- **Current:** `state: Mutex<QueueState>` holds `session`, `checks`, `n_todo`, `error_result` (checkqueue.rs). Producer and workers contend on this lock.
- **Feasibility:** `SegQueue` for checks; `AtomicUsize` for `n_todo`; `session` set once at `start_session`; error handling via `AtomicPtr` or similar. Crossbeam `SegQueue` is already used for `results`.
- **Complexity:** Moderate. Need to preserve semantics of `complete()` (wait for `n_todo == 0`, collect results, handle errors).

**Caveat:** 1.1–1.2x is modest; lock contention may not be the main bottleneck. Still worth doing if #1 and #2 are done and profiling shows queue contention.

---

## 5. Auto-tune parallelism (1.1–1.3x) — ✅ VALID

**Claim:** Derive worker count, batch size from `nproc` and cache.

**Code verification:**

- **Config:** `config.rs:928` documents `RAYON_NUM_THREADS=nproc-1` for IBD. `BLVM_CRYPTO_DRAIN_THREADS` is env-configurable (block.rs ~1151).
- **Batch size:** `DEFAULT_BATCH_SIZE = 128` (checkqueue.rs:20); can be driven by `ibd_tuning` or similar.
- **Implementation:** Use `std::thread::available_parallelism()` or `num_cpus`; derive workers, batch size, crypto drain threads; cache at startup.

**Straightforward; no major risks.**

---

## 6. Overlap producer + batch verify (1.2–1.4x) — ⚠️ PARTIAL

**Claim:** Start crypto drain threads while producer is still collecting.

**Code verification:**

- **Current flow (block.rs ~1179–1197):** `spawn_soa_drain_threads` is called before `complete()`. Drain threads run and block on `ready_queue` until chunks are ready. Workers fill SoA and push chunk IDs to `ready_queue`; drain threads consume.
- **Overlap today:** Producer adds checks → workers process and collect → drain threads verify. Drain threads start before `complete()` and overlap with worker collection.
- **Possible improvement:** Spawn drain threads earlier (e.g., at block start) so they are ready as soon as the first chunks arrive. Today they are spawned before the parallel block; the main delay may be `complete()` waiting for workers. If workers finish quickly after direct-collect, the overlap may already be good.
- **Conclusion:** Some overlap exists. Further gains depend on profiling. Estimate may be optimistic; treat as 1.1–1.2x until measured.

---

## Profile Data Caveats

- **Counter semantics:** `run_check_loop` and `p2pkh_collect` are summed across workers (CPU time), not wall-clock. A 27 ms `run_check_loop` with N workers can correspond to ~27/N ms wall time if perfectly parallel.
- **Block height:** Available logs (e.g. `blvm-ibd-20260303-053126-15min.log`) show early blocks (0–25) with no ECDSA sigs. The 180k-block numbers (22 ms/block, 27 ms run_check_loop, 24.5 ms p2pkh_collect) likely come from a different run. Re-profile at 180k–190k before and after changes.
- **Pre-extraction rate:** P2WPKH requires `precomputed_sighash_all` (block.rs:1097). If that is often `None`, P2WPKH falls back to non–pre-extracted path. Check FAST_PATH logs at target height to confirm pre-extraction rate.

---

## Recommended Order

1. **#1 Direct-collect** — Highest impact, clear path.
2. **#2 Eliminate tx_refs** — Independent, reduces allocations.
3. **#5 Auto-tune** — Low risk, quick win.
4. **#4 Lock-free queue** — If profiling shows queue contention.
5. **#6 Overlap** — Only if profiling shows drain start as bottleneck.
6. **#3 Single-writer** — Defer; limited applicability.

---

## Conclusion

The plan is **valid** overall. #1 and #2 are well-supported by the code. #3 is narrow; #4 and #5 are sound; #6 may yield less than estimated. Proceed with #1 first and re-profile to validate gains before committing to the rest.
