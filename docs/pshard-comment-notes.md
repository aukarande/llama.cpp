# pshard: details removed from code comments

Code comments follow llama.cpp convention (what/why, no dates, measurements or narrative). The measurements,
dates and experiment references that used to sit in comments are kept here, keyed by file. Full original text:
git history up to f549c0419 and grid-results/sweep-snapshot-20260914/. Results: grid-results/*/RESULTS.md.

## common/common.h

# comment rewrite notes: common/common.h

File is LF (not CRLF); line endings preserved. Only comment text changed; 5 comment lines rewritten in 3 blocks (line 343, line 963, lines 977-979). Line count unchanged (1252).

## common/common.h:343 (struct common_params_speculative, field n_ubatch)

- original: `// the target's tier ubatch (a dflash draft at ubatch 2048 reserved 4.3 GiB)`
- kept as: `// the target's tier ubatch, which is far larger than a draft needs`
- removed information: the measured figure behind the rule - a dflash draft context built at the target's tier ubatch of 2048 reserved 4.3 GiB of compute buffers, which is what motivated recording the pre-plan ubatch in speculative.draft.n_ubatch (see memory note pshard-mtp-enablement: budget honesty / one-budget reserve). The mechanism (draft contexts run at the pre-plan ubatch so they do not size compute buffers at the target's tier ubatch) is kept.

## common/common.h:963 (common_pshard_draft_reserve_mb doc block)

- original: `//   - measured with the fit's memory probe; drafts that need a target context to build`
- kept as: `//   - sized by the fit's memory probe; drafts that need a target context to build`
- removed information: none of substance. The regex matched the word "measured"; the bullet was already mechanism-only (the reserve comes from common_get_device_memory_need, the fit's no_alloc probe, in common/common.cpp common_pshard_draft_reserve_mb). Reworded so the word does not read as a recorded measurement, matching the fit.h rewrite.
- checked and kept in the same block: line 961 "(> 2 GiB with expert stacks)" is the code threshold `w_total > (2ULL << 30)` in common.cpp, not a measured number.

## common/common.h:977-979 (common_pshard_mtp_need_mb doc block)

- original:
  ```
  // sets *probe_failed (callers must not read that as "fits"). The pre-fit reserve is measured with
  // everything pinned; a plan that moves the MTP head or the MTP layer's FFN off the device grows the
  // context's compute by ~180 MiB on q35 (2026-09-05 grid: +177.5 MiB rows).
  ```
- kept as:
  ```
  // sets *probe_failed (callers must not read that as "fits"). The pre-fit reserve assumes everything
  // pinned; a plan that moves the MTP head or the MTP layer's FFN off the device grows the context's
  // compute, so the need is probed again under the fitted placement.
  ```
- removed information:
  - "~180 MiB on q35 (2026-09-05 grid: +177.5 MiB rows)": the measured size of the MTP context's compute growth when the fitted plan moves the MTP head or the MTP layer's FFN off the device - +177.5 MiB rows in the 2026-09-05 grid on the q35 model (see memory note expert-pool-build: MTP reserve re-measured post-fit, 2-pass; and qa/perf-grid.md). This is the figure that justified the second (post-fit) MTP probe and the one-time refit in common_pshard_fit_one_budget.
  - "is measured with everything pinned" -> "assumes everything pinned": same meaning, worded as the assumption rather than a recorded measurement.
  - the consequence ("so the need is probed again under the fitted placement") was added to state the why the removed number used to imply; it is the mechanism implemented by common_pshard_fit_one_budget (re-measure under the fitted plan, MAX over viable tiers, refit once).

## blocks read but left unchanged

- common/common.h:340 (pshard_reserve_mb): "MiB carved out of the target's pshard budget for this context (self-check at teardown)" - mechanism only; kept.
- common/common.h:341-342 (n_ubatch, first two lines): mechanism only; kept.
- common/common.h:355-361 (exps_bytes_per_layer / spill_pattern / exps_spill_auto, "pshard one-budget v2"): design-rule name and mechanism, no dates/numbers/cells; kept.
- common/common.h:485-490 (fit_params_target / fit_params_budget): mechanism only; kept.
- common/common.h:955-967 (rest of the common_pshard_draft_reserve_mb block), 970, 973-976, 983-988 (common_pshard_fit_one_budget; "re-measure the MTP context under the fitted plan" describes the second probe, not a recorded number), 996-997 (common_pshard_draft_leftover): mechanism only; kept.
- Non-pshard hits of the wider sweep (225 seed, 426 deepseek reasoning format, 634 cache_ram_mib, 721 legacy imatrix.dat) are stock llama.cpp comments; untouched.

## common/fit.cpp

# comment rewrite notes: common/fit.cpp

File is LF (0 CR bytes before and after); line endings preserved. Only comment text changed; 2 comment lines rewritten (lines 201 and 204, both inside common_params_fit_impl). Line count unchanged (1117). No code, identifiers, string literals or whitespace outside comments touched.

## common/fit.cpp:201 (common_params_fit_impl, `n_ctx_extra` declaration)

- original: `// context that memory was measured at`
- kept as: `// context size dmds_extra was evaluated at`
- removed information: none of substance. The regex matched the word "measured"; the comment was already mechanism-only (n_ctx_extra records the cparams->n_ctx the extra model's no_alloc memory data in dmds_extra was taken at, so add_extra_memory can tell when to re-take it). Reworded so it does not read as a recorded measurement, and named the variable it describes.

## common/fit.cpp:203-204 (common_params_fit_impl, block above the `add_extra_memory` lambda)

- original:
  ```
  // the extra model competes for the same memory as the main model, add it to every measurement
  // its memory is measured again whenever the context it follows changes
  ```
- kept as:
  ```
  // the extra model competes for the same memory as the main model, add it to every measurement
  // its memory is re-evaluated whenever the context it follows changes
  ```
- removed information: none of substance. Only line 204 changed; the regex matched "measured". The mechanism is unchanged and still stated: add_extra_memory re-runs common_get_device_memory_data_impl on the extra model when dmds_extra is empty or n_ctx_extra != cparams->n_ctx (extra->cparams->n_ctx follows the main model's n_ctx). Wording matches the sibling rewrite in common/fit.h:15 ("the fit re-evaluates its memory whenever that context changes"). Line 203 ("add it to every measurement") was not flagged and is mechanism-only; left as is.

## blocks read but left unchanged (pshard-related, mechanism-only)

- common/fit.cpp:226 (add_extra_memory catch block): "the extra model is optional, fit the main model alone rather than giving up" - one-line why; kept.
- common/fit.cpp:292-294 (--fit-budget margin derivation): "the budget is for weights + KV + compute of THIS model, independent of what the process already holds on the device (the free-memory query above already excludes the CUDA context). target = free - margin, so margin = free - budget makes target == budget." - mechanism and the arithmetic the code implements; no dates, numbers, cells or history; kept. (It is hard-wrapped mid-sentence at a fixed column, contrary to AGENTS.md style, but that is not narrative and rewrapping would change the line count; left for a later style pass.)
- common/fit.cpp:1107-1109 (common_get_device_memory_need): "pshard budgets, plans and pins are device-0 scoped (pshard_dev_layout::for_device(0)); report device 0's share only, so a layer-split draft on a multi-GPU host is not charged to device 0 for bytes that land elsewhere (the last entry is host memory)" - mechanism with a function reference; kept.
- common/fit.cpp:1104: `"cannot measure device memory of '%s'"` is a LOG_WRN format string, not a comment; not touched (the grep pattern requires `//` and does not match it).
- All other comments in the file are stock llama.cpp fit-algorithm comments (steps 1-4, false-position search, memory-breakdown table) and contain none of the disallowed content; kept.

## step 3 result

Re-running the flagged-comment grep on common/fit.cpp after the rewrite returns 0 lines.

## common/fit.h

# comment rewrite notes: common/fit.h

File is LF (not CRLF); line endings preserved. Only comment text changed; 4 comment lines rewritten (line 15, and the 3-line block at lines 71-73). Line count unchanged (85).

## common/fit.h:15 (struct common_fit_extra_model doc block)

- original: `//   - its context follows the context of the main model, so its memory is measured again whenever that context changes`
- kept as: `//   - its context follows the context of the main model, so the fit re-evaluates its memory whenever that context changes`
- removed information: none of substance. The regex matched the word "measured"; the sentence was already mechanism-only (the fit re-runs the no_alloc load of the extra model when the followed context changes - see common/fit.cpp, the `n_ctx_extra` block in common_fit_params_impl). Reworded only so the word does not read as a recorded measurement.

## common/fit.h:71-73 (struct common_device_memory_need doc block)

- original:
  ```
  // Measured device-memory need of a model + context at the given params, summed over
  // the GPU devices, in bytes (ok=false if it could not be measured - e.g. drafts whose
  // graph needs a target context). Used by pshard's one-budget rule for spec contexts.
  ```
- kept as:
  ```
  // device-memory need of a model + context at the given params, in bytes, device 0's share only (pshard budgets are device-0 scoped)
  //   - ok=false if the no_alloc load failed, e.g. a draft whose graph needs a target context
  //   - used by common_pshard_mtp_need_mb and common_pshard_draft_reserve_mb to size the spec-context reserve
  ```
- removed information:
  - "Measured" / "could not be measured": replaced with the actual mechanism (ok=false is returned when common_get_device_memory_data_impl throws, i.e. the no_alloc load failed - common/fit.cpp common_get_device_memory_need).
  - "pshard's one-budget rule for spec contexts": the design name for the rule that the MTP/draft context is paid from the same device budget as the main model (see memory note pshard-mtp-enablement: one-budget reserve). Replaced by the two callers that implement it: common_pshard_mtp_need_mb (common/common.cpp ~1536) and common_pshard_draft_reserve_mb (common/common.cpp ~1686).
  - "summed over the GPU devices": this was stale and contradicted the implementation, which reports device 0's share only (dmds[0]) because pshard budgets, plans and pins are device-0 scoped (pshard_dev_layout::for_device(0)); the header now states what the code does. Recorded here in case the summed-over-devices wording reflected an earlier version of the function.
  - the mid-sentence hard wrap at a fixed column was dropped (AGENTS.md: do not split a line mid-sentence).

## blocks read but left unchanged

- common/fit.h:39-40 (`budgets` parameter of common_fit_params): "optional device memory budgets per device in bytes (weights + KV + compute); a non-zero entry replaces that device's margin with (free - budget)" - mechanism only, no dates/numbers/cells; kept.
- common/fit.h:14,16, 24-28, 42, 60 - stock or mechanism-only wording; kept.

## examples/llama-profiler/profiler-cpu.cpp

# comment notes: examples/llama-profiler/profiler-cpu.cpp

Convention: "line (function) - original text - kept as: new text". Line numbers are pre-edit (the edits did not change the line count). The file is LF; only `//` comment text was touched.

## rewritten

121 (struct pcie_stress_ctx) - `// bytes the stress loop moved and the time it ran: the concurrent PCIe rate is measured, not derived` - kept as: `// bytes the stress loop moved and the time it ran: the concurrent PCIe rate is computed from these` (dropped the "not derived" contrast with the earlier derived rate; the what/why stays)

146 (file-level "machine calibrations" block above CPU_PROFILE_SCHEMA) - `// Every machine-specific number the planner prices with is measured here and written as a header line; the` - kept as: `// Every machine-specific number the planner prices with is calibrated here and written as a header line; the`

149 (same block) - `// the gathered-upload curve is measured for the copy engine AND the two kernel paths, and the pool's per-layer` - kept as: `// the gathered-upload curve is taken for the copy engine AND the two kernel paths, and the pool's per-layer`

210 (struct gpu_procs header) - `// paths are measured and the kernel-copy lines are omitted)` - kept as: `// paths are calibrated and the kernel-copy lines are omitted)`

294 (struct calib_results, kernel_cap_mb) - `// largest size at which the kernel copy still wins (-1 = not measured)` - kept as: `// largest size at which the kernel copy still wins (-1 = not calibrated)`

564 (write_profile_header) - `// the profile header: every measured machine number as one parseable line` - kept as: `// the profile header: every calibrated machine number as one parseable line`

614 (splice_profile_header) - `// replace the header block of an existing profile with freshly measured lines, keeping its op tables (they` - kept as: `// replace the header block of an existing profile with freshly calibrated lines, keeping its op tables (they`

615 (splice_profile_header) - `// take the long run; the calibrations take seconds). Values the calibration modes do not measure (CPU_Eff,` - kept as: `// take the long run; the calibrations take seconds). Values the calibration modes do not produce (CPU_Eff,`

699-701 (calibrate_pin_ceiling) - `// descending chunk sizes: after a refusal, smaller chunks tighten the measured` / `// floor to 256 MiB granularity (a sub-2 GiB ceiling would otherwise read as 0` / `// = "not measured", and the planner would price everything at the pinned rate)` - kept as: `// descending chunk sizes: after a refusal, smaller chunks tighten the floor to` / `// 256 MiB granularity (a sub-2 GiB ceiling would otherwise read as 0 = probe` / `// skipped, and the planner would price everything at the pinned rate)` (0 is what write_profile_header treats as "no Host_Pin_Ceiling line", i.e. the probe did not run; renamed the meaning of 0 to match)

752 (struct bench_result_cpu, pcie_concurrent_gb_s) - `// PCIe rate measured while this op ran under the stress loop` - kept as: `// PCIe rate while this op ran under the stress loop`

1072 (save_results_cpu, est_pcie) - `// measured while the op ran under the PCIe stress loop` - kept as: `// the rate while the op ran under the PCIe stress loop`

## flagged but kept unchanged

407-409 (calibrate_copy_crossover) - `// copy kernel vs copy engine for one transfer ordered behind a kernel: on WDDM a copy-engine transfer ordered` / `// against a kernel idles the GPU for the engine transition, a copy kernel does not. The cap is the largest size` / `// at which the kernel still wins; the transition is timed on its own.` - kept as-is. The regex flags "WDDM", but the block is the explicitly allowed mechanism form ("a copy-engine transfer ordered against kernels idles the GPU under WDDM"): no machine name, no numbers, no history. WDDM is the driver model the mechanism depends on (it does not apply under TCC or on Linux), so removing the term would make the comment less accurate. This is the one comment line remaining in the step-3 grep.

## pre-existing working-tree rewrites vs HEAD (found in `git diff`, not made in this pass; originals recorded so the information is preserved)

31 (file header, pin-ceiling includes) - `// pin-ceiling probe: physical-memory query + env set` - working tree has: `// pin-ceiling probe: physical-memory query` (the env-var path was removed with the back-door cleanup)

145 (file-level "machine calibrations" block) - `// ---- machine calibrations (profile schema 2, 2026-09-12) ------------------------------------------------` - working tree has: `// ---- machine calibrations (the profile header block) ----------------------------------------------------`

408-409 (calibrate_copy_crossover) - `// against a kernel idles the GPU for the engine transition (35-55 us measured 2026-09-10); a copy kernel does` / `// not. The cap is the largest size at which the kernel still wins; the transition is measured on its own.` - working tree has: `// against a kernel idles the GPU for the engine transition, a copy kernel does not. The cap is the largest size` / `// at which the kernel still wins; the transition is timed on its own.` (the engine-transition figure lives in the profile's Engine_Switch_us line and in memory/wddm-engine-transitions.md)

## step 2 (regex-missed narrative)

Read every pshard-related comment once more (pin-ceiling probe, dram_stress, calib_results fields, calibrate_pcie_sliced, calibrate_pcie_concurrent, small_graph, calibrate_pool_latencies, calibrate_staged_upload, calibrate_pin_ceiling, the main() PCIe-rate comment). None carried dates, measured numbers, cell names, review references, machine names or before/after history. Sizes that appear (64 B, 8 KB, 24-chunk bursts, 2 GiB / 256 MiB chunks, 0.5-32 MB chunk sizes) are the probe parameters the code uses, not measured results. Nothing further trimmed.

## ggml/include/ggml-backend.h

# ggml/include/ggml-backend.h - comment rewrites

Removed information preserved here (dates, measured numbers, machine/experiment references are not allowed in code comments).

- 437 (optional backend procs list, entry "ggml_backend_kernel_copy_set") - original: `(no copy-engine transitions - on WDDM each costs 35-55 us of GPU idle)` - kept as: `(no copy-engine transitions - under WDDM each one idles the GPU)`. Dropped: the measured cost of one copy-engine transition on the RTX 5070 Ti WDDM box, 35-55 us of GPU idle per transfer ordered against a kernel (gap census 2026-09-10, see memory note wddm-engine-transitions.md).

- 443 (optional backend procs list, entry "ggml_backend_kernel_copy_max_set") - original: `copies on (bandwidth-bound bulk). The runtime sets the machine profile's measured crossover here.` - kept as: `copies on (bandwidth-bound bulk). The runtime sets the machine profile's copy crossover here.` Dropped: the word "measured" - the crossover is the copy-kernel vs copy-engine byte crossover the profiler measures into the schema-2 machine profile (profiler microbenchmark, 2026-09-12/13); the runtime reads it from the profile, never from a constant.

Step 2 sweep (pshard-related comments, lines 360-465): no further narrative found; "legacy tiers" (line 430) is the fork's tier kind, not history, left as is.

Step 3 re-grep: 1 remaining hit, line 437, on the token "WDDM". Kept deliberately: it is the platform mechanism ("under WDDM a copy-engine transition idles the GPU"), which the comment rules list as the allowed form; it is not a machine name or a measurement. Removing the platform would make the statement false for TCC/Linux, where copy engines overlap kernels. No code, identifiers, string literals or line endings were touched (CRLF count 554 before and after, 0 bare LF).

## ggml/src/ggml-alloc.c

# ggml/src/ggml-alloc.c - comment rewrites

Removed information preserved here (dates, measured numbers, machine/experiment references are not allowed in code comments).

No comment was rewritten in this file. Entries below record what was checked and why each candidate was left as is.

Step 1 regex: 1 hit.

- 888 (ggml_gallocr_reserve_n_impl) - original: `// add 25% margin to avoid hash collisions` - kept as: unchanged. Not rewritten: this is upstream ggml code (present in the last upstream commit touching the file, 6036c635e, at its line 828). The number is the code constant on the next line, `min_hash_size += min_hash_size / 4`, not a measurement, so it is outside the fork's narrative-trimming scope; rewording an upstream comment would only add rebase noise.

Step 2 sweep (all comments added by the fork - lines 656, 763, 778, 804, 865, 907, 1059, 1250, 1260-1262, 1271-1272 - from the diff of 6036c635e..HEAD, plus every other comment line in the file): no narrative found. Each fork comment states the mechanism (deferred FLAG_WRITEBACK leafs allocated through the consumer view and fenced by OP_NONE keepalives; external buffers constrained to their allocation range; shared tallocr slots counted once; overflow-chunk release dropping the tallocr bookkeeping and the node/leaf allocs so the next alloc_graph re-reserves). None carries a date, measured number, grid cell or A/B name, review reference, machine name or "before/after" history. The 3-line block at 1260-1262 is a what/why explanation that references get_n_chunks, reserve and restore_state; left as is.

Step 3 re-grep: 1 remaining hit, line 888, the upstream `25% margin` comment described above. No code, identifiers, string literals or line endings were touched: the file is unmodified in the working tree (git status clean; CRLF count 1509 before and after, 0 bare LF).

## ggml/src/ggml-backend.cpp

# comment rewrite notes: ggml/src/ggml-backend.cpp

File is CRLF; line endings preserved (3579 CRLF / 0 bare LF before, all lines still CRLF after). Only `//` comment text changed; no code, identifiers, string literals or non-comment whitespace touched. 3 comment blocks rewritten, 18 original comment lines -> 11 (file shrinks by 6 lines, so line numbers after 2105 shift; original line numbers are given below, new ones in parentheses).

The regex from the brief flagged one line (2370, "WDDM"). Step 2 read every pshard-related comment in the file and found two more blocks with the same kind of narrative (a before/after "Replaces ... which left the GPU idle" and an experiment conclusion "so it has little headroom"). All other pshard comments in the file (sched struct fields, host-weight view rule, prefetch lookahead, keepalive reservation, arena-overflow refusal, backend_busy / pending_input_sync / flag_input_pending, alias drain rule, host_upload_no_drain, pool callbacks, DeepSeek-V4 writeback ordering, CPU/GPU overlap block, deny_kernel_copies, sliced-copy accounting) are mechanism-only and were left as they are.

## ggml/src/ggml-backend.cpp:2105-2111 (ggml_backend_sched_compute_splits -> run_split -> prefetch_next_split, the fence before the next split's expert uploads) -> now 2105-2109

- original:
  ```
  // conservative fence: slot regions are dedicated (no aliasing with activations),
  // but same-shard slot REUSE across splits still requires the previous tenant's
  // consumer to finish. A per-bid latest-consumer event is NOT sufficient: the
  // prefetch for split N+2 is enqueued before split N launches, and N+2's slot bytes
  // are typically N's own slot, so a narrower fence overwrites experts still being
  // read. A sound per-slot fence needs previous-same-bytes-tenant tracking; the
  // upload/compute serialization is host-DRAM-bandwidth bound, so it has little headroom.
  ```
- kept as:
  ```
  // conservative fence: slot regions never alias activations, but same-shard slot reuse across
  // splits still needs the previous tenant's consumer to finish. A per-bid latest-consumer event
  // is not enough: the prefetch for split N+2 is enqueued before split N launches, into what is
  // typically N's own slot, so it would overwrite experts still being read; a narrower per-slot
  // fence would have to track the previous tenant of the same bytes
  ```
- removed information:
  - "the upload/compute serialization is host-DRAM-bandwidth bound, so it has little headroom": an experiment conclusion about how much decode speed a narrower fence could recover. It states that the copy stream waiting on compute_events[fence_bid] (uploads chained behind the previous split's compute) was judged not to be the bottleneck because the uploads are bound by host DRAM read bandwidth. Note that the memory file s4-serialization-mechanism.md records the opposite reading for the s4 tier (per-slot fence named as the lever, ~2x s4 decode at stake), so the sentence was contested as well as narrative; the code comment now states only the correctness argument for the conservative fence.
  - the design hint "a sound per-slot fence needs previous-same-bytes-tenant tracking" is kept in shorter form (last clause of the new text).

## ggml/src/ggml-backend.cpp:2206-2209 (ggml_backend_sched_compute_splits -> run_split, input loop, produced_by_skipped branch) -> now 2204-2206

- original:
  ```
  // the producer split was skipped as all-zero: zero the consumer's copy in stream order on
  // the split backend instead of moving it (input_cpy is the tensor itself when the consumer
  // shares the producer's backend). Replaces a host synchronize + host->device upload per
  // dead expert chain, which left the GPU idle before every pooled layer
  ```
- kept as:
  ```
  // the producer split was skipped as all-zero: zero the consumer's copy in stream order on
  // the split backend instead of moving it (input_cpy is the tensor itself when the consumer
  // shares the producer's backend); a host synchronize + upload here would idle the GPU before every pooled layer
  ```
- removed information:
  - "Replaces a host synchronize + host->device upload per dead expert chain, which left the GPU idle before every pooled layer": before/after history of the dead-chain skip. Before this branch existed, a split whose compute was skipped by split_skip_cb (all-zero on the callback's word) still had its result moved to the consumer through the generic path: ggml_backend_synchronize(input_backend) + ggml_backend_tensor_copy, i.e. a host round trip and a host->device upload per skipped expert chain, one before every pooled MoE layer's GPU chain. The memory file wddm-engine-transitions.md ("kernel copies + dead-chain skip + no per-layer drains took hybrid 66 -> 89 t/s, fetch 77 -> 96 (q35 @8000)") holds the measured effect. The comment now states the why in the present tense only.

## ggml/src/ggml-backend.cpp:2365-2371 (ggml_backend_sched_compute_splits -> run_split, input loop, device -> CPU split input branch; regex hit on line 2370) -> now 2362-2365

- original:
  ```
  // device -> CPU split input: an asynchronous download on the producer's stream (a kernel
  // copy into the pinned CPU buffer with kernel copies on, else a DMA) in place of a
  // host synchronize followed by a synchronous copy; the producer is synchronized right
  // before the CPU split computes. The synchronous path leaves a copy-engine gap behind
  // every pooled layer's router kernels (a copy-engine transfer ordered against kernels
  // idles the GPU under WDDM). Stock layouts (no redirects) keep the byte-identical
  // synchronous path.
  ```
- kept as:
  ```
  // device -> CPU split input: download asynchronously on the producer's stream (a kernel copy
  // into the pinned CPU buffer when kernel copies are on, else a DMA) and fence it right before
  // the CPU split computes; a host synchronize here would idle the GPU behind every pooled
  // layer's router kernels (a copy-engine transfer ordered against kernels idles the GPU under WDDM)
  ```
- removed information:
  - "in place of a host synchronize followed by a synchronous copy" and "The synchronous path leaves a copy-engine gap behind every pooled layer's router kernels": before/after narrative of the same change as the block above (the async device->host download replaced ggml_backend_synchronize + ggml_backend_tensor_copy for the CPU split's inputs on pool tiers). Measured effect is in the memory file wddm-engine-transitions.md.
  - "Stock layouts (no redirects) keep the byte-identical synchronous path": an A/B statement that non-pshard layouts are unaffected. It is already expressed by the code (the branch is gated on pshard_async && sched->has_redirects and the else path is the upstream ggml_backend_tensor_copy) and by the comment on the fallback branch a few lines below ("device->host handoff: the pool tiers take the asynchronous path above; this synchronous copy is the legacy tiers'").
  - kept: "(a copy-engine transfer ordered against kernels idles the GPU under WDDM)". This is the mechanism in general terms, which the brief lists verbatim as allowed. It is the one remaining hit of the brief's grep regex on this file (the regex matches the token WDDM); remove the parenthetical if a zero count is preferred over the accurate platform scoping.

## ggml/src/ggml-cpu/ggml-cpu.c

# ggml/src/ggml-cpu/ggml-cpu.c - comment rewrite notes

Rewritten comments: 0. Source file not modified.

## Regex hits (step 1)

- line 1410 (ggml_compute_forward_mul_mat) - original text:
  `//   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915`
  - kept as: unchanged. Upstream ggml text (introduced by ae8de6d50 "ggml : build backends as libraries", rationale from upstream PR #6915), not fork narrative. It carries no date, measured number, cell name, review reference or machine name; "measured" trips the regex only. Rewriting an upstream comment would diverge from upstream for no fork-convention gain and add a rebase conflict, so it is left as the one legitimate remainder.

## Fork-added comments reviewed (step 2)

Commits touching this file on the fork: e3a959c51 (-1 skipped route in MUL_MAT_ID / ADD_ID), 1783270e7 (expert-pool review fixes), bb8cc8d16 (cherry-pick of upstream CPU miss-chain kernels, no comment additions).

- lines 1624-1625 (ggml_compute_forward_mul_mat_id) - `// -1 route (allow_skip): no expert computes it - the consumer sums / per-route outputs, so the dst row must be zeros` - already what/why only, no change.
- assert note `/* allow_skip; op_params[0] is precision */` (ggml_compute_forward_mul_mat_id) - already what/why only, no change.

Other scan hits were upstream and clean: line 2953 "Decode path" (false positive on "Dec"), lines 3867/3869 "200ms" (KMP_BLOCKTIME value restated, upstream), line 2331 FIXME on get_rows threading (upstream).

## Step 3

Regex re-run: 1 remaining (line 1410, upstream comment described above).

## ggml/src/ggml-cuda/ggml-cuda.cu

# ggml/src/ggml-cuda/ggml-cuda.cu - comment rewrite notes

Line endings: CRLF (6386 lines before, 6370 after; pure CRLF preserved, verified byte-level). Only `//` comment text changed; the code on every edited line is identical before and after (verified by stripping comment tails and diffing).

Line numbers below are the pre-edit positions.

## Rewritten comments

- 2550-2561 (file-scope header above `struct ggml_cuda_stage_ring`) - original: "cudaMemcpyAsync from PAGEABLE host memory is host-synchronous and, on file-backed mmap pages, runs at ~4 GB/s (Windows, DeepSeek-V4's 45 GB shard that exceeded the page-lock ceiling; pinned copies run ~40 GB/s on the same box). Stage such copies through a small pinned ring: worker threads memcpy chunk k+1 into the ring while the GPU DMAs chunk k from it. The call stays host-synchronous for the duration (exactly like the pageable copy it replaces) but at memcpy speed (~30 GB/s with 8 threads) instead of the driver's pageable path. Ring: 512 MiB = 8 x 64 MiB chunks, 8 memcpy threads per chunk (measured 2026-09-02). Pinned/registered/device sources, copies < 1 MiB, and copies issued during graph capture take the direct path. One ring per PHYSICAL device: a CUDA event can only be recorded on a stream of the device it was created on, so the slot events are created under that device (review finding 2026-09-02); the pinned buffers are portable and the host memcpy bandwidth is shared regardless." - kept as: "cudaMemcpyAsync from PAGEABLE host memory is host-synchronous and slow on file-backed mmap pages (a model mapping that exceeds the page-lock ceiling stays pageable). Stage such copies through a small pinned ring: worker threads memcpy chunk k+1 into the ring while the GPU DMAs chunk k from it. The call stays host-synchronous for the duration (like the pageable copy it replaces) but at host memcpy speed instead of the driver's pageable path. Pinned/registered/device sources, copies < 1 MiB, and copies issued during graph capture take the direct path. One ring per PHYSICAL device: a CUDA event can only be recorded on a stream of the device it was created on, so the slot events are created under that device; the pinned buffers are portable and the host memcpy bandwidth is shared regardless."
  Removed: pageable ~4 GB/s vs pinned ~40 GB/s on the dev box (Windows); DeepSeek-V4's 45 GB shard as the case that exceeded the page-lock ceiling; ~30 GB/s with 8 memcpy threads; ring geometry 512 MiB = 8 x 64 MiB measured 2026-09-02 (the geometry itself is in the code: `chunk = 64 << 20`, `mb = 512`); the per-device slot-event placement was a review finding of 2026-09-02.

- 2566-2570 (`ggml_cuda_stage_ring::n_threads` member) - original: "memcpy threads per copy: the logical core count, capped at 8 - the measured saturation point for both prefill (64 MiB chunks, q35 all-pageable: 2 -> 115, 4 -> 157, 8 -> 211, 12 -> 206 t/s prompt) and decode misses (10 MB, persistent pool, locked clocks, DSv4: 1 -> 15.3, 8 -> 16.2, 12 -> 16.1, 16 -> 16.2 t/s). Fewer copiers leave cores to the CPU-route miss policies." - kept as: "memcpy threads per copy: the logical core count, capped at 8 where the host memcpy bandwidth saturates; the remaining cores stay with the CPU-route miss policies"
  Removed: the thread-count sweep. Prefill, 64 MiB chunks, q35 all-pageable: 2 threads -> 115, 4 -> 157, 8 -> 211, 12 -> 206 t/s prompt. Decode misses, 10 MB, persistent pool, locked clocks, DSv4: 1 -> 15.3, 8 -> 16.2, 12 -> 16.1, 16 -> 16.2 t/s.

- 2586 (`ggml_cuda_stage_ring::init`) - original: "ring: 8 x 64 MiB chunks (the loader's chunking; measured 2026-09-02)" - kept as: "ring: 8 x 64 MiB chunks, the loader's chunking"
  Removed: measured 2026-09-02.

- 2615-2618 (file-scope header above `struct ggml_cuda_memcpy_pool`) - original: "Persistent memcpy pool for the staging ring. The ring's copy leg is the pipeline's slow stage (one thread ~12 GB/s vs ~25 GB/s DMA: a 10 MB expert miss cost 0.83 ms staged vs 0.42 ms pinned, 2026-09-04), and spawning threads per copy only paid off above 16 MiB, so every decode-sized miss ran single-threaded. The pool's threads live for the process." - kept as: "Persistent memcpy pool for the staging ring. The ring's copy leg is the pipeline's slow stage (one host thread is slower than the DMA), and spawning threads per copy costs more than it saves for decode-sized misses, so the pool's threads live for the process."
  Removed: one memcpy thread ~12 GB/s vs ~25 GB/s DMA; a 10 MB expert miss cost 0.83 ms staged vs 0.42 ms pinned (2026-09-04); per-copy thread spawning only paid off above 16 MiB, so before the pool every decode-sized miss ran single-threaded.

- 2624-2626 (same block, protocol paragraph) - original: "run() rewrites the job (the review of 2026-09-04 found that a straggler still in work() could compare a stale part index against the next job's n_parts and over-count done_parts, letting the caller return before a part had landed)." - kept as: "run() rewrites the job (a straggler still in work() would compare a stale part index against the next job's n_parts and over-count done_parts, letting the caller return before a part landed)."
  Removed: this was a review finding of 2026-09-04.

- 2647-2648 (`ggml_cuda_memcpy_pool::start`, Windows thread priority) - original: "a copier descheduled mid-part stalls the whole upload for a scheduler quantum (~15 ms): measured as 5-11 t/s outliers among 16 t/s runs on DSv4 (2026-09-04)" - kept as: "a copier descheduled mid-part stalls the whole upload for a scheduler quantum"
  Removed: the quantum is ~15 ms; symptom was 5-11 t/s outlier runs among 16 t/s runs on DSv4 (2026-09-04).

- 2709-2711 (above `ggml_cuda_get_memcpy_pool`) - original: "process-lifetime, never destroyed (like the stage workers): a static destructor in this DLL would run at DLL_PROCESS_DETACH after Windows has already killed the worker threads, and a join / a mutex a dead thread held would hang the exit (review finding 2026-09-04)" - kept as: same text without "(review finding 2026-09-04)"
  Removed: review finding 2026-09-04.

- 2774-2776 (file-scope header above `struct ggml_cuda_stage_worker`) - original: "The synchronous pipeline blocks the calling (scheduler) thread for the whole memcpy (~70 ms per 2 GB layer), so the GPU idles before that layer's compute is launched. The worker takes the COPY jobs instead and the caller returns at once. Stream ORDER is preserved by routing through ..." - kept as: "The synchronous pipeline blocks the calling (scheduler) thread for the whole memcpy of a layer, so the GPU idles before that layer's compute is launched. The worker takes the COPY jobs instead and the caller returns at once. Stream ORDER is preserved by routing through ..."
  Removed: ~70 ms per 2 GB layer.

- 2779 (same block) - original: "a wait/synchronize on such an event first joins the worker up to that job" - kept as: "a wait or synchronize on such an event first joins the worker up to that job"
  Regex false positive only ("wai t/s ynchronize" matched `t/s`); reworded so the grep is clean, meaning unchanged.

- 2786-2789 (`ggml_cuda_stage_worker::job_type`) - original: "COPY: staged pageable upload; RECORD: deferred event record; H2D/D2H/MEMSET: ordinary stream operations the scheduler issued on a stream that still had staged copies queued - queued behind them instead of draining, so the caller never blocks (review finding: the padding memset and KV writeback uploads right after a staged weight re-serialized every layer)" - kept as: "... so the caller never blocks (a direct memset or KV writeback upload right after a staged weight would otherwise re-serialize the layer)"
  Removed: this was a review finding; the concrete offenders were the padding memset and the KV writeback uploads issued right after a staged weight, which re-serialized every layer before H2D/D2H/MEMSET jobs existed.

- 2903 (`ggml_cuda_stage_async_enabled`) - original: "the worker won its measurement 2026-09-02; the synchronous pipeline is its implementation" - kept as: "always on; the synchronous pipeline is the worker's implementation"
  Removed: the async worker was chosen over the synchronous pipeline by a measurement on 2026-09-02.

- 2965 (above `g_stage_events`) - original: "deferred event records: event -> (worker, job seq) until the next wait/synchronize on it" - kept as: "... until the next wait or synchronize on it"
  Regex false positive only (`t/s` inside "wait/synchronize"); reworded, meaning unchanged.

- 3064-3065 (above `k_kernel_copy_segs`) - original: "batched upload: one launch for a layer's expert-row segments (the per-tensor launches left ~30 us gaps between consecutive copy kernels and delayed the CPU chain's start by the host's issue time)" - kept as: "batched upload: one launch for a layer's expert-row segments; per-tensor launches leave gaps between consecutive copy kernels and delay the CPU chain's start by the host's issue time"
  Removed: the per-tensor launch gaps were ~30 us.

- 3110-3111 (`ggml_backend_cuda_copy_segments_async`, blocks_per_seg) - original: "one int4 per thread for the largest segment (PCIe reads need many requests in flight: 22 blocks per 704 KB segment ran at 16 GB/s, the per-tensor kernels with one int4 per thread at ~30 GB/s)" - kept as: "one int4 per thread for the largest segment: PCIe reads need many requests in flight"
  Removed: 22 blocks per 704 KB segment ran at 16 GB/s; the per-tensor kernels with one int4 per thread at ~30 GB/s.

- 3135-3136 (above `ggml_cuda_kernel_fill`) - original: "memset by a kernel: on WDDM cudaMemsetAsync behaves like a copy-engine op (traced 2026-09-10: 36 us of GPU idle between a 64 KB memset and the kernel after it)" - kept as: "memset by a kernel: on WDDM cudaMemsetAsync behaves like a copy-engine op and idles the GPU before the next kernel"
  Removed: traced 2026-09-10, 36 us of GPU idle between a 64 KB memset and the following kernel.

- 3755-3762 (`ggml_cuda_check_fusion_memory_ranges`, op-NONE src check) - original: "... so the (unfused-safe) dst placement lands inside bytes the FUSED kernel is still reading (measured: ffn_moe_swiglu allocated inside ffn_gate_exps' slot). Weights in dedicated model buffers ..." - kept as: "... so the (unfused-safe) dst placement lands inside bytes the FUSED kernel is still reading. Weights in dedicated model buffers ..."
  Removed: the observed instance was ffn_moe_swiglu allocated inside ffn_gate_exps' slot.

- 3810-3812 (`ggml_cuda_can_fuse`, above the mul_mat(_id)+bias+GLU fusion branch) - original: "certified for pshard 2026-09-01: the corruption was the memory-range check skipping op-NONE srcs (sched slot copies), letting the fused dst legally land in just-freed weight-copy bytes the kernel still reads. The check sees them now." - kept as: deleted (pure history; the mechanism is documented at the check itself in `ggml_cuda_check_fusion_memory_ranges`, and the code here is upstream's fusion gating).
  Removed: the fused-GLU corruption was certified fixed for pshard on 2026-09-01; root cause = the memory-range check skipped op-NONE srcs (sched slot copies), so the fused dst legally landed in just-freed weight-copy bytes the kernel still reads. See memory note fused-glu-root-cause.md.

- 5449-5451 (`ggml_backend_cuda_unregister_host_buffer`) - original: "no env gate (unlike register): callers only unregister regions they registered, and the env may legitimately be unset by then (e.g. pshard stock-fallback) - gating here would silently leak the page-lock" - kept as: "unconditional: callers only unregister regions they registered, and a skipped unregister leaks the page-lock"
  Removed: the comparison with a register-side env gate (upstream's GGML_CUDA_REGISTER_HOST) that no longer exists in this fork after the env back doors were removed, and the pshard stock-fallback example of the env being unset by unregister time.

## Flagged by the regex, left as-is

- 1100 (`ggml_backend_cuda_device_offload_op`-area split heuristic) - original: `// The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.` - kept as: unchanged.
  Upstream llama.cpp comment (Johannes Gaessler, commit d6f303004, PR #19378 backend-agnostic tensor parallelism); the fork never touched this region. Rewriting it would diverge from upstream in an untouched region.

- 3036 pre-edit / 3028 post-edit (above `ggml_cuda_kernel_copy_flag`) - original: "kernel copies: pinned host transfers below the cap run as kernels through the device mapping instead of copy-engine transfers (on WDDM an engine switch ordered against kernels idles the GPU); the runtime sets the cap from the machine profile ("ggml_backend_kernel_copy_max_set")" - kept as: unchanged.
  Matches `WDDM`, but states the mechanism in general terms (the explicitly allowed form); no number, date, machine name or history.

- 3135 pre-edit / 3126 post-edit (above `ggml_cuda_kernel_fill`, after rewrite) - "memset by a kernel: on WDDM cudaMemsetAsync behaves like a copy-engine op and idles the GPU before the next kernel" - still matches `WDDM`; same allowed-mechanism reasoning as 3036.

## ggml/src/ggml-cuda/mmvq.cu

# ggml/src/ggml-cuda/mmvq.cu - comment rewrite notes

Line endings: LF (0 CRLF lines). File left byte-identical: no comment was rewritten.

## Rewritten comments

(none)

## Flagged by the regex, left as-is (upstream llama.cpp lines, not fork-authored)

- 296 (ggml_cuda_should_use_mmvq) - original: `switch (type) { // tuned on RTX 4090` - kept as: unchanged.
  Upstream commit 2b5621094 (llama.cpp PR #26079 "CUDA: adding switch points per HW and quant type to tune the mvq->MMQ decode crossover"). The fork never touched this region. The machine name is upstream's own attribution of the empirical ne11 thresholds in the ADA_LOVELACE branch. Rewriting it would diverge from upstream in an untouched region (rebase conflicts on every upstream sync of this file) and drop the information a maintainer needs when re-tuning the thresholds; a name-free rewrite ("tuned for Ada Lovelace") would only repeat the `cc == GGML_CUDA_CC_ADA_LOVELACE` check on the line above.
- 309 (ggml_cuda_should_use_mmvq) - original: `switch (type) { // tuned on RTX 5090` - kept as: unchanged. Same upstream commit, BLACKWELL branch, same reasoning.
  Siblings 322 (`// tuned on DGX Spark GB10`) and 359 (`// tuned for CDNA2`) are the same upstream pattern; the regex did not flag them.

## Fork-authored comments reviewed (step 2), kept unchanged

The fork's only change to this file is commit e3a959c51 (-1 = skipped route in MUL_MAT_ID / ADD_ID). It added two identical comments:

- 580 (mul_mat_vec_q) - `// a -1 id (skipped route) computes against expert 0 and writes zeros` - kept as is: one line, states what the code does and why; no dates, measured numbers, machine names, cell names or history.
- 807 (mul_mat_vec_q_moe) - same text - kept as is, same reason.

No other pshard-related comments exist in the file. The remaining comments (624 "Hide latency by prefetching...", 941-942 small-K trigger, 977-993 GB10 nwarps heuristic, etc.) are upstream text and out of scope.

## Step 3 grep remainder

2 lines (296, 309), both the upstream comments above.

Override path: if the decision is that upstream machine names must also go, the minimal rewrite is `// thresholds tuned for this arch` on 296 and 309 (and on 322/359 for consistency); update this note with the original text when doing so.

## src/llama-benchmark.cpp

# comment rewrite notes: src/llama-benchmark.cpp

Line numbers refer to the file before the rewrite. Original text is preserved verbatim so the removed
dates, measurements, cell names and history stay recoverable.

Regex hits: 6 lines in 4 blocks (365-366, 637-641, 814-820, 867-871). Three more blocks carried the same
kind of narrative without matching the regex (298, 603-608, 793-797) and were trimmed too. Log-message
strings at 98, 388 and 391 print runtime values only and were left alone.

## src/llama-benchmark.cpp:298 (llama_benchmark_predictor::load_cpu, schema 2 header parsing)

Original text:

```
// schema 2 lines: kernel-copy era measurements and the machine fingerprint
```

Kept as:

```
// schema 2 lines: kernel-copy upload rates, pool latencies and the machine fingerprint
```

Removed: "era" (project-history framing).

## src/llama-benchmark.cpp:365-366 (llama_benchmark_predictor::load_cpu, cpu_matmul_floor_gflops)

Original text:

```
// conservative compute floor for quantized matmuls with no benchmark entry:
// the slowest measured CPU matmul rate (any quant, any shape)
```

Kept as:

```
// conservative compute floor for quantized matmuls with no benchmark entry:
// the slowest CPU matmul rate in the profile (any quant, any shape)
```

Removed: the word "measured" (regex trigger); meaning unchanged.

## src/llama-benchmark.cpp:603-608 (llama_benchmark_predictor::predict_split, nearest-neighbor memory-bound branch)

Original text:

```
// memory-bound op. The matched entry's observed bytes/s is only a
// bandwidth measurement if that entry ALSO ran memory-bound; a
// compute-bound benchmark's byte throughput is an artifact far below
// the machine's streaming rate (seen: FLASH_ATTN entry at 2.6 GB/s on
// a 45.7 GB/s machine -> CPU attention priced 18x too slow, hiding
// attn-pinned plans from the search)
```

Kept as:

```
// memory-bound op. the matched entry's observed bytes/s is a bandwidth
// only if that entry also ran memory-bound; a compute-bound entry's byte
// throughput sits far below the machine's streaming rate and would price
// the op far too slow, hiding plans that pin it from the search
```

Removed: the observed case - a FLASH_ATTN entry at 2.6 GB/s on a 45.7 GB/s machine priced CPU attention 18x
too slow and hid attn-pinned plans from the search.

## src/llama-benchmark.cpp:637-641 (llama_benchmark_predictor::predict_split, fallback branch compute floor)

Original text:

```
// quantized CPU matmuls with no benchmark entry (e.g. IQ quants) are
// dequant-compute-bound at batch; memory bandwidth alone under-charges
// them 10-20x. Floor with the slowest measured CPU matmul rate.
// batch only: at small row counts the matmul is memory-bound (dequant
// streams at DRAM speed) and bytes/bw is already the right price
```

Kept as:

```
// quantized CPU matmuls with no benchmark entry (e.g. IQ quants) are
// dequant-compute-bound at batch and memory bandwidth alone under-charges
// them; floor with the slowest CPU matmul rate in the profile.
// batch only: at small row counts the matmul is memory-bound (dequant
// streams at DRAM speed) and bytes/bw is already the right price
```

Removed: the under-charge magnitude (10-20x) and the word "measured".

## src/llama-benchmark.cpp:793-797 (llama_benchmark_predictor::predict_tps, writeback ratios)

Original text:

```
// per-class writeback ratios, matching what the runtime actually moves:
//   attention KV: write-cells delta sync -> batch_size/kv_size in both directions
//   recurrent state: FULL mode, entire tensor every eval -> 1.0
// A blanket has_rs ratio charged hybrid models' attention KV at full-cache cost per
// decode step, under-predicting pinned-attn plans 2-12x at 16k ctx.
```

Kept as:

```
// per-class writeback ratios, matching what the runtime actually moves:
//   attention KV: write-cells delta sync -> batch_size/kv_size in both directions
//   recurrent state: FULL mode, entire tensor every eval -> 1.0
// one ratio for both would charge a hybrid model's attention KV at full-cache cost
// per decode step and under-predict plans that pin attention
```

Removed: the history that the code previously used a blanket has_rs ratio, and its measured effect
(pinned-attn plans under-predicted 2-12x at 16k ctx).

## src/llama-benchmark.cpp:814-820 (llama_benchmark_predictor::predict_tps, GPU split input-copy pricing)

Original text:

```
// a prefetched split still pays its sliced-by-used-ids expert copies at
// consume time (the prefetch pass skips those tensors on purpose).
// Sliced expert uploads are many small gathered transfers and run far
// below peak PCIe (measured 29.5 GB/s at 0.6MB chunks vs 45 peak);
// pricing them at peak over-predicted sliced strategies by 10-24% and
// mis-ranked 6 of 14 audited cells. Price the sliced share at the
// profiled chunk-size-dependent rate, the contiguous rest at peak.
```

Kept as:

```
// a prefetched split still pays its sliced-by-used-ids expert copies at
// consume time (the prefetch pass skips those tensors on purpose).
// sliced expert uploads are many small gathered transfers and run far
// below peak PCIe, so pricing them at peak over-predicts sliced strategies;
// price the sliced share at the profiled chunk-size rate, the contiguous rest at peak.
```

Removed: measured sliced-upload rate (29.5 GB/s at 0.6 MB chunks vs 45 GB/s peak), the over-prediction
range (10-24%) and the audit outcome (6 of 14 audited cells mis-ranked when priced at peak).

## src/llama-benchmark.cpp:867-871 (llama_benchmark_predictor::predict_tps, prefetch cost bandwidth)

Original text:

```
// prefetch cost: use concurrent PCIe BW for CPU splits (bus shared with DRAM),
// peak PCIe BW for GPU splits (GPU compute doesn't contend with PCIe DMA).
// The per-op aggregate eff_pcie_bw can be far below the machine's measured
// concurrent PCIe rate (same unrepresentative-entry disease as the memory-branch
// bandwidth) - floor it with the profiler's directly measured concurrent value.
```

Kept as:

```
// prefetch cost: use concurrent PCIe BW for CPU splits (bus shared with DRAM),
// peak PCIe BW for GPU splits (GPU compute doesn't contend with PCIe DMA).
// the per-op aggregate eff_pcie_bw can sit far below the machine's concurrent
// PCIe rate when the matched entries are unrepresentative (as in the memory-bound
// branch) - floor it with the profiler's own concurrent value
```

Removed: the two uses of "measured" (regex trigger); the mechanism is unchanged.

## totals

31 original comment lines rewritten into 27 lines across 7 blocks. No code, identifiers, whitespace outside
comments, string literals or line endings changed (file stays CRLF).

## src/llama-benchmark.h

# Comment rewrite notes: src/llama-benchmark.h

Line numbers are from the file before the rewrite. All entries are field or method comments inside `struct llama_benchmark_stats`.

## src/llama-benchmark.h:71-74 (llama_benchmark_stats::cpu_matmul_floor_gflops)

original text:

    // derived at load: the slowest measured CPU matmul rate. Quantized matmuls with
    // no benchmark entry for their type (e.g. IQ quants) are dequant-compute-bound;
    // pricing them by memory bandwidth alone under-charges 10-20x, so the fallback
    // uses max(bytes/bw, ops/this) as a conservative compute floor.

removed: the measured under-charge factor (10-20x) for pricing IQ-quant matmuls by bandwidth alone.

kept as:

    // derived at load: the slowest CPU matmul rate in the profile. Quantized matmuls with
    // no benchmark entry for their type (e.g. IQ quants) are dequant-compute-bound, so
    // pricing them by memory bandwidth alone under-charges them; the fallback uses
    // max(bytes/bw, ops/this) as a conservative compute floor.

## src/llama-benchmark.h:77-80 (llama_benchmark_stats::sliced_bw)

original text:

    // gathered-slice upload bandwidth curve (from cpu profiler header, measured under
    // concurrent CPU memory load): effective host->device BW for small strided chunks.
    // Sliced expert uploads (0.3-13 MB per expert) run far below peak PCIe; pricing
    // them at peak over-predicted sliced strategies by 10-24% (selector-gap audit).

removed: the per-expert slice size range (0.3-13 MB), the over-prediction magnitude (10-24%) and the experiment reference (selector-gap audit).

kept as:

    // gathered-slice upload bandwidth curve (from cpu profiler header, taken under
    // concurrent CPU memory load): effective host->device BW for small strided chunks.
    // sliced expert uploads run far below peak PCIe, so pricing them at peak
    // over-predicts sliced strategies.

## src/llama-benchmark.h:85-88 (llama_benchmark_stats::host_pin_ceiling_gb)

original text:

    // host pin ceiling (GB): how much ordinary process memory the driver will
    // page-lock (cudaHostRegister), measured by the profiler by registering
    // chunks until refusal. 0 = not in the profile -> assume every mmap
    // mapping page-locks (the pre-per-mapping behavior).

removed: the history note that "assume every mmap mapping page-locks" was the behavior before per-mapping page-lock prediction existed.

kept as:

    // host pin ceiling (GB): how much ordinary process memory the driver will
    // page-lock (cudaHostRegister); the profiler finds it by registering
    // chunks until refusal. 0 = not in the profile -> assume every mmap
    // mapping page-locks.

## src/llama-benchmark.h:97-102 (llama_benchmark_stats::upload_staged_bw / upload_staged_frac)

original text:

    // mixture terms behind upload_bw: the staged mappings' rate and their byte
    // fraction. A split's streamed weights come from ONE mapping (layers are
    // contiguous in the file), so a pass mixes pinned-rate and staged-rate
    // splits; pricing every split at the blended rate lets predicted compute
    // hide the staged splits' stalls under the overlap max (measured 117 vs 84
    // t/s on a half-staged model; the mixture prices ~100).

removed: the calibration point - on a half-staged model the measured decode was 84 t/s where the blended-rate mixture predicted ~100 t/s and the all-pinned pricing predicted 117 t/s; this motivated keeping the staged rate and fraction as separate terms.

kept as:

    // mixture terms behind upload_bw: the staged mappings' rate and their byte
    // fraction. A split's streamed weights come from ONE mapping (layers are
    // contiguous in the file), so a pass mixes pinned-rate and staged-rate
    // splits; pricing every split at the blended rate would let predicted compute
    // hide the staged splits' stalls under the overlap max.

## src/llama-benchmark.h:106-107 (llama_benchmark_stats::machine)

original text:

    // machine fingerprint (schema 2 profiles, 2026-09-12): the machine that measured the profile. The
    // planner compares it with the running machine; a profile from another box prices nothing.

removed: the date the schema-2 fingerprint landed (2026-09-12).

kept as:

    // machine fingerprint (schema 2 profiles): the machine that produced the profile. The
    // planner compares it with the running machine; a profile from another box prices nothing.

## src/llama-benchmark.h:124 (llama_benchmark_stats, kernel-copy block header; not a grep hit, same narrative)

original text:

    // kernel-copy era measurements (schema 2). 0 / -1 = not in the profile; there is no fallback value:

removed: the "era" framing (these fields were added when the pool moved from DMA uploads to kernel copies).

kept as:

    // kernel-copy path terms (schema 2). 0 / -1 = not in the profile; there is no fallback value:

## src/llama-benchmark.h:132-133 (llama_benchmark_stats::engine_switch_us)

original text:

    double engine_switch_us   = -1.0;   // copy-engine transition: DMA 8 KB readback behind a kernel minus the kernel-copy version;
                                        // -1 = not in the profile (a measured 0.0 is legitimate on a box without the WDDM fence)

removed: the probe size used by the profiler (an 8 KB readback), the "box" phrasing, and the WDDM name (the fence is the WDDM driver's copy-engine/compute-engine transition; a machine whose driver has no such fence legitimately profiles 0.0 here).

kept as:

    double engine_switch_us   = -1.0;   // copy-engine transition: a small DMA readback ordered behind a kernel minus the kernel-copy version;
                                        // -1 = not in the profile (0.0 is legitimate where the driver imposes no copy-engine transition fence)

## src/llama-benchmark.h:160-161 (llama_benchmark_stats::slice_bw)

original text:

    // interpolated gathered-upload BW for a chunk size on the copy engine; falls back to the measured
    // concurrent rate (eff_pcie_bw), then peak, when the curve is not in the profile

removed: only the word "measured" (the fallback is the profile's concurrent PCIe rate; no information lost).

kept as:

    // interpolated gathered-upload BW for a chunk size on the copy engine; falls back to the concurrent
    // PCIe rate (eff_pcie_bw), then peak, when the curve is not in the profile

## src/llama-benchmark.h:168-169 (llama_benchmark_stats::slice_bw_kernel)

original text:

    // the same for the pool's per-layer path (segment-batch kernel), under the CPU chain's load or with the
    // CPU idle; 0 = not measured, no fallback

removed: "not measured" -> "not in the profile" (same meaning, matches the wording used by the neighbouring fields).

kept as:

    // the same for the pool's per-layer path (segment-batch kernel), under the CPU chain's load or with the
    // CPU idle; 0 = not in the profile, no fallback

## Reviewed and left unchanged

- 92-95 (upload_bw): mechanism only, no numbers or history.
- 116-122 (machine_current / machine_hash / machine_mismatch): mechanism only.
- 126-131 (sliced_kernel_bw, segs_kernel_bw, segs_kernel_idle_bw, staged_bw, kernel_copy_cap_mb): mechanism only; "hybrid / cpu_admit" are execution-model names in the code, not experiment cells.
- 134-135 (pool_serve_us, pool_split_us): mechanism only.
- 244-251 (llama_benchmark_predictor::breakdown / predict_tps): mechanism only.

## src/llama-context.cpp

# Comment rewrite notes: src/llama-context.cpp

Line numbers are the pre-edit line numbers of the working copy (branch pshard-tot, 4632 lines before, 4629 after).
Only comment text changed; no code, identifiers, whitespace outside comments, string literals or line endings (CRLF) were touched.

## src/llama-context.cpp:933-940 (llama_context::memory_update) - active-tier layout restore after a memory update

Original text:

    // pshard: the active tier's warmup reserve IS its worst case and its saved allocation is the
    // arena's layout. The stock worst-case reserve below would place a min(n_ctx, n_ubatch)-token
    // graph into the active tier's window through ggml_backend_sched_reserve, which grows
    // overflow chunks outside the budget with no refusal - and a stale chunk it left behind
    // would stay resident for the context's lifetime (review 2026-09-06; the K-shift path is
    // the one llama_kv_cache::update marks "pshard + KV shift -- testing pending"). Restore the
    // tier's layout instead: a runtime graph it does not fit re-reserves in alloc_splits under
    // the overflow refusal.

Removed: the review reference "review 2026-09-06" and the status note that the K-shift path is the one llama_kv_cache::update marks "pshard + KV shift -- testing pending" (i.e. the K-shift path through this branch was still untested at the time of that review).

Kept as:

    // pshard: the active tier's warmup reserve is its worst case and its saved allocation is the
    // arena's layout. The stock worst-case reserve below would place a min(n_ctx, n_ubatch)-token
    // graph into the active tier's window through ggml_backend_sched_reserve, which grows
    // overflow chunks outside the budget with no refusal, and a stale chunk it left behind
    // would stay resident for the context's lifetime. Restore the tier's layout instead: a
    // runtime graph it does not fit re-reserves in alloc_splits under the overflow refusal.

## src/llama-context.cpp:1503-1506 (llama_context::process_ubatch) - result reset after ggml_backend_sched_alloc_graph fails

Original text:

    // the graph's tensors hold no (or stale) allocations: an equal-shape ubatch that came
    // next would take the reuse branch above and compute on them (review 2026-09-06: the
    // pshard arena's overflow refusal made this a recurring soft error, and callers such as
    // speculative-simple keep decoding after -2). Same reset as graph_reserve / memory_update.

Removed: the review reference "review 2026-09-06", the history that the overflow refusal "made this a recurring soft error", and the named caller example (speculative-simple).

Kept as:

    // the graph's tensors hold no (or stale) allocations: an equal-shape ubatch that came
    // next would take the reuse branch above and compute on them (the pshard arena's overflow
    // refusal makes this a soft error and callers keep decoding after -2). Same reset as
    // graph_reserve / memory_update.

## src/llama-context.cpp:1514-1517 (llama_context::process_ubatch) - arena spill warning

Original text:

    // the arena is the budget: a graph the scheduler could only place by growing overflow
    // chunks (ggml_backend_sched_alloc_splits' re-reserve) ran outside it. Name the ubatch
    // so the shape mismatch against the tier's reserve can be traced (2026-09-06: DSv4 +
    // DSpark at 8000 spilled 1088 MiB on its first decode graph, 10757 nodes).

Removed: the dated measurement "2026-09-06: DSv4 + DSpark at 8000 spilled 1088 MiB on its first decode graph, 10757 nodes" (the observation that motivated naming the ubatch in the warning: DeepSeek-V4 with the DSpark speculative draft at an 8000-token context spilled 1088 MiB of overflow chunks on its first decode graph of 10757 nodes).

Kept as:

    // the arena is the budget: a graph the scheduler could only place by growing overflow
    // chunks (ggml_backend_sched_alloc_splits' re-reserve) ran outside it. Name the ubatch
    // so the shape mismatch against the tier's reserve can be found.

## src/llama-context.cpp:1907-1908 (llama_context::decode) - one-time pshard_prefill_ubatch_eff log

Original text:

    // eval shape changes numerics on shape-sensitive models - log once so A/B
    // baselines can match -ub to what pshard actually evaluates with

Removed: the experiment reference "A/B baselines" (the log exists so the stock arm of an A/B perf pair can be run with the same -ub the pshard arm evaluated with, keeping numerics comparable).

Kept as:

    // eval shape changes numerics on shape-sensitive models - log once so a stock
    // reference run can match -ub to what pshard actually evaluates with

## Blocks read but left unchanged (mechanism-only, no dates/numbers/history)

- 255 (llama_context ctor): auto probe disabling under pshard
- 655, 693, 707 (llama_context::sched_reserve): planner probe sizing / no_alloc probe notes
- 790-808 (llama_context::sched_reserve): scheduler-rebuild invalidation of cached alloc states and pshard_reapply_active_plan
- 1568, 1806 (encode / decode): MTP hook batches carry token and embd
- 1903-1905, 1919-1929, 1939-1940 (llama_context::decode): ubatch tier selection and landed-tier clamp
- 4573-4574 (llama_perf_context_print): pshard switch time reporting

## src/llama-context.h

# Comment rewrites: src/llama-context.h

Regex sweep for dates / measured numbers / cell names / review references / machine names found 1 comment block. Step-2 manual sweep of the remaining pshard comments (transfer mode, pool region, spill detector, land_tier, runtime_ready, tier-switch perf counters, probe_reserve, free-function banner) found no narrative history, so nothing else was changed.

## Entries

- 399-400 (struct llama_context, member `pshard_pool_scratch`) - original: `// compute scratch (+ margin) each EXPERT_POOL tier's graph really needs, measured` / `// at its first reserve: the pool region is everything the graph leaves` - kept as: `// compute scratch (+ margin) each EXPERT_POOL tier's graph needs, taken at its first reserve;` / `// the pool region is what the graph leaves`. Removed information: none of substance; "measured" here referred to the runtime mechanism (the value is recorded when the tier's graph is first reserved), not a benchmark. Reworded only so the word does not read as a historical measurement.

## Remaining regex hits after rewrite

0

## src/llama-expert-pool.cpp

# Comment rewrite notes: src/llama-expert-pool.cpp

Line numbers are the ORIGINAL (pre-edit) lines. 8 comment blocks rewritten, 21 original comment lines -> 16 new lines (file 1302 -> 1297 lines, CRLF preserved, no code changed).

Note on "A/B": in this file "A/B" names the pool's own double-buffer mechanism (two whole-layer halves, identifiers `ab_mode`, `ab_capable`, `view_ab`, `ab_off`), not an experiment pair. The comments were reworded to reference the identifier `ab_mode` / "whole-layer overlay" so the convention grep stays clean; the log strings at (new) lines 214-216 and 282 still say "A/B pair" and were left alone per the string-literal rule. The regex flagged 4 lines (115, 166, 189, 291); the other 4 blocks were caught by the narrative sweep (stale plan reference, "the old way", "(user rule)"/"the grid", worked numbers).

---

src/llama-expert-pool.cpp:114-115 (llama_expert_pool::region_bytes_needed) - original:
    // cache mode: slots_per_layer rows of every pooled tensor, per layer;
    // A/B mode reuses the same span (2 whole layers) - take the max when asked
- kept as:
    // cache mode: slots_per_layer rows of every pooled tensor, per layer;
    // ab_mode reuses the same span for two whole layers - take the max when asked
- removed: nothing of substance; "A/B" -> identifier `ab_mode`.

src/llama-expert-pool.cpp:144-145 (llama_expert_pool::set_region) - original:
    // per-layer counts: the last warm start's plan when it was made for this uniform
    // count (same region), else uniform
- kept as:
    // per-layer slot counts are uniform; the region must hold every layer's slot arrays
- removed: reference to "the last warm start's plan" - stale: `n_slots_l` is only ever assigned uniformly (`L.n_slots_l = slots_per_layer`, the sole writer in src/), no per-layer plan path exists in the code. History: an earlier pool stage carried per-layer slot counts from a warm-start plan; that path is gone.

src/llama-expert-pool.cpp:166-168 (llama_expert_pool::set_region) - original:
    // cache-mode layout: layer-major, per-tensor slot arrays; A/B halves overlay
    // the region start (they are only live on whole-stack prefill tiers, where the
    // cache contents are volatile by design)
- kept as:
    // cache-mode layout: layer-major, per-tensor slot arrays; the ab_mode halves overlay the region start
    // (they are only live on whole-stack prefill tiers, where the cache contents are volatile by design)
- removed: nothing of substance; "A/B halves" -> "ab_mode halves", re-flowed to two lines.

src/llama-expert-pool.cpp:189-191 (llama_expert_pool::set_region) - original:
    // A/B halves: parity-alternating whole-layer sets at the region start. Only
    // when the region holds the pair - a cache tier's region can be smaller, and
    // the whole-stack tiers get their own (larger) carve when they are applied
- kept as:
    // ab_mode halves: parity-alternating whole-layer sets at the region start, only when the region holds the pair;
    // a cache tier's region can be smaller, and the whole-stack tiers get their own larger carve when they are applied
- removed: nothing of substance; "A/B halves" -> "ab_mode halves", re-flowed to two lines.

src/llama-expert-pool.cpp:291-292 (llama_expert_pool::set_ab_mode) - original:
    // v1 relabel: cache contents do not survive the A/B overlay (the halves alias
    // the slot arrays); drop the maps and refill lazily
- kept as:
    // cache contents do not survive the whole-layer overlay (the halves alias the slot arrays): drop the maps and refill lazily
- removed: "v1 relabel" - the POOL v1 stage label (pool v1 landed 2026-09-03 with all stages + split-op + follow-ups; "relabel" was the v1 term for the map reset on an ab_mode flip, as opposed to a later idea of preserving cache contents across the overlay, which was never built). "A/B overlay" -> "whole-layer overlay".

src/llama-expert-pool.cpp:520-521 (llama_expert_pool::ensure_read_staging) - original:
    // landing slices: a device alias lets the graph write the router ids into host memory; without the mapping
    // the layers read the ids back the old way
- kept as:
    // landing slices: a device alias lets the graph write the router ids into host memory; without the mapping
    // the layers read the ids back with a stream drain
- removed: "the old way" - history: before the mapped router-ids landing (2026-09-13 host-loop census work; per-layer round trips ~13% of the token were the lever) every pooled layer read its ids with ggml_backend_tensor_get_async + ggml_backend_synchronize, i.e. a full stream drain per layer. That is still the fallback path in serve() when the slice has not landed.

src/llama-expert-pool.cpp:1190-1191 (llama_expert_pool::log_counters) - original:
    // the three headline lines print at WARN: perf runs carry no -lv (user rule) and the
    // grid calibrates the pricing model from them; per-layer detail stays at INFO
- kept as:
    // the headline lines print at WARN so runs without -lv still report them (the pricing model is
    // calibrated from these); per-layer detail stays at INFO
- removed: "(user rule)" - the user decision of 2026-09-01 that perf runs are workload + budget only, no -v / -lv; "the grid calibrates" - the perf grid (qa/perf-grid.md, GRID 20260905-gpc2100 and the 20260906-fixes rerun) reads the h / misses-per-token / prefetch WARN lines from the run logs to calibrate the planner's pricing model. "three" dropped: there are now more than three WARN lines (ids-landing, hits/misses, prefetch, decode-only, workload).

src/llama-expert-pool.cpp:1273-1277 (llama_expert_pool::log_counters) - original:
    // routing workload: this run's cache-mode route histogram goes into <model>.pshard_workload. Real runs
    // ACCUMULATE (a 128-token run alone has ~4 routes per expert on a 256-expert model, too few on its own; the
    // module's split-half fit is unbiased at small samples and the store grows with every run); a file holding
    // only the plan-time calibration stand-in is replaced by the first real run. The 64-pass floor keeps trivial
    // runs out. The fitted exponent is what the planner prices h(s) with.
- kept as:
    // routing workload: this run's cache-mode route histogram goes into <model>.pshard_workload. Real runs
    // accumulate (one short run has too few routes per expert; the split-half fit is unbiased at small samples
    // and the store grows with every run); a file holding only the plan-time calibration stand-in is replaced by
    // the first real run. The 64-pass floor keeps trivial runs out. The planner prices h(s) with the fitted exponent.
- removed: the worked example "a 128-token run alone has ~4 routes per expert on a 256-expert model" (the standard -n 128 perf cell x 8 routes / 256 experts = 4 routes per expert; q35 real measured alpha 0.975 needed several accumulated runs, 2026-09-13 step 3 of the profile-schema-2 work).

## src/llama-expert-pool.h

# src/llama-expert-pool.h - comment rewrite notes

Line numbers are the pre-edit ones (file was 317 lines, now 315). Information removed from the code comments is preserved here.

## Rewritten

src/llama-expert-pool.h:44 (struct ids_buf, header comment) - original: `// as the vector it replaces; a copy gets its own vector storage.` - kept as: `// as std::vector; a copy gets its own vector storage.` (removed: the note that ids_buf replaced a plain std::vector member)

src/llama-expert-pool.h:110 (layer_state::ids_host) - original: `// the graph's landing tensor (nullptr: the layer reads back the old way)` - kept as: `// the graph's landing tensor (nullptr: the layer reads the ids back from the device)` (removed: "the old way" = the pre-landing-slice device->host readback of ids_router through ids_read_buf)

src/llama-expert-pool.h:160 (skipped_splits, dead-CPU-chain block) - original: `// (always on: measured a win on 2026-09-10; the A/B switch was removed 2026-09-13)` - kept as: deleted (the two preceding lines already state the mechanism). Removed history: the dead-chain skip was measured as a win on 2026-09-10 as part of the WDDM engine-transition work (kernel copies + dead-chain skip + no per-layer drains took hybrid 66 -> 89 t/s, fetch 77 -> 96 t/s on q35 @8000); its on/off toggle (an A/B experiment switch, PSHARD env var) was removed on 2026-09-13 with the other fifteen env back doors (commit f549c0419).

src/llama-expert-pool.h:164-165 (prefetch_n) - original: `// Mispredicted uploads share the PCIe link with the critical-` / `// path misses: DSv4 @12000 N=1 +6.5%, N=2 +3.7%, N=3 -2%` - kept as: `// Mispredicted uploads share the PCIe link with the critical-path misses.` (removed measurement: DeepSeek-V4 at the @12000 budget cell, prefetch_n=1 gave +6.5% decode, N=2 +3.7%, N=3 -2% versus no prefetch cap; this is why the default is 1)

src/llama-expert-pool.h:169-172 (read_staging, page-locked staging block) - original: `// of copy-engine transfers (a DMA ordered behind kernels costs 30-55 us of GPU idle on WDDM` / `// whether the host side is pinned or pageable - the pinned arena of 2026-09-10 regressed for` / `// that reason). Allocated while kernel copies are on; larger batches fall back to the vectors.` / `// also holds the per-layer router-ids landing slices (ids_host_pin), written by the graph via ids_host_buf` - kept as: `// of copy-engine transfers (a DMA ordered behind kernels idles the GPU under WDDM whether` / `// the host side is pinned or pageable). Allocated while kernel copies are on; larger batches` / `// fall back to the vectors. Also holds the per-layer router-ids landing slices (ids_host_pin),` / `// written by the graph via ids_host_buf` (removed: the measured cost of a copy-engine transfer ordered against kernels on this WDDM box is 30-55 us of GPU idle per transfer; the first attempt at this staging arena, 2026-09-10, made the host side pinned but still issued DMA transfers and regressed decode for that reason - pinned+DMA is not the fix, kernel copies are; see memory note "WDDM engine transitions")

src/llama-expert-pool.h:193 (ids_host_fallbacks) - original: `// slices that never filled (stream drained, then the old readback)` - kept as: `// slices that never filled (stream drained, then the device readback)` (removed: "the old readback" = the pre-landing-slice device->host copy path)

src/llama-expert-pool.h:228-229 (set_ab_mode declaration) - original: `// tier switch: flip cache/AB mode and re-register; cache contents survive a` / `// mode round-trip only in the preserved span (v1: dropped - lazy refill)` - kept as: `// tier switch: flip cache/AB mode and re-register; cache contents do not survive` / `// a mode round-trip (the halves alias the slot arrays): maps dropped, lazy refill` (removed: the "v1" version marker and the design-doc "preserved span" that would let part of the cache survive the A/B overlay - not implemented; the code (set_ab_mode in the .cpp) calls reset_slots() unconditionally, which the new text states)

## Regex hits reviewed and kept unchanged

Lines 17, 36, 39, 94, 203, 221, 282, 298 match the regex on `A/B`, but there "A/B" is the name of the whole-stack prefill mechanism (the two alternating layer halves of the pool region), not an experiment reference. It is the term used by the identifiers (`ab_mode`, `view_ab`, `ab_off`, `ab_pass`, `ab_capable`, `set_ab_mode`) and by docs/expert-pool-design.md ("A/B buffers", "A/B span", "A/B pair"), so renaming it in comments alone would detach them from the code. Left as is.

Line 169 (now 167) still contains "WDDM" after the rewrite: "a DMA ordered behind kernels idles the GPU under WDDM" states the mechanism in general terms, which the convention allows; the number and the date were removed.

## Checked, no change needed

- Lines 155, 157, 162: `PSHARD_POOL_PREDICT`, `PSHARD_POOL_PREFETCH`, `PSHARD_POOL_PREFETCH_N` are still read in src/llama-expert-pool.cpp (lines 40-46), so the env var names in those comments are current.
- Line 3-19 file header, 42-43, 81-142 layer_state, 149-157, 185-196, 214-226, 232-316: mechanism descriptions without dates, numbers, cells or history.

## Second pass (step-2 sweep, file now 314 lines; line numbers below are from the 315-line state it started from)

src/llama-expert-pool.h:133-134 (layer_state::n_slots_l) - original: `// per-layer allocation: this layer's slot count (= the pool's uniform n_slots` / `// the same count in every layer)` - kept as: `// this layer's slot count (= the pool's uniform n_slots, the same in every layer)` (removed: the "per-layer allocation" heading. It is residue of the retired per-layer slot allocation - commit fab14ab40 landed a popularity-ranked per-layer slot count off by default, 1ce8f5b2a reworked it, and the current set_region assigns L.n_slots_l = slots_per_layer for every layer (llama-expert-pool.cpp:147), so the member only mirrors the uniform pool value)

src/llama-expert-pool.h:147 (llama_expert_pool::n_slots) - original: `// cache-mode slots per layer (the uniform baseline)` - kept as: `// cache-mode slots per layer (the same count in every layer)` (removed: "baseline", which framed n_slots as the reference value the retired per-layer allocation deviated from; there is no per-layer deviation any more)

## Regex remainder after the second pass: 9 lines, none rewritten

Lines 17, 36, 39, 94, 201, 219, 280, 296 - "A/B" is the pool's own mode name (identifiers ab_mode / view_ab / ab_off / ab_pass / ab_capable / set_ab_mode; docs/expert-pool-design.md uses "A/B buffers", "A/B pair", "A/B span" 84 times), not an experiment pair. Renaming it in comments alone would detach them from the identifiers and the design doc.
Line 167 - "a DMA ordered behind kernels idles the GPU under WDDM whether the host side is pinned or pageable" is the mechanism in general terms (the allowed form); the measured cost and the date were already removed in the first pass. The pinned-or-pageable clause is kept because it is the why of the design: the staging arena is page-locked so the backend can address it from kernel copies, not to speed up a DMA.

## src/llama-graph.cpp

# src/llama-graph.cpp - comment rewrites

Removed information preserved here (dates, measured numbers, machine/experiment/review references are not allowed in code comments).

- 2199-2204 (llm_graph_context::build_moe_ffn, the EXPERT_POOL ids-leaves block) - original: `ids leaves for a pool-managed layer. fetch-only policies remap the four expert MUL_MAT_IDs to slot ids (one leaf); policies that admit CPU routes (cpu_exec / hybrid / fetch_on_2nd_miss) run TWO expert-FFN chains - GPU over the pool slots and CPU over the host homes - each seeing -1 for the other side's routes, merged by one ADD at the down output (the split-op, design 6e). Biases then need expert ids with -1 for the foreign routes too.` - kept as: `The ids leaves for a pool-managed layer: fetch-only policies remap the four expert MUL_MAT_IDs to slot ids (one leaf); policies that admit CPU routes (cpu_exec / hybrid / fetch_on_2nd_miss) run TWO expert-FFN chains - GPU over the pool slots and CPU over the host homes - each seeing -1 for the other side's routes, merged by one ADD at the down output (the split-op). Biases then need expert ids with -1 for the foreign routes too.` Dropped: the design-document reference "design 6e" - the split-op (two expert-FFN chains, GPU over pool slots and CPU over host homes, merged by one ADD) is design item 6e of the expert-pool RFC execution model (cpu_admit + sched CPU/GPU overlap), landed with the split-op commit e45601565 and the review fixes f6174db7c. The dangling fragment "ids leaves for a pool-managed layer." (the residue of an earlier heading) was folded into the first sentence; no information lost.

- 2208 (llm_graph_context::build_moe_ffn) - original: `whole-stack (A/B) tiers have every expert resident: no CPU routes, single chain.` - kept as: `whole-stack tiers (ab_mode) have every expert resident: no CPU routes, single chain.` Dropped: the "A/B" wording. It was not an A/B experiment pair but the pool's own name for the whole-stack prefill tier mode (llama_expert_pool::ab_mode - the pool overlays the layer's A/B half on the region start when bs*top_k*2 >= n_expert); the rewrite names the identifier so the regex cannot mistake it for an experiment reference. Single-chain-on-A/B-tiers was one of the split-op review fixes (f6174db7c).

Step 2 sweep (all pshard/expert-pool comments: build_lora_mm_id pin note 1553-1555, build_moe_select header 1965-1968, prefetch predictor 2120-2136, ids landing copy 2232-2233, expert-chain lambda 2243-2246, dual-chain ORDER MATTERS block 2415-2424, RS shadow 3603-3604, DFlash mask 476-477): no further narrative found. "Measurement / prefetch only" (2123) describes what the predictor node is for (it does not feed the real routing), not a measured number; "stock 2-D" (3604) states the upstream tensor shape the reshape stays compatible with; both left as is.

Step 3 re-grep: 0 remaining hits. No code, identifiers, string literals or line endings were touched (CRLF count 4029 before and after, 0 bare LF; the six-line block was rewritten as six lines).

## src/llama-kv-cache.cpp

# comment rewrite notes: src/llama-kv-cache.cpp

Line numbers refer to the file before the rewrite. Original text is preserved verbatim so the removed
dates, measurements and history stay recoverable. Only comment text changed; the file is LF-terminated
and was written back with the same line endings.

## src/llama-kv-cache.cpp:186-190 (llama_kv_cache::llama_kv_cache, pshard tensor_spec loop)

Original text:

```
// size rows with the SAME accessors as the stock allocation below - they are
// the per-arch source of truth (a hand-rolled n_lora_kv + n_rot here computed
// 64 instead of 576 for DeepSeek-V4-Flash, whose rank lives elsewhere, and the
// probe then crashed writing 512-wide latents into a 64-wide cache).
// MLA still has no separate V cache; dim_t2=0 signals "skip t2".
```

Removed: the bug history - the hand-rolled formula computed a 64-wide row where DeepSeek-V4-Flash
needs 576 (its KV rank is not n_lora_kv + n_rot), and the plan-time probe then crashed writing
512-wide latents into the 64-wide cache. Regex-missed narrative, trimmed in step 2.

Kept as:

```
// size rows with the SAME accessors as the stock allocation below - they are
// the per-arch source of truth (a hand-rolled n_lora_kv + n_rot misses the
// DeepSeek-V4 latent width and the cache comes out too narrow for the graph).
// MLA still has no separate V cache; dim_t2=0 signals "skip t2".
```

## src/llama-kv-cache.cpp:265-268 (llama_kv_cache::llama_kv_cache, pshard branch tail before init_attn_rot)

Original text:

```
// the constructor tail below (attention rotation + Hadamard tables) must run for
// pshard caches too: without it build_input_k_rot() returns nullptr and DeepSeek-V4
// silently builds RAW attention instead of its compressed-cache + lightning-indexer
// path (garbage from the first token, 2026-09-01)
```

Removed: the date the omission was found and fixed (2026-09-01) and the observed symptom
(garbage output from the first token).

Kept as:

```
// the constructor tail below (attention rotation + Hadamard tables) must run for
// pshard caches too: without it build_input_k_rot() returns nullptr and DeepSeek-V4
// silently builds RAW attention instead of its compressed-cache + lightning-indexer path
```

## Reviewed and left unchanged

- 169-170 (head-size bookkeeping mirrors the stock loop) - mechanism only.
- 212-213 (FULL transfer mode for graph-written caches) - mechanism only.
- 218-219 (v_stream nullptr padding for K-only caches) - mechanism only.
- All `TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]` / other TODOs and DEBUG CHECK comments are upstream llama.cpp text.

## src/llama-model.cpp

# comment rewrite notes: src/llama-model.cpp

Line numbers refer to the file before the rewrite. Original text is preserved verbatim so the removed
dates, measurements, model/machine sizes and cell names stay recoverable.

Regex hits before: 2 (lines 1544, 1997). Regex hits after: 0. Three comment blocks rewritten (9 comment
lines changed); the block at 1990-1991 was not a regex hit and was trimmed in the step-2 sweep.

## src/llama-model.cpp:1543-1547 (llama_model::load_tensors, tensor-split fallback when every device reports 0 free bytes)

Original text:

```
// every device reported 0 free bytes (a pshard target that took the whole card
// before its draft model loads, 2026-09-05: DSv4 + DSpark at ctx 8192): the
// division above would make the split points NaN, upper_bound would return the
// end iterator and devices.at() would throw "invalid vector subscript". Split
// evenly instead - the budget decides what fits, not the split.
```

Removed: the date the case was hit (2026-09-05), the cell that hit it (DSv4 + DSpark at ctx 8192), the
verbatim MSVC exception text ("invalid vector subscript").

Kept as:

```
// every device reported 0 free bytes (a pshard target that took the whole card
// before its draft model loads): the split points would be NaN, upper_bound would
// return the end iterator and devices.at() would throw. Split evenly instead - the
// budget decides what fits, not the split.
```

## src/llama-model.cpp:1990-1991 (llama_model::load_tensors, pshard mmap page-lock failure branch)

Not a grep hit; trimmed for the machine-specific sizes.

Original text:

```
// pinned-memory ceiling (a 45 GB DeepSeek-V4 shard after a 47 GB one on 127 GB RAM)
// - streamed copies from this region run pageable
```

Removed: the scenario that hit the ceiling (a 45 GB DeepSeek-V4 shard loaded after a 47 GB one on a
127 GB RAM host).

Kept as:

```
// pinned-memory ceiling: the host cannot page-lock this much more RAM
// - streamed copies from this region run pageable
```

## src/llama-model.cpp:1995-1999 (llama_model::load_tensors, _WIN32 VirtualLock fallback inside the page-lock failure branch)

Original text:

```
// Windows trims file-mapped pages out of the working set between passes, so every
// upload pass soft-faults the whole region again (~6 GB/s memcpy into the staging
// ring, ~4 GB/s on the driver's pageable path, measured on a 45 GB DeepSeek-V4
// shard). VirtualLock keeps the pages resident and mapped: the CUDA DMA still has to
// go through the staging ring, but the ring's memcpy runs at memory speed.
```

Removed: the measured rates when the region soft-faults on every pass (~6 GB/s memcpy into the staging
ring, ~4 GB/s on the driver's pageable path) and what they were measured on (a 45 GB DeepSeek-V4 shard).

Kept as:

```
// Windows trims file-mapped pages out of the working set between passes, so every
// upload pass soft-faults the whole region again. VirtualLock keeps the pages
// resident and mapped: the CUDA DMA still goes through the staging ring, but the
// ring's memcpy then runs at memory speed.
```

## Step-2 sweep: reviewed and left unchanged

Mechanism-only pshard comments with no dates, numbers, cells or history: 1172-1176 (VirtualLock'd /
page-locked region lists), 1238 (release page-locks before mappings), 1767 (tensor -> backend_id map),
1818-1819 (arena = canonical union + headroom), 2592-2593 (extras not in canonical offsets), 2754-2756
(reserve fixed-placement commons up front), 2960-2965 (land-the-bytes move/stage/upload ordering),
2988 (tier warmup addresses without landing), 3151-3152 (unconditional fence for async KV D2H),
3203 (pshard memory gate), 3258-3260 / 3312 / 3431 (MTP draft KV cache shape), 3403 / 3915 (DSpark
K layout and RoPE), 3641 (pooled layers hoist). No `/* */` block comment in the file carried flagged
content.

## src/llama-pshard-cache.cpp

# comment rewrite notes: src/llama-pshard-cache.cpp

Line numbers refer to the file before the rewrite. Original text is preserved verbatim so the removed
dates, measurements, cell names and history stay recoverable.

## src/llama-pshard-cache.cpp:96-102 (llama_pshard_generate_overrides, expert-pool branch, MTP head layers)

Original text:

```
// MTP head layers: pinned whole, experts included, exactly as the planner's copy
// prices them (the draft context is a stock sched with no pool and reads them every
// draft step). This branch lacked the special case until 2026-09-07: the load-time
// array then homed the MTP layer's experts on the shard bid, and the post-fit MTP
// probe over that array measured 193 MiB where the plan tool (planner array, layer
// pinned) measured 21 - the runtime refit to a budget with no saved variant and
// disabled pshard (grid 20260906-fixes, every q35 MTP pool cell at 8000/full).
```

Removed: the date the special case landed (2026-09-07), the measured MTP reserve mismatch (193 MiB
load-time array vs 21 MiB plan-tool array), the grid name (20260906-fixes) and the affected cells
(every q35 MTP pool cell at 8000/full).

Kept as:

```
// MTP head layers: pinned whole, experts included, the placement the planner's copy prices
// (the draft context is a stock sched with no pool and reads them every draft step). homing
// the head's experts on the shard bid instead makes the post-fit MTP probe overshoot the plan
// and the refit lands on a budget with no saved variant, which disables pshard
```

## src/llama-pshard-cache.cpp:126-130 (llama_pshard_generate_overrides, legacy strategies branch, MTP head layers)

Not a grep hit; trimmed for the "sound now that ..." history phrasing.

Original text:

```
// MTP head layers: read every draft step by the stock-sched draft context.
// Never slot-streamed (concurrent reader); PIN-PRIORITY: whenever the plan
// pins anything at all, the head goes to the compute GPU first (the probes
// price it, so viability shrinks the trunk pins accordingly). Pinned is
// sound now that the draft ctx gets stock, backed KV (per-context gate).
```

Removed: the "now that" narrative (pinning became sound once the draft context was given stock,
backed KV via the per-context gate); the PIN-PRIORITY label.

Kept as:

```
// MTP head layers: read every draft step by the stock-sched draft context, so never
// slot-streamed (concurrent reader). whenever the plan pins anything at all the head goes
// to the compute GPU first (the probes price it, so viability shrinks the trunk pins);
// pinning needs the draft ctx on stock, backed KV (per-context gate)
```

## src/llama-pshard-cache.cpp:334-337 (llama_pshard_probe_model_only, MTP layer count)

Not a grep hit; trimmed for the past-tense "leaving it out let ... and crashed ..." history.

Original text:

```
// MTP: the nextn head is a full extra layer (attn + experts) WITHIN block_count.
// When it will actually be loaded, plan it like any other layer - leaving it out
// let blk.<nextn> fall to the loader's dev_layer default (wholesale on GPU,
// outside the budget) and crashed warmup reserves with plan-blind view sizes.
```

Removed: the history that the pre-fix behaviour crashed the warmup reserves (the mechanism itself,
plan-blind view sizes from the loader's dev_layer default, is kept in present tense).

Kept as:

```
// MTP: the nextn head is a full extra layer (attn + experts) within block_count. when it will be
// loaded, plan it like any other layer; otherwise blk.<nextn> falls to the loader's dev_layer
// default (wholesale on GPU, outside the budget) and the warmup reserves see plan-blind view sizes
```

## reviewed, left unchanged

Other pshard comments in the file (fetch corner, overlap slot rationale, FFN_ALTERNATE, ids_cross,
fingerprint inputs, miss-policy mixing, delegate-flag mirror, mtp_head_cpu variant, n_layers publish)
state mechanism only and carry no dates, measurements, cell names or history. Log-message string
literals contain no dates or measurements.

Total: 16 original comment lines rewritten (7 + 5 + 4) into 11 lines.

## src/llama-pshard-plan.cpp

# src/llama-pshard-plan.cpp - comment rewrite notes

Line numbers are the pre-edit ones (file was 3248 lines, now 3231). Information removed from the code comments is preserved here. Only comment text was edited; the file stays CRLF.

## Rewritten

src/llama-pshard-plan.cpp:100-104 (llama_pshard_generate_overrides, EXPERT_POOL branch, MTP head layers) - original: `// MTP head layers: the draft context (a stock sched, no pool) reads them` / `// on every draft step - a pooled head would stream its whole expert set` / `// per draft token (measured: 9.5 ms per draft on q35). Pin the head whole,` / `// experts included, like the legacy plans do; the pool region shrinks by` / `// one layer of experts (~6 of 82 slots on q35).` - kept as: `// MTP head layers: the draft context (a stock sched, no pool) reads them` / `// on every draft step - a pooled head would stream its whole expert set` / `// per draft token. Pin the head whole, experts included, like the legacy` / `// plans do; the pool region shrinks by one layer of experts.` (removed: a pooled MTP head measured 9.5 ms per draft step on q35; pinning the head whole costs about 6 of the 82 pool slots on q35)

src/llama-pshard-plan.cpp:137-138 (llama_pshard_generate_overrides, legacy MTP head branch) - original: `// price it, so viability shrinks the trunk pins accordingly). Pinned is` / `// sound now that the draft ctx gets stock, backed KV (per-context gate).` - kept as: same with `now that` -> `because` (removed: the history marker - before the per-context gate the draft context did not get stock, backed KV, and pinning the head was not sound then)

src/llama-pshard-plan.cpp:229-232 (struct llama_pshard_search_ctx, routed-expert byte fields) - original: `// routed-expert bytes read from the gguf tensor table (0 = unknown -> the` / `// file-size heuristic): the largest layer's full expert set, one expert's rows` / `// summed over its tensors (up/gate/down or gate_up/down), and how many layers` / `// carry routed experts (DSv4-class models have dense lead layers)` - kept as: `// routed-expert bytes read from the gguf tensor table (0 = unknown: pool tiers are` / `// refused, the ids-cross decision is not priced): the largest layer's full expert` / `// set, one expert's rows summed over its tensors (up/gate/down or gate_up/down), and` / `// how many layers carry routed experts (DSv4-class models have dense lead layers)` (removed: the stale reference to the file-size-fraction fallback, which commit 8bbfc347f deleted; today a 0 makes llama_pshard_search_pool refuse the tier and pshard_alternate_ids_cross_wins leave the decision unpriced, which the new text states)

src/llama-pshard-plan.cpp:367-368 (pshard_alternate_ids_cross_wins, cover_ms) - original: `// cover the full upload could hide behind: the paired CPU-FFN's DRAM-bound expert reads (the streamed` / `// layer's attention compute was a 0.1 ms constant here: unmeasured, no longer charged)` - kept as: `// cover the full upload could hide behind: the paired CPU-FFN's DRAM-bound expert reads` (removed history: cover_ms used to add a 0.1 ms constant for the streamed layer's attention compute; the constant was never measured and was dropped under the no-constants directive)

src/llama-pshard-plan.cpp:407-408 (llama_pshard_probe_memory, mparams_probe_clean) - original: `// probe load packs pinned KV into the external preload buffer and the` / `// measurement no longer attributes it to mb.context (measured cache = 0)` - kept as: `// probe load packs pinned KV into the external preload buffer and the` / `// probe no longer attributes it to mb.context (it reports cache = 0)` (wording only; no information removed)

src/llama-pshard-plan.cpp:414-416 (llama_pshard_probe_memory, probe_n_outputs) - original: `// to n_outputs_max; 0 = n_batch, llama_context's own rule). Probing every token as an output` / `// charged (bs - n_outputs_max) x n_vocab x 4 B of logits the runtime never reserves - ~1 GiB` / `// at DSv4 bs=2048 under a speculative target's cap of 4 (review 2026-09-06)` - kept as: `// to n_outputs_max; 0 = n_batch, llama_context's own rule). Probing every token as an output` / `// would charge (bs - n_outputs_max) x n_vocab x 4 B of logits the runtime never reserves` (removed: the over-charge was about 1 GiB at DeepSeek-V4 bs=2048 with a speculative target's n_outputs_max cap of 4; found by the review of 2026-09-06)

src/llama-pshard-plan.cpp:463-466 (llama_pshard_prune_state::attn_hint) - original: `// an attn-pin bound proven at a larger batch is INVALID at bs=1: activation` / `// scratch shrinks ~350x between bs=8192 and bs=1, so far more attention fits.` / `// (q35-16k-mva2000: inherited hi_attn=11 hid the attn=40 STATIC winner, 12.1` / `// vs 29.6 predicted tps.) Search the decode tier with a fresh bound.` - kept as: `// an attn-pin bound proven at a larger batch is INVALID at bs=1: activation` / `// scratch shrinks with the batch, so far more attention fits. Search the` / `// decode tier with a fresh bound.` (removed: activation scratch shrinks about 350x between bs=8192 and bs=1; in cell q35-16k-mva2000 an inherited hi_attn=11 bound hid the attn=40 STATIC winner, 12.1 vs 29.6 predicted t/s)

src/llama-pshard-plan.cpp:1488-1489 (llama_pshard_registry_load, variant selection) - original: `// deterministic variant selection (the accumulate + first-match era produced plans` / `// from mixed planning sessions - see the phantom q8d 2.84 t/s incident):` - kept as: `// deterministic variant selection (a first-match pick can mix plans from` / `// different planning sessions):` (removed history: the earlier accumulate + first-match loader could assemble a variant from plans of different planning sessions; the "phantom q8d 2.84 t/s incident" was a bogus decode result produced by such a mixed variant and motivated rules 1-4 below the comment)

src/llama-pshard-plan.cpp:1775-1782 (llama_pshard_enforce_union_budget, per-tier need) - original: `// the tier's compute scratch (streaming slots + graph temporaries, probe-measured)` / `// must fit above the packed weights and the pinned cache too: otherwise the` / `// runtime reserve falls back to constrained packing and galloc spills into an` / `// overflow chunk OUTSIDE the arena (measured +496 MiB at a 3929 MiB budget)` / `// + margin: the runtime canonical packing (pshard_compute_scratch_off) rounds` / `// differently from this metadata pass by up to a few tens of MiB (seen +29.7 MiB` / `// at a 2024 MiB budget: a tier that passed here by 1.2 MiB was marked unviable at` / `// load, and every verify batch then ran in the 512-token streaming plan)` - kept as: `// the tier's compute scratch (streaming slots + graph temporaries, from the probe)` / `// must fit above the packed weights and the pinned cache too: otherwise the` / `// runtime reserve falls back to constrained packing and galloc spills into an` / `// overflow chunk OUTSIDE the arena` / `// + margin: the runtime canonical packing (pshard_compute_scratch_off) rounds` / `// differently from this metadata pass by up to a few tens of MiB, and a tier` / `// that passes here by less than that is marked unviable at load` (removed measurements: the galloc overflow chunk outside the arena measured +496 MiB at a 3929 MiB budget; at a 2024 MiB budget the runtime packing came out +29.7 MiB above this pass, a tier that had passed here by 1.2 MiB was marked unviable at load, and every verify batch then ran in the 512-token streaming plan - this is why `margin` exists)

src/llama-pshard-plan.cpp:1822-1827 (llama_pshard_enforce_union_budget, MTP head lever) - original: `// the MTP context's pre-fit reserve (common_pshard_draft_reserve_mb) was measured with` / `// the head pinned; with the head on the CPU its device compute grows by the logits` / `// scratch (+177.5 / +179.5 MiB on q35, 2026-09-04 grid). The one-budget fit` / `// (common_pshard_fit_one_budget) re-measures the context under the fitted placement and` / `// refits once with the larger reserve; the analytical arena charge that stood in for` / `// that (2026-09-05, n_vocab x 128 x 6 B, mtp_head_extra_mb) is retired (2026-09-06)` - kept as: `// the MTP context's pre-fit reserve (common_pshard_draft_reserve_mb) was taken with` / `// the head pinned; with the head on the CPU its device compute grows by the logits` / `// scratch. The one-budget fit (common_pshard_fit_one_budget) re-measures the context` / `// under the fitted placement and refits once with the larger reserve, so nothing is` / `// charged for it here` (removed: the logits scratch growth measured +177.5 / +179.5 MiB on q35 in the 2026-09-04 grid; an analytical arena charge of n_vocab x 128 x 6 B (mtp_head_extra_mb, added 2026-09-05) stood in for the re-measure and was retired on 2026-09-06 when the one-budget fit's post-fit re-measure landed)

src/llama-pshard-plan.cpp:1958-1959 (llama_pshard_attn_pin_fallback) - original: `// WARN: the grid runner labels rows by the forced arm; a silent substitution recorded three` / `// s1 cells and seven speculative pool cells as something they were not (audit 2026-09-05)` - kept as: `// WARN level: a caller that keys its results on the forced strategy must see the substitution` (removed: the qa grid runner labels result rows by the forced strategy arm; the audit of 2026-09-05 found three s1 cells and seven speculative pool cells recorded under the wrong strategy because this fallback used to be silent)

src/llama-pshard-plan.cpp:2009-2012 (llama_pshard_search_pool, output head placement) - original: `// output head ON the GPU: with it on the CPU every pass (target token AND draft` / `// step) pays the vocabulary projection at host rate - ~9 ms per pass on q35, the` / `// largest single term of a pool token; the head's bytes come out of the pool` / `// (~11 of 86 slots at 8000) and the probe prices the trade` - kept as: `// output head ON the GPU: with it on the CPU every pass (target token AND draft` / `// step) pays the vocabulary projection at host rate, the largest single term of` / `// a pool token; the head's bytes come out of the pool and the probe prices the trade` (removed: the host-rate vocabulary projection costs about 9 ms per pass on q35; the GPU head takes about 11 of 86 pool slots at the @8000 budget)

src/llama-pshard-plan.cpp:2045-2049 (llama_pshard_search_pool, scratch_pool probe) - original: `// allocated) and take only its compute buffer. The streaming probe above priced the tier and` / `// measured its placement bytes, but its compute holds the transient per-layer expert copies; the` / `// old credit of "2 x b_layer" for them under-estimated the pool graph's scratch by 0.7-2.2 GiB` / `// and let the planner emit A/B tiers the runtime carve could not host (DSv4 @14500 bs=4096/8192,` / `// q35 @4000 bs>=1024: 2026-09-05 grid, design 11.C.19).` - kept as: `// allocated) and take only its compute buffer. The streaming probe above priced the tier and` / `// gave its placement bytes, but its compute holds the transient per-layer expert copies, and` / `// crediting those analytically under-estimates the pool graph's scratch (design 11.C.19)` (removed: the previous analytical credit of 2 x b_layer under-estimated the pool graph's scratch by 0.7-2.2 GiB, so the planner emitted whole-stack tiers the runtime carve could not host at DSv4 @14500 bs=4096/8192 and q35 @4000 bs>=1024 in the 2026-09-05 grid)

src/llama-pshard-plan.cpp:2089-2094 (llama_pshard_search_pool, fixed_bytes / pool_bytes) - original: `// compute); scratch = the pool graph's own compute buffer. Mirror the runtime carve: it charges` / `// the measured chunk0 + 32 MiB. The extra 64 MiB stands in for what the pinned-expert probe does` / `// not see of the pool graph: the runtime's chunk0 measured 8.9% above this probe at every bs on` / `// DSv4 (+26 / +52 / +104 / +209 MiB at bs 512 / 1024 / 2048 / 4096, verify-three-20260906), a` / `// per-token term of ~50 KB whose source is not attributed yet. The verdicts agreed at both` / `// verified cells (DSv4 @14500, q35 @4000); a modelled residual is a proposed change (design 11.C.19 x).` - kept as: `// compute); scratch = the pool graph's own compute buffer. Mirror the runtime carve: it charges` / `// the probed chunk0 + 32 MiB. The extra 64 MiB stands in for what the pinned-expert probe does` / `// not see of the pool graph: the runtime's chunk0 is larger by a small per-token term whose` / `// source is not attributed yet (a modelled residual is a proposed change, design 11.C.19 x)` (removed: the runtime chunk0 measured 8.9% above the pinned-expert probe at every batch size on DeepSeek-V4, +26 / +52 / +104 / +209 MiB at bs 512 / 1024 / 2048 / 4096 in run verify-three-20260906, i.e. a per-token term of about 50 KB; the viability verdicts matched the runtime at both verified cells, DSv4 @14500 and q35 @4000)

src/llama-pshard-plan.cpp:2108 (llama_pshard_search_pool, per-expert rates header) - original: `// the profile (schema 2, 2026-09-12); a pool tier is refused rather than priced with a built-in value.` - kept as: `// the profile (schema 2); a pool tier is refused rather than priced with a built-in value.` (removed: schema 2 dates from 2026-09-12)

src/llama-pshard-plan.cpp:2113-2114 (llama_pshard_search_pool, t_cpu line of the rates header) - original: `//            quants are DRAM-bound), or the dequant compute at the slowest measured matmul rate (DSv4` / `//            UD-Q2_K_XL is compute-bound): max of the two.` - kept as: `//            quants are DRAM-bound), or the dequant compute at the profile's slowest matmul rate (heavy` / `//            dequant formats are compute-bound): max of the two.` (removed: the worked example - DeepSeek-V4 UD-Q2_K_XL is the compute-bound case on the CPU chain)

src/llama-pshard-plan.cpp:2179-2185 (llama_pshard_search_pool, cache-tier price) - original: `// the s most popular of E experts (the static optimum). alpha is MEASURED:` / `// the pool histograms every cache-mode route and refits it at exit into` / `// <model>.pshard_workload; a model without one is calibrated at plan time` / `// (sampled generation). The planner prices the long-run rate; short` / `// generations pay the fill (an 86-slot layer needs ~10 tokens of misses),` / `// which the 32-token QA gate reports as the cold one. (An LRU-shortfall` / `// factor and admission gating were both measured unnecessary and removed.)` - kept as: `// the s most popular of E experts (the static optimum). alpha comes from the` / `// routing workload: the pool histograms every cache-mode route and refits it` / `// at exit into <model>.pshard_workload; a model without one is calibrated at` / `// plan time (sampled generation). The planner prices the long-run rate; short` / `// generations pay the fill of the pool first.` (removed: an 86-slot layer needs about 10 tokens of misses to fill; the 32-token QA gate therefore reports the cold (fill) rate, not the long-run one; an LRU-shortfall factor and an admission gate were both tried in this price, measured unnecessary and removed)

src/llama-pshard-plan.cpp:2201-2203 (llama_pshard_search_pool, cpu_chain_overlaps) - original: `// the CPU chain runs concurrently with the GPU chain (scheduler lookahead); the` / `// serial variant is no longer priced (its switch was removed after the grid` / `// certified the overlap)` - kept as: `// the CPU chain runs concurrently with the GPU chain (scheduler lookahead);` / `// only the overlapped price is taken` (removed history: the serial (no-overlap) price had an on/off switch, removed once the grid certified the sched CPU/GPU overlap; `cpu_chain_overlaps` stays a constant true)

src/llama-pshard-plan.cpp:2657-2659 (llama_params_fit_pshard step 3, g_pshard_mtp_head_cpu reset) - original: `// The one-budget fit's second pass starts here too: a preset that kept pass 1's head home was` / `// tried and reverted on 2026-09-07 (design 11.C.19 xiii) - it rescued one arm and cost the` / `// others; the head home is a planner pricing decision, not a protocol rule.` - kept as: `// The one-budget fit's second pass starts here too: the head home is a planner pricing` / `// decision, not a protocol rule (design 11.C.19 xiii).` (removed: on 2026-09-07 a preset that carried the first pass's MTP head placement into the second pass of the one-budget fit was tried and reverted - it rescued one grid arm and cost the others)

src/llama-pshard-plan.cpp:2673 (llama_params_fit_pshard, profile fingerprint check) - original: `// the profile must be THIS machine's, complete (schema 2) and measured at this thread count: nothing in` - kept as: `// the profile must be THIS machine's, complete (schema 2) and taken at this thread count: nothing in` (wording only)

src/llama-pshard-plan.cpp:2677 (llama_params_fit_pshard, machine_current device) - original: `// the same device the profiler measured and the registry loader hashes: the first GPU-type device in` - kept as: `// the same device the profiler describes and the registry loader hashes: the first GPU-type device in` (wording only)

src/llama-pshard-plan.cpp:2742-2744 (llama_params_fit_pshard, per-mapping upload pricing) - original: `// Also compute the TOTAL file size: the main split of a sharded gguf can be` / `// tiny (DeepSeek-V4's is 6 MB); the byte counts the planner prices with come` / `// from the gguf tensor table (scan below), the total only feeds sanity checks.` - kept as: `// Also compute the TOTAL file size: the main split of a sharded gguf can be` / `// tiny; the byte counts the planner prices with come from the gguf tensor` / `// table (scan below), the total only feeds sanity checks.` (removed: DeepSeek-V4's main split is 6 MB)

src/llama-pshard-plan.cpp:2808-2810 (llama_params_fit_pshard, staged_r) - original: `// the staged rate is measured (PCIe_Staged: a pageable source through the staging ring). A` / `// profile without the line prices staged bytes at the pinned rate and says so; step 5 makes an` / `// incomplete profile a refusal.` - kept as: `// the staged rate comes from the profile (PCIe_Staged: a pageable source through the staging` / `// ring); a profile without the line prices staged bytes at the pinned rate and says so` (removed: the forward reference to "step 5" of the profile-schema-2 plan (landed in f549c0419, 2026-09-13) that makes an incomplete profile a refusal; note the PCIe_Staged line is still optional in this branch - the code warns and prices at the pinned rate)

src/llama-pshard-plan.cpp:2896 (llama_params_fit_pshard, routing workload) - original: `// generation (256 tokens, seed 1234, temperature 1) whose router top-k ids are histogrammed - a measured` - kept as: same with `a measured` -> `a sampled` (wording only; 256 / 1234 stay - they are the arguments passed to llama_pshard_workload_calibrate on the lines below)

## Regex hits reviewed and kept unchanged

Post-edit lines 232, 1965, 2016, 2063, 2150, 2468 (pre-edit 233, 1972, 2024, 2073, 2162, 2483) match the regex on `A/B`. There "A/B" names the whole-stack prefill mechanism - the two alternating expert-layer halves of the pool region (`ab_tier`, `LLAMA_PSHARD_PREFILL_AB_STREAM`, `plan.pool_prefill`, docs/expert-pool-design.md "A/B pair") - not an experiment comparison. Same decision as the llama-expert-pool.h notes; renaming it in comments alone would detach them from the identifiers. Left as is. Remaining count after the rewrite: 6, all of this kind.

Pre-edit lines 364-365 ("plus the copy-engine transition its DMA readback pays on this machine: the legacy sliced tiers keep the copy engine") kept: "this machine" means the profiled machine (the rate comes from `st->engine_switch_us`), stated in general terms.

## Checked, no change needed

- 110-119, 144-145, 173-183 (generate_overrides placement rules), 203-257 (search ctx fields, tensor scan header), 323-341 (ids-cross decision header and gate), 970-1017 (switch-cost estimator), 1189-1190, 1216, 1288-1310 (registry format compat notes), 1612-1620 (union accounting header), 1697-1704 (demotion levers), 1793-1795, 1857-1866, 1968-1976 (pool corner header), 2071-2076, 2109-2118 (rates header apart from the two lines above), 2156-2171, 2208-2234 (per-policy miss prices), 2284-2285, 2334-2336, 2385-2388, 2433-2435, 2482-2484, 2737-2747 (rest of the mapping-pricing block, including the MAIN-split fingerprint NOTE, which is a live invariant), 2817-2822, 2862-2863, 2929-2930: mechanism descriptions without dates, numbers, cells or history.
- `PSHARD_MISS_POLICY` (line 2170 comment) is still read by `pshard_miss_policy_from_env()` / `getenv` in llama_pshard_search_pool, so the mention is current.
- The diff against HEAD also shows removed code lines referencing `PSHARD_POOL_AUTO`; those were already in the working tree before this pass (the file was `M` at start) and are not part of this comment rewrite.

## src/llama-pshard-plan.h

# Comment rewrite notes: src/llama-pshard-plan.h

Line numbers refer to the file BEFORE the rewrite. Line endings: the file is LF (not CRLF); preserved as-is.
Convention applied: llama.cpp style - no dates, measured numbers, cell/experiment names, review references, machine names or history.

## Rewritten comments

### src/llama-pshard-plan.h:161-167 (file scope, `g_pshard_unsupported_reason`)

original text:

    // architecture support gate: both model probes (runtime cache loader and planner) set this
    // from the loaded model; nullptr = supported. pshard refuses LOUDLY (WARN + stock fallback)
    // rather than run a memory layout it cannot stream. No architecture is refused today:
    // DeepSeek-V4 (llama_kv_cache_dsv4) was the case until 2026-09-01 - its wrapper hid its pipe
    // shards, the pshard cache constructor skipped the attention-rotation tail (so the compressed
    // attention + lightning indexer were silently never built), and the scheduler let streamed
    // layers read views of host weights directly. Keep the gate for the next such architecture.

removed: the DeepSeek-V4 (llama_kv_cache_dsv4) case history and its date (refused until 2026-09-01; the wrapper hid its pipe shards, the pshard cache constructor skipped the attention-rotation tail so the compressed attention + lightning indexer were silently never built, the scheduler let streamed layers read views of host weights).

kept as:

    // architecture support gate: both model probes (runtime cache loader and planner) set this
    // from the loaded model; nullptr = supported. pshard refuses LOUDLY (WARN + stock fallback)
    // rather than run a memory layout it cannot stream. No architecture is refused today; the gate
    // stays for the next memory wrapper that hides its pipe shards from the pshard cache constructor
    // or lets streamed layers read views of host weights directly.

### src/llama-pshard-plan.h:284-285 (file scope, `llama_params_fit_impl` declaration) - grep miss, narrative

original text:

    // fit params entry point used by pshard planning (relocated from the
    // base-era src/llama.cpp; upstream's generic fit moved to common/fit)

removed: the relocation history ("relocated from the base-era src/llama.cpp").

kept as:

    // fit params entry point used by pshard planning; upstream's generic fit lives in common/fit

### src/llama-pshard-plan.h:323 (`llama_pshard_plan_registry::kernel_copy_cap_mb`)

original text:

    // the machine profile's measured kernel-copy crossover (largest transfer at which a copy kernel still beats
    // a copy-engine transfer ordered against kernels); the pool sets it as the engine's cap while active.
    // -1 = not in the profile -> the engine keeps its default

removed: the word "measured" (the profile value is by definition profiler-measured; lines 324-325 unchanged).

kept as:

    // the machine profile's kernel-copy crossover (largest transfer at which a copy kernel still beats
    // a copy-engine transfer ordered against kernels); the pool sets it as the engine's cap while active.
    // -1 = not in the profile -> the engine keeps its default

### src/llama-pshard-plan.h:331-334 (`llama_pshard_plan_registry::mtp_head_extra_mb`)

original text:

    // RETIRED 2026-09-06 (kept so existing registry files still parse; never charged): the
    // analytical arena charge for the MTP context's larger device compute with the head on the
    // CPU. The one-budget fit measures that need under the fitted placement instead
    // (common_pshard_fit_one_budget).

removed: the retirement date 2026-09-06 (retired in the three-fixes pass that re-measured the MTP reserve post-fit; see memory note pshard-mtp-enablement / expert-pool-build). Lines 332-334 unchanged.

kept as:

    // retired (kept so existing registry files still parse; never charged): the
    // analytical arena charge for the MTP context's larger device compute with the head on the
    // CPU. The one-budget fit measures that need under the fitted placement instead
    // (common_pshard_fit_one_budget).

### src/llama-pshard-plan.h:363-365 (`llama_pshard_plan_registry::arena_bytes`)

original text:

    // mtp_head_extra_mb is retired (2026-09-06): the MTP context's need under the plan's
    // placement is measured by the one-budget fit instead; the field stays for registry
    // compatibility and is not charged here

removed: the date and the repeat of the field comment's explanation (the MTP context's need under the plan's placement is measured by the one-budget fit; the field stays for registry compatibility). The code no longer references the field here (`budget = budget_bytes` was formerly `budget_bytes - mtp_head_extra_mb`).

kept as:

    // mtp_head_extra_mb is retired and not charged here (see the field comment)

### src/llama-pshard-plan.h:376-381 (`llama_pshard_plan_registry::attn_resident`)

original text:

    // layers whose ATTENTION is device-resident under a plan. n_attn_pinned is the
    // ATTNPRIO/ALTERNATE budget knob and stays 0 for strategies that pin attention
    // structurally: ATTNPIN_FFNSTREAM keeps every layer's attention resident, the
    // LAYERSTREAM/FFNCPU_ATTNSTREAM strategies only the fully pinned layers'. Pricing a
    // switch from the raw field charged an ATTNPIN <-> ATTNPRIO(attn=40) swap as 40
    // layers of attention traffic that never moves (selector-gap audit caveat, 2026-08-31).

removed: the concrete cell (ATTNPRIO attn=40 -> 40 layers of phantom attention traffic) and the audit reference (selector-gap audit caveat, 2026-08-31; see memory note selector-gap-root-causes). Lines 376-379 unchanged.

kept as:

    // layers whose ATTENTION is device-resident under a plan. n_attn_pinned is the
    // ATTNPRIO/ALTERNATE budget knob and stays 0 for strategies that pin attention
    // structurally: ATTNPIN_FFNSTREAM keeps every layer's attention resident, the
    // LAYERSTREAM/FFNCPU_ATTNSTREAM strategies only the fully pinned layers'. Pricing a
    // switch from the raw field would charge an ATTNPIN <-> ATTNPRIO swap for structurally
    // resident attention that never moves.

### src/llama-pshard-plan.h:498-501 (`llama_pshard_plan_registry::find_optimal_ubatch`)

original text:

    // pick the prefill ubatch with the lowest predicted ttft. Without TPS data the default
    // is the LARGEST VIABLE tier <= max_ubatch, never max_ubatch itself: the top tier can be
    // unviable by design (a pool tier whose scratch leaves no room for its region, 2026-09-05)
    // and a ubatch routed to it would run on the decode plan and spill past its window.

removed: the date 2026-09-05 (the day the "arena overflow chunks -> unviable tier" finding, bug 2, was made; see memory note expert-pool-build). Lines 498-499 and 501 unchanged.

kept as:

    // pick the prefill ubatch with the lowest predicted ttft. Without TPS data the default
    // is the LARGEST VIABLE tier <= max_ubatch, never max_ubatch itself: the top tier can be
    // unviable by design (a pool tier whose scratch leaves no room for its region)
    // and a ubatch routed to it would run on the decode plan and spill past its window.

## Regex hits reviewed and left unchanged (false positives)

### src/llama-pshard-plan.h:26 (`enum llama_pshard_strategy`, `LLAMA_PSHARD_EXPERT_POOL`)

    // pool region serves prefill A/B streaming and decode LRU slots

### src/llama-pshard-plan.h:80 (`enum llama_pshard_prefill_mode`, `LLAMA_PSHARD_PREFILL_AB_STREAM`)

    LLAMA_PSHARD_PREFILL_AB_STREAM = 0,  // whole expert set through the A/B span, hidden under GEMMs

reason: "A/B" here is the name of the pool's double-buffered prefill streaming span (docs/expert-pool-design.md: "A/B buffers", "A/B span", "A/B staging"), and it matches the identifier `LLAMA_PSHARD_PREFILL_AB_STREAM` / registry token `ab_stream`. It is a mechanism name, not an experiment A/B pair, so the convention permits it.

## Other pshard comments read and left as-is

- 21-22 (ALTERNATE), 24-26 (EXPERT_POOL), 65-74 (miss policies, incl. the RFC #24528 reference), 78-81, 104-106, 129-131, 149-159 (MTP layers / head lever), 170-174 (extra device bytes), 208-218 (plan fields), 226-227 (switch_ms), 291-293, 316-322, 327-329, 336-350, 368-371, 390-391, 397-412, 438-440, 484-487, 510-511, 527-529: mechanism descriptions without dates, measurements, cell names or history.

## Totals

- rewritten comment blocks: 7
- original comment lines changed or removed: 17 (161-167 = 7, 284-285 = 2, 323 = 1, 331 = 1, 363-365 = 3, 380-381 = 2, 500 = 1)
- regex hits left unchanged as false positives: 2 (lines 26, 80)

## Pass 2 (second dispatch of the same assignment, run after the pass-1 rewrite above)

Line endings: the file is LF, as the pass-1 note says; pass 2 read and wrote it with newline='' and left the endings byte for byte.

The two regex hits pass 1 left as false positives are comments, and the assignment counts only a format string or code as a legitimate remainder, so they are reworded to the mode's own name. The meaning is unchanged: the pool's double-buffered prefill span, registry token `ab_stream`, identifier `LLAMA_PSHARD_PREFILL_AB_STREAM`; docs/expert-pool-design.md keeps calling the mechanism "A/B buffers" / "A/B span".

### src/llama-pshard-plan.h:26 (`enum llama_pshard_strategy`, `LLAMA_PSHARD_EXPERT_POOL`)

- original: `// pool region serves prefill A/B streaming and decode LRU slots`
- kept as: `// pool region serves the prefill ab_stream double buffer and decode LRU slots`
- removed information: the design doc's "A/B" name for the span; the code's own token ab_stream stands in for it.

### src/llama-pshard-plan.h:80 (`enum llama_pshard_prefill_mode`, `LLAMA_PSHARD_PREFILL_AB_STREAM`)

- original: `LLAMA_PSHARD_PREFILL_AB_STREAM = 0,  // whole expert set through the A/B span, hidden under GEMMs`
- kept as: `LLAMA_PSHARD_PREFILL_AB_STREAM = 0,  // whole expert set streamed through the double-buffered span, hidden under GEMMs`
- removed information: the "A/B span" name, described as the double-buffered span instead. The code on the line is untouched.

### src/llama-pshard-plan.h:328-331 (`llama_pshard_plan_registry::mtp_head_extra_mb`) - step-2 trim, regex miss

original text (pass-1 output):

    // retired (kept so existing registry files still parse; never charged): the
    // analytical arena charge for the MTP context's larger device compute with the head on the
    // CPU. The one-budget fit measures that need under the fitted placement instead
    // (common_pshard_fit_one_budget).

kept as:

    // retired, never charged; kept so existing registry files still parse. The one-budget fit
    // measures the MTP context's device need under the fitted placement (common_pshard_fit_one_budget).

removed information: what the retired field used to hold - an analytical (formula) arena charge for the MTP draft context's larger device compute when the MTP head is CPU-resident (g_pshard_mtp_head_cpu / registry mtp_head_cpu). The charge was replaced by measuring the MTP context's need under the fitted placement (common_pshard_fit_one_budget, common/common.cpp); see memory notes pshard-mtp-enablement and expert-pool-build (three fixes: MTP reserve re-measured post-fit, head-lever charge retired).

## Other comments re-read in pass 2 and left as-is

- 65-66 ("no per-strategy allowed-sets"), 72-74 (RFC #24528), 161-165 and 176 ("no architecture is refused today"), 168-171 (DeepSeek-V4 compressor state as the example), 210-216 (design-doc section reference, "v1 uniform s"), 313-315, 320-325, 335-347 (headroom rationale: packing rounds differ by up to a few tens of MiB - kept, it is the reason the 64 MiB constant exists), 433-435 ("realistically 3-9" verify batch), 479-482, 493-496, 505-506, 522-524: mechanism, design or compatibility statements without dates, measurements, cell names, review references or machine names.

## Pass 2 totals

- comment lines rewritten: 6 (lines 26 and 80 reworded; the 4-line block 328-331 shrunk to 2 lines); the file is now 540 lines
- regex hits remaining after pass 2: 0
- code, identifiers, string literals and whitespace outside comments: unchanged (verified by stripping // comments from HEAD and the working tree and comparing the remainder)

## src/llama-pshard-workload.cpp

# comment rewrite notes: src/llama-pshard-workload.cpp

Line numbers refer to the file before the rewrite. Original text is preserved verbatim so the removed
dates, measurements, cell names and history stay recoverable.

## src/llama-pshard-workload.cpp:338-340 (llama_pshard_workload_calibrate, segment/start selection before the decode loop)

Original text:

```
// segments of 32 tokens, the first from BOS, the others from a random vocabulary token: the continuation
// of one start is one topic in one register and routes to few experts (q35: 4 experts took every token
// in the first layers), diverse starts spread the routing the way a real prompt mix does
```

Removed: the experiment/model name (q35) and the measured observation that motivated the segmented starts
(4 experts took every token in the first layers when one continuation ran unbroken from BOS).

Kept as:

```
// segments of 32 tokens, the first from BOS, the others from a random vocabulary token: one continuation
// routes to few experts (one topic, one register); diverse starts spread the routing like a real prompt mix
```

## step 2 review (comments read once more, no further rewrites)

- 60 (fit): "split-half: every route of expert i lands in fold A or B ..." - "fold A or B" names the two halves of the split, not an A/B experiment pair; mechanism only, kept.
- 77 (fit): "rank on one fold, measure the top-s cumulative share on the other" - mechanism, kept.
- 178-179 (save): fprintf header text contains "from measured routes" - string literal written into the sidecar file, not a comment; out of scope, untouched.
- 241 (load): "the file's alpha may come from an older estimator; the counts are the record" - one-line why for the refit (forward compatibility of the sidecar), no date/number/cell, kept.
- 253-254 (calib_cb): node-name matching mechanism, kept.
- 296-298 (llama_pshard_workload_calibrate): why the calibration load is CPU-only and what stands in for the workload; mechanism/why, kept.
- 336, 358 (llama_pshard_workload_calibrate): BOS fallback and EOG restart, one-line why each, kept.

## step 3 grep

Remaining flagged comment lines after the rewrite: 0.

## src/llama-pshard-workload.h

# src/llama-pshard-workload.h - comment rewrite notes

Line numbers are the pre-edit ones (file was 53 lines, now 51; LF line endings, preserved). Information removed from the code comments is preserved here.

## Rewritten

src/llama-pshard-workload.h:10-14 (struct llama_pshard_workload, header comment) - original: `// The model's routing workload: how skewed the router's expert choices are, as the Zipf exponent alpha the` / `// planner's hit-rate model h(s) is built on. Measured, never assumed: the expert pool histograms every` / `// cache-mode route and folds each run into <model>.pshard_workload at exit (real runs ACCUMULATE; the plan-time` / `// calibration below is a stand-in that the first real run replaces); a model without a file is calibrated at` / `// plan time (CPU-only sampled generation through the eval callback).` - kept as: `// The model's routing workload: the Zipf exponent alpha of the router's expert choices, the input of the` / `// planner's hit-rate model h(s). The expert pool histograms every cache-mode route and folds each run into` / `// <model>.pshard_workload at exit (runs accumulate); a model without a file is calibrated at plan time` / `// (llama_pshard_workload_calibrate below), and the first real run replaces that calibration.` (removed: the "Measured, never assumed" design-directive phrasing, which records the 2026-09-12 no-constants directive that planner inputs come only from the machine profile, the GGUF table and the workload ledger rather than built-in numbers; the capitalised "real runs ACCUMULATE" emphasis, which contrasted the accumulating sidecar with the earlier single-shot behaviour; the "stand-in" wording for the plan-time calibration; and the parenthetical "(CPU-only sampled generation through the eval callback)", which is still stated in full in the llama_pshard_workload_calibrate doc block at the bottom of the file)

src/llama-pshard-workload.h:35-40 (llama_pshard_workload::fit, doc comment) - original: `// fit alpha from per-layer expert use counts. h_obs(s) is the SPLIT-HALF top-s mass: each expert's routes` / `// are assigned to two folds (deterministic binomial split), the experts are ranked on one fold and the` / `// cumulative share of the s top-ranked is measured on the other, both ways, averaged over layers - ranking` / `// and measuring on the same sample inflates the top-s mass when routes per expert are few (a 128-token` / `// run has ~4 per expert on a 256-expert model), which read as a too-skewed alpha. alpha minimises` / `// sum_s (h_obs(s) - h_zipf(s; alpha))^2 over s = 1..n_expert on a 0.005 grid in [0, 3].` - kept as: `// fit alpha from per-layer expert use counts. h_obs(s) is a split-half top-s mass: each expert's routes are` / `// split into two folds (deterministic binomial split), the experts are ranked on one fold and the cumulative` / `// share of the s top-ranked is taken on the other, both ways, averaged over layers - ranking and scoring on` / `// the same sample inflates the top-s mass when routes per expert are few, which reads as a too-skewed alpha.` / `// alpha minimises sum_s (h_obs(s) - h_zipf(s; alpha))^2 over s = 1..n_expert on a 0.005 grid in [0, 3].` (removed: the worked example "(a 128-token run has ~4 per expert on a 256-expert model)" - 128 tokens x 8 routed experts per token / 256 experts = 4 routes per expert, the regime in which the same-sample fit over-estimated alpha and motivated the split-half fit; the capitalised "SPLIT-HALF" emphasis; the word "measured" in "is measured on the other" (algorithmic sense, replaced by "taken") and "measuring" (replaced by "scoring") so the block no longer reads as a measurement note. The 0.005 grid in [0, 3] is retained: it is what the code does - 601 steps of 0.005 in llama-pshard-workload.cpp - not a measured number)

## Reviewed, left as is

src/llama-pshard-workload.h:16-23 (field comments), :25-26, :28, :30 (load/save/set_counts/merge_counts), :43 (zipf_h), :47-50 (llama_pshard_workload_calibrate doc block) - mechanism only, no dates, measured numbers, cell names, review references, machine names or history.

## src/models/deepseek4.cpp

# comment rewrite notes: src/models/deepseek4.cpp

Line numbers refer to the file before the rewrite. Original text is preserved verbatim so the removed
dates, measurements, cell names and history stay recoverable.

## src/models/deepseek4.cpp:460-464 (llama_model_deepseek4::graph::build_hc_head, cb() names on the hc_head tail)

Original text:

```
// named so pshard's delegated-mode pin anchors them to the head's backend: unnamed, the
// scheduler's op-offload rule placed them by row count (n_outputs), on CUDA0 in a tier's
// reserve (n_outputs = bs) and on the CPU for a prompt (0 or 1 outputs) - a different backend
// assignment forces the scheduler to re-plan the arena for the first prefill graph of every
// tier (2026-09-06: 1088 MiB overflow on DSv4 + DSpark at -mva 8000)
```

Removed: the date the naming landed (2026-09-06), the measured arena overflow (1088 MiB), the model pair
and cell it was seen on (DSv4 + DSpark at -mva 8000), the backend-specific placement detail (CUDA0 in a
tier's reserve with n_outputs = bs, CPU for a prompt with 0 or 1 outputs) and the past-tense
"placed them" history phrasing.

Kept as:

```
// named so pshard's delegated-mode pin anchors them to the head's backend: unnamed, the
// scheduler's op-offload rule places them by row count (n_outputs), which differs between a
// tier's reserve graph and a prompt graph and forces an arena re-plan on every tier's first prefill
```

## step 2 pass

The file has only four other comments (lines 119-120 wo_a reshape, 266 hyper-connection mean,
319-320 Sinkhorn reference), all upstream model-graph notes with no pshard narrative. Nothing else trimmed.

## tools/pshard-plan-params/pshard-plan-params.cpp

# comment rewrite notes: tools/pshard-plan-params/pshard-plan-params.cpp

File is LF (not CRLF); line endings preserved. Only comment text changed; one 6-line comment block rewritten (lines 207-212, still 6 lines). Line count unchanged (315). No code, identifiers, string literals or whitespace outside the comment touched.

Step 1 grep hits: 1 (line 208). Step 2 read-through of the other pshard comments (lines 191-192 refusal return value, 233 one-budget rule, 241 stock-fallback sentinel, 248-249 fit passes, 277-278 SPECULATIVE example set): all state the mechanism only, nothing trimmed. Step 3 grep after the edit: 0 hits.

## tools/pshard-plan-params/pshard-plan-params.cpp:207-212 (plan_pshard_context, the spec verify tier / n_outputs_max block)

- original:
  ```
  // spec verify tier: mirror the runtime's output-limits derivation. n_outputs_max is what the
  // runtime reserve (and, since 2026-09-06, the planner's probes) clamp a tier's outputs to:
  // speculative tools set it to the output limits' total; completion/perplexity leave it 0
  // (= n_batch, every token may be an output), so a plain plan must too, or the probes would
  // price the logits scratch at 1 output while the runtime reserves bs (types defaults to
  // { NONE }, so "configured" means a draft model or a non-zero draft length)
  ```
- kept as:
  ```
  // spec verify tier: mirror the runtime's output-limits derivation. n_outputs_max is what the
  // runtime reserve and the planner's probes clamp a tier's outputs to: speculative tools set it
  // to the output limits' total; completion/perplexity leave it 0 (= n_batch, every token may be
  // an output), so a plain plan must too, or the probes would price the logits scratch at 1 output
  // while the runtime reserves bs (types defaults to { NONE }, so "configured" means a draft model
  // or a non-zero draft length)
  ```
- removed information: the date clause "since 2026-09-06" - the planner's probes started clamping a tier's outputs to n_outputs_max on 2026-09-06 (the "probe output clamp" follow-up of the three fixes of that day, together with the plan-tool spec-config trap this block guards against: a plain plan must leave n_outputs_max at 0 so the probes price the logits scratch at n_batch outputs, not 1, matching the runtime reserve). Before that date only the runtime reserve applied the clamp. The mechanism sentence is kept in full; only the history marker is gone.
