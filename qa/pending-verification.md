# Pending verification queue

Changes land with build-only checks; GPU verification is BATCHED here and drained
in one session (machine exclusive, zombie-check first: `tasklist /fi "IMAGENAME eq
llama-perplexity.exe"` etc. - TaskStop'd harness shells survive as detached sh.exe
and corrupt concurrent measurements).

## Drained 2026-08-31 (targeted items; grid + nem HELD by user)

- Loader determinism: VERIFIED - exact hit / smaller-with-warning / honest refusal.
- Switch-cost constants: VERIFIED - variant header switch_mb=509.1 attn_frac=0.12
  head_mb=885.4 pcie=45.0 (matches hand arithmetic); behavioral check rides the grid.
- ids-cross: VERIFIED - oss s4 flags exactly tier bs=1 (pair gate) with router
  patterns; q35-16k@2000 forced-s4 decode 7.65 -> 19.99 t/s (+161%), prompt
  unchanged, token gate OK. (Two silently no-op'd serialization hunks were found
  and restored en route - 7c6a48bbe. Lesson: verify batch replaces by count.)
- DSv4 perf (clean machine): s3 forced = 52.1 prompt / 10.87 decode t/s. NEW BUG
  found below.

## Queued

0. ~~DSv4 streamed-attn runtime crash~~ **FIXED**: root cause was one level up from
   the write-cells guess - llama_kv_cache_dsa(+_iswa) never overrode
   get_pipe_shards(), so the ENTIRE pipe-shard machinery was invisible for
   dual-cache models (no tensor assignment, no uploads, no writeback). Added the
   overrides ([mla, lid] / [mla, lid, swa]) + write-cells binder cases for both
   DSA context types with dedicated lid storage. Crash repro (auto, prompt-512,
   -mva 8000) now completes: 21.7 prompt / 9.9 decode t/s, coherent output.
   FOLLOW-UP observation: DSv4 auto prefill (21.7) trails forced-s3 (52.1) - the
   bs=512 FFNCPU tier's predicted 100 tps measures ~22; predictor accuracy on
   MLA/dense-lead architectures is untuned. Value A/B vs stock rides the held
   grid-class runs.
0b. ~~Harness nit: strategy_prefill parse~~ **CLOSED 2026-08-31** as part of the
   PPL-gate rework (item 8): the gate now keys placement off the effective
   prefill tier (bs=PUB) and includes STATIC in the placement-match case.

1. **Full 144-grid rerun** - restart FRESH (plans changed under items 5/1/7 +
   DSA fix; the 42 pre-change rows are stale): `sh qa/run-qa.sh /tmp/qa-full-v3 full`.
   Then compare-qa vs reference, refresh ledger (schema v2), commit, push.
2. ~~Nemotron full slice~~ **REMOVED 2026-08-31** (user): subsumed by the full
   grid rerun, which includes nem.
3. ~~Switch-cost behavioral check~~ **VERIFIED 2026-08-31**: switch_ms fields
   populated (0 for tier-0, 83-105ms for residency deltas); prompt-1500 -> ub 2048
   and prompt-16k -> ub 8192 both hand-verify as argmin of n_iters*ts/tps +
   2*switch_ms. CAVEAT (small follow-up): strategies that pin attention
   structurally (ATTNPIN/LAYERSTREAM) do not record it in n_attn_pinned, so
   switch_ms between them can overcharge; fine for 0-vs-100ms discrimination.
4. ~~DSv4 predictor tuning~~ **FIXED 2026-08-31**: IQ-quant expert matmuls had no
   benchmark entries -> priced by memory bandwidth alone (118 nodes for 224ms)
   while dequant-compute-bound at batch. Added a compute floor for quantized CPU
   matmuls without a same-type entry: max(bytes/bw, ops/floor) where floor = the
   slowest BATCH-measured (B>=32) CPU matmul rate, applied only at M>=32 rows
   (bs=1 matvecs are genuinely memory-bound - the first floor version overcharged
   them and flipped tier-0; caught by ladder inspection). Result: DSv4 ladder all
   coherent STATIC, tier-0 predicted 10.89 vs measured 10.87 (0.2%!), auto prompt
   21.7 -> 46.6 t/s (+115%), within ~10% of forced-s3. A/B-vs-stock cell rides
   the held grid-class runs.
5. ~~Slot carve-out / per-slot fence~~ **CLOSED 2026-08-30, carve-out DELETED
   (30d3f4ed1)**: ceiling probe with the prefetch fence removed entirely recovered
   nothing (437 vs 447 ms/token, with or without slot regions) - the q8d-s4
   serialization is host-DRAM-bandwidth bound (~445 ms/token invariant across
   fence-off / async-handoff / defer-prefetch / no-mmap), so a finer fence has no
   headroom and the carve-out lost its purpose. Attribution instrumentation
   (sched_sync/sched_copy under GGML_SCHED_TIMING, 371d99dab) and env-gated
   ordering experiments (GGML_DEFER_PREFETCH, GGML_ASYNC_HANDOFF, ca622d5b5)
   retained. Sanity: token-identical s4/s0 output, clean exit.
5b. **Defer-prefetch grid A/B** - GGML_DEFER_PREFETCH=1 collapses s4's download
   wait 172ms -> 1ms/token (downloads no longer queue behind layer uploads on the
   single copy queue) but only nets +2.7% on q8d (DRAM contention moves the cost
   into CPU FFN). On MoE cells with small CPU-side reads (oss/q35 sliced-expert
   configs) the contention term shrinks, so it may win there. Ride one env-on
   column on the held full-grid rerun; keep or delete by the same rule.
6. **ALTERNATE adjudication (user question)** - after the full grid rerun under the
   fixed predictor + ids-cross: does s4 ever win a cell against re-planned s1/s3?
   If auto never picks it and forced-s4 never beats the best alternative, ALTERNATE
   earns retirement from the decode tiers (stays as a prefill design point).
7a. **Selector-gap fixes LANDED + VERIFIED 2026-08-31** (user-approved, all three):
   (1) hi_attn fresh bound at bs=1 (llama-pshard-plan.cpp: tier_prune::attn_hint);
   (2) sliced-upload repricing (profiler gathered-slice microbench, measured
   23.3/29.6/30.0/29.4 GB/s at 0.5/2/8/32MB chunks vs 45 peak; curve in
   cpu_profile.txt header, log-interpolated at the split's per-expert chunk size);
   (3) N_GEN=256 decode window (switch cost stays in, amortized - user rule).
   Planner verification: all 8 q35 cells now pick STATIC attn=40 (predictions
   0.3-4.2% on measured cells); oss controls byte-identical (16k-4000 same plan).
   GPU verification (4 cells): q35-16k-2000 auto 19.9 -> 30.4 t/s (+53%, pred
   29.6); q35-2k-8000 auto 45.0 -> 57.0 (+27%, now BEATS stock 53.7); oss-16k-2000
   auto 18.5 -> 29.0 (+58%, new attn=24 plan); oss-16k-4000 control 36.2 ~= 35.0.
   NOTE: qa-full-v3's 81 grid rows are STALE (planner + N_GEN + gate all changed);
   the full grid needs a fresh restart on the fixed stack.
8. ~~oss-16k PPL parity calibration~~ **MOSTLY CLOSED 2026-08-31**: gate now
   mirrors the pshard PPL run's ACTUAL executed config (verbose pshard side ->
   parse prefill_ubatch_eff + apply_plan tier; GPUONLY_* tiers -> plain -ngl 99,
   CPU-delegate tiers -> exps=CPU list; stock always gets --swa-full + matched
   -ub). Certified: ATTNPIN@4096 mirror agrees to FIVE DIGITS (39368.10 vs
   39367.59); oss-16k-mva4000 formally TOKEN_DIVERGED_PPL_OK. Root causes were
   (a) executed-tier visibility, (b) executed ubatch, (c) SWA cache sizing
   (pshard allocates full SWA cache, stock defaults window+batch = 2.6% alone).
8b. **RESIDUAL (narrow, OPEN)**: tight-budget cells whose executed prefill tier
   is LAYERSTREAM (streamed ATTENTION weights): oss-16k-mva2000 sits 0.51% over
   the certified all-GPU mirror (39570.7748 vs 39368.0990, deterministic;
   n_batch ruled out - b=8192 mirror is bit-identical to b=4096). Suspects:
   streamed-attn slot-upload numerics or host-side KV at tight budget; needs
   logit-level bisection. Manual adjudication meanwhile: the pshard value is
   deterministic - compare against the recorded reference (39570.7748 for this
   cell); reproduction = benign, drift = real.
7. **Selector-gap audit (2026-08-31, grid stopped at 81 rows for this)** - verified
   findings from the q35/oss slice (workflow-verified, 3 adversarial lenses):
   (a) BENCH ARTIFACT: 31-token decode window swallows the prefill->decode
       pshard_apply_plan re-upload (up to 8.3 GiB = 190ms); auto pays more switch
       bytes than forced runs -> fake gaps. Fix: exclude switch from perf window or
       decode >=256 tokens. Resolves the q35-16k-12000 same-strategy anomaly
       (steady 55.3 vs 56.2; true winner there is s3 at 62.1 steady).
   (b) PLAN-SEARCH BUG (the 43% cell): llama-pshard-plan.cpp:286-290 tier prune
       inherits hi_attn from large-batch tiers into bs=1 (scratch shrinks ~350x,
       monotonicity invalid) -> auto's STATIC candidate capped attn=11 (12.1 tps)
       while forced-s1's fallback searches attn=40 fresh (29.6 pred/28.44 meas).
       Fix: re-search tier 0 with hi_attn=UINT32_MAX.
   (c) TIME-MODEL BIAS (6 mis-ranked cells): consume-time sliced expert uploads
       priced at peak PCIe 45 GB/s; measured concurrent rate 29.5 (0.6MB slices)
       to ~38 (9-13MB slices). Repricing at profiled chunk-size-aware bw flips all
       6 mis-ranked cells, flips zero controls. s3 STATIC already calibrated at 2k.
   (d) REFUTED sub-fix: serializing CPU-split prefetch "per q8d conservation"
       over-corrects dense-ALTERNATE by 21-36% (q8d predictor is already 5-8%
       pessimistic; conservation datum implies whole-token DRAM closure at PEAK
       45.7 GB/s, not serialization stacked on derated terms). Needs redesign
       gated off dense-symmetric plans; do NOT land as specified.
   (e) Secondary: FLASH_ATTN@16k over-priced ~2.4ms/tok (STATIC -13.5% under at
       16k); fully-GPU-resident plans +35% over (ranking-neutral); the
       oss-c16384-mva2000 cell spans a harness restart (weak comparability).

## Contamination note (2026-08-30)

Two "stopped" harness runs kept running detached and overlapped each other and
ad-hoc tests. Perf numbers from that window are invalid; correctness outcomes
(DSv4 runs + output text, loader behavior classes) and planner PREDICTION numbers
(pure compute, no GPU contention sensitivity) remain valid.
