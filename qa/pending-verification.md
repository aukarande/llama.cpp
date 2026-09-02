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
   **Perf recipe changed 2026-09-01 (user):** perf runs carry ONLY the workload and the
   budget - no `-ub/-b`, no `--temp 0`, no `--ignore-eos`, no logging flags - on
   either side. Stock: GOLDEN run (ub matched to cache_ubatch, temp 0, hash only) +
   perf BASELINE at defaults. Pshard: perf run at defaults + CORRECTNESS run (temp 0,
   `-lv 4`) for the hash and tier summary. One extra pshard run per config (~+2 h on
   the 144 grid). `--ignore-eos` IS workload (user, 2026-09-01) and is on both perf runs:
   without it default sampling EOS'd early (q35@4000 perf run: 2 tokens, 18 t/s). Safety
   net: windows < N_GEN/2 are retried up to 3x and recorded in the new ledger column
   `decode_tokens` (compare-qa hard-fails what stays short).
   Consequences for the rerun: (a) reference-ledger perf columns
   pre-date the change - stock rows are not perf-gated, pshard rows will most likely
   read as IMPROVEMENTS (DEBUG logging removed from the hot path), so refresh the
   reference from the new run rather than treating deltas as drift; (b) the old
   matched-ub stock numbers were not a baseline at tight budgets (q35@4000: ub 2048
   forced a ~2 GB logits scratch, fit fell to 20/41 layers, 21 t/s; at defaults 44
   t/s) - the "pshard beats stock at equal budget" claim must be re-read against the
   new stock rows. (c) The PPL-parity mirror no longer mirrors placement (user,
   2026-09-01: no `-ngl`, no `-ot`; stock uses `-fitt` + pshard's executed ubatch +
   `--swa-full`). The old `-ngl 99` branch could not even load q35 (20 GB on 16 GB ->
   NO_BASELINE, seen on the first cell run under the new recipe); the new residual
   includes GPU-vs-CPU expert math (q35@4000: 0.35% of the 0.5% band). Expect gpt-oss
   token-diverged cells to report PPL_MISMATCH from placement alone (calibration:
   stock spans 1401.9 -> 4025.0 by placement) - those need a decision, not a fix.
1b. ~~Planner canonical-union accounting~~ **CLOSED 2026-09-01 (4fadc725f)**:
   plan-time union enforcement (loader-parity packing simulation, byte-exact
   common_end 1729.71 MiB; per-tier scratch_off + measured pinned cache <=
   budget; bounded demotion loop) + the actual root cause: measurement probes
   carrying a partially-populated registry took the canonical-preload path and
   measured cache=0, mispricing fallback plans. Repro cell q8d-16k@2000-s1:
   STOCK_FALLBACK -> 6 viable tiers, all union-OK, runs rc=0. Invariants held
   (q35 pick identical, s4 smoke byte-identical, wmtp MTP unchanged).
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
5b. ~~Defer-prefetch grid A/B~~ **CLOSED 2026-08-31, code path DELETED**: A/B on
   8 MoE s4 cells (q35+oss x 2ctx x mva 4000/12000, fixed stack, N_GEN=256):
   decode deltas -1.1..+1.7% (noise), hashes identical on all cells. The MoE
   hypothesis failed - with sliced expert uploads, decode prefetch traffic is
   too small for submission order to matter. Dense (q8d) was already proven
   DRAM-conserved (+2.7%). No cell near the bar -> deleted per no-speedup rule.
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

9. **pshard+MTP LANDED 2026-09-01** (overnight; commits 3c8e0a48c ddb019848
   24237a489 7fcd9e136 12b3f837c): speculative decoding works under -pshard.
   Five stacked root causes fixed (see 12b3f837c message). Verified wmtp
   c2048/mva4000: no-spec 45.4 -> +MTP n_max=2 52.8 t/s (+16%, accept 78.3%);
   single-seq non-spec byte-identical. REMAINING (queued): [a] ~~draft VRAM
   reserve~~ CLOSED 2026-09-01 (2d368c4ed): one-budget reserve for spec contexts
   (runtime + plan tool price the same effective budget; MTP ctx measured,
   dflash/eagle modeled from gguf metadata + affine compute, spec ctx ubatch
   <= 128, footprint self-check at teardown). dflash pair @4000: 0/576
   accepted + 9067 MiB peak -> 121/219, 50.5 t/s, 4315 MiB peak (single-ctx
   pshard @4000 = 4318). Found + fixed on the way: (i) the MTP ctx and a galloc
   overflow chunk sat OUTSIDE the budget (56.9 t/s @ 5063 MiB -> 56.3 t/s @
   4367 MiB): the union enforcer now counts probe-measured compute scratch +
   a 32 MiB runtime-packing margin; (ii) a scheduler rebuild after a draft
   enables layer-input extraction left stale per-plan alloc states
   (n_accept=0) -> lazy re-reserve + immediate re-land of the active plan.
   dspark TESTED 2026-09-01 (82f78675f): q35+dspark runs clean under the budget
   (4000: 30.9 t/s, peak 4194 MiB; 12000: 46.4 t/s, peak 12190 MiB; reserve
   model within 20 MiB) but the draft is weak on this quant - 21% accepted
   vs 16% in STOCK on the same pair, so not a pshard defect. DSv4+DSpark:
   the big-draft spill rule fired as designed (9792 MiB experts -> CPU; 594
   dense + 12 KV + 1126 compute reserved, self-check -11 MiB) BUT the
   DeepSeek-V4 TARGET is broken under pshard: llama_kv_cache_dsv4 never
   exposed its pipe shards (fixed), yet pinned-attention plans still produce
   garbage from the first token and streamed-attention plans (s0/s2) hit a
   CUDA illegal memory access; stock CPU text is coherent. pshard now REFUSES
   DeepSeek-V4 loudly (WARN + stock fallback, text identical to stock). Never
   in the harness grid; the 08-30 "DSv4" validation used a different file on
   the dual-cache DSA classes. ~~DSv4 compressed-cache support~~ LANDED
   2026-09-01 (b508ae660): three root causes - (1) pshard cache ctor skipped the
   attention-rotation tail -> k_rot null -> deepseek4 silently built RAW
   attention (localized by per-layer eval-callback vs CPU: first jump at
   layer 2); (2) sched host-weight rules ignored VIEWS of host weights ->
   streamed layers read hc_* slices from host memory (compute-sanitizer);
   (3) compressed caches are graph-written -> whole-layer transfers. Text
   byte-identical to stock CPU for auto/s0/s1/s2 @12000, PPL 2.5408 vs CPU
   2.5443 with clears between chunks, DSpark pair 63/110 within budget,
   decode 10.8 t/s. Follow-ups CLOSED 2026-09-01 (319bbc6b5): compressor-state
   VRAM (11.64 MiB here; grows with n_seq_max*(1+n_rs_seq)) is now reserved
   from the pshard budget by both probes (arena 11989 + state = 12000);
   multi-sequence exercised with llama-parallel (2 clients, 4 seqs, shared
   prompt -> seq_cp) under streamed attention: clean, deterministic, 3/4
   greedy responses identical to CPU and the 4th prompt is greedy-unstable
   in every configuration (differs CPU-vs-CPU too). STILL OPEN: DSv4 not in
   the harness grid (grid held by user; a 97 GB model roughly doubles grid
   time); page-locking fails for the 45 GB third shard (pinned-memory
   limit) so DSv4 streaming runs on pageable copies - correct, slower; a
   tier marked unviable at load should fall back to the nearest viable tier
   (decode stayed in the 512-token streaming plan: 6.3 t/s at a 2024 MiB
   budget before the margin fix); joint target+draft planning (cordis v2)
   deferred until v1 shows a case where leftover budget could pin draft
   experts; [b] FNV
   logits-hash spec test port (cordis tests/test_pshard_spec.cpp shape);
   [c] stock+MTP equal-budget QA cell (stock-fallback+MTP at DEFAULT fit hit
   ~83 t/s - budget-equalized comparison pending); [d] ~~forced-s3+MTP
   n_accept=0 anomaly~~ RESOLVED (070d9fb27): the draft ctx's layer-40 KV
   cache takes the pipe-shard branch (gated on model.is_pshard()) and its
   GPU k/v stay unbacked (nothing packs a non-pshard ctx) -> scratch KV ->
   garbage history whenever the plan PINNED the nextn attn; nextn layers
   now always CPU-resident; [e] server spec path untested; [f] plan-tool
   arg parse now LLAMA_EXAMPLE_SPECULATIVE (accepts --spec-type /
   --spec-draft-n-max); [g] ~~per-context memory gating~~ CLOSED
   2026-09-01 (ed7136c04): pipe-shard KV/RS branches gated on the creating ctx's
   cparams.pshard (thread_local set in llama_model::create_memory), so the
   draft ctx gets stock backed KV and the MTP head pins again (pin-priority
   in both emitters); union enforcer demotes every violator per round with
   doubling cuts (46 MiB overshoot stalled the old single-cut loop), head-CPU
   only as the fallback lever (persisted mtp_head_cpu). wmtp draft-mtp n=2:
   4000 MiB 52.8 -> 56.9 t/s (117/155 accept), 12000 MiB 72.1 -> 83.0 t/s
   (114/158); s1@4000 single-ctx output byte-identical; [h] ~~delegate-compute
   parity~~ CLOSED 2026-09-01 (1dd7582c4): cache loader now mirrors the plan
   path; cached delegate-strategy plans no longer build early reserves
   un-delegated (s3-from-cache 46.5-47.1 vs 46.3-46.4 pre-fix, coherent).

10. ~~Fused-GLU certification~~ **CLOSED 2026-09-01 (8afa86ccf)**: root cause =
   galloc frees sched weight copies mid-split; the GLU dst legally lands in the
   just-freed slot bytes; the FUSED kernel reads gate/up while writing that dst
   (WAR hazard); the fusion memory-range check skipped op-NONE srcs and never saw
   it. Probe: 114 aliasing pairs (ffn_moe_swiglu inside ffn_gate_exps' slot).
   Check fixed; blanket env force-set removed (fusion ON by default for pshard;
   env = manual lever, value-based). PPL parity exact (1.2619 both), perf neutral.
   NOTE for grid: fusion numerics may shift temp-0 token hashes vs pre-fusion
   references - PPL parity is the gate; references refresh with the next grid.
   The deleted slot carve-out had masked this class by construction; its deletion
   stands (the range check now covers it soundly). Upstreamable fix.

## Contamination note (2026-08-30)

Two "stopped" harness runs kept running detached and overlapped each other and
ad-hoc tests. Perf numbers from that window are invalid; correctness outcomes
(DSv4 runs + output text, loader behavior classes) and planner PREDICTION numbers
(pure compute, no GPU contention sensitivity) remain valid.
