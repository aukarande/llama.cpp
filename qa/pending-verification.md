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
   (d) Stock's budget is now `-fitb MVA` (new --fit-budget: fit target = free - budget, so
   weights + KV + compute <= MVA), the same thing pshard's arena is. `-fitt (FREE - MVA)`
   had handed stock ~560 MiB less on q35@4000 (3443 MiB fitted vs pshard's 4000 MiB
   arena) because the fit's free-memory view already excludes the CUDA context.
   CUDA-side overhead (~220 MiB context + ~80 MiB workspaces) is outside the budget on
   both sides by decision (user, 2026-09-01).
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
   stock spans 1401.9 -> 4025.0 by placement). DECIDED 2026-09-02 (user): the gpt-oss
   STOCK baseline is itself unreliable on raw-text PPL - nothing to fix on our side;
   read gpt-oss PPL cells as informational, not as a gate.
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
   2*switch_ms. ~~CAVEAT: strategies that pin attention structurally
   (ATTNPIN/LAYERSTREAM) do not record it in n_attn_pinned, so switch_ms between
   them can overcharge~~ **CLOSED 2026-09-02**: the registry carries n_layers
   in-memory (planner + runtime) and prices attention residency through
   attn_resident(plan): ATTNPIN_FFNSTREAM = every layer, LAYERSTREAM/FFNCPU =
   the fully pinned layers only (n_attn_pinned stays the ATTNPRIO/ALTERNATE
   knob, so emitters and hi_attn pruning are untouched). Fresh q35+dflash
   registry @4000: LAYERSTREAM tiers 96.6 ms from the ATTNPRIO decode tier
   (2 layers + 40 attention pins + head), consistent with the formula.
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
   limit) so DSv4 streaming runs on pageable copies - correct, slower. MEASURED 2026-09-02 (q35 forced s0, GGML_CUDA_REGISTER_HOST=0 now forces pageable): pageable uploads are 3.3x slower (prompt 638 -> 189 t/s, decode 2.33 -> 0.76 t/s), i.e. ~13.5 vs 45 GB/s. It does NOT affect DSv4 @12000 today: the plan is STATIC_ATTNPRIO at every tier (attention pinned, experts on CPU), nothing streams. It would matter for a streaming plan (half the bytes pageable -> ~21 GB/s effective) and the planner prices ALL uploads at 45 GB/s - an honesty gap for such plans (follow-up: per-mapping lockability -> per-layer upload rate in the predictor). ATTRIBUTED 2026-09-02 with Nsight Systems on the streaming prefill itself (DSv4 @12000 without a draft picks GPUONLY_ATTNPIN_FFNSTREAM for tiers 512-2048, predicted 118 t/s; 343-token prompt = 12.1 s, 28.3 t/s): the GPU trace shows ALL kernels of the run total 0.69 s (MMQ for IQ2_XS/IQ3_XXS experts 0.5 s), host->device copies from PINNED memory 56 GB in 1.45 s (38.8 GB/s, 40 GB/s for the 0.7 GB slabs), copies from the PAGEABLE third shard 44.5 GB in 10.7 s (4.1 GB/s). The prompt is the pageable shard, nothing else: with it pinned the prefill would take ~2.5 s (~137 t/s, i.e. the predictor is about right for the pinned case). It is NOT disk: disk reads during the run are 0 MB/s (perf counter), the mapping is RAM-resident; 4.1 GB/s is the driver's pageable path on file-backed pages. Anonymous memory (--no-mmap, 86 GB CPU buffer, nothing page-locked) copies at ~7.6 GB/s: prompt 11.9 s, no better overall, 100 s load. Page-lock CEILING on this machine (cudaHostRegister in 4 GB chunks): 60 GB of 127 GB RAM. Options for DSv4 prefill, ranked (343-token batch, today 12.1 s): (1) staging ring for unlockable regions (CPU threads memcpy into a pinned ring, GPU DMAs from it, ~30 GB/s) -> ~2.7 s, ~4.5x; (2) pin 60 GB of the 84 GB of experts by chunked registration -> ~7.4 s, 1.6x, and combined with the ring ~2.3 s; (3) load the unlockable shard into anonymous memory -> ~7.3 s, 1.7x, costs RAM and load time. Also: the predictor must price uploads per mapping (45 pinned / ~4 file-pageable / ~7.6 anon-pageable / ~30 staged GB/s) - with that it would predict ~29 t/s for today's plan, matching the measurement, instead of 118. STAGING RING LANDED 2026-09-02 (ggml-cuda.cu, ggml_cuda_staged_h2d): set_tensor_async detects an unregistered (pageable) source via cudaPointerGetAttributes and, for copies >= 1 MiB outside graph capture, stages them through a lazily cudaHostAlloc'd pinned ring (8 x 64 MiB, GGML_CUDA_STAGE_RING_MB, 0 disables; GGML_CUDA_STAGE_THREADS memcpy threads, default 8 - measured 2/4/8/12 threads = 115/157/211/206 t/s on all-pageable q35). Outputs byte-identical with and without the ring (q35 and DSv4, greedy). Results: DSv4 @12000 prompt 28.3 -> 75.2 t/s (decode unchanged 10.1, CPU-expert tier); all-pageable q35 prompt 68 -> 211 t/s (pinned 678). nsys with the ring: every DMA now from pinned memory, 100.7 GB at 34.8 GB/s. Remaining gap to the ~3 s DMA bound (~114 t/s): the staging call stays host-synchronous (memcpy of a 2 GB layer blocks the scheduler thread ~70 ms before it can launch that layer's compute); an asynchronous variant (worker thread + stream-ordered completion) is the next step. Review (15 agents) confirmed two items, both fixed: slot events must live on the caller's device (a CUDA event records only on a stream of its own device) -> one ring per PHYSICAL device, events created under ggml_cuda_set_device; and the HIP/MUSA vendor shims lack the pointer-attribute/host-alloc/capture-status mappings -> ring compiled for CUDA proper only, stub elsewhere. Also: thread spawn only for chunks >= 16 MiB (sliced expert copies at decode stay single-memcpy), and the measurement levers (GGML_CUDA_REGISTER_HOST=0, PSHARD_PAGELOCK_SKIP) now measure pageable+ring unless GGML_CUDA_STAGE_RING_MB=0 is added. Ring caveats: it only helps while the pageable pages are RAM-resident (evicted pages make it disk-bound like any path), and staging bandwidth is host DRAM shared across GPUs (does not scale with GPU count). ASYNC VARIANT LANDED 2026-09-02 (ggml_cuda_stage_worker, one worker thread per physical device; GGML_CUDA_STAGE_ASYNC=0 restores synchronous): set_tensor_async returns at once, stream order kept by routing event records behind pending staged copies (RECORD jobs; waits join the worker first) and by draining a stream's pending jobs before any other direct operation on it (25 hook sites incl. the synchronous buffer ops). Measured: async == sync within noise (DSv4 74.7 vs 73.8 t/s, q35 all-pageable 213 vs 214) - because the memcpy into the ring, not the scheduler thread, was the bottleneck: nsys over 4 x 512-token passes showed 219 GB copied in 33 s with the copy engines busy only 5.3 s (GPU idle 84%), i.e. ~6 GB/s memcpy from the file mapping in EVERY pass (not a first-touch effect). Windows trims file-mapped pages from the working set between passes and soft-faults them back. FIX: VirtualLock the region that cudaHostRegister rejected (after raising the working-set quota; PSHARD_VIRTUALLOCK=0 disables): DSv4 prompt 75 -> 84 t/s (45 GB locked in 13.6 s at load), q35 all-pageable 213 -> 236 t/s; outputs byte-identical. That is the CEILING of staging on this box: the pass is now host-DRAM-bound (source read + ring write + two DMA reads ~177 GB per pass at ~45 GB/s = ~4 s), so ~84 t/s vs ~137 t/s fully pinned. Raising the working-set quota BEFORE cudaHostRegister does not lift the 60 GB pin ceiling (driver limit; it also drove the driver into a sticky out-of-memory) - reverted. Fully pinned performance needs more lockable memory (RAM/driver), not code. ASYNC v2 (2026-09-02, review follow-up): the first async cut drained a stream before every direct operation on it, and the scheduler issues two right after each staged push (padding memset, pinned KV upload) - so the worker was serialized anyway. v2 QUEUES those direct ops (JOB_H2D / JOB_D2H / JOB_MEMSET) behind the pending staged jobs of that stream in the worker FIFO, keeps event records deferred (JOB_RECORD), joins on event wait/synchronize/free, and drains at teardown (backend free, event free, buffer free, buffer cpy); a global atomic keeps every hook a no-op until a worker exists. Measured: DSv4 async 83.6 vs sync 82.9 t/s (md5 0ba05b4baf58 both), q35 all-pageable 249.6 t/s (md5 ecb043a95a14) - i.e. still the same ceiling, and nsys shows WHY: during the staged shard the copy engine is busy only ~35% of the time with kernels ~5%, moving ~14.5 GB/s, and GGML_CUDA_STAGE_THREADS=4/8/16 gives 83.4/83.6/82.2 t/s. The ring fill is bound by host DRAM (memcpy read + ring write + DMA read = ~44 GB/s, the same rate the pinned shard alone DMAs at), not by threads, not by serialization. Async stays the default (it is correct and costs nothing) but on this box staging tops out at ~84 t/s for DSv4 prefill; fully pinned would be ~137. Re-review of the final diff (12 agents) confirmed ONE limitation, accepted as-is: the synchronous buffer ops (buffer get/set/memset/clear on cudaStreamPerThread) drain the WHOLE device FIFO, so an ALTERNATION plan whose CPU split reads a GPU activation via buffer_get_tensor blocks behind the in-flight multi-GB staged prefetch - async == sync for s2/s4-style plans with unlockable weights too (it cannot make them WORSE than sync, and CPU-FFN compute shares DRAM with the ring fill anyway, so the theoretical overlap is small). The fix, if a config ever needs it, is an overlap check: drain only queued jobs whose src/dst ranges intersect the requested tensor. Refuted by the skeptics: working-set quota rollback, the RING_MB=0 measurement lever, VirtualUnlock scoping, --mlock interaction. Known-low: a detached stage worker can touch CUDA statics if the process exits mid-prefill from a signal handler. PER-MAPPING UPLOAD RATES LANDED 2026-09-02 (the "predictor blind to page-locking" follow-up): the profiler gains a Host_Pin_Ceiling probe (cudaHostRegister of 2 GiB anonymous chunks until the driver refuses, capped at 90% of RAM, --no-pin-ceiling to skip; measured 66.6 GB here, written as a "#   Host_Pin_Ceiling: NN.N GB" cpu_profile.txt header line - splice it via --sliced-only). The planner enumerates split-file sizes (gguf split.count + llama_split_path), simulates the runtime page-lock loop (greedy in split order, all-or-nothing, ceiling minus 2 GB margin for the ring/KV-shadow/load-staging pins), prices staged mappings at min(pcie, dram/3) (= 15.2 GB/s here, vs 14.5 nsys-measured), and reprices the predictor: serial weight-upload terms at the harmonic blend (DSv4: 49.4 GB pinned @45 + 47.4 staged @15.2 -> 23.0), and the prefetch-overlap term as a MIXTURE over mapping classes - a split's weights come from ONE mapping, so the pass mixes pinned-rate and staged-rate splits by the staged byte fraction; a single blended rate lets predicted compute hide the staged splits' stalls (blend predicted 117 vs 84 measured; mixture predicts 100). Lock prediction matches the runtime exactly (shard 2 staged). Same change fixes ctx.model_size to the TOTAL of all splits - the 6 MB main split of DSv4 had poisoned every file-size heuristic (switch_mb 0.1 -> 2071.8 MB/layer, head_mb 0.2 -> 3873.3; note the 0.04 head fraction now likely OVERSTATES the DSv4 head - single-file calibration). The registry fingerprint deliberately stays on the main-split size (the runtime computes it the same way; changing it strands cached registries into silent stock fallback). Validated: DSv4 tier tps bs512 117.89 -> 100.10 (measured 82.6-85.2), bs1024/2048 stay 118.5/118.6 - there predicted COMPUTE hides even staged uploads, i.e. the residual ~40% optimism at big-batch tiers is the pre-existing GPU compute model, not upload pricing; q35 registry bit-identical (single-file untouched); md5s unchanged (0ba05b4baf58 / ecb043a95a14). GOTCHA: profile stats are not in the registry fingerprint, so pre-existing registries keep stale tps until deleted/replanned (DSv4+q35 on this box replanned; backups in the session scratchpad).

12. EXPERT POOL BUILD (docs/expert-pool-design.md v1) - stages 1-2a LANDED 2026-09-03:
   e3a959c51 (-1 = skipped route in MUL_MAT_ID/ADD_ID, opt-in per tensor via
   ggml_mul_mat_id_set_allow_skip, all CUDA dispatch paths + CPU + sched sliced copy;
   test-backend-ops n_skipped cases green on CUDA0+CPU, existing suites unchanged),
   4808ee455 (EXPERT_POOL enum 5, per-tier pool fields, POOL-only registry columns,
   PSHARD_MISS_POLICY env fingerprinted only when set), 67bcbee46 (fetch-corner planner
   under forced PSHARD_STRATEGY=5: one probe, pool = budget remainder, A/B floor on
   whole-stack tiers (bs*top_k*2 >= E, the runtime sliced-copy gate) / fetch floor
   min(E,top_k*bs) slots/layer on cache tiers; routed ffn_*_exps stay in host RAM,
   everything else pins; runtime DISABLES pshard loudly on a viable POOL plan until
   the pool module lands). Verified: q35 @8000/2k plans all tiers s=87-90, registry
   round-trip + fingerprint parity, unforced planning byte-identical, runtime smoke
   md5-identical. NEXT (2b): pool module (region carve like the pinned-KV pack,
   per-layer slot maps/LRU/counters), graph emission over pool views + ids_gpu
   leaves, fetch via the pshard pre-compute split callback, warmup + token-hash gate
   vs always-fetch; then (2c) pool predictor terms + auto ladder entry; then split-op
   policies (cpu_exec/hybrid/fetch_on_2nd_miss/cpu_tail) with PPL-parity gates.
   STAGE 2B LANDED 2026-09-03 (e17d1d9bf) AND THE GATE PASSES: q35 @8000/2k forced-5
   with PSHARD_POOL_RUNTIME=1, 512-token prompt + 32 greedy -> md5 ecb043a95a14,
   byte-identical to the legacy pshard AND stock references, first end-to-end run.
   Numbers (untuned): prompt 574 t/s, decode 40.1 t/s at 17 slots/layer - the carve
   clamped the planner's 73 (file-size heuristic vs real nb[2] expert bytes, and
   scratch_need = max tier scratch + 64 MiB eats the region; q35 pool got 1239 MiB).
   Mechanism: sched input-copy overrides (persistent pool views replace transient
   dups; four prefetch opt-out sites; consume-time callback), pool module with
   cache-mode LRU slots + A/B halves on whole-stack tiers, ids split in the graph
   (mm gets remapped ids_gpu; probs/w_s/add_id/LoRA keep router ids), experts on
   shard bids (no pool instance = standard-streaming always-fetch fallback = the
   gate reference arm). Legacy paths regression-clean (unforced q35 md5-identical,
   MUL_MAT_ID suites green). OPEN for 2c+: pool region sizing should come from
   real expert bytes not the 0.85 heuristic (planner underprices s); A/B-mode
   prefill uploads are exposed (574 vs legacy ~640-680 prompt; prefetch-cb overlap
   pending); decode 40 vs auto-legacy 51.5 @8000/2k needs h(s) counters + pricing;
   counters exist but are not yet printed/ledgered; miss policies beyond fetch and
   auto-ladder entry not implemented; graph reuse across pooled-layer changes
   guarded only by the expert_pool pointer compare.
   SPLIT-OP LANDED 2026-09-03 (e45601565): miss policies cpu_exec / hybrid /
   fetch_on_2nd_miss execute via two expert-FFN chains per pooled layer (GPU over
   pool slots, CPU over the host homes in place, -1 for the other side's routes,
   one ADD at the down output). GATE: all three are TOKEN-IDENTICAL to fetch/legacy/
   stock (md5 ecb043a95a14) on the 512-token workload - the two-partial sum is exact
   since every route lives in exactly one chain, so the design's PPL-parity gate is
   met with room to spare (PPL c2048/8: fetch = cpu_exec = hybrid = 1.2619, stock
   1.2640 - the 0.17% is placement fusion, inside the 0.5% mirror gate). Decode
   t/s at 17 slots/layer: fetch 40.0 > hybrid 33.5 ~ fetch_on_2nd_miss 33.0 >
   cpu_exec 27.3 - fetch wins at ample slots as priced; CPU routes pay the CPU FFN
   + the GPU->CPU->GPU handoff barrier (today's sched runs the chains serially, the
   max() handshake of 6b/6e is not implemented). Prompt ~505 t/s under dual chains
   (allow_skip disables GLU fusion + memsets on A/B tiers) vs 564 fetch-only.
   PSHARD_MISS_POLICY selects the policy (fingerprinted when set). Bugs found by the
   gate run and fixed in the same commit: same-pass hits must be LRU-stamped before
   the fetch loop (latent in fetch mode too); an ids leaf nothing consumes is never
   allocated (q35 has no expert biases). OPEN: the tight-budget flip (planner
   choosing cpu_exec/hybrid below the fetch floors) is still manual via env; the
   predictor prices no miss policy yet; hybrid_frac comes from slice_bw/DRAM, the
   paired bench entry (11.B.10) is not built; per-layer h counters are collected but
   not printed/ledgered. Split-op review (18 agents) confirmed 3, fixed in the follow-up commit: A/B
   tiers are single-chain (every expert resident - the dual chain was a GiB-scale
   dead CPU chain per layer); build_lora_mm_id pins EVERY node it creates on the CPU
   chain (with a per-expert scale tensor the actual MUL_MAT_ID was unpinned and would
   have anchored on the shard bid reading the pool view with expert ids - q35 has no
   w_s so the gate could not see it); per-layer ids pointers are cleared on
   deactivate and the pool tensor assignment guards on active (stale pointers
   aliased rebuilt-graph tensors in mixed registries). Documented invariant: the GPU
   chain is the ADD's src[0] so its split runs first and the service writes the CPU
   ids in time. Refuted: LRU same-pass-hit (already fixed), hybrid trim order,
   fetch_on_2nd_miss counter decay.
   FOLLOW-UPS 1-5 LANDED 2026-09-03 (fd11c3593 real expert bytes, 75884f699 per-tier
   region, 85a9599e6 A/B prefetch overlap, faa4d870f counters, 69fc96936 pricing +
   PSHARD_POOL_AUTO): the planner sizes the pool from the gguf tensor table (q35:
   498 MiB/layer, 1.95 MiB/expert -> s=75 decode / 66 on A/B tiers, was 17 under
   the 0.85 heuristic + max-tier scratch); the region is re-carved per tier (pool +
   scratch constant, so decode turns the prefill scratch delta into slots);
   whole-stack tiers fill the A/B half from the sched prefetch pass on the copy
   stream (prompt 574 -> 662 t/s); log_counters at context free (library INFO:
   needs -lv 4, which the QA correctness log has) + ledger columns mean_h /
   misses_per_token; predict_tps breakdown (compute / exposed weight upload /
   other) + Zipf(alpha=0.8, PSHARD_POOL_ZIPF) miss term for cache tiers +
   PSHARD_POOL_AUTO=1 ladder entry. Gates: fetch / cpu_exec / hybrid / legacy md5
   ecb043a95a14 after every step. Calibration q35 @8000 fetch s=75: counted
   h=0.700, 96 misses/token (2.4/layer, per-layer 0.45-0.80; Zipf(1) would say
   0.80); priced 46.6 vs measured 45.8 t/s decode. HONEST LADDER OUTCOME: a fresh
   legacy auto plan @8000 (12 layers pinned) is priced 52.3 / measured 50.0 t/s
   decode and 764 prompt vs the pool's 45.8 / ~650, so with PSHARD_POOL_AUTO=1
   the ladder keeps legacy on every tier at this budget - and the model is right
   on both sides. The pool's case is tight budgets (legacy cannot pin whole
   layers) and expert-dominated models (DSv4) - not yet measured. Registry trap:
   PSHARD_MISS_POLICY is fingerprinted, PSHARD_POOL_AUTO is not (it rewrites the
   auto section). OPEN (6-8): planner picks cpu_exec/hybrid below the fetch
   floors; concurrent CPU chain (the max() handshake); hybrid bench entry,
   GROVEMOE chexps, MMQ ne0%512 padding, QA grid -s5; upstream cherry-picks
   #22181 #22331 #25048 #27024 and the #24528 admission gate (see the PR triage).
   Runtime map (agent-verified): uploads are per-tensor cudaMemcpyAsync on a dedicated copy stream, prefetch depth 1 with one full layer of real overlap, full 256-expert stack per layer for any ubatch >= 22 tokens, ~2 splits per layer; minor host syncs (one per streamed split on an idle placeholder stream; events[][] dead under pshard because n_copies == 1). Levers: GGML_CUDA_REGISTER_HOST=0 (all pageable), PSHARD_PAGELOCK_SKIP=<i> (one mapping pageable). Earlier two-point additive estimates in this note were wrong and are superseded by the trace. The loader now WARNs when a page-lock fails; a
   tier marked unviable at load should fall back to the nearest viable tier
   (decode stayed in the 512-token streaming plan: 6.3 t/s at a 2024 MiB
   budget before the margin fix); ~~joint target+draft planning (cordis v2)
   deferred until v1 shows a case where leftover budget could pin draft
   experts~~ **v2 LANDED + MEASURED 2026-09-02 (greedy leftover form)**: the
   planner's union enforcer persists the canonical union (variant header
   union_mb=; max over viable tiers of packed weights + pinned cache + compute
   scratch + 32 MiB margin), the arena is allocated at union + 64 MiB (whole
   MiB) instead of the whole budget, and a MoE draft whose experts the v1 rule
   spilled gets the leftover: leading layers' experts move to the device while
   they fit (generated partial-spill regex; runtime-only, registry unaffected;
   user -otd/-cmoed lists never rewritten; re-entry on the same params restores
   the canonical spill before re-measuring, so server sleep/wake derives the
   plan tool's budget). Measured: q35+dflash @4000 union 2475 -> arena 2539 of
   2671 MiB, pair peak 4206 MiB (4315 before), 45.8 t/s, 64 accepted - no
   experts to pin; DSv4+DSpark @12000 union 10120 -> arena 10184 of 10221 MiB,
   37 MiB left vs a 3264 MiB first DSpark expert layer -> nothing pinned. The
   layer granularity of the target plan leaves too little for any draft at
   hand: v2 buys nothing on these models, as suspected; the mechanism stays
   (it also releases the arena slack honestly). API: llama_pshard_registry_arena_bytes,
   llama_model_pshard_active. [b] ~~FNV logits-hash spec test port~~ **LANDED
   2026-09-02**: tests/test-pshard-spec.cpp (ctest, env-gated on
   LLAMA_TEST_PSHARD_SPEC_TARGET [+_DRAFT], label model). Per strategy: plan
   in-process (mirror of the plan tool), run the speculative-simple loop twice
   - DETERMINISM is the hard bar (token stream, n_accept/n_drafted and an
   FNV-1a hash over every verify step's full logits must match between the two
   runs; held bitwise for s0/s1/auto with dflash and with the MTP head) - plus
   pshard-engaged (llama_model_pshard_active) and n_accept > 0. Across
   strategies the first verify step's logits are compared value-wise (pairwise
   matrix + top-2 margins printed; tolerance per model via _TOL, token/hash
   equality via _STRICT) and the stock side runs at -fitb MVA as reference.
   FINDING: on q35 the GPU-only strategies s0 and s1 disagree at small-batch
   tiers (first-step max |dlogit| ~1.0 in the spec test; ub-16 perplexity
   2.8782 vs 2.8796 with no rollback ring) while plain bs=1 decode is
   byte-identical s0 == s1 == stock over 64 tokens and the DENSE 27B is
   bitwise identical at ub 16 too -> a MoE small-batch expert-path difference
   between two GPU-only placements; root-cause hunt (eval-callback per-tensor
   sums, first divergent op) recorded below under item 11.
   [c] stock+MTP equal-budget QA cell (stock-fallback+MTP at DEFAULT fit hit
   ~83 t/s - budget-equalized comparison pending; harness cells now do this
   at -fitb); [d] ~~forced-s3+MTP
   n_accept=0 anomaly~~ RESOLVED (070d9fb27): the draft ctx's layer-40 KV
   cache takes the pipe-shard branch (gated on model.is_pshard()) and its
   GPU k/v stay unbacked (nothing packs a non-pshard ctx) -> scratch KV ->
   garbage history whenever the plan PINNED the nextn attn; nextn layers
   now always CPU-resident; [e] ~~server spec path untested~~ **TESTED
   2026-09-02**: llama-server -np 2 -c 4096, two concurrent greedy completions
   per case, clean device per case (pshard vs stock -fitb 4000):
     MTP head:   pshard 4379 MiB, 31.6/32.5 t/s, acceptance 0.73/0.72 |
                 stock  4485 MiB, 25.4/26.5 t/s, acceptance 0.76/0.77
     dflash:     pshard 4332 MiB, 25.4/23.6 t/s, acceptance 0.55/0.49 |
                 stock  6160 MiB (2.1 GB OVER budget: the stock fit cannot
                 measure a dflash draft - "requires ctx_other" - so the draft
                 loads outside -fitb; one-budget is pshard-only), 21.8/21.9 t/s
   Both slots served in every case, no fallback, no errors under pshard.
   Harness lessons: sh's $! is an MSYS pid, native servers ignore its kill and
   taskkill by that pid hits nothing - stop llama-server BY IMAGE NAME and wait
   for VRAM to return to idle before the next case (a stale server kept serving
   the port and two rounds of "results" were its answers; a draft load once
   failed with "invalid vector subscript" only while a stale server held 9 GB);
   [f] plan-tool
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

11. **MoE small-batch parity between GPU-only strategies (OPEN, found by the spec
    certification test 2026-09-02)**. Facts, all on q35 Q4_K_M @4000 unless noted:
    - forced s0 (LAYERPIN_LAYERSTREAM) and s1 (ATTNPIN_FFNSTREAM) are each bitwise
      deterministic run-to-run (tokens, n_accept, FNV over every verify step's logits);
    - plain bs=1 greedy decode after a 346-token prompt: s0 == s1 == stock(-fitb) byte-
      identical over 64 tokens (auto = STATIC_ATTNPRIO diverges at token ~35, CPU experts);
    - DENSE Qwen3.6-27B-Q4_0 @6000: s0 == s1 == auto byte-identical at bs=1, and PPL at
      ub 16 bitwise identical (2.6951, per-chunk 3.4747) -> GPU-only streaming itself is
      consistent; the effect is MoE-only;
    - MoE PPL at ub 16 (c=512, 4 chunks, no rollback ring): s0 2.8782 vs s1 2.8796 vs
      stock 2.8862 - differs already in chunk 1;
    - spec test (dflash, 36-token prompt then 9-token verify batches): first verify step
      max |dlogit| s0-s1 1.05, s0-auto 0.61, s0-stock 0.95, s1-stock 0.55 (top-2 margins
      0.76-1.10, so token 3 flips); MTP head: 0.54 / 0.38;
    - eval-callback per-tensor SUMS (rel 1e-6), s0 vs s1: fresh 14-token batch identical
      over 2667 tensors; fresh 43-token batch identical; 3 x 16-token KV-warm decodes
      identical over 8001 tensors (LLAMA_EVAL_CB_CHUNK=16);
    - tier switch (36-token prompt at tier 512, then a 9-token batch at tier 16) identical
      over 5334 tensors; 346 tokens by 16 (KV to 346) identical over 58674; both again with
      ALL logits (multi-output batches, LLAMA_EVAL_CB_ALL_LOGITS=1) identical incl. the
      result_output sums -> the callback hunt sees no difference anywhere;
    - ROOT CAUSE (CLASSIFIED, not a defect): an eval callback materializes every node and
      thereby DISABLES CUDA op fusion, which is exactly what differs. With
      GGML_CUDA_DISABLE_FUSION=1 the ub-16 PPL is bitwise identical, s0 == s1 == 2.8761
      (per-chunk 3.3886), stock unchanged at 2.8862. Placement decides the scheduler
      splits: LAYERSTREAM keeps a layer in one split (fused add+norm/GLU intact),
      ATTNPIN_FFNSTREAM cuts attention|FFN inside the layer and breaks the fusions there;
      the rounding difference flips MoE expert-routing near-ties and moves logits by O(1).
      Dense has no routing to amplify (bitwise identical), and stock's CPU experts put the
      same cut inside the layer (hence stock is closer to s1: 0.55 vs 0.95). Verdict: the
      speculative-test bars are right as landed (determinism hard, cross-strategy parity
      reported); do not chase cross-placement token equality on MoE.
    Reproduce: `PSHARD_STRATEGY=0|1 llama-pshard-plan-params -m q35 -c 512 -mva 4000 -b 16
    -ub 16` then `llama-perplexity ... -c 512 --chunks 4 -b 16 -ub 16 -pshard -mva 4000`;
    the spec test with LLAMA_TEST_PSHARD_SPEC_STRATS=0,1 prints the matrix. Note: tensor
    sums cannot resolve a ~1.0 delta in one of 248k logits (sum magnitude ~1e6, rel 1e-6
    = 1.0 absolute) - an element-wise comparison of the printed elements normalized by
    max|x| (memory item 12) is the tool for non-fusion questions; fusion questions need GGML_CUDA_DISABLE_FUSION=1 on the real workload.

## Contamination note (2026-08-30)

Two "stopped" harness runs kept running detached and overlapped each other and
ad-hoc tests. Perf numbers from that window are invalid; correctness outcomes
(DSv4 runs + output text, loader behavior classes) and planner PREDICTION numbers
(pure compute, no GPU contention sensitivity) remain valid.
