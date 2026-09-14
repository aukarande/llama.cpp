# pshard optimizations done

The pshard branch runs MoE models that do not fit one GPU by planning a per-tier weight placement (pinned, streamed, expert pool) from a measured machine profile and executing it through the ggml scheduler.

All q35 numbers are Qwen3.6-35B-A3B (UD-Q4_K_M) at ctx 2048 unless a context is given; DSv4 is DeepSeek-V4-Flash; `@N` is the device budget in MiB; decode at `-n 128` unless stated.

## 1. Performance optimizations

| date | commit | area | what | measured effect |
|---|---|---|---|---|
| 2026-08-25 | 447deb3f2 | scheduler, runtime | streamed-weight transport: page-locked mmap, double buffer, sliced-expert prefetch | q35 @8192 ATTNPIN_FFNSTREAM decode 1.02 -> 31.9 t/s; q8d LAYERSTREAM 0.71 -> 2.17, ATTNPIN_FFNSTREAM 2.84 vs stock 2.12; upload ~13 -> ~45 GB/s |
| 2026-08-25 | 639d3bc44 | planner, scheduler | DYNAMIC_FFN_ALTERNATE strategy, prefetch scan-ahead, sliced-expert hardening | q8d @8192 ATTNPIN_FFNSTREAM 2.89 vs stock 2.12 t/s (+36%); q35 auto @8192 decode 54.9 vs stock 52.1, prompt 839 vs 123 |
| 2026-08-28 | 9b66dadc6 | planner | tps predictor prices the transport the runtime actually uses | q35 ctx 16k @4000 auto decode 14.50 -> 34.97 t/s (+141%); ATTNPIN @4000 predicted 2.4 -> 33.3 (measured 29.3) |
| 2026-08-30 | 13cce3042 + 7c6a48bbe | planner | ids-cross plan attribute: router on the GPU so ALTERNATE's consume-time sliced upload sees the routed ids (follow-up restores the dropped serialization) | q35 ctx 16k @2000 forced-s4 decode 7.65 -> 19.99 t/s (+161%), prompt unchanged |
| 2026-08-30 | c4ae5e311 | planner | tier-switch cost charged in prefill ubatch selection | |
| 2026-08-30 | 6513df750 | planner | compute floor for unbenchmarked quantized CPU matmuls at M >= 32 rows | DSv4 auto prompt 21.7 -> 46.6 t/s (+115%); tier-0 predicted 10.89 vs measured 10.87 |
| 2026-08-31 | c018e057e | planner, profiler | selector-gap fixes: bs=1 hi_attn fresh bound, sliced uploads priced at the profiled chunk-size curve | q35 ctx 16k @2000 auto 19.9 -> 30.4 t/s (+53%); q35 @8000 45.0 -> 57.0 (+27%); oss ctx 16k @2000 18.5 -> 29.0 (+58%) |
| 2026-09-01 | ed7136c04 | runtime, planner | per-context memory gating, MTP head pin priority, union head-CPU lever | q35 MTP n_max=2 @4000 52.8 -> 56.9 t/s, @12000 72.1 -> 83.0 |
| 2026-09-02 | e3a959c51 | CUDA, CPU backend | -1 = skipped route in MUL_MAT_ID / ADD_ID (pool split-execution groundwork) | |
| 2026-09-02 | 2b03f0ada | CUDA backend | staging ring for pageable host->device uploads (8 x 64 MiB pinned ring, memcpy threads) | DSv4 @12000 prefill 28.3 -> 75.0 t/s; q35 @4000 all-pageable prompt 68 -> 215 (pinned 678); pageable shard 4.1 -> 34.8 GB/s |
| 2026-09-02 | 006e93642 | CUDA backend | asynchronous staging worker; VirtualLock fallback for unregisterable mmap regions | DSv4 @12000 prompt 75 -> 84 t/s (VirtualLock); async vs sync 83.6 vs 82.9; q35 all-pageable prompt 213 -> 250 |
| 2026-09-03 | 4808ee455 | planner | EXPERT_POOL strategy plumbing (enum, per-tier pool fields, registry columns), inert | |
| 2026-09-03 | 67bcbee46 | planner | expert-pool fetch-corner planner (forced mode): closed-form plan, floors, placement, runtime guard | forced-5 plans all 5 q35 tiers @8000 (s=87-90 slots/layer) |
| 2026-09-03 | e17d1d9bf | expert pool, scheduler | expert-pool runtime v1: fetch policy, per-layer LRU slots, sched input-copy overrides, graph ids split | q35 @8000 forced-5 32-token md5 identical to legacy and stock; prompt 574 / decode 40 t/s at 17 slots/layer |
| 2026-09-03 | e45601565 | expert pool | split-op miss policies cpu_exec / hybrid / fetch_on_2nd_miss: GPU chain over pool slots, CPU chain over host homes, one ADD at the down output | q35 @8000 s=17: fetch 40.0, hybrid 33.5, fetch_on_2nd_miss 33.0, cpu_exec 27.3 t/s; token-identical, PPL 1.2619 |
| 2026-09-03 | 75884f699 | expert pool | pool region sized per tier (pool + scratch constant across tiers) | q35 @8000 decode tier 5467 MiB / 75 slots, 45.8 t/s |
| 2026-09-03 | 85a9599e6 | expert pool, scheduler | A/B prefetch overlap: whole-stack tiers fill the layer half from the sched prefetch pass | q35 A/B tier prompt 574 -> 662 t/s; decode unchanged |
| 2026-09-03 | 69fc96936 | planner | pool pricing: predictor breakdown, Zipf miss term for cache tiers, ladder entry | q35 @8000 pool priced 46.6 vs measured 45.8 t/s; legacy 52.3 / 50.0; ladder keeps legacy |
| 2026-09-03 | ee04cd11f | planner | planner picks the miss policy per cache tier; slot cost from the all-layer expert sum | priced / measured q35 @2700 fetch 38.1 / 37.4 t/s; @8000 fetch 47.6 / 44.7, hybrid 35.0 / 34.4, cpu_exec 27.7 / 26.7 |
| 2026-09-03 | bb8cc8d16 | CPU backend | cherry-pick upstream #22181 #22331 #25048 #27024 (CPU miss-chain kernels) | MUL_MAT_ID 885/885, MUL_MAT 1193/1193 pass; CPU-route md5 unchanged |
| 2026-09-03 | 70a4d503f | expert pool | cpu_admit miss policy: misses compute on the CPU this pass, rows upload into LRU victims off the critical path | serial chains q35 @8000 cpu_admit 40.0 vs fetch 44.2 t/s (32 tok), 51.6 vs ~50 (256 tok); DSv4 @12000 13.5 vs 13.4 |
| 2026-09-03 | d8c388569 | scheduler | CPU/GPU split overlap for the pool's dual expert chains (CPU split inputs copied before the GPU split launch) | q35 @8000 cpu_admit 32.6 -> 38.0, hybrid 36.5 -> 39.8, cpu_exec 26.9 -> 30.0 t/s; DSv4 @12000 cpu_admit 8.7 -> 11.8; grid hybrid +15..19%, cpu_admit +6..27% |
| 2026-09-03 | fcebfc614 | expert pool | prefetch predictor: layer l+k's predicted experts uploaded on the copy stream during layer l, capped at N per layer; default off | DSv4 @12000 N=1 13.4 -> 14.3 t/s (h 0.53 -> 0.63), uncapped 13.4 -> 11.9; q35 @8000 N=5 -n 256 49.1 -> 50.6 (h 0.82 -> 0.93) |
| 2026-09-04 | df120c6b1 | expert pool | output head on the GPU in the pool corner, MTP head pinned whole, prefetch on verify batches, region carved from the plan's own bounds | q35 @8000 fetch -n 32 44.7 -> 66.1 t/s, -n 256 49.8 -> 75.4 (legacy 50.0, stock 48.9); MTP 41.9 -> 77.2; DSv4 @12000 13.4 -> 14.0; draft pass 9.5 -> 0.3-0.7 ms |
| 2026-09-04 | fe41c118b | CUDA backend | persistent memcpy pool for the staging ring: every pageable upload copied by all cores | DSv4 @12000 fetch 14.6 -> 15.4-16.0 t/s (half ring), 10.6 -> 12.2 (all ring); ring miss 0.83 -> ~0.6 ms (DMA 0.42) |
| 2026-09-04 | 86585da29 | CUDA backend | memcpy pool threads above normal priority, default capped at 8 (stall outliers were OS preemption) | DSv4 @12000 fetch+pred outliers 10.8 / 5.2 t/s gone; 8 threads 16.06-16.22 over 4 runs (1 thread 14.96-15.55) |
| 2026-09-11 | f3bdcf32e | expert pool, CUDA, scheduler | pool decode without copy-engine transfers (sum of the five rows below); tier-switch perf counter | q35 @8000 hybrid 66.3 -> 88.9 t/s, fetch 77.0 -> 96.1, cpu_admit 63.2 -> 77.6, fetch+pred 75.5 -> 80.8; GPU idle per layer 152 -> 55 us |
| 2026-09-11 | f3bdcf32e | expert pool, scheduler | dead CPU chain skipped: a CPU split whose routes are all -1 is not computed, consumer copies zero-filled in stream order | q35 @8000 hybrid 66.26 -> 72.04, cpu_admit 62.87 -> 65.65 t/s; fetch 78.14 -> 77.49 |
| 2026-09-11 | f3bdcf32e | CUDA backend | kernel copies for per-token transfers when the host side is device-mapped pinned memory under the cap (K1) | q35 @8000 fetch 76.99 -> 88.42, hybrid 72.38 -> 74.97, fetch+pred 73.47 -> 77.03 t/s |
| 2026-09-11 | f3bdcf32e | scheduler, CUDA backend | event-fenced async CPU-split downloads on the producer stream; kernel zero-fill replaces cudaMemsetAsync (K2/K3) | q35 @8000 hybrid 74.97 -> 83.82, cpu_admit 67.44 -> 72.39 t/s; per-layer idle 94.5 -> 63.6 us |
| 2026-09-11 | f3bdcf32e | expert pool, scheduler | batched segment upload kernel (one launch per layer, 176 blocks per 704 KB segment), no per-layer drain, async user inputs (K4/K4b) | q35 @8000 hybrid 83.82 -> 88.82, fetch 88.78 -> 95.77, cpu_admit 72.39 -> 75.97 t/s; 1 expert in 50 us (37 GB/s); traced idle 7.9 -> 3.6 ms/token |
| 2026-09-11 | f3bdcf32e | scheduler | async host-copy paths switched on by the pool's register_sched only; legacy tiers keep the old paths | q35 @8000 legacy auto new vs old paths 57.73 vs 58.32 t/s; pool gates 88.9 / 96.1 / 77.6 / 80.8 |
| 2026-09-12 | 587b88f7a | runtime | tier switch relocates resident tensors on the device (dependency-ordered sweep, 32 MiB bounce for self-overlap) instead of re-uploading | q35 ctx 256k @14500 first-token switch 154.9 -> 46.2 ms (5746 MiB moved); ctx 8192 @8000 45.1 -> 36.9 ms, @14500 44.7 -> 33.8; outputs byte-identical |
| 2026-09-13 | uncommitted | expert pool, graph | router ids land in host memory (kernel-path ggml_cpy into the pool's pinned staging, serve() polls -1 sentinels instead of draining); generic hoist pass moves route-independent nodes (shared experts) ahead of the routed boundary | q35 @8000 pool 92.7-95.9 -> 99.16 t/s (+3.4%); hybrid / fetch / cpu_admit / fetch+pred 93.2 / 95.9 / 76.8 / 79.9 -> 96.5 / 99.6 / 78.9 / 82.2; per-layer idle 42.9 -> 13.4 us; 280 nodes hoisted over 40 boundaries; texts bit-identical; legacy 58.85 unchanged |
| 2026-09-14 | uncommitted | runtime, scheduler, CUDA | transfer mode follows the active tier: kernel copies (cap = the profile's crossover) + async scheduler paths on any viable plan; prefetch copy backends keep the copy engine (per-backend deny list); sliced-expert uploads of a streamed MoE layer as one copy-segments launch | q35 @8000 s3 59.6 -> 66.3 t/s (+11%), s4 53.0 -> 58.8 (+11%), s1 36.4 -> 38.4 (+5.6%), s2 20.95 -> 21.1, s0 3.12 -> 3.145, pool fetch 98.6 unchanged; texts and gates identical |

## 2. Measured and rejected

- 626a5a595 probationary admission gate (admit after N): q35 @8000 -n 256 h 0.822 -> 0.815 -> 0.809 for N=1/2/4; DSv4 @12000 13.3 -> 13.5 t/s; removed in 1ce8f5b2a
- fab14ab40 warm start from the prompt histogram + per-layer slot allocation: q35 @8000 -n 32 h 0.702 -> 0.731 but 44.9 -> 43.0 t/s; DSv4 @12000 10.65 -> 10.39; per-layer alloc <= 0.4 h points; landed off
- 1ce8f5b2a prompt-end LRU warm start + popularity-ranked allocation: q35 @8000 warm=8+alloc 80.2-81.3 vs 79.2-80.3 t/s (h 0.798 vs 0.787), warm alone 77.0; grid: never beats fetch+pred (q35 mean -3%, DSv4 +0/+1.5%); knobs removed in f549c0419
- fcebfc614 prefetch predictor as the default: grid q35 fetch -2% in 17 of 18 pairs, DSv4 +1..+2.6%, MTP +0.4..+1.9%; stays off, selectable as fetch+pred
- a7ec78ac9 16-thread CPU chain on DSv4: cpu_exec +13%, GPU-path policies -4..-5%; default thread count kept (a467561d7 dropped the -t 16 cells)
- 86585da29 16 memcpy copiers: identical runs spread 16.3 -> 10.8 / 5.2 t/s; capped at 8 with raised priority (16.06-16.22 stable)
- 52a8e7972 staging ring with shard 2 pinned: ring off 14.69 vs on 14.58 t/s, identical misses; the ring buys nothing once the mappings are pinned
- 677b0d599 pinning host memory past shard 2: registrations granted to 64 GB but any pin beyond 49.4 GB poisons the device (cudaMalloc OOM, VirtualLock 1450); practical pinned budget ~50 GB
- ca622d5b5 defer-prefetch transfer ordering: q8d s4 download wait 172 -> 1 ms/token but launch 221 -> 362 ms, net +2.7%; 8 MoE s4 cells -1.1..+1.7% (noise); removed in 37c7ebc9b
- 30d3f4ed1 slot carve-out: q8d @8000 with the fence removed entirely 437 vs 447 ms/token, nothing recoverable behind it; deleted
- 13cce3042 router on the CPU for the ALTERNATE (s4) tier: the consume-time sliced upload cannot see the routed ids and streams every expert; q35 ctx 16k @2000 7.65 vs 19.99 t/s with the router on the GPU
- not landed pinned staging arena for per-token copies (pinned + DMA): 32-byte pageable copies became DMA transfers with an engine switch; q35 @8000 fetch+pred 76.03 -> 68.71 t/s (-9%)
- not landed event-fenced DMA downloads for CPU splits: q35 @8000 hybrid 67.21 -> 63.44 t/s; the landed K2 uses a copy kernel behind the same fence
- not landed eager kernel launch for splits under 8 nodes: q35 @8000 hybrid 83.82 -> 78.31, fetch 88.78 -> 86.12 t/s
- not landed hits-first device slot map (hits chain launched before the host decides the misses): q35 @8000 hybrid 88.82 -> 79.04, fetch 95.77 -> 78.79, cpu_admit 75.97 -> 73.58 t/s; a copy kernel beside the hits chain slows mul_mat_vec_q 9 -> 60-114 us
- not landed first K4 cut at 22 blocks per segment: 16 GB/s, hybrid 81.05 / fetch 85.29 t/s; 176 blocks per segment (37 GB/s) landed instead
- not landed per-model shared-expert mid_hook in build_moe_ffn: q35 @8000 greedy 94.77 -> 97.61 t/s (+3.0%) but the fused-add grouping changed the text; replaced by the generic hoist pass (uncommitted, bit-identical)
- not landed transfer mode without the prefetch-backend deny list and the batched sliced upload: s3 +15%, s4 +1%, s1 -4%, s2 -4.5%; the two refinements landed instead (uncommitted)

## 3. Correctness and robustness fixes

- dd71705ee compile fixes: the branch builds against ToT
- a5e1053ea end-to-end plan -> infer works on ToT (q35 @4000 smoke: prompt 11.9 / decode 6.8 t/s)
- 732a90892 probe_reserve consumption restored in sched_reserve (QA F1); q35 @8192 decode 0.82 -> 46.9 t/s
- 1eb290015 state IO host access, switch fencing, initial-apply force upload (QA F4a/b/c)
- 9fa8c3358 decode corruption on streamed sub-layer strategies (QA F2); all 5 forced strategies q35 @8192 match stock at temp 0
- 456e94b54 unmissable stock-fallback warning; fallback packer fixed placements (QA F4e/f); q35 @8192 53.7 t/s, correct output
- 77f5a2bc9 review sweep: stale build-time views, CPU-enqueue fence, packer reserve pass, gate scope; q35 auto decode 55.4 -> 56.4 t/s with rope/rms fusions restored
- d5473c784 per-tier overlap plan attribute with budget-aware fallback; q8d @8192 ATTNPIN tier 0 viable via overlap=0 (8163.6 vs 8253.8 MiB)
- 986dd087d tiers whose runtime packing overshoots the buffer are degraded; planner packing margin; q35 forced LAYERSTREAM @2000 crash -> 24.85 t/s
- bc14b2afe apply the plan of the tier that executes, not the total-batch tier; oss ctx 16k @4000 auto prompt 83.5 -> 6749.9 t/s, decode 12.2 -> 27.0
- 787ddf869 MLA cache widths, PCIe prefetch floor, QA schema; DSv4-Flash 92 GB plans and runs on 16 GB (decode 7.40 t/s)
- b48ff5760 deterministic registry variant selection (kills the stale-plan hazard class)
- c610cc3c2 switch cost evaluated pairwise between the plans that actually swap
- 9a273944d dual-cache (DSA) pipe shards exposed; DSv4 streamed-attention crash -> 21.7 prompt / 9.9 decode t/s
- 146e03fa8 dedicated slot regions (opt-in, +0.5-1 GB VRAM) + reserve_n_size caller-array overrun; 45602 slot-vs-activation overlaps -> 0; teardown heap corruption fixed
- ddb019848 pshard page-locked mmap regions unregistered on free; 10968.7 MiB released on teardown
- 24237a489 switch uploads bypass the stale write-cells filter (F4d)
- 3c8e0a48c draft-context guard; nextn head planned when load_mtp (the decode rate claimed in that commit was a silent stock fallback)
- 12b3f837c speculative / MTP decoding end to end; q35 @4000 no-spec 45.4 -> MTP n_max=2 52.8 t/s (+16%, accept 78.3%)
- 070d9fb27 nextn layers always CPU-resident (draft KV needs backed mirrors); @12000 accept 114/163 (was n_accept=0)
- 4fadc725f plan-time canonical-union accounting + registry-free probes; q8d ctx 16k @2000 s1 STOCK_FALLBACK -> runs, 6 viable tiers (was 2)
- 8afa86ccf CUDA fusion range check sees sched input copies (certifies fused GLU); PPL fusion on vs off 1.2619 both; 114 aliasing pairs per decode pass
- 1dd7582c4 cache loader sets delegate-compute to the active plan strategy; q35 s3 from cache 46.5-47.1 t/s
- 2d368c4ed one-budget reserve for spec contexts, compute-aware union enforcer, scheduler-rebuild re-reserve; q35+dflash @4000 0/576 accepted at 9067 MiB -> 121/219 at 4315 MiB, 50.5 t/s; MTP 5063 -> 4367 MiB
- 82f78675f DSv4 compressed KV cache refused loudly; pipe shards exposed; DSpark reserve model; reserve self-check -11 MiB (was +571)
- b508ae660 DeepSeek-V4 compressed KV cache support (three root causes); 97 GB @12000 byte-identical to stock CPU, PPL 2.5408 vs 2.5443
- 319bbc6b5 DSv4 compressor state budgeted outside the arena (11.64 MiB)
- 791abc66c switch-cost structural pins, one-budget v2 (leftover -> draft experts), spec certification test, server spec path; q35+dflash @4000 pair peak 4315 -> 4206 MiB; server MTP 31.6 / 32.5 vs stock 25.4 / 26.5 t/s
- 1783270e7 expert-pool review fixes: flag slot, forced fallback, guard ordering
- 3f87889cd expert-pool runtime review: active gating, epoch-keyed reuse, async-safe buffers, floor guards; gate byte-identical
- f6174db7c split-op review: single chain on A/B tiers, whole CPU chain pinned, stale ids cleared; CPU-route prompt ~505 -> ~565 t/s
- fd11c3593 pool sizing from real expert bytes; budget-sized arena for POOL registries; q35 @8000 region 1239 -> 4811 MiB (17 -> 66 slots), decode 40.3 -> 46.4 t/s
- d9896d6ff pool region honesty: per-tier charge, optional A/B pair, measured scratch, re-registered views, decode-only counters; q35 @2700 fetch s=14 37.4 t/s / 653 prompt (was hybrid s=7 30.1 / 187)
- 7f788cf7b pool participation gated on the down tensor, not the layer (GroveMoE chunk experts)
- 9b6d6488d a reserve that leaves the arena is an unviable tier, not a spill; DSv4 @14500 decode after a 4k prompt 5.2 -> 16.5 t/s, sched buffer 19178 -> 14489 MiB
- 4e6775281 llama-bench passes n_draft=0 to llama_pshard_registry_create (compile fix)
- e3eefc566 even device split when every device reports 0 free bytes (DSv4 + DSpark ctx 8192 threw 'invalid vector subscript')
- a1a7a4881 runtime reserve failure lands a viable tier, ubatch follows it, MTP head-on-CPU arena charge (n_vocab x 128 x 6 B), fallback WARNs
- 10a87cc1d warm start seeds only while the pool owns its region (q35 @4000 4k WARM=8 garbage -> hash equals base); sched re-reserve WARNs on arena overflow
- 93b391904 three grid defects: sched re-reserve spill (DSv4 hc_head tail pinned with the head, overflow refused), MTP reserve re-measured post-fit, plan-time pool tier bound; DSv4+DSpark @8000 +1088 MiB spill gone; MTP reserve 213 vs 212.5 MiB used
- 9cc5cfe7c MTP need probed over the registry tiers' placements; load-time generator pins the MTP layer in pool tiers; grid compare tool; q35 MTP pool cells FALLBACK -> run; DSpark poolauto 4k prefill 122 -> 611 t/s; 290 of 306 rerun cells OK

## 4. Measurement, planning inputs and tooling

- 632f3131b `--fit-budget` (`-fitb`) = device memory for weights + KV + compute; stock q35 @4000 fitted 3443 -> 3907 MiB, decode tie 45.3 / 45.3 t/s at equal budget
- 66e76b1c2 + 80e52df13 perf runs at defaults with `--ignore-eos`; stock decode at tight budgets had been understated 2x (q35 @4000 21 -> 44 t/s)
- 0f5411c7a + d9cbed44d 256-token decode window (32-token window distorted decode up to 27%); PPL gate mirrors the executed config (5-digit parity on oss ctx 16k)
- e4172f033 + 417be5563 + b4ffabd5c + f27fc8e31 stock-golden harness, reference ledger (144 configs), parity gate matched on shape and placement (6 oss cells re-adjudicated, parity <= 0.0007%), 16k ledger refresh (60 of 72 improved)
- 55cff3b23 + 2080a51bd + 834341c26 + 7fcd9e136 + f83d1a5ee + 8908b879b ledger classes: resume mode, STOCK_UNAVAILABLE, stock rows over nominal budget labelled (up to 6.6x at 16k), gpt-oss PPL informational, 2 budgets
- 38c35195d + 371d99dab allocation range dump (45602 overlaps found); per-callsite sync and copy attribution (q8d s0 387 ms/token in writeback sync, s4 172 ms/token in the D2H fallback copy); both removed later (cc0c8d96d, f549c0419)
- 85d48d726 planner trusts a benchmark entry's bandwidth only if that entry ran memory-bound; q35 ctx 16k tier-0 predicted / measured @2000 12.1 / 11.2 (was 5.5), @4000 15.6 / 14.7 (was 8.0)
- 4e5700daa + 49a1ff9f7 page-lock warning; GGML_CUDA_REGISTER_HOST=0 and PSHARD_PAGELOCK_SKIP measurement levers; q35 pageable vs locked prompt 638 -> 189, decode 2.33 -> 0.76 t/s; DSv4 shard 3 pageable = 12.1 s of prompt; levers removed (da79d12f0)
- e6929caa5 DSv4 prefill attributed with Nsight: pinned copies 38.8 GB/s, pageable shard 4.1 GB/s = 10.7 of 12 s
- d23449fab per-mapping upload rates in the predictor (page-lock loop against Host_Pin_Ceiling, staged = min(pcie, dram/3)); DSv4 bs512 predicted 117.9 -> 100.1 vs 82.6-85.2 measured; switch_mb 0.1 -> 2071.8 MB/layer; pin ceiling 67.4 GB
- 40170cbb8 upload-path experiment: DSv4 misses by DMA vs staging ring vs driver pageable on the identical token stream; ring worth ~6%; pinning shard 3 is the lever
- faa4d870f pool counters: hit rate + misses/token at context free, ledger columns mean_h / misses_per_token; q35 @8000 fetch s=75 h 0.700, 96 misses/token
- 9053f024b prefetch predictor scored against the real routing; recall / miss coverage q35 @8000 k=1 0.81 / 0.72, DSv4 @12000 0.66 / 0.54
- 5f5ec2bfe pool pricing: per-model CPU expert cost, overlap-aware CPU-route terms, expert weight count from the gguf scan; priced / measured q35 @8000 fetch 53.0 / 45.5, hybrid 39.5 / 39.8, cpu_admit 37.1 / 38.0 t/s
- e4cc7f5e5 pool pricing re-fit from scheduler phase timing (handoff 0.04, service 0.10 ms/layer); fetch 43.7 / 45.5, hybrid 43.7 / 39.8, cpu_admit 40.7 / 38.0 t/s
- e56cfa6e0 hit-rate prior at the steady state (Zipf alpha 0.95, no LRU shortfall); q35 @8000 -n 256 h(86) 0.822, decode 44.7 -> 49.8 t/s vs legacy 50.0
- 09bfa60d0 + 4cac17181 + 097945237 128-token reference table; planner pick finding (t_serve priced 0.10 vs 0.03 ms measured); DSv4 legacy 10.7 t/s at 128 tokens (pool +25%); the 32-token gate reports a cold pool
- ccc7021c5 + c6912f1a7 + 0faeb745c perf grid runner: stock / legacy s0-s4 / pool policies / predictor / auto ladder, q35 + DSv4 (+MTP, +DSpark), budgets 4000 / 8000 / full, prompts 512 and 4k; 371 cells, 52 gates, 14 PPL mirrors
- a7aaed04c + 6c8aef469 + e3eefc566 + 1287f4f10 GPU clocks locked at grid start (QA_GPC_MHZ, default 2100), refusal on a non-idle GPU, idle-context whitelist, clocks in the ledger header
- 2a4e242f0 + aebf1094b fixed sampling seed for perf cells (5 of 239 cells were degenerate; DSv4 degenerate 22.3 vs prose 14.0 t/s); non-repetitive prompts (repeated shingles 41% -> 0, 4k 67% -> 1.5%)
- ddb299545 + 4bc546cd0 grid results: pool / stock decode q35 4000/512 61.1 / 47.6, 8000/512 76.6 / 56.5, full 84.0 / 83.7; DSv4 full 18.1 / 11.2 (legacy 12.2); q35 MTP 44.8 vs 25.7 steps/s; PPL within 0.004 (q35) / 0.015 (DSv4); planner fetch under-priced 11-31% q35 / 44% DSv4
- 5f22dead9 reboot analysis: bugcheck-0 hardware resets, not the code
- 6bc6820a6 grid rerun narrative: fix verification per cell, the two-generator defect, head-home A/B, machine-state finding with quiet repeats
- cc0c8d96d nine closed-investigation diagnostics removed (GGML_SCHED_NO_SLICED, TRACK, HASH, HASH_IN, TRACE_SRC, HANDOFF_DUMP, DUMP_ALLOC, GGML_CUDA_DEBUG_MM, PSHARD_DEBUG_LAYER) and the PSHARD_POOL_LRU_C pricing term
- da79d12f0 + 5ba5f2f0e + 804fb68cf PSHARD_VERIFY_PRELOAD readback, PSHARD_VIRTUALLOCK, PSHARD_PAGELOCK_SKIP, PSHARD_FORCE_PREFILL_UB removed
- 6abaa2f9f scratch error logs dropped
- a94e88b02 machine profile schema 2: fingerprint, kernel-copy upload curves, pool serve/split latencies, copy crossover, staged rate; predictor parses them; RTX 5070 Ti: segment kernel 23.0 GB/s loaded / 45.3 idle, DMA sliced 28.9, staged 16.1 GB/s, serve 17.3 us, split 18.4 us, engine switch 24-31 us
- c7c841966 planner prices the pool from the profile; 25 / 45 / 150 fallbacks deleted; a pool tier without its inputs is refused; q35 @8000 predicted fetch 61.7 -> 87.7 vs 96.6 measured (-36% -> -9%), hybrid 62.8 -> 80.0 vs 94.4; pool-auto pick hybrid -> fetch
- 8bbfc347f switch-cost estimator and pool geometry from the gguf tensor table (file-size fractions and the 25 GB/s fallback gone); estimate error +20..+75% -> within 15% (ctx 8192 @8000 69.0 -> 44.7 vs 39.4 ms); tier picks unchanged
- 28c3ba0f8 measured Zipf exponent: per-layer route histograms in every pool run, split-half fit, `<model>.pshard_workload` sidecar accumulating across runs, plan-time CPU probe for a model that never ran; q35 alpha 0.95 -> 0.975 (1.02 over 4 runs), model h(81) 0.80 vs counted 0.81; decode unchanged
- f549c0419 machine fingerprint: refusal without this machine's schema-2 profile, registry variants carry the machine hash; sixteen env back doors removed (PROFILE_UNCHECKED, POOL_ZIPF, CPU_GBS/GFLOPS, KERNEL_COPY(_MAX_MB), STAGE_*, SKIP_DEAD, WARM/ALLOC, ASYNC_HANDOFF, fusion levers, SCHED_TIMING, POOL_AUTO/RUNTIME); gates unchanged (hybrid 94.1 / fetch 96.3 / cpu_admit 77.7 / fetch+pred 80.3 t/s)
- f3bdcf32e tier-switch perf counter (switch = T ms / N, eval excl. switch); finding: the legacy `-n 128` deficit is the one-time prefill -> decode switch (45 ms at 4k, 155 ms at ctx 256k @14500), steady state within 0.7% of placement-matched stock; record C:/Aditya/grid-results/adhoc-q35-cpusample-20260908/RESULTS.md
- trace hybrid decode per-layer census: idle A 68 us (ids read + sync) + C 65 us (join sync + H2D + relaunch), promoted layers +85 us; GPU idle ~5.4 ms/token (36%); CPU chain ~0.7 ms/token, hidden; record C:/Aditya/grid-results/adhoc-q35-hybrid-trace-20260910/RESULTS.md
- microbenchmark WDDM engine transitions: kernel -> DMA D2H 8 KB -> sync 43.3 us vs copy kernel 11.6 us; kernel -> DMA H2D 64 KB -> kernel 104.6 us vs zero-copy kernel 13.3 us; per-layer skeleton 220 -> 72.6 us; record C:/Aditya/grid-results/adhoc-pool-gate-20260910/RESULTS.md
- trace per-token host loop: period 9.45 ms, GPU busy 80%, inter-token idle ~1.7% real (the 200 us log wake is an nsys artifact), intra-token idle 12.9%; token log to NUL vs `--log-disable` within 1%; not a lever; record C:/Aditya/grid-results/adhoc-hostloop-trace-20260913/RESULTS.md
- trace graph census + legacy decode: legacy 17.0 ms/token = CPU expert chains 10.75 ms (59%), copy-engine transitions 2.12 ms (12%), kernels 6.7 ms; 57 D2H + 35 H2D memcpys per token; pool fetch 82 splits and 1 copy per token, hybrid 122 splits; record C:/Aditya/grid-results/adhoc-shexp-ab-20260913/RESULTS.md
