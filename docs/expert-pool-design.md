# Expert Pool: a two-phase VRAM cache for routed MoE experts

Status: DESIGN (2026-09-01). Not implemented. Companion to the pshard planner/runtime
(PRs #22495/#22691/#22692) and a candidate implementation for upstream RFC #24528
("MoE expert cache, VRAM caching of hot CPU-resident experts with hybrid hit/miss
execution"). Numbers below are measured on the reference box (RTX 5070 Ti 16 GB,
DRAM 45.7 GB/s, PCIe standalone 45 / concurrent 29.5 GB/s, gathered-slice curve
23-30 GB/s) with q35 = Qwen3.6-35B-A3B Q4_K_M (40 layers, E=256 experts, top_k=8,
~450 MB routed experts per layer, ~1.76 MB per expert) as the running example.

## 1. Motivation: the measured constraints this design answers

1. **Streamed MoE decode is host-DRAM-bandwidth bound.** The slot-carve-out
   post-mortem (commit 30d3f4ed1) proved scheduling cannot help: with the prefetch
   fence removed entirely, q8d-s4 decode stayed ~445 ms/token across every ordering
   variant, because CPU FFN weight reads and upload-DMA source reads share host RAM
   bandwidth. "Only placement (fewer bytes through host RAM) can." A VRAM cache hit
   is exactly that: an expert whose bytes touch host RAM zero times that token.
2. **Prefill transfer is already free above a threshold ubatch.** Whole-layer
   streaming hides under the expert GEMMs once
   `B >= B* ~= t_upload_layer x GPU_rate / flops_per_token`
   (PCIe 30 GB/s -> B* ~ 6-8k, visible in the tier ladder: predicted tps flattens
   8192 -> 16384 at 1078 -> 1074). A cache cannot save time that is off the
   critical path, so prefill needs buffers, not residency.
3. **Full-layer FFN pinning is the wrong shape for MoE decode.** Hit-rate vs
   cache-slots is concave (Zipf routing). Spending 256 slots to lift one layer from
   h=0.95 to h=1.0 while other layers sit at h=0.4 is the worst marginal buy.
   Example, ~10 layer-equivalents of expert VRAM on q35:
   - corner (pin 8 layers full + 16-slot slices): ~140 misses/token
   - spread (64 slots/layer everywhere):          ~65 misses/token
   Same bytes, misses halved. Attention pinning, by contrast, is the *best* buy and
   stays (the fixed selector already converged on attn-first placement everywhere).

## 2. The core split

Routed experts stop being model weights with a per-layer placement and become a
managed resource:

- **Classic placement (unchanged machinery):** embeddings, output head, norms,
  routers, attention, shared/always-active experts, dense-layer FFNs. Planner
  assigns pin/stream/CPU per tier as today; budget goes attn-first.
- **Expert pool (new subsystem):** owns ALL routed experts of ALL layers. An expert
  has a HOME (RAM page-locked; disk when the model spills) and a CACHE STATE
  (resident in the pool region or not). `n_pinned` ceases to exist for MoE FFN.

## 3. Memory map

```
VRAM
+------------------------------------------------------------------+
| attn weights (pinned, all layers)                                 |
| routers + norms + shared experts (pinned; tiny, h=1.0 by def)     |
| KV cache                                                          |
| compute scratch (sized by the prefill tier, as today)             |
| EXPERT POOL REGION = remaining budget         <- one region,      |
|   prefill mode: [ buf A: layer L ][ buf B: layer L+1 ]  2 modes   |
|   decode  mode: [ l0: s slots ][ l1: s slots ] ... [ l39 ]        |
+------------------------------------------------------------------+
RAM   [ all routed experts, page-locked ]         <- home (fits-RAM)
DISK  [ all routed experts ] + RAM = capacity cache    (spill case)
```

Minimum region = 2 x max-layer expert bytes (the A/B buffers prefill needs anyway;
q35: 900 MB ~= 512 slots ~= 12-16 slots/layer). Every byte beyond the minimum goes
straight to decode hit rate. Per-layer share (equal split, v1):

    s = pool_bytes / (expert_bytes x n_layers)

v2: water-fill s_l by measured per-layer routing skew (equalize marginal hit-rate).

An "expert slot" is physically three parallel sub-slots (gate/up/down), which may
have different quant types (e.g. iq2/iq3/mxfp4 mixes); same slot index, three
strides. The pool region is a persistent pshard-owned buffer (pipe-shard-family
allocation), NOT galloc scratch: galloc relays out every graph, a cache must not.
This selectively restores the region mechanism of 146e03fa8 with a persistence
justification (the fence justification remains dead per 30d3f4ed1).

## 4. Graph integration

Each layer's routed-expert tensors in the graph become views of that layer's pool
sub-region (ne[2] = s slots). One step in front of MUL_MAT_ID:

```
router -> ids (expert numbers 0..E-1)
       -> remap through the layer's (expert -> slot) table
       -> hits:   slot indices -> MUL_MAT_ID over the pool view
       -> misses: fetch expert -> LRU victim slot (write lands directly in the
                  slot; no staging), patch table, then compute
```

The ids are available at copy time via the ids-crossing machinery (routers pinned
on the compute GPU; landed and verified: +161% on the q35-16k@2000 ALTERNATE cell).
Pool manager state per layer: (expert -> slot) map, LRU list, hit/miss counters.
Host-side, tiny (~E entries x layers).

## 5. Phase behavior

| aspect          | prefill (ubatch >= B*)                  | decode / small batch      |
|-----------------|------------------------------------------|---------------------------|
| region mode     | A/B whole-layer double buffer            | per-layer LRU caches      |
| expert traffic  | stream ALL experts/layer, hidden under GEMMs | hits: 0; misses: ~1-3 fetches/layer x ~50us |
| slice value     | ~0 (transfer hidden; spend VRAM on scratch/attn instead) | the whole game |

**Prefill -> decode switch:** relabel only (metadata reset). Population options:
- v1 lazy: token 1 misses everything (~560 MB total fetch ~ 20 ms once), warm by
  token 2-3. Charged to TTFT via the existing pairwise switch_ms model.
- v2 prompt-histogram warm start: prefill's routers scored every prompt token; the
  per-layer expert histogram (~41 KB of counters) is the best possible predictor of
  the continuation. Upload each layer's top-s prompt experts (~28 ms, overlappable).
- Free either way: the last two layers' A/B contents are valid full sets - those
  layers start at h=1 by remapping.

**Decode -> next prefill:** free. Only the A/B footprint gets overwritten; the rest
of the pool stays warm across prompts/turns. Mid-size prompts (~50-500 tokens: too
expert-diverse for the cache, too short to hide streaming) run A/B with partially
exposed transfer - today's behavior, still the best available; the tier ladder
already routes them.

### 5b. Transitions across strategies

Once experts live in the pool, "decode strategy" collapses into a per-tier
`miss_policy`; every transition is the same relabel:

| old strategy                | pool translation                                  |
|-----------------------------|---------------------------------------------------|
| GPU-only decode (s1-like)   | cache mode, misses = fetch                        |
| CPU-FFN decode (s3-like)    | cache mode, misses = CPU-execute (RFC hybrid);    |
|                             | hits still run on GPU -> strictly better than s3  |
| ALTERNATE-like mixes        | dissolve - nothing left to alternate              |

CPU-execute misses do not warm the cache, so that mode needs the histogram warm
start or an admit-on-second-miss upload policy. Attention is pinned in every
winning plan, so the dense side has ~0 switch bytes; where a config differs, the
existing pshard_switch_plan + switch_ms machinery handles it - orthogonal to the pool.

**Volatile / preserved layout.** Fix the A/B span at one end of the region and mark
its cache slots volatile; the remainder is the preserved sub-pool:

```
region: [ ...preserved slots (survive prompts)... ][ volatile = A/B span ]
```

Prefill reclaims only the volatile span (metadata drop, no copies); preserved
slots ride through the prompt and are warm for the next decode. Allocation biases
hot experts into preserved slots (promote-on-hit) so the cold fringe is what dies.
Skip in v1: letting prefill skip uploading cached experts (hidden above B*, buys
nothing).

Concrete cycle (q35 @ mva8000): prefill 1 streams through the volatile span ->
decode 1 relabels + histogram warm-load (~28 ms) -> prefill 2 reclaims volatile
only -> decode 2 starts warm from the preserved core + prompt-2 histogram.

## 6. Miss policy

- **bs=1 decode: fetch-on-miss (default).** ~1.8 MB into the LRU victim, ~40-60 us
  at gathered-slice rates; also warms the cache. CPU-execute costs the same host
  bytes (the DRAM-bound invariant) but warms nothing.
- **CPU-execute misses** (the FreeToken/RFC hybrid) where warming is wasted:
  predicted one-off experts (admit-on-second-miss policy) and batch/verify tiers
  where the token-union of experts makes miss volume large. v2; in v1 the planner
  simply disables cache mode on tiers where predicted miss volume kills it.
- **Prefill hybrid (three-way split), optional:** cache serves the hot head (skip
  upload), CPU absorbs the coldest tail (thin token mass; balance c so
  t_cpu(tail) ~= t_stream(middle)), PCIe streams the middle. Only relevant below
  B* (scratch-starved cells); above B* GPU-only streaming is already compute-bound.

## 7. Ubatch selection interplay

`B* = t_upload_layer x GPU_rate / flops_per_token` per transfer pipe:
- RAM home over PCIe (30 GB/s): B* ~ 6-8k -> ladder already picks it.
- Disk home over NVMe (5-7 GB/s): B* ~ 32-64k -> giant prefill ubatches; the tier
  ladder already generates them at large n_ctx, planner prices viability (scratch).
- Small-ubatch + persistent-cache prefill (working set U(B) <= s) only wins on slow
  pipes; on this box it loses 2-4x to big-ubatch streaming. Encode as tier pricing,
  not policy.

## 8. Planner and predictor integration

- Registry: `pool_mb`, per-tier `pool_mode = ab|cache` (+ later per-layer s_l).
  Plan attribute pattern identical to overlap=/ids_cross=.
- Knobs after this design: attention pins, scratch (prefill tier size), pool = rest.
- Predictor decode FFN term: `t = t_matmul + E[miss] x fetch_cost(chunk)` with
  fetch_cost from the measured gathered-slice bandwidth curve. Hit-rate h(s):
  assumed Zipf initially; replaced by measured per-layer counters written to the
  registry after warmup (task-4's sanctioned measurement - planner-facing, not a
  runtime switcher).
- Switch costs: lazy-fill dip or warm-load bytes go through the existing pairwise
  switch_cost_ms machinery.

## 9. QA / acceptance

- Value gate unchanged and non-negotiable: hit/miss execution must be
  bit-equivalent to always-fetch (same experts, same math, different bytes moved).
  Token-hash + placement-matched PPL parity as in the current harness.
- Ledger gains hit-rate columns (per run: mean h, misses/token).
- Perf acceptance: MoE decode cells (q35/q235/DSv4 class) at fixed budgets vs the
  current reference; prefill cells must be neutral (region minimum = today's A/B).
- The GGML_SCHED_DUMP_ALLOC tool verifies the pool region is disjoint from scratch.

## 10. Relation to prior art

- **FreeToken:** the S-slot region with prefill A/B repurposing and preserved area
  is FreeToken's layout; this design grounds the sizing in measured constraints
  (B*, DRAM-bound decode, concavity/water-filling) and folds it into pshard's
  per-tier planning instead of a fixed split.
- **RFC #24528:** hybrid hit/miss execution appears here as the v2 miss policy;
  this doc is effectively a concrete implementation proposal on pshard rails
  (planner-priced sizing, per-tier modes, measured-locality adaptation).
- **#26414 / #26003 / #25294 (RAM pinning, lazy experts, disk streaming):** the
  disk-spill case slots in as HOME=disk with RAM as the capacity tier; the pool
  design is unchanged (Section 3), only fetch_cost and B* change.

## 11. Open questions

1. Eviction: LRU vs LFU-with-decay (bursty locality favors LRU; v1 = LRU).
2. Water-filling cadence for s_l (per session? per N tokens?).
3. Prompt-histogram warm start: worth the 28 ms vs lazy? (Measure in v2.)
4. Batch-tier hybrid execution (GPU hits + CPU misses + merge): implementation cost
   vs simply disabling cache mode on those tiers.
5. Multi-turn: protect the pool from mid-conversation prefills of short turns
   (don't let a 50-token turn trash A/B-adjacent slots)?
