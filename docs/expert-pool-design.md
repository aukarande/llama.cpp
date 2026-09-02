# Expert Pool: a two-phase VRAM cache for routed MoE experts

Status: DESIGN (2026-09-01; rev 2026-09-02 - sections 3b, 4b, 6, 11, 12 fold in the
strategy-budget-partition discussion and the FreeToken q* study). Not implemented. Companion to the pshard planner/runtime
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
  (resident in the pool region or not). `n_pinned` ceases to exist for MoE FFN
  (amended 12.9/3b: whole-layer residency survives as the per-tier variable K, priced).

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
straight to decode hit rate. (Amended 3b.1: the remainder is priced per tier among
pool slots / attn+KV pins / K / head; scratch is a measured per-tier requirement,
never a sink; volatile span = A/B + the tier pair's scratch delta.) Per-layer share (equal split, v1):

    s = pool_bytes / (expert_bytes x n_layers)    [K whole-layer pins: / (expert_bytes x (L - K)), 3b.2]

v2: water-fill s_l by measured per-layer routing skew (equalize marginal hit-rate).

An "expert slot" is physically three parallel sub-slots (gate/up/down), which may
have different quant types (e.g. iq2/iq3/mxfp4 mixes); same slot index, three
strides. The pool region is a persistent pshard-owned buffer (pipe-shard-family
allocation), NOT galloc scratch: galloc relays out every graph, a cache must not.
This selectively restores the region mechanism of 146e03fa8 with a persistence
justification (the fence justification remains dead per 30d3f4ed1).

## 3b. Budget partition per strategy: the per-tier variable set and the s0 illustration

Refines Section 3 for two facts settled after it was written: KV is sharded with
attention (not one fixed block), and whole-layer expert residency is a priced planner
choice, not a rule. Running numbers: q35, ctx 16k, budget 8000 MB, reference box.

Supersedes (one-line amendments left at each site):
- Section 2 "`n_pinned` ceases to exist for MoE FFN": whole-layer residency survives as
  `K` (`s_l = E`, eviction off), priced per tier. Section 1.3's argument still holds at
  decode (K = 0 in plan X, 3b.4).
- Section 3: single "KV cache" block; "compute scratch (sized by the prefill tier)";
  "every byte beyond the minimum goes straight to decode hit rate";
  `s = pool_bytes / (expert_bytes x n_layers)` (now `/ (L - K)`, 3b.2).
- Section 5b "Skip in v1: letting prefill skip uploading cached experts": prefill hits
  are served in v1.
- Section 6 "CPU-execute misses ... v2; in v1 the planner simply disables cache mode":
  the split-op (GPU hits + CPU misses in one MUL_MAT_ID + merge) is v1 (user decision);
  closes 11.4. New work: today there is no split MUL_MAT_ID, no -1 skip in the CUDA/CPU
  kernels, no concurrent PCIe+CPU bench pair; `build_moe_ffn` emits one MUL_MAT_ID per
  layer.
- Section 8 `pool_mode = ab|cache` registry column: subsumed (3b.3).

### 3b.1 Per-tier partition (any strategy)

```
budget(tier) = attn pins             n_attn_pinned x (attn weights + THAT layer's KV)
             + attn/KV A/B staging   only if any attention layer streams
             + scratch               measured for this tier's graph (auto-sized)
             + routers / norms / shared experts   always pinned, never in pool; tiny
             + head / embd                        per plan (output_on_gpu -> head_mb)
             + K whole layers' experts            s_l = E, eviction off
             + POOL                               derived remainder (<= remainder if bytes left free)
```

- KV rides with attention. Attn-pinned layers: KV VRAM-resident. Streamed-attn
  layers: KV host-resident, SHARDED, moved through the KV pipe-shard
  (write-cells / writeback) every eval; a `KV_staging` slot pair (largest layer's used
  cells, double-buffered) shares transport with the attn weight A/B. So
  `KV_resident = sum over attn-pinned layers`, not `ctx x n_seq`. Pin cost = weights + KV:
  q35 @16k 55 + 64 = 119 MB/layer; @2k ~63 MB/layer.
- Streamed-attn decode cost under the pool design, per layer per token: 55 MB weights +
  KV movement (up to the full 64 MB @16k when the shard slot is transient) ~= 2 ms/layer
  at 30 GB/s. Not today's number. Forced s0 measures 2.17-3.01 t/s on q35-16k at
  2000-8000 MB (qa/reference-ledger.csv:33, :47) and 2.05-2.93 t/s at ctx 2048 (:5, :19)
  at the same budgets - ctx-independent, so KV traffic is not the driver. Cause: every
  streamed layer uploads its full 450 MB expert set + 55 MB attn; `ids_cross` is
  ALTERNATE-only (src/llama-pshard-plan.cpp:346, :549), so the sliced copy
  (ggml/src/ggml-backend.cpp:1135-1190) never fires for s0 -> ~27-39 x 505 MB per token
  ~= 300-650 ms. Attn+KV traffic alone would cap at 39 x 119 MB ~= 4.6 GB ~= 155 ms
  ~= 6 t/s.
- Scratch is a measured requirement of the tier's graph, never a sink: spare bytes
  cannot "go to scratch". q35: bs=8192 ~1400 MB, bs=2048 ~600 MB, bs=1 ~100 MB.
- Pool minimum on prefill tiers = 2 x largest layer's routed-expert bytes (the A/B
  pair); q35 900 MB. Below it a tier is not viable in pool mode -> legacy streaming.
  Decode tiers with `miss_policy = fetch`: hard floor `s_l >= min(E, top_k x tier_bs)` -
  one MUL_MAT_ID per layer, no -1 skip, so all of a token's experts must be resident at
  once (q35 bs=1: 8 slots/layer). `miss_policy = cpu_exec`: floor 0; the pool may shrink
  below the A/B minimum (s2/s3 territory, see 3b.4 tight-budget note).
- Remainder is priced per tier, never assigned by rule: whole-layer pins (K),
  preserved pool slots, extra attn+KV pins, partial-layer residency (per-layer slot
  bump, v2 `s_l`; the layer-fraction overflow only for its ATTN fraction in pool mode),
  head on GPU (`output_on_gpu`), or free. Prefill and decode tiers may choose
  differently; same bytes, phase-relabeled. `pool_mb` = what is left after the priced
  items.
- KV vs pool: no direct interaction (KV with attention, experts in the pool). They
  compete only for the remainder. At decode the predictor resolves the contest in
  favour of attn(+KV) pins: one pin (119 MB) removes ~55 MB/token of weight streaming +
  the KV delta ~= 2 ms/token; the same 119 MB in the pool is ~1.7 slots/layer (0.8
  slots/layer for the 55 MB weights alone), negligible delta-h. Outcome of pricing, not
  a rule.

```
VRAM, one tier
+---------------------------------------------------------------------+
| scratch                             measured for THIS tier's graph  |
| routers / norms / shared experts    always pinned; tiny             |
| head / embd                         per plan (output_on_gpu)        |
| attn+KV pins, n_attn_pinned layers      weights + that layer's KV   |
| attn+KV A/B staging                     only if any attn streams    |
| K whole layers' experts                 s_l = E, no eviction        |
| POOL = remainder                                                    |
|   [ preserved slots (survive prompts) ][ volatile = A/B span ]      |
+---------------------------------------------------------------------+
RAM   all routed experts, page-locked      <- home; source of every upload
HOST  KV of streamed-attn layers           <- sharded; KV pipe-shard staging
```

### 3b.2 Per-tier variable set

| variable | meaning | values | who sets | v1 / v2 |
|---|---|---|---|---|
| `n_attn_pinned` | attention + KV residency | 0..L | planner search per tier (exists today) | v1 |
| `K` | layers whose experts are ALL resident (`s_l = E`, eviction off) - today's `n_pinned` reinterpreted, decoupled from `n_attn_pinned`. PRICED: saves `max(0, t_upload - t_compute)` per layer per pass at prefill (~0 above B*); frees copy-engine + host-DRAM bw even when hidden | 0..L (s0 free; s1 0; s2/s3 0 as drafted - 3b.3 note) | planner per tier: prefill tiers by `prefill_time + switch_cost`, decode tiers by decode tps | v1 |
| `s` | pool slots per layer for the L-K unpinned layers, volatile + preserved (matches Section 3 and plan X's 43) | v1: `s = pool_mb / (expert_bytes x (L - K))`, one field. Volatile part `pool_ab / (expert_bytes x (L - K))` (q35, K = 0: 900 / (1.76 x 40) ~= 12/layer) is the A/B span on prefill tiers and cache slots at decode; only the preserved part serves prefill hits. v2: per-layer `s_l` water-filled (equalize marginal delta-h per byte), L fields. (Discussion formula `(pool_mb - A/B) / (expert_bytes x (L - K))` counted preserved slots only; superseded.) | derived from `pool_mb` - not a search dimension: budget, `n_attn_pinned`, K, head fix it | v1 uniform / v2 `s_l` |
| `pool_mb` | pool region size | remainder after scratch + routers/norms + head + attn+KV pins + staging + K; `<= remainder` when the planner leaves bytes free (option (d)); prefill tiers `>= A/B minimum`; fetch decode tiers `>= min(E, top_k x bs) x expert_bytes x (L - K)` | - | derived |
| `prefill_mode` | how a prefill tier moves an unpinned layer's experts | `ab_stream` \| `ab_stream + cpu_tail`; GPU-only s0/s1: `ab_stream` only; meaningful on prefill tiers only | planner: `B >= B*` -> `ab_stream`; `B < B*` -> `cpu_tail` if `exposed_saved(c) > merge_overhead` | v1 (`cpu_tail` needs the split-op - new) |
| `miss_policy` | what a cache miss does at decode / small batch | `fetch` \| `cpu_exec` \| `fetch_on_2nd_miss` (+ `hybrid` q*-split, pending); GPU-only s0/s1: `fetch` only; CPU-lineage s2/s3: all | planner per tier within the strategy's allowed set; pool manager executes per miss (incl. the 2nd-miss counter); applied at tier switch, never mid-tier; proposed QA override `PSHARD_MISS_POLICY=fetch\|cpu_exec\|fetch_on_2nd_miss` (not implemented; same pattern as `PSHARD_STRATEGY`) | v1 (`cpu_exec` needs the split-op - new; supersedes Section 6). Proposed defaults (OPEN, see 11): bs=1 -> `fetch` (GPU-only) / `fetch_on_2nd_miss` (CPU-lineage); batch tiers >=16 -> `cpu_exec` |
| `overlap` | transport mode (double-buffer + prefetch scan-ahead) | as today | planner | exists |
| `ids_cross` | router ids exist before the split that copies expert weights | cache-mode tiers: ALWAYS ON (remap + fetch-on-miss need ids at copy time; routers pinned on compute GPU, ~60 MB total q35, `ffn_gate_inp` ~1.5 MB/layer). Today ALTERNATE-only (src/llama-pshard-plan.cpp:346, :549; plan.h:124) - s0/s1 streaming tiers fixed 0; extending the rule `t_slice < max(0, t_full - cover)` (`pshard_alternate_ids_cross_wins`, plan.cpp:200-233) to s0/s1 legacy non-pool streaming tiers is new | forced in cache mode; planner elsewhere | exists (ALTERNATE) |
| `output_on_gpu`, `pin_from_back`, `overflow` | head placement, pin end, layer fraction | as today; `output_on_gpu` priced against the other remainder uses (head_mb VRAM vs per-token CPU head read) | planner | exist |
| `switch_ms` | pairwise switch cost; today n_pinned whole layers + attn-only pins + head; extension: attn+KV pins priced with KV, K = the whole-layer term, plus pool-content delta | ms | computed | derived |

Predictor terms per candidate:
- decode: `t = t_head(output_on_gpu) + sum_attn_streamed t_attn_stream + sum_l [ t_matmul(8 experts) + 8 x (1 - h(s_l)) x t_miss(policy) ]`,
  `t_attn_stream ~= 55 MB / 30 GB/s + KV delta ~= 2 ms/layer`, `t_miss(fetch) ~= 70 us`
  (= 1.76 MB at 25 GB/s; gathered-slice 23-30 GB/s -> 59-77 us; Sections 5/6 "~50 us" /
  "~40-60 us" -> this figure), `t_head(CPU)` = output-matrix DRAM read per token
  (TBD: measure), `t_head(GPU) ~= 0` at `head_mb` VRAM.
- prefill: `sum_l max(0, t_upload_l - t_compute_l) + compute`; `t_upload_l = 0` for the K
  layers, reduced by preserved-span hits elsewhere.
- switch: pairwise residency delta incl. expert residency -> `switch_ms`.

### 3b.3 Strategy = constraint set; tiers coupled only by switch cost

| strategy (src/llama-pshard-plan.h:17-23) | K | attn | miss_policy | prefill_mode |
|---|---|---|---|---|
| s0 GPUONLY_LAYERPIN_LAYERSTREAM | free 0..L | follows K + optional extra attn+KV pins | `fetch` | `ab_stream` |
| s1 GPUONLY_ATTNPIN_FFNSTREAM | 0 | pin-first | `fetch` | `ab_stream` |
| s2 DYNAMIC_FFNCPU_ATTNSTREAM | 0 as drafted (note) | stream (pin if budget) | `cpu_exec` | `ab_stream + cpu_tail` |
| s3 STATIC_ATTNPRIO_ALLMODELS | 0 as drafted (note) | pin-first | `cpu_exec` / `fetch_on_2nd_miss` | `ab_stream + cpu_tail` |
| s4 DYNAMIC_FFN_ALTERNATE | - | - | - | dissolves (3b.6) |

- Note on s2/s3 K = 0: drops today's `n_pinned` there (ledger: s2 `n_pinned` 7 @ 4000,
  qa/reference-ledger.csv:14, :42; s3 12 @ 8000, :46). K free 0..L is the same pricing
  rule as s0; whether `cpu_exec` makes pool slots dominate whole-layer residency is
  settled in the s2/s3 partitions (3b.6, TBD: derive).
- A strategy fixes which variables the search may move; nothing else. Every variable is
  a per-tier field of `llama_pshard_plan` (src/llama-pshard-plan.h:115), written one
  line per tier in the registry exactly like today's
  `strategy= n_pinned= n_attn_pinned= overflow= ... overlap= switch_ms= ids_cross=`
  (src/llama-pshard-plan.cpp:1026). New registry columns: `K`, `s` (derived, recorded),
  `miss_policy`, `prefill_mode`; `pool_mb` derived. Section 8's `pool_mode = ab|cache` is
  subsumed by (`prefill_mode`, `miss_policy`): cache mode = any tier with s > 0
  (preserved slots serve hits at prefill too); column retired.
- Inter-tier coupling is exactly `switch_cost_ms(from, to)` (src/llama-pshard-plan.h:272).
  Today the delta covers `n_pinned` whole layers (`switch_layer_mb`) + attention-only
  pins (`switch_attn_frac`) + head (`switch_head_mb`) over `switch_pcie_gb_s`
  (plan.h:226-229; registry variant header `switch_mb / attn_frac / head_mb / pcie`).
  `find_optimal_ubatch` already picks the prefill tier by predicted TTFT including
  `cost(active -> tier) + cost(tier -> decode)`. Extension: attn+KV pins priced with KV,
  K = the whole-layer term, plus the pool-content delta.
- Variables already differ per tier under ONE strategy today:
  q35 @ 4000 forced s2: `n_pinned` = 9, 9, 9, 8, 8 across bs 1/16/512/1024/2048
  (bigger scratch -> fewer pins; discussion figure, registry source not recorded -
  qa/reference-ledger.csv:14, :42 record `n_pinned` = 7 for the bs=1 tier at both ctx,
  TBD: reconcile). oss @ 4000 forced s4: `overlap` and `ids_cross` differ per tier
  (`ids_cross=1` only on the bs=1 tier, pair gate). Auto oss-16k @ 4000: tier0 STATIC,
  tier1 ATTNPIN, tiers 3-6 LAYERSTREAM/ATTNPIN - even the strategy differs.
- With the pool (q35 @ 8000): `K` = 0 at bs=8192 and bs=1; bs=2048 priced in 3b.4
  (derived map: 0; discussion: 9); `s` small on prefill tiers (scratch eats the bytes),
  large on decode tiers; `miss_policy` `fetch_on_2nd_miss` at bs=1 vs `cpu_exec` at
  bs=16 / verify tiers (CPU lineage); `prefill_mode` meaningful on prefill tiers only.
- Search: same per-tier binary search as `n_pinned`, +1 dimension (K), monotone in
  budget; `s` follows from the remainder; s4 drops out of the candidate set.
- One strategy, N tiers, N independent variable sets, coupled only through switch cost.

### 3b.4 s0 = GPUONLY_LAYERPIN_LAYERSTREAM, q35 @ 8000 MB, ctx 16k

Identity (final): GPU-only. Pins K whole layers = attn + KV + all 256 experts
(`s_l = E`, eviction off) for prefill AND decode; optional extra attn+KV pins; every
unpinned layer streams attn+KV through an attn A/B pair and experts through the pool
A/B span. `miss_policy = fetch` (CPU never computes FFN; floor `s_l >= 8` at bs=1),
`prefill_mode = ab_stream`. K is free 0..L - allowed, not mandated. s1 is the K = 0
attn-first corner; the planner compares the two as candidates by predicted tps.
Preserved-span hits skip upload at prefill: v1 (supersedes the "Skip in v1" line in 5b).

Unit costs (q35, 40 layers, E=256, top_k=8, ctx 16k):

| item | MB | note |
|---|---|---|
| routed experts / layer | 450 | 1.76 MB/expert; pool A/B minimum 2 x 450 = 900 |
| attn weights / layer | 55 | |
| KV / layer @16k | 64 | -> attn+KV pin 119 MB/layer |
| whole layer | 569 | 55 + 64 + 450 |
| routers + norms, total | ~80 | token_embd host |
| output head | GPU: `head_mb` (switch-model estimate 885 = 0.04 x file, qa/pending-verification.md:11-12; quantized head bytes TBD: measure) / CPU: 0 | `output_on_gpu` per tier, priced like every other remainder use. CPU head = full-matrix DRAM read per token (TBD: measure ms/token). Maps below draw the CPU case |
| scratch | 1400 / 600 / 100 | bs 8192 / 2048 / 1 (measured) |
| attn+KV A/B staging | ~240 | 2 x (55 + 64), only if any attn streams |
| RAM home | 18 GB | all 40 layers' experts, page-locked |

**Decode tier bs=1 - decided first (it sets residency; the prefill tier is indifferent
above B\*).** Fixed: scratch 100 + routers/norms 80 = 180 MB (head on CPU, 0 MB);
7820 MB to allocate. Fetch floor: 8 slots/layer.

| plan | K (whole) | attn+KV pins | streamed attn | pool | decode est. |
|---|---|---|---|---|---|
| X | 0 | 40 layers (4760 MB) | 0 | 3060 MB -> 43 slots/layer (`s` = 3060 / (1.76 x 40); 5.4 x top_k), h ~0.8 -> ~1.6 misses/layer -> ~64 fetches x 70 us ~= 4.5 ms | ~110-130 t/s before the head term |
| Y (discussion) | 9 (5121 MB) | +20 layers (2380 MB) | 11 x ~2 ms = 22 ms | 79 MB (~1.5 slots/layer) - below the fetch floor 8 and the A/B minimum -> not viable in pool mode | (discussion: ~25 t/s; illegal shape) |
| Y' (legal re-cut, derived) | 9 (5121 MB) | +13 layers (1547 MB) + staging 240 MB; 12 MB spare | 18 x ~2 ms = 36 ms | 900 MB (A/B minimum) -> ~511 slots / 31 layers ~= 16/layer (2 x top_k); h(16) between h(12) ~0.5 and h(43) ~0.8 -> 50-125 fetches -> 3.5-9 ms | ~21-24 t/s |
| Z | 13 (7397 MB) | 0 | 27 x ~2 ms = 54 ms | 0 MB -> not viable in pool mode -> legacy streaming: 27 layers' experts sliced (8 x 1.76 = 14 MB/layer ~= 0.5 ms -> ~13 ms) with `ids_cross`, or full 450 MB (27 x 15 ms = 405 ms) without | ~14 t/s with `ids_cross`; ~2-3 t/s without = today's forced s0 (qa/reference-ledger.csv:47: `n_pinned` 12, 3.01 t/s) |

Planner picks X (K = 0). One whole-layer pin (569 MB) buys one layer's expert
residency; the same bytes as attn+KV pins remove ~2 ms/token for ~5 layers. Attention
coverage beats expert residency per byte at decode. Today (auto, STATIC attn=40):
43.27 t/s @16k (qa/reference-ledger.csv:46, pre-selector-fix; post-fix all q35 cells
pick STATIC attn=40, so >= that), 57.0 t/s @2k (qa/pending-verification.md:131);
forced s0 @16k: 3.01 t/s (ledger :47).

```
DECODE TIER (bs=1)                                               8000 MB
| scratch                                     100 |
| routers / norms                              80 |
| head                                          0 |  CPU as drawn; GPU head = -head_mb from pool
| attn+KV, 40 layers                         4760 |  all pinned; no attn staging
| pool                                       3060 |  cache mode, 43 slots/layer
|   volatile  900   = A/B span footprint          |  dies each prompt
|   decode-only 1300 = scratch delta 1400 -> 100  |  reclaimed by the prefill tier
|   preserved 860                                 |  survives prompts
```
Discussion map read preserved 2160; 1300 of that is the scratch delta the prefill tier
reclaims, so 860 survives prompts. At decode all 31 non-volatile slots/layer (incl. the
1300) are promote-on-hit; only the hot core lands in the 860.
Per token: attn traffic 0 MB, KV resident, ~64 expert fetches (~4.5 ms) + GPU compute
~3 ms + head term (CPU: TBD: measure; GPU: pool shrinks by `head_mb`). Pool region,
decode mode: volatile span layer-major 12 slots/layer; remaining 31 slots/layer
promote-on-hit.

**Prefill tier bs=8192.** Fixed: scratch 1400 + 80 = 1480 MB (head on CPU, 0 MB);
6520 MB to allocate. Residency value at 8192 ~= 0 (upload hidden), so the tie-break is
switch coherence with the decode tier -> same attn+KV pins (4760 MB), K = 0. Remainder
6520 - 4760 = 1760 MB -> pool (`s` = 25: 12 volatile + 12 preserved): 900 MB A/B minimum
+ 860 MB preserved (~488 slots, ~12 experts/layer, decode-warm). Alternatives for the
860 - (a)-(e) of the spare table below - price ~equal at this tier; preserved keeps
decode warm.

```
PREFILL TIER (bs=8192)                                           8000 MB
| scratch                                    1400 |
| routers / norms                              80 |
| head                                          0 |  CPU as drawn (output_on_gpu per tier)
| attn+KV pinned, 40 layers                  4760 |  experts of these layers are NOT here
| pool A/B span                               900 |  layer L in buf A, L+1 in buf B, transient
| pool preserved                              860 |  ~488 slots ~= 12/layer, persistent
RAM (18 GB, page-locked)                         |  home of ALL 40 layers' 256 experts
```
- Where the "pinned" layers' experts are: in RAM. Under K = 0 no layer has a resident
  expert set; "40 layers pinned" in the map = attention + KV only.
- Per layer L at prefill, its 256 experts = ~12 preserved-span hits (no upload) + ~244
  uploaded RAM -> A/B for that layer's GEMMs, then overwritten by layer L+2.
- Per streamed layer: upload <= 450 MB (~15 ms at 30 GB/s) under ~23 ms compute
  (predictor estimate). Per pass <= 40 x 450 MB = 18 GB.

```
GPU:  attn(L) from pinned weights -> experts(L) GEMMs from buf A      ~23 ms
DMA:                                 upload experts(L+1) RAM -> buf B  ~15 ms  (hidden)
```

**Switch 8192 <-> 1:** residency identical (attn+KV 40 layers, same pool bytes) ->
relabel + volatile refill: ~0 ms residency delta + lazy fill (<= ~20 ms per Section 5,
less with ~12/layer preserved warm), charged through `switch_cost_ms`. Decode pool
3060 = prefill pool 1760 + scratch delta 1300 (1400 -> 100). The 2200 MB that exist only
at decode (900 A/B span + 1300 scratch delta) are volatile by construction - the next
prefill reclaims them; the 860 MB preserved span rides through.

**The s0 identity illustrated: K = 9 layout, prefill tier, scratch 1400 MB.**
8000 - 1400 - 80 - 900 (pool A/B) - 240 (attn staging) = 5380 MB -> K = 9 (9 x 569 =
5121 MB), spare 259 MB.

```
+----------------------------------------------------+ 0
| scratch (bs 8192)                          1400    |
| routers / norms                              80    |
| head                                          0    |  CPU as drawn
| layers 0-8: attn + KV + 256 experts        5121    |  9 whole layers resident
| attn+KV A/B staging (layers 9-39)           240    |  2 x (55 + 64)
| pool A/B span (layers 9-39 experts)         900    |  2 x 450
| spare                                       259    |  planner: (a)/(b)/(c)/(d)/(e) per tier
+----------------------------------------------------+ 8000
RAM: experts + attn of layers 9-39, page-locked
```

| layers | attn | KV | experts | upload / layer |
|---|---|---|---|---|
| 0-8 (whole pins) | resident | resident | resident | 0 MB |
| 9-39 (streamed) | attn A/B | staged | pool A/B | 55 + 450 = 505 MB weights + KV writeback <= 64 MB @16k -> <= 569 MB (17-19 ms) |

```
GPU:  attn(L) from attn buf A -> experts(L) from pool buf A                  ~23 ms
DMA:  attn(L+1) -> attn buf B ; experts(L+1) -> pool buf B ; KV(L)   17-19 ms   hidden
```
- Per pass 31 x 505 MB = 15.7 GB weights (+ KV writeback <= 31 x 64 MB ~= 2 GB) vs 18 GB
  at K = 0. Hidden at bs=8192 (<= 19 < 23 ms). At bs=4096 (~12 ms compute; scratch at
  4096 TBD: measure): 5-7 ms/layer exposed on 31 layers = 155-217 ms/pass vs the
  attention-pinned K = 0 plan 40 x (15 - 12) = 120 ms/pass -> K = 9 loses (derived; the
  discussion's "9 pins save ~45 ms/pass" compared against zero FFN pins with attention
  streamed, not against K = 0 with attention pinned).
- No preserved cache at prefill under this plan (A/B span all volatile); resident
  experts = the 9 pinned layers.
- Same plan at decode (s0's weakness): 31 streamed attn layers x ~2 ms ~= 62 ms/token
  -> ~14 t/s before expert misses; pool 900 MB = ~511 slots over 31 layers ~= 16/layer
  (derived from unit costs; discussion wrote 12 slots/layer). s1 at the same 8000 MB:
  ~110+ t/s.

Spare 259 MB, priced per tier:

| use | MB | prefill effect (bs 8192) | decode effect |
|---|---|---|---|
| (a) partial residency of layer 9: per-layer slot bump `s_9 += 147` (v2 `s_l`); layer-fraction overflow only for the ATTN fraction in pool mode (UP/GATE/MOE fractions would pin routed experts outside the pool - Section 2) | 259 | skips those experts' upload, hidden -> ~0 | small |
| (b) attn+KV pins for 2 more layers | 238 | ~0 | -2 streamed attn layers ~= -4 ms/token |
| (c) preserved pool slots, spread | 259 -> ~147 slots (~4.7/layer over 31) | a few hits skip upload, ~0 | small delta-h |
| (d) free | 0 | 0 | 0 |
| (e) head on GPU (`output_on_gpu`) | `head_mb` (TBD: measure; switch-model estimate 885 > 259 -> not at this spare) | 0 | removes the per-token CPU head read (TBD: measure) |

Prefill tier indifferent; a decode tier at this shape picks (b). Nothing hardcoded.
K itself is the same kind of decision: K = 0..9 evaluated against alternative uses of
569 MB/layer, per tier, by predicted tps.

**K decision rule.** `prefill_time(tier plan) + switch_cost(prefill plan -> decode
plan)`, decode plan fixed by decode tps (X above). Whole-layer pins are worth exactly
`sum(exposed upload saved) - switch delta`.

Per 16k prompt at bs=8192 (2 passes):

| | K = 0 | K = 9 (layout above, 0 extra attn pins) |
|---|---|---|
| upload / pass | 40 x 450 = 18 GB (minus preserved hits) | 31 x 450 = 14 GB (+ 31 x 55 attn, + KV) |
| exposed at bs=8192 (15-19 ms upload < 23 ms compute) | ~0 ms | ~0 ms |
| prefill wall | ~equal | ~equal (tie) |
| residency delta into X | 0 (same attn pins, pool relabel) | evict 9 x 450 MB experts (free) + upload 31 attn+KV pins = 31 x 119 MB ~= 3.7 GB (derived from unit costs; discussion: 11 pins ~= 1.3 GB, assuming plan Y's 20 extra pins) |
| `switch_ms` | ~0 ms + lazy fill (<= ~20 ms) | ~120 ms at 30 GB/s (derived; discussion ~45 ms), every prompt |
| net | K = 0 by ~120 ms/prompt (discussion ~45 ms); sign unchanged | |

**Same rule at bs=2048** (scratch ~600 MB; 16384 / 2048 = 8 passes per 16k prompt,
derived - discussion used 4). Budget map first, derived from the unit costs above:

| | K = 0 | K = 9 |
|---|---|---|
| fixed: scratch + routers/norms (head on CPU, 0) | 600 + 80 = 680 MB | 680 MB |
| K whole layers | 0 MB | 9 x 569 = 5121 MB (layers 0-8) |
| attn+KV | 40 pins = 4760 MB; no staging | staging 240 MB + 8 extra pins x 119 = 952 MB (layers 9-16); layers 17-39 stream attention |
| pool | 8000 - 680 - 4760 = 2560 MB = 900 A/B + 1660 preserved (~943 slots, ~24/layer; `s` = 36) | 900 MB A/B (`s` ~= 16 over 31 layers) |
| spare | 0 MB | 107 MB |
| streamed per pass | 40 x experts only (attention resident) | 23 x (attn + KV + experts) + 8 x experts only |

Exposed upload per pass at ~6 ms compute/layer, 30 GB/s:

| | K = 0 | K = 9 |
|---|---|---|
| upload / layer | 450 x (1 - 24/256) ~= 410 MB (~14 ms; preserved hits skip) | 23 layers x 505-569 MB (17-19 ms) + 8 layers x 450 MB (15 ms) |
| exposed / layer | ~8 ms | 11-13 ms (23 layers), 9 ms (8 layers) |
| exposed / pass | 40 x ~8 ~= 300 ms | 23 x 11..13 + 8 x 9 ~= 325-370 ms |
| per 16k prompt (8 passes) | ~2400 ms | ~2600-2960 ms |
| switch into X | ~0 ms + lazy fill | evict experts (free) + 23 attn+KV pins = 23 x 119 MB ~= 2.7 GB ~= 90 ms |
| net (derived) | K = 0 by ~300-650 ms/prompt | |
| discussion figure (equal attention residency under both K, 4 passes) | 1440 ms | 1116 ms -> K = 9 by ~275 ms/prompt |

K = 9 wins at bs=2048 only if attention stays resident on the unpinned layers (the
discussion's assumption); at 8000 MB the map says it does not. Sign is not fixed by
rule: the planner prices each cell.

Why K differs per tier, same strategy, same budget:
- bs=8192: upload hidden -> prefill tie -> the switch term picks the decode tier's
  residency -> K = 0.
- bs=2048: uploads exposed (~9 ms/layer at equal attention residency), but at 8000 MB
  K = 9 costs attention residency on 23 layers -> K = 0 by the derived map. K > 0 where
  attention residency is unchanged by K; planner prices it, not a rule.
- bs=1: attention coverage beats expert residency per byte -> K = 0 (plan X).
- Caveat: ~23 ms compute at 8192 is a predictor estimate; layers whose GEMMs are shorter
  than their upload (DSv4 dense-lead layers) expose time and pull K > 0 there too; the
  predictor prices per split.

Where GPU-only s0 loses (q35 @ 2000 MB, ctx 16k, K = 0, head on CPU): decode = scratch
100 + routers/norms 80 + pool A/B minimum 900 (12 slots/layer, >= fetch floor 8) +
attn+KV A/B staging 240 = 1320 MB -> 680 MB -> 5 attn+KV pins (595 MB); 35 streamed attn
layers x ~2 ms ~= 70 ms -> ~12-13 t/s ceiling from attention alone; misses at 12
slots/layer (h ~0.5 -> ~4/layer -> 160 x 70 us ~= 11 ms) secondary. (Discussion map:
staging 63 = 2 x ~30 MB attn-only, 7 pins / 33 streamed / 66 ms - pre-KV-correction;
recomputed here with the 240 MB attn+KV pair.) Today's auto (STATIC attn=40, FFN on
CPU): 30.4 t/s (qa/pending-verification.md:130) beats it: CPU-FFN needs no expert VRAM,
freeing bytes for attn pins -> s2/s3 territory (`miss_policy = cpu_exec`, pool below
the A/B minimum at decode).

| | budget 8000 MB | budget 2000 MB |
|---|---|---|
| attn streamed | 0 layers | 35 layers x ~2 ms (discussion: 33) |
| KV traffic | 0 MB | delta for 35 layers |
| expert misses | ~64 x 70 us ~= 4.5 ms | ~160 x 70 us ~= 11 ms |
| est. decode (s0 pool, before head term) | ~110-130 t/s | ~12-13 t/s |
| today, auto STATIC attn=40 (measured) | 43.27 t/s (qa/reference-ledger.csv:46, pre-selector-fix; post-fix >= that) | 30.4 t/s (qa/pending-verification.md:130) |
| today, forced s0 (measured) | 3.01 t/s (ledger :47) | 2.17 t/s (ledger :33) |

### 3b.5 s1 = GPUONLY_ATTNPIN_FFNSTREAM, q35 @ 8000 / 2000 MB

Identity (3b.3): K = 0 fixed; attn pin-first (pin = attn weights + that layer's KV: 55 + 64 = 119 MB/layer @16k, ~63 MB/layer @2k); `miss_policy = fetch` only (GPU-only, no -1 skip in the layer's 2-3 MUL_MAT_ID nodes (12.corrected; src/llama-graph.cpp:2119/2138/2151/2255) -> hard decode floor `s_l >= min(E, top_k x bs)` = 8 slots/layer at bs=1 = 8 x 1.76 x 40 = 563.2 MB); `prefill_mode = ab_stream` only (prefill pool >= A/B minimum 900 MB). All h(s) figures Zipf-assumed ranges (TBD: measured counters, 11.B.6).

**Allocation order.** Marginal value of the same 119 MB at decode (@16k):

| buy | decode value / token | prefill value (B < B*) | prefill value (B >= B*) |
|---|---|---|---|
| +1 attn+KV pin | -~2 ms (55 MB weights + KV delta at 30 GB/s, 3b.1) | -119 MB/pass exposed upload on its layer (~4 ms/pass) | ~0 (hidden) |
| +119 MB pool, steep region (s ~8-16) | delta-h ~0.02-0.05 -> 320 x delta-h = 6-16 fewer fetches x 70 us = 0.45-1.1 ms | -119 MB/pass of expert upload, spread over layers - exact byte tie with the pin | ~0 |
| same, flat region (s ~43) | delta-h ~0.005-0.01 -> 0.1-0.2 ms | same tie | ~0 |

- Decode: pin (2 ms) beats pool (0.45-1.1 ms) per 119 MB across the assumed h slope. Prefill: exact per-byte tie below B*, both ~0 above -> prefill indifferent at every budget; the decode tier decides, switch coherence copies its pins into the prefill tiers (zero residency delta per prompt).
- **Staging quantum.** While >= 1 attn layer streams, the A/B pair is held; staging = 2 x (55 + 64) = 238 MB = exactly two pin-widths (the doc's ~240 elsewhere is this figure rounded), so the 40th pin nets 119 - 238 = -119 MB - a wash with the 39th. Shapes @16k (exact staging): 38 pins = 180 + 563.2 + 238 + 4522 = 5503.2 MB; 39 pins = 5622.2 MB; 40 pins = 180 + 563.2 + 4760 = 5503.2 MB. The 38-pin and 40-pin shapes tie exactly - same structure as @2k below; under the doc's ~240 rounding a 2 MB pseudo-gap appears, which is rounding noise, not a real ordering - and 39 is dominated. Search consequence unchanged: the per-tier binary search on `n_attn_pinned` is non-monotone at the top - probe the 40/40 corner explicitly, bisection misses it.
- **Order (holds at every budget under assumed h):** fixed (measured scratch + 80 MB routers/norms) -> decode fetch floor 563.2 MB (hard viability, not value: without it the fetch-only tier cannot execute) -> attn+KV pins until 40/40 (staging ~240 MB carried while any attn streams) -> remainder = pool slots (head-on-GPU, est. 885 MB TBD: measure, priced against slots only once the pool is large).
- **Caveat (priced, not ruled):** the order flips where measured delta-h > ~0.09 per 1.7 slots/layer (= 2 ms / 22.4 ms per unit delta-h; equivalently delta-h > 0.42 across the 8 -> 16 slot bump that the floor's 563.2 MB = 4.7 pins ~= 9.5 ms would buy). The doc's assumed h(16) ~0.5-0.8 plus this section's extrapolated h(8) ~0.3-0.5 (introduced in the @2000 table below; the doc itself assumes h(12)/h(16)/h(43) only) straddle exactly that threshold; one measured counter set (11.B.6) adjudicates. A cliff would be priced by the predictor, not ruled out.

**Worked map @ 8000 MB, ctx 16k.** Re-derived from unit costs; byte-identical to s0 plan X and the s0 K=0 prefill map (3b.4) - verified, not copied. That identity is the dedupe motivation below.

Decode bs=1: fixed 100 + 80 = 180 MB -> 7820 MB; pins 40 x 119 = 4760 MB (40/40 -> no staging); pool = 3060 MB -> 3060 / 70.4 = 43 slots/layer (5.4 x top_k, >= floor 8).

```
DECODE TIER (bs=1)                                               8000 MB
| scratch                                     100 |
| routers / norms                              80 |
| head                                          0 |  CPU as drawn; GPU head = -885 est. from pool
| attn+KV, 40 layers (40 x 119)              4760 |  all pinned; no staging
| pool                                       3060 |  43 slots/layer
|   volatile 900 (A/B span) + 1300 (scratch delta)|  reclaimed by prefill
|   preserved 860                                 |  survives prompts
```
Per token: attn traffic 0 MB; h(43) ~0.7-0.8 (Zipf-assumed, TBD) -> 8 x (1-h) x 40 = 64-96 fetches x 70 us = 4.5-6.7 ms + GPU compute ~3 ms + head (CPU, TBD) -> **~103-134 t/s before head** (the doc's ~110-130 band sits inside it).

Prefill bs=8192: fixed 1400 + 80 = 1480 MB -> 6520 MB; same pins 4760 (indifference above B* ~6-8k -> switch coherence); pool 1760 MB = 900 A/B + 860 preserved (~488 slots ~= 12/layer; s = 25). Check: 1400 + 80 + 4760 + 900 + 860 = 8000 exactly.

```
PREFILL TIER (bs=8192)                                           8000 MB
| scratch 1400 | routers/norms 80 | head 0 |
| attn+KV pinned, 40 layers                  4760 |
| pool A/B span 900 | pool preserved 860 |
```
Per layer: upload 450 - 12 x 1.76 ~= 429 MB ~= 14.3 ms < ~23 ms compute -> hidden. Per pass ~17.1 GB (18 GB - 0.86 GB preserved hits); 16384 / 8192 = 2 passes. Switch 8192 <-> 1: residency identical -> relabel + lazy volatile fill (<= ~20 ms) via `switch_ms`; decode pool 3060 = prefill pool 1760 + scratch delta 1300.

**Worked map @ 2000 MB, ctx 16k, decode bs=1.** Fixed 180 -> 1820 MB. Two legal shapes, depending on whether the pool region may shrink below the prefill tier's 900 MB A/B span at tier switch (resize machinery unpriced - open residue; the s0 @2000 map in 3b.4 drew shape (i)):

| | (i) pool held at 900 MB (region fixed) | (ii) pool at decode floor 563.2 MB (needs region resize) |
|---|---|---|
| staging | 240 MB | 240 MB |
| pins | floor((1820-240-900)/119) = **5** (595 MB), spare 85 MB | floor((1820-240-563.2)/119) = **8** (952 MB), spare 64.8 MB -> pool |
| pool | 900 MB = 12 slots/layer | 628 MB ~= 8-9 slots/layer (floor 8) |
| streamed attn | 35 x ~2 ms = 70 ms | 32 x ~2 ms = 64 ms |
| misses (thrash regime, s ~= top_k: consecutive-token expert overlap is the only hit source; h assumed, TBD) | h(12) ~0.4-0.6 -> 128-192 fetches -> 9.0-13.4 ms | h(8-9) ~0.3-0.5 -> 160-224 fetches -> 11.2-15.7 ms |
| total (+~3 ms compute, head TBD) | 82-86 ms -> 11.6-12.2 t/s | 78-83 ms -> 12.0-12.8 t/s |

**s1 pool decode @2000/16k ~= 11.6-12.8 t/s** - the shapes agree within noise (the one budget where pin-vs-slot pricing gets close: the shapes differ by 3 pins = 357 MB ~= 6 ms of streamed attn, against the 337 MB pool delta (900 - 563.2) = +4-5 slots/layer ~ 2-3 ms). Note: 11.A.1's old "33 streamed layers" was the stale pre-KV-correction figure (attn-only 63 MB staging); with the 240 MB attn+KV pair it is 32-35 depending on the pool-floor choice; the ~13 t/s cap stands.

Measured today (q35; all q35-16k rows are PPL_MISMATCH-status, pre-recipe, stale per qa/pending-verification.md item 1):
- forced s1 @16k/2000: **falls back, `strategy_active = STATIC_ATTNPRIO`, 28.21 t/s decode / 103.44 prompt, n_pinned 0** (qa/reference-ledger.csv:34) - today's planner already refuses real ATTNPIN there. Same fallback @2k/2000: 28.51 (:6).
- real s1 @16k: 29.30 t/s @4000 (:41), 31.82 @8000 (:48), 38.55 @12000 (:55); @2k: 28.53 (:13), 32.64 (:20), 36.27 (:27).
- auto @16k: 13.33 @2000 DYNAMIC_FFNCPU (:32, pre-selector-fix, stale), 14.50 @4000 (:39), 43.27 @8000 STATIC (:46), 52.92 @12000 (:53); @2k auto: 26.57 / 42.36 / 51.46 / 65.64 (:4, :11, :18, :25).
- **TBD-cite resolved: the "30.4 t/s auto STATIC" cell is q35, ctx 16384, mva 2000 MB** - qa/pending-verification.md:130-131, item 7a, post-selector-fix 2026-08-31 (pred 29.6; the ctx-2048 cell on the same lines is q35-2k-8000, 57.0 t/s; the forced-s1 fallback at the attn=40 shape measured 28.44, :158-162). NOT ctx 2048, and not yet a ledger row (:32 still holds the stale 13.33). The doc's cites ":128/:129" (3b.4) are off by two lines (actual :130/:131). Honesty note: @16k, 40 x 55 = 2200 MB attn weights alone and 40 x 64 = 2560 MB KV each exceed 2000 MB, so "attn=40" at this cell cannot be literal under the doc unit costs (its neighbor :34 shows vram_peak_delta 3084 MB at mva 2000) - the executed plan keeps unpinned-layer KV host-side (today's KV pipe-shard) or the accounting predates KV-with-attention pricing. The run's registry line is the arbiter (TBD: pull it).

Verdict @2000/16k: derived pool-s1 ~11.6-12.8 t/s loses ~2.3-2.6x to today's measured 30.4 t/s auto STATIC (CPU-FFN frees expert VRAM for attn pins - s2/s3 territory), and today's machinery agrees by refusing to run s1 there.

**Prefill viability @ tight budgets.** Pool-mode prefill tier viable iff `scratch(bs) + 80 + staging + 119 x pins + 900 <= budget` (pins >= 0; staging 240 unless pins = 40). At 2000 MB / 16k:

| tier | check at pins = 0 | verdict |
|---|---|---|
| bs=8192 | 1400 + 80 + 240 + 900 = 2620 > 2000 | NOT viable |
| bs=4096 | viable iff scratch <= 2000 - 1220 = 780 MB (bracketed 600-1400, TBD: measure) | TBD, likely NO |
| bs=2048 | 600 + 80 + 240 + 900 = 1820 -> pins <= (2000-1820)/119 = 1.5 | VIABLE: 1 pin, used 1939 MB, spare 61 MB |
| bs<=1024 | viable iff scratch(1024) <= 600 (assumes scratch monotone in bs; measured points are 8192/2048/1 only, TBD: measure) | viable under that assumption; dominated by 2048 regardless (more passes -> more exposure) |

Surviving tier bs=2048 (B < B* -> exposure real), 1 pin + 39 streamed-attn layers, pool = 900 A/B only (no preserved hits). Per 16k prompt = 8 passes: streamed layer uploads attn 55 + experts 450 + KV writeback ~8 MB/pass (2048-token ctx share of the 64 MB; the <=64 MB figure is the full-ctx bound) ~= 513 MB ~= 17.1 ms - ~6 ms compute = ~11 ms exposed; pinned layer 450/30 = 15 - 6 = 9 ms. Per pass 39 x 11 + 9 ~= 438 ms; **per prompt ~3.5-4.1 s exposed** (upper end = KV writeback priced at the full-ctx bound) + ~1.9 s compute -> wall ~5.4-6.0 s, ~2700-3000 prompt t/s.

Non-viable tiers -> **fallback = legacy streaming, today's path** (full 450 MB/layer uploads; sliced only if the `ids_cross` ALTERNATE gate is lifted for s1 streaming tiers - src/llama-pshard-plan.cpp:346, :549). Switch legacy <-> pool tier = ordinary residency delta via `switch_ms`; pool-side metadata relabel free; volatile refill charged as today (<= ~20 ms lazy). The fallback is not hypothetical and may WIN the ladder: legacy big-ubatch streaming measures 5107.91 prompt t/s at this cell (forced s0, :33) and 4040.60 (s1 @4000, :41) - above B* uploads hide regardless of residency. Whether a legacy bs>=8192 tier fits at 2000 MB under pool-era scratch accounting is TBD (the measured 1400 MB scratch says no; the ledger row says today's transient-galloc path ran) - reconcile from the fresh grid's per-tier scratch measurements.

**All-pins threshold + crossover bound.**
- 40-pin decode shape @16k: 180 + 40 x 119 + 563.2 = **5503.2 MB** (no staging). Every byte above -> pool slots. Non-monotone corner: 39 pins needs 5622.2 MB (dominated); 38 pins ties the 40-pin shape at 5503.2 MB with exact 238 staging (the doc's ~240 rounding shows a 2 MB pseudo-gap).
- @2k ctx: 180 + 40 x 63 + 563.2 = **3263.2 MB**. Staging @2k = 2 x 63 = 126 MB = exactly two pin-widths, so the same pair effect holds exactly: 38 pins + staging = 3263.2 MB (= the 40-pin shape), 39 pins = 3326.2 MB (dominated). The quantum is structural (staging == 2 pins by construction), not a 16k artifact.
- Minimum viable s1 decode tier @16k (0 pins): 180 + 563.2 + 240 = **983 MB**; below it no fetch-only decode tier exists (GPU-only cannot shrink the floor).
- ~30 t/s bound (today's s3-class): solve 2 x n_streamed + miss_ms + ~3 ms ~= 33 ms with pool at floor (miss 11.2-15.7 ms, h 0.3-0.5 assumed) -> n_streamed = (30 - miss)/2 ~ 7-9.5 -> pins 30.6-32.9 (31-33 integer) -> budget = 180 + 240 + 563.2 + pins x 119 ~= **4.6-4.9 GB @16k**; below it pool-s1 decode falls under ~30 t/s. (A nominal 34-pin / 6-streamed shape gives 12 + 15.7 + 3 = 30.7 ms = 32.6 t/s - already above 30, so it is not on the crossover; a 5.0 GB endpoint does not reproduce.) Sanity: legacy s1 @4000/16k measures 29.30 t/s (:41) - supportive, not confirmatory. The **true s1-vs-s3 crossover stays blocked on the s3 partition** (11.A.3, 11.A.4): pool-s3 (cpu_exec frees the 563 MB floor and the 900 MB prefill minimum at decode) moves too.

**Ledger dedupe rule.** When s0's search lands K = 0 it emits a plan byte-identical to s1's (same pins, pool, miss_policy, prefill_mode).
- Auto: dedupe candidates by plan signature BEFORE measuring - signature = the plan-shape fields of the registry line src/llama-pshard-plan.cpp:1026 writes (n_pinned, n_attn_pinned, overflow, output_on_gpu, pin_from_back, overlap, ids_cross) extended with the pool-era columns still "(to add)" per the 11-intro and 12.13 (K, s/pool_mb, miss_policy, prefill_mode); the strategy label and the measured outputs on that line (tps, vram, switch_ms) are excluded - they are measurement results, not plan identity. Measure once; share {scratch_measured, cache_measured, predicted tps}.
- Forced: keep both rows as a free invariant check - identical plans must produce identical token_hash and perf within noise (the ledger already records `strategy_active` vs `strategy_forced`: rows :6, :34); drift between the s0(K=0) and s1 rows is a harness or determinism bug caught for free.
- Hook, named: candidate measurement = `llama_pshard_probe_memory` (src/llama-pshard-plan.cpp:252, serialized via `g_probe_mutex` :26/:38), called from the per-strategy binary searches (:367/:407/:449 in `llama_pshard_search_strategy` - the search s1 actually dispatches to, per the candidate-loop routing :1758-1762 and :1884-1894) and from s1's final measurement probes that write `plan.scratch_measured`/`plan.cache_measured` (:483-486/:511-514 in search_strategy) with the predictor hook `pshard_tps_probe_hook` (:242) - all inside the candidate loop `for (s = 0; s < LLAMA_PSHARD_COUNT; ...)` (:1751). `llama_pshard_search_attn_pin` (the `measure_vram` lambda :553; final probes :742-749) serves s3/s4 only - s1 reaches it solely via the forced-s1 STATIC fallback (:1722, :1827; the path behind ledger :34). If the pool-era 1-D `n_attn_pinned` search below is built by reusing search_attn_pin for s1, that is a proposal, not present-tense fact. Dedupe = a signature -> measurement memo consulted before each candidate's final probe; `prune.update` (:1804) and `pshard_plan_is_better` (:1806) unchanged.

**What s1 buys vs s0 under the pool.** s1 is the K = 0, pin-first corner of s0's search space kept as its own strategy: at every budget >= the all-pins threshold the two produce the identical best-known shape (plan X), and s1 reaches it with one fewer search dimension - a 1-D search over `n_attn_pinned` (floor and staging quantum are constants; the 40/40 corner probed explicitly) instead of a joint (K, n_attn_pinned) search whose monotonicity in budget is unproven (11.B.5) - saving ~log L probe rounds per tier per ladder. s0 can only match it (K = 0 lands byte-identical -> the dedupe above makes the second measurement free) or beat it where whole-layer residency pays: exposed-prefill cells with attention residency unchanged by K (the bs~2048 equal-residency case - refuted at 8000 MB by the derived map) or per-layer exposure (DSv4 dense-lead layers). s1 doubles as the free cross-strategy invariant when the two collide.

### 3b.6 Other strategies

- s2 FFNCPU_ATTNSTREAM: pending (K = 0 as drafted - drops today's `n_pinned` 7 @ 4000,
  qa/reference-ledger.csv:14; price K free vs 0 here, TBD: derive; `cpu_exec`,
  `ab_stream + cpu_tail`; pool may drop below the A/B minimum at decode).
- s3 STATIC_ATTNPRIO: pending (K = 0 as drafted - drops today's `n_pinned` 12 @ 8000,
  ledger :46; price K free vs 0 here, TBD: derive; attn pin-first,
  `cpu_exec` / `fetch_on_2nd_miss`, `ab_stream + cpu_tail`).
- s4 ALTERNATE: dissolves - nothing left to alternate once experts are pooled (5b).
  Retirement still needs the "does ALTERNATE ever win" adjudication; input: FreeToken
  puts head + tail layers on CPU (U-shaped per-layer miss rates), not alternating layers.

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
on the compute GPU; landed and verified: +161% on the q35-16k@2000 ALTERNATE cell). Today the flag is
ALTERNATE-gated; cache-mode tiers force it on (3b.2, 6e).
Pool manager state per layer: (expert -> slot) map, LRU list, hit/miss counters.
Host-side, tiny (~E entries x layers).

## 4b. Hybrid execution: FreeToken q* mechanics and where it lands in pshard

Reference: FreeToken (github.com/FlashML-org/FreeToken, arXiv 2608.16157). Decode-only mechanism. Their prefill = whole-layer A/B double buffer, GEMM on GPU, no q* - same as Section 5's prefill row (that row's "spend VRAM on scratch/attn instead" is superseded: scratch is a tier-measured requirement, not a sink; 3b). Split-op hybrid (GPU hits + CPU misses in one op) is **v1** in this design (user decision); this section fixes what it is and where it hooks.

Supersedes: Section 6 bullet 2 (CPU-execute "v2"), Section 10 RFC line ("v2 miss policy"), Section 11 Q4 (batch-tier hybrid open) - hybrid was v2 there. Section 9 amended: bit-equivalence (token-hash) for fetch-only tiers; any tier with CPU-executed routes uses placement-matched PPL parity, same as today's s2/s3/s4 CPU-FFN cells.

#### q* mechanics (decode, per layer, per step)

- Router -> top_k expert ids -> lookup in a global LRU slot cache -> hits + `m` misses.
- Of `m` misses: fetch `q` over PCIe into LRU victim slots; compute `m - q` on CPU from pinned host banks. Both branches read host DRAM, so the CPU sees the residual `B_R = B_H - B_P` (B_H = host DRAM bw, B_P = PCIe gather bw measured while the CPU is reading).
- Balance `q x S / B_P = (m - q) x S / (B_H - B_P)` -> **`q* = m x B_P / B_H`**. Closed form per (layer, step), no search. Rounded in-kernel to the integer that minimizes the slower branch (Triton `_ensure_experts_hybrid_kernel`, offload_kernels.py:291). Victim choice = the cache's replacement policy.
- Fetch set = the `q*` most-recently-active misses (`expert_recency`): recurring misses warm the cache, one-offs go to CPU. This is the admission filter; misses executed on CPU never enter the cache. It filters only when `m > q`: at `m = 1`, `q = 1`, the single miss is always fetched.
- Ids rewritten in place: slot id (hit or fetched) or `-1` (CPU route).
- Order: CPU branch submitted first (D2H of hidden state + topk ids/weights into pinned memory; worker pool computes only `-1` routes) -> GPU batch-copies the `q*` experts into victims -> grouped GEMM over hits + fetched -> two partial `[bs, H]` outputs summed. Exposed time = max(t_cpu, t_gpu). Whole sequence inside one CUDA graph (stream memops for the handshake).
- Constants from `ft bench bw`: real CPU MoE GEMV and real PCIe gather measured **concurrently**; `fraction = pcie_ov / (pcie_ov + cpu_ov)`. With `pcie_ov = S/B_P`, `cpu_ov = S/(B_H - B_P)` this equals `(B_H - B_P)/B_H` = the **CPU-executed share** (TBD: confirm against offload_kernels.py); pshard's `hybrid_frac` below is the fetched share `B_P/B_H = 1 - fraction`. Backend gate: hybrid only if standalone CPU MoE bw > 2x PCIe gather bw, else plain offload (q = m).
- Cache sizing: greedy MoE-first after a KV floor (cache_budget.py, `kv_reserve_tokens = 8192`). Fixed split, not priced per tier.

```
layer l, step t:  ids[top_k] -> lookup -> hits | m misses -> q* = round(m x B_P/B_H) fetched, m-q* on CPU
CPU : |- D2H x, ids_cpu -|---- GEMV over m-q* experts from host RAM @ (B_H - B_P) ----|--+
GPU : |- copy q* experts RAM -> victim slots @ B_P -|- grouped GEMM hits + q* -|--------+-- ADD -> y[bs,H]
      host DRAM sees B_P (DMA) + (B_H - B_P) (CPU) = B_H : the miss term runs at DRAM rate, not PCIe rate
```

Reference box constants: S = 1.76 MB/expert; B_H = 45.7 GB/s; **B_P = slice_bw(S)** ~ 25 GB/s (= 1.76 MB / 70 us, the fetch cost used throughout this doc; gathered-slice curve 23-30 GB/s; same constant as Section 8's `fetch_cost(chunk)`). Not used: PCIe_Concurrent = 29.5 GB/s - that header value is derived (standalone x CPU_Eff x 0.9, profiler-cpu.cpp:664-669), not a gather measurement. -> q*/m = 25/45.7 = 0.55; miss term drops from `m x S / 25` to `m x S / 45.7` = **1.83x**, asymptotically. (The header-derived 29.5 GB/s gives 0.65 / 1.55x - 6b; the paired bench, 11.B.10, decides.)

Derived here from the digest constants, not discussed or measured (TBD: measure per-layer `m` distribution and CPU expert GEMV bw). One per-miss constant everywhere: 70 us = S/B_P; 85 us = S/(B_H - B_P). CPU branch modeled bandwidth-bound (the DRAM-bound invariant of Section 1). Integer rounding at small `m` eats part of the asymptote:

| m misses/layer | q (rounded) | t_hybrid = max(q x 70 us, (m-q) x 85 us) | t_fetch = m x 70 us | gain |
|---|---|---|---|---|
| 1 | 1 | 70 us | 70 us | 1.00x |
| 2 | 1 | 85 us | 140 us | 1.65x |
| 3 | 2 | 140 us | 210 us | 1.50x |
| 4 | 2 | 170 us | 280 us | 1.65x |
| 8 | 4 | 340 us | 560 us | 1.65x |

- Derived - s0 decode illustration, plan X (q35 @ 8000, ctx 16k; 3b.4): pool 3060 MB / (1.76 MB x 40) = 43 slots/layer, h(43) ~ 0.8 -> ~1.6 misses/layer -> ~64 fetches x 70 us ~ 4.5 ms/token. s0 is GPU-only (`miss_policy = fetch`), so hybrid does not apply there; the same residency under s2/s3 would gain per the table row of each layer's `m`: `m = 1` layers 1.00x, `m >= 2` layers 1.5-1.65x. The mean 1.6 does not decide; the per-layer `m` distribution does (TBD: measure).
- Derived - s2/s3 case at budget 2000 (the tight-budget illustration: 12 slots/layer, h ~ 0.5, 4 misses/layer, 160 x 70 us = 11 ms/token): table row `m = 4` -> 40 x 170 us = ~6.8 ms/token. Batch/verify tiers (large token-union of misses per layer) approach the 1.83x asymptote.
- Derived - FreeToken's 2x gate on this box: CPU MoE GEMV bw not measured (TBD: measure); upper bound = DRAM 45.7 GB/s vs gather 23-30 GB/s = 1.5-2.0x -> their rule would likely pick plain offload. Proposed (not a user decision): pshard applies no gate; the per-tier price decides (Section 8).
- FreeToken reported gains (RTX 5090: Qwen3.6-35B-A3B 77-83 t/s vs 33-46 t/s; DeepSeek-V4-Flash 22-25 vs 12-17 t/s; RTX 4060 laptop 8 GB: 39.3 t/s on 35B) combine pipelined prefill + LRU + q*, not q* alone.

#### FreeToken -> pshard today (branch pshard-tot)

| FreeToken | pshard today |
|---|---|
| fetch misses over PCIe (q = m) | sliced-by-used-ids consume-time copy. Decision: `ggml_backend_sched_split_find_moe_consumer` / `ggml_backend_sched_prefer_sliced_expert_copy`, ggml/src/ggml-backend.cpp:1135-1170. Copy: ggml/src/ggml-backend.cpp:2198-2282 (gated on the split input being a host WEIGHTS tensor :2199-2201; syncs the ids backend, reads ids to host, builds a `used_ids` bitset, copies expert slices into the transient split input). No residency: every token refetches. |
| compute misses on CPU (`-1` routes) | none at expert granularity. s2 FFNCPU / s3 ATTNPRIO / s4 ALTERNATE put whole layers' experts on CPU via `-ot` overrides = FreeToken's `--moe-cpu-layers` (layer granularity), not q*. |
| LRU slot cache, slot_for_id, expert_recency | none (Sections 4, 5b, 6 are design only). |
| split-op: GPU part + CPU part + sum; `-1` skip in kernels | none. `build_moe_ffn` (src/llama-graph.cpp:1941) emits one `MUL_MAT_ID` per expert tensor through `build_lora_mm_id` (src/llama-graph.cpp:1542). Kernels do not skip `-1`, they assert/index on it: CUDA `ggml_cuda_mul_mat_id` (ggml/src/ggml-cuda/ggml-cuda.cu:1935, `assert(expert_to_use >= 0 ...)` :2011), CPU `ggml_compute_forward_mul_mat_id` (ggml/src/ggml-cpu/ggml-cpu.c:1534, `assert(i02 >= 0 ...)` :1631); release builds index out of bounds. `mmid.cu:28-80` compaction scan counts `expert_used < expert` with no negative guard. The sched sliced path asserts `id >= 0` (ggml/src/ggml-backend.cpp:2233). |
| `ft bench bw`: CPU MoE GEMV and PCIe gather measured concurrently, both rates recorded | partial. (a) CPU ops incl. `MUL_MAT_ID` run under a bulk 256 MB PCIe set/get stress loop (examples/llama-profiler/profiler-cpu.cpp:105-116, `run_concurrent` :382-411, `run_moe_benchmarks` :446-468) -> per-op `Concurrent_GFLOP/s` (:530), `CPU_Eff` measured; `PCIe_Concurrent` **derived** = standalone x CPU_Eff x 0.9 (:664-669), not measured. (b) gathered PCIe bursts (24 x chunk, strided) under CPU DRAM-read stress threads -> `PCIe_Sliced` curve (`calibrate_pcie_sliced` :124-171), consumed by `slice_bw(chunk)` (src/llama-benchmark.h:84). src/llama-benchmark.cpp:172-185 parses the header (`eff_pcie_bw = min(PCIe_Concurrent, DRAM_BW)`). Missing vs FreeToken: gathered PCIe and bs=1 CPU expert GEMV running simultaneously with both rates recorded; the PCIe side under CPU load is estimated, not measured. |
| KV-vs-experts budget: greedy MoE-first after a KV floor | planner per tier: attn+KV pins, tier-measured scratch, routers/norms first; `pool_mb` derived from `K` (layers with all experts resident, `s_l = E`) and `s` (slots/layer for the other L-K layers) - per-tier variable set - 3b.2; `miss_policy` chosen within the strategy's allowed set by predicted tps. |
| CPU layer set: head + tail (U-shaped per-layer miss rate) | s4 alternates layers by parity. |

#### v1 hook points

1. **Pool module** (new, pipe-shard family; v1). Persistent CUDA region disjoint from galloc scratch (Section 3). Per layer: expert -> slot map, LRU list (recency = FreeToken's `expert_recency`), hit/miss counters, and a registration {router ids tensor, pool view, `ids_gpu` leaf, `ids_cpu` leaf} the sched hook reads. Owns the fetch into victim slots.
2. **Ids split at the consume-time path** (ggml/src/ggml-backend.cpp:2198-2282; v1). Arrangement: the **host home tensor stays the split input** (src[0] of the GPU node in the user graph), so the is_host/WEIGHTS gate (:2199-2201) and `find_moe_consumer` (:1137-1147) still select the path; the **pool view replaces `input_cpy` as the copy destination** (slot offsets; no transient allocation for pool tensors) and the split-graph node's src[0] is rebound to the pool view. Pool views are CUDA-resident and would never trigger this path on their own.
   - ids: the sched reads the **router ids tensor** (from the pool registration, not `node->src[2]`); values are expert ids 0..E-1, so the `id >= 0` assert at :2233 holds for that read (reuse the ids-read block :2206-2238). Requirement: cache-mode tiers set `ids_cross = 1` (today gated to ALTERNATE at src/llama-pshard-plan.cpp:346-347 / :549-550 - lift the gate).
   - lookup pool -> hits (slot ids) + `m` misses; `q = round(m x hybrid_frac)` minimizing the slower side; fetch set = the `q` most recent misses; copy them into victim slots with the same grouped `copy_experts` lambda (:2242-2256), destination = slot offset in the pool view; patch the map.
   - write `ids_gpu` (slot id, or `-1` for CPU routes) and `ids_cpu` (expert id, or `-1` for GPU routes): two **leaf tensors** registered by the pool module per layer (graph leaves, so galloc allocates them); the sched fills them host-side and uploads `ids_gpu` before the GPU node runs. GPU chain src[2] = `ids_gpu`; CPU chain src[2] = `ids_cpu`; `get_rows` on `w_s` (src/llama-graph.cpp:1554) and `add_id` on expert biases (:2127/:2146/:2162/:2263) keep the original router ids.
   - `miss_policy = fetch` tiers never build `ids_cpu`; `ids_gpu` carries no `-1`.
3. **Graph** (src/llama-graph.cpp:1941 / :1542; v1). Per tier, by `miss_policy`:
   - `fetch`: one `MUL_MAT_ID` per expert tensor over the pool view with `ids_gpu` (slot ids). No CPU node, no ADD: same graph shape as today; token-hash gate.
   - `cpu_exec | fetch_on_2nd_miss | hybrid` (any policy admitting CPU routes): **two full expert-FFN chains** (gate/up[/bias]/act/down) - one over pool views with `ids_gpu`, one over host tensors with `ids_cpu` - and **one ADD at the down output** `[n_embd, n_expert_used, n_tokens]`; the existing per-route weighting and expert sum run once after it. Kernels gain the `-1` skip and write zeros for skipped routes; a per-route mask (0 for `-1` routes) is applied after each `add_id` so bias-carrying partials stay zero and the ADD is an exact two-partial sum (alternative: add biases once after the ADD where the position allows). `cpu_exec` is the `q = 0` case of the same split-op. Kernels: CUDA ggml/src/ggml-cuda/ggml-cuda.cu:1935 (+ `mmid.cu:28-80` compaction scan treats `-1` as "no expert"), CPU ggml/src/ggml-cpu/ggml-cpu.c:1534.
   - Placement: the `ids_cpu` chain is **pinned to CPU explicitly** (`ggml_backend_sched_set_tensor_backend` on its nodes, or an exclusion in `ggml_backend_sched_backend_id_from_cur` keyed on the `ids_cpu` tensor) - new placement code. Without it, `op_offload` (default on: src/llama-context.cpp:3863, common/common.cpp:2081) moves a host-weight op to CUDA at batch >= `GGML_OP_OFFLOAD_MIN_BATCH` = 32 (ggml/src/ggml-backend.cpp:1022-1033, ggml/src/ggml-cuda/ggml-cuda.cu:5404-5407, :5578) and uploads the whole host tensor - today's streaming path, on exactly the batch/verify tiers hybrid targets. Pinned on CPU, the host tensor is not a CUDA-split input, so the sliced-copy path does not fire for it.
   - Cost: one CPU split per layer (GPU -> CPU -> GPU boundary pair): D2H `x [n_embd, n_tokens]`, H2D partial `[n_embd, n_expert_used, n_tokens]`, plus the ADD - `t_split` in the predictor term below (TBD: measure).
   - Gate: `miss_policy = fetch` tiers: token-hash; any policy admitting CPU routes: PPL parity (Section 9, to be amended as above).
4. **Planner / registry** (v1). Per-tier fields at src/llama-pshard-plan.h:118-133, next to `overlap= / ids_cross= / switch_ms=`: `miss_policy = fetch | cpu_exec | fetch_on_2nd_miss | hybrid`, `hybrid_frac = B_P / B_H` (fetched share; applied at tier switch, never recomputed mid-tier). Bench-derived constant at the variant header :226-229 next to `switch_pcie_gb_s` (printed/parsed at src/llama-pshard-plan.cpp:1010 / :1110-1113).
   - Allowed set: `hybrid` joins the CPU-lineage set (s2/s3) next to `cpu_exec / fetch_on_2nd_miss`; GPU-only s0/s1 stay `fetch` (strategy constraint table). v1 CPU-lineage default at bs=1 stays `fetch_on_2nd_miss`; `hybrid` is a priced candidate against it.
   - Constants: extend the existing profiler (reuse `run_concurrent` :382-411 + `calibrate_pcie_sliced` :124-171) so a gathered fetch burst at chunk = S and a bs=1 CPU `MUL_MAT_ID` GEMV run simultaneously with both rates recorded, emitted in the same profile-header pattern as `PCIe_Concurrent` (profiler-cpu.cpp:520-523, parsed at src/llama-benchmark.cpp:172-185). Until then: `B_P = slice_bw(S)` ~ 25 GB/s, `B_H` = 45.7 GB/s -> `hybrid_frac` = 0.55 (TBD: measure).
   - Predictor miss term: `fetch: m x S / B_P` (= Section 8 `fetch_cost(S)`); `hybrid: max(q x S/B_P, (m-q) x S/(B_H - B_P))` at the rounded `q` (table above) `+ t_split`.
   - Proposed (not a user decision): no 2x gate; the per-tier price decides.
   - Recency-ordered fetch selection filters churn only when `m > q`. Whether it subsumes the `fetch_on_2nd_miss` counter is a TBD offline check on the existing routing counters; `fetch_on_2nd_miss` stays a separate value.

v2: tuning `cpu_exec` as the batch-tier default (available in v1 as `q = 0` of the same split-op; Section 6 must resolve the v1-default vs v2 tension explicitly); per-layer `hybrid_frac_l` only if per-layer `h_l` curves justify it (same offline check as water-filled `s_l`, Section 3).

#### Note for the ALTERNATE adjudication

FreeToken engine.py:1185 chooses **head + tail** layers for `--moe-cpu-layers` because per-layer miss rates are U-shaped. pshard s4 ALTERNATE picks layers by parity. If s4 survives as a layer-granular CPU placement, its layer set should follow measured per-layer miss rate, not parity; with the pool, the same layers surface as the low-`h_l` layers in the v2 water-fill (more slots, or `cpu_exec` / `hybrid` there). (TBD: measure per-layer miss rate vs layer index on q35 from the existing routing counters before deciding retire vs re-shape.)

## 5. Phase behavior

| aspect          | prefill (ubatch >= B*)                  | decode / small batch      |
|-----------------|------------------------------------------|---------------------------|
| region mode     | A/B whole-layer double buffer            | per-layer LRU caches      |
| expert traffic  | stream ALL experts/layer, hidden under GEMMs | hits: 0; misses: ~1-3 fetches/layer x ~70us (6a) |
| slice value     | ~0 (transfer hidden; remainder priced per tier - 3b.1) | the whole game |

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
slots ride through the prompt and are warm for the next decode. (Amended 11.B.7:
the tier pair's scratch delta is volatile too.) Allocation biases
hot experts into preserved slots (promote-on-hit) so the cold fringe is what dies.
Superseded (12.8): prefill DOES serve hits from resident experts in v1 - they skip
the upload (6c).

Concrete cycle (q35 @ mva8000): prefill 1 streams through the volatile span ->
decode 1 relabels + histogram warm-load (~28 ms) -> prefill 2 reclaims volatile
only -> decode 2 starts warm from the preserved core + prompt-2 histogram.

## 6. Miss policy and prefill mode

Two per-tier plan variables, two different questions, one shared execution primitive:

| variable       | tier it applies to                     | question it answers                                                        |
|----------------|----------------------------------------|----------------------------------------------------------------------------|
| `miss_policy`  | decode / small-batch (cache mode)      | a routed expert is NOT in the layer's VRAM slots: fetch it or compute on CPU? |
| `prefill_mode` | prefill (A/B mode)                     | all E experts are needed this ubatch: which subset NOT to upload, and compute on CPU instead? |

Both are planner-chosen per tier within the strategy's allowed set, stored per tier like
`overlap=` / `ids_cross=` / `switch_ms=` (plan fields src/llama-pshard-plan.h:123-133; registry
line format src/llama-pshard-plan.cpp:1026 (write) / :1167-1171 (parse)), applied at tier
switch, never changed mid-tier (no runtime plan switcher). The pool manager executes them.
The shared primitive is the split-op (6e).

### 6a. miss_policy

Ownership:
- Planner picks the value per tier by predicted tier cost; strategy = allowed set.
- Pool manager executes per miss. The 2nd-miss counter is a local rule inside the policy, not
  a plan change.
- QA override: `PSHARD_MISS_POLICY=fetch|fetch_on_2nd_miss|cpu_exec|hybrid` (`second` = env
  alias for `fetch_on_2nd_miss`), same pattern as `PSHARD_STRATEGY` (src/llama-pshard-plan.cpp:2053).

Values (q35, bs=1, S = 1.76 MB/expert; B_H = 45.7 GB/s DRAM; fetch rate = gathered-slice curve
23-30 GB/s, see pricing inputs):

| value               | action on a miss                                                         | warms cache?     | cost per miss                                              | best when                                                         | ver |
|---------------------|--------------------------------------------------------------------------|------------------|------------------------------------------------------------|-------------------------------------------------------------------|-----|
| `fetch`             | copy expert RAM -> LRU victim slot, compute on GPU with the hits         | yes              | S / 23-30 GB/s ~= 60-80 us; predictor constant ~70 us (PCIe_Sliced 2 MB point) | GPU-only tiers; real locality; small bs                           | v1  |
| `fetch_on_2nd_miss` | first miss: `cpu_exec`, no admit; repeat miss (per-layer counter): `fetch` | on repeat only | first: as cpu_exec; repeat: as fetch                       | low s (steep h slope -> churn); one-off filter; CPU-lineage bs=1   | v1  |
| `cpu_exec`          | compute the miss on CPU from host weights, no upload (split-op)          | no               | S / B_cpu_achieved (bs=1 CPU MUL_MAT_ID rate, <= B_H = 45.7 GB/s -> >= 38.5 us) + handoff/merge share (6e); achieved rate (TBD: measure) - profiler MUL_MAT_ID rows give GFLOP/s, not bytes/s at bs=1; the earlier ~60-80 us estimate assumes 22-29 GB/s achieved | batch/verify tiers (token-union makes miss volume large); cold one-offs | v1 mechanism (split-op); batch-tier default proposed, open (11.B.9). Supersedes the earlier build-order placement of `cpu_exec` under v2 |
| `hybrid`            | per (layer, step) with m misses: fetch q* = m x B_P / B_H by recency, CPU computes m - q* (6b) | fetched ones | m x S / B_H per layer = DRAM-ceiling lower bound (vs m x S / B_P fetch-only); holds only with the max() handshake (6e) and paired constants (6b) | m large; PCIe gather and CPU GEMV rates comparable                | v1 candidate (open, 11.B.9): rides on split-op; constants need the 6b bench entry |

Allowed sets per strategy (code names src/llama-pshard-plan.h:17-23; s0..s4 shorthand below):

| strategy                                     | allowed `miss_policy`                                   |
|----------------------------------------------|---------------------------------------------------------|
| s0 GPUONLY_LAYERPIN_LAYERSTREAM              | `fetch` only (GPU-only: no CPU FFN compute ever)        |
| s1 GPUONLY_ATTNPIN_FFNSTREAM                 | `fetch` only                                            |
| s2 DYNAMIC_FFNCPU_ATTNSTREAM                 | `fetch` / `fetch_on_2nd_miss` / `cpu_exec` / `hybrid`   |
| s3 STATIC_ATTNPRIO_ALLMODELS                 | `fetch` / `fetch_on_2nd_miss` / `cpu_exec` / `hybrid`   |
| s4 DYNAMIC_FFN_ALTERNATE                     | dissolve (section 5b); retirement pending the ALTERNATE adjudication (11.D.21) |

Pricing inputs (predictor decode term per layer: `t_matmul + 8 x (1 - h(s_l)) x t_miss(policy)`):
- fetch constant, one baseline: PCIe_Sliced curve at the 2 MB point (parsed
  src/llama-benchmark.cpp:196-201; measured examples/llama-profiler/profiler-cpu.cpp:124-179)
  -> ~70 us for S = 1.76 MB. Supersedes section 5's "~50us" and old section 6's "~40-60 us"
  (section 5 amended). Section 8's `fetch_cost(chunk)` is this curve.
- tier bs: at bs >= 16 the per-token miss union grows, warming value per fetched byte drops.
- h(s) and its slope: low s -> churn -> favor `cpu_exec` / `fetch_on_2nd_miss`.
- host-DRAM contention: `fetch` occupies PCIe + DRAM source reads; `cpu_exec` occupies cores +
  DRAM. Both hit the DRAM-bound ceiling of section 1; only hits beat it.
- `cpu_exec` never warms the cache -> pair it with 2nd-miss admission (v1) or the
  prompt-histogram warm start (v2, section 5).
- `cpu_exec` removes the A/B minimum from decode tiers: `pool_mb` may drop below 2 x 450 MB
  (s2/s3 at tight budgets, q35 @ 2000 MB); prefill tiers keep the minimum (section 3).
- v2 SSD home (section 3): a VRAM miss may also be a RAM miss; t_miss gains
  `E[miss_ram] x fetch_ssd` (~1.8 MB / 5-7 GB/s ~= 300 us + latency). SSD -> RAM slot -> then
  `fetch` or `cpu_exec` as above; SSD is never read by compute.

Proposed v1 defaults (CPU-lineage default open, 11.B.9):
- bs=1: `fetch` (GPU-only s0/s1); `fetch_on_2nd_miss` (CPU-lineage s2/s3). `hybrid` is the
  s2/s3 bs=1 candidate once the 6b bench entry supplies its constants (open).
- batch / verify tiers (bs >= 16): `cpu_exec` (CPU-lineage s2/s3); s0/s1 stay `fetch`.

Worked numbers, s0 decode plan X (q35 @ 8000 MB, ctx 16k; 3b.4 plan X): 43 slots/layer, h(43) ~= 0.8 assumed -> ~1.6 misses/layer -> ~64 misses/token.
Section 1 point 3 puts ~65 misses/token at 64 slots/layer; the two h(s) points are not from
one curve - fix one h(s) model (Zipf assumption, section 8) and recompute both places
(TBD: measured per-layer counters).
- `fetch`: 64 x ~70 us ~= 4.5 ms/token expert traffic (+ ~3 ms GPU compute -> ~110-130 t/s).
- `hybrid`: (TBD: needs the 6b bench entry). DRAM-ceiling lower bound:
  64 x 1.76 MB / 45.7 GB/s ~= 2.5 ms/token, valid only if h is unchanged (hybrid admits only
  q* of the misses), B_P equals the fetch-path rate, and the max() handshake (6e) has landed;
  today's sched gives t_gpu + t_cpu + handoff.

### 6b. hybrid = FreeToken's q*

Per (layer, decode step):
```
router -> ids -> pool lookup -> hits H (slot ids) + misses m
misses sorted by expert recency (most-recently-active first)
  first q*        -> fetch RAM -> LRU victim slots -> GPU   (ids := slot id)
  remaining m-q*  -> CPU computes from host weights          (ids := -1 on GPU side)
GPU: grouped GEMM over H + fetched ; CPU: GEMV over m-q*  ->  two partial [bs,H] -> ADD
```

- Both branches read host DRAM. PCIe gather takes B_P, CPU sees the residual B_H - B_P.
- Balance the two branches: `q x S / B_P = (m - q) x S / (B_H - B_P)` -> `q* = m x B_P / B_H`.
  Closed form per (layer, step), no search; rounded to the integer that minimizes the slower
  side (FreeToken `_ensure_experts_hybrid_kernel`, offload_kernels.py; no pinned commit, line
  refs dropped).
- Selection by recency: recurring misses get admitted and warm the cache, one-offs go to CPU.
- Order: CPU branch launched first (activations + routing D2H), then GPU copies + GEMM, then
  the sum. Exposed time = max(t_gpu, t_cpu) ONLY if the GPU expert chain is launched before
  the host computes the CPU chain: a CUDA-graph / stream-memop handshake (FreeToken) or a
  sched change (CPU split computed on a worker thread while the GPU split is in flight).
  Today's sched runs splits serially with `ggml_backend_graph_compute_async` per split
  (ggml/src/ggml-backend.cpp:2429), the CPU backend computes synchronously on the host, and
  the GPU -> CPU input handoff is a synchronous copy or get_async + synchronize = device-wide
  barrier (:2301-2352; ~5 ms drain measured on q8d-s4 per the code comment) -> sum, not max.
  The 1.55x below and the `hybrid` cost row in 6a hold only after that lands; until then
  `hybrid` / `cpu_exec` cost = t_gpu + t_cpu + handoff sync.
- Decode only: FreeToken uses q* at decode only; its prefill is whole-layer A/B on GPU
  (= `ab_stream`). `ab_stream + cpu_tail` (6c) is ours.
- Our box, header-derived estimates (TBD: measure): B_P = 29.5 GB/s (header PCIe_Concurrent,
  derived - see Constants), B_H = 45.7 GB/s -> q*/m ~= 0.65; miss time per layer
  `m x S / 29.5` -> `m x S / 45.7` = ~1.55x. The interim B_P for expert-slice chunks should
  be the PCIe_Sliced 2 MB point (consistent with the ~70 us fetch constant in 6a); it is below
  29.5 GB/s, so the fraction and the ratio move (TBD: paired bench; 4b prices with the slice point: 0.55 fetched share, 1.83x asymptotic). The ceiling is DRAM
  bandwidth: the same invariant as the q8d-s4 ~445 ms/token measurement (section 1). hybrid
  moves the miss cost to that ceiling; it does not go below it.
- FreeToken's own gate: hybrid only if standalone CPU MoE bandwidth > 2x PCIe gather
  bandwidth. Estimate: 45.7 / 29.5 ~= 1.55 < 2 -> their rule would pick plain offload here
  (achieved CPU MUL_MAT_ID rate TBD; ratio can only fall). We price it per tier instead
  (predictor), so it is a planner choice, not a rule.
- Achieved CPU MUL_MAT_ID bandwidth at bs=1 on this box: (TBD: measure); DRAM_BW is its upper
  bound.

Constants - what the profiler measures today (examples/llama-profiler/profiler-cpu.cpp):
- PCIe_Sliced (:124-179): gathered H2D bursts (24 chunks x 0.5/2/8/32 MB, strided sources)
  under synthetic CPU DRAM read stress; reports the PCIe rate only. Header line :523, parsed at
  src/llama-benchmark.cpp:196-201.
- Per-op MUL_MAT_ID rows and CPU_Eff (run_concurrent :382-415; row format :530): real CPU
  MUL_MAT_ID under a bulk 256 MB H2D+D2H stress loop (:105-116); reports the CPU rate
  (Concurrent_GFLOP/s) only.
- Header PCIe_Concurrent (:520) = PCIe_Standalone x CPU_Eff x 0.9 (:669): derived, not
  measured. The 29.5 GB/s above is this number. Parsed at src/llama-benchmark.cpp:172-186.
- Missing: gathered-slice H2D at expert-slice chunk size AND real CPU MUL_MAT_ID (bs=1,
  top_k=8) running together, BOTH achieved rates reported. FreeToken's `ft bench bw` does
  that; fraction = `pcie_ov / (pcie_ov + cpu_ov)`.
- NEW bench entry (v1, prerequisite for `hybrid`): the paired measurement above. Registry
  stores the fraction `q*/m` per tier. Until it lands: 0.65 header-derived estimate (TBD:
  measure).

Where the pieces are today and the v1 hook points: section 4b (mapping table with
file:line refs, hook list).

### 6c. prefill_mode

Applies to prefill tiers only. Plan-time choice; the partition it produces is data-dependent
per ubatch. Resident experts skip the upload in both modes - v1 (supersedes "Skip in v1" in
section 5b; 5b amended). Resident = preserved-pool hits and any layer with `s_l = E`
(K whole-layer expert pins, eviction off; a per-tier variable - 3b.2).

`ab_stream`:
- All E experts of layer L upload RAM -> buffer A while layer L-1 computes from buffer B;
  GPU expert GEMMs run from A; alternate per layer.
- Hits skip upload: `t_upload_l` = 0 for `s_l = E` layers, reduced by preserved hits elsewhere.
- Addressing: in A/B mode the expert MUL_MAT_ID also runs over the region view with slot ids;
  buf A/B are the volatile slot range, preserved hits map to their existing slots (no copy).
- CPU does no FFN work.
- Exposed per layer per pass = `max(0, t_upload_l - t_compute_l)`. q35 over PCIe:

| ubatch | t_upload (450 MB) | t_compute (est.) | exposed / layer |
|--------|-------------------|------------------|-----------------|
| 8192   | ~15 ms            | ~23 ms           | ~0              |
| 4096   | ~15 ms            | ~12 ms           | ~3 ms           |
| 2048   | ~15 ms            | ~6 ms            | ~9 ms           |

```
GPU:  attn(L) from pinned weights -> experts(L) GEMMs from buf A      ~23 ms  (bs=8192)
DMA:                                  experts(L+1) RAM -> buf B        ~15 ms  hidden
```

`ab_stream + cpu_tail`:
- Same A/B path for the hot and middle experts of layer L.
- The layer's coldest c experts (fewest tokens routed to them in THIS ubatch) are not
  uploaded: CPU computes their token groups from RAM, in parallel with the GPU GEMMs
  (parallel only with the max() handshake of 6b/6e; today's sched gives sum).
- Both read host DRAM (section 1 invariant): price `t_upload(rest)` at B_P (concurrent) and
  `t_cpu(tail)` at B_H - B_P, as in 6b. c from the balance
  `t_cpu(tail token mass) ~= t_upload(remaining experts)`. Estimate for this box: CPU absorbs
  ~2-3% of the token mass = typically 60-100 cold experts (thin Zipf tail) = 25-40% fewer
  bytes uploaded per layer (TBD: measure).
- Partition per ubatch at runtime: routing histogram (count per expert) -> sort -> cut at c.
  Mode is plan-time, the cut is data-dependent.
- Needs the split-op (6e). Not bit-identical -> PPL-parity gate.

```
experts of layer L, sorted by tokens routed this ubatch
| hot ........ middle ........ | tail (c experts, ~2-3% of tokens) |
|  upload RAM -> buf A -> GPU  |  CPU GEMM from RAM, concurrent     |
|  <- t_upload(rest) @ B_P  -> |  <- t_cpu(tail) @ B_H - B_P     -> |
```

Planner rule:
- `B >= B*` -> `ab_stream` (upload already hidden; the tail saves nothing and costs a merge).
- `B < B*` (bs 2048/4096 on this box, or a slow pipe such as SSD home) -> `cpu_tail` if
  `exposed_saved(c) > merge_overhead`.
- GPU-only strategies (s0/s1) restricted to `ab_stream`; s2/s3 may use either.
- `merge_overhead` and CPU tail GEMM rate at prefill token groups: (TBD: measure).

The old "three-way split" (cache head skips upload / PCIe middle / CPU tail) is exactly
`ab_stream + cpu_tail` with preserved hits served; the head part is now v1 in both modes.

### 6d. miss_policy vs prefill_mode

| aspect               | `miss_policy`                                                        | `prefill_mode = ab_stream + cpu_tail`                                  |
|----------------------|----------------------------------------------------------------------|------------------------------------------------------------------------|
| tier                 | decode / small batch (cache mode)                                    | prefill (A/B mode)                                                     |
| question             | needed expert not in the VRAM cache: fetch or compute on CPU?        | all E needed, none resident: which subset to not upload and run on CPU? |
| granularity          | per miss: ~1-3 experts / layer / token                               | per ubatch / layer: 60-100 experts (the cold tail)                      |
| selection            | cache lookup (resident?) + locality (worth admitting? recency)       | routing histogram of this ubatch, sorted by token mass, cut at c        |
| objective            | per-token latency + cache WARMING (fetch admits, cpu_exec does not)  | bandwidth BALANCE `t_cpu(tail) ~= t_upload(rest)`, minimize exposed upload |
| effect on residency  | fetch changes cache contents                                         | none: A/B buffers are transient                                         |
| when it matters      | every decode token                                                   | only when `B < B*` (upload exposed)                                     |
| shared               | the split-op (6e)                                                    | the split-op (6e)                                                       |

### 6e. The split-op primitive (v1)

Today: one expert sub-graph per layer with 2-3 MUL_MAT_ID nodes - fused gate_up
(src/llama-graph.cpp:2119) or up (:2138) + gate (:2151), and down (:2255), all via
build_lora_mm_id (:1547) - plus add_id bias nodes (:2127 / :2146 / :2263), all indexed by
`selected_experts`. In cache-mode tiers with a CPU branch, and in `cpu_tail` prefill, that
chain is duplicated per branch and merged once after the weighted reduce:

```
                 ids (expert ids 0..E-1, from router; requires ids_cross=1, see below)
                  |
        pool manager / consume-time path (ggml-backend.cpp:2198-2282, extended)
                  |
       +----------+-----------+
       v                      v
  ids_gpu                  ids_cpu
  slot id | -1             expert id | -1
       |                      |
  expert chain              expert chain
  pool view (VRAM)          host expert tensors (page-locked RAM, no copy)
  [ne2 = s slots]           [ne2 = E]
  up/gate MUL_MAT_ID        up/gate MUL_MAT_ID
  (+add_id) -> GLU          (+add_id) -> GLU
  -> down MUL_MAT_ID        -> down MUL_MAT_ID
  (+add_id) -> x weights    (+add_id) -> x weights
  -> reduce n_expert_used   -> reduce n_expert_used
  [n_tokens, H]             [n_tokens, H]
       |                      |
       +---------> ADD <------+      -> layer FFN output
```

- GPU chain: pool view with `ids_gpu` (hits and fetched experts as slot ids; `-1` = skip row).
- CPU chain: the layer's host expert tensors with `ids_cpu` (`-1` = skip row). Weights are
  read from RAM by the CPU kernel; nothing is uploaded for them.
- Kernels: `-1` rows must produce zero output in mul_mat_id AND add_id so GLU / down /
  weighted reduce stay correct. Asserts to remove: ggml/src/ggml-cpu/ggml-cpu.c:1631,
  ggml/src/ggml-cuda/ggml-cuda.cu:2011, ggml/src/ggml-backend.cpp:2231. None of this exists
  today.
- Sched places the two branches by buffer type; the split boundaries and copies it inserts
  are the ones priced in Cost below; no new sync primitive. Boundary = GPU -> CPU -> GPU per
  layer, as s2/s3 have today; adds a D2H barrier per layer in tiers that were GPU-only
  (ggml/src/ggml-backend.cpp:2301-2352).
- Concurrency: `max(t_gpu, t_cpu)` needs the GPU chain launched before the host computes the
  CPU chain (6b). Today's sched (:2429, serial compute_async per split) gives the sum. Open:
  CUDA-graph / stream-memop handshake (FreeToken) vs a sched change (CPU split on a worker
  thread while the GPU split is in flight); (TBD).
- Ids: cache mode requires `ids_cross=1` (routers pinned on the compute GPU, ~60 MB total on
  q35). Today `ids_cross` is documented ALTERNATE-only (src/llama-pshard-plan.h:124) - this is
  a new requirement; section 4 amended.
- Users: `miss_policy` = `cpu_exec` / `hybrid` / `fetch_on_2nd_miss` (first miss) at decode;
  `prefill_mode = ab_stream + cpu_tail` at prefill. Fetch-only tiers emit the single chain as
  today (no CPU branch, no ADD).
- Cost: the ADD over `[n_tokens, H]`, D2H/H2D of the CPU branch's activations, routing and
  partial sum, plus the handoff barrier (bs=1: tiny bytes; the plain-copy barrier drain is
  ~5 ms measured on q8d-s4 today, ggml-backend.cpp:2301-2305; prefill tail: the token groups
  of c experts). (TBD: measure.)
- Version: v1 by user decision. The old text placed it at v2 (section 6 "CPU-execute
  misses"), the earlier build order at v3, and open question 4 (section 11) asked
  implementation cost vs disabling cache mode on those tiers; all superseded (12.4; sections 10 and 11 amended).
- Numerics: two partial sums on two devices, different accumulation order -> NOT
  bit-identical to the single-device kernel. Gates:
  - fetch-only tiers: token-hash identity vs always-fetch (same experts, same kernel, only
    bytes moved differ).
  - any tier with a CPU branch (`cpu_exec`, `hybrid`, `fetch_on_2nd_miss`, `cpu_tail`):
    placement-matched PPL parity. Section 9's "bit-equivalent, non-negotiable" narrows to
    the fetch-only class (section 9 amended).

## 7. Ubatch selection interplay

`B* = t_upload_layer x GPU_rate / flops_per_token` per transfer pipe:
- RAM home over PCIe (30 GB/s): B* ~ 6-8k -> ladder already picks it.
- Disk home over NVMe (5-7 GB/s): B* ~ 32-64k -> giant prefill ubatches; the tier
  ladder already generates them at large n_ctx, planner prices viability (scratch).
- Small-ubatch + persistent-cache prefill (working set U(B) <= s) only wins on slow
  pipes; on this box it loses 2-4x to big-ubatch streaming. Encode as tier pricing,
  not policy.

## 8. Planner and predictor integration

- Registry: per-tier `K`, `s`, `miss_policy`, `prefill_mode`, `hybrid_frac`;
  `pool_mb` derived (+ later per-layer s_l). `pool_mode = ab|cache` retired -
  subsumed by (`prefill_mode`, `miss_policy`), 3b.3. Plan attribute pattern
  identical to overlap=/ids_cross=.
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
  (Amended 12.4: bit-equivalence gates fetch-only tiers; tiers with CPU-executed
  routes - cpu_exec / hybrid / fetch_on_2nd_miss / cpu_tail - change accumulation
  order and gate on placement-matched PPL parity instead.)
- Ledger gains hit-rate columns (per run: mean h, misses/token).
- Perf acceptance: MoE decode cells (q35/q235/DSv4 class) at fixed budgets vs the
  current reference; prefill cells must be neutral (region minimum = today's A/B).
- The GGML_SCHED_DUMP_ALLOC tool verifies the pool region is disjoint from scratch.

## 10. Relation to prior art

- **FreeToken:** the S-slot region with prefill A/B repurposing and preserved area
  is FreeToken's layout; this design grounds the sizing in measured constraints
  (B*, DRAM-bound decode, concavity/water-filling) and folds it into pshard's
  per-tier planning instead of a fixed split.
- **RFC #24528:** hybrid hit/miss execution appears here as the miss policy (v1
  per 12.4; the first draft had it at v2);
  this doc is effectively a concrete implementation proposal on pshard rails
  (planner-priced sizing, per-tier modes, measured-locality adaptation).
- **#26414 / #26003 / #25294 (RAM pinning, lazy experts, disk streaming):** the
  disk-spill case slots in as HOME=disk with RAM as the capacity tier; the pool
  design is unchanged (Section 3), only fetch_cost and B* change.

## 11. Open design points

Decided (section 12) vs not yet designed (this section). Items are numbered 1-21 across
groups A-D; a ref is `11.<group>.<item>` (11.D.21 = item 21, group D). Priority tags are
proposed by the drafter, not user decisions: P0 = blocks the v1 build; P1 = needed before
v1 is planner-honest / ships; P2 = v2. "(was Qn)" = carried over from the previous
open-questions list Q1-Q5 (-> 11.C.16 / 11.B.11 / 11.C.17 / 11.C.12 / 11.C.14); none
dropped (Q4 closed by 12.4, residue in 11.C.12). Labels: "Decided (12.n):" = user decision
or its direct consequence; "Today:" = present code state; "(to add)" = registry/env item
that does not exist yet (no PSHARD_MISS_POLICY, pool_mb, miss_policy or
prefill_mode in src/common/ggml).

### 11.A Budget partition still to do per strategy

s0 (3b.4) and s1 (3b.5) are done. s2, s3 have a constraint row but no allocation
order, no worked map, no crossover budget. s4 retires pending 11.D.21.

| variable          | s1 GPUONLY_ATTNPIN_FFNSTREAM        | s2 DYNAMIC_FFNCPU_ATTNSTREAM                 | s3 STATIC_ATTNPRIO_ALLMODELS                    |
|-------------------|-------------------------------------|----------------------------------------------|-------------------------------------------------|
| K                 | 0 (fixed)                           | 0 (fixed)                                    | 0 (fixed)                                       |
| n_attn_pinned     | free; pin-first (attn+KV, 119 MB/layer @16k) | free; stream default, pin if budget  | free; pin-first                                 |
| s -> pool_mb      | free; prefill tiers >= A/B min (900 MB q35); decode = remainder | free; decode tier may drop below A/B min (cpu_exec needs 0 VRAM experts) | same as s2 |
| pool mode (derived; column retired - 3b.3) | ab on prefill tiers, cache on decode / small-batch tiers | same | same |
| miss_policy       | fetch (only)                        | cpu_exec (allowed set: fetch / cpu_exec / fetch_on_2nd_miss) | cpu_exec / fetch_on_2nd_miss (allowed set as s2) |
| prefill_mode | ab_stream              | ab_stream + cpu_tail                         | ab_stream + cpu_tail                            |
| KV                | rides with attention: pinned layers resident, streamed layers via the KV pipe-shard (llama_memory_pipe_shard_i) | same | same                        |
| dense models      | classic placement (today's path), pool absent | same                               | same                                            |

1. **s1 partition - RESOLVED (3b.5).** Allocation order: fixed -> fetch floor 563.2 MB -> attn+KV pins to 40/40 (staging quantum: 38-pin ties 40-pin exactly at both ctx, 39-pin dominated - probe the 40/40 corner) -> pool; holds at every budget under assumed h. Worked maps @8000 (= s0 plan X, byte-verified) and @2000 (~11.6-12.8 t/s, loses ~2.3-2.6x to measured 30.4 t/s auto STATIC at the same cell = q35-16k-2000, qa/pending-verification.md:130 - cite resolved, not ctx 2048). Viability rule + legacy fallback: 3b.5; all-pins thresholds 5503 MB @16k / 3263 MB @2k; ~30 t/s bound ~4.6-4.9 GB @16k. Dedupe rule + hook: 3b.5. Residue, still open:
   - h(s) cliff caveat: order flips iff measured delta-h > ~0.09 per 1.7 slots/layer (0.42 across 8 -> 16 slots); the doc's h(16) plus this section's extrapolated h(8) straddle it - 11.B.6 counters adjudicate.
   - pool-region resize at tier switch (decode pool 628 MB < prefill A/B 900 MB): new machinery, unpriced; the two @2000 shapes agree within noise so it is not blocking.
   - scratch @ bs=4096 unmeasured -> that tier's viability at 2000 MB TBD; bs<=1024 viability additionally assumes scratch monotone in bs (unmeasured below 2048).
   - the 30.4 t/s cell's executed n_attn_pinned / KV accounting contradicts the unit costs (2200 MB attn + 2560 MB KV > 2000 MB) - pull the run's registry line; ledger row :32 still stale.
   - legacy bs>=8192 prefill tier at 2000 MB under pool-era scratch accounting (1400 MB says no, ledger :33 ran) - reconcile from fresh grid scratch measurements.

2. **P0 - s2 partition.** Open:
   - attn stream-vs-pin threshold under the pool: each pin removes ~2 ms/token of
     streamed attn on q35 @16k.
   - decode pool floor (0 MB allowed?) vs the prefill tier's 900 MB A/B span -> a 900 MB
     residency delta every prompt; priced by 11.B.8.
   - cpu_exec predictor terms not calibrated: expert_bytes / DRAM_bw + t_cpu_matvec
     (~60-80 us at bs=1) + merge (TBD: measure).
   - cpu_tail and the A/B upload both read host DRAM -> t_cpu(tail) and t_upload(rest)
     are not independent (DRAM-bound invariant; FreeToken's residual model B_H - B_P
     applies); needs 11.B.10's concurrent numbers.
3. **P0 - s3 partition.** Open: same as s2 plus the v1 default at bs=1
   (fetch_on_2nd_miss is the candidate; cpu_exec at bs>=16 / verify tiers); whether
   STATIC's unpinned layers keep attention on CPU or stream it (affects 11.C.15);
   ALLMODELS on dense models = unchanged path, confirm the pool code is fully bypassed
   there.
4. **P1 - crossover.** The planner must pick s1 vs s3 per cell by predicted tps. q35
   @2000 MB: today's auto STATIC at 30.4 t/s (q35-16k-2000, qa/pending-verification.md:130;
   executed attn residency vs the unit costs unreconciled - registry pull pending,
   11.A.1 residue) beats pool-s1 at ~11.6-12.8 t/s (3b.5); @8000 MB GPU-only wins.
   s1-side bound computed: falls under ~30 t/s below ~4.6-4.9 GB @16k, all-pins shape at
   5503 MB @16k / 3263 MB @2k (3b.5). The true crossover still needs the s3 partition
   (11.A.3): pool-s3's cpu_exec frees both decode floors, so its side moves too
   (TBD: derive s3, then predict + measure the pair).

### 11.B Planner

5. **P0 - K decision rule beyond the s0 example.** Decided (12.9, 12.12): K priced per
   tier by `prefill_time(tier plan) + switch_cost(prefill plan -> decode plan)`, decode
   plan fixed by decode tps; worth = `sum(exposed upload saved) - switch delta` (q35
   @8000 MB: K=0 at bs=8192 by the ~45 ms switch term; K=9 at bs=2048 by ~275 ms/prompt under the discussion's equal-attention-residency assumption - the derived 8000 MB map prices K=0 there, 3b.4).
   Open:
   - search order: decode tier first (sets residency), prefill tiers conditioned on it -
     rule, or artefact of one example?
   - ladders have 5-7 tiers; find_optimal_ubatch prices cost(active->tier) +
     cost(tier->decode) (src/llama-pshard-plan.h:401-404) - is that pair enough when
     consecutive prefill tiers hold different K?
   - constraint `n_attn_pinned >= K` (a whole-layer pin includes its attn+KV).
   - joint monotonicity of (n_attn_pinned, K, s) in budget is assumed, not shown - does
     the binary search survive three dimensions?
   - which K layers: the example pins layers 0..K-1; exposure is per layer (DSv4
     dense-lead layers) -> pin by predicted exposure, not index? pin_from_back interplay.
   - t_compute per layer at bs=8192 (~23 ms q35) is a predictor estimate; K flips sign
     when the real GEMM time drops below the 15 ms upload (TBD: measure per split).
6. **P0 - h(s) model.** Decided: Zipf assumed first. Counters: measured per-layer
   counters written to the registry after warmup = section 8 doc proposal, not a user
   decision (to add). Open: Zipf parameter source before any counters exist (the worked
   example used h(43) ~ 0.8 and h(12) ~ 0.5 - assumptions, TBD: measure); h at
   batch/verify tiers (token-union of experts, not per-token); counter format, write
   cadence, staleness rule; slope of h(s) as the churn signal that flips miss_policy
   toward cpu_exec / fetch_on_2nd_miss.
7. **P0 - pool_mb derivation vs scratch.** Decided (12.2, 12.10, 12.13): `pool_mb = s x
   expert_bytes x (L-K) + A/B minimum` (to add); scratch is a measured tier requirement
   (q35: bs 8192 ~1400 MB, bs 2048 ~600 MB, bs 1 ~100 MB), never a pool spillover; the
   scratch delta between tiers relabels to pool. Definition (one place): volatile = A/B
   span + scratch delta of the tier pair; preserved = the rest (q35 @8000 MB, tiers 8192
   <-> 1: volatile 900 MB A/B + 1300 MB scratch delta = 2200 MB, preserved 860 MB). Open
   (was 5b residue): preserved/volatile split fixed at 2 layer-widths or planner-chosen;
   does cpu_tail lower the A/B minimum to (E - c) experts x expert_bytes; viability rule
   when remainder < A/B minimum (see 11.A.1); spare-byte uses (a) partial layer pin /
   (b) attn+KV pins / (c) preserved slots / (d) free - the search must price all four,
   per tier.
8. **P0 - switch_ms pricing of pool residency deltas.** Today: switch_cost_ms prices
   |d n_pinned| x switch_layer_mb (sum when pin_from_back differs), |d attn_resident| x
   switch_layer_mb x switch_attn_frac, + switch_head_mb on an output_on_gpu flip
   (src/llama-pshard-plan.h:272-292). Decided (5/8, unchanged): the volatile refill is
   charged to TTFT via switch_ms. Needed per class of bytes:

   | residency class                    | prefill -> decode                                   | decode -> prefill            |
   |------------------------------------|-----------------------------------------------------|------------------------------|
   | preserved slots                    | 0 (survive)                                         | 0 (survive)                  |
   | volatile span (A/B + scratch delta)| refill: lazy misses (~560 MB, ~20 ms once) or warm-load ~28 ms | 0 (metadata drop) |
   | K-layer experts (s_l=E)            | evict = free; upload = 450 MB/layer at ~30 GB/s      | same                         |
   | attn+KV pins                       | upload 119 MB/layer @16k (q35 @8000: 11 pins ~1.3 GB ~45 ms); evict = free | same |

   Open: asymmetry (evict 0, upload bytes/bw) in the pairwise model; KV part of a new pin
   = used cells (grows with position), not the 64 MB @16k maximum - price at switch time
   or at max.
9. **P1 - miss_policy v1 default and 4th value.** Decided (12.15): planner-chosen per
   tier within the strategy's set, applied at tier switch only; registry field and env
   override PSHARD_MISS_POLICY=fetch|cpu_exec|fetch_on_2nd_miss|hybrid (to add);
   fetch_on_2nd_miss = admit-on-second-miss in 5b/6. Proposed v1 defaults when counters
   are absent: bs=1 -> fetch (s0/s1) / fetch_on_2nd_miss (s2/s3); bs>=16 -> cpu_exec.
   Open (was 6 residue):
   - confirm the CPU-lineage default;
   - reconcile with the build order that placed cpu_exec on batch tiers in v2 (TBD:
     decide; split-op v1 makes it feasible in v1);
   - `hybrid` (FreeToken q* split: fetch q* = m x B_P / B_H misses by recency, CPU runs
     the rest) as the 4th value, fraction stored in the registry; on this box B_P
     29.5 GB/s (profiler concurrent PCIe, 11.B.10), B_H 45.7 GB/s -> q*/m ~ 0.65, miss
     time ~1.55x better than fetch-all - predicted, not measured (slice-point B_P ~25 GB/s gives 0.55 / 1.83x - 4b).
10. **P1 - q* fraction from the bench tables.** Today: examples/llama-profiler/
    profiler-cpu.cpp measures each side under the other's load - sliced PCIe upload
    (0.5/2/8/32 MB chunks, 24-chunk bursts) under DRAM stress (profiler-cpu.cpp:118-180)
    and CPU matmul under PCIe stress (Concurrent_GFLOP/s, CPU_Eff %,
    profiler-cpu.cpp:520/530); src/llama-benchmark.cpp:172-186 and :217 ingest them as
    eff_system_bw = DRAM_BW x CPU_Eff, eff_pcie_bw = min(PCIe_Concurrent, DRAM_BW),
    eff_gflops. The 29.5 GB/s concurrent figure comes from this path. Needed: derive
    fraction = pcie_ov / (pcie_ov + cpu_ov) from those entries (candidate: q*/m =
    eff_pcie_bw / peak_system_bw) for the predictor - analytic, no plan-time probing
    (12.24). Open: the CPU-side stress is synthetic streaming reads, not an expert GEMV -
    is it representative (TBD: compare against a real gather + GEMV pair); add a
    same-workload concurrent entry only if not. FreeToken's own gate "hybrid only if CPU
    MoE bw > 2x PCIe gather bw": on the digest's DRAM/PCIe figures the ratio is
    1.55-1.99x (45.7 / 29.5; 45.7 / 23), so the gate would likely fail here if B_H stands
    in for FreeToken's CPU bw (TBD: measure with FreeToken's definitions); we decide by
    priced tps, not that rule - verify the rule is not hiding a real loss.
11. **P2 - s water-filling (v2).** Decided (12.14): v1 uniform s (one registry field, to
    add); v2 per-layer s_l by marginal dh_l(s_l) until pool spent. Open (was Q2): cadence
    - per session, per N tokens, or planner-only (relabel is metadata-only, so cheap);
    go/no-go computable offline now from captured routing data: misses/token uniform vs
    water-filled at equal total slots, <5 % -> v2 never ships, 20-30 % -> first item
    after v1 (TBD: run it).

### 11.C Runtime

12. **P0 - `-1` skip kernels and the split-op (was Q4; closed by 12.4: split-op is v1,
    the alternative of disabling cache mode on batch tiers is dropped).** Today:
    build_moe_ffn emits 2-3 MUL_MAT_ID per layer (fused gate_up, or gate + up, and down;
    src/llama-graph.cpp:2119/2138/2151/2255), then per-expert weighting and an ADD chain
    over n_expert_used (llama-graph.cpp:2285-2295); CUDA/CPU mul_mat_id kernels have no
    -1 skip; no split GPU/CPU op exists; the consume-time copy at
    ggml/src/ggml-backend.cpp:2198-2282 already syncs the ids backend and builds
    used_ids; the scheduler moves an op whose weights sit in a host buffer to a GPU
    backend when ggml_backend_offload_op fires (ggml/src/ggml-backend.cpp:1025-1027; CUDA
    op_offload_min_batch_size default 32, ggml/src/ggml-cuda/ggml-cuda.cu:5407). Design
    (v1):
    - per side one expert-FFN sub-graph (gate/up, activation, down, weighting): GPU side
      over the pool view with ids_gpu (slot ids / -1), CPU side over the host tensor with
      ids_cpu (-1 for hits); -1 rows produce zeros;
    - ADD of the two [n_embd, n_expert_used, n_tokens] outputs before the expert
      aggregation;
    - ids split extends the consume-time copy (ggml-backend.cpp:2198-2282);
    - CPU node backend pinned explicitly (ggml_backend_sched_set_tensor_backend) or
      exempted from op_offload; without it the bs>=32 offload heuristic uploads the host
      tensor at exactly the tiers 11.B.9 assigns cpu_exec and 11.C.13 assigns cpu_tail.
    Open: kernel-side -1 skip in CUDA and CPU mul_mat_id; whether ggml sched runs the CPU
    sub-graph concurrently with the GPU one (FreeToken launches CPU first; sched executes
    splits in order, GPU async) or serialises them; interaction with the pshard
    redirected-split logic; merge cost (one ADD of [n_embd, n_expert_used, n_tokens];
    merging after the aggregation instead is [n_embd, n_tokens]) vs FreeToken's exact
    sum; summation order changes -> PPL gate 11.D.19.
13. **P1 - cpu_tail cold-set cut per ubatch.** Decided (12.18): cut at c where t_cpu(tail
    token mass) ~ t_upload(remaining experts); on this box ~2-3 % of token mass =
    typically 60-100 experts = 25-40 % fewer bytes uploaded; only below B* (bs 2048/4096
    here). Open - timing:
    ```
    compute(L)  : router(L) -> ids(L) -> histogram(L) -> cold set c(L)
    DMA today   : upload experts(L+1) issued during compute(L)     <- before router(L+1) ran
    cpu_tail    : needs cold set c(L+1) at issue time              <- unknown until router(L+1)
    ```
    Candidates: issue L+1 upload after router(L+1) (exposes the upload it meant to hide);
    predict c(L+1) from a stale histogram (previous ubatch / prompt-level); upload
    hot-by-history first, cut the rest on arrival. None priced (TBD: measure). v1/v2 for
    cpu_tail itself unassigned (split-op is v1; cpu_tail rides on it).
14. **P1 - multi-turn protection (was Q5).** Decided (12.8; 5/5b): decode -> prefill
    costs nothing; only the volatile span is overwritten; preserved slots survive;
    promote-on-hit biases hot experts into preserved slots. Open: mid-size prompts
    (~50-500 tokens) run A/B with exposed transfer and kill the volatile span each turn -
    should a short turn run with pool_mode=cache (fetch / cpu_exec misses, no A/B)
    instead; threshold by predicted time; does promote-on-hit alone keep the preserved
    core intact across 50-token turns (TBD: measure hit rate before/after a short turn).
15. **P1 - ids_cross always-on cost.** Today: ids_cross pins the router on the compute
    GPU for ALTERNATE only (src/llama-pshard-plan.cpp:346). Decided (12.16): hard
    requirement of remap + fetch-on-miss -> always-on in cache-mode tiers; routers ~60 MB
    total on q35 (~1.5 MB/layer) - presumably inside the routers/norms ~80 MB fixed line
    (TBD: confirm). Open: for layers whose attention does not run on the compute GPU (s3
    STATIC unpinned layers, if attention stays on CPU) the router pin forces a
    hidden-state hop to GPU and back - cost per layer per token (TBD: measure); verify
    the pin never lands the router in the same split as the expert copy (else the
    consume-time path sees no ids and uploads all 256).
16. **P2 - eviction (was Q1).** v1 = LRU per layer, promote-on-hit into preserved slots
    (5b). Open: LFU-with-decay for bursty locality; FreeToken's recency-on-the-expert
    (not on the slot) for choosing which misses to fetch under hybrid - adopt or not;
    victim choice when the pool is non-uniform (v2 s_l).
17. **P2 - warm start (was Q3).** v1 = lazy: token 1 misses the volatile span (preserved
    core warm from the previous decode; the first prompt of a session misses everything,
    ~560 MB ~20 ms once), charged via switch_ms. v2 candidates: prompt-histogram
    warm-load (~41 KB counters, ~28 ms, overlappable); preserved core alone (free); last
    two layers' A/B contents are valid full sets -> h=1 by remap, free either way. Open:
    measure lazy vs histogram vs preserved-only on q35 decode after a 16k prompt (TBD:
    measure).
18. **P2 - SSD / RAM tiers (HOME=disk).** Decided (12.1): SSD never read by compute; path
    SSD -> RAM -> (VRAM | CPU); fetch RAM <- SSD ~1.8 MB at 5-7 GB/s ~300 us + latency;
    B* on NVMe ~32-64k tokens. Open (was 3/7/10 residue): explicit page-locked RAM expert
    cache with routing-aware LRU from day one vs mmap page cache first; RAM cache slot
    sizing; SSD prefetch for prefill (450 MB/layer at 5-7 GB/s vs ~23 ms compute at
    bs=8192 -> exposed unless the ladder reaches 32-64k tokens; scratch viability there
    unpriced).

### 11.D QA

19. **P0 - PPL-parity gate for hybrid tiers.** Decided (12.4): token-hash bit-identity
    stays the gate for fetch-only configs; hybrid configs (split-op, cpu_tail, q*) change
    summation order -> PPL parity against a placement-matched reference. Open: tolerance
    (TBD: derive from the existing PPL mirror's run-to-run spread); PPL mirror stays
    -fitb only (perf-comparison rules); which cells - one per hybrid miss_policy value and
    one cpu_tail prefill cell minimum.
20. **P1 - ledger columns.** Decided (12.13): new plan columns K, s,
    miss_policy, prefill_mode; per run mean h and misses/token. Open: per-layer h_l dump
    (input to 11.B.11's go/no-go and 11.D.21); how the ledger reads the counters and the
    write cadence (11.B.6); ALTERNATE cells keep reporting until 11.D.21 closes.
21. **P1 - ALTERNATE adjudication (12.25 = user requirement restated at L2394; not from
    the previous Q list).** Today: s4 alternates whole layers' experts between GPU and
    CPU via -ot (layer granularity). Proposed, not adjudicated: the pool leaves s4
    nothing to alternate -> retire. FreeToken engine.py:1185 puts head+tail layers on CPU
    because per-layer miss rates are U-shaped - not alternating. Open: does s4 win any
    cell in the current ledger; does q35's per-layer h_l show the U-shape (TBD: measure);
    if yes, the answer is v2 water-filled s_l (more slots to head/tail layers), not s4.
    Retire s4 only after both.

## 12. Decision log (2026-09-01/02)

User decisions made or restated in the discussion after doc commit 301e1b658 (rows marked
"earlier" were made before it and restated at L2394; rows marked "question" record a user
question whose assistant answer went uncontested). Discussion order. One line each, with
the consequence in the doc.

| #  | decision (user)                                                                 | consequence                                                                                                     |
|----|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 1  | Target = VRAM + RAM + SSD expert cache, not VRAM-only                           | three tiers; SSD -> RAM -> (VRAM or CPU), never SSD -> compute; v1 RAM home only, v2 SSD tier + RAM cache      |
| 2  | MoE budget = pinned non-FFN + compute buffer + KV + expert cache; FFN points into the cache | placement class POOL for routed experts; n_pinned no longer applies to MoE FFN (reinterpreted as K)   |
| 3  | Points, no mannered prose                                                       | doc and discussion style                                                                                        |
| 4  | Split-across-devices execution (GPU hits / CPU misses in one op) is v1 (was v2/v3) | v1 value gate = PPL parity for hybrid configs; token-hash bit-identity only for fetch-only configs; Q4's alternative (disable cache mode on batch tiers) dropped |
| 5  | Partition the budget strategy by strategy, s0 first                             | s0 done; s1/s2/s3 open (11.A); s4 retire candidate                                                             |
| 6  | KV is sharded in s0                                                             | KV rides with attention; attn pin cost = weights + KV (q35 @16k 55 + 64 = 119 MB/layer; @2k ~63 MB); streamed layers' KV via the KV pipe-shard (llama_memory_pipe_shard_i) |
| 7  | s0 keeps whole-layer FFN pins                                                   | s_l = E formulation; s0 = K whole layers (attn + KV + 256 experts, eviction off) for prefill AND decode        |
| 8  | Preserved span idle at prefill is waste                                         | prefill hits served from resident experts is v1 (reverses "skip in v1")                                        |
| 9  | Whole-layer expert pinning is a design decision, not a rule                     | K is a priced per-tier planner decision; objective = prefill + switch + decode over the prompt/decode mix       |
| 10 | Scratch is auto-sized per tier                                                  | scratch = measured tier requirement, never a pool spillover; spare bytes -> (a) partial pin / (b) attn+KV pins / (c) preserved slots / (d) free, priced per tier |
| 11 | Redo s0 at bs=8192 and bs=1                                                     | both tiers illustrated; q35 @8000 MB: K=0 at bs=8192 and bs=1 (decode ~110-130 t/s predicted); K=9 at bs=2048 under equal attention residency, K=0 by the derived 8000 MB map (3b.4) - priced per cell, not a rule |
| 12 | K is a variable (question; assistant answer uncontested)                        | K decoupled from n_attn_pinned; per tier; searched like n_pinned                                                |
| 13 | New variables per strategy                                                      | per-tier set: n_attn_pinned, K, s, pool_mb (derived), prefill_mode (ab_stream / ab_stream + cpu_tail), miss_policy + overlap, ids_cross (forced on in cache mode), output_on_gpu, pin_from_back, switch_ms; pool mode derived - ab on prefill, cache on decode (3b.3); strategy = constraint set; ledger columns K, s, miss_policy, prefill_mode |
| 14 | v1 uniform s; v2 per-layer s_l                                                  | one registry field in v1; v2 water-fill after an offline go/no-go check                                         |
| 15 | miss_policy is a variable; planner decides                                      | planner-chosen per tier within the strategy's allowed set; pool manager executes; applied at tier switch, never mid-tier; env PSHARD_MISS_POLICY=fetch/cpu_exec/fetch_on_2nd_miss/hybrid |
| 16 | ids_cross defined (question; assistant answer uncontested)                      | always-on in cache-mode tiers (routers pinned on compute GPU, ~60 MB q35); a variable only for legacy non-pool streaming tiers |
| 17 | Variables may differ per tier for the same strategy                             | one strategy, N tiers, N variable sets, coupled only by switch_ms (already true today: q35 @4000 s2 n_pinned 9,9,9,8,8) |
| 18 | ab_stream / cpu_tail defined (question; assistant answer uncontested, L2339/L2347) | prefill_mode values ab_stream / ab_stream + cpu_tail; cpu_tail cut c at t_cpu(tail token mass) ~ t_upload(remaining experts), only below B*, needs the split-op, PPL gate |
| 19 | miss_policy vs prefill_mode distinguished (question; assistant answer uncontested, L2351/L2354) | comparison table: per-miss admit decision at decode (cache mode) vs per-ubatch cold-tail cut at prefill (A/B mode); shared primitive = the split-op only |
| 20 | Study FreeToken q*; where do we do this                                         | nowhere as one mechanism today; hook points fixed (pool module, ids split in the consume-time path, per-side expert-FFN sub-graph + ADD, planner constants); miss_policy=hybrid as a 4th value; q* fraction derived from the existing profiler concurrent entries, a same-workload bench pair only if that stress proves unrepresentative (11.B.10) |
| 21 | No k:m ratio strategy (earlier, restated)                                       | no ALTERNATE ratio variable                                                                                     |
| 22 | No L2 runtime switcher (earlier, restated)                                      | miss_policy and every pool attribute change only at tier switch                                                 |
| 23 | No slot carve-out (refuted by measurement 30d3f4ed1)                            | pool region = persistent pshard buffer; region mechanism of 146e03fa8 restored for the persistent pool buffer   |
| 24 | No plan-time probing; the analytic model must work                              | h(s), q* constants, fetch costs come from bench tables and counters, never from probes at plan time             |
| 25 | Confirm whether ALTERNATE ever wins before retiring it (earlier, still binding)  | s4 retirement blocked on 11.D.21                                                                                |
| 26 | Update the design doc; list what is left                                        | this revision; section 11                                                                                       |

Corrected (assistant claims retracted in the same discussion; do not re-propose):
- corrected: "split-op hybrid is v2/v3 with a separate gate" -> v1; PPL-parity gate.
- corrected: "pool hits do not skip prefill uploads in v1; preserved span idle at prefill" -> hits served at prefill in v1; what preserved bytes do at prefill is a per-tier planner decision.
- corrected: "KV fixed by ctx x n_seq, non-negotiable" -> KV_resident = sum over attn-pinned layers; streamed layers' KV moves via the KV pipe-shard.
- corrected: "pinning experts for prefill saves zero time / buys nothing at bs=8192" -> saves max(0, t_upload - t_compute) per layer per pass (q35: ~0 at 8192, ~3 ms at 4096, ~9 ms at 2048) and frees copy-engine + host-DRAM bandwidth even when hidden.
- corrected: "spare -> scratch (bigger ubatch)" -> never; scratch is measured per tier.
- corrected: "drop layer-granular pinning; s0 and s1 converge" -> s0 keeps K free 0..L; s0 and s1 are two design points compared by predicted tps.
- corrected: "planner picks K=0 at bs=8192" (asserted) -> prefill tie; decided by the ~45 ms/prompt switch term; K=9 at bs=2048.
- corrected: map row "attn+KV, 40 layers 4760" read as 40 whole layers -> attention + KV only; all experts in RAM (18 GB page-locked) under K=0.
- corrected: ids_cross listed as a per-tier variable -> always-on in cache mode.
- flagged, not adjudicated: "s4 ALTERNATE: retire - nothing left to alternate" -> awaits 11.D.21.

Corrected on review (2026-09-02; code facts, do not re-propose):
- "build_moe_ffn emits one MUL_MAT_ID per layer" -> 2-3 per layer (src/llama-graph.cpp:2119/2138/2151/2255); the split is per-side expert-FFN sub-graphs + ADD, not two nodes (11.C.12).
- "no concurrent PCIe + CPU bench pair exists" -> the profiler measures each side under the other's load and llama-benchmark.cpp ingests it; only the q* fraction derivation is missing (11.B.10).
- "sched places the CPU-miss node by buffer type" -> the op_offload heuristic moves host-weight ops to the GPU at bs>=32 unless the node is pinned (11.C.12).
