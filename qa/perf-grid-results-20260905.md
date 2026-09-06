# perf grid 2026-09-05: what the numbers say

Run: 371 cells (+4 DSv4 stock perf cells added 2026-09-06), GPC locked 2100 MHz / DRAM 14 GHz,
seed 1234 on perf cells, 128 decode tokens, prompts prompt-512-v2.txt (CONTRIBUTING.md opening,
366 q35 / 348 DSv4 tokens) and prompt-4k-v2.txt (README + HOWTO-add-model, 3978 / 3910 tokens),
binaries a1a7a4881 (the overflow-guard + audit fixes). Ledger and per-cell logs:
`C:/Aditya/grid-results/20260905-gpc2100/`; full tables rendered by `qa/perf-grid-tables.py`
into `results.md` there. Nothing measured at 2505 MHz (the 2026-09-04 ledger) is comparable.

Excluded: 13 STRATEGY_FALLBACK rows (the planner substituted the attn-pin legacy plan for the
executing tier: DSv4 s1 at 8000, and pool fetch on every speculative verify tier where the fetch
floor bs x top_k exceeds the slots), 2 OVER_BUDGET rows (DSv4 + DSpark at 8000: the scheduler's
own re-reserve spilled 1088 MiB outside the arena), 1 OVER_RESERVE row (q35 MTP s1 at 4000/4k),
and 1 degenerate row (q35 4000/4k pool fetch+pred+warm: warm start wrote over a legacy tier's
scratch; reproduced and fixed 2026-09-06, grid-results/repro-20260906/warm.log and fix.log).
Every other row is a single run. Noise band: the one same-plan same-text repeat pair (q35 full/512
legacy auto 82.29 vs forced s3 85.70, identical .gen) differs by 4.1%, and same-plan speculative
pairs by 7-11% per target step, so plain deltas under ~4% and speculative deltas under ~10% are
not findings. Most cross-arm comparisons are cross-text: the arms' numerics differ, the sampled
continuation differs, and pool speed follows the routed text (q35 8000/512: nine arms, nine
texts). Only fetch = fetch+pred, hybrid = plan, base = noovl pairs share a text.

## 1. Stock vs legacy vs pool

Decode t/s, plain decode, 128 tokens (prompt t/s in the full tables):

| set | stock | legacy auto | pool (best) | pool/stock | pool/auto |
|---|---|---|---|---|---|
| q35 4000 / 512 | 47.6 | 47.2 | 61.1 fetch+pred | 1.28x | 1.29x |
| q35 8000 / 512 | 56.5 | 56.7 | 76.6 fetch | 1.36x | 1.35x |
| q35 full / 512 | 83.7 | 82.3 | 84.0 fetch (s3 85.7 best) | 1.00x | 1.02x |
| q35 4000 / 4k | 47.3 | 45.7 | 54.6 fetch | 1.16x | 1.20x |
| q35 8000 / 4k | 56.5 | 54.7 | 76.0 fetch | 1.34x | 1.39x |
| q35 full / 4k | 82.3 | 81.6 | 83.5 fetch | 1.01x | 1.02x |
| DSv4 full / 512 | 11.2 | 12.2 | 18.1 plan=hybrid (fetch 17.7) | 1.61x | 1.49x |
| DSv4 full / 4k | 12.7 | 11.9 | 16.7 poolauto=hybrid (fetch 15.5) | 1.32x | 1.41x |
| DSv4 8000 / 512 | 7.1 | 11.1 | no pool arm (region does not fit); legacy/stock 1.56x | | |
| DSv4 8000 / 4k | 7.6 | 10.7 | no pool arm; legacy/stock 1.41x | | |

Speculative decode, per target step (decode t/s / (1 + accept x n_draft); the arms' numerics
differ, so the sampled text and its acceptance differ even with the seed):

| set | stock | legacy auto | pool (best) | pool/auto |
|---|---|---|---|---|
| q35 MTP 4000 / 512 | 22.4 | 22.3 | 28.8 hybrid (fetch not admissible) | 1.29x |
| q35 MTP 8000 / 512 | 25.7 | 26.2 | 44.8 fetch+pred | 1.71x |
| q35 MTP full / 512 | 37.7 | 39.4 | 52.0 fetch+pred | 1.32x |
| q35 MTP 4000 / 4k | 22.9 | 20.2 | 27.3 hybrid | 1.35x |
| q35 MTP 8000 / 4k | 26.8 | 26.7 | 44.5 fetch+pred | 1.67x |
| q35 MTP full / 4k | 36.4 | 38.4 | 49.4 fetch+pred | 1.29x |
| DSv4 DSpark full / 512 | 1.8 (fit label 3000, actual peak 12.7 GB) | 3.6 | 6.1 hybrid | 1.69x |
| DSv4 DSpark full / 4k | 1.9 (fit label 3000, actual peak 12.7 GB) | 3.3 | 5.8 hybrid | 1.74x |

- The pool wins decode where the budget is tight (q35 4000/8000, +16-39%), ties at full budget
  where legacy pins nearly everything, and wins every speculative set (+29-74% per target step,
  fetch where admissible, hybrid where the verify batch's fetch floor does not fit).
- DSv4 at full budget: pool +49% over legacy auto and +61% over stock at 512; the "+77%" of the
  2026-09-04 ledger was degenerate text. Legacy auto LOSES decode to stock at DSv4 full/4k
  (11.9 vs 12.7) while winning prefill 7.6x.
- Prefill: pool tiers stream the whole expert stack through the A/B pair, so on the 512 prompt
  the pool's prompt speed is flat with budget (q35 702-720 t/s) while legacy scales with what it
  pins (684-1097). On the 4k prompt the pool matches legacy where its own bs=4096 tier is viable
  (q35 8000/full ~4150 vs ~4270); at q35 4000 the forced-pool registry's prefill tiers from
  bs=1024 up are legacy attn-pin substitutes and the prompt ran on one (2924 = s3's 2942); at
  DSv4 full the bs>=4096 pool tiers are unviable and the prompt runs in two 2048 passes (449 vs
  legacy 695).
- Legacy strategies: s3 (STATIC_ATTNPRIO) is the same tier-0 plan as auto in every plain set
  (decode within the noise band) and beats it on 512-prompt prefill by 9-13% (q35) and 17%
  (DSv4 full); s0/s1/s2/s4 never beat auto on plain decode (one speculative tie: DSv4 DSpark
  8000/4k, where auto itself landed s2).
- End-to-end (prompt + 128 tokens; table in the appendix): the pool's decode gain survives where
  its prefill keeps up - q35 4000/512 2.63 s vs auto 3.23 s, q35 8000/512 2.18 vs 2.71, q35
  8000/4k 2.63 vs 3.26, DSv4 full/512 11.0 vs 14.5 (stock 17.2), DSv4 DSpark full/512 14.4 vs
  22.6 - and is cancelled by slower prefill at q35 full/512 (2.03 vs 1.88; s3 1.79), q35 full/4k
  (tie), DSv4 full/4k (poolauto ties auto at 16.3), and reversed on DSv4 DSpark full/4k (42.3 vs
  27.0 s: the pool's 122 t/s 4k prefill).

## 2. Miss policy and the planner's pick

Measured winner at bs=1: fetch in every q35 set (55-84 vs hybrid 52-69), fetch and hybrid tie on
DSv4 (17.7 vs 17.8; 15.5 vs 16.3, inside the band). cpu_admit third of the planner's four
candidates everywhere, cpu_exec last and flat (40-42 q35, 11 DSv4), fetch_on_2nd_miss never best.
The planner picks hybrid over fetch at q35 4000/8000 and at every MTP verify tier where fetch is
admissible: -7/-14/-3/-17% plain, -12 to -19% per step.

The planner's own prices are in every gate log's warmup tier summary (appendix table "predicted
vs measured"): at q35 8000/512 it predicts fetch 62.2 and hybrid 63.0 t/s against measured 76.6
and 65.9 - fetch under-priced 11-31% on q35 and 44-45% on DSv4 (12.2 predicted vs 17.7), hybrid
within 2-5% on q35. That is the pick defect measured directly; the pool-vs-legacy margin on DSv4 is
priced +10% (12.2 vs 11.1) and measures +45%.

Term-level cause (source-derived, reproducing all 16 picks; inferred): the
hybrid term prices the CPU chain as overlapped with the uploads, max(q t_fetch, (m-q) t_cpu) +
t_split; the measured bs=1 rows fit the serial sum q t_fetch + (m-q) t_cpu + t_split (q35 s=26:
0.212 ms/layer measured, 0.203 sum, 0.155 max). Pricing the sum with the code's own constants
flips all 8 wrong picks to fetch and keeps the right ones (the MTP verify tier is bs=3, floor 24
slots; its per-layer figures were recomputed by the verifiers with the same conclusion). Do it together
with t_cpu: the FLOP floor (cpu_matmul_floor_gflops 93.8, an f16 B=64 profile entry) prices the
bs=1 expert matvec 1.7x (q35) / 2.2x (DSv4) too slow against the DRAM-rate term that matches the
measurements (q35 0.042 vs 0.0395 ms/expert measured); with the FLOP floor the sum form would flip
DSv4 to fetch against its measured tie. Cross-budget regression of the fetch rows gives t_fetch
0.055 ms/miss (priced 0.0655, ~18% high) with a 9.6-9.9 ms/token common part. The sum is an
empirical fit to overlap-ON rows: execution overlap stays on (section 3); only the PRICE of the
overlapped chain is the serial sum. t_serve (0.10 ms/layer, ~0.03 measured) is common to all
policies and cannot move a policy pick; no cell isolates it, so the "t_serve" framing of the
2026-09-04 note is retired. These are the model's constants and are proposed, not applied.

## 3. Predictor, warm start, overlap, fetch_on_2nd_miss

- Predictor (PSHARD_POOL_PREDICT=1): raises h everywhere (+0.03..+0.12) but decode moves -2% on
  average on q35 fetch (17 of 18 q35 pairs negative, cpu_admit -8%), +1..+2.6% on DSv4 at the
  noise floor, +0.4..+1.9% on MTP. Default stays off; not a ladder or planner option.
- Warm start + per-layer allocation: h +0.004..+0.019, decode -2.5%..+0.9% vs fetch+pred; 3-5%
  below plain fetch on q35 at 8000 and full, level on DSv4 (in-spread). And one correctness failure: at q35 4000 with the 4k prompt the
  A/B->cache flip fired when a LEGACY prefill tier took over and warm_start wrote 583 MiB over
  that tier's scratch (garbage from token 1). Fixed (seed only while the pool is active; WARM
  alone reproduced it, ALLOC alone was byte-identical). Recommendation: remove both knobs.
- Overlap (GGML_SCHED_NO_CPU_OVERLAP): certified in all 16 pairs with identical counters:
  hybrid +15..+19%, cpu_admit +6..+27%. Hard-code on; delete the switch and the planner's serial
  pricing branch.
- fetch_on_2nd_miss: never the best pool arm (10-26% behind the set's best in 7 of 8 sets, a tie
  in one); its DSv4 gate hash is one the stock never produces (see 5). Delete the policy.

## 4. The ladder

poolauto lands the planner's pick within -4..+4% decode in all 8 plain sets and inherits the
hybrid-over-fetch cost (-7..-15% vs the best policy at q35 4000/8000). Its cost is in the prompt:
-8..-13% vs the forced pool at 512 (the mixed-ladder switch and a colder pool after a legacy
prefill tier) and up to -20% at 4k/full. On DSv4 + DSpark the ladder never selects the pool for
the verify tier (ran STATIC_ATTNPRIO at legacy's 3.5 steps/s vs the forced hybrid's 6.0) while
its prefill used the pool's slow 4k path (122 vs legacy 580 t/s): worst of both; disable poolauto
for DSpark until the planner bounds pool tiers by scratch at plan time (design 11.C.19) and the
verify-tier pricing at s=17-18 is recalibrated.

Legacy auto's prefill tier: on 512 prompts auto loses 8-13% (q35) / 17% (DSv4 full) to forced
s3, on 4k prompts the same choice wins +38-42% (q35) / +53% (DSv4 full). The gate logs show why
(appendix "prefill tier predicted vs measured"): at bs=512 the planner prices the streaming tier
ABOVE attn-pin (q35 4000: 1173 vs 899; DSv4 full: 101 vs 64) while measurement ranks them the
other way (684 vs 779; 87 vs 105), and the pool's bs=512 tier is over-predicted at every budget
(1079 vs 697-715); at 4k the executed bs=4096 tiers are under-predicted 2.4x on q35 (1765 vs 4173)
and 8-12x on DSv4 (57 vs 695) with the ranking right. The prefill model has the wrong slope in
batch size on q35 and the wrong level on DSv4; the tier-switch time itself is not in the
artifacts.

## 5. Correctness

76 valid gates produce 10 distinct hashes, 8 of them produced by stock itself at some budget
(stock flips its own hash with budget in every family, as its fit moves attention layers between
CPU and GPU). Every legacy arm, pool fetch (+pred, +warm at 512), hybrid and cpu_admit land on a
stock hash; divergences are at tokens ~11-32 between coherent continuations. PPL mirror: max
deviation vs same-budget stock 0.004 (q35) / 0.015 (DSv4, both lower than stock) against a
stock-vs-stock budget spread of 0.0008 / 0.0076 and a reported +/- of ~0.11. Two DSv4 hashes are
NOT stock-class: pool cpu_exec and pool fetch_on_2nd_miss at full/512 diverge at token ~9. The
design certifies CPU-route policies by PPL parity; the PPL cells added for them (2026-09-06) show
every pool policy - fetch, hybrid, cpu_admit, cpu_exec, fetch_on_2nd_miss - at the SAME perplexity
per (model, budget): DSv4 full 4.1891 for all five (stock 4.2042, legacy 4.1910); q35 4.0829 at 4000,
4.0822 at 8000 and full (stock 4.0849/4.0857/4.0857). Certified: the two odd hashes are near-tie
flips with no perplexity signal.

## 6. Pricing constants (PSHARD_POOL_ZIPF / CPU_GBS / CPU_GFLOPS)

h(s) = zipf_mass(s)/zipf_mass(E), alpha 0.95: reproduces the measured fetch h within 0.04 for
s <= 81 (s=26 0.601 vs 0.594; s=81 0.814 vs 0.789; DSv4 s=23 0.609 vs 0.573) except the 4k
prompt at s=25 (0.518 measured vs 0.587, the thrash regime), and over-predicts at s~170 (0.867
vs 0.923). Whether that plateau is a hit-rate ceiling (~0.875) or the 128-token window still
filling (the 32-token gate reads 0.724 at the same s) the grid cannot say - no longer decode
exists. Least-squares alpha: 0.97 (512), 0.935 (4k), 0.98 (DSv4): freeze 0.95, delete the env.
Per-miss cost by cross-budget regression of the fetch rows: 0.055 ms/miss (q35, both prompts,
intercept 9.6-9.9 ms/token); t_cpu from cpu_exec minus that intercept: 0.040 ms/expert on q35
(DRAM-bound), 0.21-0.26 on DSv4 (one budget). No cell set CPU_GBS/CPU_GFLOPS; keep the overrides
until the t_cpu term is fixed in code (section 2), then delete them - the constants beneath them
are what mis-prices hybrid: cpu_exec is predicted 28.8 vs 41-42 measured on q35 (0.69x), 5.2 vs
11.1 on DSv4 (0.47x).

## 7. Defects the grid found and their state

- Arena overflow at the pool tiers (bs=4096/8192 at DSv4 full, bs>=1024 at q35 4000): fixed
  before the grid (9b6d6488d, a1a7a4881); the ledger's OVER_BUDGET/SCHED_GREW statuses watch it.
- Scheduler's own re-reserve spill (DSv4 + DSpark at 8000, +1088 MiB, twice): instrumented
  (WARN with bytes and graph size; runner status SCHED_GREW); the graph that spills is the
  speculative target's first decode graph, 10757 nodes, whose shape differs from the bs=512
  tier's reserved graph - root cause open.
- Warm start seeding while disengaged: fixed 2026-09-06 (10a87cc1d), knobs recommended for removal.
- MTP reserve short by ~178 MiB under streaming strategies (s1 at 4000/4k, OVER_RESERVE): the
  head-lever charge covers the lever only; the general fix is measuring the reserve after the
  plan under its placement - open.
- Missing PPL parity for CPU-route policies: cells added and run; all five pool policies are
  perplexity-identical per budget on both models (section 5).

## 8. Env-var and policy decisions (per the user's "remove after finalized")

| knob | grid verdict | action |
|---|---|---|
| GGML_SCHED_NO_CPU_OVERLAP | overlap wins all 16 pairs | delete; overlap always on |
| fetch_on_2nd_miss | dominated in 8/8 sets at s=22-170; its s<15 churn niche untested | delete the policy (a scope decision) |
| PSHARD_POOL_WARM / ALLOC | never wins; one corruption (fixed) | delete both |
| PSHARD_POOL_PREDICT | -2% q35, +1..+2.6% DSv4 (noise) | keep off; keep as opt-in |
| PSHARD_POOL_ZIPF | 0.95 fits s <= 81 (except the 4k thrash row); s~170 plateau open | freeze 0.95, delete env |
| PSHARD_POOL_CPU_GBS / CPU_GFLOPS | unused; constants beneath mis-price hybrid | fix the t_cpu term in code first (section 2), then delete the envs |
| GGML_CUDA_STAGE_ASYNC / RING_MB / THREADS | ran stably on all 109 DSv4 cells; on/off was not an arm | hard-code the defaults; keep the ring until the pinned hot-expert tier |

## Appendix: full tables (qa/perf-grid-tables.py over the ledger, 2026-09-06)

### rendered from 20260905-gpc2100

`# idle_used=850 full_mva=14500 clocks_sm=2092 clocks_mem=14001 git=2a4e242f0 date=2026-09-05T19:07`
`# idle_used=781 full_mva=14500 clocks_sm=2100 clocks_mem=14001 git=a1a7a4881 date=2026-09-05T20:43`
`# idle_used=581 full_mva=14500 clocks_sm=2100 clocks_mem=14001 git=a1a7a4881 date=2026-09-06T05:19`
`# idle_used=581 full_mva=14500 clocks_sm=2092 clocks_mem=14001 git=10a87cc1d date=2026-09-06T05:26`

Generated by `qa/perf-grid-tables.py` from `ledger.csv`. Regime: GPC 2100 MHz, seed 1234, 128 decode tokens, prompts prompt-512-v2.txt (366/348 tokens) and prompt-4k-v2.txt (3978/3910 tokens). Cells: 391. Statuses: OK 375, STRATEGY_FALLBACK 13, OVER_BUDGET 2, OVER_RESERVE 1.

## Excluded rows

| cell | reason |
|---|---|
| dsv4-8000-512-none-s1 | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=8230) |
| dsv4-8000-512-none-s1-gate | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=8242) |
| q35mtp-4000-512-mtp-pool_fetch | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=4334) |
| q35mtp-4000-512-mtp-pool_fetch-pred | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=4333) |
| dsv4-8000-512-dspark-auto | OVER_BUDGET (strategy_active=DYNAMIC_FFNCPU_ATTNSTREAM, vram_delta=9442) |
| dsv4-8000-512-dspark-s1 | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=9391) |
| dsv4-8000-512-dspark-s3 | OVER_BUDGET (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=9369) |
| dsv4-full-512-dspark-pool_fetch | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=14803) |
| dsv4-full-512-dspark-pool_fetch-pred | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=14808) |
| dsv4-8000-4k-none-s1 | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=8249) |
| q35mtp-4000-4k-mtp-s1 | OVER_RESERVE (strategy_active=GPUONLY_ATTNPIN_FFNSTREAM, vram_delta=4341) |
| q35mtp-4000-4k-mtp-pool_fetch | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=4221) |
| q35mtp-4000-4k-mtp-pool_fetch-pred | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=4221) |
| dsv4-8000-4k-dspark-s1 | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=8343) |
| dsv4-full-4k-dspark-pool_fetch | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=14603) |
| dsv4-full-4k-dspark-pool_fetch-pred | STRATEGY_FALLBACK (strategy_active=STATIC_ATTNPRIO_ALLMODELS, vram_delta=14603) |
| q35-4000-4k-none-pool_fetch-pred-warm | DEGENERATE output: no-words (130 chars, 1 tokens) |

## q35 plain decode: prompt t/s / decode t/s (pool rows add h)

| arm | 4000/512 | 8000/512 | full/512 | 4000/4k | 8000/4k | full/4k |
|---|---|---|---|---|---|---|
| stock | 185 / 47.6 | 234 / 56.5 | 407 / 83.7 | 409 / 47.3 | 521 / 56.5 | 906 / 82.3 |
| auto | 684 / 47.2 | 780 / 56.7 | 1097 / 82.3 | 4173 / 45.7 | 4261 / 54.7 | 4285 / 81.6 |
| s0 | 706 / 2.4 | 795 / 3.0 | 1056 / 5.2 | 4304 / 2.4 | 4269 / 3.0 | 4327 / 5.2 |
| s1 | 720 / 29.8 | 825 / 35.0 | 1135 / 47.4 | 4238 / 29.8 | 4320 / 34.0 | 4340 / 46.6 |
| s2 | 218 / 16.6 | 291 / 20.5 | 502 / 32.3 | 253 / 16.1 | 328 / 19.8 | 624 / 31.5 |
| s3 | 779 / 47.5 | 899 / 56.0 | 1202 / 85.7 | 2942 / 46.0 | 3004 / 52.9 | 3096 / 81.2 |
| s4 | 312 / 42.1 | 383 / 51.1 | 618 / 75.9 | 424 / 40.9 | 488 / 49.4 | 837 / 73.5 |
| pool:fetch | 710 / 59.9 h=0.601 | 708 / 76.6 h=0.814 | 707 / 84.0 h=0.867 | 2924 / 54.6 h=0.518 | 4147 / 76.0 h=0.815 | 4143 / 83.5 h=0.878 |
| pool:fetch+pred | 702 / 61.0 h=0.694 | 707 / 76.2 h=0.878 | 707 / 81.2 h=0.919 | 2925 / 53.8 h=0.617 | 4153 / 72.7 h=0.862 | 4161 / 79.6 h=0.923 |
| pool:fetch+pred+warm | 702 / 60.4 h=0.699 | 715 / 74.3 h=0.897 | 702 / 79.9 h=0.928 | excluded (DEGEN) | 4147 / 73.3 h=0.879 | 4138 / 79.8 h=0.930 |
| pool:hybrid | 708 / 55.2 h=0.591 | 708 / 65.8 h=0.785 | 697 / 69.0 h=0.821 | 2922 / 52.4 h=0.561 | 4155 / 64.5 h=0.780 | 4133 / 68.0 h=0.825 |
| pool:hybrid+pred | 706 / 53.2 h=0.690 | 707 / 63.0 h=0.859 | 707 / 65.3 h=0.903 | 2909 / 51.5 h=0.659 | 4144 / 63.0 h=0.867 | 4160 / 66.4 h=0.912 |
| pool:hybrid+noovl | 708 / 47.4 h=0.591 | 715 / 57.3 h=0.785 | 713 / 59.9 h=0.821 | 2923 / 45.3 h=0.561 | 4146 / 55.9 h=0.780 | 4130 / 57.3 h=0.825 |
| pool:cpu_admit | 706 / 48.7 h=0.633 | 715 / 63.8 h=0.818 | 706 / 67.0 h=0.869 | 2895 / 42.4 h=0.575 | 4131 / 63.3 h=0.843 | 4156 / 66.8 h=0.878 |
| pool:cpu_admit+pred | 702 / 43.6 h=0.695 | 713 / 58.9 h=0.885 | 713 / 63.4 h=0.924 | 2920 / 39.7 h=0.657 | 4149 / 56.7 h=0.871 | 4152 / 62.0 h=0.919 |
| pool:cpu_admit+noovl | 706 / 44.4 h=0.633 | 708 / 56.9 h=0.818 | 705 / 59.9 h=0.869 | 2916 / 39.9 h=0.575 | 4151 / 57.1 h=0.843 | 4137 / 59.0 h=0.878 |
| pool:cpu_exec | 713 / 41.0 h=0.000 | 707 / 42.5 h=0.000 | 705 / 42.3 h=0.000 | 2924 / 41.2 h=0.000 | 4146 / 40.9 h=0.000 | 4126 / 40.4 h=0.000 |
| pool:fetch_on_2nd_miss | 703 / 54.4 h=0.581 | 708 / 66.7 h=0.761 | 711 / 68.4 h=0.784 | 2921 / 53.9 h=0.609 | 4146 / 65.9 h=0.762 | 4139 / 66.3 h=0.780 |
| pool:plan | 699 / 55.6 h=0.591 (hybrid) | 713 / 66.2 h=0.785 (hybrid) | 708 / 83.6 h=0.867 (fetch) | 2920 / 52.8 h=0.561 (hybrid) | 4134 / 62.8 h=0.780 (hybrid) | 4124 / 82.4 h=0.878 (fetch) |
| poolauto | 640 / 55.4 h=0.598 (hybrid) | 619 / 67.1 h=0.803 (hybrid) | 625 / 83.3 h=0.867 (fetch) | 4123 / 50.8 h=0.507 (hybrid) | 3793 / 64.8 h=0.803 (hybrid) | 3304 / 80.6 h=0.879 (fetch) |

## dsv4 plain decode: prompt t/s / decode t/s (pool rows add h)

| arm | 8000/512 | full/512 | 8000/4k | full/4k |
|---|---|---|---|---|
| stock | 54 / 7.1 | 59 / 11.2 | 83 / 7.6 | 92 / 12.7 |
| auto | 103 / 11.1 | 87 / 12.2 | 428 / 10.7 | 695 / 11.9 |
| s0 | 71 / 0.2 | 74 / 0.3 | 428 / 0.2 | 439 / 0.2 |
| s1 | excluded (STRATEGY_FALLBACK) | 88 / 8.9 | excluded (STRATEGY_FALLBACK) | 697 / 7.6 |
| s2 | 18 / 2.7 | 20 / 2.8 | 19 / 2.6 | 21 / 2.7 |
| s3 | 104 / 11.4 | 105 / 12.1 | 438 / 11.0 | 454 / 11.9 |
| s4 | 27 / 3.6 | 30 / 10.3 | 34 / 2.3 | 36 / 9.9 |
| pool:fetch | - | 86 / 17.7 h=0.609 | - | 449 / 15.5 h=0.579 |
| pool:fetch+pred | - | 86 / 17.9 h=0.713 | - | 446 / 15.8 h=0.677 |
| pool:fetch+pred+warm | - | 86 / 17.7 h=0.718 | - | 448 / 15.7 h=0.681 |
| pool:hybrid | - | 87 / 17.8 h=0.592 | - | 450 / 16.2 h=0.566 |
| pool:hybrid+pred | - | 87 / 17.5 h=0.706 | - | 451 / 16.3 h=0.688 |
| pool:hybrid+noovl | - | 87 / 15.3 h=0.592 | - | 451 / 14.0 h=0.566 |
| pool:cpu_admit | - | 87 / 14.0 h=0.597 | - | 450 / 13.3 h=0.580 |
| pool:cpu_admit+pred | - | 86 / 14.4 h=0.706 | - | 451 / 13.6 h=0.698 |
| pool:cpu_admit+noovl | - | 87 / 11.0 h=0.597 | - | 450 / 10.5 h=0.580 |
| pool:cpu_exec | - | 86 / 11.1 h=0.000 | - | 449 / 11.3 h=0.000 |
| pool:fetch_on_2nd_miss | - | 86 / 16.4 h=0.570 | - | 448 / 15.1 h=0.540 |
| pool:plan | - | 87 / 18.1 h=0.611 (hybrid) | - | 451 / 16.0 h=0.560 (hybrid) |
| poolauto | - | 87 / 17.7 h=0.594 (hybrid) | - | 449 / 16.7 h=0.590 (hybrid) |

## q35mtp + mtp speculative: decode t/s (accept %) [target steps/s]

| arm | 4000/512 | 8000/512 | full/512 | 4000/4k | 8000/4k | full/4k |
|---|---|---|---|---|---|---|
| stock | 44.5 (49.2%) [22.4] | 51.7 (50.3%) [25.7] | 70.4 (43.4%) [37.7] | 46.9 (52.3%) [22.9] | 53.4 (49.6%) [26.8] | 76.3 (54.8%) [36.4] |
| auto | 38.4 (36.2%) [22.2] | 56.6 (57.9%) [26.2] | 76.4 (47.0%) [39.4] | 40.1 (49.2%) [20.2] | 53.3 (50.0%) [26.7] | 67.1 (37.4%) [38.4] |
| s0 | 4.0 (39.8%) [2.2] | 4.7 (34.8%) [2.8] | 8.6 (44.8%) [4.5] | 4.5 (52.3%) [2.2] | 5.6 (54.4%) [2.7] | 8.5 (45.2%) [4.5] |
| s1 | 24.2 (42.1%) [13.1] | 27.7 (42.1%) [15.0] | 34.7 (42.1%) [18.9] | excluded (OVER_RESERVE) | 29.0 (49.2%) [14.6] | 37.8 (55.2%) [17.9] |
| s2 | 17.7 (38.3%) [10.0] | 20.3 (37.8%) [11.6] | 30.8 (44.9%) [16.2] | 20.5 (54.8%) [9.8] | 23.2 (51.5%) [11.4] | 30.1 (44.2%) [16.0] |
| s3 | 40.7 (38.5%) [23.0] | 51.2 (43.8%) [27.2] | 77.2 (47.0%) [39.8] | 43.3 (46.6%) [22.4] | 51.6 (47.3%) [26.5] | 76.7 (52.8%) [37.3] |
| s4 | 37.4 (42.4%) [20.2] | 42.6 (34.1%) [25.3] | 74.6 (50.7%) [37.0] | 38.9 (46.6%) [20.1] | 38.5 (36.1%) [22.3] | 66.7 (40.5%) [36.8] |
| pool:fetch | excluded (STRATEGY_FALLBACK) | 79.0 (39.8%) [44.0] h=0.774 | 92.9 (39.8%) [51.7] h=0.866 | excluded (STRATEGY_FALLBACK) | 88.6 (50.7%) [43.9] h=0.768 | 100.0 (51.5%) [49.2] h=0.839 |
| pool:fetch+pred | excluded (STRATEGY_FALLBACK) | 80.6 (39.8%) [44.8] h=0.855 | 93.5 (39.8%) [52.0] h=0.921 | excluded (STRATEGY_FALLBACK) | 89.6 (50.7%) [44.5] h=0.844 | 100.4 (51.5%) [49.4] h=0.903 |
| pool:hybrid | 56.3 (47.7%) [28.8] h=0.433 | 91.8 (66.9%) [39.3] h=0.728 | 85.8 (54.4%) [41.1] h=0.795 | 52.7 (46.6%) [27.3] h=0.412 | 75.8 (49.6%) [38.0] h=0.718 | 79.6 (43.1%) [42.7] h=0.815 |
| pool:plan | 55.8 (47.7%) [28.6] h=0.433 | 90.7 (66.9%) [38.8] h=0.728 | 88.6 (54.4%) [42.4] h=0.795 | 51.2 (46.6%) [26.5] h=0.412 | 70.8 (49.6%) [35.5] h=0.718 | 79.8 (43.1%) [42.8] h=0.815 |
| poolauto | 53.7 (48.0%) [27.4] h=0.432 | 67.4 (36.6%) [38.9] h=0.752 | 83.5 (54.4%) [39.9] h=0.795 | 57.1 (66.9%) [24.4] h=0.434 | 76.9 (57.8%) [35.6] h=0.708 | 86.5 (64.9%) [37.6] h=0.759 |

Prompt t/s for the same cells:

| arm | 4000/512 | 8000/512 | full/512 | 4000/4k | 8000/4k | full/4k |
|---|---|---|---|---|---|---|
| stock | 182 | 234 | 391 | 401 | 530 | 869 |
| auto | 666 | 787 | 1089 | 2897 | 3921 | 3907 |
| s3 | 722 | 880 | 1186 | 2808 | 2866 | 3201 |
| pool:fetch | - | 718 | 720 | - | 3501 | 3832 |
| pool:hybrid | 709 | 711 | 714 | 2783 | 3511 | 3812 |
| pool:plan | 713 | 709 | 718 | 2803 | 3511 | 3825 |
| poolauto | 652 | 641 | 637 | 2906 | 3501 | 3193 |

## dsv4 + dspark speculative: decode t/s (accept %) [target steps/s]

| arm | 8000/512 | full/512 | 8000/4k | full/4k |
|---|---|---|---|---|
| stock | 3.1 (24.7%) [1.8] @3000 | 3.1 (24.7%) [1.8] @3000 | 3.8 (35.4%) [1.8] @3000 | 3.8 (35.4%) [1.8] @3000 |
| auto | excluded (OVER_BUDGET) | 7.0 (31.9%) [3.6] | 3.1 (31.9%) [1.6] | 6.4 (30.0%) [3.3] |
| s0 | 0.4 (22.5%) [0.2] | 0.5 (37.7%) [0.2] | 0.5 (40.6%) [0.2] | 0.5 (32.3%) [0.2] |
| s1 | excluded (STRATEGY_FALLBACK) | 5.5 (22.7%) [3.3] | excluded (STRATEGY_FALLBACK) | 6.0 (32.3%) [3.0] |
| s2 | 3.3 (34.2%) [1.6] | 3.4 (31.0%) [1.7] | 3.3 (33.8%) [1.6] | 3.6 (36.2%) [1.7] |
| s3 | excluded (OVER_BUDGET) | 6.9 (31.9%) [3.5] | 4.1 (33.6%) [2.1] | 6.7 (30.0%) [3.5] |
| s4 | 1.3 (27.9%) [0.7] | 7.0 (33.5%) [3.5] | 1.6 (47.2%) [0.7] | 7.4 (40.8%) [3.3] |
| pool:fetch | - | excluded (STRATEGY_FALLBACK) | - | excluded (STRATEGY_FALLBACK) |
| pool:fetch+pred | - | excluded (STRATEGY_FALLBACK) | - | excluded (STRATEGY_FALLBACK) |
| pool:hybrid | - | 12.6 (36.0%) [6.1] h=0.446 | - | 12.3 (37.0%) [5.8] h=0.440 |
| pool:plan | - | 12.5 (36.0%) [6.0] h=0.446 | - | 12.3 (37.0%) [5.8] h=0.440 |
| poolauto | - | 8.6 (49.3%) [3.5] | - | 6.8 (31.3%) [3.5] |

Prompt t/s for the same cells:

| arm | 8000/512 | full/512 | 8000/4k | full/4k |
|---|---|---|---|---|
| stock | 24 | 24 | 51 | 51 |
| auto | - | 84 | 389 | 580 |
| s3 | - | 104 | 384 | 432 |
| pool:fetch | - | - | - | - |
| pool:hybrid | - | 86 | - | 124 |
| pool:plan | - | 86 | - | 124 |
| poolauto | - | 82 | - | 122 |

## Correctness gates (32 tokens, greedy): md5 groups per (model, budget, prompt)

| model/budget/prompt | groups |
|---|---|
| dsv4/14500/4k | 083900c5fce2: auto, pool:fetch; 434885d69097: stock |
| dsv4/14500/512 | a4406c7d1a91: stock, auto, s0, s2, s3, pool:fetch, pool:fetch+pred, pool:fetch+pred+warm, pool:hybrid; 15781302de8b: s1, s4, pool:cpu_admit; fb9eed1eaacf: pool:cpu_exec; 88957c0ab95f: pool:fetch_on_2nd_miss |
| dsv4/8000/4k | 083900c5fce2: stock, auto |
| dsv4/8000/512 | a4406c7d1a91: auto, s1*, s2, s3, s4; 15781302de8b: stock, s0 |
| q35/14500/4k | 1f7410463c40: stock, auto, pool:fetch |
| q35/14500/512 | c5aacfaa3646: stock, auto, s0, s1, s2, s3, s4, pool:fetch, pool:fetch+pred, pool:fetch+pred+warm, pool:cpu_exec; e5ef51b77992: pool:hybrid, pool:cpu_admit, pool:fetch_on_2nd_miss |
| q35/4000/4k | 1f7410463c40: stock, auto, pool:fetch |
| q35/4000/512 | c5aacfaa3646: s1, s3, s4, pool:fetch, pool:fetch+pred, pool:fetch+pred+warm, pool:hybrid, pool:cpu_admit, pool:cpu_exec; e5ef51b77992: stock, auto, s0, s2, pool:fetch_on_2nd_miss |
| q35/8000/4k | 1f7410463c40: auto, pool:fetch; c55d8e1c1695: stock |
| q35/8000/512 | c5aacfaa3646: auto, s0, s1, s2, s3, pool:fetch, pool:fetch+pred, pool:fetch+pred+warm, pool:cpu_exec; e5ef51b77992: stock, s4, pool:hybrid, pool:cpu_admit, pool:fetch_on_2nd_miss |

`*` = status not OK. Stock is the reference; a different hash on a pshard arm is a near-tie flip unless the PPL mirror disagrees.

## Perplexity mirror (ppl-docs-v2.txt, c2048 x 8 chunks)

| model/budget | stock | legacy auto | pool fetch |
|---|---|---|---|
| dsv4/14500 | 4.2042 | 4.1910 | 4.1891 |
| dsv4/8000 | 4.1966 | 4.1995 | - |
| q35/14500 | 4.0857 | 4.0816 | 4.0822 |
| q35/4000 | 4.0849 | 4.0838 | 4.0829 |
| q35/8000 | 4.0857 | 4.0887 | 4.0822 |

## Pool counters (perf rows): h and misses/token

| cell | policy | slots | h | misses/token | decode t/s |
|---|---|---|---|---|---|
| q35-4000-512-none-pool_fetch | fetch | 26 | 0.601 | 127.7 | 59.92 |
| q35-4000-512-none-pool_fetch-pred | fetch | 26 | 0.694 | 97.8 | 61.05 |
| q35-4000-512-none-pool_fetch-pred-warm | fetch | 26 | 0.699 | 96.2 | 60.39 |
| q35-4000-512-none-pool_hybrid | hybrid | 26 | 0.591 | 131.0 | 55.17 |
| q35-4000-512-none-pool_hybrid-pred | hybrid | 26 | 0.690 | 99.2 | 53.21 |
| q35-4000-512-none-pool_hybrid-noovl | hybrid | 26 | 0.591 | 131.0 | 47.37 |
| q35-4000-512-none-pool_cpu_admit | cpu_admit | 26 | 0.633 | 117.4 | 48.72 |
| q35-4000-512-none-pool_cpu_admit-pred | cpu_admit | 26 | 0.695 | 97.7 | 43.56 |
| q35-4000-512-none-pool_cpu_admit-noovl | cpu_admit | 26 | 0.633 | 117.4 | 44.41 |
| q35-4000-512-none-pool_cpu_exec | cpu_exec | 26 | 0.000 | 320.0 | 40.97 |
| q35-4000-512-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 26 | 0.581 | 134.2 | 54.35 |
| q35-4000-512-none-pool_plan | hybrid | 26 | 0.591 | 131.0 | 55.58 |
| q35-4000-512-none-poolauto | hybrid | 26 | 0.598 | 128.5 | 55.38 |
| q35-8000-512-none-pool_fetch | fetch | 81 | 0.814 | 59.6 | 76.59 |
| q35-8000-512-none-pool_fetch-pred | fetch | 81 | 0.878 | 39.1 | 76.21 |
| q35-8000-512-none-pool_fetch-pred-warm | fetch | 81 | 0.897 | 33.1 | 74.32 |
| q35-8000-512-none-pool_hybrid | hybrid | 81 | 0.785 | 68.9 | 65.85 |
| q35-8000-512-none-pool_hybrid-pred | hybrid | 81 | 0.859 | 45.1 | 63.04 |
| q35-8000-512-none-pool_hybrid-noovl | hybrid | 81 | 0.785 | 68.9 | 57.33 |
| q35-8000-512-none-pool_cpu_admit | cpu_admit | 81 | 0.818 | 58.3 | 63.84 |
| q35-8000-512-none-pool_cpu_admit-pred | cpu_admit | 81 | 0.885 | 36.9 | 58.85 |
| q35-8000-512-none-pool_cpu_admit-noovl | cpu_admit | 81 | 0.818 | 58.3 | 56.90 |
| q35-8000-512-none-pool_cpu_exec | cpu_exec | 81 | 0.000 | 320.0 | 42.48 |
| q35-8000-512-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 81 | 0.761 | 76.4 | 66.68 |
| q35-8000-512-none-pool_plan | hybrid | 81 | 0.785 | 68.9 | 66.20 |
| q35-8000-512-none-poolauto | hybrid | 81 | 0.803 | 63.2 | 67.07 |
| q35-full-512-none-pool_fetch | fetch | 170 | 0.867 | 42.5 | 84.00 |
| q35-full-512-none-pool_fetch-pred | fetch | 170 | 0.919 | 25.8 | 81.18 |
| q35-full-512-none-pool_fetch-pred-warm | fetch | 170 | 0.928 | 22.9 | 79.89 |
| q35-full-512-none-pool_hybrid | hybrid | 170 | 0.821 | 57.3 | 69.00 |
| q35-full-512-none-pool_hybrid-pred | hybrid | 170 | 0.903 | 31.1 | 65.33 |
| q35-full-512-none-pool_hybrid-noovl | hybrid | 170 | 0.821 | 57.3 | 59.85 |
| q35-full-512-none-pool_cpu_admit | cpu_admit | 170 | 0.869 | 42.0 | 66.97 |
| q35-full-512-none-pool_cpu_admit-pred | cpu_admit | 170 | 0.924 | 24.3 | 63.39 |
| q35-full-512-none-pool_cpu_admit-noovl | cpu_admit | 170 | 0.869 | 42.0 | 59.92 |
| q35-full-512-none-pool_cpu_exec | cpu_exec | 170 | 0.000 | 320.0 | 42.31 |
| q35-full-512-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 170 | 0.784 | 69.0 | 68.41 |
| q35-full-512-none-pool_plan | fetch | 170 | 0.867 | 42.5 | 83.60 |
| q35-full-512-none-poolauto | fetch | 170 | 0.867 | 42.5 | 83.30 |
| dsv4-full-512-none-pool_fetch | fetch | 23 | 0.609 | 100.9 | 17.70 |
| dsv4-full-512-none-pool_fetch-pred | fetch | 23 | 0.713 | 74.0 | 17.88 |
| dsv4-full-512-none-pool_fetch-pred-warm | fetch | 23 | 0.718 | 72.8 | 17.70 |
| dsv4-full-512-none-pool_hybrid | hybrid | 23 | 0.592 | 105.3 | 17.76 |
| dsv4-full-512-none-pool_hybrid-pred | hybrid | 23 | 0.706 | 76.0 | 17.51 |
| dsv4-full-512-none-pool_hybrid-noovl | hybrid | 23 | 0.592 | 105.3 | 15.26 |
| dsv4-full-512-none-pool_cpu_admit | cpu_admit | 23 | 0.597 | 104.0 | 13.99 |
| dsv4-full-512-none-pool_cpu_admit-pred | cpu_admit | 23 | 0.706 | 75.8 | 14.36 |
| dsv4-full-512-none-pool_cpu_admit-noovl | cpu_admit | 23 | 0.597 | 104.0 | 11.04 |
| dsv4-full-512-none-pool_cpu_exec | cpu_exec | 23 | 0.000 | 258.0 | 11.12 |
| dsv4-full-512-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 23 | 0.570 | 111.1 | 16.35 |
| dsv4-full-512-none-pool_plan | hybrid | 23 | 0.611 | 100.3 | 18.09 |
| dsv4-full-512-none-poolauto | hybrid | 23 | 0.594 | 104.7 | 17.71 |
| q35mtp-4000-512-mtp-pool_hybrid | hybrid | 17 | 0.433 | 378.6 | 56.280 |
| q35mtp-4000-512-mtp-pool_plan | hybrid | 17 | 0.433 | 378.6 | 55.835 |
| q35mtp-4000-512-mtp-poolauto | hybrid | 17 | 0.432 | 379.6 | 53.714 |
| q35mtp-8000-512-mtp-pool_fetch | fetch | 70 | 0.774 | 151.4 | 79.037 |
| q35mtp-8000-512-mtp-pool_fetch-pred | fetch | 70 | 0.855 | 97.2 | 80.569 |
| q35mtp-8000-512-mtp-pool_hybrid | hybrid | 70 | 0.728 | 184.7 | 91.838 |
| q35mtp-8000-512-mtp-pool_plan | hybrid | 70 | 0.728 | 184.7 | 90.666 |
| q35mtp-8000-512-mtp-poolauto | hybrid | 70 | 0.752 | 169.2 | 67.352 |
| q35mtp-full-512-mtp-pool_fetch | fetch | 157 | 0.866 | 89.8 | 92.870 |
| q35mtp-full-512-mtp-pool_fetch-pred | fetch | 157 | 0.921 | 52.8 | 93.497 |
| q35mtp-full-512-mtp-pool_hybrid | hybrid | 157 | 0.795 | 137.1 | 85.804 |
| q35mtp-full-512-mtp-pool_plan | hybrid | 157 | 0.795 | 137.1 | 88.568 |
| q35mtp-full-512-mtp-poolauto | hybrid | 157 | 0.795 | 137.1 | 83.453 |
| dsv4-full-512-dspark-pool_hybrid | hybrid | 18 | 0.446 | 357.4 | 12.610 |
| dsv4-full-512-dspark-pool_plan | hybrid | 18 | 0.446 | 357.4 | 12.530 |
| q35-4000-4k-none-pool_fetch | fetch | 25 | 0.518 | 154.1 | 54.63 |
| q35-4000-4k-none-pool_fetch-pred | fetch | 25 | 0.617 | 122.5 | 53.76 |
| q35-4000-4k-none-pool_fetch-pred-warm | fetch | 25 | 0.993 | 2.4 | 91.95 (excluded) |
| q35-4000-4k-none-pool_hybrid | hybrid | 25 | 0.561 | 140.5 | 52.39 |
| q35-4000-4k-none-pool_hybrid-pred | hybrid | 25 | 0.659 | 109.1 | 51.52 |
| q35-4000-4k-none-pool_hybrid-noovl | hybrid | 25 | 0.561 | 140.5 | 45.32 |
| q35-4000-4k-none-pool_cpu_admit | cpu_admit | 25 | 0.575 | 136.0 | 42.40 |
| q35-4000-4k-none-pool_cpu_admit-pred | cpu_admit | 25 | 0.657 | 109.9 | 39.74 |
| q35-4000-4k-none-pool_cpu_admit-noovl | cpu_admit | 25 | 0.575 | 136.0 | 39.92 |
| q35-4000-4k-none-pool_cpu_exec | cpu_exec | 25 | 0.000 | 320.0 | 41.17 |
| q35-4000-4k-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 25 | 0.609 | 125.2 | 53.89 |
| q35-4000-4k-none-pool_plan | hybrid | 25 | 0.561 | 140.5 | 52.78 |
| q35-4000-4k-none-poolauto | hybrid | 25 | 0.507 | 157.8 | 50.77 |
| q35-8000-4k-none-pool_fetch | fetch | 79 | 0.815 | 59.2 | 75.96 |
| q35-8000-4k-none-pool_fetch-pred | fetch | 79 | 0.862 | 44.3 | 72.65 |
| q35-8000-4k-none-pool_fetch-pred-warm | fetch | 79 | 0.879 | 38.6 | 73.31 |
| q35-8000-4k-none-pool_hybrid | hybrid | 79 | 0.780 | 70.3 | 64.54 |
| q35-8000-4k-none-pool_hybrid-pred | hybrid | 79 | 0.867 | 42.4 | 63.01 |
| q35-8000-4k-none-pool_hybrid-noovl | hybrid | 79 | 0.780 | 70.3 | 55.86 |
| q35-8000-4k-none-pool_cpu_admit | cpu_admit | 79 | 0.843 | 50.2 | 63.28 |
| q35-8000-4k-none-pool_cpu_admit-pred | cpu_admit | 79 | 0.871 | 41.2 | 56.72 |
| q35-8000-4k-none-pool_cpu_admit-noovl | cpu_admit | 79 | 0.843 | 50.2 | 57.10 |
| q35-8000-4k-none-pool_cpu_exec | cpu_exec | 79 | 0.000 | 320.0 | 40.88 |
| q35-8000-4k-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 79 | 0.762 | 76.1 | 65.90 |
| q35-8000-4k-none-pool_plan | hybrid | 79 | 0.780 | 70.3 | 62.83 |
| q35-8000-4k-none-poolauto | hybrid | 79 | 0.803 | 62.9 | 64.76 |
| q35-full-4k-none-pool_fetch | fetch | 169 | 0.878 | 39.1 | 83.49 |
| q35-full-4k-none-pool_fetch-pred | fetch | 169 | 0.923 | 24.8 | 79.63 |
| q35-full-4k-none-pool_fetch-pred-warm | fetch | 169 | 0.930 | 22.4 | 79.81 |
| q35-full-4k-none-pool_hybrid | hybrid | 169 | 0.825 | 56.1 | 67.95 |
| q35-full-4k-none-pool_hybrid-pred | hybrid | 169 | 0.912 | 28.1 | 66.37 |
| q35-full-4k-none-pool_hybrid-noovl | hybrid | 169 | 0.825 | 56.1 | 57.32 |
| q35-full-4k-none-pool_cpu_admit | cpu_admit | 169 | 0.878 | 39.1 | 66.79 |
| q35-full-4k-none-pool_cpu_admit-pred | cpu_admit | 169 | 0.919 | 26.0 | 62.01 |
| q35-full-4k-none-pool_cpu_admit-noovl | cpu_admit | 169 | 0.878 | 39.1 | 58.96 |
| q35-full-4k-none-pool_cpu_exec | cpu_exec | 169 | 0.000 | 320.0 | 40.42 |
| q35-full-4k-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 169 | 0.780 | 70.4 | 66.28 |
| q35-full-4k-none-pool_plan | fetch | 169 | 0.878 | 39.1 | 82.42 |
| q35-full-4k-none-poolauto | fetch | 169 | 0.879 | 38.9 | 80.64 |
| dsv4-full-4k-none-pool_fetch | fetch | 22 | 0.579 | 108.5 | 15.50 |
| dsv4-full-4k-none-pool_fetch-pred | fetch | 22 | 0.677 | 83.4 | 15.82 |
| dsv4-full-4k-none-pool_fetch-pred-warm | fetch | 22 | 0.681 | 82.4 | 15.73 |
| dsv4-full-4k-none-pool_hybrid | hybrid | 22 | 0.566 | 111.9 | 16.25 |
| dsv4-full-4k-none-pool_hybrid-pred | hybrid | 22 | 0.688 | 80.5 | 16.32 |
| dsv4-full-4k-none-pool_hybrid-noovl | hybrid | 22 | 0.566 | 111.9 | 14.02 |
| dsv4-full-4k-none-pool_cpu_admit | cpu_admit | 22 | 0.580 | 108.4 | 13.32 |
| dsv4-full-4k-none-pool_cpu_admit-pred | cpu_admit | 22 | 0.698 | 78.0 | 13.64 |
| dsv4-full-4k-none-pool_cpu_admit-noovl | cpu_admit | 22 | 0.580 | 108.4 | 10.47 |
| dsv4-full-4k-none-pool_cpu_exec | cpu_exec | 22 | 0.000 | 258.0 | 11.26 |
| dsv4-full-4k-none-pool_fetch_on_2nd_miss | fetch_on_2nd_miss | 22 | 0.540 | 118.8 | 15.11 |
| dsv4-full-4k-none-pool_plan | hybrid | 22 | 0.560 | 113.6 | 16.00 |
| dsv4-full-4k-none-poolauto | hybrid | 22 | 0.590 | 105.8 | 16.74 |
| q35mtp-4000-4k-mtp-pool_hybrid | hybrid | 15 | 0.412 | 387.9 | 52.696 |
| q35mtp-4000-4k-mtp-pool_plan | hybrid | 15 | 0.412 | 387.9 | 51.217 |
| q35mtp-4000-4k-mtp-poolauto | hybrid | 19 | 0.434 | 374.6 | 57.130 |
| q35mtp-8000-4k-mtp-pool_fetch | fetch | 69 | 0.768 | 149.2 | 88.571 |
| q35mtp-8000-4k-mtp-pool_fetch-pred | fetch | 69 | 0.844 | 100.4 | 89.632 |
| q35mtp-8000-4k-mtp-pool_hybrid | hybrid | 69 | 0.718 | 193.4 | 75.761 |
| q35mtp-8000-4k-mtp-pool_plan | hybrid | 69 | 0.718 | 193.4 | 70.828 |
| q35mtp-8000-4k-mtp-poolauto | hybrid | 69 | 0.708 | 189.9 | 76.895 |
| q35mtp-full-4k-mtp-pool_fetch | fetch | 156 | 0.839 | 105.8 | 100.013 |
| q35mtp-full-4k-mtp-pool_fetch-pred | fetch | 156 | 0.903 | 63.5 | 100.407 |
| q35mtp-full-4k-mtp-pool_hybrid | hybrid | 156 | 0.815 | 123.7 | 79.554 |
| q35mtp-full-4k-mtp-pool_plan | hybrid | 156 | 0.815 | 123.7 | 79.761 |
| q35mtp-full-4k-mtp-poolauto | hybrid | 156 | 0.759 | 167.4 | 86.523 |
| dsv4-full-4k-dspark-pool_hybrid | hybrid | 17 | 0.440 | 374.3 | 12.303 |
| dsv4-full-4k-dspark-pool_plan | hybrid | 17 | 0.440 | 374.3 | 12.268 |

## End-to-end seconds per cell (prompt + decode tokens, from the two throughput columns; load excluded)

### q35

| arm | 4000/512 | 8000/512 | full/512 | 4000/4k | 8000/4k | full/4k |
|---|---|---|---|---|---|---|
| stock | 4.65 s | 3.81 s | 2.42 s | 12.40 s | 9.89 s | 5.93 s |
| auto | 3.23 s | 2.71 s | 1.88 s | 3.73 s | 3.26 s | 2.48 s |
| s3 | 3.14 s | 2.67 s | 1.79 s | 4.11 s | 3.73 s | 2.85 s |
| pool:fetch | 2.63 s | 2.18 s | 2.03 s | 3.69 s | 2.63 s | 2.48 s |
| pool:fetch+pred | 2.60 s | 2.18 s | 2.08 s | 3.72 s | 2.71 s | 2.55 s |
| pool:hybrid | 2.82 s | 2.45 s | 2.37 s | 3.79 s | 2.93 s | 2.83 s |
| pool:plan | 2.81 s | 2.43 s | 2.04 s | 3.77 s | 2.98 s | 2.51 s |
| poolauto | 2.86 s | 2.48 s | 2.11 s | 3.47 s | 3.01 s | 2.78 s |

### dsv4

| arm | 8000/512 | full/512 | 8000/4k | full/4k |
|---|---|---|---|---|
| stock | 24.41 s | 17.19 s | 63.74 s | 52.68 s |
| auto | 14.84 s | 14.45 s | 21.05 s | 16.30 s |
| s3 | 14.50 s | 13.82 s | 20.43 s | 19.24 s |
| pool:fetch | - | 11.21 s | - | 16.91 s |
| pool:fetch+pred | - | 11.14 s | - | 16.79 s |
| pool:hybrid | - | 11.15 s | - | 16.50 s |
| pool:plan | - | 11.01 s | - | 16.61 s |
| poolauto | - | 11.17 s | - | 16.30 s |

### q35mtp + mtp

| arm | 4000/512 | 8000/512 | full/512 | 4000/4k | 8000/4k | full/4k |
|---|---|---|---|---|---|---|
| stock | 4.95 s | 4.08 s | 2.77 s | 12.68 s | 9.92 s | 6.28 s |
| auto | 3.91 s | 2.74 s | 2.04 s | 4.64 s | 3.45 s | 2.94 s |
| s3 | 3.73 s | 2.98 s | 1.99 s | 4.44 s | 3.91 s | 2.93 s |
| pool:fetch | - | 2.14 s | 1.90 s | - | 2.59 s | 2.34 s |
| pool:fetch+pred | - | 2.12 s | 1.89 s | - | 2.57 s | 2.33 s |
| pool:hybrid | 2.84 s | 1.94 s | 2.02 s | 3.92 s | 2.86 s | 2.68 s |
| pool:plan | 2.86 s | 1.96 s | 1.97 s | 3.98 s | 2.98 s | 2.67 s |
| poolauto | 2.96 s | 2.50 s | 2.12 s | 3.66 s | 2.84 s | 2.76 s |

### dsv4 + dspark

| arm | 8000/512 | full/512 | 8000/4k | full/4k |
|---|---|---|---|---|
| stock | 56.31 s (@3000, peak 12765 MiB) | 56.31 s (@3000, peak 12765 MiB) | 111.37 s (@3000, peak 12723 MiB) | 111.37 s (@3000, peak 12723 MiB) |
| auto | - | 22.56 s | 51.36 s | 27.04 s |
| s3 | - | 21.95 s | 42.12 s | 28.25 s |
| pool:hybrid | - | 14.37 s | - | 42.29 s |
| pool:plan | - | 14.44 s | - | 42.37 s |
| poolauto | - | 19.50 s | - | 51.03 s |

## Planner predicted vs measured decode t/s (tier 0 of the gate log's warmup summary vs the perf row)

| cell | strategy | predicted | measured | measured/predicted |
|---|---|---|---|---|
| q35-4000-512-none-auto | STATIC_ATTNPRIO_ALLMODELS | 44.7 | 47.16 | 1.06x |
| q35-4000-512-none-s0 | GPUONLY_LAYERPIN_LAYERSTREAM | 2.4 | 2.39 | 1.00x |
| q35-4000-512-none-s1 | GPUONLY_ATTNPIN_FFNSTREAM | 26.0 | 29.78 | 1.15x |
| q35-4000-512-none-s2 | DYNAMIC_FFNCPU_ATTNSTREAM | 17.9 | 16.58 | 0.93x |
| q35-4000-512-none-s3 | STATIC_ATTNPRIO_ALLMODELS | 44.7 | 47.54 | 1.06x |
| q35-4000-512-none-s4 | DYNAMIC_FFN_ALTERNATE | 36.8 | 42.09 | 1.14x |
| q35-4000-512-none-pool_fetch | EXPERT_POOL | 49.6 | 59.92 | 1.21x |
| q35-4000-512-none-pool_fetch-pred | EXPERT_POOL | 49.6 | 61.05 | 1.23x |
| q35-4000-512-none-pool_fetch-pred-warm | EXPERT_POOL | 49.6 | 60.39 | 1.22x |
| q35-4000-512-none-pool_hybrid | EXPERT_POOL | 54.1 | 55.17 | 1.02x |
| q35-4000-512-none-pool_cpu_admit | EXPERT_POOL | 45.5 | 48.72 | 1.07x |
| q35-4000-512-none-pool_cpu_exec | EXPERT_POOL | 28.8 | 40.97 | 1.42x |
| q35-4000-512-none-pool_fetch_on_2nd_miss | EXPERT_POOL | 45.7 | 54.35 | 1.19x |
| q35-8000-512-none-auto | STATIC_ATTNPRIO_ALLMODELS | 52.3 | 56.70 | 1.08x |
| q35-8000-512-none-s0 | GPUONLY_LAYERPIN_LAYERSTREAM | 3.0 | 3.04 | 1.01x |
| q35-8000-512-none-s1 | GPUONLY_ATTNPIN_FFNSTREAM | 30.6 | 34.97 | 1.14x |
| q35-8000-512-none-s2 | DYNAMIC_FFNCPU_ATTNSTREAM | 21.9 | 20.50 | 0.94x |
| q35-8000-512-none-s3 | STATIC_ATTNPRIO_ALLMODELS | 52.3 | 56.03 | 1.07x |
| q35-8000-512-none-s4 | DYNAMIC_FFN_ALTERNATE | 43.5 | 51.08 | 1.17x |
| q35-8000-512-none-pool_fetch | EXPERT_POOL | 62.2 | 76.59 | 1.23x |
| q35-8000-512-none-pool_fetch-pred | EXPERT_POOL | 62.2 | 76.21 | 1.23x |
| q35-8000-512-none-pool_fetch-pred-warm | EXPERT_POOL | 62.2 | 74.32 | 1.19x |
| q35-8000-512-none-pool_hybrid | EXPERT_POOL | 63.0 | 65.85 | 1.05x |
| q35-8000-512-none-pool_cpu_admit | EXPERT_POOL | 56.2 | 63.84 | 1.14x |
| q35-8000-512-none-pool_cpu_exec | EXPERT_POOL | 28.8 | 42.48 | 1.47x |
| q35-8000-512-none-pool_fetch_on_2nd_miss | EXPERT_POOL | 56.4 | 66.68 | 1.18x |
| q35-full-512-none-auto | STATIC_ATTNPRIO_ALLMODELS | 74.5 | 82.29 | 1.10x |
| q35-full-512-none-s0 | GPUONLY_LAYERPIN_LAYERSTREAM | 5.4 | 5.19 | 0.96x |
| q35-full-512-none-s1 | GPUONLY_ATTNPIN_FFNSTREAM | 42.1 | 47.41 | 1.13x |
| q35-full-512-none-s2 | DYNAMIC_FFNCPU_ATTNSTREAM | 34.3 | 32.34 | 0.94x |
| q35-full-512-none-s3 | STATIC_ATTNPRIO_ALLMODELS | 74.5 | 85.70 | 1.15x |
| q35-full-512-none-s4 | DYNAMIC_FFN_ALTERNATE | 63.7 | 75.88 | 1.19x |
| q35-full-512-none-pool_fetch | EXPERT_POOL | 75.5 | 84.00 | 1.11x |
| q35-full-512-none-pool_fetch-pred | EXPERT_POOL | 75.5 | 81.18 | 1.08x |
| q35-full-512-none-pool_fetch-pred-warm | EXPERT_POOL | 75.5 | 79.89 | 1.06x |
| q35-full-512-none-pool_hybrid | EXPERT_POOL | 67.2 | 69.00 | 1.03x |
| q35-full-512-none-pool_cpu_admit | EXPERT_POOL | 67.2 | 66.97 | 1.00x |
| q35-full-512-none-pool_cpu_exec | EXPERT_POOL | 28.8 | 42.31 | 1.47x |
| q35-full-512-none-pool_fetch_on_2nd_miss | EXPERT_POOL | 67.2 | 68.41 | 1.02x |
| dsv4-8000-512-none-auto | STATIC_ATTNPRIO_ALLMODELS | 10.6 | 11.07 | 1.04x |
| dsv4-8000-512-none-s0 | GPUONLY_LAYERPIN_LAYERSTREAM | 0.2 | 0.24 | 1.20x |
| dsv4-8000-512-none-s2 | DYNAMIC_FFNCPU_ATTNSTREAM | 3.0 | 2.72 | 0.91x |
| dsv4-8000-512-none-s3 | STATIC_ATTNPRIO_ALLMODELS | 10.6 | 11.38 | 1.07x |
| dsv4-8000-512-none-s4 | DYNAMIC_FFN_ALTERNATE | 4.4 | 3.62 | 0.82x |
| dsv4-full-512-none-auto | STATIC_ATTNPRIO_ALLMODELS | 11.1 | 12.17 | 1.10x |
| dsv4-full-512-none-s0 | GPUONLY_LAYERPIN_LAYERSTREAM | 0.3 | 0.26 | 0.87x |
| dsv4-full-512-none-s1 | GPUONLY_ATTNPIN_FFNSTREAM | 7.1 | 8.90 | 1.25x |
| dsv4-full-512-none-s2 | DYNAMIC_FFNCPU_ATTNSTREAM | 3.2 | 2.83 | 0.88x |
| dsv4-full-512-none-s3 | STATIC_ATTNPRIO_ALLMODELS | 11.1 | 12.08 | 1.09x |
| dsv4-full-512-none-s4 | DYNAMIC_FFN_ALTERNATE | 8.9 | 10.27 | 1.15x |
| dsv4-full-512-none-pool_fetch | EXPERT_POOL | 12.2 | 17.70 | 1.45x |
| dsv4-full-512-none-pool_fetch-pred | EXPERT_POOL | 12.2 | 17.88 | 1.47x |
| dsv4-full-512-none-pool_fetch-pred-warm | EXPERT_POOL | 12.2 | 17.70 | 1.45x |
| dsv4-full-512-none-pool_hybrid | EXPERT_POOL | 12.9 | 17.76 | 1.38x |
| dsv4-full-512-none-pool_cpu_admit | EXPERT_POOL | 8.9 | 13.99 | 1.57x |
| dsv4-full-512-none-pool_cpu_exec | EXPERT_POOL | 5.2 | 11.12 | 2.14x |
| dsv4-full-512-none-pool_fetch_on_2nd_miss | EXPERT_POOL | 10.2 | 16.35 | 1.60x |
| q35-4000-4k-none-auto | STATIC_ATTNPRIO_ALLMODELS | 41.8 | 45.71 | 1.09x |
| q35-4000-4k-none-pool_fetch | EXPERT_POOL | 46.7 | 54.63 | 1.17x |
| q35-8000-4k-none-auto | STATIC_ATTNPRIO_ALLMODELS | 49.4 | 54.70 | 1.11x |
| q35-8000-4k-none-pool_fetch | EXPERT_POOL | 57.8 | 75.96 | 1.31x |
| q35-full-4k-none-auto | STATIC_ATTNPRIO_ALLMODELS | 68.7 | 81.60 | 1.19x |
| q35-full-4k-none-pool_fetch | EXPERT_POOL | 69.4 | 83.49 | 1.20x |
| dsv4-8000-4k-none-auto | STATIC_ATTNPRIO_ALLMODELS | 9.6 | 10.66 | 1.11x |
| dsv4-full-4k-none-auto | STATIC_ATTNPRIO_ALLMODELS | 9.9 | 11.90 | 1.20x |
| dsv4-full-4k-none-pool_fetch | EXPERT_POOL | 10.8 | 15.50 | 1.44x |

Prefill tier predicted vs measured prompt t/s (the tier at bs >= the prompt's ubatch; predicted from the same gate log):

| cell | tier | strategy | predicted | measured | measured/predicted |
|---|---|---|---|---|---|
| q35-4000-512-none-auto | bs=512 | GPUONLY_LAYERPIN_LAYERSTREAM | 1172.7 | 683.8 | 0.58x |
| q35-4000-512-none-s0 | bs=512 | GPUONLY_LAYERPIN_LAYERSTREAM | 1172.7 | 705.9 | 0.60x |
| q35-4000-512-none-s1 | bs=512 | GPUONLY_ATTNPIN_FFNSTREAM | 1126.3 | 720.2 | 0.64x |
| q35-4000-512-none-s2 | bs=512 | DYNAMIC_FFNCPU_ATTNSTREAM | 143.1 | 218.4 | 1.53x |
| q35-4000-512-none-s3 | bs=512 | STATIC_ATTNPRIO_ALLMODELS | 899.2 | 778.8 | 0.87x |
| q35-4000-512-none-s4 | bs=512 | DYNAMIC_FFN_ALTERNATE | 334.7 | 312.5 | 0.93x |
| q35-4000-512-none-pool_fetch | bs=512 | EXPERT_POOL | 1079.3 | 710.3 | 0.66x |
| q35-4000-512-none-pool_fetch-pred | bs=512 | EXPERT_POOL | 1079.3 | 702.5 | 0.65x |
| q35-4000-512-none-pool_fetch-pred-warm | bs=512 | EXPERT_POOL | 1079.3 | 701.5 | 0.65x |
| q35-4000-512-none-pool_hybrid | bs=512 | EXPERT_POOL | 1079.3 | 707.9 | 0.66x |
| q35-4000-512-none-pool_cpu_admit | bs=512 | EXPERT_POOL | 1079.3 | 705.5 | 0.65x |
| q35-4000-512-none-pool_cpu_exec | bs=512 | EXPERT_POOL | 1079.3 | 712.7 | 0.66x |
| q35-4000-512-none-pool_fetch_on_2nd_miss | bs=512 | EXPERT_POOL | 1079.3 | 703.2 | 0.65x |
| q35-8000-512-none-auto | bs=512 | GPUONLY_LAYERPIN_LAYERSTREAM | 1362.4 | 780.3 | 0.57x |
| q35-8000-512-none-s0 | bs=512 | GPUONLY_LAYERPIN_LAYERSTREAM | 1362.4 | 795.5 | 0.58x |
| q35-8000-512-none-s1 | bs=512 | GPUONLY_ATTNPIN_FFNSTREAM | 1324.8 | 825.0 | 0.62x |
| q35-8000-512-none-s2 | bs=512 | DYNAMIC_FFNCPU_ATTNSTREAM | 186.1 | 291.2 | 1.56x |
| q35-8000-512-none-s3 | bs=512 | STATIC_ATTNPRIO_ALLMODELS | 1060.2 | 899.1 | 0.85x |
| q35-8000-512-none-s4 | bs=512 | DYNAMIC_FFN_ALTERNATE | 414.6 | 382.9 | 0.92x |
| q35-8000-512-none-pool_fetch | bs=512 | EXPERT_POOL | 1079.3 | 708.0 | 0.66x |
| q35-8000-512-none-pool_fetch-pred | bs=512 | EXPERT_POOL | 1079.3 | 706.9 | 0.65x |
| q35-8000-512-none-pool_fetch-pred-warm | bs=512 | EXPERT_POOL | 1079.3 | 714.9 | 0.66x |
| q35-8000-512-none-pool_hybrid | bs=512 | EXPERT_POOL | 1079.3 | 708.4 | 0.66x |
| q35-8000-512-none-pool_cpu_admit | bs=512 | EXPERT_POOL | 1079.3 | 714.9 | 0.66x |
| q35-8000-512-none-pool_cpu_exec | bs=512 | EXPERT_POOL | 1079.3 | 706.8 | 0.65x |
| q35-8000-512-none-pool_fetch_on_2nd_miss | bs=512 | EXPERT_POOL | 1079.3 | 708.4 | 0.66x |
| q35-full-512-none-auto | bs=512 | GPUONLY_ATTNPIN_FFNSTREAM | 1864.2 | 1097.0 | 0.59x |
| q35-full-512-none-s0 | bs=512 | GPUONLY_LAYERPIN_LAYERSTREAM | 1799.1 | 1056.0 | 0.59x |
| q35-full-512-none-s1 | bs=512 | GPUONLY_ATTNPIN_FFNSTREAM | 1864.2 | 1134.8 | 0.61x |
| q35-full-512-none-s2 | bs=512 | DYNAMIC_FFNCPU_ATTNSTREAM | 363.4 | 502.4 | 1.38x |
| q35-full-512-none-s3 | bs=512 | STATIC_ATTNPRIO_ALLMODELS | 1544.3 | 1202.1 | 0.78x |
| q35-full-512-none-s4 | bs=512 | DYNAMIC_FFN_ALTERNATE | 692.2 | 617.8 | 0.89x |
| q35-full-512-none-pool_fetch | bs=512 | EXPERT_POOL | 1079.3 | 706.6 | 0.65x |
| q35-full-512-none-pool_fetch-pred | bs=512 | EXPERT_POOL | 1079.3 | 707.4 | 0.66x |
| q35-full-512-none-pool_fetch-pred-warm | bs=512 | EXPERT_POOL | 1079.3 | 702.3 | 0.65x |
| q35-full-512-none-pool_hybrid | bs=512 | EXPERT_POOL | 1079.3 | 697.2 | 0.65x |
| q35-full-512-none-pool_cpu_admit | bs=512 | EXPERT_POOL | 1079.3 | 705.8 | 0.65x |
| q35-full-512-none-pool_cpu_exec | bs=512 | EXPERT_POOL | 1079.3 | 704.6 | 0.65x |
| q35-full-512-none-pool_fetch_on_2nd_miss | bs=512 | EXPERT_POOL | 1079.3 | 710.8 | 0.66x |
| dsv4-8000-512-none-auto | bs=512 | STATIC_ATTNPRIO_ALLMODELS | 62.2 | 103.5 | 1.66x |
| dsv4-8000-512-none-s0 | bs=512 | GPUONLY_LAYERPIN_LAYERSTREAM | 61.5 | 71.0 | 1.15x |
| dsv4-8000-512-none-s2 | bs=512 | DYNAMIC_FFNCPU_ATTNSTREAM | 7.3 | 18.3 | 2.51x |
| dsv4-8000-512-none-s3 | bs=512 | STATIC_ATTNPRIO_ALLMODELS | 62.2 | 104.3 | 1.68x |
| dsv4-8000-512-none-s4 | bs=512 | DYNAMIC_FFN_ALTERNATE | 12.4 | 27.5 | 2.22x |
| dsv4-full-512-none-auto | bs=512 | GPUONLY_ATTNPIN_FFNSTREAM | 101.1 | 86.7 | 0.86x |
| dsv4-full-512-none-s0 | bs=512 | GPUONLY_LAYERPIN_LAYERSTREAM | 63.8 | 73.6 | 1.15x |
| dsv4-full-512-none-s1 | bs=512 | GPUONLY_ATTNPIN_FFNSTREAM | 101.1 | 88.1 | 0.87x |
| dsv4-full-512-none-s2 | bs=512 | DYNAMIC_FFNCPU_ATTNSTREAM | 7.8 | 19.7 | 2.53x |
| dsv4-full-512-none-s3 | bs=512 | STATIC_ATTNPRIO_ALLMODELS | 64.3 | 105.1 | 1.63x |
| dsv4-full-512-none-s4 | bs=512 | DYNAMIC_FFN_ALTERNATE | 13.2 | 29.8 | 2.26x |
| dsv4-full-512-none-pool_fetch | bs=512 | EXPERT_POOL | 100.3 | 86.2 | 0.86x |
| dsv4-full-512-none-pool_fetch-pred | bs=512 | EXPERT_POOL | 100.3 | 86.2 | 0.86x |
| dsv4-full-512-none-pool_fetch-pred-warm | bs=512 | EXPERT_POOL | 100.3 | 86.3 | 0.86x |
| dsv4-full-512-none-pool_hybrid | bs=512 | EXPERT_POOL | 100.3 | 87.1 | 0.87x |
| dsv4-full-512-none-pool_cpu_admit | bs=512 | EXPERT_POOL | 100.3 | 87.1 | 0.87x |
| dsv4-full-512-none-pool_cpu_exec | bs=512 | EXPERT_POOL | 100.3 | 86.0 | 0.86x |
| dsv4-full-512-none-pool_fetch_on_2nd_miss | bs=512 | EXPERT_POOL | 100.3 | 86.1 | 0.86x |
| q35-4000-4k-none-auto | bs=4096 | GPUONLY_LAYERPIN_LAYERSTREAM | 1764.7 | 4173.3 | 2.36x |
| q35-4000-4k-none-pool_fetch | bs=4096 | STATIC_ATTNPRIO_ALLMODELS | 1563.2 | 2924.2 | 1.87x |
| q35-8000-4k-none-auto | bs=4096 | GPUONLY_LAYERPIN_LAYERSTREAM | 1781.6 | 4260.7 | 2.39x |
| q35-8000-4k-none-pool_fetch | bs=4096 | EXPERT_POOL | 1708.3 | 4146.5 | 2.43x |
| q35-full-4k-none-auto | bs=4096 | GPUONLY_LAYERPIN_LAYERSTREAM | 1810.6 | 4285.4 | 2.37x |
| q35-full-4k-none-pool_fetch | bs=4096 | EXPERT_POOL | 1708.3 | 4142.8 | 2.43x |
| dsv4-8000-4k-none-auto | bs=4096 | STATIC_ATTNPRIO_ALLMODELS | 54.0 | 427.9 | 7.92x |
| dsv4-full-4k-none-auto | bs=4096 | GPUONLY_ATTNPIN_FFNSTREAM | 57.0 | 694.5 | 12.18x |

