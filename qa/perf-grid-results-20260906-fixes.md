# Perf grid rerun 2026-09-06/07: the cells affected by commit 93b391904 (the three fixes)

Companion to [perf-grid-results-20260905.md](perf-grid-results-20260905.md). Same regime: GPC 2100
locked (headers read 2092-2100, inside the lock window), memory 14001, seed 1234, 128 decode
tokens, prompts prompt-512-v2 / prompt-4k-v2, PPL corpus ppl-docs-v2. Artifacts:
`grid-results/20260906-fixes/` (ledger.csv, per-cell logs, `compare-vs-20260905.md` from
`qa/perf-grid-compare.py`, results.md), `grid-results/20260906-fixes-merged/` (the rerun rows plus
the 85 kept rows, rendered by `qa/perf-grid-tables.py`), quiet repeats in
`grid-results/20260907-repeat-*`. Every number below comes from those ledgers and logs; the claims
were checked by a 13-agent adversarial analysis (4 lenses, 8 refuters, a completeness critic).

## 1. What was rerun, and on which build

Commit 93b391904 (design 11.C.19 viii-xi) fixed the scheduler re-reserve spill (DSv4 hc_head tail
pinned with the head, arena overflow refused), re-measured the MTP reserve after the fit, and
bounded pool tiers at plan time; its follow-ups clamp the planner probe's outputs, count only pool
tiers in the pool-creation gate and measure the MTP need over tier placements. Affected set = every
cell whose code path the commit touches: all speculative cells (DSpark, MTP), all pool cells, and
the DSv4 pshard legacy arms (auto, s0-s4) - 306 of 391 cells, selected by
`QA_ONLY='dspark|mtp|pool|^dsv4-.*-none--?(auto|s[0-4])'`. The 85 kept cells (q35 plain
stock/auto/s0-s4 perf, gate and PPL; DSv4 plain stock) were not rerun: a 9-agent adversarial check
(3 lenses, 6 refuters) traced every hunk of 93b391904 and of the drift before it and found each
one inert for those cells (the reserve clamp is the identity when n_outputs_max = n_batch, the
whole-window range moves no tensor without an overflow, the refusal and the memory-update branch
never execute, the planner changes never reach a plain legacy plan).

Four builds feed the merged tables; the ledger's `git=` header records HEAD at launch, not the
binary:

| rows | build | when |
|---|---|---|
| 85 kept | a1a7a4881 (09-05 grid) | 2026-09-05 20:43 - 09-06 04:55 |
| 270 main pass | 93b391904 | 2026-09-06 22:54 - 09-07 06:31 (header 22:54) |
| 8 (the q35mtp 8000/full 4k pool cells, 12:26 header) | 93b391904 + the working tree that became 9cc5cfe7c, including a head-home preset later reverted (no effect on these cells: no lever at 8000/full) | 2026-09-07 12:26-13:02 |
| 18 re-measured (section 5) | 9cc5cfe7c (header carries the hash) | 2026-09-07 19:59-20:48, RDP session connected by the user's choice |

The main-pass binary was relinked at 06:31 for the MTP fix, so it no longer exists on disk; its
source is 93b391904.

## 2. Statuses (final ledger)

| old -> new | cells |
|---|---|
| OK -> OK | 289 |
| STRATEGY_FALLBACK -> STRATEGY_FALLBACK | 13 (the same 13 cells: s1 at 8000 on DSv4, DSpark pool fetch, MTP pool fetch at 4000) |
| OVER_BUDGET -> OK | 2 (dsv4-8000-512-dspark auto and s3: the +1088 MiB spill cells) |
| OVER_RESERVE -> OK | 1 (q35mtp-4000-4k s1: the MTP reserve short by 178 MiB) |
| OK -> NOPOOL | 1 (q35mtp-4000-4k poolauto: the ladder's verify tier is legacy after the honest refit, section 4) |

No FAIL, PLAN_FAILED, SCHED_GREW or DEGENERATE. Method caveat: the runner's status chain assigns
FALLBACK / STRATEGY_FALLBACK before OVER_BUDGET / SCHED_GREW, so fix A actually closed FOUR
CUDA0-growth cells (the two above plus dsv4-8000-512-dspark-s1, +1088 MiB, and
dsv4-8000-4k-dspark-s1, +192 MiB, both recorded STRATEGY_FALLBACK in the old grid), and the old
degenerate WARM row was "OK" in the ledger (the tables script flagged it).

## 3. The fixes in the grid

- **A, scheduler re-reserve spill.** dsv4-8000-512-dspark auto and s3: VRAM peak delta 9442 ->
  8145 and 9369 -> 8083 MiB at an 8000 budget; the old logs' `CUDA0 compute buffer size of 7289
  MiB, does not match expectation of 6201` line is gone; no `arena spill`, `allocation refused` or
  `beyond the buffer range` line in any of the 117 DSv4 pshard logs; speed unchanged (target
  steps -0.1% / +2.4%). The one `beyond ... scratch window` line left is the warmup-time
  unviable-tier landing at dsv4-8000-4k-dspark-s1 (bs=4096 substitute 384 MiB short), byte-identical
  to the old run and STRATEGY_FALLBACK in both.
- **B, MTP reserve.** q35mtp-4000-4k-mtp-s1: `needs 213 MiB ... 35 were reserved: re-fitting`,
  budget 3965 -> 3787, exit check 212.5 used vs 213 reserved, steps/s 12.9 -> 12.9. All 66 MTP
  one-budget checks in the rerun read -0.5 or -1.0 MiB (no overshoot at the measured cells).
  Seven cells took two passes: q35mtp-4000-4k auto/poolauto/s0/s1, q35mtp-4000-512 s1,
  q35mtp-8000-4k s4, q35mtp-8000-512 s0. Cost at one cell: q35mtp-4000-512 s1 was never over its
  reserve (20.5 vs 21 MiB) but the max-over-tier-placements probe found 201 MiB under a tier the
  512-token workload never runs, took 180 MiB from the target, and the refitted plan then really
  uses 200.5 MiB: prompt 721 -> 672 t/s (-6.8%), steps 13.1 -> 12.7. Honest for arbitrary
  workloads, pessimistic for this one - your call whether the reserve should follow the tiers a
  workload can reach. Both s1 cells were measured under 93b391904's probe order (load-time array
  first); the committed code probes the registry tiers first, which agree for legacy plans.
- **C, plan-time pool bound.** The planner now rejects pool tiers the runtime carve could not host,
  in 33 plans across three blocks (q35-4000-512 pool x18: bs=1024; DSv4 full/4k: bs=4096/8192;
  DSv4+DSpark full/512: bs=2048) - no `tier unviable` at warmup anywhere. Executing tiers were
  unchanged except two: dsv4-full-4k-dspark-poolauto's 4k prompt moved from the bs=512 pool tier
  (8 ubatches, 122 t/s) to the legacy prefill tier (611 t/s = auto's 615), and the q35mtp-8000-4k
  pool cells' bs=4096 pool tier, unviable at warmup before, now prefills the 4k prompt (3501 ->
  3806-3844 t/s; the sampled text changed with it, h 0.768 -> 0.735 for fetch).
- **Output clamp (xi-3) re-priced every MTP plan's prefill tiers**: 37 MTP plan logs differ in the
  union-enforcer sequence; prompt speed rose with identical text (q35mtp-4000-512 s4 +25%,
  q35mtp-full-4k s3 +7.5%, dsv4-full-4k-dspark-auto +6%), decode unchanged. A code effect, an
  improvement.
- **Generator fix (xii)**: the 16 MTP pool cells at 8000/full run with the pool again (h
  0.73-0.92); section 4.

## 4. Defect found by the rerun: two override generators

Every q35 MTP pool cell at 8000 and full first came back FALLBACK: pshard disabled itself. The
runtime's load-time override generator (src/llama-pshard-cache.cpp) and the planner's
(src/llama-pshard-plan.cpp) are two copies, and the cache copy's EXPERT_POOL branch lacked the
planner's special case that pins the MTP layer whole. The post-fit MTP probe over the load-time
array measured 193 MiB (the MTP layer's experts on the host for a stock-sched context) where the
plan tool measured 21, the runtime refit to a budget the plan tool had never saved, and `no
matching plan cache ... disabling pshard` followed. Two failure modes: at 512 the stock fallback ran
and the runner recorded stock numbers under the pool label (rc=0, ~330 t/s prompt, VRAM peak 14042
MiB at an 8000 budget - the fallback ignores -mva); at 4k the fallback died (rc=1, no numbers; those
logs were overwritten by the rerun, the rows survive in ledger.before-mtp-rerun.csv). Fixed in
9cc5cfe7c: the cache copy pins the MTP layer like the planner, and the probe measures the registry
tiers' own override lists (what every tier switch applies, identical in both processes). Open: fold
the two generator copies into one.

**The head home.** Where pass 1 took the head lever and pass 2 (a fresh plan at the lower budget)
does not, the raised reserve is idle (178 MiB at q35mtp-4000-4k) and the ladder may change shape:
poolauto lost its pool verify tier there (NOPOOL, 21.6 target steps/s vs 24.4 in the old grid's
lever plan). A protocol that kept pass 1's head home for pass 2 was tried and reverted: same-arm
A/B on the five lever cells (head-pinned pass 2 vs head-on-CPU preset) rescued poolauto (21.6 ->
23.6) but cost auto 21.9 -> 19.3 and s4 23.6 -> 22.1, every lever cell's prefill 15-24%, and ~400
MiB of peak VRAM (4400-4439 at a 4000 budget vs 3981-4003). The final rows (9cc5cfe7c, head-pinned
pass 2): auto 21.5 steps/s at a 3876 t/s prompt, s0 2.1, poolauto 21.6 NOPOOL at 3789 t/s, s4 @8000
23.4. End to end at the poolauto cell the pool plan (pass-1 lever shape) is still ~10% faster than
the legacy verify tier: the planner never prices the head home (the lever is a last resort when the
union overshoots), and pricing it is a planner-model change for you to decide.

## 5. "Pool new slower than pool old": machine state, not code

The main pass's q35 pool cells decode 1.9% slower on average than the old grid with byte-identical
plans, slot counts, hit rates, misses per token and generated text (the compare's "plan
EXPERT_POOL/81 -> /80" entries are the planner's slot estimate, lowered by fix C's 64 MiB margin;
every runtime carve is identical). An adversarial investigation (3 lenses, 6 refuters, all voting
machine state, high confidence) located it in time, not code:

- nothing in a1a7a4881..93b391904 executes per step, per miss or per prefill ubatch of a plain pool
  run; the runtime state is byte-identical across the two runs;
- the q35 512-prompt pool blocks ran 22:54-23:22 while a 9-agent code-review workflow was active
  on this machine and the user's RDP session was connected: A/B prefill -3.6..-4.0% on 36 of 36
  cells, decode -2.6..-3.7%, the CPU-bound cpu_exec probe -5..-6% at the same minutes, unchanged
  code (mmap preload, context construction) +11..26%, compute-bound PPL passes identical;
- the load-time signature recurs at 00:12-00:25 (DSv4 full/512 pool block: upload policies -3..-5%,
  CPU-route policies flat), 01:09/01:27 (MTP stock prompts -6..-9% with unchanged code) and
  02:26-02:37 (q35 4k -pred cells and a -24% warm transient with identical counters); the full/4k
  block at 02:43-02:49 reproduced the old numbers to +0.7%. RDP state during the main pass was not
  logged; idle VRAM was 953-962 MiB at both launches vs 581-781 in the old grid.

Repeats of the six q35 pool_fetch perf cells on the same build:

| cell | old grid | main pass (agents active) | RDP session active | quiet, disconnected |
|---|---|---|---|---|
| q35 4000/512 | 59.9 | 57.4 | 52.7 | 58.2 |
| q35 8000/512 | 76.6 | 75.3 | 73.4 | 76.6 |
| q35 full/512 | 84.0 | 81.5 | 80.3 | 83.6 |
| q35 4000/4k | 54.6 | 53.8 | 53.3 | 53.9 |
| q35 8000/4k | 76.0 | 72.5 | 66.9 | 70.6, then 69.6 and 75.9 |
| q35 full/4k | 83.5 | 83.5 | - | 84.1 |

The quiet machine (idle VRAM 679 MiB) brings five of six back within 0-3% of the old grid; the
8000/4k cell has a 9% run-to-run spread of its own (69.6 / 75.9 on consecutive quiet runs,
identical h), so its old 76.0 was the high end. These repeat ledgers are kept separately
(`20260907-repeat-q35pool-b`, `20260907-repeat-8000-4k-{1,2}`); the canonical rows are the ones in
`20260906-fixes/ledger.csv`. The pool's home is the mmap'd model, so miss uploads are staged
through host RAM and are the first path to suffer from host activity. Rules added to
qa/perf-grid.md.

The 18 rows re-recorded on 9cc5cfe7c (the last ledger header; RDP connected but the user idle):

| cell | old grid | before re-measure | final | reading |
|---|---|---|---|---|
| dsv4-full-512 pool_fetch | 17.7 | 16.8 (-5.3%) | 17.6 | machine state |
| q35-4000-4k pool_fetch-pred | 53.8 | 50.3 (-6.5%) | 55.5 | machine state |
| q35-4000-4k pool_hybrid-pred | 51.5 | 48.0 (-6.8%) | 49.5 | in band |
| q35-8000-4k pool_fetch-pred-warm | 73.3 | 55.8 (-24%, identical work) | 66.2 | transient gone; this cell family spreads 9% |
| q35-8000-4k pool_cpu_admit-pred | 56.7 | 53.2 (-6.2%) | 54.2 | CPU-route, session connected |
| q35mtp-8000-512 pool fetch / fetch+pred [steps/s] | 44.0 / 44.8 | 41.5 / 42.1 | 43.1 / 43.7 | in band |
| q35mtp-8000-512 pool hybrid / plan | 39.3 / 38.8 | 36.7 / 38.0 | 36.0 / 38.1 | hybrid -8%, inside the speculative band |
| q35mtp-full-512 pool fetch / fetch+pred | 51.7 / 52.0 | 49.1 / 49.9 | 50.6 / 51.2 | in band |
| q35mtp-full-512 pool hybrid / plan | 41.1 / 42.4 | 40.3 / 39.8 | 40.9 / 41.0 | in band |
| q35mtp-8000-512 s0 | 2.5 | 2.7 (preset) | 2.7 | in band |
| lever cells (auto, s0, poolauto @4000/4k; s4 @8000/4k) | 20.2, 2.2, 24.4, 22.3 | preset build | 21.5, 2.1, 21.6 NOPOOL, 23.4 | section 4 |

## 6. Headline numbers (merged tables, decode t/s; pool rows carry h)

q35 plain, prompt t/s / decode t/s (pool rows: main pass; the quiet repeats are in section 5):

| arm | 4000/512 | 8000/512 | full/512 | 4000/4k | 8000/4k | full/4k |
|---|---|---|---|---|---|---|
| stock (kept) | 185 / 47.6 | 234 / 56.5 | 407 / 83.7 | 409 / 47.3 | 521 / 56.5 | 906 / 82.3 |
| s3 (kept) | 779 / 47.5 | 899 / 56.0 | 1202 / 85.7 | 2942 / 46.0 | 3004 / 52.9 | 3096 / 81.2 |
| pool fetch | 656 / 57.4 | 686 / 75.3 | 678 / 81.5 | 2878 / 53.8 | 4151 / 72.5 | 4157 / 83.5 |
| poolauto | 621 / 53.9 (hybrid) | 587 / 64.1 (hybrid) | 603 / 81.1 (fetch) | 4113 / 49.6 (hybrid) | 3801 / 63.1 (hybrid) | 3322 / 82.0 (fetch) |

The pool keeps its wins over stock and the best legacy arm at 4000 and 8000 (+21% / +33% at 512,
+14% / +28% at 4k against stock) and ties at full; with the quiet repeats the 512 columns read
58.2 / 76.6 / 83.6.

DSv4 plain: pool fetch 17.6 (full/512, h=0.609) vs stock 11.2 and legacy 12.2-12.4; 15.4 (full/4k,
h=0.579) vs 12.7 and 12.0.

q35 MTP, target steps/s = decode / (1 + accept x 2), final ledger:

| arm | 4000/512 | 8000/512 | full/512 | 4000/4k | 8000/4k | full/4k |
|---|---|---|---|---|---|---|
| stock | 22.2 | 26.0 | 37.0 | 22.1 | 26.7 | 36.9 |
| auto | 22.2 | 25.4 | 39.0 | 21.5 | 26.3 | 38.2 |
| s3 | 22.8 | 26.9 | 39.7 | 22.5 | 26.9 | 37.2 |
| pool fetch | fallback | 43.1 (h=0.774) | 50.6 (h=0.866) | fallback | 40.3 (h=0.735) | 48.8 (h=0.839) |
| pool hybrid | 28.5 (h=0.433) | 36.0 (h=0.728) | 40.9 (h=0.795) | 27.3 (h=0.412) | 39.0 (h=0.775) | 42.0 (h=0.815) |
| poolauto | 27.2 (h=0.432) | 38.7 (h=0.752) | 40.7 (h=0.795) | 21.6 NOPOOL | 34.4 (h=0.708) | 37.7 (h=0.759) |

DSv4 + DSpark, target steps/s: stock 1.8 everywhere; auto/s3 1.6-3.6; pool hybrid 6.0 (full/512,
h=0.446) and 5.8 (full/4k, h=0.440), counters identical to the old grid; poolauto 3.5 (legacy
verify tier). The DSpark prompt story: the old narrative's "the pool's slow 4k prefill (122 vs 580
t/s)" was the target prefilling 3910 tokens as 8 x bs=512 pool ubatches, each streaming the whole
expert stack through the A/B pair; in the old grid poolauto also did this because its bs=1024/2048/
4096 pool tiers failed their runtime reserve, and the plan-time bound now leaves poolauto a legacy
prefill tier (611 t/s, end to end 25.9 s vs auto 25.8). The forced pool arms still prefill at 122
t/s: find_optimal_ubatch prices the legacy prefill tiers at 57-98 t/s against 432-695 measured and
picks the bs=512 pool tier. The verify tier (bs=4) stays legacy for poolauto because the legacy
bs=4 tier is priced ~2x too fast (32.4 batch-tokens/s planned vs ~16 measured); forced hybrid at
bs=4 is 1.7x faster than legacy. Both are prefill/verify-tier pricing items, proposed, not applied.
Also: llama-speculative-simple's `prompt eval` line is the verify batches only (the scheduler
rebuild's synchronize resets the compute timer before the prefill graph); the runner's prompt
figure is the tool's own `encoded` wall line, which is right.

## 7. Correctness

PPL: all 22 rerun PPL cells identical to the old grid to four decimals (q35 pool 4.0829 / 4.0822 /
4.0822, DSv4 auto 4.1995 / 4.1910, DSv4 pool 4.1891). Gates: 43 of 46 rerun gate hashes
unchanged (all 24 q35 gates, all 8 DSv4 pool gates, all 6 4k gates). Three DSv4 forced-legacy
512-prompt gates moved, with identical plans, n_pinned and tiers:

| gate | old | new | reading |
|---|---|---|---|
| dsv4-8000-512-none-s4 | a4406c7d1a91 | 15781302de8b | now byte-identical to stock's text at 8000; divergence at ~token 22 ("add a new feature" vs "contribute something") |
| dsv4-full-512-none-s0 | a4406c7d1a91 | 15781302de8b | leaves stock's text at full for the s1/s4/cpu_admit group; same ~token-22 flip; s0 now identical at both budgets |
| dsv4-full-512-none-s2 | a4406c7d1a91 | a17eece691a6 | NEW hash, coherent ("a new one." -> "a duplicate." after 18 words); no other arm produces it; no PPL mirror exists for DYNAMIC_FFNCPU_ATTNSTREAM |

Leading explanation: fix A names DSv4's hc_head tail nodes so they follow the head instead of
being placed by the op-offload row rule, which changes the executing backend under the strategies
that stream layers (s0, s2, s4) - no arithmetic changed, and the changes are confined to those
arms (9 of 24 s0/s2/s4 perf texts changed, 3 of 6 gates; every STATIC_ATTNPRIO, ATTNPIN and pool
text is byte-identical), but no log line records the tail's backend, so this is inferred, not
observed. The 512-vs-4k asymmetry (8 of 12 512-prompt s0/s2/s4 texts changed, 1 of 12 4k) and
three first-word flips under sampling mean same-build determinism of these arms is untested. The
merged census: 76 valid gates produce 11 distinct hashes, 8 of them produced by stock at some
budget; 3 pshard-only (a17eece691a6, fb9eed1eaacf, 88957c0ab95f). Certification of the s2 hash
(two same-build repeats plus a PPL mirror for forced s2 at full) was scheduled and CUT SHORT: the
machine lost power at 20:51:59 on 09-07 (Kernel-Power event 41, "rebooted without cleanly shutting
down") at the exact start of the s2 gate's generation - the second unexpected shutdown in three
days during a DSv4 pshard cell (the first, on 09-05, prompted the clock lowering to 2100). The s2
hash stays uncertified until the power question is settled.

WARM: q35-4000-4k pool_fetch-pred-warm now generates the base text (md5 identical to the
fetch-pred cell, h 0.993 -> 0.617), the other warm cells keep their old counters exactly; the
degenerate scan flags 0 of 306. Same-build repeats of the 12 MTP cells reproduce acceptance, h and
token counts exactly: the text is deterministic, the timing is the noisy part.

## 8. Decisions for the user (unchanged from the 09-05 narrative unless noted)

- Pricing changes remain proposed, not applied: hybrid serial-sum term, DRAM-rate t_cpu, and now
  two DSpark items - the legacy prefill tiers priced 5-8x too slow (57-98 vs 432-695 t/s, which
  keeps the forced pool arms on 8 x bs=512 pool ubatches) and the legacy bs=4 verify tier priced
  ~2x too fast (which keeps poolauto off the pool's 1.7x-faster verify tier).
- The MTP head's home is never priced (lever = last resort); the preset experiment shows +9% on
  one arm and -6..-12% on the others - a planner decision. Until then q35mtp-4000-4k poolauto is a
  NOPOOL row.
- The MTP reserve's max over all viable tier placements is honest for arbitrary workloads and
  costs q35mtp-4000-512 s1 6.8% of prompt speed for a tier it never runs.
- Planner residuals: the pinned-expert probe under-estimates the pool graph's scratch by 8.9% on
  DSv4 (not the logits), and its slot estimate now sits one below the runtime carve everywhere (the
  64 MiB margin); a modelled per-token term is proposed, not applied.
- Knob removals (WARM/ALLOC, fetch_on_2nd_miss, GGML_SCHED_NO_CPU_OVERLAP, ZIPF/CPU_GBS/GFLOPS env)
  as recommended on 09-05; the WARM variants are now honest and still never win.
- Two override-generator copies (planner and load-time) should become one function.
- The DSv4 s2 gate hash: certify (two repeats + a PPL mirror, ~5 min of GPU) once the machine is
  trusted to stay up under a DSv4 s2 cell (all routed FFN on the CPU plus streamed attention).
- Power: two unexpected shutdowns during DSv4 pshard cells (09-05 14:06, 09-07 20:52), the second
  at a locked 2100 MHz GPC - the PSU or the CPU side under the all-FFN-on-CPU strategy is the
  suspect; unattended runs of s2-class cells should wait for a decision.
- The lever cells' peak VRAM sits 120-215 MiB above the old grid (MTP context compute outside the
  arena), and the q35mtp 4k pool cells' peak rose ~240 MiB (8193 -> 8437 at an 8000 budget, within
  the +1024 OVER_BUDGET rule) - worth a look under the whole-window alloc range.
