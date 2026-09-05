# pshard perf grid: stock vs legacy vs expert pool, q35 and DSv4

Runner: `sh qa/perf-grid.sh <out-dir> [--list]` (2026-09-04). This grid is separate from
the held 144-cell QA grid (`qa/run-qa.sh`): it is a PERF grid for the two models the pool
work targets, plus the correctness gates and perplexity mirrors those perf cells need.

## Rules every cell obeys

- Decode length is **128 tokens** on every arm (user rule 2026-09-04). 32-token runs are
  correctness gates only (md5 + pool counters), never perf.
- A perf run carries the workload and the budget and nothing else:
  `-m -f <prompt> -n 128 -c <ctx> --ignore-eos -no-cnv` plus `-fitb N` (stock) or
  `-pshard -mva N` (pshard). No `--temp 0`, no `-lv`, no `-ub/-b`, no `-ngl`, no `-ot`.
  The pool's three headline counter lines (all-pass h, decode h, prefetch) print at WARN,
  so every pool perf row still carries `h` and `misses_per_token` - the pricing model's
  calibration data.
- Thread count is always the default (user rule: never set `-t`).
- Every pshard cell plans fresh with exactly the environment the run will use
  (`PSHARD_STRATEGY`, `PSHARD_MISS_POLICY`, `GGML_SCHED_NO_CPU_OVERLAP`, the spec flags and
  the budget are all fingerprinted; a mismatch silently falls back to stock - the runner
  greps the run log for that fallback and marks the row FALLBACK). Consecutive cells that
  differ only in runtime-only knobs (`PSHARD_POOL_PREDICT`, `PSHARD_POOL_WARM/ALLOC`) share
  one plan.
- Pool arms run with `PSHARD_POOL_RUNTIME=1`; the auto ladder with the pool allowed adds
  `PSHARD_POOL_AUTO=1` to both plan and run.
- "full" budget = `QA_FULL_MVA`, default 14500 MiB: the 16 GB card's idle free VRAM
  (14.6-15.0 GB) minus a margin, FIXED so the plan and the run see the same budget (the
  registry variant is keyed on the exact number) and grids stay comparable day to day.
- The machine is otherwise idle; one instance at a time (registries are per-model global
  state). The runner backs up each model's registry at start and restores it at exit.

## The user's table (2026-09-04)

| block | arms | budgets | prompts |
|---|---|---|---|
| q35 plain | stock, legacy, pool | 4000, 8000, full | 512, 4k |
| dsv4 plain | legacy at 8000 and full; pool at full only | 8000, full | 512, 4k |
| q35mtp (`--spec-type draft-mtp --spec-draft-n-max 2`) | stock, legacy, pool | 4000, 8000, full | 512, 4k |
| dsv4 + DSpark (`-md` draft, `--spec-type draft-dspark`) | stock, legacy, pool | 8000, full | 512, 4k |

"legacy" = the planner's auto pick, PLUS the five forced strategies s0..s4 as their own arms (added 2026-09-04 mid-run: +100 perf, +25 gates at 512). DSv4's pool pins ~9.2 GB
(attention, routers, norms, output head) before its first slot, so it has no pool arm at
8000. DSv4 + DSpark stock only fits at `-fitb 3000` (the stock fit ignores the 10.4 GB
draft) and is recorded at that budget. Prompts: `512` = prompt-512.txt at ctx 2048, `4k` =
prompt-4k.txt at ctx 8192.

## The pool arm = 13 variants (plain decode), 5 (speculative)

| # | variant | env | decides |
|---|---|---|---|
| 1 | fetch | `PSHARD_MISS_POLICY=fetch` | the baseline policy |
| 2 | fetch + pred | + `PSHARD_POOL_PREDICT=1` | predictor default |
| 3 | fetch + pred + warm | + `PSHARD_POOL_WARM=8 PSHARD_POOL_ALLOC=1` | prompt-end LRU seeding + per-layer slots |
| 4 | hybrid | `PSHARD_MISS_POLICY=hybrid` | the FreeToken q* split |
| 5 | hybrid + pred | + `PSHARD_POOL_PREDICT=1` | whether prefetch hurts CPU-route policies everywhere |
| 6 | hybrid noovl | + `GGML_SCHED_NO_CPU_OVERLAP=1` | certify the CPU/GPU overlap, then delete the switch |
| 7 | cpu_admit | `PSHARD_MISS_POLICY=cpu_admit` | background admission |
| 8 | cpu_admit + pred | + `PSHARD_POOL_PREDICT=1` | |
| 9 | cpu_admit noovl | + `GGML_SCHED_NO_CPU_OVERLAP=1` | |
| 10 | cpu_exec | `PSHARD_MISS_POLICY=cpu_exec` | the never-admit floor |
| 11 | fetch_on_2nd_miss | `PSHARD_MISS_POLICY=fetch_on_2nd_miss` | certify it is dominated, then delete the policy |
| 12 | plan | `PSHARD_STRATEGY=5`, planner picks the policy | does the pricing pick the measured winner |
| 13 | poolauto | `PSHARD_POOL_AUTO=1` | does the ladder reach the forced pool |

Speculative pool block: fetch, fetch + pred, hybrid, plan, poolauto (the CPU-route
policies already lost on verify batches).

Not varied by design: threads (default), the prefetch cap (1; 2 lost on DSv4), the
staging-ring knobs (every DSv4 cell exercises the ring at its defaults, which is their
confirmation), the pricing constants (calibrated FROM the pool rows' `h`).

## Cells

Perf (128 tokens):

| block | per (budget, prompt) | sets | cells |
|---|---|---|---|
| q35 plain | stock + legacy + 13 pool = 15 | 3 x 2 | 90 |
| dsv4 plain | 8000: legacy = 1; full: legacy + 13 pool = 14 | 2 x 2 | 30 |
| q35mtp | stock + legacy + 5 pool = 7 | 3 x 2 | 42 |
| dsv4 + DSpark | 8000: stock + legacy = 2; full: + 5 pool = 7 | 2 x 2 | 18 |
| **perf total** | | | **180** |

Gates (32 tokens, `--temp 0 -lv 4`, md5 vs stock + pool counters): stock (the reference,
also where its perf cell is not in the table) and legacy at every (model, budget, prompt);
at the 512 prompt every pool policy (fetch, fetch + pred, fetch + pred + warm, hybrid,
cpu_admit, cpu_exec, fetch_on_2nd_miss), at 4k pool fetch only. **52 gates.**

Perplexity mirror (`llama-perplexity -c 2048 --chunks 8` on prompt-256k, prompt-independent,
once per model and budget): stock, legacy, pool fetch. **14 ppl** (dsv4 stock ~11 min each).

**Total 371 cells** (246 + the forced strategies). Estimate: perf ~5 h (q35 cells ~1.2 min, legacy-auto and poolauto plans
3-4 min each, dsv4 cells ~3.5 min), gates ~50 min, ppl ~45 min: **~6.5 h**, splittable as
`QA_GRID_MODELS=q35` then `QA_GRID_MODELS=dsv4`.

## Ledger

One CSV row per cell in `<out-dir>/ledger.csv`:
`cell,kind,model,mva,prompt,ctx,spec,arm,predict,warm,noovl,rc,prompt_tps,decode_tps,decode_tokens,accept_pct,vram_peak_delta_mib,strategy_active,n_pinned,miss_policy,pool_slots,md5,h,misses_per_token,ppl,status`.
`strategy_active/n_pinned/miss_policy/pool_slots` come from the registry's tier-0 line after
the plan; `md5` from gate cells; `h/misses_per_token` from every pool row (perf and gate);
`accept_pct` from spec cells. Status is OK, FAIL (rc != 0), PLAN_FAILED, FALLBACK (pshard
disabled itself - a fingerprint mismatch or an unviable plan), or SHORT (decode window
< 64 tokens).

`QA_RESUME=1` skips cells already in the ledger; `QA_ONLY=<regex>` filters cell names
(e.g. `QA_ONLY='q35-8000-512'`); `QA_GRID_MODELS="q35"` or `"dsv4"` filters models.

## What the grid answers

1. Stock vs legacy vs pool at every budget and prompt length, both models, plain and
   speculative.
2. Pool: which miss policy wins where, and does the planner's pick (`plan`) match it
   (the 2026-09-04 finding: hybrid priced over fetch at q35 @8000 while fetch measures
   20% faster - `t_serve` 0.10 ms/layer in the model vs ~0.03 measured).
3. Predictor: default on? (helps fetch, hurts CPU-route policies so far.)
4. Warm start + allocation with prompt-end LRU seeding: any budget where it wins?
5. The CPU/GPU overlap switch and fetch_on_2nd_miss: certify, then delete.
6. Ladder: does `poolauto` reach the forced pool's number, and what does the mixed ladder
   cost (colder start after a legacy prefill tier, the tier-0 switch)?
7. Pricing calibration: `h` from every pool row across s = 8..100 refits alpha and the CPU
   rates, after which PSHARD_POOL_ZIPF / CPU_GBS / CPU_GFLOPS become constants.
8. Correctness: every pool policy's 32-token md5 against stock; PPL parity for the headline
   arms.
