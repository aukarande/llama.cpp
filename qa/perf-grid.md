# pshard perf grid: stock vs legacy strategies vs expert pool, q35 and DSv4

Runner: `sh qa/perf-grid.sh <out-dir> [core|ext] [--list]` (2026-09-04). This grid is
separate from the held 144-cell QA grid (`qa/run-qa.sh`): it is a PERF grid for the two
models the pool work targets, plus the correctness gates those perf cells need.

## Rules every cell obeys

- Decode length is **128 tokens** on every arm (user rule 2026-09-04). 32-token runs are
  correctness gates only (md5 + pool counters), never perf.
- A perf run carries the workload and the budget and nothing else:
  `-m -f <prompt> -n 128 -c <ctx> --ignore-eos -no-cnv` plus `-fitb N` (stock) or
  `-pshard -mva N` (pshard). No `--temp 0`, no `-lv`, no `-ub/-b`, no `-ngl`, no `-ot`.
- Every pshard cell plans fresh with exactly the environment the run will use
  (`PSHARD_STRATEGY`, `PSHARD_MISS_POLICY`, `-t`, the spec flags and the budget are all
  fingerprinted; a mismatch silently falls back to stock - the runner greps the run log for
  that fallback and marks the row FALLBACK). Consecutive cells that differ only in a
  runtime-only knob (`PSHARD_POOL_PREDICT`) share one plan.
- Pool arms run with `PSHARD_POOL_RUNTIME=1`; the auto ladder with the pool allowed adds
  `PSHARD_POOL_AUTO=1` to both plan and run.
- "full" budget = `QA_FULL_MVA`, default 14500 MiB: the 16 GB card's idle free VRAM
  (14.6-15.0 GB) minus a margin, FIXED so the plan and the run see the same budget (the
  registry variant is keyed on the exact number) and grids stay comparable day to day.
- The machine is otherwise idle; one instance at a time (registries are per-model global
  state). The runner backs up each model's registry at start and restores it at exit.

## Dimensions

| dimension | values |
|---|---|
| model | `q35` Qwen3.6-35B-A3B-UD-Q4_K_M; `q35mtp` the -wmtp variant (MTP head); `dsv4` DeepSeek-V4-Flash-UD-Q2_K_XL (3 shards) |
| budget (`-mva` / `-fitb`, MiB) | 4000, 8000, full (14500) - user's choice 2026-09-04; ext adds q35 @2700 (the tight-budget showcase) |
| prompt | `512` (prompt-512.txt, ctx 2048); `4k` (prompt-4k.txt, ctx 8192) |
| speculation | none; `mtp` (`--spec-type draft-mtp --spec-draft-n-max 2`, q35mtp); `dspark` (`-md` DSpark draft; dsv4 core, q35 ext); `dflash` (`-md` DFlash draft, q35 ext). Spec cells use the 512 prompt: speculation is a decode question |
| arm | `stock`; `auto` (legacy ladder, planner's pick); `s0..s4` (forced legacy: GPUONLY_LAYERPIN_LAYERSTREAM, GPUONLY_ATTNPIN_FFNSTREAM, DYNAMIC_FFNCPU_ATTNSTREAM, STATIC_ATTNPRIO_ALLMODELS, DYNAMIC_FFN_ALTERNATE); `pool:<policy>` (forced EXPERT_POOL with `PSHARD_MISS_POLICY`); `pool:plan` (EXPERT_POOL, planner picks the policy); `poolauto` (auto ladder with the pool allowed) |
| miss policy (pool arms) | `fetch` (copy the missed expert host -> LRU victim slot, compute on GPU); `cpu_exec` (compute every miss on the CPU from host weights, never admit); `fetch_on_2nd_miss` (first miss on CPU without admitting, a repeat miss fetches); `hybrid` (FreeToken q*: fetch a recency-chosen share of the misses, CPU computes the rest, both chains overlap); `cpu_admit` (compute the miss on CPU this pass while its rows upload on the copy stream; GPU hit next pass) |
| predictor | off; `pred` = `PSHARD_POOL_PREDICT=1` (layer l's FFN input through layer l+1's router, prefetch cap 1) on the arms that admit: fetch, hybrid, cpu_admit, poolauto |
| threads (ext) | default; `t16` = `-t 16` on the CPU-route policies (cpu_exec, hybrid, cpu_admit) |
| sched overlap (ext) | on; `noovl` = `GGML_SCHED_NO_CPU_OVERLAP=1` on hybrid and cpu_admit (plan and run) |

DSv4's pool pins ~9.2 GB (attention, routers, norms, output head) before its first slot, so
at 4000 and 8000 only stock and the legacy arms exist for dsv4; every arm runs at full.

## Cells (core)

Plain decode, per (model, budget, prompt):

| arm | perf (128) | gate 512 (32, md5 + counters) | gate 4k |
|---|---|---|---|
| stock | x | x (reference md5) | x |
| auto | x | x | x |
| s0, s1, s2, s3, s4 | x each | | |
| pool:fetch, pool:hybrid, pool:cpu_admit, each with and without pred | x each (6) | fetch, hybrid, cpu_admit, fetch+pred | fetch |
| pool:cpu_exec, pool:fetch_on_2nd_miss | x each | x each | |
| pool:plan | x | | |
| poolauto, poolauto+pred | x each | | |

q35: 3 budgets x 2 prompts = 6 sets of 18 perf (108) + gates 3x8 + 3x3 (33).
dsv4: 4000 and 8000 x 2 prompts = 4 sets of 7 perf (28) + 2 gates each (8); full x 2
prompts = 36 perf + 8 + 3 gates (11).

Speculative decoding (128 tokens, prompt 512, `llama-speculative-simple`):

| model / draft | budgets | arms |
|---|---|---|
| q35mtp / mtp | 4000, 8000, full | stock, auto, pool:fetch, pool:fetch+pred, pool:plan, poolauto (18 cells) |
| dsv4 / dspark | full | stock (fits only at `-fitb 3000`, recorded as such), auto, pool:fetch, pool:fetch+pred, pool:hybrid, pool:plan, poolauto (7 cells) |

Perplexity mirror (c2048, 8 chunks, prompt-256k): q35 stock, auto, pool:fetch at each
budget (9); dsv4 full auto, pool:fetch (2; stock is ext - 11 min).

**Core total: 260 cells** (197 perf, 52 gate, 11 ppl); ext 330. Estimate: q35 ~4 h (108 perf at
~1.2 min, the legacy-auto and poolauto plans 3-4 min each - shared by the pred variant),
dsv4 ~6 h (64 perf at ~3.5 min, dsv4 auto plans ~5 min), spec ~1 h, gates ~1 h, ppl
~0.5 h: **~10-12 h**, best run as `QA_GRID_MODELS=q35` (~5 h) then `QA_GRID_MODELS=dsv4`.

Ext adds (~70 cells): q35 @2700 both prompts (36 perf + 11 gates); q35 @8000 with the
dflash and dspark drafts (6 arms each); `t16` on cpu_exec/hybrid/cpu_admit and `noovl` on
hybrid/cpu_admit at q35 @8000 and dsv4 full (10); dsv4 stock PPL.

## Ledger

One CSV row per cell in `<out-dir>/ledger.csv`:
`cell,kind,model,mva,prompt,ctx,spec,arm,predict,threads,noovl,rc,prompt_tps,decode_tps,decode_tokens,accept_pct,vram_peak_delta_mib,strategy_active,n_pinned,miss_policy,pool_slots,md5,h,misses_per_token,ppl,status`.
`strategy_active/n_pinned/miss_policy/pool_slots` come from the registry's tier-0 line after
the plan; `md5/h/misses_per_token` from gate cells; `accept_pct` from spec cells. Status is
OK, FAIL (rc != 0), PLAN_FAILED, FALLBACK (pshard disabled itself - a fingerprint mismatch
or an unviable plan), or SHORT (decode window < 64 tokens).

`QA_RESUME=1` skips cells already in the ledger; `QA_ONLY=<regex>` filters cell names
(e.g. `QA_ONLY='q35-8000-512'`); `QA_GRID_MODELS="q35"` or `"dsv4"` filters models.

## What the grid answers

1. Legacy: does the planner's auto pick match the best forced strategy at each budget and
   prompt length?
2. Pool: which miss policy wins at each budget, and does the planner's pick (`pool:plan`)
   match it (the 2026-09-04 finding: hybrid priced over fetch at q35 @8000 while fetch
   measures 20% faster)?
3. Predictor: where prefetch helps (fetch) and where it hurts (CPU-route policies).
4. Ladder: does `poolauto` reach the forced pool's number, and what does the mixed ladder
   cost (colder start after a legacy prefill tier, the tier-0 switch)?
5. Prompt length: how the prefill tiers (A/B streaming vs legacy layer streaming) compare
   at 4k, and whether the decode after a 4k prompt keeps the 512-prompt ranking.
6. Speculation: pool vs legacy vs stock with each draft type, at 128 tokens.
7. Correctness: every pool policy's 32-token md5 against stock; PPL parity for the
   headline arms.
