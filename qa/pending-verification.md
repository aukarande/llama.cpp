# Pending verification queue

Changes land with build-only checks; GPU verification is BATCHED here and drained
in one session (machine exclusive, zombie-check first: `tasklist /fi "IMAGENAME eq
llama-perplexity.exe"` etc. - TaskStop'd harness shells survive as detached sh.exe
and corrupt concurrent measurements).

## Queued

1. **Registry loader determinism (commit pending)** - accumulate variants at two
   budgets (plan oss @4000 then @2000, same ctx), then with `-v`:
   - `-mva 4000` -> expect log `loaded ... from exact budget=4000 MiB variant`
   - `-mva 3000` -> expect `smaller budget=2000` + the leaves-VRAM-idle WARN
   - `-mva 1500` -> expect load refusal -> loud pshard DISABLED stock fallback
   Also: rerun any one 2k smoke cell to confirm no plan drift from the loader change.
2. **Full 144-grid rerun** - resume `QA_RESUME=1 sh qa/run-qa.sh /tmp/qa-full-v2 full`
   (clean through row 42; rows beyond were contamination-truncated 2026-08-30).
   Then compare-qa vs reference, refresh ledger (schema v2), commit, push.
3. **Nemotron full slice** - rerun from scratch (`/tmp/qa-nem-full` wiped, was
   contaminated): `QA_MODELS_LIST="nem:nvidia_Nemotron-3-Nano-30B-A3B-Q4_0" ...`
   Seeds the nem reference rows.
4. **DeepSeek-V4 pshard perf** - the correctness smoke stands (runs, output matches
   stock), but its 7.40 t/s decode was measured under zombie contention - remeasure
   one cell before quoting perf.
5. **Switch-cost terms (pairwise)** - plan q35 at 16k (auto): variant header must
   carry `switch_mb=/attn_frac=/head_mb=/pcie=` and tier lines `switch_ms=` (legacy
   anchor; 0 for tiers sharing the decode plan's residency). Then check
   `pshard_prefill_ubatch_eff`: a short prompt (e.g. 600 tokens) should prefer a
   residency-coherent tier over a marginally faster incoherent one; long prompts
   (16k) unchanged (switch cost amortizes). Sanity: switch_cost_ms magnitudes
   ~(pin-delta x ~450MB)/45GBps. Perf-check one short-prompt cell vs pre-change.

6. **ids-cross for ALTERNATE (commit pending)** - plan q35 at 16k, forced s4: decode
   tiers (bs=1/16) must show `ids_cross=1` in the registry (prefill tiers 0); the
   emitted ot= lines for streamed layers must carry `(ffn_gate_inp|ffn_exp_probs_b)`
   -> compute-tag patterns. Then measure s4 decode at a TIGHT budget where attn is
   only partially pinned (q35-16k@2000 measured 7.7 t/s with routers in-split):
   expect a large jump (sliced uploads engage; the @4000 all-attn-pinned cell already
   crossed incidentally at 36.1). Value gate: tokens/PPL unchanged vs stock.
7. **ALTERNATE adjudication (user question)** - after the full grid rerun under the
   fixed predictor + ids-cross: does s4 ever win a cell against re-planned s1/s3?
   If auto never picks it and forced-s4 never beats the best alternative, ALTERNATE
   earns retirement from the decode tiers (stays as a prefill design point).

## Contamination note (2026-08-30)

Two "stopped" harness runs kept running detached and overlapped each other and
ad-hoc tests. Perf numbers from that window are invalid; correctness outcomes
(DSv4 runs + output text, loader behavior classes) and planner PREDICTION numbers
(pure compute, no GPU contention sensitivity) remain valid.
