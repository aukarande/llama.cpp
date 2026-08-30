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
5. **Switch-cost terms (commit pending)** - plan q35 at 16k (auto): registry tier
   lines must carry sensible `switch_ms=` values (0 for tiers sharing the decode
   plan's residency; ~100-500ms-scale for multi-layer pin deltas at ~450 MB/layer).
   Then check `pshard_prefill_ubatch_eff`: a short prompt (e.g. 600 tokens) should
   now prefer a residency-coherent tier over a marginally faster incoherent one;
   long prompts (16k) should be unchanged (switch cost amortizes). Perf-check one
   short-prompt cell vs pre-change numbers.

## Contamination note (2026-08-30)

Two "stopped" harness runs kept running detached and overlapped each other and
ad-hoc tests. Perf numbers from that window are invalid; correctness outcomes
(DSv4 runs + output text, loader behavior classes) and planner PREDICTION numbers
(pure compute, no GPU contention sensitivity) remain valid.
