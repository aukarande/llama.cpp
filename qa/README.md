# pshard QA harness

Stock llama.cpp is the golden. Three gates, applied over the full config grid:

1. **Correctness**: pshard generation must be token-identical to stock at temp 0.
   Stock runs with `-ub` matched to pshard's planner-chosen `cache_ubatch` (evaluation
   shape changes numerics on some models — gpt-oss PPL moves 23% on ubatch alone), and
   `-fitt (free - budget)` so both sides get the same effective VRAM budget.
2. **Plan stability**: the active tier-0 strategy, overlap mode, and n_pinned must match
   the reference ledger. This catches silent fallbacks (forced strategy unviable -> stock
   or ATTNPRIO fallback) and planner drift — both produced misleading numbers before.
3. **Perf**: prompt/decode t/s within 5% of the reference ledger; sustained VRAM delta
   within 128 MiB of reference. Improvements are reported so the reference can be
   intentionally refreshed.

## Grid

full  = 3 models (q35 MoE, oss MoE/SWA-hybrid, q8d dense/SSM-hybrid)
        x ctx {2048, 16384} x mva {2000, 4096, 8192} x strategies {auto, 0..4}
smoke = 3 models x ctx 2048 x mva {4096, 8192} x auto   (per-commit, ~30 min)

## Usage

```bash
# per-commit smoke (machine must be idle - benches get the whole machine)
sh qa/run-qa.sh /tmp/qa-$(git rev-parse --short HEAD) smoke
python qa/compare-qa.py /tmp/qa-*/ledger.csv

# release gate (overnight)
sh qa/run-qa.sh /tmp/qa-full full
python qa/compare-qa.py /tmp/qa-full/ledger.csv

# after an intentional perf/plan change: refresh the reference
cp /tmp/qa-full/ledger.csv qa/reference-ledger.csv && git add qa/reference-ledger.csv
```

## Rules baked in (lessons from the 2026-08 QA arc)

- Registries are wiped and re-planned fresh for every config. Stale accumulated registry
  sections once produced a phantom +36% measurement (first-match loader picked a
  mixed-era plan whose VRAM was never sampled).
- VRAM is sampled at 1 Hz on every run; both sides' deltas land in the ledger.
- The pshard run is verbose so the tier summary (active strategy per tier) is captured;
  a `pshard DISABLED` fallback is recorded as `STOCK_FALLBACK` and fails the plan gate.
- Every number in the ledger carries the plan that produced it (strategy, overlap,
  n_pinned, cache_ubatch) — no unattributable benchmarks.
