# pshard QA harness

Stock llama.cpp is the golden. Three gates, applied over the full config grid:

1. **Correctness**: pshard generation must be token-identical to stock at temp 0.
   The stock GOLDEN run has `-ub`/`-b` matched to pshard's planner-chosen `cache_ubatch`
   (evaluation shape changes numerics on some models — gpt-oss PPL moves 23% on ubatch
   alone), and `-fitt (free - budget)` so both sides get the same effective VRAM budget.
   The golden supplies the token hash only; it is not the perf baseline (see gate 3).
   When tokens diverge (near-tie logits at temp 0; q35 diverges at token ~35 of 256),
   the PPL-parity fallback compares both sides at the SAME budget (`-fitt`) and the SAME
   eval shape: stock gets pshard's executed prefill ubatch as `-ub`/`-b` (read from the
   verbose pshard PPL log) and `--swa-full` (pshard allocates the full SWA cache; the
   stock default was a 2.6% PPL delta on gpt-oss@16k). Tolerance 0.5%.
   Compute placement is NOT mirrored (decision 2026-09-01): stock uses no `-ngl` and no
   `-ot`, it runs the placement its own fit picks for the budget, as a user would. The
   residual therefore includes GPU-vs-CPU kernel math for the experts pshard streams
   through the GPU while stock's fit keeps them on the CPU: q35@4000 pshard 1.2574 vs
   stock 1.2618 (0.35%), with forced s2 (experts mostly on CPU) at 1.2602 in between.
   Caveat: raw-text-hypersensitive models can fail this gate on placement alone — the
   2026-08-27 gpt-oss calibration had stock itself span PPL 1401.9 (all-GPU) to 4025.0
   (experts on CPU) at one ubatch, and only a placement-matched mirror reproduced pshard
   to <=0.0007%. Read gpt-oss `PPL_MISMATCH` cells with that in mind.
2. **Plan stability**: the active tier-0 strategy, overlap mode, and n_pinned must match
   the reference ledger. This catches silent fallbacks (forced strategy unviable -> stock
   or ATTNPRIO fallback) and planner drift — both produced misleading numbers before.
3. **Perf**: prompt/decode t/s within 5% of the reference ledger; sustained VRAM delta
   within 128 MiB of reference. Improvements are reported so the reference can be
   intentionally refreshed.
   **Perf rule (2026-09-01):** a perf run carries the workload (`-m -f -n -c
   --ignore-eos`, `-no-cnv` for batch completion) and the budget (`-fitt N` for stock,
   `-pshard -mva N` for pshard) and nothing else — no sampling, batch, cache or logging
   flags on either side. `--ignore-eos` is workload definition (exactly N_GEN decode
   tokens) and touches no compute path. Both
   sides run exactly what a user runs. Perf and correctness are therefore separate runs
   on BOTH sides: stock golden + stock baseline, pshard correctness + pshard perf. Forcing
   `ub = cache_ubatch` on stock was never a baseline: at ub 2048 the 248k-vocab q35 needs a
   ~2 GB logits scratch, so the 4 GB fit fell from all layers (experts on CPU, 44 t/s) to
   20/41 layers (21 t/s). Default sampling makes perf-run token streams non-reproducible
   (they are not hashed). Without `--ignore-eos` an EOS ended decode early: a 2-token
   window measured 18 t/s on q35@4000 (the prefill->decode plan switch alone). As a safety
   net a perf run whose decode window still comes out under N_GEN/2 is retried up to 3
   times; the final window length is the ledger column `decode_tokens`, and compare-qa
   hard-fails a row whose window is short and does not gate its decode_tps.
   Stock ledger rows carry `cache_ubatch` = the golden's matched shape and `prefill_ub` =
   this build's default ubatch (read from `--help`).

## Grid

full  = 3 models (q35 MoE, oss MoE/SWA-hybrid, q8d dense/SSM-hybrid)
        x ctx {2048, 16384} x mva {4000, 12000} x strategies {auto, 0..4}
smoke = 3 models x ctx 2048 x mva {4000, 8000} x auto   (per-commit, ~30 min)
(override with QA_MODELS_LIST / QA_CTX_LIST / QA_MVA_LIST / QA_STRAT_LIST)

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
- Perf runs carry no logging flag. `pshard DISABLED` and `pshard_prefill_ubatch_eff` are
  WARN lines and still parse from the perf log; the tier summary (active strategy per
  tier) is an INFO line and comes from the pshard correctness run at `-lv 4` (library
  INFO, no DEBUG). A `pshard DISABLED` in either run is recorded as `STOCK_FALLBACK` and
  fails the plan gate. Only the PPL-parity runs are verbose.
- Stock's golden (matched ubatch, hash only) and perf baseline (defaults) are separate
  runs. A golden that fails while the baseline runs is recorded as `GOLDEN_FAIL` on the
  stock row, which compare-qa hard-fails (no token reference). A pshard config whose
  perf or correctness run fails is `FAIL` (no perf without a hash, no hash without perf).
- Every number in the ledger carries the plan that produced it (strategy, overlap,
  n_pinned, cache_ubatch) — no unattributable benchmarks.
