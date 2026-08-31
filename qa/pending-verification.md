# Pending verification queue

Changes land with build-only checks; GPU verification is BATCHED here and drained
in one session (machine exclusive, zombie-check first: `tasklist /fi "IMAGENAME eq
llama-perplexity.exe"` etc. - TaskStop'd harness shells survive as detached sh.exe
and corrupt concurrent measurements).

## Drained 2026-08-31 (targeted items; grid + nem HELD by user)

- Loader determinism: VERIFIED - exact hit / smaller-with-warning / honest refusal.
- Switch-cost constants: VERIFIED - variant header switch_mb=509.1 attn_frac=0.12
  head_mb=885.4 pcie=45.0 (matches hand arithmetic); behavioral check rides the grid.
- ids-cross: VERIFIED - oss s4 flags exactly tier bs=1 (pair gate) with router
  patterns; q35-16k@2000 forced-s4 decode 7.65 -> 19.99 t/s (+161%), prompt
  unchanged, token gate OK. (Two silently no-op'd serialization hunks were found
  and restored en route - 7c6a48bbe. Lesson: verify batch replaces by count.)
- DSv4 perf (clean machine): s3 forced = 52.1 prompt / 10.87 decode t/s. NEW BUG
  found below.

## Queued

0. ~~DSv4 streamed-attn runtime crash~~ **FIXED**: root cause was one level up from
   the write-cells guess - llama_kv_cache_dsa(+_iswa) never overrode
   get_pipe_shards(), so the ENTIRE pipe-shard machinery was invisible for
   dual-cache models (no tensor assignment, no uploads, no writeback). Added the
   overrides ([mla, lid] / [mla, lid, swa]) + write-cells binder cases for both
   DSA context types with dedicated lid storage. Crash repro (auto, prompt-512,
   -mva 8000) now completes: 21.7 prompt / 9.9 decode t/s, coherent output.
   FOLLOW-UP observation: DSv4 auto prefill (21.7) trails forced-s3 (52.1) - the
   bs=512 FFNCPU tier's predicted 100 tps measures ~22; predictor accuracy on
   MLA/dense-lead architectures is untuned. Value A/B vs stock rides the held
   grid-class runs.
0b. **Harness nit**: strategy_prefill parses empty when the top (bs=CUB) tier is
   not_viable - key the parse off the effective prefill tier (bs=PUB) instead.

1. **Full 144-grid rerun** - restart FRESH (plans changed under items 5/1/7 +
   DSA fix; the 42 pre-change rows are stale): `sh qa/run-qa.sh /tmp/qa-full-v3 full`.
   Then compare-qa vs reference, refresh ledger (schema v2), commit, push.
2. **Nemotron full slice** - from scratch: `QA_MODELS_LIST="nem:nvidia_Nemotron-3-Nano-30B-A3B-Q4_0" ...`
   Seeds the nem reference rows.
3. ~~Switch-cost behavioral check~~ **VERIFIED 2026-08-31**: switch_ms fields
   populated (0 for tier-0, 83-105ms for residency deltas); prompt-1500 -> ub 2048
   and prompt-16k -> ub 8192 both hand-verify as argmin of n_iters*ts/tps +
   2*switch_ms. CAVEAT (small follow-up): strategies that pin attention
   structurally (ATTNPIN/LAYERSTREAM) do not record it in n_attn_pinned, so
   switch_ms between them can overcharge; fine for 0-vs-100ms discrimination.
4. ~~DSv4 predictor tuning~~ **FIXED 2026-08-31**: IQ-quant expert matmuls had no
   benchmark entries -> priced by memory bandwidth alone (118 nodes for 224ms)
   while dequant-compute-bound at batch. Added a compute floor for quantized CPU
   matmuls without a same-type entry: max(bytes/bw, ops/floor) where floor = the
   slowest BATCH-measured (B>=32) CPU matmul rate, applied only at M>=32 rows
   (bs=1 matvecs are genuinely memory-bound - the first floor version overcharged
   them and flipped tier-0; caught by ladder inspection). Result: DSv4 ladder all
   coherent STATIC, tier-0 predicted 10.89 vs measured 10.87 (0.2%!), auto prompt
   21.7 -> 46.6 t/s (+115%), within ~10% of forced-s3. A/B-vs-stock cell rides
   the held grid-class runs.
5. **Slot carve-out follow-up (per-slot fence redesign)** - infrastructure landed
   opt-in (GGML_SLOT_REGIONS=1): weight slots allocate in dedicated per-backend
   regions (dump: 45602 slot-vs-activation overlaps -> 0). The narrowed per-bid
   fence was UNSOUND (one-split race on same-shard slot reuse -> degenerate output,
   caught by hash gate) and is reverted to the conservative fence; perf therefore
   baseline-equal, VRAM +0.5-1GB per streamed config when enabled (why opt-in).
   Next attempt: address-keyed per-slot events - slot reuse is a strict 3-address
   rotation per shard (gate/up/down classes), so 3 events/bid keyed by slot address
   suffice; also audit writeback-staging cross-eval ordering, the suspected real
   hole. The reserve_n_size caller-array overrun fix and the fold are always-on.
6. **ALTERNATE adjudication (user question)** - after the full grid rerun under the
   fixed predictor + ids-cross: does s4 ever win a cell against re-planned s1/s3?
   If auto never picks it and forced-s4 never beats the best alternative, ALTERNATE
   earns retirement from the decode tiers (stays as a prefill design point).

## Contamination note (2026-08-30)

Two "stopped" harness runs kept running detached and overlapped each other and
ad-hoc tests. Perf numbers from that window are invalid; correctness outcomes
(DSv4 runs + output text, loader behavior classes) and planner PREDICTION numbers
(pure compute, no GPU contention sensitivity) remain valid.
