#!/bin/sh
# pshard QA harness: stock is the golden.
#
# For every config (model x ctx x mva x strategy) this:
#   1. wipes the model's plan registry and plans fresh (forced strategy via PSHARD_STRATEGY,
#      or auto), reading cache_ubatch back from the registry,
#   2. runs STOCK twice per (model, ctx, mva), both with -fitb MVA - the SAME budget pshard
#      gets: device memory for weights + KV + compute (the fit target becomes free - MVA;
#      CUDA-side overhead is outside the budget on both sides, by decision 2026-09-01):
#        golden: -ub/-b matched to pshard's cache_ubatch, temp 0, ignore-eos -> token hash
#                ONLY (eval shape changes numerics, so the correctness gate needs it),
#        perf:   DEFAULTS ONLY -> prompt/decode t/s + VRAM delta,
#   3. runs pshard twice per config:
#        perf:        DEFAULTS ONLY -> prompt/decode t/s + VRAM delta,
#        correctness: temp 0, ignore-eos, -lv 4 -> token hash + tier summary (active
#                     strategy / n_pinned for the plan gate),
#   PERF RULE (user, 2026-09-01): a perf run carries the workload (-m -f -n -c
#   --ignore-eos, -no-cnv for batch completion) and the budget (-fitb N | -pshard -mva N)
#   and NOTHING else - no sampling, batch, cache or logging flags on either side. Both
#   sides run exactly what a user runs. --ignore-eos is workload definition (exactly N_GEN
#   decode tokens), not a knob: it touches no compute path. Everything the parsers need
#   beyond WARN-level lines comes from the correctness runs or the registry.
#   4. appends one CSV row per side to the run ledger.
#
# Generation goes to stdout (*.gen files, hashed for the token gate); logs go to stderr
# (*.log files, parsed for perf and plan info). Never merge the streams: init banners and
# verbose logs differ between sides and would poison the hash.
#
# Gates are applied by qa/compare-qa.py against a committed reference ledger.
#
# Usage:
#   sh qa/run-qa.sh <out-dir> [smoke|full]
# Environment:
#   QA_MODELS_DIR (default C:/Aditya/Models), QA_PROMPTS_DIR (default C:/Aditya/Prompts)
#   QA_BIN (default <repo>/build/bin/Release)
# The machine must be otherwise idle (benches get the whole machine), and only ONE
# instance may run at a time (registries are per-model global state).

set -u
OUT=${1:?usage: run-qa.sh <out-dir> [smoke|full]}
GRID=${2:-smoke}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${QA_BIN:-$ROOT/build/bin/Release}
MODELS=${QA_MODELS_DIR:-/c/Aditya/Models}
PROMPTS=${QA_PROMPTS_DIR:-/c/Aditya/Prompts}
# 256: the prefill->decode plan-switch re-upload (up to ~8 GiB) lands inside the decode
# timer; at 32 tokens it distorted decode_tps by up to 27%. Longer decode amortizes it
# (the switch cost stays IN the measurement - it is real - just at realistic weight).
N_GEN=256
# Perf runs sample at the model's default settings; --ignore-eos (workload) pins the
# decode window to N_GEN - without it a 2-token window measured 18 t/s on q35@4000 (the
# prefill->decode plan switch, nothing else). Safety net: a perf run whose decode window
# still comes out shorter than MIN_DECODE is retried up to PERF_TRIES times; the final
# window length lands in the ledger column decode_tokens and compare-qa refuses to gate
# decode_tps on a short window.
MIN_DECODE=$((N_GEN / 2))
PERF_TRIES=3
mkdir -p "$OUT"
cd "$BIN" || exit 1

# this build's default ubatch - the perf baselines run at defaults, so the ledger records
# it from --help rather than from a logging flag on the perf run
DEF_UB=$(./llama-completion.exe --help 2>&1 | grep -aE -- '-ub, +--ubatch-size' | grep -aoE 'default: [0-9]+' | grep -aoE '[0-9]+' | head -1)
[ -z "$DEF_UB" ] && DEF_UB=default

# single-instance lock: rival harnesses clobber each other's registries mid-flight
LOCK="$OUT/../qa-run.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "another qa run (pid $(cat "$LOCK")) is active; refusing to start" >&2
    exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

LEDGER="$OUT/ledger.csv"
# QA_RESUME=1: keep an existing ledger and skip configs already recorded (crash/restart recovery)
if [ "${QA_RESUME:-0}" = "1" ] && [ -f "$LEDGER" ]; then
    echo "resuming: $(grep -c ',pshard,' "$LEDGER") pshard rows already present"
else
    echo "config,side,model,ctx,mva,strategy_forced,strategy_active,strategy_prefill,overlap,n_pinned,n_attn_pinned,cache_ubatch,prefill_ub,prompt_tps,decode_tps,decode_tokens,vram_peak_delta,token_hash,status" > "$LEDGER"
fi

IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "# idle_used=$IDLE" >> "$LEDGER"

MODELS_LIST="q35:Qwen3.6-35B-A3B-UD-Q4_K_M oss:gpt-oss-20b-Q4_0 q8d:Qwen3.6-27B-Q8_0"
if [ "$GRID" = "full" ]; then
    CTX_LIST="2048 16384"
    MVA_LIST="4000 12000"
    STRAT_LIST="auto 0 1 2 3 4"
else
    CTX_LIST="2048"
    MVA_LIST="4000 8000"
    STRAT_LIST="auto"
fi
# targeted re-adjudication overrides (e.g. after a gate change, re-run a few cells)
MODELS_LIST=${QA_MODELS_LIST:-$MODELS_LIST}
CTX_LIST=${QA_CTX_LIST:-$CTX_LIST}
MVA_LIST=${QA_MVA_LIST:-$MVA_LIST}
STRAT_LIST=${QA_STRAT_LIST:-$STRAT_LIST}

hash_gen() { # generation file (pure stdout) -> hash
    tr -d '\r\n' < "$1" | md5sum | cut -c1-16
}
perf_field() { # log pattern -> tok/s
    grep -a "$2" "$1" | tail -1 | sed 's/.*, *//;s/ tokens per second.*//'
}

decode_runs() { # log -> number of decode steps in the perf print ("eval time = ... / N runs")
    grep -a " eval time" "$1" | tail -1 | grep -aoE "/ +[0-9]+ runs" | grep -aoE "[0-9]+"
}

run_perf() { # samp-file log-file gen-file cmd... -> "rc delta runs"; retries short decode windows
    SAMP=$1; LOG=$2; GENF=$3; shift 3
    try=1
    while :; do
        RES=$(run_side "$SAMP" "$LOG" "$GENF" "$@")
        RUNS=$(decode_runs "$LOG"); [ -z "$RUNS" ] && RUNS=0
        RCp=${RES%% *}
        if [ "$RCp" != "0" ] || [ "$RUNS" -ge "$MIN_DECODE" ] || [ "$try" -ge "$PERF_TRIES" ]; then
            break
        fi
        echo "    decode window $RUNS < $MIN_DECODE tokens (EOS at default sampling), retry $((try+1))/$PERF_TRIES" >&2
        cp "$LOG" "$LOG.try$try" 2>/dev/null; cp "$GENF" "$GENF.try$try" 2>/dev/null
        try=$((try+1))
    done
    echo "$RES $RUNS"
}

run_side() { # samp-file log-file gen-file cmd...
    SAMP=$1; LOG=$2; GENF=$3; shift 3
    : > "$SAMP"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SAMP" 2>/dev/null; sleep 1; done ) &
    SPID=$!
    "$@" > "$GENF" 2> "$LOG"
    RC=$?
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    PEAK=$(sort -n "$SAMP" | tail -1); [ -z "$PEAK" ] && PEAK=$IDLE
    echo $RC $((PEAK - IDLE))
}

for MDL in $MODELS_LIST; do
    MK=${MDL%%:*}; MP="$MODELS/${MDL#*:}.gguf"
    cat "$MP" > /dev/null   # warm file cache
    for CTX in $CTX_LIST; do
        PROMPT="$PROMPTS/prompt-512.txt"
        [ "$CTX" -ge 16384 ] && PROMPT="$PROMPTS/prompt-16k.txt"
        for MVA in $MVA_LIST; do
            # stock budget == pshard budget: -fitb MVA bounds weights + KV + compute (the old
            # -fitt (free - MVA) handed stock ~560 MiB less on q35@4000 because the fit's free
            # memory view already excludes the CUDA context, which pshard's arena never paid)
            STOCK_DONE=0
            STOCK_HASH=""
            for STRAT in $STRAT_LIST; do
                CFG="${MK}-c${CTX}-mva${MVA}-s${STRAT}"
                if grep -q "^$CFG,pshard," "$LEDGER"; then
                    echo "=== $CFG (already in ledger, skipping)"
                    continue
                fi
                echo "=== $CFG"
                rm -f "$MP.tensor_overrides.pshard_registry"

                # 1. plan fresh
                PLOG="$OUT/plan_$CFG.log"
                if [ "$STRAT" = "auto" ]; then
                    ./llama-pshard-plan-params.exe -m "$MP" -c "$CTX" -mva "$MVA" > "$PLOG" 2>&1
                else
                    env PSHARD_STRATEGY=$STRAT ./llama-pshard-plan-params.exe -m "$MP" -c "$CTX" -mva "$MVA" > "$PLOG" 2>&1
                fi
                if [ $? -ne 0 ]; then
                    echo "$CFG,pshard,$MK,$CTX,$MVA,$STRAT,PLAN_FAILED,,,,,,,,,,,FAIL" >> "$LEDGER"
                    continue
                fi
                CUB=$(grep -aoE "cache_ubatch=[0-9]+" "$MP.tensor_overrides.pshard_registry" | head -1 | cut -d= -f2)
                [ -z "$CUB" ] && CUB=$CTX

                # 2. stock (once per model/ctx/mva) - TWO runs with different jobs:
                #    golden: -ub/-b matched to pshard's cache_ubatch. Supplies the token hash
                #            ONLY (evaluation shape changes numerics; README gate 1).
                #    perf:   DEFAULTS ONLY (workload + -fitb budget; see PERF RULE above).
                #            Supplies prompt/decode t/s and the VRAM delta. A ubatch forced
                #            to cache_ubatch is NOT a perf baseline: at ub 2048 the 248k-vocab
                #            q35 needs a ~2 GB logits scratch, the 4 GB fit drops from all
                #            layers (experts on CPU) to 20/41 layers and decode halves
                #            (21 vs 44 t/s, 2026-09-01). Default sampling means the token
                #            stream is not reproducible (not hashed); --ignore-eos keeps the
                #            decode window at N_GEN.
                GLOG="$OUT/stock_golden_${MK}-c${CTX}-mva${MVA}.log"
                SLOG="$OUT/stock_${MK}-c${CTX}-mva${MVA}.log"
                if [ "$STOCK_DONE" = "0" ] && [ -f "$GLOG.gen" ] && grep -q "^${MK}-c${CTX}-mva${MVA},stock," "$LEDGER"; then
                    STOCK_HASH=$(hash_gen "$GLOG.gen")
                    STOCK_DONE=1
                fi
                if [ "$STOCK_DONE" = "0" ]; then
                    RG=$(run_side "$OUT/vram_stock_golden_${MK}-c${CTX}-mva${MVA}.txt" "$GLOG" "$GLOG.gen" \
                        ./llama-completion.exe -m "$MP" -f "$PROMPT" -n $N_GEN -c "$CTX" \
                        --temp 0 -no-cnv --ignore-eos --no-display-prompt \
                        -fitb "$MVA" -ub "$CUB" -b "$CUB")
                    GRC=${RG%% *}
                    STOCK_HASH=$(hash_gen "$GLOG.gen")
                    [ "$GRC" != "0" ] && STOCK_HASH=""
                    R=$(run_perf "$OUT/vram_stock_${MK}-c${CTX}-mva${MVA}.txt" "$SLOG" "$SLOG.gen" \
                        ./llama-completion.exe -m "$MP" -f "$PROMPT" -n $N_GEN -c "$CTX" --ignore-eos -no-cnv \
                        -fitb "$MVA")
                    RC=${R%% *}; DELTA=$(echo "$R" | awk '{print $2}'); NTOK=${R##* }
                    P=$(perf_field "$SLOG" "prompt eval time"); D=$(perf_field "$SLOG" " eval time")
                    SUB=$DEF_UB
                    ST=$([ "$RC" = "0" ] && echo OK || echo FAIL)
                    # perf baseline ran but the golden did not: the cell has numbers but no
                    # token reference -> visible (not in compare-qa's PASS_CLASSES)
                    [ "$ST" = "OK" ] && [ "$GRC" != "0" ] && ST=GOLDEN_FAIL
                    # budget enforcement label: -fitb bounds weights + KV + compute as the fit
                    # models them; CUDA context + workspaces (~300 MiB) and fit misestimates
                    # land on top (up to 6.6x nominal at 16k under -fitt). Such rows are still
                    # a-fortiori references (stock over budget and pshard still compared),
                    # but the "equal budget" claim needs the honest label.
                    if [ "$ST" = "OK" ] && [ -n "$DELTA" ] && [ "$DELTA" -gt $((MVA * 110 / 100 + 512)) ]; then
                        ST=STOCK_OVER_BUDGET
                    fi
                    if [ "$RC" != "0" ]; then
                        # a failed perf run has no meaningful perf/vram telemetry (the hash
                        # comes from the golden run and stands on its own)
                        P=""; D=""; DELTA=""; NTOK=""
                    fi
                    # stock row columns: cache_ubatch = the golden's matched shape,
                    # prefill_ub = the perf baseline's effective (default) n_ubatch
                    echo "${MK}-c${CTX}-mva${MVA},stock,$MK,$CTX,$MVA,,,,,,,${CUB},${SUB},$P,$D,$NTOK,$DELTA,$STOCK_HASH,$ST" >> "$LEDGER"
                    STOCK_DONE=1
                fi

                PSH=""; [ "$STRAT" != "auto" ] && PSH="env PSHARD_STRATEGY=$STRAT"
                # 3a. pshard PERF run - DEFAULTS ONLY (PERF RULE): perf + VRAM. WARN lines
                #     are visible at default verbosity, so "pshard DISABLED" (STOCK_FALLBACK)
                #     and pshard_prefill_ubatch_eff still parse from this log.
                RLOG="$OUT/run_$CFG.log"
                R=$(run_perf "$OUT/vram_$CFG.txt" "$RLOG" "$RLOG.gen" \
                    $PSH ./llama-completion.exe -m "$MP" -f "$PROMPT" -n $N_GEN -c "$CTX" --ignore-eos -no-cnv \
                    -pshard -mva "$MVA")
                RC=${R%% *}; DELTA=$(echo "$R" | awk '{print $2}'); NTOK=${R##* }
                P=$(perf_field "$RLOG" "prompt eval time"); D=$(perf_field "$RLOG" " eval time")
                # 3b. pshard CORRECTNESS run - same shape as the stock golden (temp 0,
                #     ignore-eos, no-display-prompt) -> token hash; -lv 4 (library INFO, no
                #     DEBUG) -> tier summary for the plan gate. Not a perf measurement.
                CLOG="$OUT/gen_$CFG.log"
                RCg=$(run_side "$OUT/vram_gen_$CFG.txt" "$CLOG" "$CLOG.gen" \
                    $PSH ./llama-completion.exe -m "$MP" -f "$PROMPT" -n $N_GEN -c "$CTX" \
                    --temp 0 -no-cnv --ignore-eos --no-display-prompt -lv 4 \
                    -pshard -mva "$MVA")
                RCg=${RCg%% *}
                H=$(hash_gen "$CLOG.gen")
                [ "$RCg" != "0" ] && H=""
                T0=$(grep -a "warmup_plan_reserves:   tier 0" "$CLOG" | head -1)
                ACT=$(echo "$T0" | grep -aoE "bs=1[[:space:]]+[A-Z_]+" | awk '{print $2}')
                NP=$(echo "$T0" | grep -aoE "(^| )n_pinned=[0-9]+" | cut -d= -f2)
                OVL=$(grep -aoE "overlap=[01]" "$MP.tensor_overrides.pshard_registry" | head -1 | cut -d= -f2)
                # prefill attribution: the bs=CUB tier's strategy can differ from tier 0
                # (auto picks per tier), and the runtime may prefill at a smaller ubatch
                SP=$(grep -aA1 "bs=$CUB\]" "$MP.tensor_overrides.pshard_registry" | grep -aoE "strategy=[A-Z_]+" | tail -1 | cut -d= -f2)
                NA=$(grep -aA1 "bs=1\]" "$MP.tensor_overrides.pshard_registry" | grep -aoE "n_attn_pinned=[0-9]+" | head -1 | cut -d= -f2)
                PUB=$(grep -aoE "pshard_prefill_ubatch_eff=[0-9]+" "$RLOG" "$CLOG" | head -1 | grep -aoE "[0-9]+$")
                [ -z "$PUB" ] && PUB=$CUB
                if grep -aq "pshard DISABLED" "$RLOG" "$CLOG"; then ACT="STOCK_FALLBACK"; fi
                ST=OK
                # both pshard runs must succeed: no perf without a hash, no hash without perf
                [ "$RC" != "0" ] || [ "$RCg" != "0" ] && ST=FAIL
                if [ "$ST" = "OK" ] && [ -n "$STOCK_HASH" ] && [ "$H" != "$STOCK_HASH" ]; then
                    # secondary gate: token streams may legitimately diverge at temp 0
                    # (near-tie logits; q35 diverges at token ~35 of 256); PPL must then
                    # agree within 0.5% at the SAME budget and the SAME eval shape.
                    CH=8; [ "$CTX" -ge 16384 ] && CH=2
                    PPLC="$PROMPTS/prompt-256k.txt"
                    # run the PSHARD side FIRST (verbose: the mirror needs its EXECUTED
                    # prefill ubatch), then give stock the same eval shape. Two shape deltas
                    # proven to matter on gpt-oss@16k: the executed ubatch, and SWA cache
                    # sizing (pshard allocates the full SWA cache; stock defaults to
                    # window+batch - that alone was a 2.6% PPL delta).
                    if [ "$STRAT" = "auto" ]; then
                        ./llama-perplexity.exe -m "$MP" -f "$PPLC" -c "$CTX" --chunks $CH \
                            -pshard -mva "$MVA" -v > "$OUT/ppl_pshard_$CFG.log" 2>&1
                    else
                        env PSHARD_STRATEGY=$STRAT ./llama-perplexity.exe -m "$MP" -f "$PPLC" -c "$CTX" --chunks $CH \
                            -pshard -mva "$MVA" -v > "$OUT/ppl_pshard_$CFG.log" 2>&1
                    fi
                    PP_=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$OUT/ppl_pshard_$CFG.log" | grep -aoE "[0-9.]+$")
                    # executed prefill ubatch + the strategy that actually ran at it
                    PB=$(grep -aoE "pshard_prefill_ubatch_eff=[0-9]+" "$OUT/ppl_pshard_$CFG.log" | head -1 | cut -d= -f2)
                    [ -z "$PB" ] && PB=${PUB:-$CUB}
                    # stock mirror = the SAME budget (-fitb MVA), the SAME eval shape (pshard's
                    # executed prefill ubatch) and the same SWA cache sizing (--swa-full).
                    # Compute placement is deliberately NOT mirrored (no -ngl, no -ot;
                    # 2026-09-01, user): stock runs the placement its own fit picks for the
                    # budget, exactly as a user would (-ngl 99 also cannot load a model
                    # larger than the card: q35 Q4_K_M is 20 GB on 16 GB). The residual
                    # therefore includes GPU-vs-CPU kernel math for whatever pshard computes
                    # on the GPU that stock's fit put on the CPU (q35@4000: pshard 1.2574 vs
                    # stock 1.2618 = 0.35% of the 0.5% band). Raw-text-hypersensitive models
                    # (gpt-oss: stock alone spans 1401.9 all-GPU -> 4025.0 experts-on-CPU at
                    # one ubatch) can report PPL_MISMATCH from placement alone - read those
                    # cells with that in mind.
                    ./llama-perplexity.exe -m "$MP" -f "$PPLC" -c "$CTX" --chunks $CH \
                        -fitb "$MVA" -ub "$PB" -b "$PB" --swa-full > "$OUT/ppl_stock_$CFG.log" 2>&1
                    PS_=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$OUT/ppl_stock_$CFG.log" | grep -aoE "[0-9.]+$")
                    if [ -n "$PS_" ] && [ -n "$PP_" ] &&                        awk -v a="$PS_" -v b="$PP_" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d <= 0.005*a)}'; then
                        ST="TOKEN_DIVERGED_PPL_OK(stock=$PS_ pshard=$PP_)"
                    elif [ -z "$PS_" ]; then
                        # stock has no PPL here (typically the cell's stock baseline cannot
                        # run at all) - that is a missing baseline, not a mismatch
                        ST="NO_BASELINE(pshard=$PP_)"
                    else
                        ST="PPL_MISMATCH(stock=$PS_ pshard=$PP_)"
                    fi
                fi
                if [ "$ST" = "FAIL" ]; then P=""; D=""; DELTA=""; H=""; NTOK=""; fi
                echo "$CFG,pshard,$MK,$CTX,$MVA,$STRAT,$ACT,$SP,$OVL,$NP,$NA,$CUB,$PUB,$P,$D,$NTOK,$DELTA,$H,$ST" >> "$LEDGER"
                echo "    active=$ACT prefill=$SP/ub$PUB ovl=$OVL np=$NP attn=$NA prompt=$P decode=$D (${NTOK}tok) vram=+$DELTA $ST"
            done
        done
    done
done
echo "QA_RUN_DONE $LEDGER"
