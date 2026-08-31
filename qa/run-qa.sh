#!/bin/sh
# pshard QA harness: stock is the golden.
#
# For every config (model x ctx x mva x strategy) this:
#   1. wipes the model's plan registry and plans fresh (forced strategy via PSHARD_STRATEGY,
#      or auto), reading cache_ubatch back from the registry,
#   2. runs the STOCK baseline once per (model, ctx, mva) with -fitt equalized to the same
#      budget and -ub matched to pshard's cache_ubatch (value-comparable),
#   3. runs pshard with -v (tier summary => active strategy / overlap / n_pinned),
#      sampling VRAM at 1 Hz,
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
mkdir -p "$OUT"
cd "$BIN" || exit 1

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
    echo "config,side,model,ctx,mva,strategy_forced,strategy_active,strategy_prefill,overlap,n_pinned,n_attn_pinned,cache_ubatch,prefill_ub,prompt_tps,decode_tps,vram_peak_delta,token_hash,status" > "$LEDGER"
fi

IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "# idle_used=$IDLE" >> "$LEDGER"
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)

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
            FITT=$((FREE - MVA))
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

                # 2. stock baseline (once per model/ctx/mva), ub matched to pshard's cache_ubatch
                SLOG="$OUT/stock_${MK}-c${CTX}-mva${MVA}.log"
                if [ "$STOCK_DONE" = "0" ] && [ -f "$SLOG.gen" ] && grep -q "^${MK}-c${CTX}-mva${MVA},stock," "$LEDGER"; then
                    STOCK_HASH=$(hash_gen "$SLOG.gen")
                    STOCK_DONE=1
                fi
                if [ "$STOCK_DONE" = "0" ]; then
                    R=$(run_side "$OUT/vram_stock_${MK}-c${CTX}-mva${MVA}.txt" "$SLOG" "$SLOG.gen" \
                        ./llama-completion.exe -m "$MP" -f "$PROMPT" -n $N_GEN -c "$CTX" \
                        --temp 0 -no-cnv --ignore-eos --no-display-prompt \
                        -fitt "$FITT" -ub "$CUB" -b "$CUB")
                    RC=${R%% *}; DELTA=${R##* }
                    P=$(perf_field "$SLOG" "prompt eval time"); D=$(perf_field "$SLOG" " eval time")
                    STOCK_HASH=$(hash_gen "$SLOG.gen")
                    ST=$([ "$RC" = "0" ] && echo OK || echo FAIL)
                    if [ "$RC" != "0" ]; then
                        # a failed run has no meaningful perf/hash/vram telemetry
                        P=""; D=""; DELTA=""; STOCK_HASH=""
                    fi
                    echo "${MK}-c${CTX}-mva${MVA},stock,$MK,$CTX,$MVA,,,,,,,${CUB},,$P,$D,$DELTA,$STOCK_HASH,$ST" >> "$LEDGER"
                    STOCK_DONE=1
                fi

                # 3. pshard run (stderr verbose for tier summary; stdout = generation)
                RLOG="$OUT/run_$CFG.log"
                if [ "$STRAT" = "auto" ]; then
                    R=$(run_side "$OUT/vram_$CFG.txt" "$RLOG" "$RLOG.gen" \
                        ./llama-completion.exe -m "$MP" -f "$PROMPT" -n $N_GEN -c "$CTX" \
                        --temp 0 -no-cnv --ignore-eos --no-display-prompt -v \
                        -pshard -mva "$MVA")
                else
                    R=$(run_side "$OUT/vram_$CFG.txt" "$RLOG" "$RLOG.gen" \
                        env PSHARD_STRATEGY=$STRAT ./llama-completion.exe -m "$MP" -f "$PROMPT" -n $N_GEN -c "$CTX" \
                        --temp 0 -no-cnv --ignore-eos --no-display-prompt -v \
                        -pshard -mva "$MVA")
                fi
                RC=${R%% *}; DELTA=${R##* }
                P=$(perf_field "$RLOG" "prompt eval time"); D=$(perf_field "$RLOG" " eval time")
                H=$(hash_gen "$RLOG.gen")
                T0=$(grep -a "warmup_plan_reserves:   tier 0" "$RLOG" | head -1)
                ACT=$(echo "$T0" | grep -aoE "bs=1[[:space:]]+[A-Z_]+" | awk '{print $2}')
                NP=$(echo "$T0" | grep -aoE "(^| )n_pinned=[0-9]+" | cut -d= -f2)
                OVL=$(grep -aoE "overlap=[01]" "$MP.tensor_overrides.pshard_registry" | head -1 | cut -d= -f2)
                # prefill attribution: the bs=CUB tier's strategy can differ from tier 0
                # (auto picks per tier), and the runtime may prefill at a smaller ubatch
                SP=$(grep -aA1 "bs=$CUB\]" "$MP.tensor_overrides.pshard_registry" | grep -aoE "strategy=[A-Z_]+" | tail -1 | cut -d= -f2)
                NA=$(grep -aA1 "bs=1\]" "$MP.tensor_overrides.pshard_registry" | grep -aoE "n_attn_pinned=[0-9]+" | head -1 | cut -d= -f2)
                PUB=$(grep -aoE "pshard_prefill_ubatch_eff=[0-9]+" "$RLOG" | head -1 | cut -d= -f2)
                [ -z "$PUB" ] && PUB=$CUB
                if grep -aq "pshard DISABLED" "$RLOG"; then ACT="STOCK_FALLBACK"; fi
                ST=OK
                [ "$RC" != "0" ] && ST=FAIL
                if [ "$ST" = "OK" ] && [ -n "$STOCK_HASH" ] && [ "$H" != "$STOCK_HASH" ]; then
                    # secondary gate: token streams may legitimately diverge across placements
                    # at temp 0 (near-tie logits); PPL must agree once BOTH eval shape and
                    # compute placement are matched. Proven on gpt-oss (raw-text PPL is
                    # hypersensitive): STOCK alone spans 1401.9 (all-GPU) -> 4025.0 (experts
                    # on CPU) at the same ubatch, and stock with the plan's exact expert
                    # placement reproduces pshard's PPL to 5 digits (3440.80 vs 3440.82).
                    CH=8; [ "$CTX" -ge 16384 ] && CH=2
                    PPLC="$PROMPTS/prompt-256k.txt"
                    # run the PSHARD side FIRST (verbose: the mirror needs its EXECUTED
                    # prefill tier), then mirror stock to the actual executed config.
                    # Three deltas proven to matter (gpt-oss@16k certified to 5 digits
                    # once all matched, 39368.10 vs 39367.59): the executed tier's
                    # placement class, its ubatch, and SWA cache sizing (pshard allocates
                    # the full SWA cache; stock defaults to window+batch - that alone was
                    # a 2.6% PPL delta on gpt-oss@16k).
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
                    SPP=$(grep -a "pshard_apply_plan: strategy=" "$OUT/ppl_pshard_$CFG.log" | grep -a "bs=$PB " | tail -1 | grep -aoE "strategy=[A-Z_]+" | cut -d= -f2)
                    STOCK_PPL_ARGS="-fitt $FITT"
                    case "$SPP" in
                      GPUONLY_*)
                        # streamed weights compute on the GPU: math == fully resident
                        STOCK_PPL_ARGS="-ngl 99"
                        ;;
                      DYNAMIC_FFNCPU*|DYNAMIC_FFN_ALTERNATE|STATIC_ATTNPRIO_ALLMODELS)
                        # CPU-delegate + static-split strategies: give stock the same expert
                        # placement (MoE models; dense placements have no _exps match and
                        # fall back to the budget-equalized stock baseline)
                        # n_layer only prints in the VERBOSE (pshard) log
                        NL=$(grep -aoE "n_layer += +[0-9]+" "$RLOG" | head -1 | grep -aoE "[0-9]+$")
                        # anchor: "n_attn_pinned=N" contains the substring "n_pinned=N"
                        NPP=$(grep -aA1 "bs=$PB\]" "$MP.tensor_overrides.pshard_registry" | grep -aoE "(^| )n_pinned=[0-9]+" | tail -1 | cut -d= -f2)
                        LIST=""
                        i=${NPP:-0}
                        while [ "$i" -lt "${NL:-0}" ]; do
                            # ALTERNATE delegates only even unpinned layers to the CPU
                            if [ "$SPP" = "DYNAMIC_FFN_ALTERNATE" ] && [ $((i % 2)) -ne 0 ]; then i=$((i+1)); continue; fi
                            LIST="$LIST${LIST:+|}$i"
                            i=$((i+1))
                        done
                        if [ -n "$LIST" ]; then
                            STOCK_PPL_ARGS="-ngl 99 -ot blk\\.($LIST)\\.ffn_.*_exps=CPU --no-op-offload"
                        fi
                        ;;
                    esac
                    # --swa-full: match pshard's full-size SWA cache (no-op for non-SWA)
                    ./llama-perplexity.exe -m "$MP" -f "$PPLC" -c "$CTX" --chunks $CH \
                        $STOCK_PPL_ARGS -ub "$PB" -b "$PB" --swa-full > "$OUT/ppl_stock_$CFG.log" 2>&1
                    PS_=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$OUT/ppl_stock_$CFG.log" | grep -aoE "[0-9.]+$")
                    if [ -n "$PS_" ] && [ -n "$PP_" ] &&                        awk -v a="$PS_" -v b="$PP_" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d <= 0.005*a)}'; then
                        ST=TOKEN_DIVERGED_PPL_OK
                    elif [ -z "$PS_" ]; then
                        # stock has no PPL here (typically the cell's stock baseline cannot
                        # run at all) - that is a missing baseline, not a mismatch
                        ST="NO_BASELINE(pshard=$PP_)"
                    else
                        ST="PPL_MISMATCH(stock=$PS_ pshard=$PP_)"
                    fi
                fi
                if [ "$RC" != "0" ]; then P=""; D=""; DELTA=""; H=""; fi
                echo "$CFG,pshard,$MK,$CTX,$MVA,$STRAT,$ACT,$SP,$OVL,$NP,$NA,$CUB,$PUB,$P,$D,$DELTA,$H,$ST" >> "$LEDGER"
                echo "    active=$ACT prefill=$SP/ub$PUB ovl=$OVL np=$NP attn=$NA prompt=$P decode=$D vram=+$DELTA $ST"
            done
        done
    done
done
echo "QA_RUN_DONE $LEDGER"
