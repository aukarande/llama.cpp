#!/bin/sh
# pshard perf grid: stock vs legacy strategies vs expert pool, q35 and DSv4.
# Design and cell list: qa/perf-grid.md. This is NOT the held 144-cell QA grid (run-qa.sh).
#
# Usage:
#   sh qa/perf-grid.sh <out-dir> [core|ext] [--list]
# Environment:
#   QA_MODELS_DIR (default C:/Aditya/Models), QA_PROMPTS_DIR (default C:/Aditya/Prompts),
#   QA_BIN (default <repo>/build/bin/Release)
#   QA_FULL_MVA=N        the "full" budget in MiB (default 14500: the 16 GB card's idle
#                        free VRAM minus a margin, FIXED so plan and run agree and grids
#                        stay comparable; the registry variant is keyed on the exact budget)
#   QA_RESUME=1          keep an existing ledger, skip cells already recorded
#   QA_ONLY=<regex>      run only cells whose name matches (grep -E)
#   QA_GRID_MODELS=...   subset of "q35 dsv4" (spec cells follow their target model)
#
# RULES (user, 2026-09-01 / 2026-09-04):
#   - decode length is 128 tokens on every perf arm; 32 tokens = correctness gate only;
#   - a perf run carries workload + budget and nothing else (no temp, ub, lv, ngl, ot);
#   - thread count is always the default (user rule 2026-09-04: never set -t);
#   - every pshard cell plans fresh with the run's own environment (PSHARD_STRATEGY,
#     PSHARD_MISS_POLICY and the spec flags are fingerprinted) - consecutive cells
#     that differ only in non-fingerprinted knobs (PSHARD_POOL_PREDICT) share one plan;
#   - benches get the whole machine; one instance at a time.

set -u
OUT=${1:?usage: perf-grid.sh <out-dir> [core|ext] [--list]}
GRID=${2:-core}
LIST=0
for a in "$@"; do [ "$a" = "--list" ] && LIST=1; done
[ "$GRID" = "--list" ] && GRID=core
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${QA_BIN:-$ROOT/build/bin/Release}
MODELS=${QA_MODELS_DIR:-/c/Aditya/Models}
PROMPTS=${QA_PROMPTS_DIR:-/c/Aditya/Prompts}
PPL_TEXT="$PROMPTS/prompt-256k.txt"
FULL=${QA_FULL_MVA:-14500}
N_GEN=128          # perf decode window (user rule: always 128)
N_GATE=32          # correctness gate window (md5 + counters)
MIN_DECODE=64      # a perf row whose decode window ends shorter than this is marked SHORT
GRID_MODELS=${QA_GRID_MODELS:-"q35 dsv4"}
ONLY=${QA_ONLY:-}

# model keys -> files
mpath() { # key -> target gguf
    case $1 in
        q35)    echo "$MODELS/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" ;;
        q35mtp) echo "$MODELS/Qwen3.6-35B-A3B-UD-Q4_K_M-wmtp.gguf" ;;
        dsv4)   echo "$MODELS/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf" ;;
    esac
}
dpath() { # model spec -> draft gguf ("" for none / mtp)
    case "$1/$2" in
        q35/dspark)  echo "$MODELS/dspark-Qwen3.6-35B-A3B-Q8_0.gguf" ;;
        q35/dflash)  echo "$MODELS/dflash-Qwen3.6-35B-A3B-Q8_0.gguf" ;;
        dsv4/dspark) echo "$MODELS/DeepseekV4-Flash-20260731-DSpark.gguf" ;;
        *) echo "" ;;
    esac
}
base_model() { # spec-target key -> model family for QA_GRID_MODELS filtering
    case $1 in q35mtp) echo q35 ;; *) echo "$1" ;; esac
}
prompt_file() { case $1 in 512) echo "$PROMPTS/prompt-512.txt" ;; 4k) echo "$PROMPTS/prompt-4k.txt" ;; esac; }
prompt_ctx()  { case $1 in 512) echo 2048 ;; 4k) echo 8192 ;; esac; }
mva_of()      { case $1 in full) echo "$FULL" ;; *) echo "$1" ;; esac; }

# ---------------------------------------------------------------- cell list
# one line per cell: kind|model|mva|prompt|spec|arm|predict|threads|noovl
#   kind   perf | gate | ppl
#   mva    MiB or "full"
#   prompt 512 | 4k
#   arm    stock | auto | s0..s4 | pool:<policy> | pool:plan | poolauto
CELLS=""
add() { CELLS="$CELLS
$1|$2|$3|$4|$5|$6|$7|$8|$9"; }
plain_set() { # model mva prompt pool_ok(0/1)  -> perf + gate cells of one (model, budget, prompt)
    M=$1; B=$2; PR=$3; POOL=$4
    add perf $M $B $PR none stock 0 0 0
    add perf $M $B $PR none auto  0 0 0
    for s in 0 1 2 3 4; do add perf $M $B $PR none s$s 0 0 0; done
    if [ "$POOL" = "1" ]; then
        # the +pred variant follows its base cell: same fingerprint -> one plan for both
        for p in fetch hybrid cpu_admit; do add perf $M $B $PR none pool:$p 0 0 0; add perf $M $B $PR none pool:$p 1 0 0; done
        for p in cpu_exec fetch_on_2nd_miss; do add perf $M $B $PR none pool:$p 0 0 0; done
        add perf $M $B $PR none pool:plan 0 0 0
        add perf $M $B $PR none poolauto 0 0 0
        add perf $M $B $PR none poolauto 1 0 0
    fi
    add gate $M $B $PR none stock 0 0 0
    add gate $M $B $PR none auto  0 0 0
    if [ "$POOL" = "1" ]; then
        if [ "$PR" = "512" ]; then
            for p in fetch cpu_exec fetch_on_2nd_miss hybrid cpu_admit; do add gate $M $B $PR none pool:$p 0 0 0; done
            add gate $M $B $PR none pool:fetch 1 0 0
        else
            add gate $M $B $PR none pool:fetch 0 0 0
        fi
    fi
}
spec_set() { # model mva spec arms...   (prompt 512: speculation is a decode question)
    M=$1; B=$2; SPC=$3; shift 3
    for arm in "$@"; do
        case $arm in
            *+pred) add perf $M $B 512 $SPC "${arm%+pred}" 1 0 0 ;;
            *)      add perf $M $B 512 $SPC "$arm" 0 0 0 ;;
        esac
    done
}
# core. Budgets 4000 / 8000 / full (user, 2026-09-04); prompts 512 and 4k.
# DSv4's pool needs ~9.2 GB of pins (attention, routers, norms, head) before the first
# slot, so below the full budget its pool arms cannot plan: only stock and legacy there.
for PR in 512 4k; do
    plain_set q35  4000 $PR 1
    plain_set q35  8000 $PR 1
    plain_set q35  full $PR 1
    plain_set dsv4 4000 $PR 0
    plain_set dsv4 8000 $PR 0
    plain_set dsv4 full $PR 1
done
for B in 4000 8000 full; do spec_set q35mtp $B mtp stock auto pool:fetch pool:fetch+pred pool:plan poolauto; done
spec_set dsv4 full dspark stock auto pool:fetch pool:fetch+pred pool:hybrid pool:plan poolauto
for B in 4000 8000 full; do for arm in stock auto pool:fetch; do add ppl q35 $B 512 none $arm 0 0 0; done; done
for arm in auto pool:fetch; do add ppl dsv4 full 512 none $arm 0 0 0; done
if [ "$GRID" = "ext" ]; then
    for PR in 512 4k; do plain_set q35 2700 $PR 1; done          # the tight-budget showcase
    spec_set q35 8000 dflash stock auto pool:fetch pool:fetch+pred pool:plan poolauto
    spec_set q35 8000 dspark stock auto pool:fetch pool:fetch+pred pool:plan poolauto
    for MB in "q35 8000" "dsv4 full"; do
        set -- $MB
        for p in hybrid cpu_admit; do add perf $1 $2 512 none pool:$p 0 0 1; done
    done
    add ppl dsv4 full 512 none stock 0 0 0
fi

cell_name() { # kind model mva prompt spec arm predict threads noovl  (no subshells: fork is slow on Windows)
    case $6 in pool:*) a="pool_${6#pool:}" ;; *) a=$6 ;; esac
    n="$2-$3-$4-$5-$a"
    [ "$7" = "1" ]  && n="$n-pred"
    [ "$8" != "0" ] && n="$n-t$8"
    [ "$9" = "1" ]  && n="$n-noovl"
    [ "$1" != "perf" ] && n="$n-$1"
    echo "$n"
}

if [ "$LIST" = "1" ]; then
    echo "$CELLS" | while IFS='|' read -r K M B PR S A P T N; do
        [ -z "$K" ] && continue
        case $M in q35mtp) BM=q35 ;; *) BM=$M ;; esac
        case " $GRID_MODELS " in *" $BM "*) ;; *) continue ;; esac
        cell_name "$K" "$M" "$B" "$PR" "$S" "$A" "$P" "$T" "$N"
    done | { if [ -n "$ONLY" ]; then grep -E "$ONLY"; else cat; fi; } | awk -v g="$GRID" '{n++; print} END {print "# " n " cells (" g ")"}'
    exit 0
fi

# ---------------------------------------------------------------- run support
mkdir -p "$OUT"
cd "$BIN" || exit 1
LEDGER="$OUT/ledger.csv"
HDR="cell,kind,model,mva,prompt,ctx,spec,arm,predict,threads,noovl,rc,prompt_tps,decode_tps,decode_tokens,accept_pct,vram_peak_delta_mib,strategy_active,n_pinned,miss_policy,pool_slots,md5,h,misses_per_token,ppl,status"
if [ "${QA_RESUME:-0}" = "1" ] && [ -f "$LEDGER" ]; then
    echo "resuming: $(grep -vc -e '^#' -e '^cell,' "$LEDGER") rows already present"
else
    echo "$HDR" > "$LEDGER"
fi
IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ -z "$IDLE" ] && IDLE=0
echo "# idle_used=$IDLE full_mva=$FULL git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) date=$(date +%Y-%m-%dT%H:%M)" >> "$LEDGER"

# registries: back up at start, restore at exit (the grid rewrites them per cell)
BK="$OUT/registry-backup"; mkdir -p "$BK"
for k in q35 q35mtp dsv4; do
    R="$(mpath $k).tensor_overrides.pshard_registry"
    [ -f "$R" ] && cp "$R" "$BK/$k.registry"
done
restore() {
    for k in q35 q35mtp dsv4; do
        R="$(mpath $k).tensor_overrides.pshard_registry"
        if [ -f "$BK/$k.registry" ]; then cp "$BK/$k.registry" "$R"; else rm -f "$R"; fi
    done
}
trap restore EXIT INT TERM

strip() { sed 's/\x1b\[[0-9;]*m//g'; }
hash_gen() { tr -d '\r\n' < "$1" | md5sum | cut -c1-12; }
tps_field() { # log pattern -> tokens per second (llama perf print)
    strip < "$1" | grep -a "$2" | tail -1 | grep -aoE '[0-9.]+ tokens per second' | grep -aoE '^[0-9.]+'
}
decode_runs() { strip < "$1" | grep -a " eval time" | tail -1 | grep -aoE "/ +[0-9]+ runs" | grep -aoE "[0-9]+"; }
spec_speed()  { strip < "$1" | grep -a "decoded .* tokens in" | tail -1 | grep -aoE 'speed: +[0-9.]+' | grep -aoE '[0-9.]+$'; }
spec_ntok()   { strip < "$1" | grep -a "decoded .* tokens in" | tail -1 | grep -aoE 'decoded +[0-9]+' | grep -aoE '[0-9]+'; }
spec_accept() { strip < "$1" | grep -a "accept  *=" | tail -1 | grep -aoE '[0-9.]+%' | tr -d '%'; }
fallback()    { strip < "$1" | grep -aqE 'pshard not active|disabling pshard|pshard DISABLED|invalid PSHARD_'; }
tier0() { # registry -> "strategy n_pinned miss_policy pool_slots" of the first tier-0 line
    L=$(grep -a -A1 '^\[tier 0 ' "$1" 2>/dev/null | grep -a '^strategy=' | head -1)
    echo "$(echo "$L" | grep -aoE 'strategy=[A-Z_]+' | cut -d= -f2) $(echo "$L" | grep -aoE 'n_pinned=[0-9]+' | cut -d= -f2) $(echo "$L" | grep -aoE 'miss_policy=[a-z_0-9]+' | cut -d= -f2) $(echo "$L" | grep -aoE 'pool_slots=[0-9]+' | cut -d= -f2)"
}
pool_counters() { # gate log -> "h misses_per_token"
    L=$(strip < "$1" | grep -a 'log_counters: expert pool:' | head -1)
    echo "$(echo "$L" | grep -aoE 'h=[0-9.]+' | cut -d= -f2) $(echo "$L" | grep -aoE 'misses/token=[0-9.]+' | cut -d= -f2)"
}

run_side() { # samp log gen cmd... -> "rc vram_delta"
    SAMP=$1; LOG=$2; GENF=$3; shift 3
    : > "$SAMP"
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$SAMP" 2>/dev/null; sleep 1; done ) &
    SPID=$!
    "$@" > "$GENF" 2> "$LOG"
    RC=$?
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    PEAK=$(sort -n "$SAMP" | tail -1); [ -z "$PEAK" ] && PEAK=$IDLE
    echo "$RC $((PEAK - IDLE))"
}

# ---------------------------------------------------------------- main loop
LAST_PLAN_KEY=""; LAST_TIER0=""
echo "$CELLS" | while IFS='|' read -r K M B PR S A P T N; do
    [ -z "$K" ] && continue
    case " $GRID_MODELS " in *" $(base_model $M) "*) ;; *) continue ;; esac
    NM=$(cell_name "$K" "$M" "$B" "$PR" "$S" "$A" "$P" "$T" "$N")
    if [ -n "$ONLY" ] && ! echo "$NM" | grep -qE "$ONLY"; then continue; fi
    if grep -q "^$NM," "$LEDGER"; then echo "=== $NM (in ledger, skipping)"; continue; fi
    T0=$(date +%s)
    MP=$(mpath $M); DP=$(dpath $M $S); REG="$MP.tensor_overrides.pshard_registry"
    MVA=$(mva_of $B); PROMPT=$(prompt_file $PR); CTX=$(prompt_ctx $PR)

    # tool + workload
    case $S in
        none)   TOOL=./llama-completion.exe;        SPECF="" ;;
        mtp)    TOOL=./llama-speculative-simple.exe; SPECF="--spec-type draft-mtp --spec-draft-n-max 2" ;;
        dspark) TOOL=./llama-speculative-simple.exe; SPECF="-md $DP --spec-type draft-dspark" ;;
        dflash) TOOL=./llama-speculative-simple.exe; SPECF="-md $DP --spec-type draft-dflash" ;;
    esac
    THR=""; [ "$T" != "0" ] && THR="-t $T"
    case $K in
        perf) WORK="-f $PROMPT -n $N_GEN -c $CTX --ignore-eos"; [ "$S" = "none" ] && WORK="$WORK -no-cnv" ;;
        gate) WORK="-f $PROMPT -n $N_GATE -c $CTX --temp 0 --ignore-eos -no-cnv --no-display-prompt -lv 4"; TOOL=./llama-completion.exe; SPECF="" ;;
        ppl)  WORK="-f $PPL_TEXT -c $CTX --chunks 8"; TOOL=./llama-perplexity.exe; SPECF="" ;;
    esac

    # environment for plan + run (identical: the registry fingerprint must match the run)
    ENVF=""   # fingerprinted part (also the plan-sharing key)
    case $A in
        stock)    ;;
        auto)     ;;
        s[0-4])   ENVF="PSHARD_STRATEGY=${A#s}" ;;
        pool:plan) ENVF="PSHARD_STRATEGY=5 PSHARD_POOL_RUNTIME=1" ;;
        pool:*)   ENVF="PSHARD_STRATEGY=5 PSHARD_MISS_POLICY=${A#pool:} PSHARD_POOL_RUNTIME=1" ;;
        poolauto) ENVF="PSHARD_POOL_AUTO=1 PSHARD_POOL_RUNTIME=1" ;;
    esac
    [ "$N" = "1" ] && ENVF="$ENVF GGML_SCHED_NO_CPU_OVERLAP=1"   # the planner prices the overlap: plan + run
    ENVV=$ENVF
    [ "$P" = "1" ] && ENVV="$ENVV PSHARD_POOL_PREDICT=1"        # runtime-only: not part of the plan key

    # budget flag: stock -fitb, pshard -pshard -mva. DSv4 + DSpark stock only fits at 3000
    # (the stock fit ignores the 10.4 GB draft and OOMs otherwise) - recorded as mva 3000.
    BUDGET=$MVA
    if [ "$A" = "stock" ]; then
        [ "$M" = "dsv4" ] && [ "$S" = "dspark" ] && BUDGET=3000
        BUDF="-fitb $BUDGET"
    else
        BUDF="-pshard -mva $MVA"
    fi

    echo "=== $NM"
    STRAT=""; NPIN=""; MPOL=""; SLOTS=""; STATUS=OK
    if [ "$A" != "stock" ]; then
        PLAN_KEY="$M|$MVA|$CTX|$SPECF|$ENVF|$THR"
        if [ "$PLAN_KEY" = "$LAST_PLAN_KEY" ] && [ -f "$REG" ]; then
            echo "    plan shared with the previous cell"
        else
            rm -f "$REG"
            PLOG="$OUT/plan_$NM.log"
            # shellcheck disable=SC2086
            env $ENVF ./llama-pshard-plan-params.exe -m "$MP" $SPECF -c $CTX -mva $MVA $THR > "$PLOG" 2>&1
            PRC=$?
            LAST_PLAN_KEY=""; LAST_TIER0=""
            if [ $PRC -ne 0 ]; then
                echo "$NM,$K,$M,$BUDGET,$PR,$CTX,$S,$A,$P,$T,$N,$PRC,,,,,,,,,,,,,,PLAN_FAILED" >> "$LEDGER"
                echo "    plan failed (rc=$PRC) in $(( $(date +%s) - T0 )) s"
                continue
            fi
            LAST_PLAN_KEY=$PLAN_KEY; LAST_TIER0=$(tier0 "$REG")
        fi
        set -- $LAST_TIER0; STRAT=${1:-}; NPIN=${2:-}; MPOL=${3:-}; SLOTS=${4:-}
        if [ -z "$STRAT" ]; then
            echo "$NM,$K,$M,$BUDGET,$PR,$CTX,$S,$A,$P,$T,$N,0,,,,,,,,,,,,,,PLAN_FAILED" >> "$LEDGER"
            echo "    plan wrote no tier-0 line in $(( $(date +%s) - T0 )) s"
            continue
        fi
    fi

    LOG="$OUT/$NM.log"; GENF="$OUT/$NM.gen"
    # shellcheck disable=SC2086
    R=$(run_side "$OUT/vram_$NM.txt" "$LOG" "$GENF" env $ENVV $TOOL -m "$MP" $SPECF $WORK $BUDF $THR)
    RC=${R%% *}; DELTA=${R##* }
    PT=""; DT=""; NTOK=""; ACC=""; MD5=""; H=""; MPT=""; PPL=""
    case $K in
        perf)
            PT=$(tps_field "$LOG" "prompt eval time")
            if [ "$S" = "none" ]; then
                DT=$(tps_field "$LOG" " eval time"); NTOK=$(decode_runs "$LOG")
            else
                DT=$(spec_speed "$LOG"); NTOK=$(spec_ntok "$LOG"); ACC=$(spec_accept "$LOG")
            fi
            if [ -n "$NTOK" ] && [ "$NTOK" -lt "$MIN_DECODE" ] 2>/dev/null; then STATUS=SHORT; fi
            ;;
        gate)
            MD5=$(hash_gen "$GENF")
            case $A in pool:*) set -- $(pool_counters "$LOG"); H=${1:-}; MPT=${2:-} ;; esac
            ;;
        ppl)
            PPL=$(strip < "$LOG" | grep -aoE 'Final estimate: PPL = [0-9.]+' | grep -aoE '[0-9.]+$')
            ;;
    esac
    [ "$RC" != "0" ] && STATUS=FAIL
    [ "$A" != "stock" ] && fallback "$LOG" && STATUS=FALLBACK
    echo "$NM,$K,$M,$BUDGET,$PR,$CTX,$S,$A,$P,$T,$N,$RC,$PT,$DT,$NTOK,$ACC,$DELTA,$STRAT,$NPIN,$MPOL,$SLOTS,$MD5,$H,$MPT,$PPL,$STATUS" >> "$LEDGER"
    echo "    $STATUS rc=$RC prompt=$PT decode=$DT n=$NTOK acc=$ACC md5=$MD5 h=$H ppl=$PPL tier0=$STRAT/$NPIN/$MPOL/$SLOTS in $(( $(date +%s) - T0 )) s"
done
echo "done: $LEDGER"
