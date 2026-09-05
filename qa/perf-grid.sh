#!/bin/sh
# pshard perf grid: stock vs legacy (planner's auto pick) vs expert pool, q35 and DSv4.
# Design and cell list: qa/perf-grid.md. This is NOT the held 144-cell QA grid (run-qa.sh).
#
# Usage:
#   sh qa/perf-grid.sh <out-dir> [--list]
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
#     the pool's headline counters print at WARN, so perf rows still carry h;
#   - thread count is always the default (never set -t);
#   - every pshard cell plans fresh with the run's own environment (PSHARD_STRATEGY,
#     PSHARD_MISS_POLICY, GGML_SCHED_NO_CPU_OVERLAP and the spec flags are fingerprinted);
#     consecutive cells that differ only in runtime-only knobs (PSHARD_POOL_PREDICT,
#     PSHARD_POOL_WARM/ALLOC) share one plan;
#   - benches get the whole machine; one instance at a time.

set -u
OUT=${1:?usage: perf-grid.sh <out-dir> [--list]}
LIST=0
for a in "$@"; do [ "$a" = "--list" ] && LIST=1; done
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
        dsv4/dspark) echo "$MODELS/DeepseekV4-Flash-20260731-DSpark.gguf" ;;
        *) echo "" ;;
    esac
}
# 4k = prompt-4k-v2.txt (2026-09-05): repository documentation, no internal repetition. The old
# prompt-4k.txt repeats one 50-sentence article 12x: models copied it, the 4k gates produced
# identical text across MODELS, and a warm start that seeded the copied passage looked like +50%.
prompt_file() { case $1 in 512) echo "$PROMPTS/prompt-512.txt" ;; 4k) echo "$PROMPTS/prompt-4k-v2.txt" ;; esac; }
prompt_ctx()  { case $1 in 512) echo 2048 ;; 4k) echo 8192 ;; esac; }
mva_of()      { case $1 in full) echo "$FULL" ;; *) echo "$1" ;; esac; }

# ---------------------------------------------------------------- cell list
# one line per cell: kind|model|mva|prompt|spec|arm|predict|warm|noovl
#   kind   perf | gate | ppl
#   mva    MiB or "full"
#   prompt 512 | 4k
#   arm    stock | auto | s0..s4 (forced legacy) | pool:<policy> | pool:plan | poolauto
#   warm   1 = PSHARD_POOL_WARM=8 PSHARD_POOL_ALLOC=1 (prompt-end LRU seeding + per-layer slots)
#   noovl  1 = GGML_SCHED_NO_CPU_OVERLAP=1 (plan and run)
CELLS=""
add() { CELLS="$CELLS
$1|$2|$3|$4|$5|$6|$7|$8|$9"; }
pool_block() { # model mva prompt  -> the 13 pool perf variants (pred/warm variants right after their base: one plan)
    M=$1; B=$2; PR=$3
    add perf $M $B $PR none pool:fetch     0 0 0
    add perf $M $B $PR none pool:fetch     1 0 0
    add perf $M $B $PR none pool:fetch     1 1 0
    add perf $M $B $PR none pool:hybrid    0 0 0
    add perf $M $B $PR none pool:hybrid    1 0 0
    add perf $M $B $PR none pool:hybrid    0 0 1
    add perf $M $B $PR none pool:cpu_admit 0 0 0
    add perf $M $B $PR none pool:cpu_admit 1 0 0
    add perf $M $B $PR none pool:cpu_admit 0 0 1
    add perf $M $B $PR none pool:cpu_exec  0 0 0
    add perf $M $B $PR none pool:fetch_on_2nd_miss 0 0 0
    add perf $M $B $PR none pool:plan      0 0 0
    add perf $M $B $PR none poolauto       0 0 0
}
plain_set() { # model mva prompt stock_ok pool_ok
    M=$1; B=$2; PR=$3; ST=$4; POOL=$5
    [ "$ST" = "1" ] && add perf $M $B $PR none stock 0 0 0
    add perf $M $B $PR none auto 0 0 0
    for st in 0 1 2 3 4; do add perf $M $B $PR none s$st 0 0 0; done   # forced legacy strategies (user, 2026-09-04)
    [ "$POOL" = "1" ] && pool_block $M $B $PR
    # gates: 32-token md5 vs stock (stock always runs as the reference, even where its
    # perf cell is not in the table) + pool counters; every policy at 512, fetch at 4k
    add gate $M $B $PR none stock 0 0 0
    add gate $M $B $PR none auto  0 0 0
    if [ "$PR" = "512" ]; then for st in 0 1 2 3 4; do add gate $M $B $PR none s$st 0 0 0; done; fi
    if [ "$POOL" = "1" ]; then
        if [ "$PR" = "512" ]; then
            add gate $M $B $PR none pool:fetch 0 0 0
            add gate $M $B $PR none pool:fetch 1 0 0
            add gate $M $B $PR none pool:fetch 1 1 0
            for p in hybrid cpu_admit cpu_exec fetch_on_2nd_miss; do add gate $M $B $PR none pool:$p 0 0 0; done
        else
            add gate $M $B $PR none pool:fetch 0 0 0
        fi
    fi
    # perplexity mirror (prompt-independent: once per model/budget, on the 512 pass)
    if [ "$PR" = "512" ]; then
        add ppl $M $B $PR none stock 0 0 0
        add ppl $M $B $PR none auto  0 0 0
        [ "$POOL" = "1" ] && add ppl $M $B $PR none pool:fetch 0 0 0
    fi
}
spec_set() { # model mva spec prompt pool_ok  -> stock + auto (+ 5 pool variants)
    M=$1; B=$2; SPC=$3; PR=$4; POOL=$5
    add perf $M $B $PR $SPC stock 0 0 0
    add perf $M $B $PR $SPC auto  0 0 0
    for st in 0 1 2 3 4; do add perf $M $B $PR $SPC s$st 0 0 0; done
    if [ "$POOL" = "1" ]; then
        add perf $M $B $PR $SPC pool:fetch  0 0 0
        add perf $M $B $PR $SPC pool:fetch  1 0 0
        add perf $M $B $PR $SPC pool:hybrid 0 0 0
        add perf $M $B $PR $SPC pool:plan   0 0 0
        add perf $M $B $PR $SPC poolauto    0 0 0
    fi
}
# the user's table (2026-09-04): q35 stock/legacy/pool at 4000, 8000, full; dsv4 legacy at
# 8000 and full, pool at full (its ~9.2 GB of fixed pins do not fit below); q35mtp and
# dsv4+DSpark stock/legacy/pool; prompts 512 and 4k everywhere
for PR in 512 4k; do
    plain_set q35  4000 $PR 1 1
    plain_set q35  8000 $PR 1 1
    plain_set q35  full $PR 1 1
    plain_set dsv4 8000 $PR 0 0
    plain_set dsv4 full $PR 0 1
    spec_set q35mtp 4000 mtp    $PR 1
    spec_set q35mtp 8000 mtp    $PR 1
    spec_set q35mtp full mtp    $PR 1
    spec_set dsv4   8000 dspark $PR 0
    spec_set dsv4   full dspark $PR 1
done

cell_name() { # kind model mva prompt spec arm predict warm noovl  (no subshells: fork is slow on Windows)
    case $6 in pool:*) a="pool_${6#pool:}" ;; *) a=$6 ;; esac
    n="$2-$3-$4-$5-$a"
    [ "$7" = "1" ] && n="$n-pred"
    [ "$8" = "1" ] && n="$n-warm"
    [ "$9" = "1" ] && n="$n-noovl"
    [ "$1" != "perf" ] && n="$n-$1"
    echo "$n"
}

if [ "$LIST" = "1" ]; then
    echo "$CELLS" | while IFS='|' read -r K M B PR S A P W N; do
        [ -z "$K" ] && continue
        case $M in q35mtp) BM=q35 ;; *) BM=$M ;; esac
        case " $GRID_MODELS " in *" $BM "*) ;; *) continue ;; esac
        cell_name "$K" "$M" "$B" "$PR" "$S" "$A" "$P" "$W" "$N"
    done | { if [ -n "$ONLY" ]; then grep -E "$ONLY"; else cat; fi; } | awk '{n++; print} END {print "# " n " cells"}'
    exit 0
fi

# ---------------------------------------------------------------- run support
mkdir -p "$OUT"
cd "$BIN" || exit 1
LEDGER="$OUT/ledger.csv"
HDR="cell,kind,model,mva,prompt,ctx,spec,arm,predict,warm,noovl,rc,prompt_tps,decode_tps,decode_tokens,accept_pct,vram_peak_delta_mib,strategy_active,n_pinned,miss_policy,pool_slots,md5,h,misses_per_token,ppl,status"
if [ "${QA_RESUME:-0}" = "1" ] && [ -f "$LEDGER" ]; then
    echo "resuming: $(grep -vc -e '^#' -e '^cell,' "$LEDGER") rows already present"
else
    echo "$HDR" > "$LEDGER"
fi
# ---- preconditions (user rules): locked clocks, and the whole machine ------------------------
# clocks: C:/Aditya/gb203_lock_clocks.bat = five perfdebug.exe lines (must run from C:/Aditya):
# GPC 2505 MHz, DRAM 14 GHz, fixed-frequency regime. Re-run here (idempotent; locks do not
# survive a driver reset). QA_FORCE=1 skips both checks.
if [ -x /c/Aditya/perfdebug.exe ]; then
    ( cd /c/Aditya && for a in "--lock_strict set dramclkkHz 14000000" "--lock_strict set gpcclkkHz 2505000"         "--lock_loose set sysclkkHz 2230000" "--lock_loose set xbarclkkHz 2230000" "--force_regime ffr"; do
        MSYS_NO_PATHCONV=1 ./perfdebug.exe $a > /dev/null 2>&1; done )
    sleep 2
fi
CLK=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
SM=${CLK%%,*}; MEM=${CLK##*,}
if [ "${QA_FORCE:-0}" != "1" ]; then
    case $SM in 249[0-9]|250[0-9]) ;; *) echo "ABORT: GPU clocks not locked (sm=$SM MHz, mem=$MEM MHz); run C:/Aditya/gb203_lock_clocks.bat from C:/Aditya, or QA_FORCE=1"; exit 2 ;; esac
    # GPU idle: utilization over 3 s and foreign GPU contexts. The Windows shell and the Claude
    # desktop app (claude.exe + its msedgewebview2 renderer - this session's host) are
    # whitelisted: idle graphics contexts. Browsers, the Codex/ChatGPT app and anything else
    # are not - 2026-09-04 a background ChatGPT.exe (compute + graphics context) plus a
    # browser cost 15-30% on both prefill and decode.
    UTIL=$(for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null; sleep 1; done | sort -n | tail -1 | tr -d ' ')
    FOREIGN=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -viE 'dwm.exe|explorer.exe|LogonUI|ShellExperienceHost|StartMenuExperienceHost|TextInputHost|TabTip|WUDFHost|SearchHost|ShellHost|CrossDeviceResume|claude.exe|msedgewebview2|nvidia' | tr '\134' '/' | sed 's#.*/##' | sort -u | tr '\n' ' ')
    if [ "${UTIL:-0}" -gt 3 ] 2>/dev/null || [ -n "$FOREIGN" ]; then
        echo "ABORT: GPU is not idle (utilization ${UTIL}%, foreign GPU contexts: ${FOREIGN:-none}); close them or QA_FORCE=1"; exit 2
    fi
fi
IDLE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ -z "$IDLE" ] && IDLE=0
echo "# idle_used=$IDLE full_mva=$FULL clocks_sm=$SM clocks_mem=$MEM git=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) date=$(date +%Y-%m-%dT%H:%M)" >> "$LEDGER"

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
# strip ANSI colour codes before hashing: with stderr on a character device (NUL or a tty)
# Windows' isatty says yes and llama-completion colours the prompt on STDOUT (\e[33m ... \e[0m)
hash_gen() { sed 's/\x1b\[[0-9;]*m//g' "$1" | tr -d '\r\n' | md5sum | cut -c1-12; }
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
pool_counters() { # log -> "h misses_per_token" from the pool's headline line (WARN level: present in perf logs too)
    L=$(strip < "$1" | grep -a 'log_counters: expert pool: ' | head -1)
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
echo "$CELLS" | while IFS='|' read -r K M B PR S A P W N; do
    [ -z "$K" ] && continue
    case $M in q35mtp) BM=q35 ;; *) BM=$M ;; esac
    case " $GRID_MODELS " in *" $BM "*) ;; *) continue ;; esac
    NM=$(cell_name "$K" "$M" "$B" "$PR" "$S" "$A" "$P" "$W" "$N")
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
    esac
    case $K in
        perf) WORK="-f $PROMPT -n $N_GEN -c $CTX --ignore-eos"; [ "$S" = "none" ] && WORK="$WORK -no-cnv"
              # llama-speculative-simple refuses a prompt longer than its logical batch (2048): stock
              # cells with the 4k prompt get -b = ctx (pshard sizes its own batch from the plan)
              [ "$S" != "none" ] && [ "$A" = "stock" ] && [ "$PR" = "4k" ] && WORK="$WORK -b $CTX" ;;
        gate) WORK="-f $PROMPT -n $N_GATE -c $CTX --temp 0 --ignore-eos -no-cnv --no-display-prompt -lv 4"; TOOL=./llama-completion.exe; SPECF="" ;;
        ppl)  WORK="-f $PPL_TEXT -c $CTX --chunks 8"; TOOL=./llama-perplexity.exe; SPECF="" ;;
    esac

    # environment: fingerprinted part (plan + run, also the plan-sharing key) and runtime-only knobs
    ENVF=""
    case $A in
        stock)    ;;
        auto)     ;;
        s[0-4])   ENVF="PSHARD_STRATEGY=${A#s}" ;;   # forced legacy strategy (fingerprinted)
        pool:plan) ENVF="PSHARD_STRATEGY=5 PSHARD_POOL_RUNTIME=1" ;;
        pool:*)   ENVF="PSHARD_STRATEGY=5 PSHARD_MISS_POLICY=${A#pool:} PSHARD_POOL_RUNTIME=1" ;;
        poolauto) ENVF="PSHARD_POOL_AUTO=1 PSHARD_POOL_RUNTIME=1" ;;
    esac
    [ "$N" = "1" ] && ENVF="$ENVF GGML_SCHED_NO_CPU_OVERLAP=1"   # the planner prices the overlap
    ENVV=$ENVF
    [ "$P" = "1" ] && ENVV="$ENVV PSHARD_POOL_PREDICT=1"
    [ "$W" = "1" ] && ENVV="$ENVV PSHARD_POOL_WARM=8 PSHARD_POOL_ALLOC=1"

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
        PLAN_KEY="$M|$MVA|$CTX|$SPECF|$ENVF"
        if [ "$PLAN_KEY" = "$LAST_PLAN_KEY" ] && [ -f "$REG" ]; then
            echo "    plan shared with the previous cell"
        else
            rm -f "$REG"
            PLOG="$OUT/plan_$NM.log"
            # shellcheck disable=SC2086
            env $ENVF ./llama-pshard-plan-params.exe -m "$MP" $SPECF -c $CTX -mva $MVA > "$PLOG" 2>&1
            PRC=$?
            LAST_PLAN_KEY=""; LAST_TIER0=""
            if [ $PRC -ne 0 ]; then
                echo "$NM,$K,$M,$BUDGET,$PR,$CTX,$S,$A,$P,$W,$N,$PRC,,,,,,,,,,,,,,PLAN_FAILED" >> "$LEDGER"
                echo "    plan failed (rc=$PRC) in $(( $(date +%s) - T0 )) s"
                continue
            fi
            LAST_PLAN_KEY=$PLAN_KEY; LAST_TIER0=$(tier0 "$REG")
        fi
        set -- $LAST_TIER0; STRAT=${1:-}; NPIN=${2:-}; MPOL=${3:-}; SLOTS=${4:-}
        if [ -z "$STRAT" ]; then
            echo "$NM,$K,$M,$BUDGET,$PR,$CTX,$S,$A,$P,$W,$N,0,,,,,,,,,,,,,,PLAN_FAILED" >> "$LEDGER"
            echo "    plan wrote no tier-0 line in $(( $(date +%s) - T0 )) s"
            continue
        fi
    fi

    LOG="$OUT/$NM.log"; GENF="$OUT/$NM.gen"
    # shellcheck disable=SC2086
    R=$(run_side "$OUT/vram_$NM.txt" "$LOG" "$GENF" env $ENVV $TOOL -m "$MP" $SPECF $WORK $BUDF)
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
            ;;
        ppl)
            PPL=$(strip < "$LOG" | grep -aoE 'Final estimate: PPL = [0-9.]+' | grep -aoE '[0-9.]+$')
            ;;
    esac
    case $A in pool:*|poolauto) set -- $(pool_counters "$LOG"); H=${1:-}; MPT=${2:-} ;; esac
    [ "$RC" != "0" ] && STATUS=FAIL
    [ "$A" != "stock" ] && fallback "$LOG" && STATUS=FALLBACK
    echo "$NM,$K,$M,$BUDGET,$PR,$CTX,$S,$A,$P,$W,$N,$RC,$PT,$DT,$NTOK,$ACC,$DELTA,$STRAT,$NPIN,$MPOL,$SLOTS,$MD5,$H,$MPT,$PPL,$STATUS" >> "$LEDGER"
    echo "    $STATUS rc=$RC prompt=$PT decode=$DT n=$NTOK acc=$ACC md5=$MD5 h=$H ppl=$PPL tier0=$STRAT/$NPIN/$MPOL/$SLOTS in $(( $(date +%s) - T0 )) s"
done
echo "done: $LEDGER"
