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
PPL_TEXT="$PROMPTS/ppl-docs-v2.txt"   # 2026-09-05: repo docs corpus (571 KB, 173k q35 tokens, 5% repeated shingles); prompt-256k.txt was one article repeated
FULL=${QA_FULL_MVA:-14500}
GPC=${QA_GPC_MHZ:-2100}   # locked GPC clock, MHz (2026-09-05: lowered from 2505, suspected of tripping the PSU; 2505-clock ledgers are NOT comparable)
N_GEN=128          # perf decode window (user rule: always 128)
N_GATE=32          # correctness gate window (md5 + counters)
SEED=${QA_SEED:-1234}  # perf cells sample with a FIXED seed (2026-09-05): the pool
                   # decode speed follows the generated text (degenerate digit runs route to a
                   # few experts: h 0.75 vs 0.46 for prose on the same cell), so arms must
                   # generate the same text to be comparable. Gates are greedy (--temp 0).
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
prompt_file() { case $1 in 512) echo "$PROMPTS/prompt-512-v2.txt" ;; 4k) echo "$PROMPTS/prompt-4k-v2.txt" ;; esac; }
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
    plain_set dsv4 8000 $PR 1 0   # stock perf cells added 2026-09-06 (user: the DSv4 table needs a stock column)
    plain_set dsv4 full $PR 1 1
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
    done | { if [ -n "$ONLY" ]; then grep -E -e "$ONLY"; else cat; fi; } | awk '{n++; print} END {print "# " n " cells"}'
    exit 0
fi

# ---------------------------------------------------------------- run support
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)   # absolute: every artifact path is built from it after the cd below
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
# GPC $QA_GPC_MHZ (default 2100) MHz, DRAM 14 GHz, fixed-frequency regime. Re-run here (idempotent; locks do not
# survive a driver reset). QA_FORCE=1 skips both checks.
if [ -x /c/Aditya/perfdebug.exe ]; then
    ( cd /c/Aditya && for a in "--lock_strict set dramclkkHz 14000000" "--lock_strict set gpcclkkHz ${GPC}000"         "--lock_loose set sysclkkHz 2230000" "--lock_loose set xbarclkkHz 2230000" "--force_regime ffr"; do
        MSYS_NO_PATHCONV=1 ./perfdebug.exe $a > /dev/null 2>&1; done )
    sleep 2
fi
CLK=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
SM=${CLK%%,*}; MEM=${CLK##*,}
if [ "${QA_FORCE:-0}" != "1" ]; then
    # numeric guard first: a non-numeric nvidia-smi field ("[N/A]") made both arithmetic tests
    # false and let the run through unlocked (audit 2026-09-05); the DRAM lock (14 GHz) is
    # checked as well, it was never tested before
    case "$SM$MEM" in ''|*[!0-9]*) echo "ABORT: cannot read GPU clocks (sm='$SM' mem='$MEM')"; exit 2 ;; esac
    if [ "$SM" -lt $((GPC - 30)) ] || [ "$SM" -gt $((GPC + 10)) ] || [ "$MEM" -lt 13900 ]; then echo "ABORT: GPU clocks not locked at $GPC MHz / 14 GHz (sm=$SM MHz, mem=$MEM MHz); lock with perfdebug from C:/Aditya, or QA_FORCE=1"; exit 2; fi
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

# registries: back up ONCE per out-dir (a QA_RESUME after a reboot mid-run must not overwrite
# the good backup with the forced registry the crashed run left behind - audit 2026-09-05),
# restore at exit; an absent registry is recorded so restore() removes it rather than keeping
# a forced leftover
BK="$OUT/registry-backup"; mkdir -p "$BK"
if [ ! -f "$BK/.taken" ]; then
    for k in q35 q35mtp dsv4; do
        R="$(mpath $k).tensor_overrides.pshard_registry"
        if [ -f "$R" ]; then cp "$R" "$BK/$k.registry"; else : > "$BK/$k.absent"; fi
    done
    : > "$BK/.taken"
fi
restore() {
    for k in q35 q35mtp dsv4; do
        R="$(mpath $k).tensor_overrides.pshard_registry"
        if [ -f "$BK/$k.registry" ]; then cp "$BK/$k.registry" "$R"; elif [ -f "$BK/$k.absent" ]; then rm -f "$R"; fi
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
# llama-speculative-simple books every verification batch as "prompt eval": its prompt figure is
# the target's aggregate, 33-56% below its own prefill line (audit 2026-09-05)
spec_prefill() { strip < "$1" | grep -a "encoded .* tokens in" | tail -1 | grep -aoE 'speed: +[0-9.]+' | grep -aoE '[0-9.]+$'; }
# the spec context's real device footprint vs what the target left it ("(+X MiB)" = over)
reserve_over() { strip < "$1" | grep -a 'pshard one-budget check' | tail -1 | grep -aoE '\(\+[0-9.]+ MiB\)' | grep -aoE '[0-9]+' | head -1; }
# a perf cell whose generation collapsed (one token repeated, digit runs, a 3-gram loop) routes to
# the same few experts every step: h inflates and the pool row measures nothing (the 2026-09-04
# DSv4 "+77%" and the 2026-09-05 q35 warm-start row were such text). Tail = the .gen minus the
# echoed prompt; flagged when its whitespace tokens are too few or too repetitive.
degenerate() { # gen prompt_file -> 0 (degenerate) / 1
    PLEN=$(wc -c < "$2" | tr -d ' ')
    strip < "$1" | tr -d '\r' | tail -c +"$((PLEN + 1))" | awk '
        { for (i = 1; i <= NF; i++) { n++; t[n] = $i; c[$i]++ } }
        END {
            if (n == 0) { print 0; exit }                       # no whitespace tokens at all
            if (n < 12) { print (length(t[1]) > 40 ? 0 : 1); exit }
            for (i = 1; i <= n - 2; i++) g[t[i] " " t[i+1] " " t[i+2]]++
            u = 0; for (k in g) u++
            top = 0; for (k in c) if (c[k] > top) top = c[k]
            print (u / (n - 2) < 0.35 || top / n > 0.30) ? 0 : 1
        }'
}
# ~llama_context prints each backend's sched buffer against its post-warmup size; a larger CUDA0
# size at exit means the scheduler grew the target's buffer during the run = an arena overflow
sched_grew() { # log -> 0 (grew) / 1
    strip < "$1" | grep -a 'CUDA0 compute buffer size of' | grep -a 'does not match' | tail -1 |
        awk '{ for (i = 1; i <= NF; i++) { if ($i == "of" && $(i+2) == "MiB,") sz = $(i+1); if ($i == "expectation") ex = $(i+2) } }
             END { print (sz + 0 > ex + 1) ? 0 : 1 }'
}
expected_strategy() { # arm -> the tier-0 strategy name the arm must produce ("" = any)
    case $1 in s0) echo GPUONLY_LAYERPIN_LAYERSTREAM ;; s1) echo GPUONLY_ATTNPIN_FFNSTREAM ;; s2) echo DYNAMIC_FFNCPU_ATTNSTREAM ;;
               s3) echo STATIC_ATTNPRIO_ALLMODELS ;; s4) echo DYNAMIC_FFN_ALTERNATE ;; pool:*) echo EXPERT_POOL ;; *) echo "" ;; esac
}
fallback()    { strip < "$1" | grep -aqE 'pshard not active|disabling pshard|pshard DISABLED|invalid PSHARD_'; }
tier0() { # registry [tier] -> "strategy n_pinned miss_policy pool_slots" of that tier's first line
    # (tier 0 = bs=1 decode; speculative cells execute tier 1 = the n_draft+1 verify batch, which
    # the planner can substitute with the attn-pin legacy fallback while tier 0 stays a pool tier)
    L=$(grep -a -A1 "^\[tier ${2:-0} " "$1" 2>/dev/null | grep -a '^strategy=' | head -1)
    echo "$(echo "$L" | grep -aoE 'strategy=[A-Z_]+' | cut -d= -f2) $(echo "$L" | grep -aoE '(^| )n_pinned=[0-9]+' | tr -d ' ' | cut -d= -f2) $(echo "$L" | grep -aoE 'miss_policy=[a-z_0-9]+' | cut -d= -f2) $(echo "$L" | grep -aoE '(^| )s=[0-9]+' | tr -d ' ' | cut -d= -f2)"
}
pool_counters() { # log -> "h misses_per_token": the decode-only headline when present (a pool
    # prefill/warmup tier adds cold passes to the all-passes line), else the all-passes line
    L=$(strip < "$1" | grep -aE 'log_counters: expert pool( decode)?: ' | tail -1)
    echo "$(echo "$L" | grep -aoE 'h=[0-9.]+' | cut -d= -f2) $(echo "$L" | grep -aoE 'misses/token=[0-9.]+' | cut -d= -f2)"
}

run_side() { # samp log gen cmd... -> "rc vram_delta clock_ok"; samples memory.used and clocks.sm
    # once a second: a TDR / driver reset mid-run drops the lock silently (audit 2026-09-05)
    SAMP=$1; LOG=$2; GENF=$3; shift 3
    : > "$SAMP"
    ( while :; do nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv,noheader,nounits >> "$SAMP" 2>/dev/null; sleep 1; done ) &
    SPID=$!
    "$@" > "$GENF" 2> "$LOG"
    RC=$?
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    PEAK=$(cut -d, -f1 "$SAMP" | tr -d ' ' | grep -E '^[0-9]+$' | sort -n | tail -1); [ -z "$PEAK" ] && PEAK=$IDLE
    # the SM clock reads below the lock while the GPU idles between the model load and the
    # first kernel; judge only samples where the card was busy (memory above the idle level)
    CLKBAD=$(awk -F, -v idle="$IDLE" -v lo=$((GPC - 30)) -v hi=$((GPC + 10)) '{gsub(/ /,""); if ($1+0 > idle+1024 && ($2+0 < lo || $2+0 > hi)) n++} END {print n+0}' "$SAMP")
    echo "$RC $((PEAK - IDLE)) $([ "${CLKBAD:-0}" -gt 2 ] && echo 0 || echo 1)"
}

# ---------------------------------------------------------------- main loop
LAST_PLAN_KEY=""; LAST_TIER0=""
echo "$CELLS" | while IFS='|' read -r K M B PR S A P W N; do
    [ -z "$K" ] && continue
    case $M in q35mtp) BM=q35 ;; *) BM=$M ;; esac
    case " $GRID_MODELS " in *" $BM "*) ;; *) continue ;; esac
    NM=$(cell_name "$K" "$M" "$B" "$PR" "$S" "$A" "$P" "$W" "$N")
    if [ -n "$ONLY" ] && ! echo "$NM" | grep -qE -e "$ONLY"; then continue; fi
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
        perf) WORK="-f $PROMPT -n $N_GEN -c $CTX --ignore-eos -s $SEED"; [ "$S" = "none" ] && WORK="$WORK -no-cnv"
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
            LAST_PLAN_KEY=$PLAN_KEY; LAST_TIER0=$(tier0 "$REG" $([ "$S" = "none" ] && echo 0 || echo 1))
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
    set -- $R; RC=$1; DELTA=$2; CLKOK=${3:-1}
    PT=""; DT=""; NTOK=""; ACC=""; MD5=""; H=""; MPT=""; PPL=""
    case $K in
        perf)
            if [ "$S" = "none" ]; then
                PT=$(tps_field "$LOG" "prompt eval time")
                DT=$(tps_field "$LOG" " eval time"); NTOK=$(decode_runs "$LOG")
            else
                PT=$(spec_prefill "$LOG")
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
    if [ "$STATUS" = OK ] && [ "$A" != "stock" ]; then
        # the planner substitutes the attn-pin legacy plan for a forced strategy or a pool
        # policy that does not fit the tier (three s1 cells and seven speculative pool cells
        # of the 2026-09-04 grid were recorded as what they were not); poolauto may pick legacy
        EXP=$(expected_strategy "$A")
        if [ -n "$EXP" ] && [ -n "$STRAT" ] && [ "$STRAT" != "$EXP" ]; then STATUS=STRATEGY_FALLBACK
        # a pool row without counters is a run the pool never served (decode at legacy speed)
        elif [ "$STRAT" = EXPERT_POOL ] && [ "$K" != ppl ] && [ -z "$H" ]; then STATUS=NOPOOL
        # the arena is the budget; anything the run allocated far beyond it is an overflow (the
        # 2026-09-05 bug class: +1.2 / +4.2 GB in the 09-04 ledger). Outside the arena by design:
        # the CUDA context + workspaces (~250 MiB, both sides, the user's decision) and, on pool
        # arms at 4k, ~250 MiB more of CUDA temporaries on the redirect backends during the A/B
        # prefill (measured +492 total on dsv4-full-4k pool_fetch; not priced by the planner)
        elif [ -n "$DELTA" ] && [ "$DELTA" -gt $((MVA + 1024)) ] 2>/dev/null; then STATUS=OVER_BUDGET
        # the spec context outgrew what the target left it (MTP head moved to the CPU, ...)
        elif [ "$S" != none ] && [ "$(reserve_over "$LOG")" -gt 64 ] 2>/dev/null; then STATUS=OVER_RESERVE
        elif [ "$CLKOK" = 0 ]; then STATUS=CLOCK
        fi
    fi
    if [ "$STATUS" = OK ] && [ "$K" = perf ] && [ "$(degenerate "$GENF" "$PROMPT")" = 0 ]; then STATUS=DEGENERATE; fi
    if [ "$STATUS" = OK ] && [ "$A" != "stock" ] && [ "$(sched_grew "$LOG")" = 0 ]; then STATUS=SCHED_GREW; fi
    echo "$NM,$K,$M,$BUDGET,$PR,$CTX,$S,$A,$P,$W,$N,$RC,$PT,$DT,$NTOK,$ACC,$DELTA,$STRAT,$NPIN,$MPOL,$SLOTS,$MD5,$H,$MPT,$PPL,$STATUS" >> "$LEDGER"
    echo "    $STATUS rc=$RC prompt=$PT decode=$DT n=$NTOK acc=$ACC md5=$MD5 h=$H ppl=$PPL tier0=$STRAT/$NPIN/$MPOL/$SLOTS in $(( $(date +%s) - T0 )) s"
done
echo "done: $LEDGER"
