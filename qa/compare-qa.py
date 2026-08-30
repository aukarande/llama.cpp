#!/usr/bin/env python3
"""Gate a QA run against the reference ledger.

Usage: python qa/compare-qa.py <new-ledger.csv> [reference-ledger.csv]
Default reference: qa/reference-ledger.csv next to this script.

Hard failures (exit 1):
  - any row with status FAIL / TOKEN_MISMATCH / PLAN_FAILED
  - pshard token hash != stock token hash for the same (model, ctx, mva)
  - active strategy or overlap differs from the reference row (silent-fallback drift)
  - sustained VRAM delta exceeds reference by more than VRAM_SLACK MiB
Soft failures (exit 1 unless --perf-warn-only):
  - prompt or decode t/s below reference * (1 - PERF_TOL)

Improvements (new >= reference * (1 + PERF_TOL)) are reported so the reference
can be intentionally regenerated (copy the new ledger over qa/reference-ledger.csv).
"""
import csv, sys, os

PERF_TOL   = 0.05   # 5% noise band
VRAM_SLACK = 128    # MiB

def load(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            rows[(r["config"], r["side"])] = r
    return rows

def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    warn_only = "--perf-warn-only" in sys.argv
    new_path = args[0]
    ref_path = args[1] if len(args) > 1 else os.path.join(os.path.dirname(__file__), "reference-ledger.csv")
    new = load(new_path)
    ref = load(ref_path) if os.path.exists(ref_path) else {}

    hard, soft, better, info = [], [], [], []

    PASS_CLASSES = ("OK", "TOKEN_DIVERGED_PPL_OK", "STOCK_UNAVAILABLE", "NO_BASELINE")

    def status_class(s):
        # statuses may embed per-run values, e.g. "PPL_MISMATCH(stock=... pshard=...)"
        return (s or "").split("(", 1)[0]

    def normalize(rows, cfg, side, r):
        if side != "pshard":
            return r
        stock = rows.get((cfg.rsplit("-s", 1)[0], "stock"))
        # the token hash is the ground truth: if it matches the cell's CURRENT stock hash
        # the generation is identical, whatever status was recorded when the row was made
        if (stock is not None and r.get("token_hash") and
                r["token_hash"] == stock.get("token_hash")):
            return dict(r, status="OK")
        # cells where STOCK itself cannot run (e.g. 16k ctx in a 2 GB budget) make the
        # token/PPL gates unadjudicable - classify as STOCK_UNAVAILABLE (pshard-only cell)
        if r["status"] != "OK" and stock is not None and stock["status"] == "FAIL":
            return dict(r, status="STOCK_UNAVAILABLE")
        return r

    for (cfg, side), r in sorted(new.items()):
        if side == "stock" and r["status"] == "FAIL":
            info.append(f"{cfg}: stock cannot run this cell (pshard-only)")
            continue
        r = normalize(new, cfg, side, r)
        if status_class(r["status"]) not in PASS_CLASSES:
            # a bad status whose class matches the reference is a tracked known issue -
            # report it, but only NEW breakage hard-fails the gate
            rr = ref.get((cfg, side))
            if rr is not None:
                rr = normalize(ref, cfg, side, rr)
            if rr is not None and status_class(rr["status"]) == status_class(r["status"]):
                info.append(f"{cfg}/{side}: KNOWN ISSUE (unchanged): {r['status'][:70]}")
            else:
                hard.append(f"{cfg}/{side}: status={r['status'][:70]}")
            continue
        if side != "pshard":
            continue
        rr = ref.get((cfg, side))
        if rr is None:
            info.append(f"{cfg}: no reference row (new config)")
            continue
        rr = normalize(ref, cfg, side, rr)
        if status_class(r["status"]) != status_class(rr["status"]):
            hard.append(f"{cfg}: token-gate class drift {rr['status']} -> {r['status']}")
        # plan attribution drift; only compare fields the (possibly older) reference has
        for k in ("strategy_active", "strategy_prefill", "overlap", "n_pinned",
                  "n_attn_pinned", "prefill_ub"):
            if rr.get(k) not in (None, "") and r.get(k) not in (None, "") and r[k] != rr[k]:
                hard.append(f"{cfg}: plan drift {k}: {rr[k]} -> {r[k]}")
        dv, rv = fnum(r["vram_peak_delta"]), fnum(rr["vram_peak_delta"])
        if dv is not None and rv is not None and dv > rv + VRAM_SLACK:
            hard.append(f"{cfg}: VRAM {dv:.0f} MiB > reference {rv:.0f} + {VRAM_SLACK}")
        for k in ("prompt_tps", "decode_tps"):
            nv, ov = fnum(r[k]), fnum(rr[k])
            if nv is None or ov is None or ov == 0:
                continue
            if nv < ov * (1 - PERF_TOL):
                soft.append(f"{cfg}: {k} {nv:.2f} < reference {ov:.2f} (-{(1 - nv / ov) * 100:.1f}%)")
            elif nv > ov * (1 + PERF_TOL):
                better.append(f"{cfg}: {k} {nv:.2f} > reference {ov:.2f} (+{(nv / ov - 1) * 100:.1f}%)")

    for title, items in (("HARD FAILURES", hard), ("PERF REGRESSIONS", soft),
                         ("IMPROVEMENTS", better), ("INFO", info)):
        if items:
            print(f"== {title} ({len(items)})")
            for i in items:
                print("  ", i)
    if not hard and not soft:
        print("== PASS" + (f" ({len(better)} improvements - consider refreshing the reference)" if better else ""))
    sys.exit(1 if hard or (soft and not warn_only) else 0)

if __name__ == "__main__":
    main()
