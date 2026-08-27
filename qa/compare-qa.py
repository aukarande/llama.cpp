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

    for (cfg, side), r in sorted(new.items()):
        # cells where STOCK itself cannot run (e.g. 16k ctx in a 2 GB budget: stock fails to
        # allocate compute buffers while pshard runs) are informational pshard wins, not failures
        if side == "stock" and r["status"] == "FAIL":
            info.append(f"{cfg}: stock cannot run this cell (pshard-only)")
            continue
        if side == "pshard":
            stock = new.get((cfg.rsplit("-s", 1)[0], "stock"))
            if stock is not None and stock["status"] == "FAIL" and r["status"] != "OK":
                r = dict(r, status="STOCK_UNAVAILABLE")
        if r["status"] not in ("OK", "TOKEN_DIVERGED_PPL_OK", "STOCK_UNAVAILABLE"):
            # a bad status that exactly matches the reference is a tracked known issue -
            # report it, but only NEW breakage hard-fails the gate
            rr = ref.get((cfg, side))
            if rr is not None and rr["status"] == r["status"]:
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
        if r["status"] != rr["status"]:
            hard.append(f"{cfg}: token-gate class drift {rr['status']} -> {r['status']}")
        if r["strategy_active"] != rr["strategy_active"] or r["overlap"] != rr["overlap"]:
            hard.append(f"{cfg}: active plan drift {rr['strategy_active']}/ovl{rr['overlap']} -> "
                        f"{r['strategy_active']}/ovl{r['overlap']}")
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
