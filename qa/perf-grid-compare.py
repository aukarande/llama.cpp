#!/usr/bin/env python3
"""Compare two perf-grid ledgers cell by cell and (optionally) merge them.

    python qa/perf-grid-compare.py OLD_DIR NEW_DIR [--md out.md] [--merge MERGED_DIR]

OLD_DIR/NEW_DIR hold ledger.csv (qa/perf-grid.sh). The report lists, per cell present in NEW:
status old -> new, prompt t/s and decode t/s deltas (speculative cells also as target steps/s =
decode / (1 + accept x n_draft), the tables script's rate), md5/h/ppl changes. Cells only in OLD are
listed as "kept" (not rerun). --merge writes MERGED_DIR/ledger.csv = NEW rows plus the OLD rows for
cells NEW did not run (and copies the OLD cells' logs), so qa/perf-grid-tables.py renders one
complete grid; the header comment of NEW's ledger is kept and a second comment line records the merge.
The noise bands used for the verdict column: 4% plain decode, 10% speculative, 6% prompt (design
11.C.19 / qa/perf-grid.md).
"""
import csv
import os
import shutil
import sys

N_DRAFT = {"mtp": 2, "dspark": 3, "none": 0}
BAND = {"plain_decode": 0.04, "spec_decode": 0.10, "prompt": 0.06}


def read_ledger(path):
    rows, header, comments = [], None, []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
                continue
            if header is None:
                header = line.rstrip("\n").split(",")
                continue
            vals = line.rstrip("\n").split(",")
            if len(vals) < len(header):
                vals += [""] * (len(header) - len(vals))
            rows.append(dict(zip(header, vals[:len(header)])))
    return header, comments, rows


def fnum(s):
    try:
        return float(s)
    except Exception:
        return None


def steps_per_s(r):
    d = fnum(r.get("decode_tps"))
    a = fnum(r.get("accept_pct"))
    if d is None:
        return None
    n = N_DRAFT.get(r.get("spec", "none"), 0)
    if n == 0 or a is None:
        return d
    return d / (1.0 + a / 100.0 * n)


def pct(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def fmt_pct(p):
    return "-" if p is None else "%+.1f%%" % p


def fmt(v, nd=1):
    return "-" if v is None else ("%%.%df" % nd) % v


def verdict(o, n):
    so, sn = o.get("status", ""), n.get("status", "")
    if so != sn:
        return "status %s -> %s" % (so or "?", sn or "?")
    if sn != "OK":
        return "both %s" % sn
    kind = n.get("kind", "")
    if kind == "gate":
        return "hash same" if o.get("md5") == n.get("md5") else "HASH CHANGED"
    if kind == "ppl":
        po, pn = fnum(o.get("ppl")), fnum(n.get("ppl"))
        if po is None or pn is None:
            return "-"
        return "ppl %+.4f" % (pn - po)
    spec = n.get("spec", "none") != "none"
    band = BAND["spec_decode"] if spec else BAND["plain_decode"]
    dp = pct(steps_per_s(n), steps_per_s(o))
    pp = pct(fnum(n.get("prompt_tps")), fnum(o.get("prompt_tps")))
    tags = []
    if dp is not None and abs(dp) > band * 100:
        tags.append("decode %s" % fmt_pct(dp))
    if pp is not None and abs(pp) > BAND["prompt"] * 100:
        tags.append("prompt %s" % fmt_pct(pp))
    if n.get("strategy_active") != o.get("strategy_active") or n.get("pool_slots") != o.get("pool_slots"):
        tags.append("plan %s/%s -> %s/%s" % (o.get("strategy_active"), o.get("pool_slots"),
                                             n.get("strategy_active"), n.get("pool_slots")))
    return "; ".join(tags) if tags else "within noise"


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    old_dir, new_dir = argv[1], argv[2]
    md_path, merge_dir = None, None
    i = 3
    while i < len(argv):
        if argv[i] == "--md":
            md_path = argv[i + 1]; i += 2
        elif argv[i] == "--merge":
            merge_dir = argv[i + 1]; i += 2
        else:
            print("unknown arg", argv[i]); return 2
    header_o, comments_o, rows_o = read_ledger(os.path.join(old_dir, "ledger.csv"))
    header_n, comments_n, rows_n = read_ledger(os.path.join(new_dir, "ledger.csv"))
    by_o = {r["cell"]: r for r in rows_o}
    by_n = {r["cell"]: r for r in rows_n}

    out = []
    def w(s=""):
        out.append(s)

    w("# perf grid comparison: %s -> %s" % (old_dir, new_dir))
    w()
    # every launch header (a ledger gets one per launch / resume; a header may cover no rows)
    for c in comments_o:
        w("old: `%s`" % c)
    for c in comments_n:
        w("new: `%s`" % c)
    w()
    w("%d cells in old, %d in new, %d rerun, %d kept from old, %d new-only" % (
        len(rows_o), len(rows_n), len(set(by_o) & set(by_n)), len(set(by_o) - set(by_n)), len(set(by_n) - set(by_o))))
    w()

    # status transitions
    trans = {}
    for c, n in by_n.items():
        o = by_o.get(c)
        key = ((o or {}).get("status", "(new)"), n.get("status", ""))
        trans[key] = trans.get(key, 0) + 1
    w("## status transitions (old -> new: cells)")
    w()
    w("| old | new | cells |")
    w("|---|---|---|")
    for (a, b), k in sorted(trans.items(), key=lambda kv: -kv[1]):
        w("| %s | %s | %d |" % (a, b, k))
    w()

    # perf cells
    w("## perf cells (rerun)")
    w()
    w("| cell | status old -> new | prompt t/s old -> new | decode old -> new (target steps/s for spec) | plan old -> new (strategy/n_pinned/slots) | h old -> new | verdict |")
    w("|---|---|---|---|---|---|---|")
    for c in sorted(by_n):
        n = by_n[c]
        if n.get("kind") != "perf":
            continue
        o = by_o.get(c, {})
        so, sn = steps_per_s(o) if o else None, steps_per_s(n)
        spec = n.get("spec", "none") != "none"
        dec = "%s -> %s" % (fmt(fnum(o.get("decode_tps"))), fmt(fnum(n.get("decode_tps"))))
        if spec:
            dec += " [%s -> %s]" % (fmt(so), fmt(sn))
        plan_o = "%s/%s/%s" % (o.get("strategy_active", "-"), o.get("n_pinned", "-"), o.get("pool_slots", "-"))
        plan_n = "%s/%s/%s" % (n.get("strategy_active", "-"), n.get("n_pinned", "-"), n.get("pool_slots", "-"))
        w("| %s | %s -> %s | %s -> %s (%s) | %s (%s) | %s -> %s | %s -> %s | %s |" % (
            c, o.get("status", "(new)"), n.get("status", ""),
            fmt(fnum(o.get("prompt_tps"))), fmt(fnum(n.get("prompt_tps"))), fmt_pct(pct(fnum(n.get("prompt_tps")), fnum(o.get("prompt_tps")))),
            dec, fmt_pct(pct(sn, so)),
            plan_o, plan_n, o.get("h", "-") or "-", n.get("h", "-") or "-",
            verdict(o, n) if o else "new cell"))
    w()

    # gates and ppl
    w("## gate cells (rerun): token hash")
    w()
    w("| cell | status old -> new | md5 old -> new | same |")
    w("|---|---|---|---|")
    for c in sorted(by_n):
        n = by_n[c]
        if n.get("kind") != "gate":
            continue
        o = by_o.get(c, {})
        w("| %s | %s -> %s | %s -> %s | %s |" % (c, o.get("status", "(new)"), n.get("status", ""),
                                                 o.get("md5", "-") or "-", n.get("md5", "-") or "-",
                                                 "yes" if o.get("md5") == n.get("md5") else "NO"))
    w()
    w("## ppl cells (rerun)")
    w()
    w("| cell | status old -> new | ppl old -> new | delta |")
    w("|---|---|---|---|")
    for c in sorted(by_n):
        n = by_n[c]
        if n.get("kind") != "ppl":
            continue
        o = by_o.get(c, {})
        po, pn = fnum(o.get("ppl")), fnum(n.get("ppl"))
        w("| %s | %s -> %s | %s -> %s | %s |" % (c, o.get("status", "(new)"), n.get("status", ""),
                                                 fmt(po, 4), fmt(pn, 4), "-" if po is None or pn is None else "%+.4f" % (pn - po)))
    w()
    kept = sorted(set(by_o) - set(by_n))
    w("## kept from old (not rerun): %d cells" % len(kept))
    w()
    w(", ".join(kept))
    w()

    text = "\n".join(out) + "\n"
    if md_path:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote", md_path)
    else:
        sys.stdout.write(text)

    if merge_dir:
        os.makedirs(merge_dir, exist_ok=True)
        with open(os.path.join(merge_dir, "ledger.csv"), "w", encoding="utf-8", newline="") as f:
            for c in comments_n:
                f.write(c + "\n")
            f.write("# merged: %d rerun rows from %s + %d kept rows from %s (cells the rerun did not cover)\n" % (
                len(rows_n), new_dir, len(kept), old_dir))
            f.write(",".join(header_n) + "\n")
            for r in rows_n:
                f.write(",".join(r.get(h, "") for h in header_n) + "\n")
            for c in kept:
                r = by_o[c]
                f.write(",".join(r.get(h, "") for h in header_n) + "\n")
        # per-cell files the tables script reads next to the ledger: <cell>.log / .gen (degenerate
        # scan) and plan_<cell>.log (predicted-vs-measured)
        n_copied = 0
        for name in os.listdir(new_dir):
            if name.endswith((".log", ".gen", ".txt")):
                shutil.copy2(os.path.join(new_dir, name), os.path.join(merge_dir, name))
                n_copied += 1
        for c in kept:
            for name in (c + ".log", c + ".gen", c + ".txt", "plan_" + c + ".log"):
                src = os.path.join(old_dir, name)
                dst = os.path.join(merge_dir, name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    n_copied += 1
        print("merged ledger: %d rows -> %s (%d files copied)" % (len(rows_n) + len(kept), merge_dir, n_copied))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
