"""
collect_runtimes.py
===================

Collect per-network runtimes and iteration counts for the two gradient oracles
(method 1 = EXACT, method 2 = STOCHASTIC) out of exp2's *_summary.csv files into
a single tidy CSV.

For every network in every summary file we emit one row:

    source, network_id, n_controls, total_Q,
    runtime_exact_s, runtime_stoch_s, iters_exact, iters_stoch

We report the time/iterations the method needed to converge
(`runtime_to_converge_s` / `iters_to_converge`); if a run never converged, we
fall back to the full-run totals (`total_runtime_s` / `total_iters`). A method
that is absent from a file (e.g. an exact-only run) leaves its cells blank.

Usage
-----
    python3 collect_runtimes.py --out runtimes.csv            # every *_summary.csv here
    python3 collect_runtimes.py grad_b1r1_summary.csv --out one.csv
    python3 collect_runtimes.py --glob "grad_*_summary.csv" --out runtimes.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

METHOD_NAME = {"1": "exact", "2": "stochastic"}

OUT_FIELDS = ["source", "network_id", "n_controls", "total_Q",
              "runtime_exact_s", "runtime_stoch_s", "iters_exact", "iters_stoch"]


def _tag_of(path):
    """'grad_b1r1_summary.csv' -> 'b1r1', 'grad_rw_b10r5_summary.csv' -> 'rw_b10r5',
    otherwise the basename minus '_summary.csv'."""
    base = os.path.basename(path)
    if base.endswith("_summary.csv"):
        base = base[:-len("_summary.csv")]
    if base.startswith("grad_"):
        base = base[len("grad_"):]
    return base


def _runtime(row):
    """Seconds the method needed: time-to-converge, else the full-run total."""
    if row is None:
        return ""
    val = (row.get("runtime_to_converge_s") or "").strip()
    return val if val else (row.get("total_runtime_s") or "").strip()


def _iters(row):
    """Iterations the method needed: to-converge, else the full-run total."""
    if row is None:
        return ""
    val = (row.get("iters_to_converge") or "").strip()
    return val if val else (row.get("total_iters") or "").strip()


def rows_from_file(path):
    """One output row per network in a single summary file."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"[skip] {os.path.basename(path)}: empty")
        return []
    if "method" not in rows[0]:
        print(f"[skip] {os.path.basename(path)}: no 'method' column "
              f"(not an exp2 summary?)")
        return []

    # network_id -> {method_code: row}
    by_id = {}
    for r in rows:
        by_id.setdefault(r["network_id"], {})[r.get("method")] = r

    tag = _tag_of(path)
    out = []
    for nid in sorted(by_id, key=lambda s: int(s)):
        methods = by_id[nid]
        m1, m2 = methods.get("1"), methods.get("2")
        meta = m1 or m2                       # n_controls/total_Q are method-agnostic
        out.append({
            "source": tag,
            "network_id": nid,
            "n_controls": meta.get("n_controls", ""),
            "total_Q": meta.get("total_Q", ""),
            "runtime_exact_s": _runtime(m1),
            "runtime_stoch_s": _runtime(m2),
            "iters_exact": _iters(m1),
            "iters_stoch": _iters(m2),
        })
    return out


def run(args):
    if args.files:
        paths = [f if os.path.isabs(f) else os.path.join(args.dir, f) for f in args.files]
    else:
        paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    # never ingest our own output
    out_abs = os.path.abspath(args.out)
    paths = [p for p in paths if os.path.abspath(p) != out_abs]
    if not paths:
        raise SystemExit(f"no files matched (dir={args.dir}, glob={args.glob!r})")

    all_rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path}: not found")
            continue
        rows = rows_from_file(path)
        all_rows.extend(rows)
        print(f"[{_tag_of(path)}] {len(rows)} networks  ({os.path.basename(path)})")

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {args.out}  ({len(all_rows)} rows from {len(paths)} files)")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*",
                   help="specific summary files (default: all matching --glob in --dir)")
    p.add_argument("--dir", default=_HERE, help="directory to scan (default: exp2/)")
    p.add_argument("--glob", default="*_summary.csv",
                   help="filename pattern when no files are given (default: *_summary.csv)")
    p.add_argument("--out", default=os.path.join(_HERE, "runtimes.csv"),
                   help="output CSV path (default: exp2/runtimes.csv)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
