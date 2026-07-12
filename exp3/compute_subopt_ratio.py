"""
compute_subopt_ratio.py
=======================

Build a PER-INSTANCE CSV of the paper's defender SUBOPTIMALITY ratio

        (GI/GI) / (GR/GI)   ( < 1 ).

    GI/GI : the min-max value -- the OPTIMAL defender's allocation faced by the
            OPTIMAL (Gittins) attacker. This is exp2's exact `v_opt`; read from
            exp2/grad_<tag>_summary.csv.
    GR/GI : the GREEDY-trained allocation l*_GR faced by the OPTIMAL attacker.
            Read from exp3/grgi_res/grgi_<tag>.csv (attacker_value).

Both are rewards conceded to the optimal attacker, so their ratio measures how
much better the optimal defender does than the greedy-trained one:

    (GI/GI)/(GR/GI) < 1  ==>  optimising against the true (optimal) attacker beats
    optimising against a greedy heuristic; the smaller the ratio, the worse the
    greedy assumption is.

b100 has no GR/GI counterpart in exp3 (grgi_res only holds b1/b10), so b100 tags
are skipped -- driving --all from grgi_res already excludes them.

Usage
-----
    python3 compute_subopt_ratio.py b10r5 --out subopt_b10r5.csv
    python3 compute_subopt_ratio.py --all --out subopt_all.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _grgi_name(tag):
    """exp3 grgi_res filename: 'b10r5' -> 'grgi_b10r5.csv',
    'rw_b10r5' -> 'rw_grgi_b10r5.csv'."""
    if tag.startswith("rw_"):
        return f"rw_grgi_{tag[len('rw_'):]}.csv"
    return f"grgi_{tag}.csv"


def _gigi_name(tag):
    """exp2 summary filename: 'b10r5' -> 'grad_b10r5_summary.csv',
    'rw_b10r5' -> 'grad_rw_b10r5_summary.csv'."""
    if tag.startswith("rw_"):
        return f"grad_rw_{tag[len('rw_'):]}_summary.csv"
    return f"grad_{tag}_summary.csv"


def _load_grgi(path, column):
    """{network_id: full row} from an exp3 grgi file (must have `column`)."""
    if not os.path.exists(path):
        raise SystemExit(f"GR/GI file not found: {path}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        raise SystemExit(f"{path}: missing '{column}' column -- is this a gdga output?")
    return {r["network_id"]: r for r in rows}


def _load_gigi(path):
    """{network_id: v_opt(float)} from an exp2 summary (exact method)."""
    if not os.path.exists(path):
        raise SystemExit(f"GI/GI (exp2) file not found: {path}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "v_opt" not in rows[0]:
        raise SystemExit(f"{path}: no 'v_opt' column -- is this an exp2 summary?")
    out = {}
    for r in rows:
        if r.get("method") in (None, "", "1"):     # exact optimum (same across methods)
            out[r["network_id"]] = float(r["v_opt"])
    return out


CSV_FIELDS = ["tag", "network_id", "n_controls", "total_Q", "rho", "samples",
              "GI_GI", "GR_GI", "GI_GI_over_GR_GI"]


def instance_rows(tag, grgi_dir, gigi_dir, column):
    grgi_path = os.path.join(grgi_dir, _grgi_name(tag))
    gigi_path = os.path.join(gigi_dir, _gigi_name(tag))
    grgi = _load_grgi(grgi_path, column)
    gigi = _load_gigi(gigi_path)

    common = sorted(set(grgi) & set(gigi), key=lambda s: int(s))
    only_gr, only_gi = set(grgi) - set(gigi), set(gigi) - set(grgi)
    if only_gr or only_gi:
        print(f"[warn] {tag}: id sets differ (only grgi={sorted(only_gr)}, "
              f"only gigi={sorted(only_gi)}); using {len(common)} common ids.")
    if not common:
        raise SystemExit(f"{tag}: no common network_id between grgi and exp2 files")

    rows = []
    for nid in common:
        gr_row = grgi[nid]
        grgi_val = float(gr_row[column])            # GR/GI
        gigi_val = gigi[nid]                        # GI/GI
        ratio = gigi_val / grgi_val if grgi_val else float("nan")
        rows.append({
            "tag": tag,
            "network_id": nid,
            "n_controls": gr_row.get("n_controls", ""),
            "total_Q": gr_row.get("total_Q", ""),
            "rho": gr_row.get("rho", ""),
            "samples": gr_row.get("samples", ""),
            "GI_GI": gigi_val,
            "GR_GI": grgi_val,
            "GI_GI_over_GR_GI": ratio,
        })
    return rows, grgi_path, gigi_path


def run(args):
    tags = list(args.files)
    if args.all:
        found = sorted(glob.glob(os.path.join(args.grgi_dir, "*grgi*.csv")))
        if not found:
            raise SystemExit(f"--all: no '*grgi*.csv' in {args.grgi_dir}")
        for p in found:
            base = os.path.basename(p)[:-4].replace("grgi", "")
            while "__" in base:
                base = base.replace("__", "_")
            tags.append(base.strip("_"))
    if not tags:
        raise SystemExit("give one or more tags, or use --all")

    all_rows = []
    for tag in tags:
        if "b100" in tag:
            print(f"[skip] {tag}: b100 has no GR/GI counterpart in grgi_res.")
            continue
        rows, grgi_path, gigi_path = instance_rows(
            tag, args.grgi_dir, args.gigi_dir, args.column)
        all_rows.extend(rows)
        mean_gi = st.mean(r["GI_GI"] for r in rows)
        mean_gr = st.mean(r["GR_GI"] for r in rows)
        agg = mean_gi / mean_gr if mean_gr else float("nan")
        print(f"[{tag}] {len(rows)} instances | GI/GI<-{os.path.basename(gigi_path)} "
              f"GR/GI<-{os.path.basename(grgi_path)}")
        print(f"    mean GI/GI={mean_gi:.6f}  mean GR/GI={mean_gr:.6f}  "
              f"(mean GI/GI)/(mean GR/GI)={agg:.4f}")

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in all_rows:
                row = dict(r)
                row["GI_GI"] = f"{r['GI_GI']:.8f}"
                row["GR_GI"] = f"{r['GR_GI']:.8f}"
                row["GI_GI_over_GR_GI"] = f"{r['GI_GI_over_GR_GI']:.8f}"
                writer.writerow(row)
        print(f"\nwrote {args.out}  ({len(all_rows)} instance rows)")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="tags, e.g. b10r5 rw_b1r10")
    p.add_argument("--all", action="store_true",
                   help="process every tag found in --grgi-dir (b100 excluded)")
    p.add_argument("--out", default=None, help="write per-instance results to this CSV")
    p.add_argument("--grgi-dir", default=os.path.join(_HERE, "grgi_res"),
                   help="directory holding the GR/GI files (default: exp3/grgi_res)")
    p.add_argument("--gigi-dir", default=os.path.join(_ROOT, "exp2"),
                   help="directory holding the exp2 summaries with v_opt (default: exp2)")
    p.add_argument("--column", default="attacker_value",
                   help="GR/GI reward column in the grgi file (default: attacker_value)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
