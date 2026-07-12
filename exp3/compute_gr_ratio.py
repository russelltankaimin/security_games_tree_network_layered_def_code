"""
compute_gr_ratio.py
===================

Build a PER-INSTANCE CSV of the paper's GR/GR and GR/GI rewards and their ratio
(GR/GR) / (GR/GI) from a pair of corresponding gdga output files.

    GR/GI : reward the GREEDY-trained allocation l*_GR concedes to the OPTIMAL
            (Gittins/index) attacker.  Read from grgi_res/<...>  (attacker_value).
    GR/GR : reward l*_GR concedes to the GREEDY attacker it was trained against.
            Read from grgr_res/<...>  (attacker_value).

Both files hold one row per network (same l*_GR, since the defender's training is
identical). We match rows by network_id and, FOR EACH NETWORK, emit one output
row with that instance's GR/GR, GR/GI and (GR/GR)/(GR/GI):

    (GR/GR)/(GR/GI) < 1  ==>  the defender overestimates its own defence -- it
    thinks it concedes GR/GR to a greedy attacker but actually concedes the
    larger GR/GI to an optimal one.

Usage
-----
    # one file -> a per-network CSV
    python3 compute_gr_ratio.py grgi_b10r5.csv --out gr_b10r5.csv

    # several tags (or --all) -> one CSV, rows tagged by their file
    python3 compute_gr_ratio.py b10r5 b1r10 rw_b10r1 --out gr_rows.csv
    python3 compute_gr_ratio.py --all --out gr_rows.csv

You may pass either the grgi_ or grgr_ filename (or a bare tag); the script
resolves BOTH corresponding files from --grgi-dir and --grgr-dir.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_names(spec):
    """From a user-supplied filename/tag, derive the grgi_ and grgr_ basenames.

    Handles 'grgi_<tag>.csv', 'grgr_<tag>.csv', 'rw_grgi_<tag>.csv', or a bare
    '<tag>' / '<tag>.csv' (optionally 'rw_<tag>')."""
    base = os.path.basename(spec)
    if not base.endswith(".csv"):
        base += ".csv"
    if "grgi" in base:
        return base, base.replace("grgi", "grgr")
    if "grgr" in base:
        return base.replace("grgr", "grgi"), base
    if base.startswith("rw_"):
        tag = base[len("rw_"):]
        return f"rw_grgi_{tag}", f"rw_grgr_{tag}"
    return f"grgi_{base}", f"grgr_{base}"


def _tag_of(grgi_name):
    """Short label: 'grgi_b10r5.csv' -> 'b10r5', 'rw_grgi_b1r10.csv' -> 'rw_b1r10'."""
    base = os.path.basename(grgi_name)
    if base.endswith(".csv"):
        base = base[:-4]
    tag = base.replace("grgi", "")
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag.strip("_")


def _rows_by_id(path, column):
    """Return {network_id: row_dict} for a gdga output CSV (must have `column`)."""
    if not os.path.exists(path):
        raise SystemExit(f"file not found: {path}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    if column not in rows[0]:
        raise SystemExit(
            f"{path}: no '{column}' column (columns: {sorted(rows[0])}) -- "
            f"is this a gdga output file?")
    return {r["network_id"]: r for r in rows}


# One output row per network instance.
CSV_FIELDS = ["tag", "network_id", "n_controls", "total_Q", "rho", "samples",
              "GR_GR", "GR_GI", "GR_GR_over_GR_GI"]


def instance_rows(spec, grgi_dir, grgr_dir, column):
    """Per-network GR/GR, GR/GI and ratio for one tag. Returns a list of dicts."""
    grgi_name, grgr_name = _resolve_names(spec)
    grgi_path = os.path.join(grgi_dir, grgi_name)
    grgr_path = os.path.join(grgr_dir, grgr_name)
    tag = _tag_of(grgi_name)

    grgi = _rows_by_id(grgi_path, column)
    grgr = _rows_by_id(grgr_path, column)

    common = sorted(set(grgi) & set(grgr), key=lambda s: int(s))
    only_i, only_r = set(grgi) - set(grgr), set(grgr) - set(grgi)
    if only_i or only_r:
        print(f"[warn] {tag}: id sets differ (only grgi={sorted(only_i)}, "
              f"only grgr={sorted(only_r)}); using {len(common)} common ids.")
    if not common:
        raise SystemExit(f"{tag}: no common network_id between the two files")

    out = []
    for nid in common:
        gi_row, gr_row = grgi[nid], grgr[nid]
        grgi_val = float(gi_row[column])            # GR/GI: index attacker on l*_GR
        grgr_val = float(gr_row[column])            # GR/GR: greedy attacker on l*_GR
        ratio = grgr_val / grgi_val if grgi_val else float("nan")
        out.append({
            "tag": tag,
            "network_id": nid,
            "n_controls": gi_row.get("n_controls", ""),
            "total_Q": gi_row.get("total_Q", ""),
            "rho": gi_row.get("rho", ""),
            "samples": gi_row.get("samples", ""),
            "GR_GR": grgr_val,
            "GR_GI": grgi_val,
            "GR_GR_over_GR_GI": ratio,
        })
    return out, grgi_path, grgr_path


def run(args):
    specs = list(args.files)
    if args.all:
        found = sorted(glob.glob(os.path.join(args.grgi_dir, "*grgi*.csv")))
        if not found:
            raise SystemExit(f"--all: no '*grgi*.csv' in {args.grgi_dir}")
        specs += [os.path.basename(p) for p in found]
    if not specs:
        raise SystemExit("give one or more files/tags, or use --all")

    all_rows = []
    for spec in specs:
        rows, grgi_path, grgr_path = instance_rows(
            spec, args.grgi_dir, args.grgr_dir, args.column)
        all_rows.extend(rows)
        tag = rows[0]["tag"]
        mean_gr = st.mean(r["GR_GR"] for r in rows)
        mean_gi = st.mean(r["GR_GI"] for r in rows)
        agg = mean_gr / mean_gi if mean_gi else float("nan")
        print(f"[{tag}] {len(rows)} instances | GR/GI<-{os.path.basename(grgi_path)} "
              f"GR/GR<-{os.path.basename(grgr_path)}")
        print(f"    mean GR/GR={mean_gr:.6f}  mean GR/GI={mean_gi:.6f}  "
              f"(mean GR/GR)/(mean GR/GI)={agg:.4f}")

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in all_rows:
                row = dict(r)
                row["GR_GR"] = f"{r['GR_GR']:.8f}"
                row["GR_GI"] = f"{r['GR_GI']:.8f}"
                row["GR_GR_over_GR_GI"] = f"{r['GR_GR_over_GR_GI']:.8f}"
                writer.writerow(row)
        print(f"\nwrote {args.out}  ({len(all_rows)} instance rows)")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*",
                   help="grgi_/grgr_ filenames or bare tags (e.g. grgi_b10r5.csv, b10r5)")
    p.add_argument("--all", action="store_true",
                   help="process every '*grgi*.csv' found in --grgi-dir")
    p.add_argument("--out", default=None,
                   help="write the per-instance results to this CSV")
    p.add_argument("--grgi-dir", default=os.path.join(_HERE, "grgi_res"),
                   help="directory holding the GR/GI files (default: exp3/grgi_res)")
    p.add_argument("--grgr-dir", default=os.path.join(_HERE, "grgr_res"),
                   help="directory holding the GR/GR files (default: exp3/grgr_res)")
    p.add_argument("--column", default="attacker_value",
                   help="reward column to read from each file (default: attacker_value)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
