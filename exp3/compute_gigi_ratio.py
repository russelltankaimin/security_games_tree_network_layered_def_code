"""
compute_gigi_ratio.py
=====================

Build a PER-INSTANCE CSV of GI/GI and the ratio (GR/GR) / (GI/GI).

    GI/GI : the min-max value -- the OPTIMAL defender's allocation faced by the
            OPTIMAL (Gittins) attacker. This is exactly exp2's `v_opt` (the exact
            optimum reached by regret matching); it is read from
            exp2/grad_<tag>_summary.csv (method 1 = exact).
    GR/GR : reward the GREEDY-trained allocation l*_GR concedes to the greedy
            attacker it was trained against. Read from
            exp3/grgr_res/grgr_<tag>.csv (attacker_value).

For each network we emit one row with GR/GR, GI/GI and their ratio

    (GR/GR) / (GI/GI)

i.e. how the greedy defender's *believed* reward compares to the genuine
optimal-vs-optimal game value.

The b100 batch has no GR/GR counterpart in exp3 (grgr_res only holds b1/b10), so
b100 tags are skipped -- driving --all from grgr_res already excludes them.

Usage
-----
    python3 compute_gigi_ratio.py b10r5 --out gigi_b10r5.csv
    python3 compute_gigi_ratio.py --all --out gigi_all.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _grgr_name(tag):
    """exp3 grgr_res filename for a tag: 'b10r5' -> 'grgr_b10r5.csv',
    'rw_b10r5' -> 'rw_grgr_b10r5.csv'."""
    if tag.startswith("rw_"):
        return f"rw_grgr_{tag[len('rw_'):]}.csv"
    return f"grgr_{tag}.csv"


def _gigi_name(tag):
    """exp2 summary filename for a tag: 'b10r5' -> 'grad_b10r5_summary.csv',
    'rw_b10r5' -> 'grad_rw_b10r5_summary.csv'."""
    if tag.startswith("rw_"):
        return f"grad_rw_{tag[len('rw_'):]}_summary.csv"
    return f"grad_{tag}_summary.csv"


def _load_grgr(path, column):
    """{network_id: full row} from an exp3 grgr file (must have `column`)."""
    if not os.path.exists(path):
        raise SystemExit(f"GR/GR file not found: {path}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        raise SystemExit(f"{path}: missing '{column}' column -- is this a gdga output?")
    return {r["network_id"]: r for r in rows}


def _load_gigi(path):
    """{network_id: v_opt(float)} from an exp2 summary (exact method only)."""
    if not os.path.exists(path):
        raise SystemExit(f"GI/GI (exp2) file not found: {path}")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "v_opt" not in rows[0]:
        raise SystemExit(f"{path}: no 'v_opt' column -- is this an exp2 summary?")
    out = {}
    for r in rows:
        if r.get("method") in (None, "", "1"):     # exact optimum (identical across methods)
            out[r["network_id"]] = float(r["v_opt"])
    return out


CSV_FIELDS = ["tag", "network_id", "n_controls", "total_Q", "rho", "samples",
              "GR_GR", "GI_GI", "GR_GR_over_GI_GI"]


def instance_rows(tag, grgr_dir, gigi_dir, column):
    grgr_path = os.path.join(grgr_dir, _grgr_name(tag))
    gigi_path = os.path.join(gigi_dir, _gigi_name(tag))
    grgr = _load_grgr(grgr_path, column)
    gigi = _load_gigi(gigi_path)

    common = sorted(set(grgr) & set(gigi), key=lambda s: int(s))
    only_gr, only_gi = set(grgr) - set(gigi), set(gigi) - set(grgr)
    if only_gr or only_gi:
        print(f"[warn] {tag}: id sets differ (only grgr={sorted(only_gr)}, "
              f"only gigi={sorted(only_gi)}); using {len(common)} common ids.")
    if not common:
        raise SystemExit(f"{tag}: no common network_id between grgr and exp2 files")

    rows = []
    for nid in common:
        gr_row = grgr[nid]
        grgr_val = float(gr_row[column])            # GR/GR
        gigi_val = gigi[nid]                        # GI/GI
        ratio = grgr_val / gigi_val if gigi_val else float("nan")
        rows.append({
            "tag": tag,
            "network_id": nid,
            "n_controls": gr_row.get("n_controls", ""),
            "total_Q": gr_row.get("total_Q", ""),
            "rho": gr_row.get("rho", ""),
            "samples": gr_row.get("samples", ""),
            "GR_GR": grgr_val,
            "GI_GI": gigi_val,
            "GR_GR_over_GI_GI": ratio,
        })
    return rows, grgr_path, gigi_path


def run(args):
    tags = list(args.files)
    if args.all:
        found = sorted(glob.glob(os.path.join(args.grgr_dir, "*grgr*.csv")))
        if not found:
            raise SystemExit(f"--all: no '*grgr*.csv' in {args.grgr_dir}")
        for p in found:
            base = os.path.basename(p)[:-4].replace("grgr", "")
            while "__" in base:
                base = base.replace("__", "_")
            tags.append(base.strip("_"))
    if not tags:
        raise SystemExit("give one or more tags, or use --all")

    all_rows = []
    for tag in tags:
        if "b100" in tag:
            print(f"[skip] {tag}: b100 has no GR/GR counterpart in grgr_res.")
            continue
        rows, grgr_path, gigi_path = instance_rows(
            tag, args.grgr_dir, args.gigi_dir, args.column)
        all_rows.extend(rows)
        mean_gr = st.mean(r["GR_GR"] for r in rows)
        mean_gi = st.mean(r["GI_GI"] for r in rows)
        agg = mean_gr / mean_gi if mean_gi else float("nan")
        print(f"[{tag}] {len(rows)} instances | GR/GR<-{os.path.basename(grgr_path)} "
              f"GI/GI<-{os.path.basename(gigi_path)}")
        print(f"    mean GR/GR={mean_gr:.6f}  mean GI/GI={mean_gi:.6f}  "
              f"(mean GR/GR)/(mean GI/GI)={agg:.4f}")

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in all_rows:
                row = dict(r)
                row["GR_GR"] = f"{r['GR_GR']:.8f}"
                row["GI_GI"] = f"{r['GI_GI']:.8f}"
                row["GR_GR_over_GI_GI"] = f"{r['GR_GR_over_GI_GI']:.8f}"
                writer.writerow(row)
        print(f"\nwrote {args.out}  ({len(all_rows)} instance rows)")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="tags, e.g. b10r5 rw_b1r10")
    p.add_argument("--all", action="store_true",
                   help="process every tag found in --grgr-dir (b100 excluded)")
    p.add_argument("--out", default=None, help="write per-instance results to this CSV")
    p.add_argument("--grgr-dir", default=os.path.join(_HERE, "grgr_res"),
                   help="directory holding the GR/GR files (default: exp3/grgr_res)")
    p.add_argument("--gigi-dir", default=os.path.join(_ROOT, "exp2"),
                   help="directory holding the exp2 summaries with v_opt (default: exp2)")
    p.add_argument("--column", default="attacker_value",
                   help="GR/GR reward column in the grgr file (default: attacker_value)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
