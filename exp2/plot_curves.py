"""
plot_curves.py — plot the exact vs average-stochastic objective curves produced
by gradient_rm.py (<out>_curves.csv).

For each selected network: V*(l_t) vs iteration t, the deterministic EXACT curve
against the mean STOCHASTIC curve with a +/- std band across the runs.

    python3 plot_curves.py synth_b1r1_curves.csv --nets 0,3,9,15,22,35 --out curves.png
    python3 plot_curves.py synth_b1r1_curves.csv --logy         # log y-axis
"""
import argparse, csv, math, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    data = defaultdict(list)   # nid -> list of (t, exact, smean, sstd)
    for r in csv.DictReader(open(path)):
        def g(k):
            v = (r.get(k) or "").strip()
            return float(v) if v else None
        data[r["network_id"]].append((int(r["t"]), g("exact_obj"),
                                      g("stoch_mean_obj"), g("stoch_std_obj")))
    for nid in data:
        data[nid].sort()
    return data


def run(args):
    data = load(args.csv)
    nets = args.nets.split(",") if args.nets else list(data.keys())[:6]
    nets = [n for n in nets if n in data]
    ncol = min(3, len(nets)); nrow = math.ceil(len(nets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    tag = os.path.basename(args.csv).replace("_curves.csv", "")

    for i, nid in enumerate(nets):
        ax = axes[i // ncol][i % ncol]
        rows = data[nid]
        te = [t for t, e, m, s in rows if e is not None]
        ye = [e for t, e, m, s in rows if e is not None]
        tm = [t for t, e, m, s in rows if m is not None]
        ym = [m for t, e, m, s in rows if m is not None]
        ys = [s for t, e, m, s in rows if m is not None]
        ax.plot(te, ye, color="C0", lw=2, label="exact")
        ax.plot(tm, ym, color="C1", lw=1.6, label="stochastic (mean)")
        lo = [m - s for m, s in zip(ym, ys)]; hi = [m + s for m, s in zip(ym, ys)]
        ax.fill_between(tm, lo, hi, color="C1", alpha=0.25, label="stoch ±1 std")
        if ye:
            ax.axhline(ye[-1], color="C0", ls=":", lw=1, alpha=0.7)   # target
        if args.logy:
            ax.set_yscale("log")
        if args.logx:
            ax.set_xscale("log")
        ax.set_title(f"network {nid}")
        ax.set_xlabel("iteration t"); ax.set_ylabel(r"$V^*(\ell_t)$")
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(nets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(f"Objective convergence — {tag}  (exact vs stochastic)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(nets)} networks)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="a <tag>_curves.csv from gradient_rm.py")
    p.add_argument("--nets", default="", help="comma-separated network ids (default: first 6)")
    p.add_argument("--out", default=None, help="output PNG (default: <tag>_curves.png)")
    p.add_argument("--logy", action="store_true", help="log-scale the y-axis")
    p.add_argument("--logx", action="store_true", help="log-scale the x-axis (iterations)")
    a = p.parse_args()
    if a.out is None:
        a.out = a.csv.replace("_curves.csv", "_curves.png")
    run(a)
