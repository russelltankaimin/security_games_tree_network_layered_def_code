"""Data visualisation -- reads results/*.csv, writes publication-quality PDF
figures to figures/. AAAI-style: serif fonts (match LaTeX), vector PDF, no junk.
Run after the experiments, or via run_all.py."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (4.4, 3.1), "font.size": 9, "font.family": "serif",
    "mathtext.fontset": "cm", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "legend.frameon": False,
})
C = {"opt": "#1b4965", "a": "#1b4965", "b": "#bc4749", "c": "#e09f3e", "d": "#6a994e", "e": "#8e7dbe"}

def load(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        return list(csv.DictReader(f))

def _agg(rows, key, val, fn=np.median):
    out = {}
    for r in rows:
        out.setdefault(float(r[key]), []).append(float(r[val]))
    xs = sorted(out)
    return np.array(xs), np.array([fn(out[x]) for x in xs])

def fig1_validation(path="results/exp1_validation.csv", out="figures/fig1_validation.pdf"):
    rows = load(path)
    if not rows: return
    opt = np.array([float(r["mdp_opt"]) for r in rows])
    idx = np.array([float(r["index_gameval"]) for r in rows])
    gap = np.array([float(r["gap_polval"]) for r in rows])
    fig, ax = plt.subplots()
    ax.scatter(opt, idx, s=12, color=C["a"], alpha=0.6, edgecolor="none")
    lim = [min(opt.min(), idx.min()), 1.0]
    ax.plot(lim, lim, "--", color="0.4", lw=0.8)
    ax.set_xlabel("brute-force MDP optimum"); ax.set_ylabel("index algorithm value")
    ax.set_title("Exactness: index value vs MDP optimum")
    ax.text(0.05, 0.92, f"max policy gap $={gap.max():.1e}$\n$N={len(rows)}$ instances",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.savefig(out); plt.close(fig); print("wrote", out)

def fig2_scalability(path="results/exp2_scalability.csv", out="figures/fig2_scalability.pdf"):
    rows = load(path)
    if not rows: return
    fig, ax = plt.subplots()
    xi, yi = _agg(rows, "n", "t_index_ms")
    ax.semilogy(xi, yi, "-o", ms=4, color=C["a"], label="index  $O(Q^2)$")
    mrows = [r for r in rows if r["t_mdp_ms"] not in ("", "nan") and r["t_mdp_ms"] == r["t_mdp_ms"]]
    mrows = [r for r in rows if r["t_mdp_ms"] not in ("",) and float(r["t_mdp_ms"]) == float(r["t_mdp_ms"])]
    if mrows:
        xm, ym = _agg(mrows, "n", "t_mdp_ms")
        ax.semilogy(xm, ym, "-s", ms=4, color=C["b"], label="brute-force MDP")
        ax.axvline(xm.max() + 0.5, ls=":", color="0.6", lw=0.8)
        ax.text(xm.max() + 0.6, ax.get_ylim()[1], " MDP\n infeasible", fontsize=7, va="top", color="0.4")
    ax.set_xlabel("number of controls $n$"); ax.set_ylabel("median wall-clock (ms)")
    ax.set_title("Scalability: exact index vs product MDP"); ax.legend(loc="center right")
    fig.savefig(out); plt.close(fig); print("wrote", out)

def fig3_baselines(path="results/exp3_baselines.csv", out="figures/fig3_baselines.pdf"):
    rows = load(path)
    if not rows: return
    order = ["optimal_index", "pc_index", "myopic_raw", "greedy_success", "single_route", "round_robin"]
    present = [p for p in order if any(r["policy"] == p for r in rows)]
    data = [[100 * float(r["rel_loss"]) for r in rows if r["policy"] == p] for p in present]
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    cols = [C["opt"], C["b"], C["c"], C["d"], C["e"], "0.5"]
    for patch, col in zip(bp["boxes"], cols): patch.set_facecolor(col); patch.set_alpha(0.55)
    for med in bp["medians"]: med.set_color("black")
    ax.set_xticklabels([p.replace("_", "\n") for p in present], fontsize=7.5)
    ax.set_ylabel("relative value loss (\\%)")
    ax.set_title("Value loss vs optimal index policy")
    fig.savefig(out); plt.close(fig); print("wrote", out)

def fig4_nesting(path="results/exp4_nesting.csv", out="figures/fig4_nesting.pdf"):
    rows = load(path)
    if not rows: return
    fig, ax = plt.subplots()
    style = {"pc_index": ("-o", C["b"], "parallel-chains (AAAI-26)"),
             "myopic_raw": ("-s", C["c"], "myopic (raw index)"),
             "single_route": ("-^", "0.5", "single route")}
    for pol, (mk, col, lab) in style.items():
        sub = [r for r in rows if r["policy"] == pol]
        if not sub: continue
        x, y = _agg(sub, "levels", "rel_loss", fn=np.mean)
        ax.plot(x, 100 * y, mk, ms=4, color=col, label=lab)
    ax.set_xlabel("nesting depth (Par-inside-Ser levels)")
    ax.set_ylabel("mean relative value loss (\\%)")
    ax.set_title("Generalisation beyond parallel chains"); ax.legend(loc="upper left")
    fig.savefig(out); plt.close(fig); print("wrote", out)

def fig5_sensitivity(path="results/exp5_sensitivity.csv", out="figures/fig5_sensitivity.pdf"):
    rows = load(path)
    if not rows: return
    betas = sorted({float(r["beta"]) for r in rows}); qs = sorted({int(float(r["qmax"])) for r in rows})
    M = np.zeros((len(qs), len(betas)))
    for r in rows:
        i = qs.index(int(float(r["qmax"]))); j = betas.index(float(r["beta"]))
        M[i, j] = 100 * float(r["loss_pc_index"])
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(betas))); ax.set_xticklabels(betas)
    ax.set_yticks(range(len(qs))); ax.set_yticklabels(qs)
    ax.set_xlabel(r"discount $\beta$"); ax.set_ylabel(r"max jam threshold $q$")
    ax.set_title("pc_index loss (\\%) across regimes")
    for i in range(len(qs)):
        for j in range(len(betas)):
            ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out); plt.close(fig); print("wrote", out)

def all_figs():
    fig1_validation(); fig2_scalability(); fig3_baselines(); fig4_nesting(); fig5_sensitivity()

if __name__ == "__main__":
    all_figs()
