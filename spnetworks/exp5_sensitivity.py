"""Experiment 5 -- sensitivity.
Sweeps over discount beta and jam threshold q on small random networks, reporting
the mean parallel-chains and myopic gap, to locate regimes where downstream
coupling matters most. Optimal reference = exact index value; baselines by MC."""
import csv, random, argparse
import networks as nw, sp_index, mdp, policies

def run(seed=4, betas=(0.8, 0.9, 0.95), qmaxes=(1, 2), n=6, trials=12, samples=3000,
        out="results/exp5_sensitivity.csv"):
    rng = random.Random(seed); mcrng = random.Random(seed + 100); rows = []
    for beta in betas:
        for qm in qmaxes:
            losses = {p: [] for p in ['pc_index', 'myopic_raw']}; opts = []
            for t in range(trials):
                G = nw.random_network(rng, n, beta=beta, qmax=qm, monotone=True, p_par=0.5)
                opt = sp_index.game_value(G)
                if opt < 1e-9: continue
                opts.append(opt)
                for p in losses:
                    v = mdp.mc_policy_value(G, policies.ALL[p](G), n_samples=samples, rng=mcrng)
                    losses[p].append(max(0.0, (opt - v) / opt))
            row = dict(beta=beta, qmax=qm, mean_opt=sum(opts) / len(opts))
            for p in losses: row[f"loss_{p}"] = sum(losses[p]) / len(losses[p])
            rows.append(row)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"exp5: {len(rows)} rows -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=12); ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args(); run(seed=a.seed, trials=a.trials)
