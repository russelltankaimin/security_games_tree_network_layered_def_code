"""Experiment 1 -- exactness/optimality validation.
On small random networks, verify the index policy attains the exact MDP optimum
(gap ~ 0) and that the algorithm's game value matches the MDP."""
import csv, random, argparse
import networks as nw, sp_index, mdp, policies

def run(seed=0, sizes=range(2, 7), trials=15, out="results/exp1_validation.csv"):
    rng = random.Random(seed); rows = []
    for n in sizes:
        for t in range(trials):
            G = nw.random_network(rng, n, qmax=2, monotone=True, p_par=0.5)
            opt, _ = mdp.optimal_value(G)
            pol = mdp.policy_value(G, policies.optimal_index(G))
            gv = sp_index.game_value(G)
            rows.append(dict(n=n, Q=nw.total_Q(G), depth=nw.depth(G), mdp_opt=opt,
                             index_polval=pol, index_gameval=gv,
                             gap_polval=opt - pol, gap_gameval=abs(opt - gv)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    mg = max(r["gap_polval"] for r in rows); mv = max(r["gap_gameval"] for r in rows)
    print(f"exp1: {len(rows)} instances, max policy gap = {mg:.2e}, max game-value gap = {mv:.2e} -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); run(seed=a.seed, trials=a.trials)
