"""Experiment 2 -- scalability (headline plot).
Wall-clock time of the O(Q^2) index algorithm vs the brute-force product MDP as
the network grows. The MDP is state-capped, so oversized instances are skipped
(its blow-up is the point); the index method stays flat."""
import csv, random, time, argparse
import networks as nw, sp_index, mdp
from mdp import CapExceeded

def run(seed=1, sizes=range(2, 13), trials=3, mdp_state_cap=60_000, out="results/exp2_scalability.csv"):
    rng = random.Random(seed); rows = []
    for n in sizes:
        for t in range(trials):
            G = nw.random_network(rng, n, qmax=2, monotone=True, p_par=0.5)
            Q = nw.total_Q(G)
            t0 = time.perf_counter(); sp_index.index_table(G); sp_index.game_value(G)
            t_idx = (time.perf_counter() - t0) * 1e3
            t_mdp, states = float("nan"), float("nan")
            try:
                t0 = time.perf_counter(); _, states = mdp.optimal_value_capped(G, mdp_state_cap)
                t_mdp = (time.perf_counter() - t0) * 1e3
            except (CapExceeded, RecursionError):
                pass
            rows.append(dict(n=n, Q=Q, t_index_ms=t_idx, t_mdp_ms=t_mdp, mdp_states=states))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"exp2: {len(rows)} instances -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=3); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); run(seed=a.seed, trials=a.trials)
