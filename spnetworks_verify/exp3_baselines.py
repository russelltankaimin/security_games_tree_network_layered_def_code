"""Experiment 3 -- value over baselines.
Relative value loss of each heuristic vs the optimal index policy, across small
random networks of mixed topology (exact policy evaluation)."""
import csv, random, argparse
import networks as nw, mdp, policies

BASELINES = ['pc_index', 'myopic_raw', 'greedy_success', 'single_route', 'round_robin']

def run(seed=2, sizes=range(3, 8), trials=15, out="results/exp3_baselines.csv"):
    rng = random.Random(seed); rows = []
    for n in sizes:
        for t in range(trials):
            G = nw.random_network(rng, n, qmax=2, monotone=True, p_par=0.5)
            opt, _ = mdp.optimal_value(G)
            if opt < 1e-9: continue
            for name in BASELINES:
                v = mdp.policy_value(G, policies.ALL[name](G))
                rows.append(dict(n=n, Q=nw.total_Q(G), depth=nw.depth(G),
                                 policy=name, value=v, opt=opt, rel_loss=(opt - v) / opt))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"exp3: {len(rows)} rows -> {out}")
    for name in BASELINES:
        ls = [r["rel_loss"] for r in rows if r["policy"] == name]
        print(f"   {name:15s} mean loss {sum(ls)/len(ls)*100:6.3f}%  max {max(ls)*100:6.3f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=15); ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args(); run(seed=a.seed, trials=a.trials)
