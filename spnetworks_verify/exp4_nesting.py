"""Experiment 4 -- generalisation beyond parallel chains.
On small nested (Par-inside-Ser) instances, the parallel-chains relaxation
(pc_index = AAAI-26 applied by flattening) is suboptimal and its gap grows with
nesting depth. Optimal reference = exact index value (validated in exp1);
baselines evaluated by Monte-Carlo."""
import csv, random, argparse
import networks as nw, sp_index, mdp, policies

def run(seed=3, levels=range(1, 4), trials=8, samples=4000, out="results/exp4_nesting.csv"):
    rng = random.Random(seed); mcrng = random.Random(seed + 100); rows = []
    for L in levels:
        for t in range(trials):
            G = nw.nested_network(rng, L, qmax=2, monotone=True)
            opt = sp_index.game_value(G)
            if opt < 1e-9: continue
            for name in ['pc_index', 'myopic_raw', 'single_route']:
                v = mdp.mc_policy_value(G, policies.ALL[name](G), n_samples=samples, rng=mcrng)
                rows.append(dict(levels=L, n=nw.count_controls(G), depth=nw.depth(G),
                                 policy=name, rel_loss=max(0.0, (opt - v) / opt)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"exp4: {len(rows)} rows -> {out}")
    for L in levels:
        ls = [r["rel_loss"] for r in rows if r["levels"] == L and r["policy"] == 'pc_index']
        if ls:
            nn = [r['n'] for r in rows if r['levels'] == L][0]
            print(f"   levels={L} (n={nn}): pc_index mean loss {sum(ls)/len(ls)*100:6.3f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=8); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(); run(seed=a.seed, trials=a.trials)
