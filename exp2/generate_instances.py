"""
generate_instances.py
=====================

Make more `series_heavy` / `highly_parallel` test instances at a SPECIFIED number
of controls `n` and total state count `Q`, tuned to each family's reward regime so
the intended exact-vs-stochastic behaviour holds. Output loads directly into
`gradient_rm.py`.

Families
--------
  series_heavy     `Par` of a FEW LONG `Ser` chains (narrow, deep) with LOW success
                   probs -> RARE reward (V* in ~[0.01, 0.08]) -> high stochastic-
                   gradient noise -> EXACT-favouring (by right lah).
  highly_parallel  `Par` of MANY SHORT chains (wide, shallow) with HIGH success
                   probs -> FREQUENT, consistent reward (V* high) -> low noise ->
                   STOCHASTIC-favouring.

Sizing
------
Given `--n-controls n` and `--total-Q Q`, every control gets lockout
q = Q // n (the first Q % n controls get one extra) so `total_Q` is exactly `Q`.
The shape is set by `--n-chains` (series_heavy: few chains -> long) or
`--chain-len` (highly_parallel: short chains -> many). Success probabilities are
searched over the family's range grid to land V* in its target band; if no range
lands in-band, the closest is used and a warning is printed.

Examples
--------
    # 5 exact-favouring nets, 200 controls, Q=4000, 6 long chains each
    python3 generate_instances.py --family series_heavy \
        --n-controls 200 --total-Q 4000 --n-chains 6 --n-networks 5 --out sh_new.csv

    # 3 stochastic-favouring nets, 300 controls, Q=6000, chains of length 3
    python3 generate_instances.py --family highly_parallel \
        --n-controls 300 --total-Q 6000 --chain-len 3 --n-networks 3 --out hp_new.csv

    python3 gradient_rm.py --csv sh_new.csv --out sh_run --batch 1

Note: verifying V* runs one exact fold per candidate; on large / long-chain
instances that fold is expensive, so generation of big `series_heavy` nets is slow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gradient_rm as G                      # reuse parse/build helpers  # noqa: E402
from sp_attacker import IndexAttacker        # noqa: E402

FAMILY = {
    "series_heavy": dict(
        v_target=0.04, v_band=(0.01, 0.08),
        pranges=[(0.03, 0.10), (0.05, 0.15), (0.08, 0.20), (0.12, 0.28),
                 (0.15, 0.40), (0.20, 0.45)],
    ),
    "highly_parallel": dict(
        v_target=0.80, v_band=(0.30, 0.97),
        pranges=[(0.50, 0.75), (0.55, 0.80), (0.60, 0.85), (0.65, 0.88),
                 (0.70, 0.90), (0.75, 0.95)],
    ),
}


# ---------------------------------------------------------------------------
def balanced_sizes(n, k):
    """Split `n` controls into `k` chain sizes as evenly as possible (each >= 1)."""
    k = max(1, min(k, n))
    base, rem = divmod(n, k)
    return [base + 1] * rem + [base] * (k - rem)


def build_structure(sizes):
    """`Par` of chains; a size-1 chain is a bare control, longer ones are `Ser(...)`."""
    chains, idx = [], 0
    for s in sizes:
        names = [f"v{idx + j}" for j in range(s)]
        idx += s
        chains.append(names[0] if s == 1 else "Ser(" + ",".join(names) + ")")
    if len(chains) == 1:
        return chains[0]
    return "Par(" + ",".join(chains) + ")"


def distribute_q(n, total_q):
    """Per-control lockout so the sum is exactly `total_q` (base q, +1 for the first r)."""
    base, rem = divmod(total_q, n)
    return [base + 1 if i < rem else base for i in range(n)]


def ladder(pmin, pmax, q, rng):
    """Monotone non-increasing success ladder of length q in [pmin, pmax]."""
    return sorted((round(rng.uniform(pmin, pmax), 6) for _ in range(q)), reverse=True)


def tree_depth(node):
    return 0 if node[0] == "c" else 1 + max(tree_depth(c) for c in node[1])


# ---------------------------------------------------------------------------
def make_one(nid, args, spec):
    n, Q = args.n_controls, args.total_Q
    if args.family == "series_heavy":
        n_chains = args.n_chains
    else:
        n_chains = max(2, -(-n // args.chain_len))     # ceil(n / chain_len)
    sizes = balanced_sizes(n, n_chains)
    structure = build_structure(sizes)
    node = G.flatten_generic(G.parse_generic(structure))
    names = G.leaf_names(node)
    ell = {v: 1.0 / n for v in names}
    q_list = distribute_q(n, Q)
    q_by_name = {names[i]: q_list[i] for i in range(n)}

    best = None                                        # (score, V*, probs, prange)
    for pmin, pmax in spec["pranges"]:
        rng = random.Random(f"{args.seed}:{nid}:{pmin}")
        probs = {nm: ladder(pmin, pmax, q_by_name[nm], rng) for nm in names}
        V = G.exact_value(G.build_defender(node, probs), ell, args.rho)
        score = abs(V - spec["v_target"])
        lo, hi = spec["v_band"]
        in_band = lo <= V <= hi
        if best is None or (in_band and not best[3]) or (in_band == best[3] and score < best[0]):
            best = (score, V, probs, in_band, (pmin, pmax))
    _, V, probs, in_band, prange = best
    if not in_band:
        print(f"  warning: net {nid} V*={V:.4f} outside target band {spec['v_band']} "
              f"(closest achievable at n={n}, Q={Q}); using it anyway.", flush=True)

    net = IndexAttacker(G.build_attacker(node, probs, ell, args.rho))
    rr = random.Random(1)
    pw = sum(1 for _ in range(300) if net.simulate(rr).reward > 0) / 300
    G.value_profile.cache_clear()

    row = {"network_id": nid, "structure": structure, "n_controls": n,
           "total_Q": sum(q_list), "depth": tree_depth(node), "probs": json.dumps(probs)}
    return row, V, pw, prange, n_chains


def run(args):
    if args.family not in FAMILY:
        raise SystemExit(f"--family must be one of {list(FAMILY)}")
    if args.total_Q < args.n_controls:
        raise SystemExit(f"--total-Q ({args.total_Q}) must be >= --n-controls ({args.n_controls})")
    spec = FAMILY[args.family]

    print(f"family={args.family}  n_controls={args.n_controls}  total_Q={args.total_Q}  "
          f"q~{args.total_Q // args.n_controls}  rho={args.rho}\n"
          f"{'net':>3} {'chains':>6} {'depth':>5} {'V*':>8} {'P(win)':>7} {'p-range':>12}", flush=True)
    rows = []
    for nid in range(args.start_id, args.start_id + args.n_networks):
        row, V, pw, pr, C = make_one(nid, args, spec)
        rows.append(row)
        print(f"{nid:>3} {C:>6} {row['depth']:>5} {V:>8.4f} {pw:>7.3f} "
              f"{f'[{pr[0]},{pr[1]}]':>12}", flush=True)

    with open(args.out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["network_id", "structure", "n_controls",
                                          "total_Q", "depth", "probs"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {len(rows)} network(s) to {args.out}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=list(FAMILY))
    p.add_argument("--n-controls", type=int, required=True, help="number of controls n")
    p.add_argument("--total-Q", type=int, required=True, help="total state count Q = sum_v q_v")
    p.add_argument("--n-chains", type=int, default=6,
                   help="series_heavy: number of (long) chains -- keep small for narrow/deep")
    p.add_argument("--chain-len", type=int, default=3,
                   help="highly_parallel: chain length -- keep small for wide/shallow")
    p.add_argument("--rho", type=float, default=5.0, help="discount rate rho (> 0)")
    p.add_argument("--n-networks", type=int, default=5, help="how many instances")
    p.add_argument("--start-id", type=int, default=0, help="first network_id")
    p.add_argument("--seed", type=int, default=0, help="master RNG seed")
    p.add_argument("--out", default=os.path.join(_HERE, "instances.csv"), help="output CSV path")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
