"""
large_instance_generator.py
===========================

Generate LARGE series--parallel network instances for `gradient_rm.py`.

A "large" instance here means a big total state count Q (default target
>= 50,000), reached by combining MANY controls with a DEEP per-control lockout
ladder (each control has q_v >= 10 attempts). Recall Q = sum_v q_v is what sizes
the attacker/defender fold (O(Q * depth) on a balanced tree, so this generator
keeps the tree balanced to stay tractable).

Output CSV
----------
One row per network, with exactly the columns `gradient_rm.py` reads
(`network_id, structure, n_controls, total_Q, probs`) plus `depth`/`ell` for
round-trip compatibility with the other exp1/exp2 scripts:

    network_id, structure, n_controls, total_Q, depth, probs, ell

  * structure : n-ary "Ser(...)" / "Par(...)" over bare-name leaves v0, v1, ...
  * probs     : JSON {name: [p(0), p(1), ...]} -- a MONOTONE NON-INCREASING
                success ladder (Assumption 2), so `sp_attacker` uses its fast
                O(q + |R|) closed-form leaf solve rather than the O(q|R|) fallback.
                Probabilities are kept "not so low" (default in [0.35, 0.95]) so
                the game value is non-degenerate.
  * ell       : JSON {name: l_v} on the simplex (gradient_rm re-initialises the
                allocation to uniform, so this is informational / for other scripts).

Topologies (`--topology`)
-------------------------
  random          : a balanced random SP tree (n-ary nodes, arity in
                    [2, --max-arity]); depth ~ log(n), keeps the fold tractable.
  parallel_chains : Par(Ser(chain_1), ..., Ser(chain_C)) -- C independent chains
                    raced in parallel (the classic layered-defence family).

Examples
--------
    # ~50k states: 2000 controls, q_v in [10, 40] distributed to hit total_Q.
    python3 large_instance_generator.py --out big.csv \
        --n-controls 2000 --total-Q 50000 --q-min 10 --q-max 40

    # parallel chains: 50 chains, 5000 controls, each control 10-20 attempts.
    python3 large_instance_generator.py --out big_pc.csv \
        --topology parallel_chains --n-chains 50 --n-controls 5000 \
        --q-min 10 --q-max 20

    # feed straight into gradient_rm.py
    python3 gradient_rm.py --csv big.csv --out big_rm

Memory note. A single instance carries Q floating-point success probabilities;
at Q = 50,000 the `probs` JSON alone is a few hundred KB per row, and running
`gradient_rm.py` on it holds O(Q) piecewise-linear breakpoints in memory. This is
expected -- these instances are meant to be heavy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random


# ===========================================================================
# Generic nested tree  ('c', name) | ('S', [kids]) | ('P', [kids])
# (same shape gradient_rm.parse_generic produces; we just serialise it back)
# ===========================================================================
def balanced_sizes(m, k):
    """Split `m` leaves into `k` group sizes as evenly as possible (each >= 1)."""
    base, rem = divmod(m, k)
    return [base + 1] * rem + [base] * (k - rem)


def build_random_tree(names, rng, max_arity):
    """A balanced random SP tree over `names` (n-ary; arity in [2, max_arity])."""
    if len(names) == 1:
        return ("c", names[0])
    op = "S" if rng.random() < 0.5 else "P"
    k = rng.randint(2, min(max_arity, len(names)))
    sizes = balanced_sizes(len(names), k)
    kids, i = [], 0
    for s in sizes:
        kids.append(build_random_tree(names[i:i + s], rng, max_arity))
        i += s
    return (op, kids)


def build_parallel_chains(names, n_chains, rng):
    """Par(Ser(chain_1), ..., Ser(chain_C)) over `names` split into C chains."""
    n_chains = max(2, min(n_chains, len(names)))
    sizes = balanced_sizes(len(names), n_chains)
    chains, i = [], 0
    for s in sizes:
        group = names[i:i + s]
        i += s
        chains.append(("c", group[0]) if s == 1
                      else ("S", [("c", nm) for nm in group]))
    return ("P", chains)


def to_structure(node):
    """Serialise the nested tree to a 'Ser(...)'/'Par(...)' structure string."""
    if node[0] == "c":
        return node[1]
    op = "Ser" if node[0] == "S" else "Par"
    return f"{op}({','.join(to_structure(k) for k in node[1])})"


def tree_depth(node):
    """Nesting depth (a single control has depth 0)."""
    if node[0] == "c":
        return 0
    return 1 + max(tree_depth(k) for k in node[1])


# ===========================================================================
# Per-control synthesis: lockouts q_v and monotone success ladders p_v(k)
# ===========================================================================
def distribute_lockouts(n, total_q, q_min, q_max, rng):
    """Return n lockouts q_v in [q_min, q_max] summing to `total_q`.

    Each control starts at q_min; the surplus (total_q - n*q_min) is dealt out
    one at a time to controls that still have head-room, in O(surplus)."""
    lo, hi = n * q_min, n * q_max
    if not (lo <= total_q <= hi):
        raise SystemExit(
            f"--total-Q={total_q} unreachable with n={n}, q in [{q_min},{q_max}] "
            f"(feasible range [{lo}, {hi}])")
    q = [q_min] * n
    caps = [q_max - q_min] * n
    avail = [i for i in range(n) if caps[i] > 0]
    surplus = total_q - lo
    while surplus > 0:
        pos = rng.randrange(len(avail))
        ci = avail[pos]
        q[ci] += 1
        caps[ci] -= 1
        surplus -= 1
        if caps[ci] == 0:                 # swap-remove a saturated control
            avail[pos] = avail[-1]
            avail.pop()
    return q


def random_lockouts(n, q_min, q_max, rng):
    """Independent q_v ~ Uniform{q_min, ..., q_max} (total_Q emerges)."""
    return [rng.randint(q_min, q_max) for _ in range(n)]


def monotone_ladder(q_v, rng, p_min, p_max, digits):
    """A monotone NON-INCREASING success ladder of length q_v in [p_min, p_max].

    Sorting descending guarantees p(0) >= p(1) >= ... (Assumption 2); the range
    floor keeps the probabilities 'not so low'."""
    vals = sorted((rng.uniform(p_min, p_max) for _ in range(q_v)), reverse=True)
    return [round(v, digits) for v in vals]


def sample_ell(names, rng, uniform):
    """Allocation on the simplex (sum = 1): uniform 1/n, or a random simplex point."""
    n = len(names)
    if uniform:
        return {nm: 1.0 / n for nm in names}
    raw = [rng.expovariate(1.0) for _ in names]     # -> uniform on the simplex
    s = sum(raw)
    return {nm: raw[i] / s for i, nm in enumerate(names)}


# ===========================================================================
# Driver
# ===========================================================================
FIELDNAMES = ["network_id", "structure", "n_controls", "total_Q", "depth",
              "probs", "ell"]


def generate_one(nid, args):
    """Build one instance; returns the CSV row dict."""
    rng = random.Random(f"{args.seed}:{nid}")
    n = args.n_controls
    names = [f"v{i}" for i in range(n)]

    # topology
    if args.topology == "parallel_chains":
        tree = build_parallel_chains(names, args.n_chains, rng)
    else:
        tree = build_random_tree(names, rng, args.max_arity)

    # lockouts q_v (target a total_Q, or sample independently when --total-Q 0)
    if args.total_Q > 0:
        q = distribute_lockouts(n, args.total_Q, args.q_min, args.q_max, rng)
    else:
        q = random_lockouts(n, args.q_min, args.q_max, rng)

    probs = {names[i]: monotone_ladder(q[i], rng, args.p_min, args.p_max, args.digits)
             for i in range(n)}
    ell = sample_ell(names, rng, uniform=(args.ell_mode == "uniform"))

    return {
        "network_id": nid,
        "structure": to_structure(tree),
        "n_controls": n,
        "total_Q": sum(q),
        "depth": tree_depth(tree),
        "probs": json.dumps(probs),
        "ell": json.dumps({nm: round(ell[nm], 6) for nm in names}),
    }


def run(args):
    if args.n_controls < 2:
        raise SystemExit("--n-controls must be >= 2")
    if args.q_min < 1 or args.q_max < args.q_min:
        raise SystemExit("require 1 <= --q-min <= --q-max")

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for k in range(args.n_networks):
            nid = args.start_id + k
            row = generate_one(nid, args)
            writer.writerow(row)
            handle.flush()
            print(f"network {nid}: topology={args.topology} "
                  f"n_controls={row['n_controls']} total_Q={row['total_Q']} "
                  f"depth={row['depth']} "
                  f"(probs {len(row['probs']) // 1024} KB)", flush=True)

    print(f"\nwrote {args.n_networks} network(s) to {args.out}")
    print(f"feed in with:  python3 gradient_rm.py --csv {args.out} --out <tag>")


def build_arg_parser():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(here, "large_instances.csv"),
                   help="output CSV path")
    p.add_argument("--topology", choices=["random", "parallel_chains"],
                   default="random", help="network family")
    p.add_argument("--n-controls", type=int, default=2000,
                   help="number of controls (leaves)")
    p.add_argument("--n-chains", type=int, default=50,
                   help="number of parallel chains (--topology parallel_chains)")
    p.add_argument("--max-arity", type=int, default=4,
                   help="max children per internal node (--topology random)")
    p.add_argument("--total-Q", type=int, default=50000,
                   help="target total state count Q = sum_v q_v; 0 = sample q_v "
                        "independently in [q_min, q_max] (Q emerges)")
    p.add_argument("--q-min", type=int, default=10,
                   help="min attempts (lockout) per control; the '10+' floor")
    p.add_argument("--q-max", type=int, default=40, help="max attempts per control")
    p.add_argument("--p-min", type=float, default=0.35,
                   help="min success prob in the ladder ('not so low')")
    p.add_argument("--p-max", type=float, default=0.95, help="max success prob")
    p.add_argument("--digits", type=int, default=6, help="rounding for probabilities")
    p.add_argument("--ell-mode", choices=["uniform", "random"], default="uniform",
                   help="allocation written to the ell column (gradient_rm re-inits it)")
    p.add_argument("--n-networks", type=int, default=1, help="how many instances")
    p.add_argument("--start-id", type=int, default=0, help="first network_id")
    p.add_argument("--seed", type=int, default=0, help="master RNG seed")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
