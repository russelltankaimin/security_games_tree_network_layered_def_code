"""
run_experiments.py
==================

Sweep runner: generate a specified number of random series--parallel (SP)
networks, evaluate every implemented attacker policy on each, and log all
metrics to a single CSV (one row per network x policy).

Reuses the tested building blocks:
    sp_attacker.IndexAttacker, game_value, controls   (the optimal policy + Phase 1)
    other_policies.baseline_policies                  (every baseline)
    experiment_attacker.evaluate_policy               (Monte-Carlo + exact metrics)

Example
-------
    # 50 networks of 3-8 controls, 20k rollouts each, into sweep_metrics.csv
    python3 run_experiments.py --num-networks 50 --min-controls 3 --max-controls 8 \
        --samples 20000 --seed 0 --out sweep_metrics.csv

The whole sweep is reproducible from --seed alone: all network structures, control
parameters, and Monte-Carlo streams are derived from a single seeded master RNG.

CSV columns
-----------
    network_id, structure, n_controls, total_Q, depth,           (network descriptors)
    policy, exact_value, expected_reward, std_error, ci95_halfwidth,
    win_probability, mean_attempts, mean_attempts_given_win,
    optimal_value, absolute_gap, relative_loss, num_samples, seed (per-policy metrics)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List

from sp_attacker import (
    Control,
    IndexAttacker,
    Network,
    Par,
    Series,
    controls,
    game_value,
)
from other_policies import baseline_policies
from experiment_attacker import evaluate_policy

# Column order for the output CSV.
FIELDNAMES = [
    "network_id",
    "structure",
    "n_controls",
    "total_Q",
    "depth",
    "policy",
    "exact_value",
    "expected_reward",
    "std_error",
    "ci95_halfwidth",
    "win_probability",
    "mean_attempts",
    "mean_attempts_given_win",
    "optimal_value",
    "absolute_gap",
    "relative_loss",
    "num_samples",
    "seed",
]


# ===========================================================================
# Random network generation
# ===========================================================================


def random_network(
    rng: random.Random,
    n_controls: int,
    *,
    qmax: int,
    p_min: float,
    p_max: float,
    beta_min: float,
    beta_max: float,
    p_par: float,
) -> Network:
    """Generate a random SP network with exactly `n_controls` controls.

    A random binary tree is built by repeatedly splitting the control budget;
    each internal node is Par with probability `p_par`, else Series. Each leaf is
    a control with `q ~ U{1..qmax}` lockout, discount `beta ~ U[beta_min, beta_max]`,
    and monotone non-increasing success probabilities drawn in [p_min, p_max]
    (Assumption 2).
    """
    counter = [0]

    def make_leaf() -> Control:
        name = f"v{counter[0]}"
        counter[0] += 1
        q = rng.randint(1, qmax)
        ps = tuple(sorted((round(rng.uniform(p_min, p_max), 4) for _ in range(q)), reverse=True))
        beta = round(rng.uniform(beta_min, beta_max), 4)
        return Control(name, beta, ps)

    def build(n: int) -> Network:
        if n == 1:
            return make_leaf()
        left = rng.randint(1, n - 1)
        node = Par if rng.random() < p_par else Series
        return node(build(left), build(n - left))

    return build(n_controls)


def network_depth(net: Network) -> int:
    """Nesting depth of the parse tree (a single control has depth 0)."""
    if isinstance(net, Control):
        return 0
    return 1 + max(network_depth(net.left), network_depth(net.right))


def describe(net: Network) -> str:
    """Compact structural string, e.g. 'Par(Ser(v0,Par(v1,v2)),v3)'."""
    if isinstance(net, Control):
        return net.name
    op = "Par" if isinstance(net, Par) else "Ser"
    return f"{op}({describe(net.left)},{describe(net.right)})"


# ===========================================================================
# Sweep
# ===========================================================================


def run_sweep(args: argparse.Namespace) -> List[Dict]:
    """Generate `args.num_networks` networks, evaluate all policies, return rows."""
    master = random.Random(args.seed)  # single source of randomness for the whole run
    rows: List[Dict] = []

    for network_id in range(args.num_networks):
        n_controls = master.randint(args.min_controls, args.max_controls)

        # Independent, reproducible sub-streams derived from the master RNG.
        gen_rng = random.Random(master.randrange(2**31))      # network structure + params
        eval_seed = master.randrange(2**31)                   # Monte-Carlo outcome stream
        policy_seed = master.randrange(2**31)                 # stochastic policies' choices

        net = random_network(
            gen_rng,
            n_controls,
            qmax=args.qmax,
            p_min=args.p_min,
            p_max=args.p_max,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            p_par=args.p_par,
        )

        optimal_value = game_value(net)  # V* from the exact Phase-1 fold
        net_controls = controls(net)
        descriptors = {
            "network_id": network_id,
            "structure": describe(net),
            "n_controls": len(net_controls),
            "total_Q": sum(c.q for c in net_controls),
            "depth": network_depth(net),
        }

        # All implemented algorithms: the optimal index policy plus every baseline.
        policies: Dict[str, object] = {"index_policy": IndexAttacker(net)}
        policies.update(baseline_policies(net, rng=random.Random(policy_seed)))

        for label, policy in policies.items():
            row = evaluate_policy(net, label, policy, args.samples, eval_seed, optimal_value)
            row.update(descriptors)
            rows.append(row)

        print(
            f"[{network_id + 1:>3}/{args.num_networks}] "
            f"n={descriptors['n_controls']} Q={descriptors['total_Q']} "
            f"depth={descriptors['depth']} V*={optimal_value:.4f}  {descriptors['structure']}"
        )

    _write_csv(rows, args.out)
    _print_aggregate(rows, args.out)
    return rows


# ===========================================================================
# Output
# ===========================================================================


def _format(value) -> str:
    """CSV cell formatting: blank for None/NaN, six decimals for floats."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def _write_csv(rows: List[Dict], out_path: str) -> None:
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in FIELDNAMES})


def _print_aggregate(rows: List[Dict], out_path: str) -> None:
    """Print mean relative loss and win probability per policy across all networks."""
    by_policy: Dict[str, List[Dict]] = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)

    print(f"\nmean over {len({r['network_id'] for r in rows})} networks:")
    print(f"{'policy':18s} {'mean rel.loss':>14s} {'mean win p':>11s}")
    ordered = sorted(by_policy.items(), key=lambda kv: sum(r["relative_loss"] for r in kv[1]))
    for label, policy_rows in ordered:
        mean_loss = sum(r["relative_loss"] for r in policy_rows) / len(policy_rows)
        mean_win = sum(r["win_probability"] for r in policy_rows) / len(policy_rows)
        print(f"{label:18s} {100 * mean_loss:13.3f}% {mean_win:11.3f}")
    print(f"\nwrote {len(rows)} rows to {out_path}")


# ===========================================================================
# CLI
# ===========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gen = parser.add_argument_group("network generation")
    gen.add_argument("--num-networks", type=int, default=20, help="how many networks to generate")
    gen.add_argument("--min-controls", type=int, default=3, help="min controls per network")
    gen.add_argument("--max-controls", type=int, default=7, help="max controls per network")
    gen.add_argument("--qmax", type=int, default=2, help="max lockout count q per control")
    gen.add_argument("--p-min", type=float, default=0.2, help="min success probability")
    gen.add_argument("--p-max", type=float, default=0.9, help="max success probability")
    gen.add_argument("--beta-min", type=float, default=0.9, help="min per-attempt discount")
    gen.add_argument("--beta-max", type=float, default=0.9, help="max per-attempt discount")
    gen.add_argument("--p-par", type=float, default=0.5, help="probability an internal node is Par")

    eval_group = parser.add_argument_group("evaluation")
    eval_group.add_argument("--samples", type=int, default=20_000, help="Monte-Carlo rollouts per policy")
    eval_group.add_argument("--seed", type=int, default=0, help="master RNG seed (reproduces the whole sweep)")
    eval_group.add_argument("--out", type=str, default="sweep_metrics.csv", help="output CSV path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.min_controls > args.max_controls:
        raise SystemExit("--min-controls must not exceed --max-controls")
    run_sweep(args)


if __name__ == "__main__":
    main()

"""
Sample Command run:
python3 run_experiments.py --num-networks 50 --min-controls 3 --max-controls 8 --samples 20000 --seed 0 --out sweep.csv
"""
