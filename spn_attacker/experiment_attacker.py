"""
experiment_attacker.py
======================

Simulate the optimal index attacker and every baseline on a SINGLE network,
record metrics useful for a paper, and write them to a CSV (one row per policy).

Edit `build_network()` to choose the network, then run:

    python3 experiment_attacker.py                       # defaults
    python3 experiment_attacker.py --samples 100000 --seed 1 --out results.csv

Metrics (per policy)
--------------------
    exact_value             exact policy value by backward induction (deterministic
                            policies only, and only when the joint state space is
                            small enough); noise-free ground truth when available
    expected_reward         Monte-Carlo mean realised discounted reward
    std_error               standard error of expected_reward (for error bars)
    ci95_halfwidth          1.96 * std_error (95% confidence half-width)
    win_probability         fraction of rollouts that reach the target
    mean_attempts           mean number of attempts (all rollouts)
    mean_attempts_given_win mean attempts conditional on success (time-to-compromise)
    optimal_value           V* = game_value(net) (exact optimum; same for all rows)
    absolute_gap            V* - policy value
    relative_loss           (V* - policy value) / V*   <-- the headline suboptimality

The "policy value" used for the gap/loss is the exact value when available, else
the Monte-Carlo mean. The index policy should show ~0 gap (it is optimal).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Optional

from sp_attacker import (
    Control,
    IndexAttacker,
    Network,
    Par,
    Series,
    Status,
    apply_outcome,
    block_status,
    controls,
    frontier,
    game_value,
    initial_state,
)
from other_policies import baseline_policies, simulate

# Column order for the CSV / printed table.
FIELDNAMES = [
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
    "n_controls",
    "total_Q",
]


# ===========================================================================
# 1. The network to study  --  EDIT HERE
# ===========================================================================


def build_network() -> Network:
    """Return the SP network to simulate. Replace the body to study your own.

    Default: a two-branch nested instance,
        N = Par( Series(a, Par(b, c)), Series(d, e) ),
    where the optimal policy opens on branch a (its forward cone is the cheap
    redundant OR-block {b, c}) even though d is the better individual lock --
    so coupling-blind baselines mis-open and lose value.
    """
    a = Control("a", 0.9, (0.5, 0.2))
    b = Control("b", 0.9, (0.7, 0.4))
    c = Control("c", 0.9, (0.6, 0.3))
    d = Control("d", 0.9, (0.8, 0.5))
    e = Control("e", 0.9, (0.55, 0.25))
    return Par(Series(a, Par(b, c)), Series(d, e))


# ===========================================================================
# 2. Exact policy value (noise-free) for deterministic policies
# ===========================================================================


class _StateSpaceTooLarge(Exception):
    """Raised when the joint state space exceeds the exact-evaluation cap."""


def exact_policy_value(net: Network, policy, state_cap: int = 300_000) -> Optional[float]:
    """Exact value of a deterministic policy by backward induction over states.

    Returns None if `policy` is stochastic or the reachable state space exceeds
    `state_cap` (in which case fall back to the Monte-Carlo estimate).
    """
    if not getattr(policy, "deterministic", True):
        return None
    cmap = {c.name: c for c in controls(net)}
    cache: Dict = {}
    seen = [0]

    def value(state) -> float:
        status = block_status(net, state)
        if status is Status.WON:
            return 1.0
        if status is Status.DEAD:
            return 0.0
        key = tuple(sorted(state.items(), key=lambda kv: kv[0]))
        if key in cache:
            return cache[key]
        seen[0] += 1
        if seen[0] > state_cap:
            raise _StateSpaceTooLarge
        name = policy.action(state)
        if name is None:  # policy stops -> no reward from here
            cache[key] = 0.0
            return 0.0
        control = cmap[name]
        p = control.ps[state[name]]
        result = control.beta * (
            p * value(apply_outcome(state, control, True))
            + (1.0 - p) * value(apply_outcome(state, control, False))
        )
        cache[key] = result
        return result

    try:
        return value(initial_state(net))
    except _StateSpaceTooLarge:
        return None


# ===========================================================================
# 3. Evaluate one policy: Monte-Carlo metrics + exact value + suboptimality
# ===========================================================================


def evaluate_policy(
    net: Network,
    label: str,
    policy,
    num_samples: int,
    seed: int,
    optimal_value: float,
) -> Dict:
    """Run `num_samples` rollouts and return the metric row for this policy."""
    rng = random.Random(seed)
    rewards: List[float] = []
    attempts: List[int] = []
    attempts_given_win: List[int] = []
    wins = 0

    for _ in range(num_samples):
        result = simulate(net, policy, rng)
        rewards.append(result.reward)
        attempts.append(result.attempts)
        if result.won:
            wins += 1
            attempts_given_win.append(result.attempts)

    n = num_samples
    mean_reward = sum(rewards) / n
    variance = sum((r - mean_reward) ** 2 for r in rewards) / (n - 1) if n > 1 else 0.0
    std_error = math.sqrt(variance / n)

    exact = exact_policy_value(net, policy)
    value_for_gap = exact if exact is not None else mean_reward
    absolute_gap = optimal_value - value_for_gap
    relative_loss = absolute_gap / optimal_value if optimal_value > 0 else 0.0

    return {
        "policy": label,
        "exact_value": exact,  # None -> written as "" in the CSV
        "expected_reward": mean_reward,
        "std_error": std_error,
        "ci95_halfwidth": 1.96 * std_error,
        "win_probability": wins / n,
        "mean_attempts": sum(attempts) / n,
        "mean_attempts_given_win": (
            sum(attempts_given_win) / len(attempts_given_win) if attempts_given_win else float("nan")
        ),
        "optimal_value": optimal_value,
        "absolute_gap": absolute_gap,
        "relative_loss": relative_loss,
        "num_samples": n,
        "seed": seed,
    }


# ===========================================================================
# 4. Run all policies on one network and write the CSV
# ===========================================================================


def run(net: Network, num_samples: int, seed: int, out_path: str) -> List[Dict]:
    """Evaluate the index policy and all baselines; write the CSV; return rows."""
    optimal_value = game_value(net)  # V* (exact, from the Phase-1 fold)
    net_controls = controls(net)
    meta = {"n_controls": len(net_controls), "total_Q": sum(c.q for c in net_controls)}

    # The optimal index policy plus all baselines, keyed by label.
    policies: Dict[str, object] = {"index_policy": IndexAttacker(net)}
    policies.update(baseline_policies(net, rng=random.Random(seed)))

    rows: List[Dict] = []
    for label, policy in policies.items():
        row = evaluate_policy(net, label, policy, num_samples, seed, optimal_value)
        row.update(meta)
        rows.append(row)

    rows.sort(key=lambda r: -r["expected_reward"])  # best policy first
    _write_csv(rows, out_path)
    _print_summary(rows, optimal_value, out_path)
    return rows


def _format(value) -> str:
    """CSV cell formatting: blanks for None, six decimals for floats."""
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
            writer.writerow({key: _format(row[key]) for key in FIELDNAMES})


def _print_summary(rows: List[Dict], optimal_value: float, out_path: str) -> None:
    print(f"optimal value V* = {optimal_value:.6f}   ({rows[0]['n_controls']} controls, "
          f"Q={rows[0]['total_Q']}, {rows[0]['num_samples']} samples)\n")
    print(f"{'policy':18s} {'reward':>9s} {'+/-95%':>8s} {'win p':>7s} "
          f"{'E[att]':>7s} {'rel.loss':>9s}")
    for r in rows:
        print(f"{r['policy']:18s} {r['expected_reward']:9.4f} {r['ci95_halfwidth']:8.4f} "
              f"{r['win_probability']:7.3f} {r['mean_attempts']:7.2f} "
              f"{100 * r['relative_loss']:8.3f}%")
    print(f"\nwrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50_000, help="Monte-Carlo rollouts per policy")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--out", type=str, default="attacker_metrics.csv", help="output CSV path")
    args = parser.parse_args()
    run(build_network(), args.samples, args.seed, args.out)


if __name__ == "__main__":
    main()
