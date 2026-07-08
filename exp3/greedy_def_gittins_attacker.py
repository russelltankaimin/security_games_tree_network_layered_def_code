"""
greedy_def_gittins_attacker.py
==============================

MISMATCHED game: the defender optimises its allocation l ASSUMING the attacker is
GREEDY, then the true GITTINS-INDEX (optimal) attacker best-responds to that fixed
l. We measure the price of the defender's wrong assumption.

    greedy attacker : always attacks the live (frontier) control with the greatest
                      one-step success probability p(k)  (myopic; see
                      other_policies.GreedyProbabilityPolicy).

Why the greedy defender problem is well posed
---------------------------------------------
The greedy rule scores by p(k) alone, and p(k) does NOT depend on l (l only sets
beta_v = exp(-rho l_v), the per-attempt discount). So the greedy policy is a FIXED,
l-independent policy pi_g, hence

    V_greedy(l) = E_{pi_g}[ 1_win * prod_v beta_v^{n_v} ]

is a smooth, convex function of l (an expectation of log-linear terms), and its
pathwise gradient carries NO envelope/Danskin term:

    dV_greedy/dl_v = -rho * E_{pi_g}[ n_v * R ],   R = realised discounted reward.

We estimate it by averaging n_v * R over `--samples` greedy rollouts (unbiased),
and the defender minimises V_greedy over the simplex {l >= 0, sum l = 1} by the
same regret matching used in exp2, from the uniform start.

What we log
-----------
  * defender_opt_time_s : wall-clock to compute l* (only the gradient + regret-
                          matching update are timed; monitoring is untimed).
  * minimax_value       : V_index(l*) -- the value when the OPTIMAL (Gittins)
                          attacker exploits the greedy-optimal l*, via the exact
                          O(Q^2) fold (sp_gradients2). This is the game value the
                          defender actually suffers.
  * greedy_value_est    : V_greedy(l*) -- the value the defender THOUGHT it would
                          get (fresh Monte-Carlo estimate). The gap
                          minimax_value - greedy_value_est is the cost of wrongly
                          assuming a greedy adversary.

Termination (first to fire; `stop_reason` records it)
-----------------------------------------------------
  * "regret"   : average external regret max_v R_v / t < --epsilon; or
  * "plateau"  : the running-average allocation l_bar has stopped moving -- its
                 per-control spread over the last --patience monitored points is
                 below --obj-tol (and t >= --min-iters). Allocation-based (not
                 value-based) so it needs no noisy V_greedy evaluation. Set
                 --patience 0 to disable.
  * else "max_iters" at --max-iters.

Output CSV (--out): one row per network.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from math import exp

# --- make the sibling attacker/defender packages importable ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _pkg in ("spn_attacker", "spn_defender"):
    _path = os.path.join(_ROOT, _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from sp_attacker import (  # noqa: E402
    Control as AttackerControl,
    Par as AttackerPar,
    Series as AttackerSeries,
    index_table,
    value_profile,
)
from other_policies import (  # noqa: E402
    GreedyProbabilityPolicy,
    simulate as simulate_policy,
)
from sp_gradients2 import (  # noqa: E402
    Control as DefenderControl,
    Parallel as DefenderParallel,
    Series as DefenderSeries,
    value_and_gradient,
)


# ===========================================================================
# Structure parsing:  "Ser(A,B)" / "Par(A,B)" / "name"  ->  a tree
# ===========================================================================


def parse_structure(text, make_control, make_series, make_par):
    """Generic binary Ser/Par parser; node builders are supplied by the caller."""
    pos = 0

    def parse():
        nonlocal pos
        head = text[pos:pos + 4]
        if head in ("Ser(", "Par("):
            op = head[:3]
            pos += 4
            left = parse()
            assert text[pos] == ","
            pos += 1
            right = parse()
            assert text[pos] == ")"
            pos += 1
            return make_series(left, right) if op == "Ser" else make_par(left, right)
        start = pos
        while pos < len(text) and text[pos] not in ",)":
            pos += 1
        return make_control(text[start:pos])

    tree = parse()
    assert pos == len(text), f"trailing input: {text[pos:]!r}"
    return tree


def leaf_names(text):
    names = []
    parse_structure(text, lambda n: names.append(n) or n,
                    lambda a, b: None, lambda a, b: None)
    return names


def build_defender_tree(structure, probs):
    """sp_gradients2 tree (allocation-agnostic; carries lockouts and probs)."""
    return parse_structure(
        structure,
        lambda name: DefenderControl(name, len(probs[name]), list(probs[name])),
        lambda a, b: DefenderSeries(a, b),
        lambda a, b: DefenderParallel(a, b),
    )


def build_attacker_net(structure, probs, ell, rho):
    """sp_attacker network with beta_v = exp(-rho * l_v)."""
    return parse_structure(
        structure,
        lambda name: AttackerControl(name, exp(-rho * ell[name]), tuple(probs[name])),
        lambda a, b: AttackerSeries(a, b),
        lambda a, b: AttackerPar(a, b),
    )


# ===========================================================================
# Oracles
# ===========================================================================


def greedy_gradient(structure, probs, ell, rho, samples, rng, names, policy=None):
    """Monte-Carlo V_greedy(l) and its (unbiased) pathwise gradient from `samples`
    rollouts of the GREEDY attacker.

        dV_greedy/dl_v = -rho * E_{pi_g}[ n_v * R ].

    `policy` (a GreedyProbabilityPolicy) may be reused across iterations: its
    action rule depends only on p(k), not on l, so its decisions are identical at
    every l -- only the realised discount (via the freshly built net) changes."""
    net = build_attacker_net(structure, probs, ell, rho)
    if policy is None:
        policy = GreedyProbabilityPolicy(net)
    sum_nvR = {name: 0.0 for name in names}
    total_R = 0.0
    for _ in range(samples):
        roll = simulate_policy(net, policy, rng)
        R = roll.reward
        if R > 0.0:
            total_R += R
            for name, _k, _succ in roll.history:
                sum_nvR[name] += R          # summed over occurrences => n_v * R
    grad = {name: -rho * sum_nvR[name] / samples for name in names}
    return total_R / samples, grad


def index_value(defender_tree, ell, rho):
    """V*(l): the OPTIMAL (Gittins index) attacker's value via the exact O(Q^2)
    fold. This is the minimax value the greedy defender actually suffers."""
    value, _ = value_and_gradient(defender_tree, allocation=ell,
                                  discount_rate=rho, break_ties=False)
    return value


def indices_by_control(structure, probs, ell, rho, names):
    """Gittins index table alpha(v, k) at allocation `ell`, grouped by control."""
    table = index_table(build_attacker_net(structure, probs, ell, rho))
    grouped = {}
    for name in names:
        ks = sorted(k for (nm, k) in table if nm == name)
        grouped[name] = [round(table[(name, k)], 6) for k in ks]
    return grouped


# ===========================================================================
# Defender optimisation: regret matching against the GREEDY attacker
# ===========================================================================


def optimise_greedy_defender(structure, probs, names, rho, samples, seed,
                             max_iters, epsilon, obj_tol, patience, min_iters,
                             log_every, label="", progress_every=0):
    """Minimise V_greedy(l) over the simplex by regret matching from the uniform
    allocation, using stochastic (greedy-rollout) gradients.

    Returns (ell_star, stop_reason, iters, opt_time_s, final_regret), where
    ell_star is the running-average iterate at termination and opt_time_s is the
    wall-clock of the gradient + update work only (monitoring is untimed)."""
    n = len(names)
    rng = random.Random(f"{seed}:greedy")
    ell = {v: 1.0 / n for v in names}          # uniform initialisation
    cumulative_regret = {v: 0.0 for v in names}
    ell_sum = {v: 0.0 for v in names}
    opt_time = 0.0

    # Greedy decisions are l-independent -> build the policy once and reuse it.
    ref_net = build_attacker_net(structure, probs, ell, rho)
    greedy_policy = GreedyProbabilityPolicy(ref_net)

    window = []                                # recent l_bar snapshots (for plateau)
    stop_reason = "max_iters"
    ell_star = dict(ell)
    final_regret = float("inf")
    t = 0

    for t in range(1, max_iters + 1):
        for v in names:                        # accumulate the played iterate
            ell_sum[v] += ell[v]

        start = time.perf_counter()
        _v_est, grad = greedy_gradient(structure, probs, ell, rho, samples, rng,
                                       names, policy=greedy_policy)
        expected_loss = sum(ell[v] * grad[v] for v in names)
        for v in names:
            cumulative_regret[v] += expected_loss - grad[v]
        positive = {v: max(cumulative_regret[v], 0.0) for v in names}
        total = sum(positive.values())
        ell = ({v: positive[v] / total for v in names} if total > 0.0
               else {v: 1.0 / n for v in names})
        opt_time += time.perf_counter() - start

        # Fresh attacker net per iteration -> drop value_profile's unbounded cache
        # (pure function; cannot change any result).
        value_profile.cache_clear()

        final_regret = max(0.0, max(cumulative_regret.values())) / t
        regret_hit = epsilon is not None and final_regret < epsilon

        is_log = (t == 1 or t % log_every == 0 or t == max_iters or regret_hit)
        is_progress = progress_every and (
            t == 1 or t % progress_every == 0 or t == max_iters or regret_hit)
        plateau_hit = False
        if is_log or is_progress:
            ell_avg = {v: ell_sum[v] / t for v in names}
            if is_log:
                # Allocation-plateau test on the fixed log cadence: has l_bar
                # stopped moving? (value-free, so no MC noise enters the test.)
                window.append(ell_avg)
                if len(window) > patience:
                    window.pop(0)
                if patience and t >= min_iters and len(window) == patience:
                    spread = max(max(w[v] for w in window) - min(w[v] for w in window)
                                 for v in names)
                    if spread < obj_tol:
                        plateau_hit = True
            if is_progress and not (regret_hit or plateau_hit):
                print(f"    [{label}] iter {t:>6}/{max_iters}  "
                      f"V_greedy~{_v_est:.6f}  regret={final_regret:.3e}  "
                      f"{opt_time:6.1f}s", flush=True)

        if regret_hit or plateau_hit:
            stop_reason = "regret" if regret_hit else "plateau"
            ell_star = {v: ell_sum[v] / t for v in names}
            print(f"    [{label}] *** stop [{stop_reason}] at iter {t} "
                  f"({opt_time:.2f}s, regret={final_regret:.3e}) ***", flush=True)
            break
    else:
        ell_star = {v: ell_sum[v] / max(t, 1) for v in names}

    return ell_star, stop_reason, t, opt_time, final_regret


# ===========================================================================
# Experiment driver
# ===========================================================================


def load_networks(csv_path):
    seen = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            nid = row["network_id"]
            if nid in seen:
                continue
            seen[nid] = {
                "network_id": int(nid),
                "structure": row["structure"],
                "n_controls": int(row.get("n_controls") or 0),
                "total_Q": int(row.get("total_Q") or 0),
                "probs": json.loads(row["probs"]) if row.get("probs") else None,
            }
    return sorted(seen.values(), key=lambda r: r["network_id"])


FIELDS = ["network_id", "n_controls", "total_Q", "rho", "samples",
          "converged", "stop_reason", "iters", "defender_opt_time_s",
          "final_regret", "greedy_value_est", "minimax_value", "regret_gap",
          "final_ell", "final_indices"]


def run(args):
    networks = load_networks(args.csv)
    networks = [nw for nw in networks if nw["network_id"] >= args.start_id]
    if args.end_id is not None:
        networks = [nw for nw in networks if nw["network_id"] <= args.end_id]
    if args.limit:
        networks = networks[: args.limit]
    if not networks:
        raise SystemExit(f"no networks in id range [{args.start_id}, {args.end_id}]")

    mode = "a" if args.append else "w"
    write_header = not (args.append and os.path.exists(args.out)
                        and os.path.getsize(args.out) > 0)
    out_file = open(args.out, mode, newline="")
    writer = csv.DictWriter(out_file, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
    out_file.flush()

    print(f"{len(networks)} networks | greedy-defender vs Gittins-attacker | "
          f"rho={args.rho} samples={args.samples} eps={args.epsilon} "
          f"obj_tol={args.obj_tol} patience={args.patience}\n", flush=True)

    for idx, info in enumerate(networks, 1):
        if info["probs"] is None:
            raise ValueError(f"network {info['network_id']}: needs a probs column")
        names = leaf_names(info["structure"])
        defender_tree = build_defender_tree(info["structure"], info["probs"])
        print(f"=== [{idx}/{len(networks)}] network {info['network_id']}: "
              f"n={info['n_controls']} Q={info['total_Q']} ===", flush=True)

        # 1) Defender optimises l assuming a GREEDY attacker (this is timed).
        ell_star, stop_reason, iters, opt_time, regret = optimise_greedy_defender(
            info["structure"], info["probs"], names, args.rho, args.samples,
            args.seed, args.max_iters, args.epsilon, args.obj_tol, args.patience,
            args.min_iters, args.log_every,
            label=f"net {info['network_id']}", progress_every=args.progress_every)

        # 2) The GITTINS attacker best-responds to the fixed l* -> minimax value.
        minimax = index_value(defender_tree, ell_star, args.rho)

        # 3) What the defender THOUGHT it would get (fresh, larger MC estimate).
        eval_rng = random.Random(f"{args.seed}:eval")
        greedy_val, _ = greedy_gradient(info["structure"], info["probs"], ell_star,
                                        args.rho, args.eval_samples, eval_rng, names)
        value_profile.cache_clear()

        final_ell = {v: round(ell_star[v], 6) for v in names}
        final_indices = indices_by_control(
            info["structure"], info["probs"], ell_star, args.rho, names)

        writer.writerow({
            "network_id": info["network_id"], "n_controls": info["n_controls"],
            "total_Q": info["total_Q"], "rho": args.rho, "samples": args.samples,
            "converged": stop_reason in ("regret", "plateau"),
            "stop_reason": stop_reason, "iters": iters,
            "defender_opt_time_s": f"{opt_time:.6f}",
            "final_regret": f"{regret:.8f}",
            "greedy_value_est": f"{greedy_val:.8f}",
            "minimax_value": f"{minimax:.8f}",
            "regret_gap": f"{minimax - greedy_val:.8f}",
            "final_ell": json.dumps(final_ell),
            "final_indices": json.dumps(final_indices),
        })
        out_file.flush()
        print(f"  -> stop=[{stop_reason}] iters={iters} opt_time={opt_time:.3f}s "
              f"greedy_val~{greedy_val:.6f}  minimax(Gittins)={minimax:.6f}  "
              f"gap={minimax - greedy_val:+.6f}", flush=True)

    out_file.close()
    print(f"\nwrote {args.out}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True,
                   help="input networks CSV (columns: network_id, structure, "
                        "probs, [n_controls, total_Q]); the stored ell is ignored "
                        "-- the defender optimises l from the uniform start")
    p.add_argument("--rho", type=float, default=0.9, help="discount rate rho (> 0)")
    p.add_argument("--out", default=os.path.join(_HERE, "greedy_def_gittins_attacker.csv"),
                   help="output CSV path")
    p.add_argument("--samples", type=int, default=300,
                   help="greedy rollouts per stochastic gradient")
    p.add_argument("--eval-samples", type=int, default=20000,
                   help="rollouts for the final V_greedy(l*) estimate (untimed)")
    p.add_argument("--epsilon", type=float, default=1e-3,
                   help="regret convergence threshold: avg external regret < this")
    p.add_argument("--obj-tol", type=float, default=1e-4,
                   help="allocation-plateau tolerance: stop when the per-control "
                        "spread of l_bar over the last --patience points < this")
    p.add_argument("--patience", type=int, default=50,
                   help="monitored points l_bar must stay within --obj-tol to stop "
                        "(0 disables the plateau rule)")
    p.add_argument("--min-iters", type=int, default=2000,
                   help="earliest iteration a plateau stop is allowed")
    p.add_argument("--max-iters", type=int, default=500000, help="iteration cap")
    p.add_argument("--log-every", type=int, default=10, help="monitor cadence (iters)")
    p.add_argument("--progress-every", type=int, default=2000,
                   help="print a progress line every N iterations (0 = silent)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for rollouts")
    p.add_argument("--start-id", type=int, default=0,
                   help="only networks with network_id >= this")
    p.add_argument("--end-id", type=int, default=None,
                   help="optional last network_id (default: the last)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap to the first N networks after the id filter (0 = no cap)")
    p.add_argument("--append", action="store_true",
                   help="append to an existing --out (header written only if new)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
