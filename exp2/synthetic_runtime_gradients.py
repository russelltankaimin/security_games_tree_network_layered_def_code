"""
synthetic_runtime_gradients.py
==============================

Compare the convergence of defender regret matching under two gradient oracles:

    (1) EXACT gradient      -- sp_gradients2.value_and_gradient (the O(Q^2) fold;
                               exact dV*/dl at the attacker's optimal response).
    (2) STOCHASTIC gradient -- a Monte-Carlo estimate of the same dV*/dl from
                               rollouts of the optimal (Gittins index) attacker.

Both drive the SAME regret-matching descent of the defender's convex value V*(l)
over the simplex {l >= 0, sum l = 1}, starting from the UNIFORM allocation.

Stochastic gradient estimator
-----------------------------
With beta_v = exp(-rho * l_v) and realised reward R = prod_v beta_v^{n_v} on a
winning rollout (0 on a loss), Danskin's theorem at the optimal policy pi* gives

    dV*/dl_v = -rho * beta_v * E_{pi*}[ n_v * R / beta_v ] = -rho * E_{pi*}[ n_v * R ],

where n_v is the number of attempts on control v in a rollout. We estimate it by
averaging n_v * R over `--samples` rollouts of the index attacker. This is an
unbiased estimate of exactly what (1) computes.

Convergence, regret, termination
--------------------------------
Each iteration t we log, for the running-average iterate l_bar_t:
    * rm_regret -- the Hart-Mas-Colell average external regret max_v R_v / t
                   (the regret-matching certificate), and
    * opt_gap   -- the true optimality gap V*(l_bar_t) - V*_opt (for reference).
CONVERGENCE CRITERION (both methods, same threshold): the average external
regret falls below --epsilon,  max_v R_v / t < epsilon. V* evaluations are cheap
(O(Q^2)); V*_opt (the best value the EXACT run attains) is logged only to report
opt_gap. Only the gradient computation + regret-matching update are timed; the
monitoring V* evaluations are untimed.

Networks come from exp1/synth_nets_random_ell.csv (structure + probs are used;
the stored ell is ignored -- l is initialised uniform).

Output
------
  <out>_iterations.csv : per-logged-iteration rows
        network_id, method, iteration, opt_gap, rm_regret, objective, cum_runtime_s
  <out>_summary.csv    : one row per (network, method)
        network_id, n_controls, total_Q, method, converged, iters_to_converge,
        runtime_to_converge_s, total_iters, total_runtime_s, final_opt_gap,
        v_opt, epsilon, samples
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
    IndexAttacker,
    Par as AttackerPar,
    Series as AttackerSeries,
    index_table,
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
# Gradient oracles (both return dV*/dl as {name: value})
# ===========================================================================


def exact_gradient(defender_tree, ell, rho):
    """Exact value V*(l) and gradient dV*/dl via the O(Q^2) fold."""
    return value_and_gradient(defender_tree, allocation=ell, discount_rate=rho)


def stochastic_gradient(structure, probs, ell, rho, samples, rng, names):
    """Monte-Carlo estimate of dV*/dl from `samples` optimal-attacker rollouts.

    dV*/dl_v = -rho * E[n_v * R]  (n_v = attempts on v, R = realised reward).
    Also returns the MC estimate of V*(l) = E[R]. Building the index attacker is
    part of the cost (it needs the optimal policy)."""
    net = build_attacker_net(structure, probs, ell, rho)
    policy = IndexAttacker(net)
    sum_nvR = {name: 0.0 for name in names}
    total_R = 0.0
    for _ in range(samples):
        roll = policy.simulate(rng)
        R = roll.reward
        if R > 0.0:
            total_R += R
            for name, _k, _succ in roll.history:
                sum_nvR[name] += R          # summed over occurrences => n_v * R
    grad = {name: -rho * sum_nvR[name] / samples for name in names}
    return total_R / samples, grad


def exact_value(defender_tree, ell, rho):
    """V*(l) only (skip the tie-breaking gradient pass -- faster, untimed)."""
    value, _ = value_and_gradient(defender_tree, allocation=ell,
                                  discount_rate=rho, break_ties=False)
    return value


# ===========================================================================
# Regret matching with either oracle
# ===========================================================================


def run_regret_matching(method, structure, probs, names, defender_tree, rho,
                        max_iters, samples, seed, log_every, label="", progress_every=0,
                        epsilon=None, v_ref=None):
    """Run RM from the uniform allocation. Returns a trajectory of monitoring
    points (iteration, v_avg, rm_regret, cum_runtime, ell_avg) plus totals.

    Only the gradient + update are timed; monitoring V* evals are untimed.
    Prints a progress line every `progress_every` iterations (0 = silent),
    including the optimality gap vs `v_ref` (or vs the running-best V* if None),
    and announces the first iteration whose gap falls to <= `epsilon`."""
    n = len(names)
    rng = random.Random(f"{seed}:{method}")
    ell = {v: 1.0 / n for v in names}          # uniform initialisation
    cumulative_regret = {v: 0.0 for v in names}
    ell_sum = {v: 0.0 for v in names}
    cum_time = 0.0
    trajectory = []
    v0 = None                                  # initial (uniform) objective

    for t in range(1, max_iters + 1):
        for v in names:                        # accumulate the played iterate
            ell_sum[v] += ell[v]

        start = time.perf_counter()
        if method == "exact":
            _value, grad = exact_gradient(defender_tree, ell, rho)
        else:
            _value, grad = stochastic_gradient(structure, probs, ell, rho, samples, rng, names)
        expected_loss = sum(ell[v] * grad[v] for v in names)
        for v in names:
            cumulative_regret[v] += expected_loss - grad[v]
        positive = {v: max(cumulative_regret[v], 0.0) for v in names}
        total = sum(positive.values())
        ell = ({v: positive[v] / total for v in names} if total > 0.0
               else {v: 1.0 / n for v in names})
        cum_time += time.perf_counter() - start

        # Average external regret (cheap) drives the convergence test every iter.
        rm_regret = max(0.0, max(cumulative_regret.values())) / t
        converged = epsilon is not None and rm_regret < epsilon

        is_log = (t == 1 or t % log_every == 0 or t == max_iters or converged)
        is_progress = progress_every and (
            t == 1 or t % progress_every == 0 or t == max_iters or converged)
        if is_log or is_progress:
            ell_avg = {v: ell_sum[v] / t for v in names}
            v_avg = exact_value(defender_tree, ell_avg, rho)     # untimed monitor
            if v0 is None:
                v0 = v_avg
            gap = None if v_ref is None else v_avg - v_ref        # gap vs v_opt
            if is_log:
                trajectory.append((t, v_avg, rm_regret, cum_time, ell_avg))
            if is_progress and not converged:
                gap_str = f"gap={gap:+.2e}  " if gap is not None else ""
                print(f"    [{label}] iter {t:>6}/{max_iters}  V*={v_avg:.6f}  "
                      f"{gap_str}regret={rm_regret:.3e}  drop={v0 - v_avg:.4f}  "
                      f"{cum_time:6.1f}s", flush=True)

        if converged:                          # TERMINATE on the criterion
            print(f"    [{label}] *** converged: avg regret < {epsilon:g} at iter {t} "
                  f"({cum_time:.2f}s) ***", flush=True)
            break

    return trajectory, t, cum_time


def indices_by_control(structure, probs, ell, rho, names):
    """Gittins index table alpha(v, k) at allocation `ell`, grouped by control."""
    table = index_table(build_attacker_net(structure, probs, ell, rho))
    grouped = {}
    for name in names:
        ks = sorted(k for (nm, k) in table if nm == name)
        grouped[name] = [round(table[(name, k)], 6) for k in ks]
    return grouped


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


ITER_FIELDS = ["network_id", "method", "iteration", "opt_gap", "rm_regret",
               "objective", "cum_runtime_s"]
SUMMARY_FIELDS = ["network_id", "n_controls", "total_Q", "method", "converged",
                  "iters_to_converge", "runtime_to_converge_s", "avg_regret_at_converge",
                  "total_iters", "total_runtime_s", "converged_minmax_value",
                  "final_opt_gap", "v_opt", "epsilon", "samples",
                  "final_ell", "final_indices"]


def summarise(trajectory, v_opt, epsilon):
    """Find the first monitored point with average regret < epsilon; also return
    the min-max value V*(l_bar) and allocation there (or the last point if never
    converged)."""
    converged, iters_c, runtime_c = False, None, None
    v_at, ell_at, regret_at = trajectory[-1][1], trajectory[-1][4], trajectory[-1][2]
    for (t, v_avg, rm_regret, cum_time, ell_avg) in trajectory:
        if rm_regret < epsilon:
            converged, iters_c, runtime_c = True, t, cum_time
            v_at, ell_at, regret_at = v_avg, ell_avg, rm_regret
            break
    final_gap = trajectory[-1][1] - v_opt
    return converged, iters_c, runtime_c, final_gap, v_at, ell_at, regret_at


def run(args):
    networks = load_networks(args.csv)
    if args.limit:
        networks = networks[: args.limit]
    methods = (["exact", "stochastic"] if args.methods == "both" else [args.methods])

    iter_path = args.out + "_iterations.csv"
    summary_path = args.out + "_summary.csv"
    iter_file = open(iter_path, "w", newline="")
    iter_writer = csv.DictWriter(iter_file, fieldnames=ITER_FIELDS)
    iter_writer.writeheader()
    summary_file = open(summary_path, "w", newline="")
    summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
    summary_writer.writeheader()
    summary_file.flush()

    print(f"{len(networks)} networks | methods={methods} eps={args.epsilon} "
          f"max_iters={args.max_iters} samples={args.samples} rho={args.rho}\n", flush=True)

    for idx, info in enumerate(networks, 1):
        if info["probs"] is None:
            raise ValueError(f"network {info['network_id']}: needs a probs column")
        names = leaf_names(info["structure"])
        defender_tree = build_defender_tree(info["structure"], info["probs"])
        print(f"=== [{idx}/{len(networks)}] network {info['network_id']}: "
              f"n={info['n_controls']} Q={info['total_Q']} ===", flush=True)

        # EXACT run first -- also fixes the reference optimum V*_opt.
        traj = {}
        print(f"  running exact ...", flush=True)
        traj["exact"] = run_regret_matching(
            "exact", info["structure"], info["probs"], names, defender_tree,
            args.rho, args.max_iters, args.samples, args.seed, args.log_every,
            label=f"net {info['network_id']} exact", progress_every=args.progress_every,
            epsilon=args.epsilon, v_ref=None)
        v_opt = min(pt[1] for pt in traj["exact"][0])

        if "stochastic" in methods:
            print(f"  running stochastic ({args.samples} rollouts/grad), v_opt={v_opt:.6f} ...",
                  flush=True)
            traj["stochastic"] = run_regret_matching(
                "stochastic", info["structure"], info["probs"], names, defender_tree,
                args.rho, args.max_iters, args.samples, args.seed, args.log_every,
                label=f"net {info['network_id']} stoch", progress_every=args.progress_every,
                epsilon=args.epsilon, v_ref=v_opt)

        for method in methods:
            trajectory, total_iters, total_time = traj[method]
            converged, iters_c, runtime_c, final_gap, v_at, ell_at, regret_at = summarise(
                trajectory, v_opt, args.epsilon)

            for (t, v_avg, rm_regret, cum_time, _ell) in trajectory:
                iter_writer.writerow({
                    "network_id": info["network_id"], "method": method, "iteration": t,
                    "opt_gap": f"{v_avg - v_opt:.8f}", "rm_regret": f"{rm_regret:.8f}",
                    "objective": f"{v_avg:.8f}", "cum_runtime_s": f"{cum_time:.6f}",
                })
            iter_file.flush()

            # Final allocation at convergence + the index table it induces.
            final_ell = {v: round(ell_at[v], 6) for v in names}
            final_indices = indices_by_control(
                info["structure"], info["probs"], ell_at, args.rho, names)

            summary_writer.writerow({
                "network_id": info["network_id"], "n_controls": info["n_controls"],
                "total_Q": info["total_Q"], "method": method,
                "converged": converged,
                "iters_to_converge": iters_c if iters_c is not None else "",
                "runtime_to_converge_s": f"{runtime_c:.6f}" if runtime_c is not None else "",
                "avg_regret_at_converge": f"{regret_at:.8f}",
                "total_iters": total_iters, "total_runtime_s": f"{total_time:.6f}",
                "converged_minmax_value": f"{v_at:.8f}",
                "final_opt_gap": f"{final_gap:.8f}", "v_opt": f"{v_opt:.8f}",
                "epsilon": args.epsilon,
                "samples": args.samples if method == "stochastic" else "",
                "final_ell": json.dumps(final_ell),
                "final_indices": json.dumps(final_indices),
            })
            summary_file.flush()                 # persist per (network, method)
            iters_disp = iters_c if iters_c else total_iters
            rt_disp = runtime_c if runtime_c is not None else total_time
            print(f"  -> {method:>10}: converged={str(converged):>5} "
                  f"iters={iters_disp:>6} runtime={rt_disp:8.3f}s "
                  f"avg_regret={regret_at:.3e} minmax={v_at:.6f} "
                  f"final_gap={final_gap:.2e}", flush=True)

    iter_file.close()
    summary_file.close()
    print(f"\nwrote {iter_path}\nwrote {summary_path}")


def build_arg_parser():
    default_csv = os.path.join(_ROOT, "exp1", "synth_nets_random_ell.csv")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=default_csv, help="networks (structure + probs)")
    p.add_argument("--out", default=os.path.join(_HERE, "gradient_convergence"),
                   help="output base path (writes <out>_iterations.csv and <out>_summary.csv)")
    p.add_argument("--methods", choices=["exact", "stochastic", "both"], default="both")
    p.add_argument("--epsilon", type=float, default=1e-2,
                   help="convergence threshold: average external regret < epsilon")
    p.add_argument("--max-iters", type=int, default=100000, help="iteration cap per run")
    p.add_argument("--samples", type=int, default=300, help="rollouts per stochastic gradient")
    p.add_argument("--rho", type=float, default=0.9, help="discount rate rho (> 0)")
    p.add_argument("--log-every", type=int, default=10, help="monitor/log cadence (iterations)")
    p.add_argument("--progress-every", type=int, default=500,
                   help="print an in-run progress line every N iterations (0 = silent)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for stochastic rollouts")
    p.add_argument("--limit", type=int, default=0, help="only the first N networks (0 = all)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
