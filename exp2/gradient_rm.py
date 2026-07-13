"""
gradient_rm.py
==============

Regret-matching optimisation of the defender allocation l under two gradient
oracles, at a FIXED stochastic batch size b in {1, 10, 100}. Both arms minimise
the SAME objective -- the optimal (Gittins) attacker's value V*(l) -- and the
objective is always evaluated EXACTLY (deterministic n-ary fold), so the two
arms' "curves" are directly comparable.

Protocol
--------
Let T = --max-iters (default 100000) be the per-run iteration cap, and log every
--log-every (default 50) iterations. At iteration t we hold the running-average
iterate  l_t = (1/t) * sum_{s<=t} l_s  and log (l_t, V*(l_t)).

  EXACT arm: the gradient is the exact O(Q^2) fold. We run a FIXED number of
    iterations --exact-iters (default 5000; NOT until convergence), then record
    the objective V*(l_t) at that iteration (the TARGET) and the allocation
    l_{exact-iters}. Because the run is deterministic, we repeat it --exact-runs
    times (default 10) purely to AVERAGE the wall-clock (reporting mean and std);
    the iterate/objective are identical across repeats.

  STOCHASTIC arm (--runs independent runs, default 10): the gradient is the
    pathwise Monte-Carlo estimate from `b` rollouts of the optimal attacker. Each
    run logs V*(l_t) every --log-every iterations; averaging across runs gives the
    stochastic "average curve". We report how many MORE iterations and how much
    MORE wall-clock the stochastic arm needs to first fall AT OR BELOW the exact
    arm's target objective (it need not exactly match it, and may never reach it
    within T -- in which case the run is marked not-reached).

Timing (symmetric and fair). In BOTH arms we time only the essential optimisation
work per iteration: the running-average update, the gradient, and the RM step. The
exact objective evaluations V*(l_t) are UNTIMED in both arms -- they are the shared
measurement instrument (the same O(Q^2) fold) used to draw the curves and locate
the target crossing, not part of either method. (Charging the stochastic arm for
an exact-fold objective check would be unfair: it would make it pay for the very
exact computation it exists to avoid; a real stochastic method would detect
convergence with a cheap self-estimate, not the exact fold.) The target crossing
is located every --check-every iterations at the untimed objective. All folds and
attacker rollouts use the native n-ary implementations in spn_defender/spn_attacker.

Outputs
-------
  <out>_summary.csv : one row per network (exact stop + stochastic-to-target).
  <out>_curves.csv  : network_id, t, exact_obj, stoch_mean_obj, stoch_runs.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _pkg in ("spn_attacker", "spn_defender"):
    p = os.path.join(_ROOT, _pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from sp_attacker import (  # noqa: E402
    Control as AControl, Series as ASeries, Par as APar,
    IndexAttacker, value_profile,
)
from sp_gradients2 import (  # noqa: E402
    Control as DControl, Series as DSeries, Parallel as DParallel,
    value_and_gradient,
)


# ===========================================================================
# Structure parsing -> generic nested form -> n-ary trees
# ===========================================================================
def parse_generic(text):
    """Parse 'Ser(A,B)'/'Par(A,B)'/'name' into ('c',name)|('S',[..])|('P',[..])."""
    pos = 0

    def parse():
        nonlocal pos
        head = text[pos:pos + 4]
        if head in ("Ser(", "Par("):
            op = "S" if head[:3] == "Ser" else "P"
            pos += 4
            left = parse(); assert text[pos] == ","; pos += 1
            right = parse(); assert text[pos] == ")"; pos += 1
            return (op, [left, right])
        start = pos
        while pos < len(text) and text[pos] not in ",)":
            pos += 1
        return ("c", text[start:pos])

    tree = parse()
    assert pos == len(text), f"trailing input: {text[pos:]!r}"
    return tree


def flatten_generic(node):
    """Collapse nested same-op nodes into n-ary Series/Par (associativity)."""
    if node[0] == "c":
        return node
    op, kids = node
    merged = []
    for c in (flatten_generic(k) for k in kids):
        if c[0] == op:
            merged.extend(c[1])
        else:
            merged.append(c)
    return (op, merged)


def leaf_names(node):
    if node[0] == "c":
        return [node[1]]
    out = []
    for c in node[1]:
        out += leaf_names(c)
    return out


def build_defender(node, probs):
    """sp_gradients2 n-ary tree (allocation-agnostic)."""
    if node[0] == "c":
        name = node[1]
        return DControl(name, len(probs[name]), list(probs[name]))
    children = [build_defender(c, probs) for c in node[1]]
    return DSeries(*children) if node[0] == "S" else DParallel(*children)


def build_attacker(node, probs, ell, rho):
    """sp_attacker n-ary net with beta_v = exp(-rho * l_v)."""
    if node[0] == "c":
        name = node[1]
        return AControl(name, exp(-rho * ell[name]), tuple(probs[name]))
    children = [build_attacker(c, probs, ell, rho) for c in node[1]]
    return ASeries(*children) if node[0] == "S" else APar(*children)


# ===========================================================================
# Oracles (objective is always the EXACT n-ary fold)
# ===========================================================================
def exact_gradient(tree, ell, rho):
    """Exact value and gradient dV*/dl via the native n-ary fold."""
    return value_and_gradient(tree, allocation=ell, discount_rate=rho, native=True)


def exact_value(tree, ell, rho):
    """V*(l) only (no tie-breaking pass); the monitored objective."""
    value, _ = value_and_gradient(tree, allocation=ell, discount_rate=rho,
                                  break_ties=False, native=True)
    return value


def stochastic_gradient(node, probs, ell, rho, batch, rng, names):
    """MC estimate dV*/dl_v = -rho * E[n_v * R] from `batch` optimal-attacker rollouts."""
    net = build_attacker(node, probs, ell, rho)
    policy = IndexAttacker(net)
    sum_nvR = {v: 0.0 for v in names}
    for _ in range(batch):
        roll = policy.simulate(rng)
        R = roll.reward
        if R > 0.0:
            for name, _k, _s in roll.history:
                sum_nvR[name] += R
    value_profile.cache_clear()          # fresh net each iter -> drop unbounded cache
    return {v: -rho * sum_nvR[v] / batch for v in names}


def naive_fd_gradient(node, probs, ell, rho, batch, rng, names, h):
    """The MOST PRIMITIVE stochastic gradient: forward finite differences of
    Monte-Carlo value estimates. For each control v,

        dV*/dl_v ~= ( Vhat(l + h e_v) - Vhat(l) ) / h,

    where Vhat(.) is the mean reward over `batch` optimal-attacker rollouts. This
    needs n+1 MC value estimates per gradient (vs one batch for the pathwise
    estimator), each rebuilding the attacker and its Gittins indices -- so it is
    ~n times more work per iteration, biased O(h), and much higher variance
    (a difference of two noisy estimates divided by a small h)."""
    def mc_value(alloc):
        att = IndexAttacker(build_attacker(node, probs, alloc, rho))
        total = sum(att.simulate(rng).reward for _ in range(batch))
        value_profile.cache_clear()
        return total / batch

    base = mc_value(ell)
    grad = {}
    for v in names:
        bumped = dict(ell)
        bumped[v] = ell[v] + h
        grad[v] = (mc_value(bumped) - base) / h
    return grad


# ===========================================================================
# Regret-matching update
# ===========================================================================
def rm_step(ell, grad, cumulative_regret, names):
    """One Hart--Mas-Colell regret-matching update; returns the next allocation."""
    n = len(names)
    expected_loss = sum(ell[v] * grad[v] for v in names)
    for v in names:
        cumulative_regret[v] += expected_loss - grad[v]
    positive = {v: max(cumulative_regret[v], 0.0) for v in names}
    total = sum(positive.values())
    if total > 0.0:
        return {v: positive[v] / total for v in names}
    return {v: 1.0 / n for v in names}


# ===========================================================================
# Exact arm: one run of a FIXED number of iterations; returns curve + target
# ===========================================================================
def run_exact(tree, names, rho, exact_iters, log_every, time_runs):
    """Run the (deterministic) exact-gradient RM for `exact_iters` iterations. The
    iterate/objective are identical every time, so we repeat the run `time_runs`
    times ONLY to average the wall-clock (timing jitter) and report its std. The
    curve and target allocation are taken from the first pass."""
    n = len(names)
    curve = []                                   # (t, obj) -- deterministic
    target_ell = None
    final_times = []
    for run_i in range(time_runs):
        ell = {v: 1.0 / n for v in names}
        cumulative_regret = {v: 0.0 for v in names}
        ell_sum = {v: 0.0 for v in names}
        cum_time = 0.0
        for t in range(1, exact_iters + 1):
            start = time.perf_counter()
            for v in names:
                ell_sum[v] += ell[v]
            _v, grad = exact_gradient(tree, ell, rho)
            ell = rm_step(ell, grad, cumulative_regret, names)
            cum_time += time.perf_counter() - start
            if run_i == 0 and (t == 1 or t % log_every == 0 or t == exact_iters):
                ell_bar = {v: ell_sum[v] / t for v in names}
                curve.append((t, exact_value(tree, ell_bar, rho)))   # untimed
                if t == exact_iters:
                    target_ell = {v: round(ell_bar[v], 6) for v in names}
        final_times.append(cum_time)
    mean_t = sum(final_times) / len(final_times)
    std_t = ((sum((x - mean_t) ** 2 for x in final_times) / (len(final_times) - 1)) ** 0.5
             if len(final_times) > 1 else 0.0)
    return curve, {"stop_iter": exact_iters, "time_s": mean_t, "time_std": std_t,
                   "runs": time_runs, "obj": curve[-1][1], "ell": target_ell}


# ===========================================================================
# Stochastic arm: `runs` independent runs at batch b, until each reaches `target`
# ===========================================================================
def run_stochastic(node, probs, tree, names, rho, batch, T, log_every, runs,
                   seed, target, check_every, estimator, fd_step):
    """Curve checkpoints are logged every `log_every` iterations, but the FIRST
    dip below `target` is detected every `check_every` iterations (default 1 =
    every iteration) so `iters_to_target` is not snapped to the log grid."""
    n = len(names)
    per_run = []              # list of {t: (obj, cum_time)}
    reached = []              # list of (iters_to_target, time_to_target) or None
    for r in range(runs):
        rng = random.Random(f"{seed}:{r}")
        ell = {v: 1.0 / n for v in names}
        cumulative_regret = {v: 0.0 for v in names}
        ell_sum = {v: 0.0 for v in names}
        cum_time = 0.0
        checkpoints = {}
        hit = None
        for t in range(1, T + 1):
            # TIMED: only the essential optimisation work (avg update + gradient
            # + RM step), symmetric with the exact arm.
            start = time.perf_counter()
            for v in names:
                ell_sum[v] += ell[v]
            if estimator == "fd":
                grad = naive_fd_gradient(node, probs, ell, rho, batch, rng, names, fd_step)
            else:
                grad = stochastic_gradient(node, probs, ell, rho, batch, rng, names)
            ell = rm_step(ell, grad, cumulative_regret, names)
            cum_time += time.perf_counter() - start

            # UNTIMED: objective evaluation (shared measurement instrument), used
            # for the curve and to locate the target crossing every check_every.
            is_curve = (t == 1 or t % log_every == 0 or t == T)
            is_check = (hit is None and t % check_every == 0)
            if is_curve or is_check:
                ell_bar = {v: ell_sum[v] / t for v in names}
                obj = exact_value(tree, ell_bar, rho)
                if is_curve:
                    checkpoints[t] = (obj, cum_time)
                if hit is None and is_check and obj <= target * (1.0 + 1e-9):
                    hit = (t, cum_time)
                    checkpoints[t] = (obj, cum_time)         # keep the crossing pt
                    break                                    # this run reached target
        per_run.append(checkpoints)
        reached.append(hit)
        value_profile.cache_clear()
    return per_run, reached


# ===========================================================================
# Driver
# ===========================================================================
def load_networks(csv_path):
    seen = {}
    with open(csv_path, newline="") as h:
        for row in csv.DictReader(h):
            nid = row["network_id"]
            if nid in seen:
                continue
            seen[nid] = {
                "network_id": int(nid), "structure": row["structure"],
                "n_controls": int(row.get("n_controls") or 0),
                "total_Q": int(row.get("total_Q") or 0),
                "probs": json.loads(row["probs"]) if row.get("probs") else None,
            }
    return sorted(seen.values(), key=lambda r: r["network_id"])


SUMMARY_FIELDS = ["network_id", "n_controls", "total_Q", "rho", "batch", "runs",
                  "exact_runs", "exact_iters", "max_iters",
                  "exact_time_s", "exact_time_std", "target_obj",
                  "stoch_reached", "stoch_iters_to_target", "stoch_time_to_target_s",
                  "extra_iters", "extra_time_s", "iter_ratio", "time_ratio",
                  "target_ell"]
CURVE_FIELDS = ["network_id", "t", "exact_obj", "stoch_mean_obj",
                "stoch_std_obj", "stoch_sem_obj", "stoch_runs"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _mean_std(xs):
    """Return (mean, sample std, standard error of the mean) over xs. std/sem are
    the error-bar half-widths across the stochastic runs; std=0 if fewer than 2."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None
    m = sum(xs) / len(xs)
    if len(xs) >= 2:
        std = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    else:
        std = 0.0
    return m, std, std / (len(xs) ** 0.5)


def run(args):
    nets = load_networks(args.csv)
    nets = [nw for nw in nets if nw["network_id"] >= args.start_id]
    if args.end_id is not None:
        nets = [nw for nw in nets if nw["network_id"] <= args.end_id]
    if args.limit:
        nets = nets[: args.limit]
    if not nets:
        raise SystemExit("no networks in range")

    sfile = open(args.out + "_summary.csv", "w", newline="")
    swr = csv.DictWriter(sfile, fieldnames=SUMMARY_FIELDS); swr.writeheader()
    cfile = open(args.out + "_curves.csv", "w", newline="")
    cwr = csv.DictWriter(cfile, fieldnames=CURVE_FIELDS); cwr.writeheader()

    print(f"{len(nets)} networks | batch={args.batch} runs={args.runs} "
          f"exact_iters={args.exact_iters} T={args.max_iters} "
          f"log_every={args.log_every} rho={args.rho}\n", flush=True)

    for idx, info in enumerate(nets, 1):
        if info["probs"] is None:
            raise ValueError(f"network {info['network_id']}: needs probs")
        node = flatten_generic(parse_generic(info["structure"]))
        names = leaf_names(node)
        tree = build_defender(node, info["probs"])
        print(f"=== [{idx}/{len(nets)}] network {info['network_id']}: "
              f"n={info['n_controls']} Q={info['total_Q']} ===", flush=True)

        # EXACT arm -> fixed-iteration target objective + allocation (wall-clock
        # averaged over --exact-runs deterministic repeats).
        exact_curve, tgt = run_exact(
            tree, names, args.rho, args.exact_iters, args.log_every, args.exact_runs)
        target = tgt["obj"]
        print(f"  exact: iter {tgt['stop_iter']}  time={tgt['time_s']:.3f}"
              f"+-{tgt['time_std']:.3f}s (avg of {tgt['runs']})  "
              f"target V*={target:.6f}", flush=True)

        # STOCHASTIC arm
        per_run, reached = run_stochastic(
            node, info["probs"], tree, names, args.rho, args.batch, args.max_iters,
            args.log_every, args.runs, args.seed, target, args.check_every,
            args.estimator, args.fd_step)
        iters_to = _mean([r[0] for r in reached if r is not None])
        time_to = _mean([r[1] for r in reached if r is not None])
        n_reached = sum(1 for r in reached if r is not None)

        extra_iters = (iters_to - tgt["stop_iter"]) if iters_to is not None else None
        extra_time = (time_to - tgt["time_s"]) if time_to is not None else None
        iter_ratio = (iters_to / tgt["stop_iter"]) if iters_to else None
        time_ratio = (time_to / tgt["time_s"]) if (time_to and tgt["time_s"] > 0) else None

        swr.writerow({
            "network_id": info["network_id"], "n_controls": info["n_controls"],
            "total_Q": info["total_Q"], "rho": args.rho, "batch": args.batch,
            "runs": args.runs, "exact_runs": tgt["runs"],
            "exact_iters": tgt["stop_iter"], "max_iters": args.max_iters,
            "exact_time_s": f"{tgt['time_s']:.6f}", "exact_time_std": f"{tgt['time_std']:.6f}",
            "target_obj": f"{target:.8f}", "stoch_reached": f"{n_reached}/{args.runs}",
            "stoch_iters_to_target": f"{iters_to:.1f}" if iters_to else "",
            "stoch_time_to_target_s": f"{time_to:.6f}" if time_to else "",
            "extra_iters": f"{extra_iters:.1f}" if extra_iters is not None else "",
            "extra_time_s": f"{extra_time:.6f}" if extra_time is not None else "",
            "iter_ratio": f"{iter_ratio:.3f}" if iter_ratio else "",
            "time_ratio": f"{time_ratio:.3f}" if time_ratio else "",
            "target_ell": json.dumps(tgt["ell"]),
        })
        sfile.flush()

        # curves: align exact + stochastic-average on the checkpoint grid
        exact_by_t = {t: obj for t, obj in exact_curve}
        all_t = set(exact_by_t)
        for cp in per_run:
            all_t |= set(cp)
        for t in sorted(all_t):
            stoch_objs = [cp[t][0] for cp in per_run if t in cp]
            m, std, sem = _mean_std(stoch_objs)
            cwr.writerow({
                "network_id": info["network_id"], "t": t,
                "exact_obj": f"{exact_by_t[t]:.8f}" if t in exact_by_t else "",
                "stoch_mean_obj": f"{m:.8f}" if m is not None else "",
                "stoch_std_obj": f"{std:.8f}" if std is not None else "",
                "stoch_sem_obj": f"{sem:.8f}" if sem is not None else "",
                "stoch_runs": len(stoch_objs),
            })
        cfile.flush()

        rep_extra = f"{extra_iters:.0f}" if extra_iters is not None else "n/a"
        print(f"  stoch(b={args.batch}): reached {n_reached}/{args.runs}  "
              f"iters_to_target={iters_to and round(iters_to)}  "
              f"time={time_to and round(time_to, 3)}s  extra_iters={rep_extra}", flush=True)

    sfile.close(); cfile.close()
    print(f"\nwrote {args.out}_summary.csv and {args.out}_curves.csv")


def build_arg_parser():
    default_csv = os.path.join(_ROOT, "exp1", "synth_nets_random_ell.csv")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=default_csv, help="networks (structure + probs)")
    p.add_argument("--out", default=os.path.join(_HERE, "gradient_rm"), help="output base path")
    p.add_argument("--batch", type=int, default=10,
                   help="stochastic gradient batch size b (rollouts per MC value estimate)")
    p.add_argument("--estimator", choices=["pathwise", "fd"], default="pathwise",
                   help="pathwise = -rho*E[n_v R] (smart); fd = naive forward "
                        "finite differences of MC values (primitive, O(n) per gradient)")
    p.add_argument("--fd-step", type=float, default=1e-2,
                   help="finite-difference step h (only for --estimator fd)")
    p.add_argument("--runs", type=int, default=10, help="independent stochastic runs to average")
    p.add_argument("--exact-runs", type=int, default=10,
                   help="repeats of the (deterministic) exact run to average its wall-clock")
    p.add_argument("--exact-iters", type=int, default=5000,
                   help="fixed number of exact-arm iterations defining the target")
    p.add_argument("--max-iters", type=int, default=100000, help="T: stochastic iteration cap per run")
    p.add_argument("--log-every", type=int, default=50, help="curve checkpoint cadence")
    p.add_argument("--check-every", type=int, default=1,
                   help="cadence (iters) for detecting the stochastic arm's first "
                        "dip below target; 1 = every iteration (precise), larger = coarser/faster")
    p.add_argument("--rho", type=float, default=5.0, help="discount rate rho/lambda (> 0)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for stochastic rollouts")
    p.add_argument("--start-id", type=int, default=0)
    p.add_argument("--end-id", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
