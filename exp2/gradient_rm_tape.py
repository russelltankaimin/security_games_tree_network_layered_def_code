"""
gradient_rm_tape.py
===================

Run ONE gradient oracle's regret-matching descent of the defender value V*(l) on
selected instances -- unlike gradient_rm.py, which runs the exact vs stochastic
comparison, this runs a SINGLE chosen method individually and reports its own
convergence and wall-clock.

Methods (--method)
------------------
  stochastic  Monte-Carlo pathwise gradient from `--batch` optimal-attacker
              rollouts (sp_attacker); `--runs` independent runs, averaged.
  fold        exact gradient via sp_gradients2 (weight-list fold, O(Q^2) worst case).
  tape        exact gradient via sp_gradients3 binary reverse-tape (O(Q d_b)).
  mary        exact gradient via sp_gradients3 direct m-ary (O(sum_G Q_G log m_G)).
  par-tape    `tape` spread across `--cores` processes (sp_par_gradients subtree cut).
  par-mary    `mary` spread across `--cores` processes (sp_par_gradients subtree cut).
  par-fold    `fold` spread across `--cores` processes (sp_par_gradients_fold subtree cut).

The `par-*` methods give a bit-identical gradient to their serial counterparts and
help most on series_heavy (deep/narrow) nets; on small/wide instances they fall
back to serial internally. The persistent worker pool amortises process spawn over
the RM iterations.

Instance selection
------------------
`--csv` chooses the network file; `--start-id` / `--end-id` (inclusive, matched on
the CSV `network_id`) and `--limit` pick which instances to run.

Protocol
--------
RM descent from the uniform allocation for `--iters` iterations, monitoring the
EXACT objective V*(l_bar_t) of the (polynomially-weighted, --avg-power) running
average every `--log-every` iterations. Only the essential optimisation work
(running-average update + gradient + RM step) is timed; the exact objective checks
are UNTIMED (a shared measurement instrument, --monitor-impl). Deterministic
methods (fold/tape/mary) run once; `stochastic` runs `--runs` times and averages.

Outputs
-------
  <out>_summary.csv : one row per instance (method, iters, wall-clock, final V*).
  <out>_curves.csv  : network_id, t, obj_mean, obj_std, runs.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gradient_rm as G                              # shared helpers  # noqa: E402
import sp_gradients2 as _G2                          # noqa: E402
import sp_gradients3 as _G3                          # noqa: E402
import sp_par_gradients as _GP                       # noqa: E402
import sp_par_gradients_fold as _GPF                  # noqa: E402


# ===========================================================================
# Gradient oracles and the (exact) objective monitor
# ===========================================================================
def opt_gradient(method, tree, node, probs, ell, rho, batch, rng, names, plan=None):
    """The gradient dV*/dl used to drive RM, per the chosen method."""
    if method == "stochastic":
        return G.stochastic_gradient(node, probs, ell, rho, batch, rng, names)
    if method == "tape":
        _v, grad = _G3.value_and_gradient(tree, ell, rho, method="binary")
        return grad
    if method == "mary":
        _v, grad = _G3.value_and_gradient(tree, ell, rho, method="mary")
        return grad
    if method in ("par-tape", "par-mary", "par-fold"):  # reuse the resident CutPlan
        _v, grad = plan.value_and_gradient(ell, rho)
        return grad
    _v, grad = _G2.value_and_gradient(tree, allocation=ell, discount_rate=rho, native=True)
    return grad                                       # fold


def monitor_value(monitor_impl, tree, ell, rho):
    """Exact V*(l) -- the UNTIMED objective, measured the same way for every method."""
    if monitor_impl == "fold":
        v, _ = _G2.value_and_gradient(tree, allocation=ell, discount_rate=rho,
                                      break_ties=False, native=True)
        return v
    v, _ = _G3.value_and_gradient(tree, ell, rho,
                                  method=("mary" if monitor_impl == "mary" else "binary"))
    return v


# ===========================================================================
# One RM run of the chosen method
# ===========================================================================
def rm_run(method, node, probs, tree, names, rho, batch, iters, log_every,
           avg_power, monitor_impl, run_seed, plan=None, target_v=None):
    """RM descent. If `target_v` is set, run until the (untimed, exact) monitored
    objective V*(l_bar) <= target_v (no iteration cap). Otherwise run exactly
    `iters` iterations. Returns (curve, cum_time, hit_iter): the iteration the
    target was reached, or None when there is no target."""
    n = len(names)
    rng = random.Random(f"{run_seed}")
    ell = {v: 1.0 / n for v in names}
    cumulative_regret = {v: 0.0 for v in names}
    ell_sum = {v: 0.0 for v in names}
    w_total = 0.0
    cum_time = 0.0
    curve = {}                                        # t -> (obj, cum_time)
    hit_iter = None
    t = 0
    while True:
        t += 1
        start = time.perf_counter()                   # TIMED: essential work only
        w = G.avg_weight(t, avg_power)
        for v in names:
            ell_sum[v] += w * ell[v]
        w_total += w
        grad = opt_gradient(method, tree, node, probs, ell, rho, batch, rng, names, plan)
        ell = G.rm_step(ell, grad, cumulative_regret, names)
        cum_time += time.perf_counter() - start
        if t == 1 or t % log_every == 0 or (target_v is None and t == iters):
            ell_bar = {v: ell_sum[v] / w_total for v in names}
            obj = monitor_value(monitor_impl, tree, ell_bar, rho)          # UNTIMED
            curve[t] = (obj, cum_time)
            if target_v is not None and obj <= target_v:                   # reached target
                hit_iter = t
                break
        if target_v is None and t == iters:           # no target: stop at --iters
            break
    G.value_profile.cache_clear()
    return curve, cum_time, hit_iter


# ===========================================================================
# Driver
# ===========================================================================
SUMMARY_FIELDS = ["network_id", "n_controls", "total_Q", "method", "batch", "runs",
                  "iters", "avg_power", "rho", "wall_time_s", "wall_time_std",
                  "final_obj", "final_obj_std",
                  "target_v", "reached", "hit_iter_mean", "hit_iter_std", "hit_wall_mean"]
CURVE_FIELDS = ["network_id", "t", "obj_mean", "obj_std", "runs"]


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, 0.0
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def run(args):
    if args.method in ("par-tape", "par-mary"):
        _GP.set_cores(args.cores)                     # the global core cap
    nets = G.load_networks(args.csv)
    nets = [nw for nw in nets if nw["network_id"] >= args.start_id]
    if args.end_id is not None:
        nets = [nw for nw in nets if nw["network_id"] <= args.end_id]
    if args.limit:
        nets = nets[: args.limit]
    if not nets:
        raise SystemExit("no networks in range")

    n_runs = args.runs if args.method == "stochastic" else 1

    sfile = open(args.out + "_summary.csv", "w", newline="")
    swr = csv.DictWriter(sfile, fieldnames=SUMMARY_FIELDS); swr.writeheader()
    cfile = open(args.out + "_curves.csv", "w", newline="")
    cwr = csv.DictWriter(cfile, fieldnames=CURVE_FIELDS); cwr.writeheader()

    print(f"{len(nets)} networks | method={args.method} "
          f"{'batch=' + str(args.batch) + ' runs=' + str(n_runs) if args.method == 'stochastic' else ''}"
          f"{' cores=' + str(args.cores) if args.method in ('par-tape', 'par-mary', 'par-fold') else ''} "
          f"{'target_v=' + str(args.target_v) + ' (until reached)' if args.target_v is not None else 'iters=' + str(args.iters)} "
          f"rho={args.rho} avg_power={args.avg_power} "
          f"monitor={args.monitor_impl}\n", flush=True)

    for idx, info in enumerate(nets, 1):
        if info["probs"] is None:
            raise ValueError(f"network {info['network_id']}: needs probs")
        node = G.flatten_generic(G.parse_generic(info["structure"]))
        names = G.leaf_names(node)
        tree = G.build_defender(node, info["probs"])
        print(f"=== [{idx}/{len(nets)}] network {info['network_id']}: "
              f"n={info['n_controls']} Q={info['total_Q']} ===", flush=True)

        # One resident CutPlan per network (partition + subtrees shipped once);
        # reused across every RM iteration so only allocation/adjoints move.
        plan = None
        if args.method in ("par-tape", "par-mary"):
            plan = _GP.make_plan(tree, method=("binary" if args.method == "par-tape" else "mary"),
                                 cores=args.cores)
        elif args.method == "par-fold":
            plan = _GPF.make_plan(tree, cores=args.cores)

        per_run, times, hits = [], [], []
        for r in range(n_runs):
            curve, wall, hit = rm_run(args.method, node, info["probs"], tree, names,
                                      args.rho, args.batch, args.iters, args.log_every,
                                      args.avg_power, args.monitor_impl, f"{args.seed}:{r}",
                                      plan, args.target_v)
            per_run.append(curve); times.append(wall); hits.append(hit)
        if plan is not None:
            plan.close()

        # aligned curve (mean/std across runs at each logged t)
        all_t = sorted({t for cp in per_run for t in cp})
        for t in all_t:
            objs = [cp[t][0] for cp in per_run if t in cp]
            m, sd = _mean_std(objs)
            cwr.writerow({"network_id": info["network_id"], "t": t,
                          "obj_mean": f"{m:.8f}" if m is not None else "",
                          "obj_std": f"{sd:.8f}", "runs": len(objs)})
        cfile.flush()

        final_objs = [cp[max(cp)][0] for cp in per_run if cp]   # each run's last checkpoint
        fm, fsd = _mean_std(final_objs)
        tm, tsd = _mean_std(times)

        # target-stop stats: iterations and TIMED wall to first hit V* <= target
        reached_iters = [float(h) for h in hits if h is not None]
        reached_walls = [per_run[i][hits[i]][1] for i in range(len(hits)) if hits[i] is not None]
        hm, hsd = _mean_std(reached_iters)
        wm, _ = _mean_std(reached_walls)

        swr.writerow({
            "network_id": info["network_id"], "n_controls": info["n_controls"],
            "total_Q": info["total_Q"], "method": args.method, "batch": args.batch,
            "runs": n_runs, "iters": args.iters, "avg_power": args.avg_power,
            "rho": args.rho, "wall_time_s": f"{tm:.6f}", "wall_time_std": f"{tsd:.6f}",
            "final_obj": f"{fm:.8f}", "final_obj_std": f"{fsd:.8f}",
            "target_v": ("" if args.target_v is None else f"{args.target_v:.6f}"),
            "reached": ("" if args.target_v is None else f"{len(reached_iters)}/{n_runs}"),
            "hit_iter_mean": (f"{hm:.1f}" if hm is not None else ""),
            "hit_iter_std": f"{hsd:.1f}" if reached_iters else "",
            "hit_wall_mean": (f"{wm:.6f}" if wm is not None else ""),
        })
        sfile.flush()
        if args.target_v is not None:
            if reached_iters:
                pm = f" +-{hsd:.0f}" if len(reached_iters) > 1 else ""
                print(f"  {args.method}: reached V*<={args.target_v:g} at iter {hm:.0f}{pm}  "
                      f"wall={wm:.3f}s  ({len(reached_iters)}/{n_runs} run"
                      f"{'s' if n_runs > 1 else ''})  final V*={fm:.6f}", flush=True)
            else:
                print(f"  {args.method}: did NOT reach V*<={args.target_v:g} within "
                      f"{args.iters} iters  (final V*={fm:.6f}, wall={tm:.3f}s)", flush=True)
        else:
            print(f"  {args.method}: final V*={fm:.6f}  wall={tm:.3f}s"
                  f"{' +-' + format(tsd, '.3f') if n_runs > 1 else ''}  "
                  f"(over {n_runs} run{'s' if n_runs > 1 else ''}, {args.iters} iters)", flush=True)

    sfile.close(); cfile.close()
    _GP.shutdown(); _GPF.shutdown()                   # release worker pools (no-op if unused)
    print(f"\nwrote {args.out}_summary.csv and {args.out}_curves.csv")


def build_arg_parser():
    default_csv = os.path.join(os.path.dirname(_HERE), "exp1", "synth_nets_random_ell.csv")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True,
                   choices=["stochastic", "fold", "tape", "mary",
                            "par-tape", "par-mary", "par-fold"],
                   help="which single gradient oracle to run")
    p.add_argument("--cores", type=int, default=_GP.CORES,
                   help="core cap for par-* methods (default: os.cpu_count() - 1)")
    p.add_argument("--csv", default=default_csv, help="networks (structure + probs)")
    p.add_argument("--out", default=os.path.join(_HERE, "gradient_rm_tape"), help="output base path")
    p.add_argument("--start-id", type=int, default=0, help="first network_id (inclusive)")
    p.add_argument("--end-id", type=int, default=None, help="last network_id (inclusive)")
    p.add_argument("--limit", type=int, default=0, help="only the first N of the selected range")
    p.add_argument("--iters", type=int, default=2000,
                   help="RM iterations to run (ignored when --target-v is set)")
    p.add_argument("--target-v", type=float, default=None,
                   help="run until the monitored V*(l_bar) <= this target, with NO "
                        "iteration cap (checked at the --log-every cadence). If unset, "
                        "run exactly --iters.")
    p.add_argument("--batch", type=int, default=1, help="stochastic gradient batch size b")
    p.add_argument("--runs", type=int, default=1, help="independent stochastic runs to average")
    p.add_argument("--log-every", type=int, default=50, help="objective-curve checkpoint cadence")
    p.add_argument("--avg-power", type=float, default=0.0,
                   help="polynomial averaging exponent for the reported iterate (0 = uniform)")
    p.add_argument("--rho", type=float, default=5.0, help="discount rate rho/lambda (> 0)")
    p.add_argument("--monitor-impl", choices=["tape", "fold", "mary"], default="tape",
                   help="exact oracle used for the UNTIMED objective V*(l_bar) (default tape: "
                        "exact and fast on large nets)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (stochastic)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
