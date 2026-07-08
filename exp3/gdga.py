"""
general_defender_general_attacker.py
==================================

GENERALISED mismatched game (a superset of greedy_def_gittins_attacker.py). The
defender optimises its allocation l ASSUMING the attacker plays a SPECIFIED policy
(--defender-policy); then a (possibly different) SPECIFIED attacker policy
(--attacker-policy) is turned loose on the fixed l. We log the time to compute l*
and the value the actual attacker achieves.

Specifiable policies (both roles)
---------------------------------
    index              : the OPTIMAL Gittins-index attacker (sp_attacker)
    greedy_probability : attack the highest one-step success prob p(k)
    greedy_discounted  : attack the highest discounted success beta*p(k)
    uncoupled_index    : attack the highest single-control index c_hat(v,k)
    single_route       : commit to the best route; never pivot
    fixed_priority     : structure-blind fixed lexicographic order
    random_frontier    : attack a uniformly random live control

Gradient oracle for the defender's optimisation
------------------------------------------------
The defender minimises V_def(l) = E_{pi_def}[ 1_win * prod_v beta_v^{n_v} ] over
the simplex by regret matching from the uniform start. Two oracles:

  * stochastic : the pathwise Monte-Carlo estimate  dV/dl_v = -rho E[n_v R]  from
                 `--samples` rollouts of pi_def. This is a correct (sub)gradient
                 for EVERY policy above -- the l-independent ones trivially, the
                 deterministic argmax ones because they are piecewise-constant in
                 l (locally fixed trajectory law), and the optimal `index` one by
                 the envelope theorem.
  * exact      : the closed-form O(Q^2) fold dV*/dl (sp_gradients2). This ONLY
                 exists for the OPTIMAL `index` attacker, so --gradient exact is
                 valid only with --defender-policy index; requested otherwise it
                 warns and falls back to stochastic.

Value of the actual attacker at l*
----------------------------------
  * index policy      : exact value via the O(Q^2) fold.
  * any other policy  : Monte-Carlo average reward over --eval-samples rollouts.

We log both defender_assumed_value = V_{pi_def}(l*) (what the defender thought it
would get) and attacker_value = V_{pi_att}(l*) (what the specified attacker
actually achieves); their difference value_gap is the price of the mismatch.

Termination (first to fire; `stop_reason`): "regret" (avg external regret
< --epsilon), "plateau" (running-average allocation l_bar's per-control spread
over the last --patience monitored points < --obj-tol, and t >= --min-iters;
--patience 0 disables), else "max_iters".

Only the gradient + regret-matching update are timed (defender_opt_time_s); all
monitoring / evaluation is untimed. Output CSV (--out): one row per network.
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
    value_profile,
)
from other_policies import (  # noqa: E402
    FixedPriorityPolicy,
    GreedyDiscountedPolicy,
    GreedyProbabilityPolicy,
    RandomFrontierPolicy,
    SingleRoutePolicy,
    UncoupledIndexPolicy,
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
# Policy factory + a uniform rollout entry point
# ===========================================================================

POLICIES = ("index", "greedy_probability", "greedy_discounted", "uncoupled_index",
            "single_route", "fixed_priority", "random_frontier")

_BASELINE_CLASSES = {
    "greedy_probability": GreedyProbabilityPolicy,
    "greedy_discounted": GreedyDiscountedPolicy,
    "uncoupled_index": UncoupledIndexPolicy,
    "single_route": SingleRoutePolicy,
    "fixed_priority": FixedPriorityPolicy,
}


def make_policy(name, net, rng):
    """Instantiate the named policy on `net`. `index` is the optimal Gittins
    attacker; the rest are the baselines from other_policies."""
    if name == "index":
        return IndexAttacker(net)
    if name == "random_frontier":
        return RandomFrontierPolicy(net, rng)
    return _BASELINE_CLASSES[name](net)


def do_rollout(policy, net, rng):
    """One rollout of any policy, returning a Rollout (with .reward, .history)."""
    if isinstance(policy, IndexAttacker):
        return policy.simulate(rng)
    return simulate_policy(net, policy, rng)


# ===========================================================================
# Oracles
# ===========================================================================


def stochastic_gradient(policy_name, structure, probs, ell, rho, samples, rng, names):
    """Monte-Carlo V_pi(l) and pathwise gradient dV/dl_v = -rho E_pi[n_v R] from
    `samples` rollouts of the named policy at allocation l."""
    net = build_attacker_net(structure, probs, ell, rho)
    policy = make_policy(policy_name, net, rng)
    sum_nvR = {name: 0.0 for name in names}
    total_R = 0.0
    for _ in range(samples):
        roll = do_rollout(policy, net, rng)
        R = roll.reward
        if R > 0.0:
            total_R += R
            for name, _k, _succ in roll.history:
                sum_nvR[name] += R          # summed over occurrences => n_v * R
    grad = {name: -rho * sum_nvR[name] / samples for name in names}
    return total_R / samples, grad


def index_value(defender_tree, ell, rho):
    """V*(l): the OPTIMAL (Gittins index) attacker's value via the exact O(Q^2) fold."""
    value, _ = value_and_gradient(defender_tree, allocation=ell,
                                  discount_rate=rho, break_ties=False)
    return value


def policy_value(policy_name, structure, probs, defender_tree, ell, rho,
                 eval_samples, rng, names):
    """Value of the named policy at l: exact fold for `index`, else MC average."""
    if policy_name == "index":
        return index_value(defender_tree, ell, rho)
    net = build_attacker_net(structure, probs, ell, rho)
    policy = make_policy(policy_name, net, rng)
    total = sum(do_rollout(policy, net, rng).reward for _ in range(eval_samples))
    value_profile.cache_clear()
    return total / eval_samples


def indices_by_control(structure, probs, ell, rho, names):
    """Gittins index table alpha(v, k) at allocation `ell`, grouped by control."""
    table = index_table(build_attacker_net(structure, probs, ell, rho))
    grouped = {}
    for name in names:
        ks = sorted(k for (nm, k) in table if nm == name)
        grouped[name] = [round(table[(name, k)], 6) for k in ks]
    return grouped


# ===========================================================================
# Defender optimisation: regret matching against the specified policy
# ===========================================================================


def optimise_defender(defender_policy, use_exact, structure, probs, names,
                      defender_tree, rho, samples, seed, max_iters, epsilon,
                      obj_tol, patience, min_iters, log_every, label="",
                      progress_every=0):
    """Minimise V_{defender_policy}(l) over the simplex by regret matching from the
    uniform allocation. `use_exact` selects the O(Q^2) fold gradient (valid only
    for defender_policy == 'index'); otherwise the pathwise MC gradient is used.

    Returns (ell_star, stop_reason, iters, opt_time_s, final_regret)."""
    n = len(names)
    rng = random.Random(f"{seed}:{defender_policy}")
    ell = {v: 1.0 / n for v in names}          # uniform initialisation
    cumulative_regret = {v: 0.0 for v in names}
    ell_sum = {v: 0.0 for v in names}
    opt_time = 0.0

    window = []                                # recent l_bar snapshots (for plateau)
    stop_reason = "max_iters"
    ell_star = dict(ell)
    final_regret = float("inf")
    t = 0

    for t in range(1, max_iters + 1):
        for v in names:                        # accumulate the played iterate
            ell_sum[v] += ell[v]

        start = time.perf_counter()
        if use_exact:
            _v_est, grad = value_and_gradient(defender_tree, allocation=ell,
                                              discount_rate=rho)
        else:
            _v_est, grad = stochastic_gradient(defender_policy, structure, probs,
                                               ell, rho, samples, rng, names)
        expected_loss = sum(ell[v] * grad[v] for v in names)
        for v in names:
            cumulative_regret[v] += expected_loss - grad[v]
        positive = {v: max(cumulative_regret[v], 0.0) for v in names}
        total = sum(positive.values())
        ell = ({v: positive[v] / total for v in names} if total > 0.0
               else {v: 1.0 / n for v in names})
        opt_time += time.perf_counter() - start

        # Stochastic path builds a fresh attacker net each iteration -> drop
        # value_profile's unbounded cache (pure function; cannot alter results).
        if not use_exact:
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
                # Allocation-plateau test on the fixed log cadence (value-free, so
                # no MC noise enters the stopping decision).
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
                      f"V_def~{_v_est:.6f}  regret={final_regret:.3e}  "
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


FIELDS = ["network_id", "n_controls", "total_Q", "rho", "defender_policy",
          "attacker_policy", "gradient", "samples", "converged", "stop_reason",
          "iters", "defender_opt_time_s", "final_regret",
          "defender_assumed_value", "attacker_value", "value_gap",
          "final_ell", "final_indices"]


def run(args):
    if args.rho <= 0:
        raise SystemExit("--rho must be > 0")

    # Exact gradients only model the OPTIMAL (index) attacker.
    use_exact = (args.gradient == "exact")
    if use_exact and args.defender_policy != "index":
        print(f"[warn] --gradient exact is only valid with --defender-policy index; "
              f"got '{args.defender_policy}'. Falling back to stochastic gradients.",
              flush=True)
        use_exact = False
    gradient_used = "exact" if use_exact else "stochastic"

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

    print(f"{len(networks)} networks | defender={args.defender_policy} "
          f"(grad={gradient_used}) vs attacker={args.attacker_policy} | "
          f"rho={args.rho} samples={args.samples} eps={args.epsilon} "
          f"obj_tol={args.obj_tol} patience={args.patience}\n", flush=True)

    for idx, info in enumerate(networks, 1):
        if info["probs"] is None:
            raise ValueError(f"network {info['network_id']}: needs a probs column")
        names = leaf_names(info["structure"])
        defender_tree = build_defender_tree(info["structure"], info["probs"])
        print(f"=== [{idx}/{len(networks)}] network {info['network_id']}: "
              f"n={info['n_controls']} Q={info['total_Q']} ===", flush=True)

        # 1) Defender optimises l assuming --defender-policy (this is timed).
        ell_star, stop_reason, iters, opt_time, regret = optimise_defender(
            args.defender_policy, use_exact, info["structure"], info["probs"],
            names, defender_tree, args.rho, args.samples, args.seed,
            args.max_iters, args.epsilon, args.obj_tol, args.patience,
            args.min_iters, args.log_every,
            label=f"net {info['network_id']}", progress_every=args.progress_every)

        # 2) Value the ACTUAL attacker (--attacker-policy) achieves at the fixed l*.
        #    Exact where possible: the `index` attacker is evaluated by the exact
        #    O(Q^2) fold; any other policy is estimated by Monte-Carlo rollouts.
        att_rng = random.Random(f"{args.seed}:eval_att")
        attacker_value = policy_value(args.attacker_policy, info["structure"],
                                      info["probs"], defender_tree, ell_star,
                                      args.rho, args.eval_samples, att_rng, names)
        # 3) Value the defender THOUGHT it would get (its assumed policy at l*).
        def_rng = random.Random(f"{args.seed}:eval_def")
        defender_assumed = policy_value(args.defender_policy, info["structure"],
                                        info["probs"], defender_tree, ell_star,
                                        args.rho, args.eval_samples, def_rng, names)

        final_ell = {v: round(ell_star[v], 6) for v in names}
        # Gittins indices describe only the OPTIMAL attacker's policy at l*, so
        # log them only when some role is `index`; otherwise no one uses them.
        uses_index = "index" in (args.defender_policy, args.attacker_policy)
        final_indices = (indices_by_control(
            info["structure"], info["probs"], ell_star, args.rho, names)
            if uses_index else {})

        writer.writerow({
            "network_id": info["network_id"], "n_controls": info["n_controls"],
            "total_Q": info["total_Q"], "rho": args.rho,
            "defender_policy": args.defender_policy,
            "attacker_policy": args.attacker_policy, "gradient": gradient_used,
            "samples": args.samples,
            "converged": stop_reason in ("regret", "plateau"),
            "stop_reason": stop_reason, "iters": iters,
            "defender_opt_time_s": f"{opt_time:.6f}",
            "final_regret": f"{regret:.8f}",
            "defender_assumed_value": f"{defender_assumed:.8f}",
            "attacker_value": f"{attacker_value:.8f}",
            "value_gap": f"{attacker_value - defender_assumed:.8f}",
            "final_ell": json.dumps(final_ell),
            "final_indices": json.dumps(final_indices),
        })
        out_file.flush()
        X, Y = args.defender_policy, args.attacker_policy
        print(f"  -> stop=[{stop_reason}] iters={iters} opt_time={opt_time:.3f}s\n"
              f"       V* defender      V_{X}(l*) = {defender_assumed:.6f}\n"
              f"       attacker value   V_{Y}(l*) = {attacker_value:.6f}   ({Y} playing l*)",
              flush=True)

    out_file.close()
    print(f"\nwrote {args.out}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True,
                   help="input networks CSV (columns: network_id, structure, "
                        "probs, [n_controls, total_Q]); stored ell is ignored -- "
                        "the defender optimises l from the uniform start")
    p.add_argument("--rho", type=float, default=0.9, help="discount rate rho (> 0)")
    p.add_argument("--defender-policy", choices=POLICIES, default="greedy_probability",
                   help="the attacker policy the DEFENDER optimises l against")
    p.add_argument("--attacker-policy", choices=POLICIES, default="index",
                   help="the policy the ACTUAL attacker plays against the fixed l")
    p.add_argument("--gradient", choices=["exact", "stochastic"], default="stochastic",
                   help="defender gradient oracle; 'exact' (O(Q^2) fold) is valid "
                        "only with --defender-policy index, else falls back to "
                        "stochastic")
    p.add_argument("--out", default=os.path.join(_HERE, "general_greedy_general_attacker.csv"),
                   help="output CSV path")
    p.add_argument("--samples", type=int, default=300,
                   help="rollouts per stochastic gradient")
    p.add_argument("--eval-samples", type=int, default=20000,
                   help="rollouts for the final MC value estimates (untimed)")
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
