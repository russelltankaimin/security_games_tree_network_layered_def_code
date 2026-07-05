"""
regret_matching.py
==================

Regret-matching solver for the defender's time-allocation game on a
series-parallel (SP) security network.

Setup
-----
The defender splits one unit of "time" across the controls: a per-control
allocation l_v >= 0 with  sum_v l_v = 1.  That allocation sets each control's
per-attempt discount factor

        beta_v = exp(-rho * l_v)                                    (rho > 0)

This is a min-max (Stackelberg) problem:  min_l max_pi  u(l, pi), where the
attacker (follower) maximises its value pi and the defender (leader) minimises.

The attacker best-responds with the optimal *Gittins index policy*: given the
betas it computes the coupled index table alpha(v, k) (one index per control v
and failure count k) via `sp_attacker.index_table`.  Because the attacker plays
optimally, the resulting Stackelberg value V*(l) is a convex function of l, and
`sp_gradients2.value_and_gradient` returns both V*(l) and the exact
(sub)gradient  dV*/dl_v  --  evaluated at the attacker's fixed optimal response
(Danskin / envelope theorem), i.e. it is the defender reading off the gradient
*from* the attacker's indexed best response.

The defender wants to DETER the attacker, i.e. minimise V*(l) over the simplex.

Algorithm (Hart & Mas-Colell regret matching)
----------------------------------------------
Treat each control as an "action".  Initialise l uniform (l_v = 1/n).  Each
round t, with gradient g^t = dV*/dl at the current allocation l^t, the linearised
loss of the mixed play is <g^t, l^t> and the loss of pure action v is g_v^t, so
the instantaneous regret for not having concentrated on v is

        r_v^t = <g^t, l^t> - g_v^t.

Accumulate  R_v += r_v^t  and play proportional to positive regret,

        l_v^{t+1} = [R_v]^+ / sum_u [R_u]^+     (uniform if all R_u <= 0).

External regret vanishes at O(1/sqrt(T)), so the AVERAGE allocation converges to
the minimiser of the convex V* over the simplex.  A control with strongly
negative gradient (high deterrence) earns positive regret and attracts budget --
exactly the defender's incentive.

Output
------
  * the final time allocation l_v  (the running average -- the no-regret iterate)
  * the attacker's index table alpha(v, k) evaluated at that allocation.

Run `python3 regret_matching.py` for a worked demo.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import namedtuple
from math import exp
from typing import Dict, List

# --- make the sibling attacker/defender packages importable ----------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _pkg in ("spn_attacker", "spn_defender"):
    _path = os.path.join(_ROOT, _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Attacker side: Gittins index table (+ a value fn for cross-checking).
from sp_attacker import (  # noqa: E402
    Control as AttackerControl,
    Par as AttackerPar,
    Series as AttackerSeries,
    game_value,
    index_table,
)

# Defender side: convex value V*(l) and its exact gradient dV*/dl.
from sp_gradients2 import (  # noqa: E402
    Control as DefenderControl,
    Parallel as DefenderParallel,
    Series as DefenderSeries,
    value_and_gradient,
)


# ===========================================================================
# Neutral network specification (built once, rendered into either library).
# ===========================================================================

# Names mirror the underlying libraries (`Control`/`Series` are shared; the
# parallel node is `Par` in sp_attacker and `Parallel` in sp_gradients2 -- we use
# `Parallel` here). A leaf's lockout is implied by len(success_probs).
Control = namedtuple("Control", ["name", "success_probs"])
Series = namedtuple("Series", ["left", "right"])       # clear left, then right
Parallel = namedtuple("Parallel", ["left", "right"])   # clear either branch (race)


def control_names(spec) -> List[str]:
    """Control names in left-to-right tree order."""
    if isinstance(spec, Control):
        return [spec.name]
    return control_names(spec.left) + control_names(spec.right)


def build_attacker_tree(spec, allocation: Dict[str, float], rho: float):
    """Render the spec into an `sp_attacker` tree, baking beta_v = exp(-rho*l_v)."""
    if isinstance(spec, Control):
        beta = exp(-rho * allocation[spec.name])
        return AttackerControl(spec.name, beta, tuple(spec.success_probs))
    child = build_attacker_tree
    if isinstance(spec, Series):
        return AttackerSeries(child(spec.left, allocation, rho), child(spec.right, allocation, rho))
    return AttackerPar(child(spec.left, allocation, rho), child(spec.right, allocation, rho))


def build_defender_tree(spec):
    """Render the spec into an `sp_gradients2` tree (allocation-agnostic)."""
    if isinstance(spec, Control):
        return DefenderControl(spec.name, len(spec.success_probs), list(spec.success_probs))
    child = build_defender_tree
    if isinstance(spec, Series):
        return DefenderSeries(child(spec.left), child(spec.right))
    return DefenderParallel(child(spec.left), child(spec.right))


# ===========================================================================
# Regret matching
# ===========================================================================

RegretMatchingResult = namedtuple(
    "RegretMatchingResult",
    ["allocation", "last_allocation", "indices", "value", "uniform_value", "value_history"],
)


def regret_matching(
    spec,
    rho: float = 0.9,
    iterations: int = 2000,
    verbose: bool = True,
) -> RegretMatchingResult:
    """Solve the defender's allocation game by regret matching.

    Returns the average (no-regret) allocation, the last-iterate allocation, the
    attacker's index table at the average allocation, and the value trajectory.
    """
    names = control_names(spec)
    n = len(names)
    defender_tree = build_defender_tree(spec)

    cumulative_regret = {v: 0.0 for v in names}
    allocation = {v: 1.0 / n for v in names}   # uniform initialisation
    allocation_sum = {v: 0.0 for v in names}   # for the running-average iterate
    value_history: List[float] = []
    uniform_value = None

    for t in range(iterations):
        # Attacker: compute the Gittins index table at the current betas. (Not
        # needed numerically for the gradient -- the fold re-derives the optimal
        # response internally -- but this is the attacker's explicit best-response
        # step in the loop.)
        _ = index_table(build_attacker_tree(spec, allocation, rho))

        # Defender: value V*(l) and exact gradient dV*/dl at the attacker's
        # optimal response (Danskin / envelope theorem).
        value, gradient = value_and_gradient(defender_tree, allocation=allocation, discount_rate=rho)
        value_history.append(value)
        if uniform_value is None:
            uniform_value = value

        # Accumulate the strategy actually played this round (for the average).
        for v in names:
            allocation_sum[v] += allocation[v]

        # Regret matching update (minimising the convex value).
        expected_loss = sum(allocation[v] * gradient[v] for v in names)
        for v in names:
            cumulative_regret[v] += expected_loss - gradient[v]

        positive = {v: max(cumulative_regret[v], 0.0) for v in names}
        total = sum(positive.values())
        if total > 0.0:
            allocation = {v: positive[v] / total for v in names}
        else:
            allocation = {v: 1.0 / n for v in names}

        if verbose and (t < 3 or (t + 1) % max(1, iterations // 10) == 0):
            print(f"  iter {t + 1:>5}/{iterations}   V* = {value:.6f}")

    last_allocation = allocation
    average_allocation = {v: allocation_sum[v] / iterations for v in names}

    # Attacker index table + value at the reported (average) allocation.
    attacker_tree = build_attacker_tree(spec, average_allocation, rho)
    indices = index_table(attacker_tree)
    final_value, _ = value_and_gradient(defender_tree, allocation=average_allocation, discount_rate=rho)

    return RegretMatchingResult(
        allocation=average_allocation,
        last_allocation=last_allocation,
        indices=indices,
        value=final_value,
        uniform_value=uniform_value,
        value_history=value_history,
    )


# ===========================================================================
# Reporting
# ===========================================================================


def print_result(spec, result: RegretMatchingResult, rho: float) -> None:
    names = control_names(spec)

    print(f"\nV* at uniform allocation : {result.uniform_value:.6f}")
    print(f"V* at final  allocation  : {result.value:.6f}   "
          f"(deterrence gain {result.uniform_value - result.value:+.6f})")
    print(f"\nmin-max value  min_l max_pi u(l, pi) = {result.value:.6f}   "
          f"(attacker's optimal value at the solved allocation)")

    print("\nfinal time allocation  l_v   (sum = "
          f"{sum(result.allocation.values()):.6f}):")
    for v in names:
        print(f"   l[{v}] = {result.allocation[v]:.6f}   "
              f"(beta = exp(-rho*l) = {exp(-rho * result.allocation[v]):.6f})")

    print("\nattacker Gittins index table  alpha(v, k)  at the final allocation:")
    for v in names:
        ks = sorted(k for (name, k) in result.indices if name == v)
        for k in ks:
            print(f"   alpha({v}, {k}) = {result.indices[(v, k)]:.6f}")


# ===========================================================================
# Custom network input
# ===========================================================================
#
# A network is entered as an expression in the constructors below, e.g.
#
#   Parallel(Series(Control("a", [0.5, 0.4]), Parallel(Control("b", 0.6),
#            Control("c", 0.55))), Series(Control("d", 0.5), Control("e", 0.45)))
#
#   * Control(name, probs)   probs is a list/tuple of per-failure success
#                            probabilities (lockout = len); a bare float is
#                            shorthand for a lockout-1 control.
#   * Series(a, b, ...)      clear children in order   (n-ary, >= 2)
#   * Parallel(a, b, ...)    clear any one child        (n-ary, >= 2)
#
# Series/Parallel are associative, so n-ary nodes are right-folded into the
# binary spec used internally. `Ser`/`Par` are accepted as aliases.


def _mk_control(name, probs):
    if isinstance(probs, (int, float)):
        probs = (float(probs),)
    return Control(name, tuple(float(p) for p in probs))


def _fold(node_type, children):
    if len(children) < 2:
        raise ValueError(f"{node_type.__name__} needs >= 2 children")
    folded = children[-1]
    for child in reversed(children[:-1]):
        folded = node_type(child, folded)
    return folded


def parse_tree(text: str):
    """Parse a network expression (see above) into the internal spec."""
    namespace = {
        "Control": _mk_control,
        "Series": lambda *children: _fold(Series, children),
        "Parallel": lambda *children: _fold(Parallel, children),
        "Ser": lambda *children: _fold(Series, children),
        "Par": lambda *children: _fold(Parallel, children),
    }
    tree = eval(text, {"__builtins__": {}}, namespace)  # restricted namespace
    names = control_names(tree)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate control names: {names}")
    return tree


# ===========================================================================
# Default demo network
# ===========================================================================


def default_network():
    """Parallel( Series(a[q=2], Parallel(b, c)), Series(d, e) ) -- the coupled example."""
    return Parallel(
        Series(Control("a", (0.5, 0.4)), Parallel(Control("b", (0.6,)), Control("c", (0.55,)))),
        Series(Control("d", (0.5,)), Control("e", (0.45,))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=2000, help="regret-matching rounds")
    parser.add_argument("--rho", type=float, default=0.9, help="discount rate rho (> 0)")
    parser.add_argument("--tree", type=str, default=None,
                        help='custom SP network as an expression, e.g. '
                             '"Parallel(Series(Control(\'a\',[0.5,0.4]), Control(\'b\',0.6)), '
                             'Control(\'c\',0.55))"')
    parser.add_argument("--tree-file", type=str, default=None,
                        help="path to a file containing the network expression (see --tree)")
    parser.add_argument("--verify", action="store_true",
                        help="cross-check the two libraries agree on V* at the final allocation")
    args = parser.parse_args()

    if args.tree_file is not None:
        with open(args.tree_file) as handle:
            spec = parse_tree(handle.read())
        source = f"tree from {args.tree_file}"
    elif args.tree is not None:
        spec = parse_tree(args.tree)
        source = "custom tree"
    else:
        spec = default_network()
        source = "default network"
    print(f"regret matching on {source}, rho = {args.rho}, "
          f"{args.iterations} iterations")
    result = regret_matching(spec, rho=args.rho, iterations=args.iterations)
    print_result(spec, result, args.rho)

    if args.verify:
        attacker_tree = build_attacker_tree(spec, result.allocation, args.rho)
        attacker_value = game_value(attacker_tree)
        diff = abs(attacker_value - result.value)
        print(f"\n[verify] attacker game_value = {attacker_value:.8f}   "
              f"defender V* = {result.value:.8f}   |diff| = {diff:.2e}")


if __name__ == "__main__":
    main()
