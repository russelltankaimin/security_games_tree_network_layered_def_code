"""
regret_matching_arbitrary.py
============================

Defender time-allocation solver when the feasible set is an arbitrary
(nonempty) polyhedron rather than the probability simplex.

Setup (same min-max as regret_matching.py)
------------------------------------------
The defender picks an allocation l in a polyhedron

        P = { l in R^n : l >= 0,  A l + c >= d }          (nonempty)

setting beta_v = exp(-rho * l_v).  The attacker best-responds with the optimal
Gittins index policy, giving the Stackelberg value V*(l) = max_pi u(l, pi).  The
defender minimises it:

        min_{l in P}  max_pi  u(l, pi).

The attacker's index table alpha(v, k) and the value/gradient of V* come from
`sp_attacker.index_table` and `sp_gradients2.value_and_gradient` -- the latter
returns the exact (sub)gradient dV*/dl at the attacker's fixed optimal response
(Danskin / envelope theorem).

Why not "regret matching" literally
-----------------------------------
Hart-Mas-Colell regret matching is a *simplex* algorithm -- its play-proportional
-to-positive-regret step is exactly the projection onto the probability simplex.
For an arbitrary convex P we keep the same no-regret idea (linearise with the
gradient each round; the averaged iterate converges) but replace the simplex step
with **projected subgradient descent** (Zinkevich online gradient descent):

        l_{t+1} = Proj_P( l_t - eta_t * g_t ),   g_t = dV*/dl at l_t
        eta_t   = eta0 / sqrt(t+1)  (diminishing),   report the average iterate.

This is the direct generalisation of regret matching to any convex set and,
unlike Frank-Wolfe, is robust to the fact that V*(l) is convex but NONSMOOTH
(the gradient module returns a subgradient at index-tie kinks).  On P = simplex
it reproduces the regret-matching solution.

The only P-dependent operation is the Euclidean projection onto P.  P is an
intersection of halfspaces / a hyperplane / the nonnegative orthant, each with a
closed-form projection, combined by **Dykstra's algorithm** (alternating
projection corrected to converge to the true projection onto the intersection).

Well-posedness
--------------
For the min-max to be attained, P must be bounded in the descent directions:
more time deters the attacker (V* is nonincreasing in each l_v), so if P is
unbounded upward the defender drives l -> infinity and V* -> 0.  Include an upper
budget (e.g. sum_v l_v <= B, or sum_v l_v = B) in the constraints.

Run `python3 regret_matching_arbitrary.py` for a worked demo.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import namedtuple
from math import exp, sqrt
from typing import Dict, List, Optional, Sequence, Tuple

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
# Polyhedron and Euclidean projection (Dykstra)
# ===========================================================================

Vector = List[float]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _project_halfspace(x: Vector, a: Sequence[float], b: float) -> Vector:
    """Project onto { z : a . z >= b } (unchanged if already satisfied)."""
    s = _dot(a, x)
    if s >= b:
        return list(x)
    norm2 = _dot(a, a)
    if norm2 <= 1e-15:
        return list(x)
    t = (b - s) / norm2
    return [xi + t * ai for xi, ai in zip(x, a)]


def _project_hyperplane(x: Vector, a: Sequence[float], b: float) -> Vector:
    """Project onto { z : a . z = b }."""
    norm2 = _dot(a, a)
    if norm2 <= 1e-15:
        return list(x)
    t = (b - _dot(a, x)) / norm2
    return [xi + t * ai for xi, ai in zip(x, a)]


class Polyhedron:
    """A nonempty polyhedron P = { l >= 0 (optional) : a_i . l >= b_i, e_j . l = f_j }.

    Rows are given aligned to a fixed control ordering `names`.
      * inequalities `ineq`: list of (a, b) meaning  a . l >= b
      * equalities   `eq`  : list of (a, b) meaning  a . l  = b
      * nonneg             : if True, also intersect with { l >= 0 }
    """

    def __init__(
        self,
        names: Sequence[str],
        ineq: Optional[Sequence[Tuple[Sequence[float], float]]] = None,
        eq: Optional[Sequence[Tuple[Sequence[float], float]]] = None,
        nonneg: bool = True,
    ) -> None:
        self.names = list(names)
        self.n = len(self.names)
        # Each set: (kind, a, b) with kind in {"ineq", "eq", "orthant"}.
        self._sets: List[Tuple[str, Optional[Vector], float]] = []
        for a, b in (eq or []):
            self._sets.append(("eq", list(a), float(b)))
        for a, b in (ineq or []):
            self._sets.append(("ineq", list(a), float(b)))
        if nonneg:
            self._sets.append(("orthant", None, 0.0))

    # -- named-constructor helpers -----------------------------------------
    @classmethod
    def from_A_c_d(cls, names, A, c, d, nonneg: bool = True) -> "Polyhedron":
        """Build P from the form  A l + c >= d  (row i:  A[i] . l >= d[i] - c[i])."""
        ineq = [(list(A[i]), float(d[i]) - float(c[i])) for i in range(len(A))]
        return cls(names, ineq=ineq, nonneg=nonneg)

    @classmethod
    def simplex(cls, names) -> "Polyhedron":
        """The probability simplex { l >= 0, sum l = 1 } (recovers regret matching)."""
        ones = [1.0] * len(names)
        return cls(names, eq=[(ones, 1.0)], nonneg=True)

    # -- projection ---------------------------------------------------------
    def project(self, y: Sequence[float], max_iter: int = 5000, tol: float = 1e-12) -> Vector:
        """Euclidean projection of `y` onto P via Dykstra's algorithm."""
        x = list(y)
        corrections = [[0.0] * self.n for _ in self._sets]
        for _ in range(max_iter):
            x_prev = list(x)
            for k, (kind, a, b) in enumerate(self._sets):
                temp = [x[j] + corrections[k][j] for j in range(self.n)]
                if kind == "orthant":
                    proj = [v if v > 0.0 else 0.0 for v in temp]
                elif kind == "eq":
                    proj = _project_hyperplane(temp, a, b)
                else:  # ineq
                    proj = _project_halfspace(temp, a, b)
                corrections[k] = [temp[j] - proj[j] for j in range(self.n)]
                x = proj
            if sum((x[j] - x_prev[j]) ** 2 for j in range(self.n)) <= tol * tol:
                break
        return x


# ===========================================================================
# Projected subgradient solver
# ===========================================================================

Result = namedtuple(
    "Result",
    ["allocation", "last_allocation", "indices", "value", "start_value", "value_history"],
)


def solve(
    spec,
    polyhedron: Polyhedron,
    rho: float = 0.9,
    iterations: int = 2000,
    eta0: float = 1.0,
    verbose: bool = True,
) -> Result:
    """Minimise the convex V*(l) over the polyhedron P by projected subgradient
    descent; return the averaged (no-regret) allocation and the min-max value."""
    names = control_names(spec)
    n = len(names)
    if list(polyhedron.names) != list(names):
        raise ValueError("polyhedron.names must match the network's control order")
    defender_tree = build_defender_tree(spec)

    # Feasible start: project the uniform point onto P.
    x = polyhedron.project([1.0 / n] * n)
    x_sum = [0.0] * n
    value_history: List[float] = []
    start_value: Optional[float] = None

    for t in range(iterations):
        alloc = {names[i]: x[i] for i in range(n)}

        # Attacker best-response: the Gittins index table at the current betas.
        _ = index_table(build_attacker_tree(spec, alloc, rho))

        # Defender: value and (sub)gradient at the attacker's optimal response.
        value, gradient = value_and_gradient(defender_tree, allocation=alloc, discount_rate=rho)
        value_history.append(value)
        if start_value is None:
            start_value = value

        for i in range(n):
            x_sum[i] += x[i]

        # Projected subgradient step (diminishing step size).
        eta = eta0 / sqrt(t + 1)
        y = [x[i] - eta * gradient[names[i]] for i in range(n)]
        x = polyhedron.project(y)

        if verbose and (t < 3 or (t + 1) % max(1, iterations // 10) == 0):
            print(f"  iter {t + 1:>5}/{iterations}   V* = {value:.6f}")

    # The average of feasible iterates is feasible (P is convex).
    average = [x_sum[i] / iterations for i in range(n)]
    avg_alloc = {names[i]: average[i] for i in range(n)}
    indices = index_table(build_attacker_tree(spec, avg_alloc, rho))
    final_value, _ = value_and_gradient(defender_tree, allocation=avg_alloc, discount_rate=rho)

    return Result(
        allocation=avg_alloc,
        last_allocation={names[i]: x[i] for i in range(n)},
        indices=indices,
        value=final_value,
        start_value=start_value,
        value_history=value_history,
    )


# ===========================================================================
# Reporting
# ===========================================================================


def print_result(spec, result: Result, rho: float) -> None:
    names = control_names(spec)
    print(f"\nV* at feasible start     : {result.start_value:.6f}")
    print(f"V* at final  allocation  : {result.value:.6f}   "
          f"(deterrence gain {result.start_value - result.value:+.6f})")
    print(f"\nmin-max value  min_{{l in P}} max_pi u(l, pi) = {result.value:.6f}   "
          f"(attacker's optimal value at the solved allocation)")

    print(f"\nfinal time allocation  l_v   (sum = {sum(result.allocation.values()):.6f}):")
    for v in names:
        print(f"   l[{v}] = {result.allocation[v]:.6f}   "
              f"(beta = exp(-rho*l) = {exp(-rho * result.allocation[v]):.6f})")

    print("\nattacker Gittins index table  alpha(v, k)  at the final allocation:")
    for v in names:
        for k in sorted(kk for (name, kk) in result.indices if name == v):
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
    parser.add_argument("--iterations", type=int, default=2000, help="projected-subgradient steps")
    parser.add_argument("--rho", type=float, default=0.9, help="discount rate rho (> 0)")
    parser.add_argument("--eta0", type=float, default=1.0, help="initial step size (eta_t = eta0/sqrt(t))")
    parser.add_argument("--tree", type=str, default=None,
                        help="custom SP network expression (Control/Series/Parallel)")
    parser.add_argument("--tree-file", type=str, default=None,
                        help="path to a file containing the network expression")
    parser.add_argument("--min-share", type=float, default=0.0,
                        help="lower bound l_v >= min_share for every control (a genuine polyhedron)")
    parser.add_argument("--budget", type=float, default=1.0, help="total budget: sum_v l_v = budget")
    parser.add_argument("--verify", action="store_true",
                        help="cross-check the attacker/defender libraries agree on V* at the solution")
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

    names = control_names(spec)
    n = len(names)
    if args.min_share * n > args.budget + 1e-12:
        raise SystemExit(f"infeasible: min_share*{n} = {args.min_share * n} > budget {args.budget}")

    # P = { l >= 0,  sum l = budget,  l_v >= min_share for all v }.
    ones = [1.0] * n
    ineq = []
    if args.min_share > 0.0:
        for i in range(n):
            row = [0.0] * n
            row[i] = 1.0
            ineq.append((row, args.min_share))   # l_i >= min_share
    polyhedron = Polyhedron(names, ineq=ineq, eq=[(ones, args.budget)], nonneg=True)

    print(f"projected-subgradient min-max on {source}   "
          f"(sum l = {args.budget}, l_v >= {args.min_share}), rho = {args.rho}, "
          f"{args.iterations} iterations")
    result = solve(spec, polyhedron, rho=args.rho, iterations=args.iterations, eta0=args.eta0)
    print_result(spec, result, args.rho)

    if args.verify:
        attacker_value = game_value(build_attacker_tree(spec, result.allocation, args.rho))
        diff = abs(attacker_value - result.value)
        print(f"\n[verify] attacker game_value = {attacker_value:.8f}   "
              f"defender V* = {result.value:.8f}   |diff| = {diff:.2e}")


if __name__ == "__main__":
    main()
