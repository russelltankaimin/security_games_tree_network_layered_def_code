"""
sp_attacker.py
==============

Reference implementation of the optimal adaptive attacker on a series--parallel
(SP) layered-defence network, in three phases:

    Phase 1  value_profile / game_value   -- the bottom-up *value fold*
    Phase 2  index_table                  -- *index threading* (per-control Gittins indices)
    Phase 3  IndexAttacker                 -- *online execution* of the greedy index policy

Model (see the accompanying note for the full formalism)
--------------------------------------------------------
A network is built from single *controls* by two n-ary constructors (>= 2
children each):

    Series(G1, ..., Gk)  : clear G1, then G2, ... then Gk       (sequential)
    Par(G1, ..., Gk)     : clear ANY one of G1..Gk -- a race    (any branch wins)

Attacking a control takes one attempt; with the control's per-attempt discount
`beta` the attempt succeeds with probability `p(k)` where `k` is the number of
prior failures on that control. Success advances/wins; failure increments `k`,
and after `q = len(ps)` failures the control *jams* (is permanently dead). A jam
strands everything series-downstream of it; a win prunes the losing OR-sibling.
Reaching the target pays reward 1, discounted by the product of the `beta`s of
all attempts made; if every route dies the reward is 0.

Calibrated value and index
--------------------------
`Psi_G(gamma)` is the value of block G when, at any point, the attacker may
instead "retire" for a one-off buyout `gamma`. It is convex and piecewise linear
in `gamma in [0, 1]`. The *index* of a control state is the buyout at which
retiring first becomes optimal, `alpha(v, k) = inf{ gamma : Psi(gamma) = gamma }`.
The optimal policy is greedy: always attack the live control of highest index
(Gittins' theorem, specialised to this setting).

With the linear-merge Series compose and the closed-form leaf solve, the whole
pipeline runs in O(Q * d) time and O(Q) space, where Q is the total number of
control states and d is the height of the series--parallel decomposition tree
(so O(Q log n) on balanced networks, O(Q^2) only on a degenerate chain). This
sidesteps the exponential joint-state MDP. Non-monotone controls (violating
Assumption 2) fall back to an exact O(Q^2)-per-leaf recursion.
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

# ===========================================================================
# Network datatype
# ===========================================================================


@dataclass(frozen=True)
class Control:
    """A single security control (a leaf of the network).

    Attributes
    ----------
    name : unique identifier used in states and the index table.
    beta : per-attempt discount factor in (0, 1); beta = exp(-rho * length).
    ps   : success probabilities (p(0), p(1), ...); ps[k] applies after k
           failures. The control jams after q = len(ps) failures.
    """

    name: str
    beta: float
    ps: Tuple[float, ...]

    def __post_init__(self) -> None:
        # Normalise ps to a tuple so the dataclass stays hashable/cacheable.
        object.__setattr__(self, "ps", tuple(self.ps))

    @property
    def q(self) -> int:
        """Number of attempts before lockout (== number of failure states)."""
        return len(self.ps)


class _CompositeNode:
    """Base for the n-ary composites Series and Par (>= 2 children).

    Children are stored in a tuple. For backward compatibility with the binary
    recursions and external callers, `.left`/`.right` expose the RIGHT-FOLD binary
    view: `.left` is the first child and `.right` is the composite of the rest (a
    single child if only one remains). Because Series and Par are associative (Par
    also commutative), walking this binary view yields identical values and indices
    to folding the children natively -- which the native n-ary functions verify.
    """

    __slots__ = ("children",)

    def __init__(self, *children: "Network") -> None:
        if len(children) < 2:
            raise ValueError(f"{type(self).__name__} needs >= 2 children")
        self.children: Tuple["Network", ...] = tuple(children)

    @property
    def left(self) -> "Network":
        return self.children[0]

    @property
    def right(self) -> "Network":
        rest = self.children[1:]
        return rest[0] if len(rest) == 1 else type(self)(*rest)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.children == other.children  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.children))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(repr(c) for c in self.children)})"


class Series(_CompositeNode):
    """Sequential composition: clear every child in order (n-ary, >= 2)."""


class Par(_CompositeNode):
    """Parallel (OR) composition: completing ANY child wins the block (n-ary, >= 2)."""


Network = Union[Control, Series, Par]


def controls(net: Network) -> List[Control]:
    """Return all controls (leaves) of `net`, left-to-right (n-ary safe)."""
    if isinstance(net, Control):
        return [net]
    result: List[Control] = []
    for child in net.children:
        result += controls(child)
    return result


def single_control_index(beta: float, p: float) -> float:
    """Closed-form index of a single control with success prob `p`, toward a unit
    reward: c_hat = beta * p / (1 - beta * (1 - p)) in [0, 1).

    For a control whose success leads *directly* to reward r, the index is
    c_hat * r. When a non-trivial block lies downstream, use `index_table`.
    """
    return beta * p / (1.0 - beta * (1.0 - p))


# ===========================================================================
# Convex piecewise-linear functions on [0, 1]
# ===========================================================================
#
# Every calibrated value Psi_G is convex and piecewise linear in the buyout
# gamma, with at most Q + 1 breakpoints. We store such a function by its
# breakpoints (xs, ys) and provide the four operations the fold needs:
#   * affine        -- form cf*f + cg*g
#   * maximum       -- pointwise max (inserts the crossing to stay exact)
#   * parallel_with -- the Par rule, 1 - integral of the two derivatives
#   * series        -- the Series rule, R(gamma) * P(gamma / R(gamma))
# plus `index_crossing`, the smallest gamma with f(gamma) = gamma.


class PiecewiseLinear:
    """A continuous piecewise-linear function on [0, 1], given by breakpoints."""

    __slots__ = ("xs", "ys")

    def __init__(self, xs: List[float], ys: List[float]) -> None:
        # Merge breakpoints that coincide (keeps the representation minimal and
        # avoids zero-width pieces that would break slope computations).
        merged_x: List[float] = []
        merged_y: List[float] = []
        for x, y in zip(xs, ys):
            if merged_x and abs(x - merged_x[-1]) < 1e-15:
                merged_y[-1] = y
            else:
                merged_x.append(x)
                merged_y.append(y)
        self.xs = merged_x
        self.ys = merged_y

    # -- construction helpers ------------------------------------------------
    @classmethod
    def identity(cls) -> "PiecewiseLinear":
        """The buyout line f(gamma) = gamma."""
        return cls([0.0, 1.0], [0.0, 1.0])

    @classmethod
    def constant(cls, c: float) -> "PiecewiseLinear":
        """The constant reward f(gamma) = c."""
        return cls([0.0, 1.0], [c, c])

    # -- evaluation ----------------------------------------------------------
    def __call__(self, x: float) -> float:
        """Evaluate (linearly extrapolating just outside [first, last] knot)."""
        xs, ys = self.xs, self.ys
        if len(xs) == 1:
            return ys[0]
        if x <= xs[0]:
            return ys[0] + (x - xs[0]) / (xs[1] - xs[0]) * (ys[1] - ys[0])
        if x >= xs[-1]:
            return ys[-1] + (x - xs[-1]) / (xs[-1] - xs[-2]) * (ys[-1] - ys[-2])
        i = bisect.bisect_right(xs, x) - 1
        return ys[i] + (x - xs[i]) / (xs[i + 1] - xs[i]) * (ys[i + 1] - ys[i])

    # -- algebra used by the fold -------------------------------------------
    @staticmethod
    def affine(
        f: "PiecewiseLinear", g: "PiecewiseLinear", cf: float, cg: float
    ) -> "PiecewiseLinear":
        """Return cf * f + cg * g (exact: PWL is closed under affine combos)."""
        grid = sorted(set(f.xs) | set(g.xs))
        return PiecewiseLinear(grid, [cf * f(x) + cg * g(x) for x in grid])

    def maximum(self, other: "PiecewiseLinear") -> "PiecewiseLinear":
        """Pointwise max, inserting any interior crossing so the result is exact."""
        grid = sorted(set(self.xs) | set(other.xs))
        xs: List[float] = []
        ys: List[float] = []
        for i in range(len(grid) - 1):
            a, b = grid[i], grid[i + 1]
            xs.append(a)
            ys.append(max(self(a), other(a)))
            diff_a = self(a) - other(a)
            diff_b = self(b) - other(b)
            if diff_a * diff_b < 0:  # they cross strictly inside (a, b)
                t = diff_a / (diff_a - diff_b)
                xc = a + t * (b - a)
                xs.append(xc)
                ys.append(self(xc))  # equals other(xc) at the crossing
        xs.append(grid[-1])
        ys.append(max(self(grid[-1]), other(grid[-1])))
        return PiecewiseLinear(xs, ys)

    def parallel_with(self, other: "PiecewiseLinear") -> "PiecewiseLinear":
        """The Par rule with unit reward:

            Psi_Par(gamma) = 1 - integral_gamma^1 self'(z) * other'(z) dz.

        Both branch derivatives are piecewise constant, so the integrand is
        piecewise constant and we accumulate it backwards from gamma = 1.
        """
        grid = sorted(set(self.xs) | set(other.xs))
        ys = [0.0] * len(grid)
        ys[-1] = 1.0  # Psi_Par(1) = 1
        for i in range(len(grid) - 2, -1, -1):
            a, b = grid[i], grid[i + 1]
            slope_self = (self(b) - self(a)) / (b - a)
            slope_other = (other(b) - other(a)) / (b - a)
            ys[i] = ys[i + 1] - slope_self * slope_other * (b - a)
        return PiecewiseLinear(grid, ys)

    @staticmethod
    def parallel_many(profiles: List["PiecewiseLinear"]) -> "PiecewiseLinear":
        """N-ary race of k >= 2 branches: the k-way generalisation of
        `parallel_with`, where the integrand is the PRODUCT of all branch
        derivatives (independent branches -> product survival):

            Psi_Par(gamma) = 1 - integral_gamma^1 prod_i g_i'(z) dz.
        """
        grid = sorted(set().union(*(set(p.xs) for p in profiles)))
        ys = [0.0] * len(grid)
        ys[-1] = 1.0
        for i in range(len(grid) - 2, -1, -1):
            a, b = grid[i], grid[i + 1]
            slope_product = 1.0
            for p in profiles:
                slope_product *= (p(b) - p(a)) / (b - a)
            ys[i] = ys[i + 1] - slope_product * (b - a)
        return PiecewiseLinear(grid, ys)

    @staticmethod
    def series(
        downstream: "PiecewiseLinear", upstream: "PiecewiseLinear"
    ) -> "PiecewiseLinear":
        """The Series rule (upstream block first, then downstream block):

            Psi_Series(gamma) = R(gamma) * P(gamma / R(gamma)),

        where R = downstream value, P = upstream value. On each cell where R is
        linear and P is linear in its argument, the product is again linear in
        gamma, so the result is exactly PWL. Its breakpoints are the knots of R
        together with the gammas at which the argument u(gamma) = gamma / R(gamma)
        hits a knot of P.

        Linear-merge implementation. Because R is nondecreasing, 1-Lipschitz, and
        R(gamma) >= gamma, the argument u(gamma) = gamma / R(gamma) is monotone
        nondecreasing. Hence u evaluated at R's knots is already sorted, and each
        upstream knot's pre-image can be found by advancing a single pointer over
        R's pieces (a merge), rather than testing every (R-piece, P-knot) pair.
        This costs O(|R| + |P|) instead of O(|R| * |P|).
        """
        R, P = downstream, upstream
        Rx, Ry = R.xs, R.ys

        def argument(gamma: float, r_value: float) -> float:
            """u(gamma) = gamma / R(gamma), clamped to [0, 1] (0 if R(gamma) ~ 0)."""
            return 0.0 if r_value <= 1e-15 else min(max(gamma / r_value, 0.0), 1.0)

        # u at each downstream knot -- sorted, since u is monotone.
        u_at_knot = [argument(Rx[i], Ry[i]) for i in range(len(Rx))]

        # Pre-image gamma of each interior upstream knot, in increasing order,
        # located by a single forward sweep over R's pieces.
        preimages: List[float] = []
        piece = 0
        for u in P.xs:
            if u <= 1e-15 or u >= 1.0 - 1e-15:
                continue  # u = 0 -> gamma = 0; u = 1 -> gamma = alpha_R; both are R knots
            # If u exceeds every knot (u_at_knot[-1] < u, e.g. R's last knot < 1),
            # stay on the LAST piece -- the pre-image is clamped below. `piece + 2 <
            # len(Rx)` keeps piece <= len(Rx) - 2 so Rx[piece + 1] stays in range.
            while piece + 2 < len(Rx) and u_at_knot[piece + 1] < u:
                piece += 1
            ax, bx = Rx[piece], Rx[piece + 1]
            ay, by = Ry[piece], Ry[piece + 1]
            slope = (by - ay) / (bx - ax)
            intercept = ay - slope * ax  # R(gamma) = intercept + slope * gamma here
            denom = 1.0 - u * slope  # solve gamma / R(gamma) = u for gamma
            if abs(denom) < 1e-15:
                continue
            preimages.append(min(max(u * intercept / denom, ax), bx))

        # Merge the two already-sorted breakpoint sources (R's knots and the
        # pre-images), dropping duplicates.
        breakpoints: List[float] = []
        i = j = 0
        while i < len(Rx) or j < len(preimages):
            if j >= len(preimages) or (i < len(Rx) and Rx[i] <= preimages[j]):
                candidate = Rx[i]
                i += 1
            else:
                candidate = preimages[j]
                j += 1
            candidate = min(max(candidate, 0.0), 1.0)
            if not breakpoints or candidate - breakpoints[-1] > 1e-12:
                breakpoints.append(candidate)

        # Evaluate the exact product at each breakpoint.
        values = []
        for gamma in breakpoints:
            r_value = R(gamma)
            values.append(r_value * P(argument(gamma, r_value)))
        return PiecewiseLinear(breakpoints, values)

    def index_crossing(self) -> float:
        """Smallest gamma in [0, 1] with self(gamma) = gamma (the index).

        Convexity guarantees a single crossing; `self - identity` is decreasing
        until it first reaches 0, so we return that root.
        """
        diff = PiecewiseLinear.affine(self, PiecewiseLinear.identity(), 1.0, -1.0)
        xs, ys = diff.xs, diff.ys
        for i in range(len(xs)):
            if ys[i] <= 1e-12:
                if i == 0:
                    return xs[0]
                y0, y1, x0, x1 = ys[i - 1], ys[i], xs[i - 1], xs[i]
                return x0 if y0 == y1 else x0 + y0 / (y0 - y1) * (x1 - x0)
        return xs[-1]


# Shared constants (built once).
_IDENTITY = PiecewiseLinear.identity()
_UNIT_REWARD = PiecewiseLinear.constant(1.0)


def _control_recursion(
    control: Control, reward: PiecewiseLinear
) -> Tuple[PiecewiseLinear, List[float]]:
    """Solve the single-control Bellman recursion against a reward *profile*.

        Psi(.; k) = max{ gamma, beta[ p(k) * reward + (1 - p(k)) * Psi(.; k+1) ] },
        Psi(.; q) = gamma.

    Returns the head profile Psi(.; 0) and the list of indices [alpha(.,0), ...,
    alpha(.,q-1)]. With `reward = unit`, the head profile is the leaf's value
    (Phase 1); with a threaded downstream reward, the indices are the coupled
    Gittins indices (Phase 2).
    """
    beta = control.beta
    psi = _IDENTITY  # Psi(.; q) = gamma
    indices = [0.0] * control.q
    for k in range(control.q - 1, -1, -1):
        p = control.ps[k]
        continue_value = PiecewiseLinear.affine(reward, psi, beta * p, beta * (1.0 - p))
        psi = _IDENTITY.maximum(continue_value)
        indices[k] = psi.index_crossing()
    return psi, indices


def _is_monotone(control: Control) -> bool:
    """Whether the success probabilities are non-increasing (Assumption 2)."""
    ps = control.ps
    return all(ps[i] >= ps[i + 1] - 1e-15 for i in range(len(ps) - 1))


def _solve_leaf_closed_form(
    control: Control, reward: PiecewiseLinear
) -> Tuple[PiecewiseLinear, List[float]]:
    """Closed-form single-control solve against a reward *profile*, in
    O(q_v + |reward|) time.

    Returns the same head profile Psi(.; 0) and indices as `_control_recursion`,
    but without rebuilding the profile at every failure count. Requires a
    monotone control; two facts (both from Assumption 2) make it linear:

    * Indices. alpha(v, k) is the unique fixed point gamma = c_hat(v, k) * R(gamma),
      and c_hat(v, k) is non-increasing in k, so the fixed points are too. A
      single pointer descending over R's linear pieces locates all q_v of them.

    * Head profile. Where the optimal policy attacks exactly k times before
      stopping, the value is A_k * R(gamma) + B_k * gamma, with
          B_k = prod_{j<k} beta (1 - p(j)),
          A_k = sum_{j<k} [prod_{i<j} beta (1 - p(i))] beta p(j),
      accumulated in O(1) per k. The region boundaries are exactly the
      alpha(v, k), which we merge with R's own knots.
    """
    beta, ps, q = control.beta, control.ps, control.q
    Rx, Ry = reward.xs, reward.ys
    chat = [beta * p / (1.0 - beta * (1.0 - p)) for p in ps]  # non-increasing in k

    # -- indices: alpha(v, k) = fixed point of gamma = chat[k] * R(gamma) -------
    # Solutions decrease with k, so sweep R's pieces top-down with one pointer.
    alphas = [0.0] * q
    piece = len(Rx) - 2
    for k in range(q):
        ck = chat[k]
        while True:
            ax, bx = Rx[piece], Rx[piece + 1]
            ay, by = Ry[piece], Ry[piece + 1]
            b = (by - ay) / (bx - ax)
            a = ay - b * ax  # R(gamma) = a + b * gamma on this piece
            denom = 1.0 - ck * b
            gamma = ck * a / denom if abs(denom) > 1e-15 else 0.0
            if gamma >= ax - 1e-9 or piece == 0:
                break
            piece -= 1
        alphas[k] = min(max(gamma, ax), bx)

    # -- head profile Psi(.; 0): value A_k * R + B_k * gamma on region k --------
    A = [0.0] * (q + 1)
    B = [1.0] * (q + 1)
    for k in range(1, q + 1):
        B[k] = B[k - 1] * beta * (1.0 - ps[k - 1])
        A[k] = A[k - 1] + B[k - 1] * beta * ps[k - 1]

    # Breakpoints = region boundaries (the alphas) merged with R's own knots.
    knots = {0.0, 1.0}
    knots.update(alphas)
    knots.update(min(max(x, 0.0), 1.0) for x in Rx)
    xs = sorted(knots)
    ascending_alphas = sorted(alphas)
    ys: List[float] = []
    for gamma in xs:
        # region k = number of attacks before stopping = q - (#alphas <= gamma)
        k = q - bisect.bisect_right(ascending_alphas, gamma + 1e-12)
        k = 0 if k < 0 else q if k > q else k
        ys.append(A[k] * reward(gamma) + B[k] * gamma)
    return PiecewiseLinear(xs, ys), alphas


def _leaf_solve(
    control: Control, reward: PiecewiseLinear
) -> Tuple[PiecewiseLinear, List[float]]:
    """Single-control solve used by both passes. Uses the O(q + |R|) closed form
    for monotone controls (Assumption 2) and falls back to the exact O(q|R|)
    recursion otherwise, so correctness never depends on monotonicity.
    """
    if _is_monotone(control):
        return _solve_leaf_closed_form(control, reward)
    return _control_recursion(control, reward)


# ===========================================================================
# Phase 1 -- the value fold
# ===========================================================================


@lru_cache(maxsize=None)
def value_profile(net: Network) -> PiecewiseLinear:
    """Phase 1: the calibrated value Psi_net as a function of the buyout gamma.

    Native n-ary fold, bottom-up over the parse tree:
        Control : single-control recursion against a unit reward;
        Par     : 1 - integral of the PRODUCT of all branch derivatives;
        Series  : suffix fold of the series rule over the children in order.

    (Series and Par are associative, so this equals the right-folded binary
    result; Par is also commutative.) Profiles are immutable (lru_cache shares them).
    """
    if isinstance(net, Control):
        head_profile, _ = _leaf_solve(net, _UNIT_REWARD)
        return head_profile
    if isinstance(net, Par):
        return PiecewiseLinear.parallel_many(
            [value_profile(child) for child in net.children])
    # Series: value = series(downstream = value(rest), upstream = value(first)),
    # accumulated right-to-left over the children.
    children = net.children
    profile = value_profile(children[-1])
    for child in reversed(children[:-1]):
        profile = PiecewiseLinear.series(profile, value_profile(child))
    return profile


def game_value(net: Network) -> float:
    """The optimal attacker's expected discounted reward in the true game.

    This is the value at zero buyout, Psi_net(0).
    """
    return value_profile(net)(0.0)


# ===========================================================================
# Phase 2 -- index threading
# ===========================================================================

ControlState = Tuple[str, int]  # (control name, failure count k)


def index_table(net: Network) -> Dict[ControlState, float]:
    """Phase 2: the index alpha(v, k) of every control state in `net`.

    Each control is solved against the value profile of *its forward cone* (the
    prize unlocked by clearing it), threaded through the tree from the unit reward
    at the root (native n-ary):

    * Par threads the SAME incoming reward to EVERY branch. Winning any one branch
      completes the race, so a branch's forward cone never involves its siblings --
      this independence is exactly the Gittins decomposition ("alpha(i, j, k) is a
      function of chain i alone").
    * Series threads the reward through the value profiles of the DOWNSTREAM
      siblings: control c_i (i-th of k, in order) sees
          series(reward, value(composite(c_{i+1..k}))),
      built by sweeping the children right-to-left while extending the downstream
      cone. The last child sees the incoming reward directly.
    * Control records the crossings from the single-control recursion.
    """
    table: Dict[ControlState, float] = {}

    def collect(node: Network, reward: PiecewiseLinear) -> None:
        if isinstance(node, Control):
            _, indices = _leaf_solve(node, reward)
            for k, alpha in enumerate(indices):
                table[(node.name, k)] = alpha
        elif isinstance(node, Par):
            for child in node.children:          # every branch shares the reward
                collect(child, reward)
        else:  # Series: sweep right-to-left, extending the downstream forward cone
            downstream: Optional[PiecewiseLinear] = None
            for child in reversed(node.children):
                if downstream is None:           # last child: incoming reward directly
                    collect(child, reward)
                else:
                    collect(child, PiecewiseLinear.series(reward, downstream))
                downstream = (
                    value_profile(child) if downstream is None
                    else PiecewiseLinear.series(downstream, value_profile(child)))

    collect(net, _UNIT_REWARD)
    return table


# ===========================================================================
# Phase 3 -- online execution of the greedy index policy
# ===========================================================================


class Status(Enum):
    """Resolution status of a block (or of a single control)."""

    LIVE = "live"  # still attackable / in progress
    WON = "won"  # cleared / target reached
    DEAD = "dead"  # jammed or stranded -- no route through this block


# An attack state maps each control name to its current status:
#   an int k    -> active, with k failures so far (attackable at count k)
#   Status.WON  -> cleared
#   Status.DEAD -> jammed
AttackState = Dict[str, Union[int, Status]]


def initial_state(net: Network) -> AttackState:
    """Fresh state: every control active at failure count 0."""
    return {c.name: 0 for c in controls(net)}


def block_status(net: Network, state: AttackState) -> Status:
    """Status of a block, derived from its controls.

    Series is dead if its left dies, won once both halves are won, otherwise
    follows the right half after the left is cleared. Par (a race) is won as
    soon as either branch wins and dead only if both branches die. This is what
    implicitly realises 'win prunes sibling' and 'jam strands downstream'.
    """
    if isinstance(net, Control):
        s = state[net.name]
        return s if isinstance(s, Status) else Status.LIVE
    statuses = [block_status(child, state) for child in net.children]
    if isinstance(net, Series):
        for status in statuses:                  # the first not-yet-won child decides
            if status is Status.DEAD:
                return Status.DEAD
            if status is Status.LIVE:
                return Status.LIVE
        return Status.WON                         # all children won
    # Par
    if any(status is Status.WON for status in statuses):
        return Status.WON
    if all(status is Status.DEAD for status in statuses):
        return Status.DEAD
    return Status.LIVE


def frontier(net: Network, state: AttackState) -> List[ControlState]:
    """The controls currently attackable, as (name, k) pairs.

    Only live blocks contribute; a series exposes its right half only after the
    left is won; a parallel exposes both live branches.
    """
    if block_status(net, state) is not Status.LIVE:
        return []
    if isinstance(net, Control):
        return [(net.name, state[net.name])]  # state value is the int count
    if isinstance(net, Series):
        for child in net.children:               # skip won children, expose the first live one
            if block_status(child, state) is Status.WON:
                continue
            return frontier(child, state)
        return []
    result: List[ControlState] = []               # Par: expose every live branch
    for child in net.children:
        result += frontier(child, state)
    return result


def apply_outcome(state: AttackState, control: Control, success: bool) -> AttackState:
    """Return a new state after attacking `control` with the given outcome.

    Success clears the control; failure increments its count, jamming it (DEAD)
    once it would reach `q`.
    """
    new_state = dict(state)
    if success:
        new_state[control.name] = Status.WON
    else:
        k = state[control.name] + 1  # one more failure
        new_state[control.name] = Status.DEAD if k >= control.q else k
    return new_state


@dataclass
class Rollout:
    """Outcome of one simulated attack."""

    won: bool
    reward: float  # realised discounted reward (0 if every route died)
    attempts: int
    history: List[Tuple[str, int, bool]] = field(default_factory=list)  # (name, k, success)


class IndexAttacker:
    """The optimal attacker: precompute indices once, then play greedily.

    Usage
    -----
        attacker = IndexAttacker(net)
        attacker.indices[("A", 0)]          # the index of state (A, 0)
        name = attacker.action(state)       # which control to attack now
        result = attacker.simulate(rng)     # one stochastic trajectory
        attacker.monte_carlo_value(10_000)  # estimate of the game value
    """

    def __init__(self, net: Network) -> None:
        self.net = net
        self.indices: Dict[ControlState, float] = index_table(net)
        self._control_by_name: Dict[str, Control] = {c.name: c for c in controls(net)}

    def action(self, state: AttackState) -> Optional[str]:
        """The control to attack in `state`: the live one of highest index.

        Returns None only if there is no live control (the game is already won
        or lost). In the true game every live control has a positive index, so
        the attacker never voluntarily stops.
        """
        live = frontier(self.net, state)
        if not live:
            return None
        # Highest index wins; break ties by name for reproducibility.
        best_name, _ = max(live, key=lambda nk: (self.indices[nk], nk[0]))
        return best_name

    def simulate(self, rng: Optional[random.Random] = None) -> Rollout:
        """Play one trajectory under the greedy index policy.

        The realised discounted reward is the product of the attempted controls'
        `beta`s if the target is reached, and 0 if every route dies.
        """
        rng = rng or random.Random()
        state = initial_state(self.net)
        discount = 1.0
        attempts = 0
        history: List[Tuple[str, int, bool]] = []
        while True:
            status = block_status(self.net, state)
            if status is Status.WON:
                return Rollout(True, discount, attempts, history)
            if status is Status.DEAD:
                return Rollout(False, 0.0, attempts, history)
            name = self.action(state)
            assert name is not None  # LIVE implies a non-empty frontier
            control = self._control_by_name[name]
            k = state[name]
            success = rng.random() < control.ps[k]
            discount *= control.beta
            attempts += 1
            history.append((name, k, success))
            state = apply_outcome(state, control, success)

    def monte_carlo_value(
        self, num_samples: int, rng: Optional[random.Random] = None
    ) -> float:
        """Average realised reward over `num_samples` rollouts.

        Converges to `game_value(net)` as the sample size grows -- a cheap check
        that Phases 1-3 are mutually consistent.
        """
        rng = rng or random.Random()
        return sum(self.simulate(rng).reward for _ in range(num_samples)) / num_samples


# ===========================================================================
# Demonstration
# ===========================================================================

if __name__ == "__main__":
    # Nested, non-parallel-chains example: clear (A OR B), then C.
    A = Control("A", 0.9, (0.5, 0.2))
    B = Control("B", 0.9, (0.8, 0.4))
    C = Control("C", 0.9, (0.6, 0.3))
    network = Series(Par(A, B), C)

    print("Phase 1 - value fold")
    print(f"  game value Psi_N(0) = {game_value(network):.6f}\n")

    print("Phase 2 - index threading (priority order)")
    table = index_table(network)
    for (name, k), alpha in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  alpha({name},{k}) = {alpha:.6f}")
    print()

    print("Phase 3 - online execution")
    attacker = IndexAttacker(network)
    rng = random.Random(0)
    for trial in range(3):
        result = attacker.simulate(rng)
        trail = " -> ".join(
            f"{name}@{k}:{'OK' if ok else 'x'}" for name, k, ok in result.history
        )
        outcome = "WIN" if result.won else "dead"
        print(f"  rollout {trial}: {trail}  ==> {outcome}, reward={result.reward:.4f}")

    estimate = attacker.monte_carlo_value(50_000, random.Random(1))
    print(f"\n  Monte-Carlo value (50k rollouts) = {estimate:.6f}")
    print(f"  exact game value                 = {game_value(network):.6f}")
    