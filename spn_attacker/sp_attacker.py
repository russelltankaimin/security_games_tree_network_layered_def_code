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
A network is built from single *controls* by two constructors:

    Series(G1, G2)  : clear G1, then clear G2          (sequential)
    Par(G1, G2)     : clear G1 OR G2 -- a race          (either branch wins)

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

The whole pipeline runs in O(Q^2) time, where Q is the total number of control
states, sidestepping the exponential joint-state MDP.
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


@dataclass(frozen=True)
class Series:
    """Sequential composition: clear `left`, then clear `right`."""

    left: "Network"
    right: "Network"


@dataclass(frozen=True)
class Par:
    """Parallel (OR) composition: completing either branch wins the block."""

    left: "Network"
    right: "Network"


Network = Union[Control, Series, Par]


def controls(net: Network) -> List[Control]:
    """Return all controls (leaves) of `net`, left-to-right."""
    if isinstance(net, Control):
        return [net]
    return controls(net.left) + controls(net.right)


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
    def series(
        downstream: "PiecewiseLinear", upstream: "PiecewiseLinear"
    ) -> "PiecewiseLinear":
        """The Series rule (upstream block first, then downstream block):

            Psi_Series(gamma) = R(gamma) * P(gamma / R(gamma)),

        where R = downstream value, P = upstream value. On each cell where R is
        linear (R = u + v*gamma) and P is linear in its argument, the product is
        again *linear in gamma* -- so the result is exactly PWL. The breakpoints
        are those of R together with the gammas at which gamma / R(gamma) hits a
        breakpoint of P.
        """
        R, P = downstream, upstream
        knots = set(R.xs)
        for i in range(len(R.xs) - 1):
            ax, bx = R.xs[i], R.xs[i + 1]
            ay, by = R.ys[i], R.ys[i + 1]
            slope = (by - ay) / (bx - ax)
            intercept = ay - slope * ax  # R(gamma) = intercept + slope * gamma here
            for u in P.xs:
                denom = 1.0 - u * slope  # solve gamma / R(gamma) = u
                if abs(denom) < 1e-15:
                    continue
                gamma = u * intercept / denom
                if ax - 1e-12 <= gamma <= bx + 1e-12:
                    knots.add(min(max(gamma, ax), bx))
        xs = sorted(g for g in knots if -1e-9 <= g <= 1 + 1e-9)
        ys = []
        for gamma in xs:
            Rg = R(gamma)
            arg = 0.0 if Rg <= 1e-15 else min(max(gamma / Rg, 0.0), 1.0)
            ys.append(Rg * P(arg))
        return PiecewiseLinear(xs, ys)

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


# ===========================================================================
# Phase 1 -- the value fold
# ===========================================================================


@lru_cache(maxsize=None)
def value_profile(net: Network) -> PiecewiseLinear:
    """Phase 1: the calibrated value Psi_net as a function of the buyout gamma.

    Computed bottom-up over the parse tree:
        Control  : the single-control recursion against a unit reward;
        Par(L,R) : combine child values by the parallel rule;
        Series(L,R): compose child values by the series rule (R downstream).

    Returned profiles are treated as immutable (the lru_cache shares them).
    """
    if isinstance(net, Control):
        head_profile, _ = _control_recursion(net, _UNIT_REWARD)
        return head_profile
    if isinstance(net, Par):
        return value_profile(net.left).parallel_with(value_profile(net.right))
    # Series: upstream = left is solved against downstream = value of right.
    return PiecewiseLinear.series(value_profile(net.right), value_profile(net.left))


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
    prize unlocked by clearing it). We thread that downstream reward profile
    through the tree, starting from the unit reward at the root:
        Series(L, R): R keeps the incoming reward; L sees `series(reward, value(R))`;
        Par(L, R)   : both branches inherit the same reward;
        Control     : record the crossings from the single-control recursion.
    """
    table: Dict[ControlState, float] = {}

    def collect(node: Network, reward: PiecewiseLinear) -> None:
        if isinstance(node, Control):
            _, indices = _control_recursion(node, reward)
            for k, alpha in enumerate(indices):
                table[(node.name, k)] = alpha
        elif isinstance(node, Par):
            collect(node.left, reward)
            collect(node.right, reward)
        else:  # Series: clearing left leads to right, which leads to `reward`.
            collect(node.right, reward)
            collect(node.left, PiecewiseLinear.series(reward, value_profile(node.right)))

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
    left, right = block_status(net.left, state), block_status(net.right, state)
    if isinstance(net, Series):
        if left is Status.DEAD:
            return Status.DEAD
        if left is Status.WON:
            return right
        return Status.LIVE
    # Par
    if left is Status.WON or right is Status.WON:
        return Status.WON
    if left is Status.DEAD and right is Status.DEAD:
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
        if block_status(net.left, state) is Status.WON:
            return frontier(net.right, state)
        return frontier(net.left, state)
    return frontier(net.left, state) + frontier(net.right, state)


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