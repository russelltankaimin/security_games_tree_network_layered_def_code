"""
other_policies.py
=================

Baseline attacker policies for comparison against the optimal index policy
(`IndexAttacker` in `sp_attacker.py`). Every policy exposes the same interface

    action(state) -> Optional[str]      # which live control to attack, or None to stop

so a single `simulate` driver can roll any of them out on a network. The
baselines are deliberately spread from "ignores almost nothing" to "ignores
almost everything":

    GreedyProbabilityPolicy : attack the highest one-step success prob p(k)
    GreedyDiscountedPolicy  : attack the highest discounted success beta*p(k)
    UncoupledIndexPolicy    : attack the highest *single-control* index c_hat(v,k)
                              (the index policy minus the downstream coupling)
    SingleRoutePolicy       : commit to one best route; never pivot across routes
    FixedPriorityPolicy     : structure-blind, fixed lexicographic priority
    RandomFrontierPolicy    : attack a uniformly random live control

`UncoupledIndexPolicy` is the most informative baseline: comparing it with the
optimal index policy isolates exactly the value of accounting for the forward
cone (the coupling), since the two differ only in that term.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from sp_attacker import (
    Control,
    Network,
    Par,
    Rollout,
    Series,
    Status,
    apply_outcome,
    block_status,
    controls,
    frontier,
    game_value,
    initial_state,
    single_control_index,
)


def _control_map(net: Network) -> Dict[str, Control]:
    """Map each control name to its Control object (for O(1) lookup)."""
    return {c.name: c for c in controls(net)}


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


class Policy(ABC):
    """A deterministic-by-default attack policy.

    Subclasses set `name` and implement `action`. `deterministic` flags whether
    the policy's choice is a function of the state alone (used downstream to
    decide whether an exact, noise-free policy value can be computed).
    """

    name: str = "policy"
    deterministic: bool = True

    @abstractmethod
    def action(self, state) -> Optional[str]:
        """Return the name of the control to attack, or None to stop."""
        raise NotImplementedError


class _ScorePolicy(Policy):
    """Convenience base: attack the live control maximising `score`.

    Ties are broken by control name so that runs are reproducible.
    """

    def __init__(self, net: Network) -> None:
        self.net = net
        self._cmap = _control_map(net)

    def score(self, control: Control, k: int) -> float:
        raise NotImplementedError

    def action(self, state) -> Optional[str]:
        live = frontier(self.net, state)
        if not live:
            return None
        best = max(live, key=lambda nk: (self.score(self._cmap[nk[0]], nk[1]), nk[0]))
        return best[0]


# ---------------------------------------------------------------------------
# Greedy / myopic baselines
# ---------------------------------------------------------------------------


class GreedyProbabilityPolicy(_ScorePolicy):
    """Attack the control most likely to be cracked on the next attempt."""

    name = "greedy_probability"

    def score(self, control: Control, k: int) -> float:
        return control.ps[k]


class GreedyDiscountedPolicy(_ScorePolicy):
    """Attack the control with the highest discounted one-step success beta*p(k)."""

    name = "greedy_discounted"

    def score(self, control: Control, k: int) -> float:
        return control.beta * control.ps[k]


class UncoupledIndexPolicy(_ScorePolicy):
    """Attack the highest *single-control* index c_hat(v,k), ignoring the work
    that still lies downstream. This is the index policy with the forward-cone
    coupling removed."""

    name = "uncoupled_index"

    def score(self, control: Control, k: int) -> float:
        return single_control_index(control.beta, control.ps[k])


# ---------------------------------------------------------------------------
# Structure-blind / non-adaptive baselines
# ---------------------------------------------------------------------------


class FixedPriorityPolicy(Policy):
    """Attack live controls in a fixed lexicographic order (no use of structure
    or state)."""

    name = "fixed_priority"

    def __init__(self, net: Network) -> None:
        self.net = net

    def action(self, state) -> Optional[str]:
        live = frontier(self.net, state)
        if not live:
            return None
        return min(name for name, _ in live)


class RandomFrontierPolicy(Policy):
    """Attack a uniformly random live control. Holds its own RNG so that the
    simulation outcome RNG and the policy's choices are independent and each
    reproducible."""

    name = "random_frontier"
    deterministic = False

    def __init__(self, net: Network, rng: Optional[random.Random] = None) -> None:
        self.net = net
        self.rng = rng or random.Random()

    def action(self, state) -> Optional[str]:
        live = frontier(self.net, state)
        if not live:
            return None
        return self.rng.choice(live)[0]


def source_target_routes(net: Network) -> List[List[str]]:
    """Enumerate every source-to-target path as an ordered list of control names.

    Series concatenates the two halves; Par offers the union of each branch's
    routes; a control is a length-one route.
    """
    if isinstance(net, Control):
        return [[net.name]]
    if isinstance(net, Par):
        return source_target_routes(net.left) + source_target_routes(net.right)
    return [
        left + right
        for left in source_target_routes(net.left)
        for right in source_target_routes(net.right)
    ]


class SingleRoutePolicy(Policy):
    """Commit to the single best route (highest standalone chain value) and only
    attack along it; never pivot to another route. If that route can no longer
    progress, give up (return None) -- the non-adaptive "constant" heuristic."""

    name = "single_route"

    def __init__(self, net: Network) -> None:
        self.net = net
        self._cmap = _control_map(net)
        routes = source_target_routes(net)
        self.route = max(routes, key=lambda r: game_value(self._as_chain(r)))

    def _as_chain(self, names: List[str]) -> Network:
        """Build the Series chain of the given controls (to score a route)."""
        node: Network = self._cmap[names[-1]]
        for name in reversed(names[:-1]):
            node = Series(self._cmap[name], node)
        return node

    def action(self, state) -> Optional[str]:
        live = {name for name, _ in frontier(self.net, state)}
        for name in self.route:  # attack the next exposed control on the route
            if name in live:
                return name
        return None


# ---------------------------------------------------------------------------
# Generic simulator and a convenience factory
# ---------------------------------------------------------------------------


def simulate(net: Network, policy, rng: random.Random) -> Rollout:
    """Roll out any policy once and return the realised outcome.

    The realised discounted reward is the product of the attempted controls'
    `beta`s if the target is reached, else 0 (every route died, or the policy
    chose to stop). Works for the index policy and every baseline alike, since
    all expose `action(state)`.
    """
    cmap = _control_map(net)
    state = initial_state(net)
    discount = 1.0
    attempts = 0
    history = []
    while True:
        status = block_status(net, state)
        if status is Status.WON:
            return Rollout(True, discount, attempts, history)
        if status is Status.DEAD:
            return Rollout(False, 0.0, attempts, history)
        name = policy.action(state)
        if name is None:  # policy voluntarily stops -> no reward
            return Rollout(False, 0.0, attempts, history)
        control = cmap[name]
        k = state[name]
        success = rng.random() < control.ps[k]
        discount *= control.beta
        attempts += 1
        history.append((name, k, success))
        state = apply_outcome(state, control, success)


def baseline_policies(
    net: Network, rng: Optional[random.Random] = None
) -> Dict[str, Policy]:
    """Construct one instance of every baseline policy, keyed by its name."""
    rng = rng or random.Random(0)
    policies: List[Policy] = [
        GreedyProbabilityPolicy(net),
        GreedyDiscountedPolicy(net),
        UncoupledIndexPolicy(net),
        SingleRoutePolicy(net),
        FixedPriorityPolicy(net),
        RandomFrontierPolicy(net, rng),
    ]
    return {p.name: p for p in policies}
