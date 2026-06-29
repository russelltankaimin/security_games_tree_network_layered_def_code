"""
test_sp_attacker.py
===================

Regression tests for `sp_attacker.py`. Run with:  pytest -q

The strongest checks use a brute-force MDP oracle (`_mdp_optimal_value`) that
solves the game by exhaustive backward induction over the joint state space --
a computation completely independent of the piecewise-linear fold. We assert:

    Phase 1 (fold)            == MDP optimum            (value is correct)
    Phase 3 (index policy)    == MDP optimum            (the policy is optimal)

plus structural invariants (index monotonicity, the series discount, the OR
gain), the single-control closed form, the win/jam reward semantics, frontier
behaviour, and Monte-Carlo consistency.
"""

import random

import pytest

import sp_attacker as spa
from sp_attacker import Control, Par, Series, IndexAttacker, game_value, index_table


# ---------------------------------------------------------------------------
# Test helpers: an independent brute-force MDP oracle and a random generator
# ---------------------------------------------------------------------------


def _state_key(state):
    """Hashable key for an attack state (dict name -> int|Status)."""
    return tuple(sorted(state.items(), key=lambda kv: kv[0]))


def _mdp_optimal_value(net):
    """Optimal game value by exhaustive backward induction over joint states.

    Uses the model's transition rules (status/frontier/apply_outcome) but, unlike
    the index method, maximises over *every* frontier action at every state.
    """
    cmap = {c.name: c for c in spa.controls(net)}
    cache = {}

    def value(state):
        status = spa.block_status(net, state)
        if status is spa.Status.WON:
            return 1.0
        if status is spa.Status.DEAD:
            return 0.0
        key = _state_key(state)
        if key in cache:
            return cache[key]
        best = 0.0  # the buyout-0 "stop" option, dominated whenever a route lives
        for name, k in spa.frontier(net, state):
            ctrl = cmap[name]
            p = ctrl.ps[k]
            succ = value(spa.apply_outcome(state, ctrl, True))
            fail = value(spa.apply_outcome(state, ctrl, False))
            best = max(best, ctrl.beta * (p * succ + (1.0 - p) * fail))
        cache[key] = best
        return best

    return value(spa.initial_state(net))


def _mdp_policy_value(net, attacker):
    """Exact value of a fixed (deterministic) policy via backward induction."""
    cmap = {c.name: c for c in spa.controls(net)}
    cache = {}

    def value(state):
        status = spa.block_status(net, state)
        if status is spa.Status.WON:
            return 1.0
        if status is spa.Status.DEAD:
            return 0.0
        key = _state_key(state)
        if key in cache:
            return cache[key]
        name = attacker.action(state)
        ctrl = cmap[name]
        p = ctrl.ps[state[name]]
        succ = value(spa.apply_outcome(state, ctrl, True))
        fail = value(spa.apply_outcome(state, ctrl, False))
        result = ctrl.beta * (p * succ + (1.0 - p) * fail)
        cache[key] = result
        return result

    return value(spa.initial_state(net))


def _random_network(rng, n_leaves, qmax=2):
    """A random SP tree with exactly `n_leaves` monotone controls."""
    counter = [0]

    def leaf():
        name = f"v{counter[0]}"
        counter[0] += 1
        q = rng.randint(1, qmax)
        ps = tuple(sorted((round(rng.uniform(0.2, 0.9), 3) for _ in range(q)), reverse=True))
        return Control(name, round(rng.uniform(0.8, 0.95), 3), ps)

    def build(n):
        if n == 1:
            return leaf()
        left = rng.randint(1, n - 1)
        op = Par if rng.random() < 0.5 else Series
        return op(build(left), build(n - left))

    return build(n_leaves)


# A small, fixed, nested instance reused across several tests.
A = Control("A", 0.9, (0.5, 0.2))
B = Control("B", 0.9, (0.8, 0.4))
C = Control("C", 0.9, (0.6, 0.3))
WORKED = Series(Par(A, B), C)

TOL = 1e-9


# ---------------------------------------------------------------------------
# PiecewiseLinear primitives
# ---------------------------------------------------------------------------


def test_pwl_evaluation_and_affine():
    f = spa.PiecewiseLinear([0.0, 0.5, 1.0], [0.0, 0.25, 1.0])
    assert f(0.5) == pytest.approx(0.25)
    assert f(0.25) == pytest.approx(0.125)  # interpolated on the first piece
    g = spa.PiecewiseLinear.affine(f, spa.PiecewiseLinear.identity(), 2.0, 1.0)
    assert g(0.5) == pytest.approx(2 * 0.25 + 0.5)


def test_pwl_maximum_inserts_crossing():
    line = spa.PiecewiseLinear.identity()
    flat = spa.PiecewiseLinear.constant(0.4)
    top = line.maximum(flat)
    assert top(0.2) == pytest.approx(0.4)  # flat wins on the left
    assert top(0.8) == pytest.approx(0.8)  # line wins on the right
    assert 0.4 in [pytest.approx(x) for x in top.xs]  # crossing was inserted


def test_index_crossing_of_constant():
    # f(gamma) = c crosses the diagonal at gamma = c.
    assert spa.PiecewiseLinear.constant(0.6).index_crossing() == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Phase 1 / Phase 2: single-control closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta,p", [(0.9, 0.5), (0.8, 0.8), (0.95, 0.3), (0.85, 0.6)])
def test_single_control_index_closed_form(beta, p):
    ctrl = Control("x", beta, (p,))
    expected = spa.single_control_index(beta, p)  # c_hat
    assert index_table(ctrl)[("x", 0)] == pytest.approx(expected, abs=TOL)
    # For a lone control the game value equals beta*p (win in one attempt, else jam).
    assert game_value(ctrl) == pytest.approx(beta * p, abs=TOL)


def test_chain_of_certain_controls_value():
    # Two controls in series that always succeed: win in 2 attempts.
    a = Control("a", 0.9, (1.0,))
    b = Control("b", 0.8, (1.0,))
    assert game_value(Series(a, b)) == pytest.approx(0.9 * 0.8, abs=TOL)


# ---------------------------------------------------------------------------
# Phase 1 vs the brute-force MDP oracle  (the value is correct)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_value_matches_bruteforce_mdp(seed):
    rng = random.Random(seed)
    net = _random_network(rng, n_leaves=rng.randint(2, 6), qmax=2)
    assert game_value(net) == pytest.approx(_mdp_optimal_value(net), abs=1e-9)


# ---------------------------------------------------------------------------
# Phase 3 vs the brute-force MDP oracle  (the index policy is optimal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_index_policy_is_optimal(seed):
    rng = random.Random(seed)
    net = _random_network(rng, n_leaves=rng.randint(2, 6), qmax=2)
    optimal = _mdp_optimal_value(net)
    policy_value = _mdp_policy_value(net, IndexAttacker(net))
    # Zero optimality gap, and consistent with the Phase-1 fold.
    assert policy_value == pytest.approx(optimal, abs=1e-9)
    assert policy_value == pytest.approx(game_value(net), abs=1e-9)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_index_monotone_in_failures(seed):
    # Assumption 2: with monotone controls, alpha(v,k) >= alpha(v,k+1).
    rng = random.Random(100 + seed)
    net = _random_network(rng, n_leaves=rng.randint(2, 6), qmax=3)
    table = index_table(net)
    for ctrl in spa.controls(net):
        for k in range(ctrl.q - 1):
            assert table[(ctrl.name, k)] >= table[(ctrl.name, k + 1)] - 1e-9


def test_series_discount_lowers_head_index():
    # The head of a series can only lose value from the work still downstream:
    # alpha(head, 0) <= c_hat(head, 0).
    head = Control("h", 0.9, (0.6, 0.3))
    tail = Control("t", 0.9, (0.5, 0.2))
    alpha_head = index_table(Series(head, tail))[("h", 0)]
    assert alpha_head <= spa.single_control_index(0.9, 0.6) + 1e-9


def test_parallel_or_gain():
    # Racing both branches is at least as good as either alone (at gamma = 0).
    both = game_value(Par(A, B))
    assert both >= max(game_value(A), game_value(B)) - 1e-12


def test_worked_example_regression():
    # Pin the known numbers for the nested instance Series(Par(A,B), C).
    assert game_value(WORKED) == pytest.approx(0.533995, abs=1e-6)
    expected = {
        ("A", 0): 0.640130, ("A", 1): 0.479546,
        ("B", 0): 0.698616, ("B", 1): 0.606293,
        ("C", 0): 0.843750, ("C", 1): 0.729730,
    }
    table = index_table(WORKED)
    for state, alpha in expected.items():
        assert table[state] == pytest.approx(alpha, abs=1e-6)


# ---------------------------------------------------------------------------
# Phase 3: execution semantics
# ---------------------------------------------------------------------------


def test_certain_win_reward_is_discount_product():
    a = Control("a", 0.9, (1.0,))
    b = Control("b", 0.8, (1.0,))
    result = IndexAttacker(Series(a, b)).simulate(random.Random(0))
    assert result.won
    assert result.attempts == 2
    assert result.reward == pytest.approx(0.9 * 0.8, abs=TOL)


def test_all_routes_jam_gives_zero_reward():
    # Controls that can never succeed: every route dies, reward 0 (and value 0).
    dead = Par(Control("x", 0.9, (0.0,)), Control("y", 0.9, (0.0,)))
    assert game_value(dead) == pytest.approx(0.0, abs=TOL)
    result = IndexAttacker(dead).simulate(random.Random(0))
    assert not result.won
    assert result.reward == 0.0


def test_frontier_initial_and_pruning():
    state = spa.initial_state(WORKED)
    # Initially only the OR-block heads A, B are attackable; C is hidden.
    assert {name for name, _ in spa.frontier(WORKED, state)} == {"A", "B"}
    # After A is won, the OR-block is cleared (B pruned) and C is exposed.
    state = spa.apply_outcome(state, A, success=True)
    assert {name for name, _ in spa.frontier(WORKED, state)} == {"C"}
    assert spa.block_status(WORKED, state) is spa.Status.LIVE


def test_jam_strands_downstream():
    # In Series(A, B), jamming A (q=1) kills the branch; B is never exposed.
    a = Control("a", 0.9, (0.5,))  # q = 1: one failure jams it
    b = Control("b", 0.9, (0.5,))
    net = Series(a, b)
    state = spa.apply_outcome(spa.initial_state(net), a, success=False)
    assert spa.block_status(net, state) is spa.Status.DEAD
    assert spa.frontier(net, state) == []


# ---------------------------------------------------------------------------
# Monte-Carlo consistency
# ---------------------------------------------------------------------------


def test_monte_carlo_matches_exact_value():
    attacker = IndexAttacker(WORKED)
    estimate = attacker.monte_carlo_value(40_000, random.Random(2024))
    assert estimate == pytest.approx(game_value(WORKED), abs=0.01)


def test_index_table_is_deterministic():
    assert index_table(WORKED) == index_table(WORKED)