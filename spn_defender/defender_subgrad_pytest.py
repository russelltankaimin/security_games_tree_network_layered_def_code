"""
Pytest suite for sp_gradient.py.

Run with:   pytest test_sp_gradient.py            (quiet)
            pytest -v test_sp_gradient.py         (verbose)

The trust anchor for every numeric assertion is the independent brute-force MDP
solver (brute_force_value), which shares no logic with the fold / reverse-mode
under test. We check:

  * value          : fold V*  ==  brute-force MDP V*
  * gradient       : reverse-mode  ==  central finite differences (smooth points)
  * subgradient    : the definition (subgradient inequality + directional test)
                     holds, including at index ties (kinks)
  * structure      : n-ary == nested-binary, series/parallel associativity
  * monotonicity   : more delay never helps the attacker (gradient <= 0)
  * convexity      : random simplex chords satisfy the convexity inequality
  * error handling : malformed inputs raise ValueError
"""
import math
import random
import pytest

from sp_gradients2 import (
    Control, Series, Parallel,
    value_and_gradient, brute_force_value, to_binary_tree, collect_controls,
)
from verify_subgrad import (
    subgradient_inequality_min_slack, directional_max_violation,
)

# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------
VALUE_TOL = 1e-9          # fold vs brute-force value
GRAD_TOL = 1e-5           # reverse-mode vs finite differences (smooth points)
SLACK_TOL = -1e-6         # subgradient inequality: min slack must exceed this
DIR_TOL = 1e-5            # directional test: max violation must be below this

# --------------------------------------------------------------------------
# Helpers (independent reference quantities)
# --------------------------------------------------------------------------
def _factors(names, allocation, rho):
    return {n: math.exp(-rho * allocation[n]) for n in names}

def reference_value(tree, allocation, rho):
    binary = to_binary_tree(tree)
    names = [c.name for c in collect_controls(binary)]
    return brute_force_value(binary, _factors(names, allocation, rho))

def finite_difference_gradient(tree, allocation, rho, step=1e-6):
    binary = to_binary_tree(tree)
    names = [c.name for c in collect_controls(binary)]
    grad = {}
    for n in names:
        up = dict(allocation); up[n] += step
        down = dict(allocation); down[n] -= step
        grad[n] = (brute_force_value(binary, _factors(names, up, rho))
                   - brute_force_value(binary, _factors(names, down, rho))) / (2 * step)
    return grad

# --------------------------------------------------------------------------
# Test instances: (id, tree, allocation, rho). allocation=None -> uniform.
# All chosen small enough for the brute-force solver, and away from ties.
# --------------------------------------------------------------------------
SMOOTH_INSTANCES = [
    ("single_q1",
     Control("a", 1, 0.5), {"a": 0.5}, 1.0),
    ("single_q2",
     Control("a", 2, [0.6, 0.4]), {"a": 0.4}, 1.0),
    ("parallel_asymmetric",
     Parallel(Control("a", 1, 0.6), Control("b", 1, 0.4)), {"a": 0.3, "b": 0.7}, 1.0),
    ("series_ab",
     Series(Control("a", 2, [0.8, 0.6]), Control("b", 1, 0.7)), {"a": 0.4, "b": 0.6}, 1.0),
    ("par_series_c",
     Parallel(Series(Control("a", 1, 0.6), Control("b", 2, [0.5, 0.4])),
              Control("c", 1, 0.55)),
     {"a": 0.25, "b": 0.35, "c": 0.40}, 1.0),
    ("tutorial2_nested_q2",
     Parallel(Series(Control("a", 2, [0.5, 0.4]),
                     Parallel(Control("b", 1, 0.6), Control("c", 1, 0.55))),
              Series(Control("d", 1, 0.5), Control("e", 1, 0.45))),
     {"a": 0.2, "b": 0.1, "c": 0.15, "d": 0.3, "e": 0.25}, 1.0),
    ("nary_parallel_three_chains",
     Parallel(Series(Control("x1", 1, 0.6), Control("x2", 1, 0.5)),
              Series(Control("y1", 2, [0.5, 0.3]), Control("y2", 1, 0.55)),
              Control("z", 1, 0.45)),
     None, 1.0),
    ("series_of_parallels",
     Series(Parallel(Control("a", 1, 0.6), Control("b", 1, 0.5)),
            Parallel(Control("c", 1, 0.55), Control("d", 1, 0.45))),
     {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}, 1.3),
]
SMOOTH_IDS = [case[0] for case in SMOOTH_INSTANCES]

# A representative subset for the (slower) subgradient-certification tests.
CERTIFY_SUBSET = {"parallel_asymmetric", "par_series_c", "tutorial2_nested_q2"}


# ==========================================================================
# Value correctness
# ==========================================================================
@pytest.mark.parametrize("name,tree,allocation,rho", SMOOTH_INSTANCES, ids=SMOOTH_IDS)
def test_fold_value_matches_brute_force(name, tree, allocation, rho):
    value, _ = value_and_gradient(tree, allocation, rho)
    expected = reference_value(tree, allocation if allocation is not None
                               else {c.name: 1.0 / len(collect_controls(to_binary_tree(tree)))
                                     for c in collect_controls(to_binary_tree(tree))},
                               rho)
    assert abs(value - expected) < VALUE_TOL


# ==========================================================================
# Gradient correctness at smooth points (vs finite differences)
# ==========================================================================
@pytest.mark.parametrize("name,tree,allocation,rho", SMOOTH_INSTANCES, ids=SMOOTH_IDS)
def test_gradient_matches_finite_differences(name, tree, allocation, rho):
    if allocation is None:
        names = [c.name for c in collect_controls(to_binary_tree(tree))]
        allocation = {n: 1.0 / len(names) for n in names}
    _, grad = value_and_gradient(tree, allocation, rho)
    fd = finite_difference_gradient(tree, allocation, rho)
    for n in grad:
        assert abs(grad[n] - fd[n]) < GRAD_TOL, f"{name}: control {n}"


# ==========================================================================
# Monotonicity: more delay never helps the attacker -> gradient <= 0
# ==========================================================================
@pytest.mark.parametrize("name,tree,allocation,rho", SMOOTH_INSTANCES, ids=SMOOTH_IDS)
def test_gradient_is_nonpositive(name, tree, allocation, rho):
    _, grad = value_and_gradient(tree, allocation, rho)
    for n, g in grad.items():
        assert g <= 1e-12, f"{name}: control {n} has positive gradient {g}"


# ==========================================================================
# Subgradient certification (definition-based; valid even at kinks)
# ==========================================================================
@pytest.mark.parametrize(
    "name,tree,allocation,rho",
    [case for case in SMOOTH_INSTANCES if case[0] in CERTIFY_SUBSET],
    ids=[c[0] for c in SMOOTH_INSTANCES if c[0] in CERTIFY_SUBSET],
)
def test_smooth_gradient_certifies_as_subgradient(name, tree, allocation, rho):
    if allocation is None:
        names = [c.name for c in collect_controls(to_binary_tree(tree))]
        allocation = {n: 1.0 / len(names) for n in names}
    _, grad = value_and_gradient(tree, allocation, rho)
    min_slack = subgradient_inequality_min_slack(tree, allocation, rho, grad, n=300, seed=0)
    max_viol = directional_max_violation(tree, allocation, rho, grad, n=200, seed=1)
    assert min_slack >= SLACK_TOL
    assert max_viol <= DIR_TOL


def test_certifier_rejects_a_wrong_vector():
    """The certifier must REJECT a non-subgradient (otherwise it proves nothing)."""
    tree = Parallel(Series(Control("a", 1, 0.6), Control("b", 2, [0.5, 0.4])),
                    Control("c", 1, 0.55))
    allocation = {"a": 0.25, "b": 0.35, "c": 0.40}
    _, grad = value_and_gradient(tree, allocation, 1.0)
    wrong = {k: v + 0.2 for k, v in grad.items()}          # shift off the gradient
    min_slack = subgradient_inequality_min_slack(tree, allocation, 1.0, wrong, n=300, seed=0)
    max_viol = directional_max_violation(tree, allocation, 1.0, wrong, n=200, seed=1)
    assert (min_slack < -1e-3) or (max_viol > 1e-3)


# ==========================================================================
# The index-tie (kink) case and the tie-breaking fix
# ==========================================================================
SYMMETRIC_TIE_TREE = Parallel(Control("a", 1, 0.5), Control("b", 1, 0.5))
SYMMETRIC_TIE_ALLOC = {"a": 0.5, "b": 0.5}


def test_tie_break_returns_valid_subgradient():
    _, grad = value_and_gradient(SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0,
                                 break_ties=True)
    min_slack = subgradient_inequality_min_slack(
        SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0, grad, n=1500, seed=0)
    max_viol = directional_max_violation(
        SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0, grad, n=800, seed=1)
    assert min_slack >= SLACK_TOL
    assert max_viol <= DIR_TOL


def test_raw_pass_without_tiebreak_is_not_a_subgradient_at_tie():
    """Regression guard: without tie-breaking, the raw pass returns a vector that
    is NOT a subgradient at an exact tie (this is the defect break_ties fixes)."""
    _, raw = value_and_gradient(SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0,
                                break_ties=False)
    max_viol = directional_max_violation(
        SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0, raw, n=800, seed=1)
    assert max_viol > 1e-2                                  # clearly violates the definition


def test_tie_value_independent_of_tiebreak():
    v_on, _ = value_and_gradient(SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0, break_ties=True)
    v_off, _ = value_and_gradient(SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0, break_ties=False)
    assert abs(v_on - v_off) < VALUE_TOL


def test_tie_gradient_is_a_subdifferential_vertex():
    """At the symmetric tie the subdifferential is the segment between
    (-b/2 - b^2/4, -b^2/4) and its swap; the tie-break should return a vertex."""
    b = math.exp(-0.5)
    vertex_a = -(0.5 * b + 0.25 * b * b)
    vertex_b = -(0.25 * b * b)
    _, grad = value_and_gradient(SYMMETRIC_TIE_TREE, SYMMETRIC_TIE_ALLOC, 1.0, break_ties=True)
    got = (grad["a"], grad["b"])
    options = [(vertex_a, vertex_b), (vertex_b, vertex_a)]
    assert any(abs(got[0] - va) < 1e-4 and abs(got[1] - vb) < 1e-4 for va, vb in options)


# ==========================================================================
# Structural equivalences
# ==========================================================================
def test_value_independent_of_break_ties_on_smooth_point():
    tree = Parallel(Series(Control("a", 1, 0.6), Control("b", 2, [0.5, 0.4])),
                    Control("c", 1, 0.55))
    allocation = {"a": 0.25, "b": 0.35, "c": 0.40}
    v_on, g_on = value_and_gradient(tree, allocation, 1.0, break_ties=True)
    v_off, g_off = value_and_gradient(tree, allocation, 1.0, break_ties=False)
    assert abs(v_on - v_off) < VALUE_TOL
    for n in g_on:                                          # smooth: identical either way
        assert abs(g_on[n] - g_off[n]) < 1e-9


def test_nary_parallel_equals_nested_binary():
    allocation = {"a": 0.3, "b": 0.3, "c": 0.4}
    nary = Parallel(Control("a", 1, 0.6), Control("b", 1, 0.5), Control("c", 1, 0.55))
    left_nested = Parallel(Parallel(Control("a", 1, 0.6), Control("b", 1, 0.5)),
                           Control("c", 1, 0.55))
    right_nested = Parallel(Control("a", 1, 0.6),
                            Parallel(Control("b", 1, 0.5), Control("c", 1, 0.55)))
    v0, g0 = value_and_gradient(nary, allocation, 1.0)
    for variant in (left_nested, right_nested):
        v, g = value_and_gradient(variant, allocation, 1.0)
        assert abs(v - v0) < 1e-9
        for n in g0:
            assert abs(g[n] - g0[n]) < 1e-7


def test_nary_series_equals_nested_binary():
    allocation = {"a": 0.3, "b": 0.3, "c": 0.4}
    nary = Series(Control("a", 1, 0.6), Control("b", 1, 0.5), Control("c", 1, 0.55))
    left_nested = Series(Series(Control("a", 1, 0.6), Control("b", 1, 0.5)),
                         Control("c", 1, 0.55))
    right_nested = Series(Control("a", 1, 0.6),
                          Series(Control("b", 1, 0.5), Control("c", 1, 0.55)))
    v0, g0 = value_and_gradient(nary, allocation, 1.0)
    for variant in (left_nested, right_nested):
        v, g = value_and_gradient(variant, allocation, 1.0)
        assert abs(v - v0) < 1e-9
        for n in g0:
            assert abs(g[n] - g0[n]) < 1e-7


# ==========================================================================
# Convexity (a light random-chord check on one instance)
# ==========================================================================
def test_value_is_convex_on_random_chords():
    tree = Parallel(Series(Control("a", 1, 0.6), Control("b", 2, [0.5, 0.4])),
                    Control("c", 1, 0.55))
    names = ["a", "b", "c"]
    rng = random.Random(0)

    def value_at(allocation):
        v, _ = value_and_gradient(tree, allocation, 1.0, break_ties=False)
        return v

    def random_simplex_point():
        raw = [rng.random() for _ in names]
        total = sum(raw)
        return {n: r / total for n, r in zip(names, raw)}

    worst_violation = 0.0
    for _ in range(400):
        x = random_simplex_point()
        y = random_simplex_point()
        lam = rng.random()
        mid = {n: lam * x[n] + (1 - lam) * y[n] for n in names}
        lhs = value_at(mid)
        rhs = lam * value_at(x) + (1 - lam) * value_at(y)
        worst_violation = max(worst_violation, lhs - rhs)     # convex => lhs <= rhs
    assert worst_violation < 1e-9


# ==========================================================================
# Determinism
# ==========================================================================
def test_results_are_deterministic():
    tree = Parallel(Series(Control("a", 2, [0.5, 0.4]),
                           Parallel(Control("b", 1, 0.6), Control("c", 1, 0.55))),
                    Series(Control("d", 1, 0.5), Control("e", 1, 0.45)))
    allocation = {"a": 0.2, "b": 0.1, "c": 0.15, "d": 0.3, "e": 0.25}
    v1, g1 = value_and_gradient(tree, allocation, 1.0)
    v2, g2 = value_and_gradient(tree, allocation, 1.0)
    assert v1 == v2 and g1 == g2


# ==========================================================================
# Error handling
# ==========================================================================
def test_control_rejects_mismatched_probability_length():
    with pytest.raises(ValueError):
        Control("a", 2, [0.5])

def test_control_rejects_nonpositive_lockout():
    with pytest.raises(ValueError):
        Control("a", 0, [])

def test_series_requires_two_children():
    with pytest.raises(ValueError):
        Series(Control("a", 1, 0.5))

def test_parallel_requires_two_children():
    with pytest.raises(ValueError):
        Parallel(Control("a", 1, 0.5))

def test_duplicate_control_names_rejected():
    tree = Parallel(Control("a", 1, 0.6), Control("a", 1, 0.5))
    with pytest.raises(ValueError):
        value_and_gradient(tree, {"a": 0.5}, 1.0)

def test_missing_allocation_entry_rejected():
    tree = Series(Control("a", 1, 0.6), Control("b", 1, 0.5))
    with pytest.raises(ValueError):
        value_and_gradient(tree, {"a": 0.5}, 1.0)

def test_nonpositive_discount_rate_rejected():
    tree = Series(Control("a", 1, 0.6), Control("b", 1, 0.5))
    with pytest.raises(ValueError):
        value_and_gradient(tree, {"a": 0.5, "b": 0.5}, 0.0)

def test_single_float_probability_shorthand_accepted():
    control = Control("a", 1, 0.5)
    assert control.success_probs == [0.5]