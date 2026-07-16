"""
sp_gradients3_pytest.py
=======================

Tests for the binary reverse-tape / direct m-ary exact-gradient implementations
in sp_gradients3, validated against the three independent references from the
paper's checklist (Sec. 16):

  * brute-force dynamic programming (sp_gradients2.brute_force_value) for the value;
  * the well-tested sp_gradients2 fold (value AND gradient) at regular points; and
  * centered finite differences of the value.

Both methods ("binary", "mary") are checked. Regular points (all l_e > 0) are used
for the exact-match / FD tests -- at l_e = 0 (beta = 1) the value is a kink and the
subgradient is set-valued (Sec. 15), so implementations may pick different valid
vertices; that case is covered by a separate value-only test.

Run:  python3 -m pytest sp_gradients3_pytest.py -q
"""

import math
import random

import pytest

import sp_gradients2 as G2
import sp_gradients3 as G3
from sp_gradients2 import (Control, Series, Parallel, value_and_gradient,
                           brute_force_value, to_binary_tree, collect_controls)

METHODS = ["binary", "mary"]


# --------------------------------------------------------------------------
# Test instances: a few fixed structures + reproducible random SP trees.
# --------------------------------------------------------------------------
def _fixed_cases():
    return [
        ("single_q1", Control("a", 1, [0.6])),
        ("single_q3", Control("a", 3, [0.7, 0.5, 0.3])),
        ("series2", Series(Control("a", 2, [0.6, 0.4]), Control("b", 1, [0.5]))),
        ("parallel2", Parallel(Control("a", 1, [0.6]), Control("b", 1, [0.5]))),
        ("mary_par3", Parallel(Control("a", 2, [0.6, 0.3]),
                               Control("b", 1, [0.55]), Control("c", 3, [0.7, 0.5, 0.2]))),
        ("mary_ser3", Series(Control("a", 1, [0.6]),
                             Control("b", 2, [0.5, 0.3]), Control("c", 1, [0.45]))),
        ("nested", Parallel(Series(Control("a", 2, [0.5, 0.4]),
                                   Parallel(Control("b", 1, [0.6]), Control("c", 1, [0.55]))),
                            Series(Control("d", 1, [0.5]), Control("e", 1, [0.45])))),
        ("wide_par", Parallel(*[Control(f"v{i}", 2, [0.5, 0.3]) for i in range(6)])),
        ("long_ser", Series(*[Control(f"v{i}", 2, [0.4, 0.2]) for i in range(6)])),
    ]


def _random_tree(names, rng, bias, arity):
    if len(names) == 1:
        q = rng.randint(1, 4)
        ps = sorted((round(rng.uniform(0.05, 0.95), 4) for _ in range(q)), reverse=True)
        return Control(names[0], q, ps)
    op = Series if rng.random() < bias else Parallel
    k = rng.randint(2, min(arity, len(names)))
    base, rem = divmod(len(names), k)
    sizes = [base + 1] * rem + [base] * (k - rem)
    kids, i = [], 0
    for s in sizes:
        kids.append(_random_tree(names[i:i + s], rng, bias, arity))
        i += s
    return op(*kids)


def _random_cases(n=30):
    out = []
    for t in range(n):
        rng = random.Random(1000 + t)
        m = rng.randint(1, 12)
        tree = _random_tree([f"v{i}" for i in range(m)], rng,
                            bias=rng.choice([0.3, 0.5, 0.7]), arity=rng.choice([2, 3, 4]))
        out.append((f"rand{t}", tree))
    return out


ALL_CASES = _fixed_cases() + _random_cases()
RHOS = [1.0, 5.0]


def _regular_alloc(tree, rng):
    """A strictly-positive allocation on the simplex (regular point)."""
    names = [c.name for c in collect_controls(tree)]
    raw = {n: rng.random() + 0.05 for n in names}
    tot = sum(raw.values())
    return {n: raw[n] / tot for n in names}


# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name,tree", ALL_CASES, ids=lambda x: x if isinstance(x, str) else "")
def test_matches_fold_at_regular_points(method, name, tree):
    rng = random.Random(hash((name, method)) & 0xFFFF)
    for rho in RHOS:
        alloc = _regular_alloc(tree, rng)
        v2, g2 = value_and_gradient(tree, alloc, rho, native=True, break_ties=False)
        v3, g3 = G3.value_and_gradient(tree, alloc, rho, method=method)
        assert abs(v2 - v3) < 1e-9, f"{name}/{method}: value {v2} vs {v3}"
        gerr = max(abs(g2[k] - g3[k]) for k in g2)
        assert gerr < 1e-6, f"{name}/{method}: gradient max err {gerr:.2e}"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name,tree", ALL_CASES, ids=lambda x: x if isinstance(x, str) else "")
def test_value_matches_brute_force(method, name, tree):
    controls = collect_controls(tree)
    if sum(c.lockout for c in controls) > 22:          # brute force is exponential
        pytest.skip("too large for brute force")
    rng = random.Random(hash((name, "bf")) & 0xFFFF)
    rho = 2.0
    alloc = _regular_alloc(tree, rng)
    names = [c.name for c in controls]
    factors = {n: math.exp(-rho * alloc[n]) for n in names}
    v_bf = brute_force_value(to_binary_tree(tree), factors)
    v3, _ = G3.value_and_gradient(tree, alloc, rho, method=method)
    assert abs(v_bf - v3) < 1e-7, f"{name}/{method}: brute {v_bf} vs tape {v3}"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name,tree", ALL_CASES, ids=lambda x: x if isinstance(x, str) else "")
def test_gradient_matches_finite_diff(method, name, tree):
    rng = random.Random(hash((name, "fd")) & 0xFFFF)
    rho, h = 3.0, 1e-6
    alloc = _regular_alloc(tree, rng)
    _, g3 = G3.value_and_gradient(tree, alloc, rho, method=method)
    for n in g3:
        up = dict(alloc); up[n] += h
        dn = dict(alloc); dn[n] -= h
        vu, _ = G3.value_and_gradient(tree, up, rho, method=method)
        vd, _ = G3.value_and_gradient(tree, dn, rho, method=method)
        fd = (vu - vd) / (2 * h)
        assert abs(fd - g3[n]) < 1e-4, f"{name}/{method}: d/d{n} fd={fd:.6f} grad={g3[n]:.6f}"


@pytest.mark.parametrize("name,tree", ALL_CASES, ids=lambda x: x if isinstance(x, str) else "")
def test_binary_and_mary_agree(name, tree):
    rng = random.Random(hash((name, "bm")) & 0xFFFF)
    alloc = _regular_alloc(tree, rng)
    vb, gb = G3.value_and_gradient(tree, alloc, 5.0, method="binary")
    vm, gm = G3.value_and_gradient(tree, alloc, 5.0, method="mary")
    assert abs(vb - vm) < 1e-9
    assert max(abs(gb[k] - gm[k]) for k in gb) < 1e-7


@pytest.mark.parametrize("method", METHODS)
def test_value_correct_with_zero_allocations(method):
    """At l_e = 0 the gradient is set-valued, but the VALUE is still exact."""
    rng = random.Random(7)
    for name, tree in ALL_CASES:
        names = [c.name for c in collect_controls(tree)]
        zeros = set(rng.sample(names, rng.randint(0, len(names))))
        raw = {n: (0.0 if n in zeros else rng.random() + 0.05) for n in names}
        tot = sum(raw.values()) or 1.0
        alloc = {n: raw[n] / tot for n in names}
        v2, _ = value_and_gradient(tree, alloc, 5.0, native=True, break_ties=False)
        v3, _ = G3.value_and_gradient(tree, alloc, 5.0, method=method)
        assert abs(v2 - v3) < 1e-7, f"{name}/{method}: value {v2} vs {v3} (zeros)"
