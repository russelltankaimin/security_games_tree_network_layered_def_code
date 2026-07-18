"""
sp_par_gradients_pytest.py
==========================

Safety tests for the MULTI-CORE cut implementation in `sp_par_gradients`: both
the stateless `value_and_gradient` and the reusable `CutPlan`. The cut regroups
the SAME associative merges as the serial reverse tape, exchanging only float
boundary data, so the result must be BIT-IDENTICAL to `sp_gradients3` for the
same method at regular points -- and the VALUE exact even at l = 0 (where the
subgradient is set-valued). We force the parallel path with `min_parallel_q=0`
so small random trees still exercise the partition / worker round-trip.

Worker functions are module-level in `sp_par_gradients`, so this runs under the
`spawn` start method (macOS default). Run:

    python3 -m pytest sp_par_gradients_pytest.py -q
"""

import random

import pytest

import sp_gradients3 as G3
import sp_par_gradients as GP
from sp_gradients2 import Control, Parallel, Series, collect_controls

METHODS = ["binary", "mary"]
CORES = 3                                             # small, for a fast suite


# --------------------------------------------------------------------------
def _rand_tree(names, rng, bias, arity):
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
        kids.append(_rand_tree(names[i:i + s], rng, bias, arity))
        i += s
    return op(*kids)


def _cases(n, seed0):
    out = []
    for t in range(n):
        rng = random.Random(seed0 + t)
        m = rng.randint(2, 14)
        tree = _rand_tree([f"v{i}" for i in range(m)], rng,
                          bias=rng.choice([0.3, 0.5, 0.7]), arity=rng.choice([2, 3, 4]))
        out.append((f"rand{t}", tree))
    return out


def _regular_alloc(tree, rng):
    names = [c.name for c in collect_controls(tree)]
    raw = {n: rng.random() + 0.05 for n in names}
    tot = sum(raw.values())
    return {n: raw[n] / tot for n in names}


@pytest.fixture(scope="module", autouse=True)
def _cleanup_pool():
    yield
    GP.shutdown()                                     # release the stateless global pool


# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_stateless_matches_serial(method):
    """Stateless value_and_gradient == serial, bit-identical at regular points."""
    worst_v = worst_g = 0.0
    for name, tree in _cases(40, 5000):
        rng = random.Random(hash((name, method)) & 0xFFFF)
        alloc = _regular_alloc(tree, rng)
        vs, gs = G3.value_and_gradient(tree, alloc, 5.0, method=method)
        vp, gp = GP.value_and_gradient(tree, alloc, 5.0, method=method,
                                       cores=CORES, min_parallel_q=0)
        worst_v = max(worst_v, abs(vs - vp))
        worst_g = max(worst_g, max(abs(gs[n] - gp[n]) for n in gs))
    assert worst_v == 0.0, f"value drift {worst_v:.2e}"
    assert worst_g == 0.0, f"gradient drift {worst_g:.2e}"


@pytest.mark.parametrize("method", METHODS)
def test_cutplan_reuse_matches_serial(method):
    """One CutPlan, MANY allocations -> each bit-identical to serial (proves the
    resident-subtree reuse is correct across repeated calls, as in RM)."""
    worst_v = worst_g = 0.0
    for name, tree in _cases(20, 6000):
        plan = GP.make_plan(tree, method=method, cores=CORES, min_parallel_q=0)
        try:
            rng = random.Random(hash((name, method, "plan")) & 0xFFFF)
            for _ in range(4):                        # reuse the plan across allocations
                alloc = _regular_alloc(tree, rng)
                vs, gs = G3.value_and_gradient(tree, alloc, 5.0, method=method)
                vp, gp = plan.value_and_gradient(alloc, 5.0)
                worst_v = max(worst_v, abs(vs - vp))
                worst_g = max(worst_g, max(abs(gs[n] - gp[n]) for n in gs))
        finally:
            plan.close()
    assert worst_v == 0.0, f"value drift {worst_v:.2e}"
    assert worst_g == 0.0, f"gradient drift {worst_g:.2e}"


@pytest.mark.parametrize("method", METHODS)
def test_value_exact_with_zero_allocations(method):
    """At l_e = 0 the subgradient is set-valued, but V* must stay exact."""
    rng = random.Random(11)
    for name, tree in _cases(20, 7000):
        names = [c.name for c in collect_controls(tree)]
        zeros = set(rng.sample(names, rng.randint(0, len(names))))
        raw = {n: (0.0 if n in zeros else rng.random() + 0.05) for n in names}
        tot = sum(raw.values()) or 1.0
        alloc = {n: raw[n] / tot for n in names}
        vs, _ = G3.value_and_gradient(tree, alloc, 5.0, method=method)
        vp, _ = GP.value_and_gradient(tree, alloc, 5.0, method=method,
                                      cores=CORES, min_parallel_q=0)
        assert abs(vs - vp) < 1e-12, f"{name}/{method}: {vs} vs {vp} (zeros)"


@pytest.mark.parametrize("method", METHODS)
def test_serial_fallbacks(method):
    """cores=1, a single control, and a tree below min_parallel_q must fall back
    to the serial path and still return the exact serial result."""
    rng = random.Random(3)
    _, tree = _cases(1, 8000)[0]
    alloc = _regular_alloc(tree, rng)
    vs, gs = G3.value_and_gradient(tree, alloc, 5.0, method=method)

    # cores = 1
    v1, g1 = GP.value_and_gradient(tree, alloc, 5.0, method=method, cores=1, min_parallel_q=0)
    assert abs(vs - v1) == 0.0 and max(abs(gs[n] - g1[n]) for n in gs) == 0.0

    # below the size threshold -> serial
    v2, g2 = GP.value_and_gradient(tree, alloc, 5.0, method=method,
                                   cores=CORES, min_parallel_q=10 ** 9)
    assert abs(vs - v2) == 0.0 and max(abs(gs[n] - g2[n]) for n in gs) == 0.0

    # a single control (nothing to split)
    c = Control("solo", 3, [0.6, 0.4, 0.2])
    a = {"solo": 1.0}
    vc, gc = G3.value_and_gradient(c, a, 5.0, method=method)
    vpc, gpc = GP.value_and_gradient(c, a, 5.0, method=method, cores=CORES, min_parallel_q=0)
    assert abs(vc - vpc) == 0.0 and abs(gc["solo"] - gpc["solo"]) == 0.0

    # CutPlan on the single control marks itself serial (no pool)
    plan = GP.make_plan(c, method=method, cores=CORES, min_parallel_q=0)
    assert plan.serial and plan.pool is None
    vpl, gpl = plan.value_and_gradient(a, 5.0)
    plan.close()
    assert abs(vc - vpl) == 0.0
