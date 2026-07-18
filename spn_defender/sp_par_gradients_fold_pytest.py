"""
sp_par_gradients_fold_pytest.py
===============================

Safety tests for the MULTI-CORE fold cut (`sp_par_gradients_fold`): stateless
`value_and_gradient` and the reusable `CutPlan`, both against the serial fold
reference `sp_gradients2.value_and_gradient(..., break_ties=False, native=True)`.
The cut regroups the same associative fold and injects the exact weight-list
context at the boundary, so results must be BIT-IDENTICAL at regular points, and
the value exact at l = 0. `min_parallel_q=0` forces the parallel path on small
random trees. Worker fns are module-level -> spawn-safe.

Run:  python3 -m pytest sp_par_gradients_fold_pytest.py -q
"""

import random

import pytest

import sp_par_gradients_fold as GPF
from sp_gradients2 import (Control, Parallel, Series, collect_controls,
                           value_and_gradient as serial_vg)

CORES = 3


def _serial(tree, alloc, rho, break_ties=False):
    return serial_vg(tree, alloc, rho, break_ties=break_ties, native=True)


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
def _cleanup():
    yield
    GPF.shutdown()


def test_stateless_matches_serial_fold():
    worst_v = worst_g = 0.0
    for name, tree in _cases(40, 15000):
        rng = random.Random(hash(name) & 0xFFFF)
        alloc = _regular_alloc(tree, rng)
        vs, gs = _serial(tree, alloc, 5.0)
        vp, gp = GPF.value_and_gradient(tree, alloc, 5.0, cores=CORES, min_parallel_q=0,
                                        break_ties=False)
        worst_v = max(worst_v, abs(vs - vp))
        worst_g = max(worst_g, max(abs(gs[n] - gp[n]) for n in gs))
    assert worst_v == 0.0, f"value drift {worst_v:.2e}"
    assert worst_g == 0.0, f"gradient drift {worst_g:.2e}"


def test_cutplan_reuse_matches_serial_fold():
    worst_v = worst_g = 0.0
    for name, tree in _cases(20, 16000):
        plan = GPF.make_plan(tree, cores=CORES, min_parallel_q=0, break_ties=False)
        try:
            rng = random.Random(hash((name, "plan")) & 0xFFFF)
            for _ in range(4):
                alloc = _regular_alloc(tree, rng)
                vs, gs = _serial(tree, alloc, 5.0)
                vp, gp = plan.value_and_gradient(alloc, 5.0)
                worst_v = max(worst_v, abs(vs - vp))
                worst_g = max(worst_g, max(abs(gs[n] - gp[n]) for n in gs))
        finally:
            plan.close()
    assert worst_v == 0.0, f"value drift {worst_v:.2e}"
    assert worst_g == 0.0, f"gradient drift {worst_g:.2e}"


def test_break_ties_matches_serial_fold():
    """Default break_ties=True must match the serial fold's break_ties=True vertex
    subgradient -- at regular points AND at kinks (some l_v = 0). This is the
    convention the runner's `fold` uses; the raw-only version diverged in RM."""
    worst = 0.0
    for name, tree in _cases(25, 15500):
        names = [c.name for c in collect_controls(tree)]
        rng = random.Random(hash((name, "bt")) & 0xFFFF)
        for zeros_frac in (0.0, 0.4):                 # 0.0 = regular, 0.4 = force kinks
            zeros = set(rng.sample(names, int(len(names) * zeros_frac)))
            raw = {n: (0.0 if n in zeros else rng.random() + 0.05) for n in names}
            tot = sum(raw.values()) or 1.0
            alloc = {n: raw[n] / tot for n in names}
            vs, gs = serial_vg(tree, alloc, 5.0, break_ties=True, native=True)
            vp, gp = GPF.value_and_gradient(tree, alloc, 5.0, cores=CORES, min_parallel_q=0)
            assert abs(vs - vp) == 0.0, f"{name}: value {vs} vs {vp}"
            worst = max(worst, max(abs(gs[n] - gp[n]) for n in gs))
    assert worst == 0.0, f"tie-broken gradient drift {worst:.2e}"


def test_value_exact_with_zero_allocations():
    rng = random.Random(11)
    for name, tree in _cases(20, 17000):
        names = [c.name for c in collect_controls(tree)]
        zeros = set(rng.sample(names, rng.randint(0, len(names))))
        raw = {n: (0.0 if n in zeros else rng.random() + 0.05) for n in names}
        tot = sum(raw.values()) or 1.0
        alloc = {n: raw[n] / tot for n in names}
        vs, _ = _serial(tree, alloc, 5.0)
        vp, _ = GPF.value_and_gradient(tree, alloc, 5.0, cores=CORES, min_parallel_q=0)
        assert abs(vs - vp) < 1e-12, f"{name}: {vs} vs {vp} (zeros)"


def test_rel_tol_pruning_matches_serial():
    """With adjoint pruning on, the parallel cut must still match the serial fold
    run with the SAME rel_tol (the pruning is applied identically at Par nodes)."""
    for name, tree in _cases(15, 18000):
        rng = random.Random(hash((name, "prune")) & 0xFFFF)
        alloc = _regular_alloc(tree, rng)
        vs, gs = serial_vg(tree, alloc, 5.0, break_ties=False, native=True, adjoint_rel_tol=1e-3)
        vp, gp = GPF.value_and_gradient(tree, alloc, 5.0, cores=CORES, min_parallel_q=0,
                                        rel_tol=1e-3, break_ties=False)
        assert abs(vs - vp) == 0.0
        assert max(abs(gs[n] - gp[n]) for n in gs) == 0.0


def test_serial_fallbacks():
    rng = random.Random(3)
    _, tree = _cases(1, 19000)[0]
    alloc = _regular_alloc(tree, rng)
    vs, gs = _serial(tree, alloc, 5.0)

    v1, g1 = GPF.value_and_gradient(tree, alloc, 5.0, cores=1, min_parallel_q=0, break_ties=False)
    assert abs(vs - v1) == 0.0 and max(abs(gs[n] - g1[n]) for n in gs) == 0.0

    v2, g2 = GPF.value_and_gradient(tree, alloc, 5.0, cores=CORES, min_parallel_q=10 ** 9,
                                    break_ties=False)
    assert abs(vs - v2) == 0.0 and max(abs(gs[n] - g2[n]) for n in gs) == 0.0

    c = Control("solo", 3, [0.6, 0.4, 0.2])
    a = {"solo": 1.0}
    vc, gc = _serial(c, a, 5.0)
    vpc, gpc = GPF.value_and_gradient(c, a, 5.0, cores=CORES, min_parallel_q=0, break_ties=False)
    assert abs(vc - vpc) == 0.0 and abs(gc["solo"] - gpc["solo"]) == 0.0

    plan = GPF.make_plan(c, cores=CORES, min_parallel_q=0, break_ties=False)
    assert plan.serial and plan.pool is None
    vpl, _ = plan.value_and_gradient(a, 5.0)
    plan.close()
    assert abs(vc - vpl) == 0.0
