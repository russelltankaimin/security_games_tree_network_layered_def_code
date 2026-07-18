"""
sp_par_gradients_fold.py
========================

MULTI-CORE exact value-and-gradient for the **fold** algorithm (`sp_gradients2`,
the weight-list adjoint), using the SAME subtree-boundary cut as
`sp_par_gradients` (which parallelises the reverse-tape / m-ary kernels).

Why the same cut works for the fold
-----------------------------------
The fold computes the gradient by a TOP-DOWN pass (`compute_gradient_nary`) that
hands each node a **weight list** -- the linear functional "how V* responds to
perturbing this node's profile." That weight list is exactly the boundary adjoint
context, the fold-analogue of the tape's boundary `.adj`. So we cut the tree into
a frontier of ~`CORES` subtrees and exchange only floats:

    forward (parallel)  : each worker folds its subtree -> boundary PROFILE
                          (breakpoints, values)                       [floats]
    top (master)        : combine the boundary profiles (series/parallel fold)
                          above the frontier, then run the fold's top-down
                          backward down to each cut -> that subtree's WEIGHT LIST
    backward (parallel) : each worker re-folds its subtree (checkpoint) and runs
                          `compute_gradient_nary` SEEDED with its weight list
                          -> its leaves' gradient

Only the subtree profiles (round 1) and the seed weight lists (round 2) cross the
process boundary; the `PiecewiseLinearProfile` / weight-list objects stay inside
each process. Result is bit-identical to the serial fold
(`sp_gradients2.value_and_gradient(..., break_ties=False, native=True)`).

Same regimes as the tape cut (Amdahl on the serial top merge): good on
series_heavy (`Par` of few deep chains), little gain on wide-shallow -- and for the
fold the top `Par` backward is the O(Q^2) leave-one-out, so a wide top is an even
sharper serial bottleneck. Small / narrow instances fall back to the serial fold.

Cores are capped by `sp_par_gradients.CORES` (shared with the tape cut); pass
`cores=` per call or use `sp_par_gradients.set_cores(n)`.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import sp_par_gradients as _PT                      # reuse partition + core cap
from sp_gradients2 import (ControlNode, ParallelNode, SeriesNode, PiecewiseLinearProfile,
                           collect_controls, series_profile, parallel_profile_nary,
                           compute_profiles_nary, compute_gradient_nary,
                           cumulative_weight, _leave_one_out_weight_lists,
                           _prune_weight_list, _ZERO_DIVISION_GUARD, value_and_gradient as _serial_vg)

_MIN_PARALLEL_Q = _PT._MIN_PARALLEL_Q


def set_cores(n):
    return _PT.set_cores(n)


# ===========================================================================
# Shared helpers
# ===========================================================================
def _discount_factors(node, allocation, rho):
    """beta_v = exp(-rho l_v) for a (sub)tree's leaves -- exactly as the serial
    fold (`_value_and_raw_gradient`); NOTE: no beta clamp (the fold handles the
    l=0 kink via break_ties, which we mirror by matching break_ties=False)."""
    return {n: math.exp(-rho * allocation[n]) for n in _PT._leaf_names(node)}


# ===========================================================================
# Worker tasks (module-level -> picklable for ProcessPoolExecutor)
# ===========================================================================
def _worker_forward(payload):
    """Round 1: fold subtree forward, return its boundary profile as floats."""
    subtree, sub_alloc, rho = payload
    dfs = _discount_factors(subtree, sub_alloc, rho)
    profile = compute_profiles_nary(subtree, dfs, {})
    return (list(profile.breakpoints), list(profile.values))


def _worker_backward(payload):
    """Round 2: re-fold subtree (checkpoint), seed its weight list, backward,
    return this subtree's leaf gradient {name: dV*/dl}."""
    subtree, sub_alloc, rho, weight_list, rel_tol = payload
    dfs = _discount_factors(subtree, sub_alloc, rho)
    forward_cache = {}
    compute_profiles_nary(subtree, dfs, forward_cache)
    return compute_gradient_nary(subtree, dfs, rho, forward_cache, rel_tol, seed=weight_list)


# Resident variants (CutPlan): subtree shipped once via the pool initializer.
_RESIDENT = None


def _init_worker(subtrees):
    global _RESIDENT
    _RESIDENT = subtrees


def _resident_forward(payload):
    idx, sub_alloc, rho = payload
    subtree = _RESIDENT[idx]
    dfs = _discount_factors(subtree, sub_alloc, rho)
    profile = compute_profiles_nary(subtree, dfs, {})
    return (list(profile.breakpoints), list(profile.values))


def _resident_backward(payload):
    idx, sub_alloc, rho, weight_list, rel_tol = payload
    subtree = _RESIDENT[idx]
    dfs = _discount_factors(subtree, sub_alloc, rho)
    forward_cache = {}
    compute_profiles_nary(subtree, dfs, forward_cache)
    return compute_gradient_nary(subtree, dfs, rho, forward_cache, rel_tol, seed=weight_list)


# ===========================================================================
# Master-side top: fold ABOVE the frontier, cut nodes are precomputed profiles.
# Mirrors sp_gradients2.compute_profiles_nary / compute_gradient_nary, treating a
# cut subtree as a leaf (forward: use its profile; backward: capture its weight
# list instead of emitting a gradient).
# ===========================================================================
def _top_forward(node, cut_profiles, top_cache):
    nid = id(node)
    if nid in cut_profiles:
        bps, vals = cut_profiles[nid]
        profile = PiecewiseLinearProfile(list(bps), list(vals))
        top_cache[nid] = ("cut", profile)
        return profile
    child_profiles = [_top_forward(c, cut_profiles, top_cache) for c in node.children]
    if isinstance(node, SeriesNode):
        k = len(child_profiles)
        suffix = [None] * k
        suffix[k - 1] = child_profiles[k - 1]
        for i in range(k - 2, -1, -1):
            suffix[i] = series_profile(child_profiles[i], suffix[i + 1])
        top_cache[nid] = ("S", child_profiles, suffix)
        return suffix[0]
    profile = parallel_profile_nary(child_profiles)
    top_cache[nid] = ("P", child_profiles)
    return profile


def _top_backward(node, weight_list, top_cache, cut_weights, rel_tol):
    entry = top_cache[id(node)]
    if entry[0] == "cut":
        cut_weights[id(node)] = weight_list          # boundary context for the worker
        return
    if isinstance(node, ControlNode):
        raise AssertionError("control node above the frontier was not cut")

    if isinstance(node, ParallelNode):
        child_profiles = entry[1]
        cumulative = cumulative_weight(weight_list)
        query_points = {point for point, _ in weight_list}
        all_breakpoints = set()
        for p in child_profiles:
            all_breakpoints |= set(p.breakpoints)
        knots = all_breakpoints | query_points
        weight_lists = _leave_one_out_weight_lists(child_profiles, cumulative, knots)
        for child, child_weights in zip(node.children, weight_lists):
            _top_backward(child, _prune_weight_list(child_weights, rel_tol),
                          top_cache, cut_weights, rel_tol)
    else:                                             # SeriesNode: peel left->right
        child_profiles, suffix_profiles = entry[1], entry[2]
        k = len(child_profiles)
        weights = weight_list
        for i in range(k - 1):
            upstream_profile = child_profiles[i]
            downstream_profile = suffix_profiles[i + 1]

            def relocate(point, downstream=downstream_profile):
                downstream_value = downstream(point)
                return (point / downstream_value
                        if downstream_value > _ZERO_DIVISION_GUARD else 0.0)

            upstream_weights = [
                (relocate(point), weight * downstream_profile(point))
                for point, weight in weights]
            downstream_weights = [
                (point, weight * (upstream_profile(relocate(point))
                                  - relocate(point)
                                  * upstream_profile.right_slope(relocate(point))))
                for point, weight in weights]
            _top_backward(node.children[i], upstream_weights, top_cache, cut_weights, rel_tol)
            weights = downstream_weights
        _top_backward(node.children[k - 1], weights, top_cache, cut_weights, rel_tol)


# ===========================================================================
# Persistent pool (stateless path) + serial fallback
# ===========================================================================
_POOL = None
_POOL_CORES = None


def _get_pool(cores):
    global _POOL, _POOL_CORES
    if _POOL is None or _POOL_CORES != cores:
        if _POOL is not None:
            _POOL.shutdown()
        _POOL = ProcessPoolExecutor(max_workers=cores)
        _POOL_CORES = cores
    return _POOL


def shutdown():
    global _POOL, _POOL_CORES
    if _POOL is not None:
        _POOL.shutdown()
        _POOL = None
        _POOL_CORES = None


def _serial(tree, allocation, rho, rel_tol, break_ties=True, tie_eps=1e-7, tie_tol=1e-4):
    """Serial fold reference, matching the tie-breaking convention we reproduce."""
    return _serial_vg(tree, allocation, rho, break_ties=break_ties, tie_eps=tie_eps,
                      tie_tol=tie_tol, native=True, adjoint_rel_tol=rel_tol)


def _apply_break_ties(raw, allocation, names, tie_eps, tie_tol):
    """Wrap a raw-gradient oracle `raw(alloc) -> (v, grad)` with the SAME tie-break
    as sp_gradients2.value_and_gradient: pick the perturbed-direction subgradient at
    a kink (a genuine vertex), leave smooth points unchanged. Two raw evals -- the
    serial fold with break_ties=True does exactly the same two."""
    v, grad = raw(allocation)
    offsets = {name: (i + 1) for i, name in enumerate(sorted(names))}
    perturbed = {name: allocation[name] + tie_eps * offsets[name] for name in names}
    _, perturbed_grad = raw(perturbed)
    if any(abs(perturbed_grad[n] - grad[n]) > tie_tol for n in names):
        grad = perturbed_grad
    return v, grad


def _run_cut(tree, rel_tol, frontier, pool,
             forward_fn, backward_fn, payload_fwd, payload_bwd):
    """Shared body: round1 forward -> master top -> round2 backward."""
    boundary = list(pool.map(forward_fn, payload_fwd(frontier)))
    cut_profiles = {id(sub): prof for sub, prof in zip(frontier, boundary)}

    top_cache = {}
    root_profile = _top_forward(tree, cut_profiles, top_cache)
    v_star = root_profile(0.0)
    cut_weights = {}
    _top_backward(tree, [(0.0, 1.0)], top_cache, cut_weights, rel_tol)

    grad_parts = list(pool.map(backward_fn, payload_bwd(frontier, cut_weights)))
    gradient = {}
    for part in grad_parts:
        gradient.update(part)
    return v_star, gradient


# ===========================================================================
# Public API (drop-in for the serial fold, plus `cores`)
# ===========================================================================
def value_and_gradient(tree, allocation=None, discount_rate=1.0, cores=None,
                       min_parallel_q=_MIN_PARALLEL_Q, rel_tol=0.0,
                       break_ties=True, tie_eps=1e-7, tie_tol=1e-4):
    """Exact V* and gradient via the fold, across `cores` processes (subtree cut).
    Matches sp_gradients2.value_and_gradient(..., native=True) -- including the
    default break_ties=True subgradient at kinks. Falls back to serial when
    parallelism can't pay."""
    if cores is None:
        cores = _PT.CORES
    if discount_rate <= 0:
        raise ValueError("discount_rate must be > 0")
    controls = collect_controls(tree)
    names = [c.name for c in controls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate control names: {names}")
    if allocation is None:
        allocation = {n: 1.0 / len(names) for n in names}

    total_q = sum(c.lockout for c in controls)
    if cores <= 1 or total_q < min_parallel_q or isinstance(tree, ControlNode):
        return _serial(tree, allocation, discount_rate, rel_tol, break_ties, tie_eps, tie_tol)
    frontier = _PT._partition(tree, cores)
    if len(frontier) < 2:
        return _serial(tree, allocation, discount_rate, rel_tol, break_ties, tie_eps, tie_tol)

    pool = _get_pool(cores)
    rho = discount_rate

    def raw(a):                                        # one parallel raw gradient at `a`
        def payload_fwd(fr):
            return [(sub, {n: a[n] for n in _PT._leaf_names(sub)}, rho) for sub in fr]

        def payload_bwd(fr, cut_weights):
            return [(sub, {n: a[n] for n in _PT._leaf_names(sub)}, rho,
                     cut_weights[id(sub)], rel_tol) for sub in fr]

        return _run_cut(tree, rel_tol, frontier, pool,
                        _worker_forward, _worker_backward, payload_fwd, payload_bwd)

    if not break_ties:
        return raw(allocation)
    return _apply_break_ties(raw, allocation, names, tie_eps, tie_tol)


# ===========================================================================
# CutPlan: reusable plan for a FIXED tree (the RM hot path). Partition once,
# subtrees resident in workers; per call ship only allocation + weight lists.
# ===========================================================================
class CutPlan:
    def __init__(self, tree, cores=None, min_parallel_q=_MIN_PARALLEL_Q, rel_tol=0.0,
                 break_ties=True, tie_eps=1e-7, tie_tol=1e-4):
        if cores is None:
            cores = _PT.CORES
        self.tree = tree
        self.cores = cores
        self.rel_tol = rel_tol
        self.break_ties = break_ties
        self.tie_eps = tie_eps
        self.tie_tol = tie_tol
        controls = collect_controls(tree)
        self.names = [c.name for c in controls]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate control names: {self.names}")
        total_q = sum(c.lockout for c in controls)
        splittable = (cores > 1 and total_q >= min_parallel_q
                      and not isinstance(tree, ControlNode))
        self.frontier = _PT._partition(tree, cores) if splittable else [tree]
        self.serial = len(self.frontier) < 2
        if self.serial:
            self.pool = None
            self.piece_leaves = None
            return
        self.piece_leaves = [_PT._leaf_names(s) for s in self.frontier]
        self.pool = ProcessPoolExecutor(max_workers=cores, initializer=_init_worker,
                                        initargs=(self.frontier,))

    def value_and_gradient(self, allocation=None, discount_rate=1.0):
        if discount_rate <= 0:
            raise ValueError("discount_rate must be > 0")
        if allocation is None:
            allocation = {n: 1.0 / len(self.names) for n in self.names}
        if self.serial:
            return _serial(self.tree, allocation, discount_rate, self.rel_tol,
                           self.break_ties, self.tie_eps, self.tie_tol)
        rho, rel_tol = discount_rate, self.rel_tol

        def raw(a):                                   # one parallel raw gradient at `a`
            def payload_fwd(fr):
                return [(i, {n: a[n] for n in self.piece_leaves[i]}, rho)
                        for i in range(len(fr))]

            def payload_bwd(fr, cut_weights):
                return [(i, {n: a[n] for n in self.piece_leaves[i]}, rho,
                         cut_weights[id(fr[i])], rel_tol) for i in range(len(fr))]

            return _run_cut(self.tree, rel_tol, self.frontier, self.pool,
                            _resident_forward, _resident_backward, payload_fwd, payload_bwd)

        if not self.break_ties:
            return raw(allocation)
        return _apply_break_ties(raw, allocation, self.names, self.tie_eps, self.tie_tol)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_plan(tree, cores=None, min_parallel_q=_MIN_PARALLEL_Q, rel_tol=0.0,
              break_ties=True, tie_eps=1e-7, tie_tol=1e-4):
    return CutPlan(tree, cores=cores, min_parallel_q=min_parallel_q, rel_tol=rel_tol,
                   break_ties=break_ties, tie_eps=tie_eps, tie_tol=tie_tol)


# ===========================================================================
# Quick self-check (the full battery lives in the pytest file).
# ===========================================================================
if __name__ == "__main__":
    import time

    from sp_gradients2 import Control, Parallel, Series

    def chain(prefix, depth):
        return Series(*[Control(f"{prefix}_{i}", 6, [0.25, 0.2, 0.15, 0.1, 0.08, 0.05])
                        for i in range(depth)])
    tree = Parallel(*[chain(f"c{k}", 30) for k in range(8)])
    names = _PT._leaf_names(tree)
    alloc = {n: 1.0 / len(names) for n in names}
    rho = 5.0

    vs, gs = _serial(tree, alloc, rho, 0.0)
    t = time.perf_counter()
    vs, gs = _serial(tree, alloc, rho, 0.0)
    t_ser = time.perf_counter() - t

    value_and_gradient(tree, alloc, rho, cores=_PT.CORES)       # warm
    t = time.perf_counter()
    vp, gp = value_and_gradient(tree, alloc, rho, cores=_PT.CORES)
    t_par = time.perf_counter() - t

    dv = abs(vs - vp)
    dg = max(abs(gs[n] - gp[n]) for n in gs)
    print(f"fold: |dV*|={dv:.2e} |dgrad|={dg:.2e}  serial={t_ser * 1000:.0f}ms  "
          f"parallel(cores={_PT.CORES})={t_par * 1000:.0f}ms  speedup={t_ser / t_par:.2f}x")
    shutdown()
