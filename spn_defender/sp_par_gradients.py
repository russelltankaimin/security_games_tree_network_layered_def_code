"""
sp_par_gradients.py
===================

MULTI-CORE exact value-and-subgradient for the defender problem, wrapping the
serial reverse-tape / m-ary implementations in `sp_gradients3` with a
subtree-boundary CUT so a single `value_and_gradient` call spreads across cores.

Why a cut (and not a naive fork-join)
-------------------------------------
The reverse tape is pure-Python float arithmetic, so the GIL rules out any
speedup from `threading` -- parallelism here must use PROCESSES. But the tape's
`Var` adjoint graph (`a.adj += ...`) is a shared mutable object graph that cannot
cross a process boundary. So we never ship `Var`s: we cut the SP tree into a
frontier of ~`CORES` heavy subtrees and exchange only PLAIN FLOATS across the
boundary --

    forward (parallel)  : each worker builds its subtree -> boundary profile floats
    top merge (master)  : wrap the boundary floats as fresh Vars, run the merges
                          ABOVE the frontier, seed V*.adj = 1, backward
                          -> boundary-profile adjoints (floats)
    backward (parallel) : each worker rebuilds its subtree (checkpoint), seeds the
                          boundary adjoints, backward -> its leaves' gradient

This is textbook reverse-mode with a checkpoint at the cut: correct, and the only
data crossing processes are O(#boundary-breakpoints) floats. The subtree forward
is recomputed once in the backward phase (gradient checkpointing) to keep workers
stateless and robust to any pool scheduling -- costing ~3 forward-units per
subtree instead of 2, spread across cores.

When it helps (and when it doesn't)
-----------------------------------
The `top` merge above the frontier is SERIAL and its cost scales with tree WIDTH
(a k-way parallel merge touches every boundary breakpoint). So, per Amdahl:

  * series_heavy (`Par` of a few DEEP `Ser` chains) -> heavy subtrees, tiny top
    -> good speedup (cut at the top `Par`, one chain per core).
  * highly_parallel / wide-shallow -> the O(Q) top merge dominates -> little gain.
  * a single deep `Ser` chain is inherently sequential (its merges are one fold)
    -> no speedup; we fall back to serial.

Result matches `sp_gradients3.value_and_gradient` for the same `method`: the same
associative merges in the same order, only regrouped at the cut, so it is
bit-identical up to the float copy of the boundary values.

Cores
-----
`CORES` (module global) caps the worker pool; `set_cores(n)` sets it, or pass
`cores=` per call. A persistent pool is reused across calls (RM makes thousands),
so pay process-spawn once. Small / narrow instances fall back to the serial path.

    from sp_par_gradients import value_and_gradient, set_cores
    set_cores(8)
    v, grad = value_and_gradient(tree, allocation, rho, method="mary")
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import sp_gradients3 as _G3
from sp_gradients2 import ControlNode, ParallelNode, collect_controls

# Beta clamp identical to sp_gradients3 (l -> 0+ limit at the non-regular tie).
_BCAP = 1.0 - 1e-9

# Cores cap (the "global variable"). Default leaves headroom like the rest of the
# stack; override with set_cores() or the per-call `cores=` argument.
CORES = max(1, (os.cpu_count() or 2) - 1)

# Below this total state count the IPC / checkpoint overhead outweighs the split,
# so we run serially. Tunable per call via `min_parallel_q=`.
_MIN_PARALLEL_Q = 2000


def set_cores(n):
    """Set the global core cap for subsequent parallel gradient calls."""
    global CORES
    CORES = max(1, int(n))
    return CORES


# ===========================================================================
# Small shared helpers (used on master and inside workers)
# ===========================================================================
def _leaf_names(node):
    return [c.name for c in collect_controls(node)]


def _betas_for(node, allocation, rho):
    """Leaf betas for a (sub)tree, clamped exactly as sp_gradients3 does."""
    return {n: _G3.Var(min(math.exp(-rho * allocation[n]), _BCAP))
            for n in _leaf_names(node)}


def _build_profile(tape, node, betas, method):
    """Forward profile for a (sub)tree, identical to the serial build."""
    if method == "mary":
        return _G3._build_mary(tape, node, betas)
    return _G3._build(tape, _G3._binarize(node), betas)


def _subtree_q(node, memo):
    key = id(node)
    if key in memo:
        return memo[key]
    if isinstance(node, ControlNode):
        q = node.lockout
    else:
        q = sum(_subtree_q(c, memo) for c in node.children)
    memo[key] = q
    return q


# ===========================================================================
# Worker tasks (module-level so they pickle by name for ProcessPoolExecutor)
# ===========================================================================
def _worker_forward(payload):
    """Round 1: build subtree forward, return its boundary profile as floats."""
    subtree, sub_alloc, rho, method = payload
    tape = _G3.Tape()
    betas = _betas_for(subtree, sub_alloc, rho)
    xs, ys = _build_profile(tape, subtree, betas, method)
    return ([x.val for x in xs], [y.val for y in ys])


def _worker_backward(payload):
    """Round 2: rebuild subtree (checkpoint), seed boundary adjoints, backward,
    return this subtree's leaf gradient {name: dV*/dl}."""
    subtree, sub_alloc, rho, method, xs_adj, ys_adj = payload
    tape = _G3.Tape()
    betas = _betas_for(subtree, sub_alloc, rho)
    xs, ys = _build_profile(tape, subtree, betas, method)
    for k in range(len(xs)):          # seed the adjoints handed down by the master
        xs[k].adj = xs_adj[k]
    for k in range(len(ys)):
        ys[k].adj = ys_adj[k]
    tape.backward()
    return {n: -rho * betas[n].val * betas[n].adj for n in betas}


# ---------------------------------------------------------------------------
# Resident-subtree variants (for CutPlan): the subtree structure is shipped ONCE
# via the pool initializer and looked up by index, so repeated calls on a fixed
# tree send only the changing allocation / adjoints, never the tree again.
# ---------------------------------------------------------------------------
_RESIDENT = None                                      # per-worker: list of subtrees


def _init_worker(subtrees):
    global _RESIDENT
    _RESIDENT = subtrees


def _resident_forward(payload):
    idx, sub_alloc, rho, method = payload
    subtree = _RESIDENT[idx]
    tape = _G3.Tape()
    betas = _betas_for(subtree, sub_alloc, rho)
    xs, ys = _build_profile(tape, subtree, betas, method)
    return ([x.val for x in xs], [y.val for y in ys])


def _resident_backward(payload):
    idx, sub_alloc, rho, method, xs_adj, ys_adj = payload
    subtree = _RESIDENT[idx]
    tape = _G3.Tape()
    betas = _betas_for(subtree, sub_alloc, rho)
    xs, ys = _build_profile(tape, subtree, betas, method)
    for k in range(len(xs)):
        xs[k].adj = xs_adj[k]
    for k in range(len(ys)):
        ys[k].adj = ys_adj[k]
    tape.backward()
    return {n: -rho * betas[n].val * betas[n].adj for n in betas}


# ===========================================================================
# Tree partition: a frontier of ~CORES heavy subtrees (cut at node boundaries).
# ===========================================================================
def _partition(tree, k_target):
    """Greedily split the largest INTERNAL frontier node until we have >= k_target
    pieces (or nothing splittable remains). Cutting at node boundaries commutes
    with both the binarisation and the merges, so each piece is exact."""
    memo = {}
    frontier = [tree]
    while len(frontier) < k_target:
        splittable = [n for n in frontier if not isinstance(n, ControlNode)]
        if not splittable:
            break
        biggest = max(splittable, key=lambda n: _subtree_q(n, memo))
        frontier.remove(biggest)
        frontier.extend(biggest.children)
    return frontier


# ===========================================================================
# Master-side top build: merges ABOVE the frontier, cut nodes are leaf-profiles.
# ===========================================================================
def _reduce_binary(tape, merge, profiles):
    """Balanced binary fold matching sp_gradients3._balanced (mid = len // 2)."""
    if len(profiles) == 1:
        return profiles[0]
    mid = len(profiles) // 2
    left = _reduce_binary(tape, merge, profiles[:mid])
    right = _reduce_binary(tape, merge, profiles[mid:])
    return merge(tape, left, right)


def _top_build(tape, node, method, cut_profiles, wrapped):
    """Build the profile above the frontier. `cut_profiles[id(node)]` holds the
    boundary floats for a cut subtree; we wrap them as fresh Vars once and record
    the wrapped Vars in `wrapped` so their adjoints can be read after backward."""
    nid = id(node)
    if nid in cut_profiles:
        xs_vals, ys_vals = cut_profiles[nid]
        w = ([_G3.Var(x) for x in xs_vals], [_G3.Var(y) for y in ys_vals])
        wrapped[nid] = w
        return w
    if isinstance(node, ControlNode):                 # every leaf is under a cut
        raise AssertionError("control node above the frontier was not cut")
    child_profiles = [_top_build(tape, c, method, cut_profiles, wrapped)
                      for c in node.children]
    if method == "mary":
        if isinstance(node, ParallelNode):
            return _G3._parallel_mary(tape, child_profiles)
        return _G3._series_mary(tape, child_profiles)
    merge = _G3._parallel_merge if isinstance(node, ParallelNode) else _G3._series_merge
    return _reduce_binary(tape, merge, child_profiles)


# ===========================================================================
# Persistent process pool (reused across calls; RM issues thousands).
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
    """Tear down the worker pool (optional; freed at interpreter exit anyway)."""
    global _POOL, _POOL_CORES
    if _POOL is not None:
        _POOL.shutdown()
        _POOL = None
        _POOL_CORES = None


# ===========================================================================
# Public API (drop-in for sp_gradients3.value_and_gradient, plus `cores`)
# ===========================================================================
def value_and_gradient(tree, allocation=None, discount_rate=1.0, method="binary",
                       cores=None, min_parallel_q=_MIN_PARALLEL_Q):
    """Exact V* and gradient {control_name: dV*/dl}, computed across `cores`
    processes via a subtree-boundary cut. Matches sp_gradients3.value_and_gradient
    for the same `method`. Falls back to the serial path when parallelism can't
    pay (cores <= 1, tree too small/narrow to split)."""
    if cores is None:
        cores = CORES
    if discount_rate <= 0:
        raise ValueError("discount_rate must be > 0")

    controls = collect_controls(tree)
    names = [c.name for c in controls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate control names: {names}")
    if allocation is None:
        allocation = {n: 1.0 / len(names) for n in names}

    total_q = sum(c.lockout for c in controls)
    # Cheap / un-splittable cases: serial is faster (no IPC, no checkpoint).
    if cores <= 1 or total_q < min_parallel_q or isinstance(tree, ControlNode):
        return _G3.value_and_gradient(tree, allocation, discount_rate, method=method)

    frontier = _partition(tree, cores)
    if len(frontier) < 2:                             # nothing useful to split
        return _G3.value_and_gradient(tree, allocation, discount_rate, method=method)

    pool = _get_pool(cores)

    # ---- Round 1: parallel forward -> boundary profiles (floats) ----
    fwd_payloads = []
    for sub in frontier:
        sub_alloc = {n: allocation[n] for n in _leaf_names(sub)}
        fwd_payloads.append((sub, sub_alloc, discount_rate, method))
    boundary = list(pool.map(_worker_forward, fwd_payloads))
    cut_profiles = {id(sub): prof for sub, prof in zip(frontier, boundary)}

    # ---- Master: merges above the frontier, then one reverse pass ----
    tape = _G3.Tape()
    wrapped = {}
    _xs, ys = _top_build(tape, tree, method, cut_profiles, wrapped)
    v_star = ys[0]                                    # V* = Psi_N(0), xs[0] == 0
    v_star.adj = 1.0
    tape.backward()

    # ---- Round 2: hand each subtree its boundary adjoints, parallel backward ----
    bwd_payloads = []
    for sub in frontier:
        xs_w, ys_w = wrapped[id(sub)]
        xs_adj = [v.adj for v in xs_w]
        ys_adj = [v.adj for v in ys_w]
        sub_alloc = {n: allocation[n] for n in _leaf_names(sub)}
        bwd_payloads.append((sub, sub_alloc, discount_rate, method, xs_adj, ys_adj))
    grad_parts = list(pool.map(_worker_backward, bwd_payloads))

    gradient = {}
    for part in grad_parts:
        gradient.update(part)                        # leaves are disjoint across cuts
    return v_star.val, gradient


# ===========================================================================
# CutPlan: reusable plan for a FIXED tree (the RM hot path).
# Partition once, keep the subtrees resident in the workers, and per call ship
# only the changing allocation + boundary adjoints. Bit-identical to the
# stateless value_and_gradient / serial sp_gradients3 for the same method.
# ===========================================================================
class CutPlan:
    def __init__(self, tree, method="binary", cores=None,
                 min_parallel_q=_MIN_PARALLEL_Q):
        if cores is None:
            cores = CORES
        self.tree = tree
        self.method = method
        self.cores = cores
        controls = collect_controls(tree)
        self.names = [c.name for c in controls]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate control names: {self.names}")
        total_q = sum(c.lockout for c in controls)

        splittable = (cores > 1 and total_q >= min_parallel_q
                      and not isinstance(tree, ControlNode))
        self.frontier = _partition(tree, cores) if splittable else [tree]
        self.serial = len(self.frontier) < 2
        if self.serial:                               # nothing to parallelise: no pool
            self.pool = None
            self.piece_leaves = None
            return
        self.piece_leaves = [_leaf_names(s) for s in self.frontier]
        # Ship the (invariant) subtree structures to every worker exactly once.
        self.pool = ProcessPoolExecutor(max_workers=cores, initializer=_init_worker,
                                        initargs=(self.frontier,))

    def value_and_gradient(self, allocation=None, discount_rate=1.0):
        """Same contract as sp_par_gradients.value_and_gradient, reusing the
        resident partition/subtrees (only allocation + adjoints cross per call)."""
        if discount_rate <= 0:
            raise ValueError("discount_rate must be > 0")
        if allocation is None:
            allocation = {n: 1.0 / len(self.names) for n in self.names}
        if self.serial:
            return _G3.value_and_gradient(self.tree, allocation, discount_rate,
                                          method=self.method)
        method, rho = self.method, discount_rate

        # ---- forward: send only (piece idx, sub-allocation) ----
        fwd = [(i, {n: allocation[n] for n in self.piece_leaves[i]}, rho, method)
               for i in range(len(self.frontier))]
        boundary = list(self.pool.map(_resident_forward, fwd))
        cut_profiles = {id(s): prof for s, prof in zip(self.frontier, boundary)}

        # ---- master top merges + one reverse pass ----
        tape = _G3.Tape()
        wrapped = {}
        _xs, ys = _top_build(tape, self.tree, method, cut_profiles, wrapped)
        v_star = ys[0]
        v_star.adj = 1.0
        tape.backward()

        # ---- backward: send only (piece idx, sub-allocation, boundary adjoints) ----
        bwd = []
        for i, s in enumerate(self.frontier):
            xs_w, ys_w = wrapped[id(s)]
            bwd.append((i, {n: allocation[n] for n in self.piece_leaves[i]}, rho, method,
                        [v.adj for v in xs_w], [v.adj for v in ys_w]))
        grad_parts = list(self.pool.map(_resident_backward, bwd))

        gradient = {}
        for part in grad_parts:
            gradient.update(part)
        return v_star.val, gradient

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_plan(tree, method="binary", cores=None, min_parallel_q=_MIN_PARALLEL_Q):
    """Build a reusable CutPlan for repeated gradient calls on a fixed `tree`."""
    return CutPlan(tree, method=method, cores=cores, min_parallel_q=min_parallel_q)


# ===========================================================================
# Quick self-check + timing (cheap; the full battery lives in the pytest file).
# ===========================================================================
if __name__ == "__main__":
    import time

    from sp_gradients2 import Control, Parallel, Series

    # series_heavy-like: Par of a few deep Ser chains -> the regime the cut wins on.
    def chain(prefix, depth):
        return Series(*[Control(f"{prefix}_{i}", 6, [0.25, 0.2, 0.15, 0.1, 0.08, 0.05])
                        for i in range(depth)])
    tree = Parallel(*[chain(f"c{k}", 40) for k in range(8)])
    names = _leaf_names(tree)
    alloc = {n: 1.0 / len(names) for n in names}
    rho = 5.0

    for method in ("binary", "mary"):
        vs, gs = _G3.value_and_gradient(tree, alloc, rho, method=method)
        t = time.perf_counter()
        vs2, gs2 = _G3.value_and_gradient(tree, alloc, rho, method=method)
        t_ser = time.perf_counter() - t

        t = time.perf_counter()
        vp, gp = value_and_gradient(tree, alloc, rho, method=method, cores=CORES)
        t_par = time.perf_counter() - t

        dv = abs(vs - vp)
        dg = max(abs(gs[n] - gp[n]) for n in gs)
        print(f"{method:6s}: |dV*|={dv:.2e} |dgrad|={dg:.2e}  "
              f"serial={t_ser * 1000:.0f}ms  parallel(cores={CORES})={t_par * 1000:.0f}ms  "
              f"speedup={t_ser / t_par:.2f}x")
    shutdown()
