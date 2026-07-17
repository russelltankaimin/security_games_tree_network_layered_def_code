"""
sp_gradients3.py
================

Binary full-profile REVERSE-TAPE exact value-and-subgradient for the defender
problem on a series-parallel network -- the O(Q * d_b) algorithm of

    "Exact Subgradient Computation on Series-Parallel Security Networks"
    (Sec. 5, Algorithms 1-3),

which replaces the O(Q^2) weight-list adjoint of sp_gradients2 by differentiating
the *concrete* piecewise-linear profile construction with reverse-mode AD.

Idea (Sec. 5.1): build every calibrated profile Psi_G with a finite sequence of
scalar operations, recording each on a tape; then one reverse pass produces the
full gradient. Breakpoint POSITIONS are differentiable scalars (Sec. 4): interval
widths enter the parallel integral and series breakpoints move with the child
profiles, so they must be taped, not frozen.

Realisation: a small scalar reverse-mode autodiff (`Var` + tape). Taping every
forward arithmetic op automatically applies the paper's local reverse rules
(Alg. 1/2 reverse rules, leaf tape) -- so they need not be hand-coded. The
m-ary tree is balanced-binarised (Sec. 6) so d_b = O(log-arity depth), avoiding
the right-deep O(Qm) trap.

Public API mirrors sp_gradients2:
    value_and_gradient(tree, allocation=None, discount_rate=1.0) -> (V*, {name: dV*/dl})

Verified elsewhere to match sp_gradients2.value_and_gradient to ~1e-9.
"""

from __future__ import annotations

import heapq
import math

from sp_gradients2 import ControlNode, SeriesNode, ParallelNode, collect_controls

_GUARD = 1e-15
_MERGE_TOL = 1e-12


# =========================================================================
# Scalar reverse-mode autodiff (the "reverse tape").
# =========================================================================
class Var:
    __slots__ = ("val", "adj")

    def __init__(self, val):
        self.val = float(val)
        self.adj = 0.0


# Op codes for the flat tape (avoids a Python closure per arithmetic op).
_ADD, _SUB, _MUL, _DIV = 0, 1, 2, 3


class Tape:
    """Flat tape of (opcode, a, b, out) records, replayed by one inline reverse pass.

    Recording a tuple per op (instead of a closure) and accumulating adjoints
    inline in `backward` (instead of a per-adjoint helper call) removes the two
    dominant constant-factor costs of the reference reverse-mode AD. The `g == 0`
    skip prunes ops that carry no gradient (adding 0 is a no-op), so the result is
    bit-identical to the closure version."""

    __slots__ = ("_ops",)

    def __init__(self):
        self._ops = []

    def add(self, a, b):
        o = Var(a.val + b.val)
        self._ops.append((_ADD, a, b, o))
        return o

    def sub(self, a, b):
        o = Var(a.val - b.val)
        self._ops.append((_SUB, a, b, o))
        return o

    def mul(self, a, b):
        o = Var(a.val * b.val)
        self._ops.append((_MUL, a, b, o))
        return o

    def div(self, a, b):
        # Defensive against a zero (or degenerate) denominator: callers already
        # guard their divisors (piece widths, hull slope gaps, 1 - x*a), but never
        # let one crash the tape. A degenerate ratio contributes nothing, so return
        # a constant 0 and don't record it -- so `backward` never sees bv == 0 either.
        if abs(b.val) < _GUARD:
            return Var(0.0)
        o = Var(a.val / b.val)
        self._ops.append((_DIV, a, b, o))
        return o

    def backward(self):
        for opcode, a, b, o in reversed(self._ops):
            g = o.adj
            if g == 0.0:                          # no gradient flows through this op
                continue
            if opcode == _MUL:
                a.adj += g * b.val
                b.adj += g * a.val
            elif opcode == _ADD:
                a.adj += g
                b.adj += g
            elif opcode == _SUB:
                a.adj += g
                b.adj -= g
            else:                                 # _DIV
                bv = b.val
                a.adj += g / bv
                b.adj -= g * a.val / (bv * bv)


# =========================================================================
# Profile = (xs, ys): breakpoint positions and values, all Var; domain [0, 1].
# Piece j (on [xs[j], xs[j+1]]) has right-slope (ys[j+1]-ys[j])/(xs[j+1]-xs[j]).
# =========================================================================
def _piece_index(xs, sval):
    """Index k with xs[k].val <= sval < xs[k+1].val (clamped to a valid piece)."""
    lo, hi = 0, len(xs) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if xs[mid].val <= sval:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _slope(tape, xs, ys, k):
    # Deep series chains drive V* = prod_i Psi_ci(0) -> 0 and cram breakpoints
    # toward 0 until adjacent ones round to the same float (a zero-width piece).
    # Guard the division: a piece with no width integrates to nothing, so its
    # slope is immaterial -- return 0 rather than dividing by ~0 (ZeroDivisionError).
    if abs(xs[k + 1].val - xs[k].val) < _GUARD:
        return Var(0.0)
    return tape.div(tape.sub(ys[k + 1], ys[k]), tape.sub(xs[k + 1], xs[k]))


# =========================================================================
# Leaf profile: upper hull of the lines L_h(s) = A_h + B_h s (Sec. 2.3).
# =========================================================================
def _leaf_profile(tape, control, beta):
    q, ps = control.lockout, control.success_probs
    A = [Var(0.0)]
    B = [Var(1.0)]
    for h in range(q):                                # recurrence Eq. (4)
        A.append(tape.add(A[h], tape.mul(tape.mul(beta, Var(ps[h])), B[h])))
        B.append(tape.mul(tape.mul(beta, Var(1.0 - ps[h])), B[h]))
    n = len(A)                                        # q + 1 lines

    def xint(i, j):                                   # s where line i == line j
        denom = B[j].val - B[i].val
        return (A[i].val - A[j].val) / denom if abs(denom) > _GUARD else math.inf

    # Upper envelope (max) via a monotone stack, lines by increasing slope.
    order = sorted(range(n), key=lambda h: (B[h].val, A[h].val))
    st = []
    for h in order:
        if st and abs(B[st[-1]].val - B[h].val) < 1e-14:   # equal slope: keep higher intercept
            if A[h].val >= A[st[-1]].val:
                st.pop()
            else:
                continue
        while len(st) >= 2 and xint(st[-2], st[-1]) >= xint(st[-1], h) - _GUARD:
            st.pop()
        st.append(h)

    cross = [xint(st[k], st[k + 1]) for k in range(len(st) - 1)]   # increasing
    start = sum(1 for c in cross if c <= _MERGE_TOL)              # first line active at s=0
    stop = len(st) - 1 - sum(1 for c in cross if c >= 1.0 - _MERGE_TOL)  # active at s=1
    stop = max(stop, start)

    def line_at(h, s_var):
        return tape.add(A[h], tape.mul(B[h], s_var))

    xs = [Var(0.0)]
    plines = [st[start]]
    for k in range(start, stop):
        num = tape.sub(A[st[k]], A[st[k + 1]])
        den = tape.sub(B[st[k + 1]], B[st[k]])
        xs.append(tape.div(num, den))                 # differentiable crossover
        plines.append(st[k + 1])
    xs.append(Var(1.0))

    ys = [line_at(plines[min(j, len(plines) - 1)], xs[j]) for j in range(len(xs))]
    return xs, ys


# =========================================================================
# Binary parallel merge (Algorithm 1): slope product then backward integrate.
# =========================================================================
def _merge_positions(xs1, xs2):
    """Union of two sorted Var-position lists (dedup within tol); shares Vars."""
    out, i, j = [], 0, 0
    while i < len(xs1) or j < len(xs2):
        if j >= len(xs2) or (i < len(xs1) and xs1[i].val <= xs2[j].val):
            cand = xs1[i]; i += 1
        else:
            cand = xs2[j]; j += 1
        if not out or cand.val - out[-1].val > _MERGE_TOL:
            out.append(cand)
    return out


def _parallel_merge(tape, p1, p2):
    (xs1, ys1), (xs2, ys2) = p1, p2
    xs = _merge_positions(xs1, xs2)
    slopes = []
    for j in range(len(xs) - 1):
        mid = 0.5 * (xs[j].val + xs[j + 1].val)
        s1 = _slope(tape, xs1, ys1, _piece_index(xs1, mid))
        s2 = _slope(tape, xs2, ys2, _piece_index(xs2, mid))
        slopes.append(tape.mul(s1, s2))               # g_par = g1 * g2
    ys = [None] * len(xs)
    ys[-1] = Var(1.0)                                 # Psi_G(1) = 1
    for j in range(len(xs) - 2, -1, -1):              # Eq. (13) backward integrate
        width = tape.sub(xs[j + 1], xs[j])
        ys[j] = tape.sub(ys[j + 1], tape.mul(slopes[j], width))
    return xs, ys


# =========================================================================
# Binary series merge (Algorithm 2): transfer-map two-pointer sweep.
# up = upstream, down = downstream.
# =========================================================================
def _series_merge(tape, up, down):
    (xs1, ys1), (xs2, ys2) = up, down
    ONE = Var(1.0)

    def coeffs(j1, j2):
        a1 = _slope(tape, xs1, ys1, j1)
        b1 = tape.sub(ys1[j1], tape.mul(a1, xs1[j1]))
        a2 = _slope(tape, xs2, ys2, j2)
        b2 = tape.sub(ys2[j2], tape.mul(a2, xs2[j2]))
        A = tape.add(a1, tape.mul(b1, a2))            # Eq. (14)
        B = tape.mul(b1, b2)
        return a2, b2, A, B

    j1 = j2 = 0
    a2, b2, A, B = coeffs(j1, j2)
    xs = [Var(0.0)]
    ys = [B]                                          # A*0 + B
    guard = 0
    while xs[-1].val < 1.0 - _MERGE_TOL and guard < len(xs1) + len(xs2) + 4:
        guard += 1
        a2, b2, A, B = coeffs(j1, j2)
        u = xs2[j2 + 1] if j2 + 1 < len(xs2) else ONE            # downstream breakpoint
        if j1 + 1 < len(xs1) and xs1[j1 + 1].val < 1.0 - _MERGE_TOL:
            x = xs1[j1 + 1]                                       # upstream breakpoint (t-coord)
            if abs(1.0 - x.val * a2.val) > _GUARD:
                v = tape.div(tape.mul(x, b2), tape.sub(ONE, tape.mul(x, a2)))  # Eq. (15)
            else:
                v = ONE
        else:
            v = ONE
        uval, vval = u.val, v.val
        snew = ONE if min(uval, vval) >= 1.0 - _MERGE_TOL else (u if uval <= vval else v)
        xs.append(snew)
        ys.append(tape.add(tape.mul(A, snew), B))
        if snew is ONE:
            break
        if uval <= vval + _MERGE_TOL:
            j2 += 1
        if vval <= uval + _MERGE_TOL:
            j1 += 1
    if xs[-1] is not ONE and xs[-1].val < 1.0 - _MERGE_TOL:       # ensure domain ends at 1
        xs.append(ONE)
        ys.append(tape.add(tape.mul(A, ONE), B))
    # NB: do NOT dedup coincident breakpoints here -- an s-position dedup at
    # _MERGE_TOL floors V* at ~1e-12 on deep chains (their true V* underflows far
    # below that). The _slope guard alone handles the zero-width division safely
    # while preserving the tiny values (see mary, which has no dedup and is exact).
    return xs, ys


# =========================================================================
# Direct m-ary algorithms (Sec. 7-8): operate on the original m-ary node,
# O(Q_G log m_G) via a balanced event tree (parallel) / segment tree (series).
# Same profiles as the binary merges (associativity), so the reverse tape gives
# the identical gradient; kept as an alternative that preserves node structure.
# =========================================================================
def _parallel_mary(tape, profiles):
    """Algorithm 4: product/event tree over m children's slopes."""
    m = len(profiles)
    if m == 1:
        return profiles[0]
    ptr = [0] * m                                     # active piece per child

    def child_slope(i):
        return _slope(tape, profiles[i][0], profiles[i][1], ptr[i])

    size = 1
    while size < m:
        size *= 2
    tree = [Var(1.0)] * (2 * size)                    # tournament PRODUCT tree
    for i in range(m):
        tree[size + i] = child_slope(i)
    for i in range(size - 1, 0, -1):
        tree[i] = tape.mul(tree[2 * i], tree[2 * i + 1])

    def update(i):                                    # one child changed -> O(log m) path
        pos = size + i
        tree[pos] = child_slope(i)
        pos //= 2
        while pos >= 1:
            tree[pos] = tape.mul(tree[2 * pos], tree[2 * pos + 1])
            pos //= 2

    heap = []                                         # next breakpoint of each child
    for i in range(m):
        xsi = profiles[i][0]
        if ptr[i] + 1 < len(xsi):
            heapq.heappush(heap, (xsi[ptr[i] + 1].val, i))

    xs = [Var(0.0)]
    slopes = []
    while heap and heap[0][0] < 1.0 - _MERGE_TOL:
        val = heap[0][0]
        slopes.append(tree[1])                        # parent slope on [xs[-1], val]
        bp_var = None
        while heap and heap[0][0] <= val + _MERGE_TOL:
            _, i = heapq.heappop(heap)
            xsi = profiles[i][0]
            if bp_var is None:
                bp_var = xsi[ptr[i] + 1]
            ptr[i] += 1
            update(i)
            if ptr[i] + 1 < len(xsi):
                heapq.heappush(heap, (xsi[ptr[i] + 1].val, i))
        xs.append(bp_var)
    slopes.append(tree[1])
    xs.append(Var(1.0))

    ys = [None] * len(xs)
    ys[-1] = Var(1.0)
    for j in range(len(xs) - 2, -1, -1):
        ys[j] = tape.sub(ys[j + 1], tape.mul(slopes[j], tape.sub(xs[j + 1], xs[j])))
    return xs, ys


def _series_mary(tape, profiles):
    """Algorithm 5: balanced segment tree of transfer-map summaries (A, B, eta).

    Summary (A, B) means tau(s) = s / (A s + B), so the parent piece is A s + B
    (Eq. 22). Combine upstream(left) with downstream(right): Eqs. (24)-(26)."""
    m = len(profiles)
    if m == 1:
        return profiles[0]
    ptr = [0] * m

    def leaf_summary(i):
        xsi, ysi = profiles[i]
        k = ptr[i]
        A = _slope(tape, xsi, ysi, k)
        B = tape.sub(ysi[k], tape.mul(A, xsi[k]))
        eta = xsi[k + 1] if k + 1 < len(xsi) else Var(1.0)
        return (A, B, eta, i)

    def combine(up, dn):
        Aup, Bup, etaup, own_up = up
        Adn, Bdn, etadn, own_dn = dn
        A = tape.add(Aup, tape.mul(Bup, Adn))          # Eq. (24)
        B = tape.mul(Bup, Bdn)                         # Eq. (25)
        if etaup.val < 1.0 - _MERGE_TOL and abs(1.0 - etaup.val * Adn.val) > _GUARD:
            pb = tape.div(tape.mul(etaup, Bdn),            # Eq. (26): pull etaup back
                          tape.sub(Var(1.0), tape.mul(etaup, Adn)))
        else:
            pb = Var(1.0)                                  # upstream exhausted -> no event
        if etadn.val <= pb.val:
            return (A, B, etadn, own_dn)
        return (A, B, pb, own_up)

    nodes = []

    def build(lo, hi):
        nd = {"parent": None}
        nodes.append(nd)
        if lo == hi:
            nd["leaf"] = lo
            nd["sum"] = leaf_summary(lo)
        else:
            mid = (lo + hi) // 2
            left = build(lo, mid)
            right = build(mid + 1, hi)
            left["parent"] = right["parent"] = nd
            nd["l"], nd["r"] = left, right
            nd["sum"] = combine(left["sum"], right["sum"])
        return nd

    root = build(0, m - 1)
    leaves = {nd["leaf"]: nd for nd in nodes if "leaf" in nd}

    def update(i):
        nd = leaves[i]
        nd["sum"] = leaf_summary(i)
        nd = nd["parent"]
        while nd is not None:
            nd["sum"] = combine(nd["l"]["sum"], nd["r"]["sum"])
            nd = nd["parent"]

    A, B, eta, owner = root["sum"]
    xs = [Var(0.0)]
    ys = [B]                                           # A*0 + B
    total = sum(len(p[0]) for p in profiles)
    guard = 0
    while eta.val < 1.0 - _MERGE_TOL and guard < total + 4:
        guard += 1
        A, B, eta, owner = root["sum"]
        if eta.val >= 1.0 - _MERGE_TOL:
            break
        xs.append(eta)
        ys.append(tape.add(tape.mul(A, eta), B))
        ptr[owner] += 1
        update(owner)
    A, B, _, _ = root["sum"]
    xs.append(Var(1.0))
    ys.append(tape.add(tape.mul(A, Var(1.0)), B))
    return xs, ys


def _build_mary(tape, node, betas):
    """Native n-ary build (no binarisation), using the direct m-ary merges."""
    if isinstance(node, ControlNode):
        return _leaf_profile(tape, node, betas[node.name])
    child_profiles = [_build_mary(tape, c, betas) for c in node.children]
    if isinstance(node, ParallelNode):
        return _parallel_mary(tape, child_profiles)
    return _series_mary(tape, child_profiles)


# =========================================================================
# Balanced binarisation (Sec. 6) + recursive profile build.
# =========================================================================
def _balanced(op, kids):
    if len(kids) == 1:
        return kids[0]
    mid = len(kids) // 2
    return (op, _balanced(op, kids[:mid]), _balanced(op, kids[mid:]))


def _binarize(node):
    if isinstance(node, ControlNode):
        return ("c", node)
    op = "S" if isinstance(node, SeriesNode) else "P"
    return _balanced(op, [_binarize(c) for c in node.children])


def _build(tape, bnode, betas):
    tag = bnode[0]
    if tag == "c":
        control = bnode[1]
        return _leaf_profile(tape, control, betas[control.name])
    left = _build(tape, bnode[1], betas)
    right = _build(tape, bnode[2], betas)
    if tag == "P":
        return _parallel_merge(tape, left, right)
    return _series_merge(tape, left, right)            # left upstream, right downstream


# =========================================================================
# Public API
# =========================================================================
def value_and_gradient(tree, allocation=None, discount_rate=1.0, method="binary"):
    """Exact V* and gradient {control_name: dV*/dl} via the reverse tape.

    method="binary" (default): balanced binarisation + binary merges, O(Q d_b).
    method="mary": direct m-ary event/segment trees on the original tree,
    O(sum_G Q_G log m_G). Both give the identical exact gradient (associativity);
    they match sp_gradients2.value_and_gradient (native, break_ties=False) at
    regular points. `allocation` defaults to uniform on the simplex; rho > 0."""
    controls = collect_controls(tree)
    names = [c.name for c in controls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate control names: {names}")
    if allocation is None:
        allocation = {n: 1.0 / len(names) for n in names}
    if discount_rate <= 0:
        raise ValueError("discount_rate must be > 0")

    tape = Tape()
    # Clamp beta just below 1: at l_e = 0 (beta = 1) the leaf's identity line L0 = s
    # meets the attack line exactly at the boundary s = 1 -- a non-regular tie (Sec.
    # 15) at which the raw hull would drop L0 and return an invalid subgradient.
    # beta = 1 - 1e-9 takes the l -> 0+ limit (a valid subgradient), value error ~1e-9.
    _BCAP = 1.0 - 1e-9
    betas = {n: Var(min(math.exp(-discount_rate * allocation[n]), _BCAP)) for n in names}
    if method == "mary":
        xs, ys = _build_mary(tape, tree, betas)
    else:
        xs, ys = _build(tape, _binarize(tree), betas)

    v_star = ys[0]                                     # V* = Psi_N(0), and xs[0] == 0
    v_star.adj = 1.0
    tape.backward()

    # dV*/dl_e = -rho * beta_e * dV*/dbeta_e   (Eq. 12 / Sec. 5.4)
    gradient = {n: -discount_rate * betas[n].val * betas[n].adj for n in names}
    return v_star.val, gradient
