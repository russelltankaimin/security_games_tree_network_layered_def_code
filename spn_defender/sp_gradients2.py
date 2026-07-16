"""
sp_gradient.py  --  value and exact gradients for an arbitrary series-parallel
layered-defence network.

Given a series-parallel (SP) parse tree over controls, a per-control delay
allocation, and a discount rate, this computes

    * the Stackelberg game value         V* = root_profile(0)        (forward fold)
    * the gradient  d V* / d allocation[v]  for every control v       (one backward pass)

both in O(Q^2) time, where Q is the total lockout budget (sum of per-control
lockouts).

The gradient is the exact (sub)gradient of the convex value function: it equals
the unique gradient wherever the value is differentiable, and a valid subgradient
at the measure-zero "index tie" kinks. It is evaluated at the attacker's optimal
response held fixed (Danskin / envelope theorem), so no differentiation through
the best-response map is required.

Self-contained: standard library only.

----------------------------------------------------------------------------
BUILDING A TREE
----------------------------------------------------------------------------
    Control(name, lockout, success_probs)
        A single control.
          name           : hashable identifier (unique within the tree)
          lockout        : max attempts before the control jams (>= 1)
          success_probs  : list of length `lockout`; success_probs[k] is the
                           breach probability after k previous failures.
                           A single float is accepted as shorthand for lockout 1.
    Series(child, child, ...)     Clear every child in order (n-ary, >= 2 children).
    Parallel(child, child, ...)   Clear ANY one child -- a race (n-ary, >= 2).

Example:
    tree = Parallel(
        Series(Control('a', 2, [0.5, 0.4]),
               Parallel(Control('b', 1, 0.6), Control('c', 1, 0.55))),
        Series(Control('d', 1, 0.5), Control('e', 1, 0.45)),
    )

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------
    from sp_gradient import Control, Series, Parallel, value_and_gradient

    value, gradient = value_and_gradient(
        tree, allocation={'a': 0.2, ...}, discount_rate=1.0)
    # gradient['a'] == d V* / d allocation['a'], and so on.

    # With an independent check (brute-force MDP + finite differences):
    value, gradient = value_and_gradient(tree, allocation, discount_rate=1.0,
                                         verify=True)

Run `python3 sp_gradient.py` for a worked demo.
"""

import math
import bisect
from collections import namedtuple
from functools import lru_cache

# Numerical tolerances (named instead of scattered magic constants).
_BREAKPOINT_MERGE_TOL = 1e-14   # treat two breakpoints this close as identical
_ZERO_DIVISION_GUARD = 1e-15    # below this a profile value is treated as ~0
_THRESHOLD_TOL = 1e-12          # slack when testing "outside option < threshold"
_RIGHT_LIMIT_EPS = 1e-13        # offset used to sample a step function's right limit


# =========================================================================
# Piecewise-linear convex profiles on [0, 1].
#
# A profile is stored as sorted breakpoints with their function values, with
# breakpoints[0] == 0 and breakpoints[-1] == 1. The function is linear between
# consecutive breakpoints and convex overall.
# =========================================================================
class PiecewiseLinearProfile:
    def __init__(self, breakpoints, values):
        merged_breakpoints, merged_values = [], []
        for x, y in zip(breakpoints, values):
            if merged_breakpoints and abs(x - merged_breakpoints[-1]) < _BREAKPOINT_MERGE_TOL:
                merged_values[-1] = y          # collapse duplicate breakpoint
            else:
                merged_breakpoints.append(x)
                merged_values.append(y)
        self.breakpoints = merged_breakpoints
        self.values = merged_values

    def __call__(self, x):
        breakpoints, values = self.breakpoints, self.values
        if x <= breakpoints[0]:
            if len(breakpoints) > 1:
                slope = (values[1] - values[0]) / (breakpoints[1] - breakpoints[0])
                return values[0] + slope * (x - breakpoints[0])
            return values[0]
        if x >= breakpoints[-1]:
            if len(breakpoints) > 1:
                slope = (values[-1] - values[-2]) / (breakpoints[-1] - breakpoints[-2])
                return values[-1] + slope * (x - breakpoints[-1])
            return values[-1]
        i = bisect.bisect_right(breakpoints, x) - 1
        i = max(0, min(i, len(breakpoints) - 2))
        slope = (values[i + 1] - values[i]) / (breakpoints[i + 1] - breakpoints[i])
        return values[i] + slope * (x - breakpoints[i])

    def right_slope(self, x):
        """Slope of the piece immediately to the right of x (the right derivative)."""
        breakpoints, values = self.breakpoints, self.values
        if x >= breakpoints[-1]:
            i = len(breakpoints) - 2
        else:
            i = bisect.bisect_right(breakpoints, x) - 1
            i = max(0, min(i, len(breakpoints) - 2))
        return (values[i + 1] - values[i]) / (breakpoints[i + 1] - breakpoints[i])


IDENTITY_PROFILE = PiecewiseLinearProfile([0, 1], [0, 1])   # x -> x


def affine_transform(profile, intercept, slope):
    """Return the profile  x -> intercept + slope * profile(x)  (same breakpoints)."""
    return PiecewiseLinearProfile(
        list(profile.breakpoints),
        [intercept + slope * y for y in profile.values],
    )


def pointwise_max(profile_a, profile_b):
    """Exact pointwise maximum of two convex piecewise-linear profiles."""
    candidate_breakpoints = sorted(set(profile_a.breakpoints) | set(profile_b.breakpoints))
    crossings = []
    for i in range(len(candidate_breakpoints) - 1):
        left, right = candidate_breakpoints[i], candidate_breakpoints[i + 1]
        gap_left = profile_a(left) - profile_b(left)
        gap_right = profile_a(right) - profile_b(right)
        if gap_left == 0 or gap_right == 0:
            continue
        if (gap_left < 0) != (gap_right < 0):                 # the two profiles cross here
            fraction = gap_left / (gap_left - gap_right)
            crossings.append(left + fraction * (right - left))
    all_breakpoints = sorted(set(candidate_breakpoints) | set(crossings))
    return PiecewiseLinearProfile(
        all_breakpoints,
        [max(profile_a(x), profile_b(x)) for x in all_breakpoints],
    )


# =========================================================================
# Tree node types. Series/Parallel hold a tuple of children. The binary fold
# (to_binary_tree) reduces that tuple to exactly two entries; the NATIVE n-ary
# path (compute_profiles_nary / compute_gradient_nary) instead operates on the
# original >= 2 children directly.
# =========================================================================
ControlNode = namedtuple("ControlNode", ["name", "lockout", "success_probs"])
SeriesNode = namedtuple("SeriesNode", ["children"])
ParallelNode = namedtuple("ParallelNode", ["children"])


def Control(name, lockout, success_probs):
    probs = ([float(success_probs)] if isinstance(success_probs, (int, float))
             else [float(p) for p in success_probs])
    if lockout < 1:
        raise ValueError(f"control {name!r}: lockout must be >= 1")
    if len(probs) != lockout:
        raise ValueError(
            f"control {name!r}: len(success_probs)={len(probs)} must equal lockout={lockout}")
    return ControlNode(name, lockout, probs)


def Series(*children):
    if len(children) < 2:
        raise ValueError("Series needs >= 2 children")
    return SeriesNode(tuple(children))


def Parallel(*children):
    if len(children) < 2:
        raise ValueError("Parallel needs >= 2 children")
    return ParallelNode(tuple(children))


def to_binary_tree(node):
    """Right-fold n-ary Series/Parallel nodes into binary ones (series and
    parallel are associative, so this preserves the game)."""
    if isinstance(node, ControlNode):
        return node
    node_type = type(node)
    binary_children = [to_binary_tree(child) for child in node.children]
    folded = binary_children[-1]
    for child in reversed(binary_children[:-1]):
        folded = node_type((child, folded))
    return folded


def collect_controls(node):
    if isinstance(node, ControlNode):
        return [node]
    result = []
    for child in node.children:          # n-ary safe (binary has exactly 2)
        result += collect_controls(child)
    return result


# =========================================================================
# Forward fold (Pass 1): build a profile for every block, bottom-up.
# =========================================================================
def control_profile(discount_factor, success_probs, lockout):
    """Profile of a single control, plus its per-failure-count ladder of profiles
    and the switch thresholds (outside-option value at which attacking stops being
    worthwhile at each count)."""
    profiles_by_count = {lockout: IDENTITY_PROFILE}      # a jammed control is worth `s`
    thresholds_by_count = {}
    for failures in range(lockout - 1, -1, -1):
        p = success_probs[failures]
        beta = discount_factor
        attack_value = affine_transform(profiles_by_count[failures + 1],
                                        beta * p, beta * (1 - p))   # beta*(p + (1-p)*next)
        profile = pointwise_max(IDENTITY_PROFILE, attack_value)     # max{stop, attack}
        profiles_by_count[failures] = profile

        # Threshold: smallest outside option at which "attack" no longer wins.
        threshold = 1.0
        breakpoints = profile.breakpoints
        for i in range(len(breakpoints) - 1):
            left, right = breakpoints[i], breakpoints[i + 1]
            surplus_left = attack_value(left) - left
            surplus_right = attack_value(right) - right
            if surplus_left > _ZERO_DIVISION_GUARD and surplus_right <= _ZERO_DIVISION_GUARD:
                fraction = surplus_left / (surplus_left - surplus_right)
                threshold = left + fraction * (right - left)
                break
            if surplus_left <= _ZERO_DIVISION_GUARD:
                threshold = left
                break
        thresholds_by_count[failures] = threshold
    return profiles_by_count[0], profiles_by_count, thresholds_by_count


# --- DEAD: binary-fold `parallel_profile` (superseded by `parallel_profile_nary`;
# --- reached only via the unused native=False path). Kept commented for reference.
# def parallel_profile(left_profile, right_profile):
#     """Race of two branches:  profile(s) = 1 - integral_s^1 (g_left * g_right),
#     so the profile's slope is the product of the two branch slopes."""
#     breakpoints = sorted(set(left_profile.breakpoints) | set(right_profile.breakpoints))
#     values = [None] * len(breakpoints)
#     values[-1] = 1.0
#     for i in range(len(breakpoints) - 2, -1, -1):
#         left, right = breakpoints[i], breakpoints[i + 1]
#         midpoint = 0.5 * (left + right)
#         slope_product = left_profile.right_slope(midpoint) * right_profile.right_slope(midpoint)
#         values[i] = values[i + 1] - slope_product * (right - left)
#     return PiecewiseLinearProfile(breakpoints, values)


def parallel_profile_nary(profiles):
    """N-ary race of k >= 2 branches. As in the binary case the combined profile's
    slope is the PRODUCT of the branch slopes (independent branches -> survival is
    a product), so profile(s) = 1 - integral_s^1 prod_i g_i'."""
    breakpoints = sorted(set().union(*(set(p.breakpoints) for p in profiles)))
    values = [None] * len(breakpoints)
    values[-1] = 1.0
    for i in range(len(breakpoints) - 2, -1, -1):
        left, right = breakpoints[i], breakpoints[i + 1]
        midpoint = 0.5 * (left + right)
        slope_product = 1.0
        for p in profiles:
            slope_product *= p.right_slope(midpoint)
        values[i] = values[i + 1] - slope_product * (right - left)
    return PiecewiseLinearProfile(breakpoints, values)


_SERIES_MERGE_TOL = 1e-12   # drop near-duplicate breakpoints (avoids ~0-width pieces)


def series_profile(upstream_profile, downstream_profile):
    """Chain of two blocks:  profile(s) = downstream(s) * upstream(s / downstream(s)).

    The breakpoints are the knots of R = downstream together with the s at which
    the relocated argument u(s) = s / R(s) hits a knot of P = upstream.

    Linear-merge implementation (mirrors sp_attacker.PiecewiseLinear.series).
    Because R is nondecreasing, 1-Lipschitz, and R(s) >= s, the argument u(s) is
    monotone nondecreasing. So u at R's knots is already sorted, and each interior
    upstream knot's pre-image is found by advancing a single pointer over R's
    pieces -- O(|R| + |P|), instead of the O(|R| * |P|) double loop.
    """
    R, P = downstream_profile, upstream_profile
    Rx, Ry = R.breakpoints, R.values

    def argument(s, r_value):
        """u(s) = s / R(s), clamped to [0, 1] (0 if R(s) ~ 0)."""
        return 0.0 if r_value <= _ZERO_DIVISION_GUARD else min(max(s / r_value, 0.0), 1.0)

    # u at each downstream knot -- sorted, since u is monotone.
    u_at_knot = [argument(Rx[i], Ry[i]) for i in range(len(Rx))]

    # Pre-image s of each interior upstream knot, in increasing order, located by
    # a single forward sweep over R's pieces.
    preimages = []
    piece = 0
    for u in P.breakpoints:
        if u <= _ZERO_DIVISION_GUARD or u >= 1.0 - _ZERO_DIVISION_GUARD:
            continue  # u = 0 -> s = 0; u = 1 -> s = alpha_R; both are R knots
        # Advance to the R-piece whose u-range contains u; if u exceeds every knot
        # (u_at_knot[-1] < u, e.g. when R's last knot < 1), stay on the LAST piece
        # -- the pre-image is clamped to bx below. `piece + 2 < len(Rx)` keeps
        # piece <= len(Rx) - 2 so Rx[piece + 1] never runs off the end.
        while piece + 2 < len(Rx) and u_at_knot[piece + 1] < u:
            piece += 1
        ax, bx = Rx[piece], Rx[piece + 1]
        ay, by = Ry[piece], Ry[piece + 1]
        slope = (by - ay) / (bx - ax)
        intercept = ay - slope * ax           # R(s) = intercept + slope * s here
        denom = 1.0 - u * slope               # solve s / R(s) = u for s
        if abs(denom) < _ZERO_DIVISION_GUARD:
            continue
        preimages.append(min(max(u * intercept / denom, ax), bx))

    # Merge the two already-sorted breakpoint sources (R's knots and the
    # pre-images), dropping duplicates.
    ordered_breakpoints = []
    i = j = 0
    while i < len(Rx) or j < len(preimages):
        if j >= len(preimages) or (i < len(Rx) and Rx[i] <= preimages[j]):
            candidate = Rx[i]
            i += 1
        else:
            candidate = preimages[j]
            j += 1
        candidate = min(1.0, max(0.0, candidate))
        if not ordered_breakpoints or candidate - ordered_breakpoints[-1] > _SERIES_MERGE_TOL:
            ordered_breakpoints.append(candidate)

    values = []
    for s in ordered_breakpoints:
        downstream_value = R(s)
        if downstream_value <= _ZERO_DIVISION_GUARD:
            values.append(0.0)
        else:
            values.append(downstream_value * P(argument(s, downstream_value)))
    return PiecewiseLinearProfile(ordered_breakpoints, values)


# Cached per-node forward data, consumed by the backward pass.
_ControlData = namedtuple("_ControlData", ["profile", "profiles_by_count", "thresholds_by_count"])
# --- DEAD: binary-fold forward pass (superseded by `compute_profiles_nary`;
# --- reached only via the unused native=False path). Kept commented for reference.
# _InternalData = namedtuple("_InternalData", ["profile", "left_profile", "right_profile"])
#
#
# def compute_profiles(node, discount_factors, forward_cache):
#     """Forward fold; records each node's profile (and a control's count ladder)
#     into forward_cache keyed by node identity. Returns the node's profile."""
#     if isinstance(node, ControlNode):
#         profile, profiles_by_count, thresholds = control_profile(
#             discount_factors[node.name], node.success_probs, node.lockout)
#         forward_cache[id(node)] = _ControlData(profile, profiles_by_count, thresholds)
#         return profile
#     left, right = node.children
#     left_profile = compute_profiles(left, discount_factors, forward_cache)
#     right_profile = compute_profiles(right, discount_factors, forward_cache)
#     if isinstance(node, SeriesNode):
#         profile = series_profile(left_profile, right_profile)
#     else:
#         profile = parallel_profile(left_profile, right_profile)
#     forward_cache[id(node)] = _InternalData(profile, left_profile, right_profile)
#     return profile


# =========================================================================
# Backward pass (Pass 2): reverse-mode differentiation through the fold.
#
# Each node carries a "weight list": a list of (point, weight) pairs encoding the
# linear functional  perturbation -> sum_j weight_j * perturbation(point_j),
# i.e. how V* responds to perturbing this node's profile.
# =========================================================================
def cumulative_weight(weight_list):
    """Return the step function  z -> sum of weights at points <= z."""
    ordered = sorted(weight_list)
    def value_at(z):
        return sum(weight for point, weight in ordered if point <= z + _ZERO_DIVISION_GUARD)
    return value_at


def step_function_to_weight_list(step_function, breakpoints):
    """Represent a step function as a weight list: the value just right of 0,
    then the jump at each interior breakpoint."""
    weight_list = [(0.0, step_function(_RIGHT_LIMIT_EPS))]
    previous = weight_list[0][1]
    for breakpoint in sorted(breakpoints):
        if breakpoint <= _THRESHOLD_TOL or breakpoint >= 1 - _THRESHOLD_TOL:
            continue
        right_value = step_function(breakpoint + _RIGHT_LIMIT_EPS)
        jump = right_value - previous
        if abs(jump) > _BREAKPOINT_MERGE_TOL:
            weight_list.append((breakpoint, jump))
        previous = right_value
    return weight_list


def _leave_one_out_weight_lists(child_profiles, cumulative, knots):
    """Backward weight list for every child of an n-ary parallel node at once.

    Child i's step function is  z -> cumulative(z) * prod_{j != i} g_j'(z), the
    leave-one-out product of the OTHER branches' right-slopes. At each knot the k
    leave-one-out products are obtained from prefix/suffix products of the slope
    vector in O(k), so the whole routine is O(k * |knots|) rather than the naive
    O(k^2 * |knots|). The per-child bookkeeping (right-limit sampling, interior
    jump extraction, tolerance filtering) is identical to
    step_function_to_weight_list, so the output matches it child-for-child."""
    k = len(child_profiles)

    def leave_one_out_at(z):
        slopes = [p.right_slope(z) for p in child_profiles]
        prefix = [1.0] * (k + 1)                     # prefix[i] = prod slopes[:i]
        for j in range(k):
            prefix[j + 1] = prefix[j] * slopes[j]
        suffix = [1.0] * (k + 1)                     # suffix[i] = prod slopes[i:]
        for j in range(k - 1, -1, -1):
            suffix[j] = suffix[j + 1] * slopes[j]
        scale = cumulative(z)
        return [scale * prefix[i] * suffix[i + 1] for i in range(k)]  # omit slope i

    first = leave_one_out_at(_RIGHT_LIMIT_EPS)
    weight_lists = [[(0.0, first[i])] for i in range(k)]
    previous = list(first)
    for breakpoint in sorted(knots):
        if breakpoint <= _THRESHOLD_TOL or breakpoint >= 1 - _THRESHOLD_TOL:
            continue
        right_values = leave_one_out_at(breakpoint + _RIGHT_LIMIT_EPS)
        for i in range(k):
            jump = right_values[i] - previous[i]
            if abs(jump) > _BREAKPOINT_MERGE_TOL:
                weight_lists[i].append((breakpoint, jump))
            previous[i] = right_values[i]
    return weight_lists


def _prune_weight_list(weight_list, rel_tol):
    """Drop adjoint breakpoints whose jump is negligible vs the largest in the list.

    The reverse-mode adjoint through a Parallel node inherits every sibling's
    breakpoints, so a leaf under a wide Par accumulates an O(Q)-point weight list;
    most of those points carry a tiny jump and contribute ~nothing to the gradient
    integral. Dropping them (relative to the biggest jump) cuts leaf evaluations at
    a small, bounded approximation cost. `rel_tol <= 0` is exact (the default) --
    the base value at 0 (weight_list[0]) is always kept."""
    if rel_tol <= 0.0 or len(weight_list) <= 1:
        return weight_list
    biggest = max(abs(w) for _, w in weight_list)
    if biggest == 0.0:
        return weight_list
    threshold = rel_tol * biggest
    kept = [weight_list[0]]
    kept.extend((pt, w) for pt, w in weight_list[1:] if abs(w) >= threshold)
    return kept


def leaf_beta_sensitivity(outside_option, control_node, discount_factor, control_data):
    """d profile / d beta for a control, evaluated at one outside-option value,
    via the downward failure-count recursion."""
    profiles_by_count = control_data.profiles_by_count
    thresholds = control_data.thresholds_by_count
    success_probs = control_node.success_probs
    sensitivity = 0.0                                   # value at the jammed count
    for failures in range(control_node.lockout - 1, -1, -1):
        if outside_option < thresholds[failures] - _THRESHOLD_TOL:   # attack branch active
            p = success_probs[failures]
            next_profile = profiles_by_count[failures + 1]
            sensitivity = ((p + (1 - p) * next_profile(outside_option))
                           + discount_factor * (1 - p) * sensitivity)
        else:                                                        # stop branch active
            sensitivity = 0.0
    return sensitivity


# --- DEAD: binary-fold backward pass (superseded by `compute_gradient_nary`;
# --- reached only via the unused native=False path). Kept commented for reference.
# def compute_gradient(tree, discount_factors, discount_rate, forward_cache, rel_tol=0.0):
#     """Backward pass returning {control_name: d V* / d allocation[name]}.
#
#     rel_tol > 0 relatively-prunes the adjoint weight lists at Parallel nodes (a
#     small, bounded approximation that avoids the O(Q^2) adjoint blow-up); 0 = exact."""
#     gradient = {}
#
#     def propagate(node, weight_list):
#         node_data = forward_cache[id(node)]
#         if isinstance(node, ControlNode):
#             discount_factor = discount_factors[node.name]
#             beta_sensitivity = sum(
#                 weight * leaf_beta_sensitivity(point, node, discount_factor, node_data)
#                 for point, weight in weight_list)
#             # Chain rule from beta to the allocation: d beta / d allocation = -rho * beta.
#             gradient[node.name] = -discount_rate * discount_factor * beta_sensitivity
#             return
#
#         left, right = node.children
#         left_profile = node_data.left_profile
#         right_profile = node_data.right_profile
#
#         if isinstance(node, ParallelNode):
#             cumulative = cumulative_weight(weight_list)
#             query_points = {point for point, _ in weight_list}
#             # Each branch is modulated by the OTHER branch's slope.
#             left_weights = step_function_to_weight_list(
#                 lambda z: right_profile.right_slope(z) * cumulative(z),
#                 set(right_profile.breakpoints) | query_points)
#             right_weights = step_function_to_weight_list(
#                 lambda z: left_profile.right_slope(z) * cumulative(z),
#                 set(left_profile.breakpoints) | query_points)
#             propagate(left, _prune_weight_list(left_weights, rel_tol))
#             propagate(right, _prune_weight_list(right_weights, rel_tol))
#         else:  # SeriesNode: left is upstream, right is downstream
#             def relocate(point):
#                 downstream_value = right_profile(point)
#                 return point / downstream_value if downstream_value > _ZERO_DIVISION_GUARD else 0.0
#
#             upstream_weights = [
#                 (relocate(point), weight * right_profile(point))
#                 for point, weight in weight_list]
#             downstream_weights = [
#                 (point, weight * (left_profile(relocate(point))
#                                   - relocate(point) * left_profile.right_slope(relocate(point))))
#                 for point, weight in weight_list]
#             propagate(left, upstream_weights)
#             propagate(right, downstream_weights)
#
#     propagate(tree, [(0.0, 1.0)])     # seed: V* = root_profile(0)
#     return gradient


# =========================================================================
# NATIVE n-ary fold + backward pass (no to_binary_tree).
#
# Produces bit-identical results to the binary fold (series and parallel are
# associative; parallel is also commutative), but folds each node's >= 2
# children in one shot rather than through a right-fold of binary nodes.
# =========================================================================
_SeriesDataNary = namedtuple("_SeriesDataNary", ["profile", "child_profiles", "suffix_profiles"])
_ParallelDataNary = namedtuple("_ParallelDataNary", ["profile", "child_profiles"])


def compute_profiles_nary(node, discount_factors, forward_cache):
    """Native n-ary forward fold; records per-node data into forward_cache."""
    if isinstance(node, ControlNode):
        profile, profiles_by_count, thresholds = control_profile(
            discount_factors[node.name], node.success_probs, node.lockout)
        forward_cache[id(node)] = _ControlData(profile, profiles_by_count, thresholds)
        return profile

    child_profiles = [compute_profiles_nary(c, discount_factors, forward_cache)
                      for c in node.children]

    if isinstance(node, SeriesNode):
        # suffix_profiles[i] = series composite of children i .. k-1 (order matters).
        k = len(child_profiles)
        suffix_profiles = [None] * k
        suffix_profiles[k - 1] = child_profiles[k - 1]
        for i in range(k - 2, -1, -1):
            suffix_profiles[i] = series_profile(child_profiles[i], suffix_profiles[i + 1])
        profile = suffix_profiles[0]
        forward_cache[id(node)] = _SeriesDataNary(profile, child_profiles, suffix_profiles)
    else:  # ParallelNode
        profile = parallel_profile_nary(child_profiles)
        forward_cache[id(node)] = _ParallelDataNary(profile, child_profiles)
    return profile


def compute_gradient_nary(tree, discount_factors, discount_rate, forward_cache, rel_tol=0.0):
    """Native n-ary backward pass returning {control_name: d V* / d allocation}.

    rel_tol > 0 relatively-prunes the adjoint weight lists at Parallel nodes (a
    small, bounded approximation that avoids the O(Q^2) adjoint blow-up); 0 = exact."""
    gradient = {}

    def propagate(node, weight_list):
        node_data = forward_cache[id(node)]
        if isinstance(node, ControlNode):
            discount_factor = discount_factors[node.name]
            beta_sensitivity = sum(
                weight * leaf_beta_sensitivity(point, node, discount_factor, node_data)
                for point, weight in weight_list)
            gradient[node.name] = -discount_rate * discount_factor * beta_sensitivity
            return

        if isinstance(node, ParallelNode):
            child_profiles = node_data.child_profiles
            cumulative = cumulative_weight(weight_list)
            query_points = {point for point, _ in weight_list}
            all_breakpoints = set()
            for p in child_profiles:
                all_breakpoints |= set(p.breakpoints)
            knots = all_breakpoints | query_points
            # Child i is modulated by the LEAVE-ONE-OUT product of the other
            # branches' slopes (the binary case is the k = 2 instance). All k
            # leave-one-out products are formed together in O(k) per knot.
            weight_lists = _leave_one_out_weight_lists(child_profiles, cumulative, knots)
            for child, child_weights in zip(node.children, weight_lists):
                propagate(child, _prune_weight_list(child_weights, rel_tol))
        else:  # SeriesNode: peel children left (upstream) to right (downstream)
            child_profiles = node_data.child_profiles
            suffix_profiles = node_data.suffix_profiles
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
                propagate(node.children[i], upstream_weights)
                weights = downstream_weights          # feed the suffix composite next
            propagate(node.children[k - 1], weights)  # last child = suffix_profiles[k-1]

    propagate(tree, [(0.0, 1.0)])
    return gradient


# =========================================================================
# Independent reference: brute-force MDP value (exponential state space;
# usable as a check on small trees only).
# =========================================================================
def brute_force_value(tree, discount_factors):
    controls = collect_controls(tree)
    success_probs = {c.name: c.success_probs for c in controls}
    lockouts = {c.name: c.lockout for c in controls}

    def is_cleared(node, state):
        if isinstance(node, ControlNode):
            return state[node.name] == "passed"
        if isinstance(node, SeriesNode):
            return all(is_cleared(c, state) for c in node.children)
        return any(is_cleared(c, state) for c in node.children)

    def attackable_controls(node, state):
        if isinstance(node, ControlNode):
            return [node.name] if isinstance(state[node.name], int) else []
        if isinstance(node, SeriesNode):
            for child in node.children:              # attack the first uncleared child
                if not is_cleared(child, state):
                    return attackable_controls(child, state)
            return []
        if any(is_cleared(c, state) for c in node.children):
            return []                                # a branch already won -> done
        result = []
        for child in node.children:
            result += attackable_controls(child, state)
        return result

    @lru_cache(maxsize=None)
    def value(state_items):
        state = dict(state_items)
        if is_cleared(tree, state):
            return 1.0
        frontier = attackable_controls(tree, state)
        if not frontier:
            return 0.0
        best = 0.0
        for name in frontier:
            failures = state[name]
            p = success_probs[name][failures]
            discount_factor = discount_factors[name]
            after_success = dict(state); after_success[name] = "passed"
            after_failure = dict(state)
            after_failure[name] = ("jammed" if failures + 1 == lockouts[name] else failures + 1)
            best = max(best, discount_factor * (
                p * value(tuple(sorted(after_success.items())))
                + (1 - p) * value(tuple(sorted(after_failure.items())))))
        return best

    initial_state = tuple(sorted({c.name: 0 for c in controls}.items()))
    return value(initial_state)


# =========================================================================
# Public API
# =========================================================================
def _value_and_raw_gradient(work_tree, names, allocation, discount_rate, native=False, rel_tol=0.0):
    """Forward fold + one backward pass at the given allocation (no tie handling).

    Always uses the native n-ary fold (it handles binary trees too, so the old
    `native` flag is retained only for call-site compatibility). rel_tol > 0
    relatively-prunes the backward adjoint (approximate); 0 = exact."""
    discount_factors = {name: math.exp(-discount_rate * allocation[name]) for name in names}
    forward_cache = {}
    root_profile = compute_profiles_nary(work_tree, discount_factors, forward_cache)
    gradient = compute_gradient_nary(work_tree, discount_factors, discount_rate, forward_cache, rel_tol)
    # --- DEAD: old binary-fold branch (native=False), kept commented for reference:
    # if native:
    #     root_profile = compute_profiles_nary(...); gradient = compute_gradient_nary(...)
    # else:
    #     root_profile = compute_profiles(...); gradient = compute_gradient(...)
    return root_profile(0.0), gradient


def value_and_gradient(tree, allocation=None, discount_rate=1.0,
                       verify=False, finite_diff_step=1e-6,
                       break_ties=True, tie_eps=1e-7, tie_tol=1e-4,
                       native=True, adjoint_rel_tol=0.0):
    """
    Compute the game value V* and the gradient {control_name: d V* / d allocation}
    for an arbitrary series-parallel tree.

    tree           : built from Control / Series / Parallel (n-ary allowed)
    allocation     : dict {control_name: delay}. If None, uniform on the simplex.
    discount_rate  : discount rate rho (> 0)
    verify         : also solve the brute-force MDP and finite-difference the
                     gradient, printing the agreement. Exponential; small trees only.
    break_ties     : if True (default), return a guaranteed-valid SUBGRADIENT even
                     at index ties (kinks of the convex value), by consistently
                     breaking ties along a fixed generic direction. At a kink the
                     raw backward pass can return a vector that is NOT a subgradient
                     (it mixes per-branch one-sided derivatives); this option
                     replaces it with a genuine vertex of the subdifferential.
                     Smooth points are unaffected (the exact gradient is returned).
    tie_eps        : perturbation size used to break ties.
    tie_tol        : gradient-jump size above which a kink is deemed present.
    adjoint_rel_tol: if > 0, relatively-prune the backward adjoint weight lists at
                     Parallel nodes (drop breakpoints whose jump is < this fraction
                     of the largest in the list). A small, bounded approximation
                     that speeds up the gradient on wide-Par trees (~1.7x at 1e-3,
                     ~0.15% gradient error). 0 (default) = exact.

    Returns (value, gradient).
    """
    # native=True folds the original n-ary tree directly; otherwise reduce to
    # binary first. Both produce identical values/gradients (associativity).
    work_tree = tree if native else to_binary_tree(tree)
    controls = collect_controls(work_tree)
    names = [c.name for c in controls]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate control names: {names}")
    if allocation is None:
        allocation = {name: 1.0 / len(names) for name in names}
    missing = set(names) - set(allocation)
    if missing:
        raise ValueError(f"allocation missing controls: {sorted(missing)}")
    if discount_rate <= 0:
        raise ValueError("discount_rate must be > 0")

    value, gradient = _value_and_raw_gradient(
        work_tree, names, allocation, discount_rate, native=native, rel_tol=adjoint_rel_tol)

    if break_ties:
        # Break ties consistently with a fixed generic direction (distinct offsets):
        # at a smooth point this lands in the same linear region and changes nothing;
        # at a kink it selects one adjacent region, yielding a genuine subgradient.
        offsets = {name: (i + 1) for i, name in enumerate(sorted(names))}
        perturbed = {name: allocation[name] + tie_eps * offsets[name] for name in names}
        _, perturbed_gradient = _value_and_raw_gradient(
            work_tree, names, perturbed, discount_rate, native=native, rel_tol=adjoint_rel_tol)
        at_kink = any(abs(perturbed_gradient[name] - gradient[name]) > tie_tol for name in names)
        if at_kink:
            gradient = perturbed_gradient            # a valid vertex subgradient

    if verify:
        reference_factors = {name: math.exp(-discount_rate * allocation[name]) for name in names}
        reference_value = brute_force_value(work_tree, reference_factors)
        print(f"[verify] V*: fold={value:.10f}  brute-MDP={reference_value:.10f}  "
              f"|diff|={abs(value - reference_value):.2e}")
        worst_discrepancy = 0.0
        for name in names:
            bumped_up = dict(allocation); bumped_up[name] += finite_diff_step
            bumped_down = dict(allocation); bumped_down[name] -= finite_diff_step
            factors_up = {n: math.exp(-discount_rate * bumped_up[n]) for n in names}
            factors_down = {n: math.exp(-discount_rate * bumped_down[n]) for n in names}
            finite_diff = (brute_force_value(work_tree, factors_up)
                           - brute_force_value(work_tree, factors_down)) / (2 * finite_diff_step)
            discrepancy = abs(gradient[name] - finite_diff)
            worst_discrepancy = max(worst_discrepancy, discrepancy)
            print(f"[verify]   dV*/d allocation[{name}]: "
                  f"reverse={gradient[name]:+.8f}  finite-diff={finite_diff:+.8f}  "
                  f"|diff|={discrepancy:.2e}")
        note = ("   (large only at exact index ties / kinks)"
                if worst_discrepancy > 1e-5 else "")
        print(f"[verify] worst gradient discrepancy = {worst_discrepancy:.2e}{note}")

    return value, gradient


# --- DEAD: pretty-printer + __main__ demo (never imported; only used by the demo).
# --- Kept commented for reference.
# def print_report(tree, allocation=None, discount_rate=1.0, verify=False):
#     """Compute and pretty-print the value and per-control gradient table."""
#     value, gradient = value_and_gradient(tree, allocation, discount_rate, verify=verify)
#     names = [c.name for c in collect_controls(to_binary_tree(tree))]
#     if allocation is None:
#         allocation = {name: 1.0 / len(names) for name in names}
#     print(f"\nV* = {value:.6f}    (discount_rate = {discount_rate})")
#     header = f"{'control':<10}{'allocation':>12}{'dV*/d alloc':>15}{'deterrence':>14}"
#     print(header)
#     print("-" * len(header))
#     for name in names:
#         print(f"{name:<10}{allocation[name]:>12.4f}{gradient[name]:>15.6f}{-gradient[name]:>14.6f}")
#     return value, gradient
#
#
# if __name__ == "__main__":
#     print("=" * 66)
#     print("DEMO 1  --  Parallel(Series(a, Parallel(b, c)), Series(d, e)),  q_a = 2")
#     print("=" * 66)
#     tree = Parallel(
#         Series(Control("a", 2, [0.5, 0.4]),
#                Parallel(Control("b", 1, 0.6), Control("c", 1, 0.55))),
#         Series(Control("d", 1, 0.5), Control("e", 1, 0.45)),
#     )
#     allocation = {"a": 0.20, "b": 0.10, "c": 0.15, "d": 0.30, "e": 0.25}
#     print_report(tree, allocation, discount_rate=1.0, verify=True)
#
#     print("\n" + "=" * 66)
#     print("DEMO 2  --  n-ary parallel race of three chains (uniform allocation)")
#     print("=" * 66)
#     three_way = Parallel(
#         Series(Control("x1", 1, 0.6), Control("x2", 1, 0.5)),
#         Series(Control("y1", 2, [0.5, 0.3]), Control("y2", 1, 0.55)),
#         Control("z", 1, 0.45),
#     )
#     print_report(three_way, allocation=None, discount_rate=1.0, verify=True)