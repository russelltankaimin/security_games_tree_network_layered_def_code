# Gradient backward-pass redesign — spec

## Problem

`value_and_gradient` (sp_gradients2.py) computes `dV*/dl_v` for every control via a
reverse-mode adjoint over piecewise-linear (PWL) profiles. The **forward** value
fold is O(Q·depth) and matches the attacker's `index_table`. The **backward**
gradient pass is **O(Q²) on wide-`Par`-of-long-chains**, which is the blocker for
the "large exact win" regime (H7).

### Root cause (measured)

The backward pass propagates an adjoint `weight_list` (a measure over the buyout
γ). Through node types:

- **Series** (`compute_gradient_nary`, series branch): size-preserving — each
  `(point, weight)` maps to one relocated `(point', weight')`. No growth.
- **Parallel** (`_leave_one_out_weight_lists`): each child's new weight list is
  built over `union(sibling breakpoints) ∪ query_points`. **Every branch inherits
  its siblings' breakpoints.**

So a leaf under a wide `Par` of `C` branches accumulates a weight list of size
≈ the full merged grid (O(Q)); each of the ~`n` leaves then runs
`leaf_beta_sensitivity` over O(Q) points → **O(n·Q) = O(Q²)**.

Profiled at Par(8×Ser(20)), q=32 (Q=5120): defender fold 2.8 s vs attacker
index_table 0.36 s (8×); the ratio grows to ~17× by Q≈15k. `leaf_beta_sensitivity`
drives ~3.2M PWL evaluations; the profiles reach the full Q breakpoints.

Mathematically the adjoint reaching branch `j` is
`w_j(z) = A(z) · ∏_{i≠j} g_i'(z)`, where `g_i = Ψ_child_i` and `A` is the incoming
adjoint. `∏_{i≠j} g_i'` is piecewise-constant with **all siblings'** breakpoints,
so representing `w_j(z)` exactly as a function of z costs O(Q). The blow-up is
inherent to *this representation*, not to the code — hence a redesign, not a
rewrite. (An opt-in relative pruning of `w_j`, `adjoint_rel_tol`, already recovers
~1.7× by dropping negligible-jump points, but caps there; the survivors are all
significant.)

## Target

Compute the exact `dV*/dl_v` for all controls in **O(Q·depth)** (same as the
forward fold / attacker index fold), keeping each control's work bounded by its
**own forward cone**, not the global grid. Bit-identical to the current exact
gradient (default `adjoint_rel_tol=0`).

## Key identity

By the envelope/Danskin theorem (V* is the optimal-attacker value, a max over
policies), the gradient is the pathwise derivative under the optimal policy π*:

```
dV*/dl_v = -rho * beta_v * dV*/dbeta_v ,   dV*/dbeta_v = E_{π*}[ n_v * R / beta_v ]
```

i.e. `dV*/dl_v = -rho * E_{π*}[n_v · R]` — a **local** reward-weighted expected
visit count. Locality is what the redesign must exploit; the current global
adjoint discards it.

## Approach A — cumulative-moment adjoint (reverse-mode, minimal reformulation)

Keep the reverse-mode structure but never materialise `w_j(z)` at sibling
breakpoints. At each `Par` node:

1. A child's contribution to a downstream leaf is `∫ w_j(z)·s(z) dz`, where `s(z)`
   is the leaf/branch sensitivity — **piecewise-linear on the child's OWN knots**.
2. On each child interval `s` is linear, and `w_j` is piecewise-constant; so the
   integral over that interval needs only the **0th and 1st moments** of `w_j`
   there: `∫ w_j` and `∫ z·w_j`.
3. Precompute the cumulative moments `W0(z)=∫_0^z w_j`, `W1(z)=∫_0^z u·w_j du` for
   every branch `j` in **O(Q) per Par node** using the existing leave-one-out
   prefix/suffix product trick (as `_leave_one_out_weight_lists` already does for
   the raw product).
4. Push to child `j` a compact adjoint = `(W0, W1)` **sampled at child `j`'s own
   knots** (O(|child grid|)). The child integrates against its own PWL sensitivity
   in O(|child grid|).

Result: per-`Par` work O(Q); each child sees only its own grid → total
**O(Q·depth)**. Series stays size-preserving (already local). This is the smallest
change that removes the sibling-breakpoint inheritance.

**Derivation to complete:** the exact interval formula for `∫ w_j·s` from
`(W0, W1)` and the child's PWL `s`, and confirming the seed/normalisation match the
current pass.

### Prototype result (2026-07) — simple re-gridding is INCORRECT

A first realisation of Approach A — at each `Par`, project each child's adjoint
atoms onto the child's own profile breakpoints preserving the 0th+1st moment per
cell — was prototyped and checked against the exact gradient. It **fails**:

- `Ser(a,b)` (no Par): error 0 (Series alone is exact).
- **`Par(a,b,c)` single controls: error 0.45** — wrong already at one Par level.
- `Par(Ser(a,b),Ser(c,d))`: error 6e-3; random trees: worst 4.2.

Why: moment-projection onto a grid is exact only if the downstream integrand is
piecewise-LINEAR on that grid. It is not — the leaf sensitivity's kinks sit at the
control's *thresholds*, which do not coincide with the composite profile's
breakpoints, and the Series relocation `z -> z / down(z)` warps atom positions
nonlinearly, so 0th/1st-moment preservation on a coarse grid does not preserve the
integral. Conclusion: Approach A cannot be done by discretised re-gridding.

The correct Approach A must propagate the cumulative moments `M0, M1`
**analytically** and transform them exactly through the Series relocation and the
Par leave-one-out coupling (no discretisation), integrating against the leaf
sensitivity on ITS OWN threshold grid. The Series transform of `M1` under the
nonlinear relocation is the crux and needs derivation. This is deeper than the
bullet list above implied — genuinely research-level, with real correctness risk.

## Approach B — forward `E[n_v·R]` fold (pathwise, cleaner but more new math)

Fold the pathwise quantity directly:
- **Forward:** value profiles Ψ per node (already computed) + the reward carried
  into each node's forward cone.
- **Backward:** the probability of *reaching* each control's decision context
  under π* (Gittins order), and the expected reward-to-go, both decomposing over
  the SP tree because the Gittins index of a control depends only on its own chain.
- Combine to `E[n_v·R]` per control in O(Q·depth).

Higher payoff (conceptually matches the attacker fold and reuses its calibrated-
value/index machinery) but requires deriving the reach-probability recursion under
the interleaved optimal policy — more new math than A.

**Recommendation:** prototype **A** first (localises the existing pass with the
least new derivation); fall back to **B** if the moment formula is unwieldy.

## Verification plan

1. `defender_subgrad_pytest.py` (45 cases) must pass unchanged.
2. `value_and_gradient(..., verify=True)` finite-difference check on a spread of
   small trees (series-heavy, parallel-heavy, mixed) → agreement to ~1e-6.
3. New scaling test: assert wall-clock grows ~O(Q·depth), not O(Q²), on
   Par(C×Ser(L)) as C,L,q scale (the prof_chain / prof_full harness).
4. Cross-check against the current exact pass (`adjoint_rel_tol=0`) to machine
   precision on the `series_heavy.csv` / `h7_targets.csv` instances.

## Effort & risk

- **A:** ~moderate. New: cumulative-moment computation at Par + child-local
  integration; the forward fold and Series backward are reused. Main risk is
  getting the moment/interval algebra bit-identical (mitigated by the FD verify).
- **B:** larger; new reach-probability recursion. Higher upside.
- Payoff: removes the O(Q²) → unlocks H7 (large-Q series-heavy exact wins) and
  makes `gradient_rm`'s exact arm runnable at Q ≳ 10k (folds drop from ~17 s to
  ~1 s, matching the attacker).
