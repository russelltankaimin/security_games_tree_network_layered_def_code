# SP layered-defence — value, gradient & validation

Defender-side code for the series–parallel (SP) layered-defence model: given an
SP parse tree over controls, a per-control delay allocation, and a discount
rate, compute the Stackelberg game value `V*` and its exact (sub)gradient with
respect to the allocation — plus the scripts that prove those numbers correct.
Everything is plain Python 3 (standard library only — no numpy/scipy). The one
exception is the pytest suite, which needs `pytest`.

## The two generations

The directory holds two independent tracks. Both compute the same `V*` and
gradient; they differ in API and in how the gradient is certified.

- **`sp_gradients` track (current).** A clean, single-file library with a
  public `value_and_gradient(tree, allocation, discount_rate)` API, an n-ary
  `Control` / `Series` / `Parallel` tree builder, and an embedded brute-force
  MDP for self-checking. `sp_gradients2.py` is the maintained version: it adds
  `break_ties=True`, which guarantees a *valid subgradient* even at index ties
  (kinks). Use this track for new work.
- **`verify.py` track (original).** The first validation harness, exposing a
  lower-level API (`Vstar_fold`, `grad_reverse`, `brute_Vstar`, `fold`, `PWL`,
  …). The trace/convexity/dump scripts all build on it. Kept for the worked
  numerical walkthroughs.

## The verification logic

The trust anchor in **both** tracks is an **independent brute-force solver** that
shares no code or ideas with the method under test:

- It builds the attacker's MDP over the *full joint state* (status of every
  control) and solves the Bellman equation by backward induction over the DAG of
  states. No profiles, no fold, no Gittins indices. Its cost is exponential in
  the number of controls, so it is a reference oracle for small instances only.
  (`brute_force_value` in the `sp_gradients` track; `brute_Vstar` in `verify.py`.)

Three checks ride on that anchor:

1. **Value.** The forward fold must equal the brute-force MDP value — this tests
   the value computation (and, indirectly, that the block decoupling is exact).
2. **Gradient (smooth points).** The reverse-mode pass must equal central
   **finite differences** of the brute-force value — testing the gradient rules
   with no reference to the fold's own internals. Differentiating the
   brute-force value, not the fold value, is what makes this independent.
3. **Subgradient (everywhere, incl. kinks).** Where the value is *not*
   differentiable, finite differences are not a valid criterion, so
   `verify_subgrad.py` instead certifies the returned vector against the
   **definition** of a subgradient of the convex value `V*` (see below).

All checks pass: value to machine precision, gradient to ~1e-5..1e-11 at
differentiable points, and the subgradient definition is satisfied at the kinks.

## Files

### `sp_gradients` track

| file | what it is | run |
|---|---|---|
| `sp_gradients2.py` | **Maintained library.** Value + exact (sub)gradient fold in O(Q²), n-ary tree builder, embedded brute-force MDP (`verify=True`), and `break_ties=True` for valid subgradients at kinks. | `python3 sp_gradients2.py` (worked demo) |
| `sp_gradients.py` | Earlier single-file version of the same library, **without** the tie-breaking fix. Kept for reference. | `python3 sp_gradients.py` |
| `verify_subgrad.py` | **Definition-based subgradient certifier.** Tests the returned vector by (A) the global subgradient inequality and (B) the directional-derivative inequality, both against the brute-force value. Valid at kinks, where finite differences are not. Run directly for a smooth case, the symmetric-tie kink, and a deliberately wrong vector (which it rejects). | `python3 verify_subgrad.py` |
| `defender_subgrad_pytest.py` | **Pytest suite (45 tests).** Value vs brute force, gradient vs finite differences, subgradient certification, the tie-breaking fix and its regression guard, n-ary↔nested equivalence, monotonicity (`gradient ≤ 0`), convexity, determinism, and input validation. | `pytest defender_subgrad_pytest.py -q` |

`verify_subgrad.py` and the pytest suite import `sp_gradients2`, so keep them in
the same directory.

### `verify.py` track

| file | what it validates | run |
|---|---|---|
| `verify.py` | **Core library + self-test.** PWL value fold, brute-force MDP, reverse-mode gradient, finite differences. Importable (silent); run directly for the full suite. | `python3 verify.py` |
| `convex.py` | Convexity of `V*` on the simplex: random-chord midpoint inequality across instances and discount rates (Prop. A2). | `python3 convex.py` |
| `parallel_chain_verify.py` | The method on the **AAAI-26 parallel-chains topology** `Par(Series(a1,a2),Series(b1,b2))`: fold==brute-MDP value, reverse-mode subgradient==finite differences (the authors' own acceptance check). | `python3 parallel_chain_verify.py` |
| `trace.py` | **Tutorial 1** full numerical trace: `Par(Series(a,b),c)`, single-shot. Prints every forward/backward intermediate + verification. | `python3 trace.py` |
| `trace2.py` | **Tutorial 2** full numerical trace: `Par(Series(a,Par(b,c)),Series(d,e))` with a `q=2` control. | `python3 trace2.py` |
| `dump.py` | Dumps the backward-pass weight lists at every node of `Par(Series(a,b),c)` (the numbers behind the walkthrough). | `python3 dump.py` |

`convex.py`, `parallel_chain_verify.py`, `trace*.py`, and `dump.py` all
`import verify`, so keep them in the same directory.

## The index-tie (kink) story

`V*` is convex but not everywhere differentiable: at an **index tie** (e.g. the
perfectly symmetric `Par(a,b)` at `l_a = l_b = 0.5`) it has a kink, so there is
no unique gradient — only a *subdifferential* (an interval/polytope of valid
subgradients). The two tracks treat this differently:

- **Old behaviour (`verify.py`, `sp_gradients.py`).** The raw reverse pass mixes
  per-branch one-sided derivatives, so at an exact tie it can return a vector
  that is **not** a subgradient. The `verify.py` self-test deliberately surfaces
  this: it prints `ALL TESTS PASS: False` for the symmetric tie and a diagnostic
  showing reverse-mode returns the *right* derivative while central FD returns
  the *midpoint* — both inside the subdifferential, but the strict
  "reverse == central-FD" equality cannot hold at a kink.
- **Current fix (`sp_gradients2.py`).** With `break_ties=True` (the default), the
  pass consistently breaks ties along a fixed generic direction and returns a
  genuine **vertex of the subdifferential** — a guaranteed-valid subgradient.
  Smooth points are unaffected (the exact gradient is returned, and the value is
  identical with or without tie-breaking).

`verify_subgrad.py` is what makes this rigorous at the kink: rather than finite
differences (invalid there), it certifies the returned vector directly against
the subgradient *definition*. The pytest suite encodes both directions —
`test_tie_break_returns_valid_subgradient` (the fix works) and
`test_raw_pass_without_tiebreak_is_not_a_subgradient_at_tie` (a regression guard
that the defect is real). This is the intended behaviour: exact gradient off a
measure-zero set, valid subgradient on it.

## Honest scope note on the AAAI comparison

`parallel_chain_verify.py` runs **our** fold/reverse-mode on the AAAI-26
parallel-chains *topology* and cross-checks it against the brute-force MDP and
finite differences. It is **not** a reimplementation of the AAAI authors' own
gradient algorithm (their stage / product-distribution method). The equivalence
argument is: both their `g(l)` and our gradient are (sub)gradients of the *same*
convex function `V*_l`, each validated against the brute-force value; therefore
they coincide wherever `V*_l` is differentiable and are both valid subgradients
at the measure-zero kinks. A true head-to-head against their algorithm would
require coding their method separately; happy to add that if wanted.
