# Validation scripts — SP layered-defence value & gradient

These are the scripts used to check the series–parallel (SP) defender work:
the value fold, the convexity of `V*`, and the exact reverse-mode gradient.
Everything is plain Python 3 (standard library only — no numpy/scipy needed).

## The verification logic

The trust anchor is an **independent brute-force solver** that shares no code or
ideas with the method under test:

- `brute_Vstar` builds the attacker's MDP over the *full joint state* (status of
  every control) and solves the Bellman equation by backward induction
  (memoized recursion over the DAG of states). It uses no profiles, no fold, no
  Gittins indices. Its cost is exponential in the number of controls, so it is a
  reference oracle for small instances only.

Two checks ride on that anchor:

1. **Value.** The forward fold `Vstar_fold` must equal `brute_Vstar` — this tests
   the value computation (and, indirectly, that the "defer-to-`s`" block
   decoupling is exact).
2. **Gradient.** The reverse-mode pass `grad_reverse` must equal central
   **finite differences** of `brute_Vstar` — this tests the gradient rules
   (B2–B4) with no reference to the fold's own internals. (Differentiating the
   brute-force value, not the fold value, is what makes this independent.)

Both checks have passed: value to machine precision, gradient to ~1e-11 at every
differentiable point, across parallel / series / nested instances.

## Files

| file | what it validates | run |
|---|---|---|
| `verify.py` | **Core library + self-test.** PWL value fold, brute-force MDP, reverse-mode gradient, finite differences. Importable (silent); run directly for the full suite. | `python3 verify.py` |
| `convex.py` | Convexity of `V*` on the simplex: random-chord midpoint inequality across instances and discount rates (Prop. A2). | `python3 convex.py` |
| `pc.py` | The method on the **AAAI-26 parallel-chains topology** `Par(Series(a1,a2),Series(b1,b2))`: fold==brute-MDP value, reverse-mode subgradient==finite differences (the authors' own acceptance check). | `python3 pc.py` |
| `trace.py` | **Tutorial 1** full numerical trace: `Par(Series(a,b),c)`, single-shot. Prints every forward/backward intermediate + verification. | `python3 trace.py` |
| `trace2.py` | **Tutorial 2** full numerical trace: `Par(Series(a,Par(b,c)),Series(d,e))` with a `q=2` control. | `python3 trace2.py` |
| `dump.py` | Dumps the backward-pass weight lists at every node of `Par(Series(a,b),c)` (the numbers behind the walkthrough). | `python3 dump.py` |

`convex.py`, `pc.py`, `trace*.py`, `dump.py` all `import verify`, so keep them in
the same directory.

## Reading the self-test output (`python3 verify.py`)

The suite prints six instances, then a diagnostic. One line reads
`ALL TESTS PASS: False`. This is **expected and not a bug**: the failing case is
the *perfectly symmetric* `Par(a,b)` at `l_a=l_b=0.5`, which is a point where `V*`
is **not differentiable** (an index tie / kink). There is no unique gradient
there, so the strict "reverse == central-FD" equality cannot hold:

- reverse-mode returns the **right derivative** (`-0.092`),
- central FD returns the **midpoint** of the one-sided derivatives (`-0.244`),
- both lie inside the subdifferential interval `[-0.395, -0.092]`, i.e. both are
  *valid subgradients*.

The diagnostic block immediately below confirms exactly this (it prints the
subdifferential interval and `reverse-mode value in subdifferential? True`) and
shows that moving infinitesimally off the tie restores reverse==FD to ~1e-11.
This is the B5 behaviour: exact gradient off a measure-zero set, valid
subgradient on it.

## Honest scope note on the AAAI comparison

`pc.py` runs **our** fold/reverse-mode on the AAAI-26 parallel-chains *topology*
and cross-checks it against the brute-force MDP and finite differences. It is
**not** a reimplementation of the AAAI authors' own gradient algorithm (their
stage / product-distribution method). The equivalence argument is: both their
`g(l)` and our gradient are (sub)gradients of the *same* convex function `V*_l`,
each validated against finite differences; therefore they coincide wherever
`V*_l` is differentiable and are both valid subgradients at the measure-zero
kinks. A true head-to-head against their algorithm would require coding their
method separately; happy to add that if wanted.
