# SP layered-defence — the optimal defender

Defender-side solvers for the series–parallel (SP) layered-defence model. Where
`spn_attacker` computes the **optimal attacker** and `spn_defender` computes the
**value and gradient** of the game, this directory closes the loop: it finds the
defender's optimal allocation of a fixed *time* budget across controls so as to
minimise the attacker's value — the **min–max (Stackelberg) solution**.

Everything is plain Python 3 (standard library only — no numpy/scipy). The two
solvers import the attacker/defender building blocks from the sibling
`spn_attacker/` and `spn_defender/` directories (wired up at import time via
`sys.path`, so no installation step is needed).

## The min–max problem

The defender splits a time budget across the controls: an allocation `l_v >= 0`
in a convex feasible set. Each control's per-attempt discount is set by its time
share,

```text
beta_v = exp(-rho * l_v)          (rho > 0 is the discount rate)
```

so spending more time on a control lowers its `beta` and deters attacks through
it. The attacker best-responds optimally, yielding the Stackelberg value
`V*(l) = max_pi u(l, pi)`. The defender minimises it:

```text
min_{l in K}  max_pi  u(l, pi).
```

- **Inner max** (attacker) is solved *exactly* every round by the optimal Gittins
  index policy — `sp_attacker.index_table` produces the coupled index table
  `alpha(v, k)` (one index per control `v` and failure count `k`).
- **Outer min** (defender) is a convex program: `sp_gradients2.value_and_gradient`
  returns `V*(l)` and the exact `(sub)gradient dV*/dl` evaluated at the attacker's
  fixed optimal response (Danskin / envelope theorem — no differentiating through
  the best-response map). `V*` is convex in `l` but **nonsmooth** (kinks at
  index ties), which drives the choice of optimiser below.

## `regret_matching.py` — feasible set = probability simplex

Solves the case `K = { l >= 0, sum_v l_v = 1 }` with **Hart–Mas-Colell regret
matching**. Each control is an "action"; starting from the uniform allocation,
each round accumulates regret `r_v = <g, l> - g_v` (with `g = dV*/dl`) and replays
proportional to positive regret:

```text
l_v <- [R_v]^+ / sum_u [R_u]^+     (uniform if all R_u <= 0).
```

External regret vanishes at `O(1/sqrt(T))`, so the **average** allocation
converges to the min–max solution. A control with strongly negative gradient
(high deterrence) earns positive regret and attracts budget.

```bash
# default network, rho = 0.9
python3 regret_matching.py

# custom SP network + cross-check the two libraries agree on V*
python3 regret_matching.py --rho 0.9 --iterations 2000 --verify \
  --tree "Parallel(Series(Control('a',[0.5,0.4]), Parallel(Control('b',0.6), Control('c',0.55))), Series(Control('d',0.5), Control('e',0.45)))"
```

Outputs the final time allocation `l_v`, the resulting `beta_v`, the attacker's
index table `alpha(v, k)` at the solution, and the **min–max value**.

## `projected_subgrad.py` — feasible set = any polyhedron

Generalises the feasible set to an arbitrary nonempty polyhedron

```text
P = { l in R^n : l >= 0,  A l + c >= d }.
```

Regret matching is intrinsically a *simplex* algorithm (its update **is** the
simplex projection), so for a general `P` it is replaced by **projected
subgradient descent** (Zinkevich online gradient descent) — the direct
convex-set generalisation, and robust to the nonsmoothness of `V*`:

```text
l_{t+1} = Proj_P( l_t - eta_t * g_t ),   eta_t = eta0 / sqrt(t+1),
```

reporting the averaged iterate. The only `P`-dependent operation is the
Euclidean **projection onto `P`**, done with **Dykstra's algorithm** (closed-form
projections onto each halfspace / hyperplane / the nonnegative orthant, corrected
to converge to the true projection onto their intersection).

```bash
# simplex (reproduces regret_matching.py)
python3 projected_subgrad.py --verify

# genuine polyhedron: floor every control at l_v >= 0.05, with sum_v l_v = 1
python3 projected_subgrad.py --min-share 0.05 --verify
```

Programmatic use with the raw `A l + c >= d` form:

```python
from projected_subgrad import Polyhedron, solve, default_network, control_names

spec  = default_network()
names = control_names(spec)                 # fixes the column order of A
P     = Polyhedron.from_A_c_d(names, A, c, d)   # or Polyhedron(names, ineq=..., eq=...)
result = solve(spec, P, rho=0.9, iterations=2000)
print(result.value, result.allocation)      # min-max value and optimal l_v
```

**Well-posedness.** More time always deters (`V*` is nonincreasing in each
`l_v`), so if `P` is unbounded upward the defender drives `l -> infinity` and
`V* -> 0`. Make `P` bounded in the descent directions — e.g. include a budget
`sum_v l_v = B` (or `<= B`).

## Network specification

Both solvers share a compact spec that mirrors the underlying libraries:

- `Control(name, probs)` — a leaf; `probs` is the success-probability ladder
  `(p(0), p(1), ...)` and the lockout is `len(probs)`. A bare float is shorthand
  for a lockout-1 control.
- `Series(a, b, ...)` — clear children in order (n-ary, right-folded to binary).
- `Parallel(a, b, ...)` — clear any one child, a race (n-ary). `Ser`/`Par` alias.

Pass a network on the command line via `--tree "<expr>"` / `--tree-file <path>`,
or in code with `parse_tree(text)` / the constructors directly.

## Output

Each solver prints, at the converged allocation:

- `V* at final allocation` — the **min–max value** `min_{l in K} max_pi u(l, pi)`;
- the time allocation `l_v` and induced discount `beta_v = exp(-rho * l_v)`;
- the attacker's Gittins index table `alpha(v, k)`.

The `--verify` flag additionally checks that the attacker library
(`sp_attacker.game_value`) and the defender library (`sp_gradients2`) agree on
`V*` at the solution (they match to machine precision).
