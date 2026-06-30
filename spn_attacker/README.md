# SP layered-defence — the optimal attacker

Attacker-side code for the series–parallel (SP) layered-defence model: the
optimal adaptive attacker on an SP network, the baseline policies it is compared
against, and the experiment harnesses + tests that measure and certify it.
Everything is plain Python 3 (standard library only — no numpy/scipy). The one
exception is the pytest suite, which needs `pytest`.

## The model

A network is built from single **controls** by two constructors:

- `Series(G1, G2)` — clear `G1`, then clear `G2` (sequential).
- `Par(G1, G2)` — clear `G1` **or** `G2`; a race, either branch wins.

Attacking a control takes one attempt. With the control's per-attempt discount
`beta`, the attempt succeeds with probability `p(k)`, where `k` is the number of
prior failures on that control. Success advances (and prunes the losing
OR-sibling); failure increments `k`, and after `q = len(ps)` failures the control
**jams** permanently (stranding everything series-downstream of it). Reaching the
target pays reward 1, discounted by the product of the `beta`s of all attempts
made; if every route dies, the reward is 0.

## The optimal policy (`sp_attacker.py`)

The optimal attacker is computed in three phases, all in **O(Q²)** time (Q = total
number of control states), sidestepping the exponential joint-state MDP:

1. **Value fold** (`value_profile` / `game_value`) — bottom-up over the parse
   tree. Each block's *calibrated value* `Psi_G(gamma)` (its value when the
   attacker may instead retire for a one-off buyout `gamma`) is convex and
   piecewise-linear; the `PiecewiseLinear` class carries the Series, Par, affine,
   and max rules exactly. `game_value(net) = Psi_net(0)`.
2. **Index threading** (`index_table`) — each control is solved against the value
   profile of its *forward cone* (the prize unlocked by clearing it), yielding the
   coupled Gittins index `alpha(v, k)` of every control state.
3. **Online execution** (`IndexAttacker`) — precompute the indices once, then play
   greedily: always attack the live control of highest index (Gittins' theorem).
   Supports `action(state)`, `simulate(rng)`, and `monte_carlo_value(n)`.

Run `python3 sp_attacker.py` for a worked three-phase demo on `Series(Par(A,B), C)`.

## Files

| file | what it is | run |
|---|---|---|
| `sp_attacker.py` | **Core library.** Network datatypes (`Control`/`Series`/`Par`), the `PiecewiseLinear` calibrated-value algebra, and the three phases above (`game_value`, `index_table`, `IndexAttacker`). | `python3 sp_attacker.py` (demo) |
| `other_policies.py` | **Baseline policies** for comparison, all exposing `action(state)`, plus a generic `simulate` driver and the `baseline_policies(net)` factory. | imported |
| `experiment_attacker.py` | **Single-network experiment.** Evaluates the index policy + every baseline on one (editable) network; reports Monte-Carlo metrics, an exact noise-free value for deterministic policies, and the suboptimality gap; writes a CSV. | `python3 experiment_attacker.py` |
| `runner.py` | **Sweep harness.** Generates many random SP networks, evaluates every policy on each, and logs one row per network×policy to a CSV. Fully reproducible from `--seed`. | `python3 runner.py --num-networks 50 ...` |
| `attacker_pytest.py` | **Pytest suite (49 tests).** Asserts value & policy optimality against an independent brute-force MDP oracle, plus structural invariants. | `pytest attacker_pytest.py -q` |
| `attacker_metrics.csv` | Sample output of `experiment_attacker.py` (one row per policy). | — |

`other_policies.py`, `experiment_attacker.py`, `runner.py`, and the pytest suite
all import `sp_attacker`, so keep them in the same directory.

## Baseline policies (`other_policies.py`)

Every policy exposes `action(state) -> Optional[str]`, so the single `simulate`
driver can roll any of them out. They are deliberately spread from "ignores almost
nothing" to "ignores almost everything":

| policy | rule |
|---|---|
| `greedy_probability` | attack the highest one-step success prob `p(k)` |
| `greedy_discounted` | attack the highest discounted success `beta·p(k)` |
| `uncoupled_index` | attack the highest *single-control* index `c_hat(v,k)` — the index policy **minus** the forward-cone coupling |
| `single_route` | commit to the single best route; never pivot across routes |
| `fixed_priority` | structure-blind, fixed lexicographic priority |
| `random_frontier` | attack a uniformly random live control (stochastic) |

`uncoupled_index` is the most informative baseline: comparing it with the optimal
index policy isolates exactly the value of accounting for the forward cone, since
the two differ only in that coupling term.

## The verification logic

The trust anchor in `attacker_pytest.py` is an **independent brute-force MDP
oracle** (`_mdp_optimal_value`) that solves the game by exhaustive backward
induction over the full joint state space — a computation that shares no logic
with the piecewise-linear fold. Two headline checks ride on it:

- **Value.** Phase-1 fold value == MDP optimum.
- **Optimality.** The Phase-3 index policy's value == MDP optimum (it is optimal).

plus structural invariants (index monotonicity, the series discount, the OR gain),
the single-control closed form, win/jam reward semantics, frontier behaviour, and
Monte-Carlo consistency. The MDP is exponential in the number of controls, so it
is a reference oracle for small instances only.

## Experiments

### Single network — `experiment_attacker.py`

Edit `build_network()` to choose the instance, then:

```bash
python3 experiment_attacker.py                          # defaults
python3 experiment_attacker.py --samples 100000 --seed 1 --out results.csv
```

For each policy it reports the Monte-Carlo `expected_reward` (with 95% CI),
`win_probability`, attempt counts, and the headline `relative_loss` against the
exact optimum `V*`. For deterministic policies it also computes an exact,
noise-free value by backward induction (falling back to Monte-Carlo when the state
space is too large). The optimal index policy should show ~0 gap.

### Random sweep — `runner.py`

```bash
python3 runner.py --num-networks 50 --min-controls 3 --max-controls 8 \
    --samples 20000 --seed 0 --out sweep.csv
```

Generates random SP networks (controls obey Assumption 2: monotone non-increasing
success probabilities), evaluates every policy on each, and writes one row per
network×policy. The whole sweep — structures, parameters, and Monte-Carlo
streams — is reproducible from `--seed` alone via a single seeded master RNG. A
per-policy aggregate (mean relative loss, mean win probability) is printed at the
end.
