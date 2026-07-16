# exp2 — Defender optimisation: exact vs stochastic gradients

Regret-matching (RM) optimisation of the **defender allocation** `l` on the
simplex `{l ≥ 0, Σ l = 1}`, where control `v` gets survival factor
`β_v = exp(-ρ · l_v)`. The defender minimises the **optimal (Gittins) attacker's
value** `V*(l)`, which is convex in `l`. Two gradient oracles drive the SAME RM
descent and are compared:

- **EXACT** — `spn_defender/sp_gradients2.value_and_gradient`: the deterministic
  n-ary fold giving `dV*/dl` exactly at the attacker's optimal response.
- **STOCHASTIC** — a Monte-Carlo estimate `dV*/dl_v = -ρ·E[n_v·R]` from `b`
  rollouts of the optimal attacker (`spn_attacker`), where `n_v` is the number of
  attempts on control `v` and `R` the realised discounted reward.

The central question: **how much more time / how many more iterations does the
stochastic method (at batch `b`) need to reach the quality the exact method
reaches**, and **how does that depend on network structure**.

Depends on the sibling packages `spn_attacker/` (optimal attacker + rollouts) and
`spn_defender/` (exact value/gradient fold).

---

## Scripts

### Main experiment
- **`gradient_rm.py`** — the core driver. Runs the exact arm for a fixed
  `--exact-iters` to define a target `V*`, then the stochastic arm (`--runs`
  independent runs at batch `--batch`) and reports the extra iterations / wall
  clock to reach that target. Timing is symmetric (only the essential per-iter
  work is timed; the exact-fold objective checks are untimed). Extra modes:
  - `--grad-var` — log the per-component **variance of the stochastic gradient**
    at every iteration (from an independent, untimed measurement batch
    `--var-batch`); stops at the target like a normal run. → `<out>_gradvar.csv`.
  - `--fixed-var` — sample the gradient variance at ONE fixed representative
    allocation reached by `--fixed-var-iters` exact-RM steps. → `<out>_fixedvar.csv`.
  - `--avg-power` — polynomially-weighted averaging of the stochastic iterate
    (0 = uniform, 1 = linear/recency).
  - Outputs: `<out>_summary.csv`, `<out>_curves.csv` (+ the two above).
- **`synthetic_runtime_gradients.py`** — a related convergence study of the same
  two oracles, terminating on a regret certificate (`max_v R_v / t < ε`) or an
  objective plateau, logging RM regret and the true optimality gap per iteration.

### Instance generation
- **`large_instance_generator.py`** — generate large series-parallel instances
  (`--topology random | parallel_chains`) with monotone success ladders, sized to
  a target `Q`. Output is directly readable by `gradient_rm.py`.

### Analysis & plotting
- **`plot_curves.py`** — plot `V*(l_t)` convergence (exact vs mean stochastic with
  a ±std band) from a `<tag>_curves.csv`.
- **`collect_runtimes.py`** — aggregate any `*_summary.csv` files into one tidy
  `runtimes.csv` (per network: exact vs stochastic runtime and iteration counts).
- **`estimate_nary_runtime.py`** — analytically rescale measured binary-fold
  runtimes to estimate the native n-ary fold runtime (structure-only cost ratio).

### Helper
- **`aaai_26.py`** — brute-force MDP solver and success-probability generators
  (exp-backoff / beta / constant); reference utilities.

---

## Test-case instances

Structured instance families (5 networks each) built to probe how **structure**
decides the exact-vs-stochastic winner. All share the columns
`network_id, structure, n_controls, total_Q, depth, probs` and load straight into
`gradient_rm.py`. (Other CSVs in this directory are experiment *results*.)

| file | structure | Q range | reward | gradient noise* | intended use |
|------|-----------|---------|--------|-----------------|--------------|
| `series_heavy.csv` | `Par` of a **few long** `Ser` chains (narrow, deep) | 1k–10k | rare (V*≈0.02–0.05) | high (1.4–3.4) | **exact-favouring** — b=1 needs many more iters |
| `highly_parallel.csv` | `Par` of **many short** chains (wide, shallow) | 1k–20k | frequent (V*≈0.86–0.96) | low (0.56–0.63) | **stochastic-favouring** contrast |
| `sp_par80ser20.csv` | ~80% parallel / 20% series, forced series-root spine | 11k–25k | moderate | wide, LONG trajectories (16–55 attempts) | wide sampling variation + long attacker rollouts |
| `h7_targets.csv` | **long-chain** series-heavy at large `n` (390–540) | 12k–19k | rare | max (~6) | maximises the stochastic-gradient penalty; **note:** the long chains make the *exact* fold very expensive — see caveat below |

\* relative gradient noise = `trace(Cov(ĝ)) / ‖E[ĝ]‖²` at the uniform allocation;
`> 1` means variance exceeds signal² (favours exact via more stochastic iterations).

**Caveat on `h7_targets.csv`:** these have the ideal *structure* for a large exact
win (rare reward, very high noise) but the long chains inflate the exact fold to
~15–20 s per gradient (O(Q²) reverse-mode blow-up), so a full `gradient_rm` run is
impractical on them as-is. See `spn_defender/gradient_redesign_spec.md` for the
diagnosis and the (research-level) fix.

---

## Quick start

```bash
# exact vs stochastic (batch 1) on the exact-favouring set, 1 run, coarse checks
python3 gradient_rm.py --csv series_heavy.csv --out sh_b1 --batch 1 \
    --exact-iters 2000 --exact-runs 1 --runs 5 --log-every 500 --check-every 100

# just network 2, batch 1, per-iteration gradient variance
python3 gradient_rm.py --csv series_heavy.csv --start-id 2 --end-id 2 --batch 1 \
    --grad-var --runs 1 --exact-runs 1 --var-batch 200 --out net2

# plot the convergence curves
python3 plot_curves.py sh_b1_curves.csv --nets 0,1,2 --logy

# generate a fresh large instance and run on it
python3 large_instance_generator.py --out big.csv --n-controls 2000 --total-Q 50000
python3 gradient_rm.py --csv big.csv --out big_rm --batch 1 --exact-iters 50
```

## Input format

`gradient_rm.py` needs, per network: `network_id`, `structure` (n-ary
`Ser(...)`/`Par(...)` over bare-name leaves), `n_controls`, `total_Q`, and `probs`
(JSON `{name: [p(0), p(1), …]}`, a monotone non-increasing ladder). Default input
is `../exp1/synth_nets_random_ell.csv`.

## Background

The exact-vs-stochastic wall-clock outcome factors into two levers — per-iteration
cost and iteration count (driven by rare-reward gradient variance) — which pull in
opposite directions with structure. The full analysis, including why a clean large
exact win is hard and what an O(Q·depth) gradient redesign would require, is in
`spn_defender/gradient_redesign_spec.md`.
