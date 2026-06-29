# Experiment harness — SP layered-defence index policy (AAAI)

Reproducible experiments for the optimal adaptive-attacker index policy on
series–parallel networks. Pure Python + numpy + matplotlib (no other deps).

## Modules
- `networks.py`  — SP construction and random generators (random SP, parallel
  chains, nested non-PC, canonical instances).
- `sp_index.py`  — the exact O(Q^2) index algorithm (PWL profiles + index
  threading); `index_table(G)`, `game_value(G)`.
- `mdp.py`       — brute-force product-MDP ground truth: `optimal_value(G)` and
  `policy_value(G, policy)`.
- `policies.py`  — `optimal_index` plus baselines: `pc_index` (AAAI-26 parallel-
  chains relaxation), `myopic_raw`, `greedy_success`, `single_route`,
  `round_robin`.
- `plots.py`     — reads `results/*.csv`, writes `figures/*.pdf`.

## Experiments
1. `exp1_validation.py` — index policy attains the exact MDP optimum (gap ≈ 0).
2. `exp2_scalability.py` — runtime of index vs MDP (the MDP blows up; headline).
3. `exp3_baselines.py`  — relative value loss of each baseline vs optimal.
4. `exp4_nesting.py`    — pc_index gap grows with nesting depth (why SP is needed).
5. `exp5_sensitivity.py`— sweep over discount β and jam threshold q.

## Run
    python3 run_all.py            # quick (small ensembles)
    python3 run_all.py --full     # paper-scale ensembles
    python3 exp2_scalability.py --trials 12   # any single experiment

Each experiment writes a CSV to `results/`; `plots.py` turns them into figures.
All randomness is seeded, so runs are reproducible.

## Notes
- All values are for the true game (buyout γ=0, reward 1 at the target).
- `pc_index` equals the optimum on parallel-chains instances by construction;
  its loss appears only on nested topologies (exp4) — the generalisation claim.
- The MDP is exponential ( ∏(q_v+1) states ) and is capped in exp2.
