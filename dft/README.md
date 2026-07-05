# DFT → SP-network parser

The translation bridge between the **FFORT** dynamic-fault-tree benchmarks
(written in extended **Galileo** notation) and the Stackelberg time-allocation
solvers in [`sp_network_main/`](../sp_network_main). `attack_tree_parser.py`
reads a Galileo attack/fault tree, aggressively validates that it fits the
solver's structural assumptions, synthesises the missing adversarial parameters,
and emits a series–parallel (SP) network spec.

Plain Python 3, standard library only (`shlex`, `math`).

> This README covers `attack_tree_parser.py`. `attack_fault_tree_parser.py` is a
> separate variant and is not documented here.

## The setup it feeds

The solvers in `sp_network_main/` play a min–max (Stackelberg) game on an SP
network of security **controls**: the defender splits a unit time budget as an
allocation `l_v` (setting each control's per-attempt discount `beta_v =
exp(-rho*l_v)`), and the optimal attacker best-responds via a Gittins index
policy. Solving it exactly in `O(Q^2)` relies on two hard structural invariants:

- the network is **series–parallel** — built only from `Series` (clear children
  in order) and `Parallel` (clear any one child, a race); and
- it is a **vertex-disjoint tree** — no component is shared across branches (no
  DAGs), as the index policy requires independent forward cones.

FFORT models are richer than this (dynamic gates, shared subtrees, continuous
failure rates instead of adversarial success probabilities), so the parser's job
is to *filter out* what the solver cannot handle and *synthesise* what FFORT
does not provide.

## Galileo input

A tree is a list of `;`-terminated statements. One line names the root; internal
nodes declare a gate over children; leaves declare their parameters:

```text
toplevel "System";
"System"    or   "Subnet_A" "Subnet_B";     # gate: name  gate_type  children...
"Subnet_A"  and  "Firewall_A" "Server_A";
"Subnet_B"  seq  "WAF_B" "Server_B";
"Firewall_A" prob=0.15;                      # leaf: discrete one-shot event
"Server_A"   lambda=0.08;                    # leaf: continuous failure rate
"WAF_B"      prob=0.20;
"Server_B"   lambda=0.05;
```

Names may be quoted (parsed with `shlex`, so embedded spaces are preserved).

## How the SP network is built (pipeline)

**Phase 1 — Lexing & ingestion.** Split on `;`, `shlex`-tokenise each line, and
sort statements into two tables: `gates` (`name -> (gate_type, [children])`) and
`leaves` (`name -> {attr: value}`). A stray space after `=` (e.g. `prob= 8.61e-5`)
is repaired in place.

**Phase 2 — Read-once reduction (the gatekeeper).** The solver needs a
vertex-disjoint SP *tree*, i.e. a **read-once** function (each basic event used
once). A shared event makes the input a DAG, but the DAG may still *denote* a
read-once function, so instead of blanket-rejecting every shared node we decide
representability semantically:

- **Safelist.** Only `{or, and, seq, pand}` are allowed. Any spare gate (`csp`),
  functional dependency (`fdep`), maintenance/inspection loop (`inspT`), etc. is
  rejected — this keeps restless-bandit / infinite-horizon dynamics out.
- **Reduce with sound Boolean identities.** *Flatten* associative same-type gates
  (`OR(a, OR(b,c)) → OR(a,b,c)`, and `AND`), *dedup* identical children
  (idempotency `OR(a,a)=a`, `AND(a,a)=a`, and identical shared subtrees), and
  *collapse* unary gates.
- **Accept iff read-once.** After reduction, every gate's children must have
  **disjoint variable sets**. Disjoint ⟹ the function is read-once (SP-
  representable). A residual overlap is a genuine coupling — a 2-of-3 vote / a
  Wheatstone bridge — that no SP tree can express, and is rejected. This runs on
  leaf-*sets* with memoisation, so it is polynomial: a shared DAG is **never**
  expanded into an exponential tree.

  So an all-`or` tree that merely repeats an event (e.g. `T10 Chopper`, where
  `X6` sits under three `or`-gates) collapses to one `Parallel` and is accepted,
  while a true bridge is rejected. *Not handled* (conservatively rejected):
  distributive factoring of a shared common `AND`-factor across `OR` branches,
  `OR(AND(c,a), AND(c,b)) = AND(c, OR(a,b))`.

**Phase 3 — Build & synthesise.** Recurse from `toplevel` down:

- **Gate translation.** `or` → **`Parallel`**; `and` / `seq` / `pand` →
  **`Series`**, preserving the left-to-right child order. (A unary gate collapses
  to its single child.)
- **Parameter synthesis.** FFORT has no notion of an attacker, so each leaf's
  success-probability ladder `p(0), p(1), …` (and its lockout `q = len(p)`) is
  derived from the reliability attributes:

  | Leaf attribute | Meaning | Ladder produced |
  |---|---|---|
  | `prob=P` | discrete one-shot event | `[P]` (lockout `q = 1`) |
  | `lambda=R` | continuous rate | geometric back-off `p(k) = (1 − e^{−R·Δ})·δ^k`, length `lockout_limit` |
  | `phases=N`, `lambda=R` | Erlang / phased resilient control | flat `p(k) = 1 − e^{−R·Δ}`, length `2N` |
  | *(none)* | unparametrised | `[0.5]` |

  Here `Δ = time_block` and `δ = decay_factor`. The geometric decay models an
  IDS/back-off response and keeps `p(k)` **monotone non-increasing**, which the
  index policy assumes.

### A note on time

In `sp_network_main` the per-control time `l_v` is the **defender's decision
variable** — the solver assigns it. It is *not* a parser input. A `Control`
therefore carries only its probability ladder (`lockout = len`). The parser's
`time_block` is merely a *nominal* reference time used to convert a continuous
rate `lambda` into a base success probability; do not confuse it with `l_v`.

## Output

Both outputs are valid input to `sp_network_main/`:

- **`to_expression(text) -> str`** — a DSL string in the solver's own grammar,
  e.g.

  ```text
  Parallel(Series(Control('Firewall_A', [0.15]), Control('Server_A', [0.07688, 0.06151, 0.04921, 0.03936])),
           Series(Control('WAF_B', [0.2]), Control('Server_B', [0.04877, 0.03902, 0.03121, 0.02497])))
  ```

  Paste it into the solvers' `--tree` flag, or feed it to `parse_tree(...)`.
  Standalone — no solver import required.

- **`to_spec(text)`** — the live `Control` / `Series` / `Parallel` spec object,
  ready to hand straight to `solve(...)` / `regret_matching(...)`. It builds the
  namedtuple objects directly (n-ary gates right-folded into the solver's binary
  nodes), lazily importing the constructors from `sp_network_main/`.

## Usage

```bash
python3 attack_tree_parser.py       # parses the built-in example, prints both forms
```

```python
from attack_tree_parser import SPNetworkParser

parser = SPNetworkParser(lockout_limit=4, time_block=1.0, decay_factor=0.8)

# (1) as a string for the CLI / parse_tree
expr = parser.to_expression(ffort_text)

# (2) as a live spec, solved directly
spec = parser.to_spec(ffort_text)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sp_network_main"))
from regret_matching import regret_matching, print_result
result = regret_matching(spec, rho=0.9, iterations=2000)
print_result(spec, result, 0.9)
```

Networks that violate the SP / tree invariants raise `ValueError` — catch it to
skip a benchmark:

```python
try:
    spec = parser.to_spec(ffort_text)
except ValueError as e:
    print(f"Benchmark skipped: {e}")
```

## Parser knobs (`SPNetworkParser(...)`)

| Argument | Default | Effect |
|---|---|---|
| `lockout_limit` | `5` | lockout `q_v` for continuous (`lambda`) controls |
| `time_block` | `1.0` | nominal reference time Δ for the `lambda → p0` conversion |
| `decay_factor` | `0.8` | geometric decay `δ` of `p(k)` across attempts |
