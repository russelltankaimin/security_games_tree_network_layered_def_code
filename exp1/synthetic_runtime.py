"""
synthetic_runtime.py
====================

Runtime comparison on synthetic series-parallel (SP) networks:

    (a) computing the Gittins index table  -- `sp_attacker.index_table`, the
        O(Q^2) fold that yields the optimal index policy; versus
    (b) computing the optimal policy by BRUTE-FORCE MDP -- backward induction
        over the full joint state space (exponential).

The point is to show that (b) is intractable: for realistic SP networks it
blows past a 15-minute wall-clock budget (recorded as -1), while (a) finishes in
milliseconds.

Running the two SEPARATELY, on the SAME instances
-------------------------------------------------
Every instance (topology, per-control lockouts q_v, success probabilities
p_v(k), and allocation ell) is generated deterministically from
(--seed, network_id, --mode, --rho, --p-min, --p-max). So

    python3 synthetic_runtime.py --task index --out runtime_index.csv
    python3 synthetic_runtime.py --task mdp   --out runtime_mdp.csv

operate on byte-identical networks, and each CSV logs the `ell` and `probs`
actually used so you can confirm they match (join on network_id). `--task both`
runs them together.

Input CSV
---------
Required columns: `network_id` and `structure` (binary "Ser(A,B)" / "Par(A,B)"
with bare-name leaves, e.g. "Ser(Ser(Par(v0,v1),v2),...)").

Optional columns:
    total_Q, n_controls, depth  -- carried through / used to size synthesis.
    probs  -- JSON {name: [p(0), p(1), ...]} giving each control's exact success
              ladder (lockout q_v = len). If present, probabilities and lockouts
              are taken from here instead of being synthesised.
    ell    -- JSON {name: l_v} giving the exact allocation (normalised to the
              simplex). If present, it overrides the --mode ell choice.

Both `probs` and `ell` use the same JSON format this script WRITES, so an output
CSV can be fed straight back in as input (round-trip on the same instances).

When `probs`/`ell` are absent they are synthesised reproducibly: q_v is
distributed to match the recorded `total_Q`, p_v(k) is a monotone non-increasing
ladder, and ell is chosen per --mode. (Runtime depends on the topology and the
q_v -- the state-space size -- not on the probability values.)

Choosing ell (sets beta_v = exp(-rho * l_v)), both on the simplex (sum l_v = 1):
    * random     -- l sampled uniformly from the simplex.
    * predefined -- l fixed to uniform 1/n (or an explicit vector via --ell).

Output
------
One CSV row per network with the index-table and MDP runtimes in seconds; the MDP
runtime is -1 when it fails to finish within the timeout (or trips the
state-count safety cap). `mdp_status` records ok / timeout / statecap / oom /
error, and `ell` / `probs` record the exact instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from math import exp

# --- make the sibling attacker package importable --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ATTACKER = os.path.join(_ROOT, "spn_attacker")
if _ATTACKER not in sys.path:
    sys.path.insert(0, _ATTACKER)

from sp_attacker import (  # noqa: E402
    Control,
    Par,
    Series,
    Status,
    apply_outcome,
    block_status,
    controls,
    frontier,
    index_table,
    initial_state,
)


# ===========================================================================
# Structure parsing:  "Ser(A,B)" / "Par(A,B)" / "vK"  ->  sp_attacker Network
# ===========================================================================


def parse_structure(text, make_control):
    """Parse a `structure` string into an sp_attacker network.

    `make_control(name)` supplies the Control (with its synthesised beta / ps)
    for each leaf. Ser/Par nodes are n-ary (>= 2 children), matching both the
    binary CSVs and the wide/nested n-ary CSVs (Series/Par are n-ary-capable).
    """
    pos = 0

    def parse():
        nonlocal pos
        head = text[pos:pos + 4]
        if head in ("Ser(", "Par("):
            op = head[:3]
            pos += 4
            kids = [parse()]
            while text[pos] == ",":               # collect all children (n-ary)
                pos += 1
                kids.append(parse())
            assert text[pos] == ")", f"expected ')' at {pos}"
            pos += 1
            return Series(*kids) if op == "Ser" else Par(*kids)
        start = pos
        while pos < len(text) and text[pos] not in ",)":
            pos += 1
        return make_control(text[start:pos])

    tree = parse()
    assert pos == len(text), f"trailing input at {pos}: {text[pos:]!r}"
    return tree


def leaf_names(text):
    """Control names in left-to-right order, via the structure parser."""
    names = []
    parse_structure(text, lambda name: names.append(name) or name)
    return names


# ===========================================================================
# Reproducible synthesis of the missing per-control parameters and of ell
# ===========================================================================


def synth_lockouts(names, total_q, rng):
    """Assign q_v >= 1 to each control, summing to `total_q` (matching the CSV)."""
    q = {name: 1 for name in names}
    for _ in range(max(0, total_q - len(names))):
        q[rng.choice(names)] += 1
    return q


def synth_success_probs(q_v, rng, p_min, p_max):
    """A monotone non-increasing success-probability ladder of length q_v."""
    vals = sorted((rng.uniform(p_min, p_max) for _ in range(q_v)), reverse=True)
    return tuple(round(v, 4) for v in vals)


def sample_ell(names, mode, rng, predefined=None):
    """Return {name: l_v} on the simplex (sum = 1)."""
    n = len(names)
    if mode == "random":
        raw = [rng.expovariate(1.0) for _ in names]   # uniform on the simplex
        total = sum(raw)
        return {name: raw[i] / total for i, name in enumerate(names)}
    # predefined
    if predefined is not None:
        if len(predefined) != n:
            raise ValueError(
                f"--ell has {len(predefined)} entries but network has {n} controls")
        total = sum(predefined)
        return {name: predefined[i] / total for i, name in enumerate(names)}
    return {name: 1.0 / n for name in names}          # uniform 1/n


def build_network(structure, ell, rho, ps_map):
    """Assemble the sp_attacker network with beta_v = exp(-rho * l_v)."""
    def make_control(name):
        return Control(name, exp(-rho * ell[name]), ps_map[name])

    return parse_structure(structure, make_control)


def network_depth(net):
    """Nesting depth of the parse tree (a single control has depth 0)."""
    if isinstance(net, Control):
        return 0
    return 1 + max(network_depth(net.left), network_depth(net.right))


def make_instance(info, args, predefined):
    """Build one test instance from a CSV network row.

    `probs` / `ell` columns (JSON) are used verbatim when present; otherwise they
    are synthesised deterministically from (seed, network_id, mode, rho, p_min,
    p_max), so the instance is reproducible across separate --task runs. Returns
    (net, names, ps_map, ell).
    """
    rng = random.Random(f"{args.seed}:{info['network_id']}")
    names = leaf_names(info["structure"])

    # Success ladders p_v(k) (and hence lockouts q_v): from `probs`, else synth.
    if "probs" in info:
        raw = json.loads(info["probs"])
        missing = [n for n in names if n not in raw]
        if missing:
            raise ValueError(f"network {info['network_id']}: probs missing {missing}")
        ps_map = {n: tuple(float(p) for p in raw[n]) for n in names}
    else:
        q_map = synth_lockouts(names, info.get("total_Q", len(names)), rng)
        ps_map = {n: synth_success_probs(q_map[n], rng, args.p_min, args.p_max)
                  for n in names}

    # Allocation ell: from `ell` column (normalised to the simplex), else --mode.
    if "ell" in info:
        raw = json.loads(info["ell"])
        missing = [n for n in names if n not in raw]
        if missing:
            raise ValueError(f"network {info['network_id']}: ell missing {missing}")
        vals = {n: float(raw[n]) for n in names}
        total = sum(vals.values())
        ell = {n: vals[n] / total for n in names} if total > 0 else vals
    else:
        ell = sample_ell(names, args.mode, rng, predefined)

    net = build_network(info["structure"], ell, args.rho, ps_map)
    return net, names, ps_map, ell


# ===========================================================================
# Index-table timing (the tractable O(Q^2) computation)
# ===========================================================================


def time_index(net):
    start = time.perf_counter()
    index_table(net)
    return time.perf_counter() - start


# ===========================================================================
# Brute-force MDP: optimal policy by backward induction over joint states
# ===========================================================================


class _StateCapExceeded(Exception):
    pass


class _Timeout(Exception):
    pass


def brute_force_policy(net, max_states):
    """Optimal attacker policy via exhaustive backward induction.

    Enumerates every reachable joint state (exponential) and records the optimal
    action at each. Raises `_StateCapExceeded` once the number of distinct states
    exceeds `max_states` (a memory safeguard; 0 disables the cap). A pending
    SIGALRM interrupts the recursion for the wall-clock timeout.
    """
    cmap = {c.name: c for c in controls(net)}
    value_cache = {}
    policy = {}

    def encode(state):
        return tuple(sorted(
            (n, v.name if isinstance(v, Status) else v) for n, v in state.items()))

    def value(state):
        status = block_status(net, state)
        if status is Status.WON:
            return 1.0
        if status is Status.DEAD:
            return 0.0
        key = encode(state)
        cached = value_cache.get(key)
        if cached is not None:
            return cached
        if max_states and len(value_cache) >= max_states:
            raise _StateCapExceeded
        best, best_action = 0.0, None
        for name, k in frontier(net, state):
            control = cmap[name]
            p = control.ps[k]
            v = control.beta * (
                p * value(apply_outcome(state, control, True))
                + (1.0 - p) * value(apply_outcome(state, control, False)))
            if v > best:
                best, best_action = v, name
        value_cache[key] = best
        policy[key] = best_action
        return best

    value(initial_state(net))
    return len(value_cache)


def time_mdp(net, timeout, max_states):
    """Solve the MDP in-process, aborting at `timeout` seconds via SIGALRM.

    Returns (seconds, status, n_states). seconds is -1 on any non-completion
    (timeout / statecap / oom) so the CSV always carries a number.
    """
    use_alarm = timeout and hasattr(signal, "SIGALRM")
    if use_alarm:
        def _handler(signum, frame):
            raise _Timeout
        previous = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)

    sys.setrecursionlimit(1_000_000)
    start = time.perf_counter()
    try:
        n_states = brute_force_policy(net, max_states)
        return time.perf_counter() - start, "ok", n_states
    except _Timeout:
        return -1.0, "timeout", None
    except _StateCapExceeded:
        return -1.0, "statecap", None
    except MemoryError:
        return -1.0, "oom", None
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


# ===========================================================================
# Experiment driver
# ===========================================================================


def load_networks(csv_path):
    """One entry per network. Requires `network_id` and `structure`; picks up
    optional `total_Q`/`n_controls`/`depth` (ints) and `probs`/`ell` (JSON) when
    present."""
    csv.field_size_limit(10 ** 9)      # probs/ell can be large JSON blobs
    seen = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            nid = row["network_id"]
            if nid in seen:
                continue
            entry = {"network_id": int(nid), "structure": row["structure"]}
            for col in ("n_controls", "total_Q", "depth"):
                if row.get(col) not in (None, ""):
                    entry[col] = int(row[col])
            for col in ("probs", "ell"):
                if row.get(col):
                    entry[col] = row[col]           # raw JSON string
            seen[nid] = entry
    return sorted(seen.values(), key=lambda r: r["network_id"])


FIELDNAMES = [
    "network_id", "structure", "n_controls", "total_Q", "depth", "mode", "rho",
    "seed", "task", "index_time_s", "mdp_time_s", "mdp_status", "mdp_states",
    "ell", "probs",
]


def run(args):
    predefined = [float(x) for x in args.ell.split(",")] if args.ell else None

    networks = load_networks(args.csv)
    if args.limit:
        networks = networks[: args.limit]

    print(f"{len(networks)} networks | task={args.task} mode={args.mode} "
          f"rho={args.rho} timeout={args.timeout}s max_states={args.max_states}\n")
    print(f"{'id':>3} {'n':>3} {'Q':>4} {'depth':>5}  "
          f"{'index(s)':>10}  {'mdp(s)':>10}  status")

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for info in networks:
            net, names, ps_map, ell = make_instance(info, args, predefined)
            # Derive descriptors from the actual instance (authoritative when
            # probs/ell come from the CSV).
            n_controls = len(names)
            total_q = sum(len(ps_map[n]) for n in names)
            depth = info.get("depth", network_depth(net))

            index_time = ""
            if args.task in ("index", "both"):
                index_time = time_index(net)

            mdp_time, mdp_status, mdp_states = "", "", ""
            if args.task in ("mdp", "both"):
                seconds, mdp_status, states = time_mdp(net, args.timeout, args.max_states)
                mdp_time = f"{seconds:.6f}" if seconds >= 0 else "-1"
                mdp_states = states if states is not None else ""

            writer.writerow({
                "network_id": info["network_id"],
                "structure": info["structure"],
                "n_controls": n_controls,
                "total_Q": total_q,
                "depth": depth,
                "mode": args.mode,
                "rho": args.rho,
                "seed": args.seed,
                "task": args.task,
                "index_time_s": f"{index_time:.6f}" if index_time != "" else "",
                "mdp_time_s": mdp_time,
                "mdp_status": mdp_status,
                "mdp_states": mdp_states,
                "ell": json.dumps({n: round(ell[n], 6) for n in names}),
                "probs": json.dumps({n: list(ps_map[n]) for n in names}),
            })
            handle.flush()                       # persist partial progress

            it = f"{index_time:>10.4f}" if index_time != "" else f"{'-':>10}"
            mt = (f"{mdp_time:>10}" if mdp_time == "" else
                  f"{float(mdp_time):>10.2f}")
            print(f"{info['network_id']:>3} {n_controls:>3} "
                  f"{total_q:>4} {depth:>5}  {it}  {mt}  {mdp_status}")

    print(f"\nwrote {args.out}")


def build_arg_parser():
    default_csv = os.path.join(_ATTACKER, "sweep_metrics_large.csv")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=default_csv, help="input CSV of network structures")
    p.add_argument("--out", default=os.path.join(_HERE, "runtime.csv"), help="output CSV")
    p.add_argument("--task", choices=["index", "mdp", "both"], default="both",
                   help="which computation(s) to run (instances are identical across tasks)")
    p.add_argument("--mode", choices=["random", "predefined"], default="random",
                   help="how to choose ell on the simplex")
    p.add_argument("--ell", default=None,
                   help="predefined ell as comma-separated values (with --mode predefined; "
                        "must match n_controls); default is uniform 1/n")
    p.add_argument("--rho", type=float, default=0.9, help="discount rate rho (> 0)")
    p.add_argument("--timeout", type=float, default=900.0, help="MDP wall-clock budget (s)")
    p.add_argument("--max-states", type=int, default=5_000_000,
                   help="MDP state-count safety cap (0 disables); exceeding it records -1")
    p.add_argument("--seed", type=int, default=0, help="master seed for synthesis")
    p.add_argument("--p-min", type=float, default=0.2, help="min synthesised success prob")
    p.add_argument("--p-max", type=float, default=0.9, help="max synthesised success prob")
    p.add_argument("--limit", type=int, default=0, help="only the first N networks (0 = all)")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
