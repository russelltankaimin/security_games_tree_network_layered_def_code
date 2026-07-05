"""
attack_tree_parser.py
=====================

Translate FFORT-style (extended Galileo) attack/fault trees into the
series-parallel network spec consumed by `sp_network_main/` (the Stackelberg
time-allocation solvers `regret_matching.py` and `projected_subgrad.py`).

The bridge is deliberately strict: the solvers rely on the structural invariants
of the O(Q^2) series-parallel fold and the Gittins index policy, so this parser
aggressively filters, validates, and synthesises.

Pipeline
--------
Phase 1 -- Lexing & ingestion:   split on ';', shlex-tokenise each line, and
           separate `gates` (internal logic nodes) from `leaves` (basic events).
Phase 2 -- Structural SP filter: reject any gate outside the safelist
           {or, and, seq, pand}, and reject DAGs (any node with in-degree > 1),
           enforcing the vertex-disjoint tree topology the solver requires.
Phase 3 -- Build & synthesise:   recurse from `toplevel` down, mapping `or` ->
           Parallel and `and`/`seq`/`pand` -> Series (preserving child order),
           and synthesising each leaf's success-probability ladder p(k).

Output
------
`to_expression(text)` returns a string in the sp_network_main DSL, e.g.

    Parallel(Series(Control('Firewall_A', [0.15]), Control('Server_A', [...])),
             Series(Control('WAF_B', [0.2]), Control('Server_B', [...])))

which is exactly what `--tree` / `parse_tree(...)` accept.  `to_spec(text)` goes
one step further and returns the live spec object (Control/Series/Parallel
namedtuples) ready to hand to `solve(...)` / `regret_matching(...)`.

Note on time: in sp_network_main the per-control time l_v is the DEFENDER's
decision variable (the solver assigns it), NOT an input.  A Control therefore
carries only its success-probability ladder (lockout = len).  `time_block` below
is a *nominal* reference time used solely to convert a continuous rate lambda
into a base success probability; it is not the solver's l_v.
"""

import math
import os
import shlex
import sys


class SPNetworkParser:
    def __init__(self, lockout_limit=5, time_block=1.0, decay_factor=0.8):
        """
        lockout_limit : q_v assigned to continuous (lambda) controls (multi-shot).
        time_block    : nominal reference time for the lambda -> p0 conversion
                        (NOT the solver's decision variable l_v).
        decay_factor  : geometric decay of success probability across attempts,
                        modelling an IDS/back-off response (keeps p(k) monotone
                        non-increasing, as the index policy assumes).
        """
        self.lockout_limit = lockout_limit
        self.time_block = time_block
        self.decay_factor = decay_factor

        # The exact topological safelist the solver's SP fold supports.
        self.safelist_gates = {'or', 'and', 'seq', 'pand'}

    # ------------------------------------------------------------------
    # Phase 1 + 2: lex, ingest, and validate structure.
    # ------------------------------------------------------------------
    def _parse_and_validate(self, text):
        """Return (toplevel, gates, leaves) after the structural SP filter.

        gates  : Dict[name, (gate_type, [children])]
        leaves : Dict[name, {attr: value}]
        """
        lines = [line.strip() for line in text.split(';') if line.strip()]

        toplevel = None
        gates = {}
        leaves = {}

        # --- PHASE 1: LEXING & INGESTION ---
        for line in lines:
            parts = shlex.split(line)
            if not parts:
                continue

            if parts[0] == 'toplevel':
                toplevel = parts[1]
            elif len(parts) >= 2 and all('=' in p for p in parts[1:]):
                # Basic event (leaf): "Name" lambda=R prob=P
                attrs = {}
                for p in parts[1:]:
                    key, val = p.split('=', 1)
                    attrs[key] = float(val)
                leaves[parts[0]] = attrs
            else:
                # Gate (internal node): "Name" gate "Child1" "Child2" ...
                gates[parts[0]] = (parts[1].lower(), parts[2:])

        if not toplevel:
            raise ValueError("Invalid FFORT: No 'toplevel' defined.")

        # --- PHASE 2: STRUCTURAL FILTER ---
        # 2a. Reject disallowed dynamic/cyclic gates.
        for name, (g_type, _) in gates.items():
            if g_type not in self.safelist_gates:
                raise ValueError(
                    f"Rejected: disallowed gate '{g_type}' at node '{name}' "
                    f"(safelist = {sorted(self.safelist_gates)}).")

        # 2b. Reject DAGs -- enforce strict vertex-disjoint tree topology.
        in_degree = {name: 0 for name in list(gates) + list(leaves)}
        for name, (_, children) in gates.items():
            for child in children:
                in_degree[child] = in_degree.get(child, 0) + 1
                if in_degree[child] > 1:
                    raise ValueError(
                        f"Rejected: DAG detected -- node '{child}' has multiple "
                        f"parent branches (in-degree > 1).")

        return toplevel, gates, leaves

    # ------------------------------------------------------------------
    # Phase 3a: leaf parameter synthesis.
    # ------------------------------------------------------------------
    def _success_probs(self, attrs):
        """Synthesise the success-probability ladder p(0..q-1) for one leaf."""
        # Rule A: discrete probability -> single-shot control (q_v = 1).
        if 'prob' in attrs:
            return [round(attrs['prob'], 5)]
        # Rule B: Erlang Distribution (Phased/Resilient Control)
        elif 'phases' in attrs and 'lambda' in attrs:
            N = int(attrs['phases'])
            # Give the attacker more attempts to grind through the phases
            q_v = N * 2
            
            # Calculate the baseline probability for a single phase
            p_phase = 1.0 - math.exp(-attrs['lambda'] * self.time_block)
            
            # Flat probability profile: the defense doesn't weaken easily
            p_array = [round(p_phase, 5) for _ in range(q_v)]
            return p_array

        # Rule C: continuous rate -> multi-shot with geometric back-off.
        elif 'lambda' in attrs:
            p0 = 1.0 - math.exp(-attrs['lambda'] * self.time_block)
            return [round(p0 * (self.decay_factor ** k), 5)
                    for k in range(self.lockout_limit)]

        # Fallback for an unparametrised basic event.
        return [0.5]

    # ------------------------------------------------------------------
    # Phase 3b: build the sp_network_main DSL expression.
    # ------------------------------------------------------------------
    def to_expression(self, text):
        """Parse FFORT `text` and return an sp_network_main DSL expression string
        (the input accepted by `--tree` / `parse_tree`)."""
        toplevel, gates, leaves = self._parse_and_validate(text)

        def build(node_name):
            if node_name in leaves:
                probs = self._success_probs(leaves[node_name])
                return f"Control({node_name!r}, {probs!r})"

            if node_name in gates:
                g_type, children = gates[node_name]
                child_exprs = [build(c) for c in children]
                if len(child_exprs) == 1:
                    return child_exprs[0]          # unary gate = pass-through
                ctor = "Parallel" if g_type == 'or' else "Series"
                return f"{ctor}({', '.join(child_exprs)})"

            raise ValueError(f"Missing definition for node '{node_name}'.")

        return build(toplevel)

    # ------------------------------------------------------------------
    # Phase 3c: build the live spec object (ready for solve()/regret_matching()).
    # ------------------------------------------------------------------
    def to_spec(self, text):
        """Parse FFORT `text` and return the native Control/Series/Parallel spec
        object consumed by the sp_network_main solvers.

        Lazily imports `parse_tree` from sp_network_main, so `to_expression` stays
        usable without the solver packages present.
        """
        expression = self.to_expression(text)
        parse_tree = _import_parse_tree()
        return parse_tree(expression)


def _import_parse_tree():
    """Locate sibling sp_network_main/ and return its `parse_tree`."""
    here = os.path.dirname(os.path.abspath(__file__))
    sp_main = os.path.join(os.path.dirname(here), "sp_network_main")
    if sp_main not in sys.path:
        sys.path.insert(0, sp_main)
    from regret_matching import parse_tree  # noqa: E402
    return parse_tree


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    ffort_data = """
    toplevel "System";
    "System" or "Subnet_A" "Subnet_B";
    "Subnet_A" and "Firewall_A" "Server_A";
    "Subnet_B" seq "WAF_B" "Server_B";
    "Firewall_A" prob=0.15;
    "Server_A" lambda=0.08 phases=3;
    "WAF_B" prob=0.20;
    "Server_B" lambda=0.05;
    """

    ffort_data = """
    toplevel "System";
    "System" or "Subnet_A" "Subnet_B";
    "Subnet_A" and "Firewall_A" "Server_A";
    "Subnet_B" seq "WAF_B" "Server_B";
    "Firewall_A" prob=0.15;
    "Server_A" lambda=0.08;
    "WAF_B" prob=0.20;
    "Server_B" lambda=0.05;
    """

    parser = SPNetworkParser(lockout_limit=4, time_block=1.0, decay_factor=0.8)

    try:
        expression = parser.to_expression(ffort_data)
        print("sp_network_main DSL expression (pass to --tree / parse_tree):\n")
        print(expression)

        # Also confirm it round-trips into a live spec object.
        spec = parser.to_spec(ffort_data)
        print("\nRound-tripped into a live spec object:\n")
        print(spec)
    except ValueError as e:
        print(f"Benchmark skipped: {e}")
