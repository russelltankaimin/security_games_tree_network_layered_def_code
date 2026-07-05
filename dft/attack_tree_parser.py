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
           A stray space after '=' (e.g. `prob= 8.61e-5`) is repaired in place.
Phase 2 -- Read-once reduction:   reject any gate outside the safelist
           {or, and, seq, pand}, then reduce the gate DAG with sound Boolean
           identities and accept iff the result is READ-ONCE (each basic event
           appears exactly once) -- the exact class representable as a
           vertex-disjoint SP tree.  See the reduction notes below.
Phase 3 -- Build & synthesise:   convert the reduced read-once tree, mapping `or`
           -> Parallel and `and`/`seq`/`pand` -> Series (preserving child order),
           and synthesising each leaf's success-probability ladder p(k).

Read-once reduction (why some shared-node DAGs are still accepted)
------------------------------------------------------------------
A shared basic event makes the FFORT input a DAG, not a tree -- but the DAG may
still denote a read-once function that IS SP-representable.  We decide this with
semantics-preserving rewrites plus a disjointness test:

    * flatten  associative same-type gates: OR(a, OR(b,c)) -> OR(a,b,c), and AND.
    * dedup    identical children (idempotency): OR(a,a) -> a, AND(a,a) -> a,
               and identical shared subtrees collapse to one.
    * collapse unary gates to their single child.
    * accept   iff every gate's (deduped) children have DISJOINT variable sets.

Disjoint children => the formula is read-once (representable). A residual overlap
means a genuine coupling (e.g. 2-of-3 voting / a Wheatstone bridge) that no SP
tree can express, so it is rejected. Everything runs on leaf-SETS with
memoisation, so it is polynomial -- a shared DAG is never expanded into an
exponential tree.

Not handled (conservatively rejected): distributive factoring of a shared common
AND-factor across OR branches, e.g. OR(AND(c,a), AND(c,b)) = AND(c, OR(a,b)).
Such inputs are read-once but are reported as coupled.

Output
------
`to_spec(text)` returns the live spec object (Control/Series/Parallel namedtuples)
ready to hand to `solve(...)` / `regret_matching(...)`.  `to_expression(text)`
returns the equivalent sp_network_main DSL string (for `--tree` / `parse_tree`).

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


class ReadOnceError(ValueError):
    """Raised when a network is not representable as a read-once SP tree."""


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
    # Phase 1: lex & ingest.
    # ------------------------------------------------------------------
    def _lex(self, text):
        """Return (toplevel, gates, leaves); reject non-safelist gates.

        gates  : Dict[name, (gate_type, [children])]
        leaves : Dict[name, {attr: value}]
        """
        lines = [line.strip() for line in text.split(';') if line.strip()]

        toplevel = None
        gates = {}
        leaves = {}

        for line in lines:
            parts = shlex.split(line)
            if not parts:
                continue
            # Repair a stray space after '=' (e.g. `prob= 8.61e-5`): glue a token
            # ending in '=' onto the following token.
            merged = []
            for tok in parts:
                if merged and merged[-1].endswith('='):
                    merged[-1] += tok
                else:
                    merged.append(tok)
            parts = merged

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

        for name, (g_type, _) in gates.items():
            if g_type not in self.safelist_gates:
                raise ReadOnceError(
                    f"Rejected: disallowed gate '{g_type}' at node '{name}' "
                    f"(safelist = {sorted(self.safelist_gates)}).")

        return toplevel, gates, leaves

    # ------------------------------------------------------------------
    # Phase 2: reduce the gate DAG and verify it is read-once.
    #
    # A reduced node is a tuple:  ('leaf', name) | ('or', kids) | ('and', kids),
    # where kids is a tuple of reduced nodes. These tuples are hashable, so they
    # double as canonical keys for dedup and leaf-set memoisation.
    # ------------------------------------------------------------------
    def _reduce(self, toplevel, gates, leaves):
        memo = {}                       # name -> reduced node
        leafset_cache = {}              # reduced node -> frozenset of leaf names

        def leaves_of(node):
            cached = leafset_cache.get(node)
            if cached is not None:
                return cached
            if node[0] == 'leaf':
                result = frozenset((node[1],))
            else:
                result = frozenset().union(*(leaves_of(c) for c in node[1]))
            leafset_cache[node] = result
            return result

        def reduce(name, stack):
            if name in memo:
                return memo[name]
            if name in stack:
                raise ReadOnceError(f"Rejected: cycle through node '{name}'.")
            if name in leaves:
                node = ('leaf', name)
                memo[name] = node
                return node
            if name not in gates:
                raise ValueError(f"Missing definition for node '{name}'.")

            g_type, children = gates[name]
            op = 'or' if g_type == 'or' else 'and'
            reduced = [reduce(c, stack | {name}) for c in children]

            # Flatten associative same-op children.
            flat = []
            for ch in reduced:
                if ch[0] == op:
                    flat.extend(ch[1])
                else:
                    flat.append(ch)

            # Dedup identical children (idempotency), preserving order.
            seen, uniq = set(), []
            for ch in flat:
                if ch not in seen:
                    seen.add(ch)
                    uniq.append(ch)

            if len(uniq) == 1:
                node = uniq[0]                       # unary gate -> its child
            else:
                # Read-once test: distinct children must not share any variable.
                accumulated = set()
                for ch in uniq:
                    ls = leaves_of(ch)
                    overlap = accumulated & ls
                    if overlap:
                        raise ReadOnceError(
                            f"Rejected: not read-once -- node '{name}' has "
                            f"branches sharing {sorted(overlap)} (a coupling that "
                            f"no series-parallel tree can represent).")
                    accumulated |= ls
                node = (op, tuple(uniq))

            memo[name] = node
            return node

        return reduce(toplevel, frozenset())

    # ------------------------------------------------------------------
    # Phase 3a: leaf parameter synthesis.
    # ------------------------------------------------------------------
    def _success_probs(self, attrs):
        """Synthesise the success-probability ladder p(0..q-1) for one leaf."""
        # Rule A: discrete probability -> single-shot control (q_v = 1).
        if 'prob' in attrs:
            return [round(attrs['prob'], 5)]
        # Rule B: Erlang / phased resilient control (flat profile, length 2N).
        if 'phases' in attrs and 'lambda' in attrs:
            q_v = int(attrs['phases']) * 2
            p_phase = 1.0 - math.exp(-attrs['lambda'] * self.time_block)
            return [round(p_phase, 5) for _ in range(q_v)]
        # Rule C: continuous rate -> multi-shot with geometric back-off.
        if 'lambda' in attrs:
            p0 = 1.0 - math.exp(-attrs['lambda'] * self.time_block)
            return [round(p0 * (self.decay_factor ** k), 5)
                    for k in range(self.lockout_limit)]
        # Fallback for an unparametrised basic event.
        return [0.5]

    # ------------------------------------------------------------------
    # Phase 3b/3c: convert the reduced read-once tree to the output form.
    # ------------------------------------------------------------------
    def to_expression(self, text):
        """Parse FFORT `text` and return an sp_network_main DSL expression string
        (the input accepted by `--tree` / `parse_tree`)."""
        toplevel, gates, leaves = self._lex(text)
        node = self._reduce(toplevel, gates, leaves)

        def emit(nd):
            if nd[0] == 'leaf':
                probs = self._success_probs(leaves[nd[1]])
                return f"Control({nd[1]!r}, {probs!r})"
            ctor = "Parallel" if nd[0] == 'or' else "Series"
            return f"{ctor}({', '.join(emit(c) for c in nd[1])})"

        return emit(node)

    def to_spec(self, text):
        """Parse FFORT `text` and return the native Control/Series/Parallel spec
        object consumed by the sp_network_main solvers.

        Accepts any input reducible to a read-once tree (see module docstring);
        raises ReadOnceError otherwise. n-ary gates are right-folded into the
        solver's binary Series/Parallel nodes.
        """
        Control, Series, Parallel = _import_constructors()
        toplevel, gates, leaves = self._lex(text)
        node = self._reduce(toplevel, gates, leaves)

        def build(nd):
            if nd[0] == 'leaf':
                return Control(nd[1], tuple(self._success_probs(leaves[nd[1]])))
            node_type = Parallel if nd[0] == 'or' else Series
            specs = [build(c) for c in nd[1]]
            folded = specs[-1]
            for child in reversed(specs[:-1]):
                folded = node_type(child, folded)
            return folded

        return build(node)


def _import_constructors():
    """Locate sibling sp_network_main/ and return (Control, Series, Parallel)."""
    here = os.path.dirname(os.path.abspath(__file__))
    sp_main = os.path.join(os.path.dirname(here), "sp_network_main")
    if sp_main not in sys.path:
        sys.path.insert(0, sp_main)
    from regret_matching import Control, Parallel, Series  # noqa: E402
    return Control, Series, Parallel


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
    "Server_A" lambda=0.08;
    "WAF_B" prob=0.20;
    "Server_B" lambda=0.05;
    """

    parser = SPNetworkParser(lockout_limit=4, time_block=1.0, decay_factor=0.8)

    try:
        spec = parser.to_spec(ffort_data)
        print("Parsed SP-network spec object:\n")
        print(spec)
        print("\nEquivalent sp_network_main DSL expression:\n")
        print(parser.to_expression(ffort_data))
    except ValueError as e:
        print(f"Benchmark skipped: {e}")
