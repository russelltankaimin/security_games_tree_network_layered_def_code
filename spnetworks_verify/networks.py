"""
Series-parallel network construction and random generators.

A network is a nested tuple:
    ('ctrl', name, beta, ps)   ps = [p(0),...,p(q-1)], q = len(ps)
    ('ser', G1, G2)            clear G1 then G2
    ('par', G1, G2)            OR-race: clear either branch
This single representation is shared by the index algorithm and the MDP solver.
"""
import random

def Ctrl(name, beta, ps): return ('ctrl', name, beta, list(ps))
def Ser(a, b):            return ('ser', a, b)
def Par(a, b):            return ('par', a, b)

def controls(G):
    if G[0] == 'ctrl': return [G[1]]
    return controls(G[1]) + controls(G[2])

def count_controls(G):
    return 1 if G[0] == 'ctrl' else count_controls(G[1]) + count_controls(G[2])

def total_Q(G):
    info = net_info(G)
    return sum(len(ps) for _, ps in info.values())

def depth(G):
    if G[0] == 'ctrl': return 0
    return 1 + max(depth(G[1]), depth(G[2]))

def net_info(G, d=None):
    if d is None: d = {}
    if G[0] == 'ctrl': d[G[1]] = (G[2], G[3])
    else: net_info(G[1], d); net_info(G[2], d)
    return d

# ---------- random pieces ----------
def random_control(rng, name, beta=0.9, qmax=3, monotone=True):
    q = rng.randint(1, qmax)
    if monotone:                       # Assumption 2: p(0) >= p(1) >= ...
        ps = sorted((round(rng.uniform(0.15, 0.9), 4) for _ in range(q)), reverse=True)
    else:
        ps = [round(rng.uniform(0.15, 0.9), 4) for _ in range(q)]
    b = beta if beta is not None else round(rng.uniform(0.8, 0.95), 3)
    return Ctrl(name, b, ps)

def _counter():
    n = [0]
    def nxt():
        s = f"v{n[0]}"; n[0] += 1; return s
    return nxt

def random_network(rng, n_leaves, beta=0.9, qmax=3, monotone=True, p_par=0.5):
    """Random SP tree with exactly n_leaves controls."""
    nxt = _counter()
    def build(n):
        if n == 1:
            return random_control(rng, nxt(), beta, qmax, monotone)
        left = rng.randint(1, n - 1)
        op = Par if rng.random() < p_par else Ser
        return op(build(left), build(n - left))
    return build(n_leaves)

def parallel_chains(rng, n_chains, chain_len, beta=0.9, qmax=3, monotone=True):
    """N = Par of Ser-chains (the AAAI-26 topology)."""
    nxt = _counter()
    def chain(L):
        node = random_control(rng, nxt(), beta, qmax, monotone)
        for _ in range(L - 1):
            node = Ser(node, random_control(rng, nxt(), beta, qmax, monotone))
        return node
    chains = [chain(chain_len) for _ in range(n_chains)]
    node = chains[0]
    for c in chains[1:]:
        node = Par(node, c)
    return node

def nested_network(rng, levels, beta=0.9, qmax=2, monotone=True):
    """Forces Par-inside-Ser nesting of a given depth (NOT parallel-chains)."""
    nxt = _counter()
    def block(L):
        if L == 0:
            return random_control(rng, nxt(), beta, qmax, monotone)
        # Ser( Par(block, block), block ) : a parallel sub-goal followed by more work
        return Ser(Par(block(L - 1), block(L - 1)), random_control(rng, nxt(), beta, qmax, monotone))
    return block(levels)

# ---------- canonical instances ----------
def worked_example():
    A = Ctrl('A', 0.9, [0.5, 0.3]); B = Ctrl('B', 0.9, [0.8, 0.5]); C = Ctrl('C', 0.9, [0.6, 0.4])
    return Par(Ser(A, B), C)

def ser_par_c():
    A = Ctrl('A', 0.9, [0.6, 0.4]); B = Ctrl('B', 0.9, [0.7, 0.5]); C = Ctrl('C', 0.9, [0.5, 0.3])
    return Ser(Par(A, B), C)        # smallest non-parallel-chains topology
