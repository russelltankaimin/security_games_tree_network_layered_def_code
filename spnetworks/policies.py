"""
Policies for the online attack. Each returns a function
    policy(state, frontier_dict) -> control_name | None
where frontier_dict maps each attackable control to its current count k.

  optimal_index : attack arg max_v alpha(v,k_v)            (Theorem 20 -- optimal)
  pc_index      : parallel-chains relaxation (AAAI-26 applied by flattening the
                  network to its source-sink paths, treated as independent chains)
  myopic_raw    : rank by raw single-control index c_hat(v,k) (ignores downstream)
  greedy_success: rank by immediate success probability p_v(k)
  single_route  : commit to the best single path; no switching across routes
  round_robin   : structure-blind fixed priority over frontier controls
"""
from networks import net_info, controls, Ctrl, Ser
import sp_index

def _chat(beta, p):
    return beta * p / (1 - beta * (1 - p))

# ---------- the optimal policy ----------
def optimal_index(G):
    table = sp_index.index_table(G)
    def policy(state, fr):
        return max(fr, key=lambda n: (table[(n, fr[n])], n))
    return policy

# ---------- baselines ----------
def myopic_raw(G):
    info = net_info(G)
    def policy(state, fr):
        return max(fr, key=lambda n: (_chat(info[n][0], info[n][1][fr[n]]), n))
    return policy

def greedy_success(G):
    info = net_info(G)
    def policy(state, fr):
        return max(fr, key=lambda n: (info[n][1][fr[n]], n))
    return policy

def round_robin(G):
    def policy(state, fr):
        return min(fr)                      # fixed lexicographic priority
    return policy

def routes(G):
    if G[0] == 'ctrl': return [[G[1]]]
    if G[0] == 'par':  return routes(G[1]) + routes(G[2])
    return [a + b for a in routes(G[1]) for b in routes(G[2])]

def _chain_from(names, info):
    node = Ctrl(names[-1], *info[names[-1]])
    for n in reversed(names[:-1]):
        node = Ser(Ctrl(n, *info[n]), node)
    return node

def single_route(G):
    info = net_info(G); paths = routes(G)
    best = max(paths, key=lambda pth: sp_index.game_value(_chain_from(pth, info)))
    bestset = set(best)
    def policy(state, fr):
        cand = [n for n in fr if n in bestset]
        return max(cand, key=lambda n: best.index(n)) if cand else None  # deepest on the route
    return policy

def pc_index(G, max_paths=4000):
    """AAAI-26 relaxation: value each source-sink path as an independent chain,
    index its controls by the exact chain (Gittins) index, and take, for each
    control, the best index over the paths through it. Ignores the OR-sharing
    that the true alpha captures."""
    info = net_info(G); paths = routes(G)
    if len(paths) > max_paths: paths = paths[:max_paths]
    table = {}
    for pth in paths:
        for (n, k), a in sp_index.index_table(_chain_from(pth, info)).items():
            if (n, k) not in table or a > table[(n, k)]: table[(n, k)] = a
    def policy(state, fr):
        return max(fr, key=lambda n: (table.get((n, fr[n]), 0.0), n))
    return policy

ALL = {
    'optimal_index': optimal_index,
    'pc_index':      pc_index,
    'myopic_raw':    myopic_raw,
    'greedy_success':greedy_success,
    'single_route':  single_route,
    'round_robin':   round_robin,
}

if __name__ == "__main__":
    from networks import worked_example
    import mdp
    G = worked_example(); opt, _ = mdp.optimal_value(G)
    print(f"{'policy':16s} value     rel.loss")
    for name, ctor in ALL.items():
        v = mdp.policy_value(G, ctor(G))
        print(f"{name:16s} {v:.6f}  {(opt - v) / opt * 100:6.3f}%")
