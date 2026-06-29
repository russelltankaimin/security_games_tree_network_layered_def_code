"""
Brute-force exact MDP for the true game (gamma=0, reward 1 at the sink).
Independent ground truth: backward induction over the DAG of reachable
joint states. Provides the optimal value and the value of any fixed policy.

State mirrors the network tree, annotating each control with its status:
    ('c', name, k)            active at failure count k
    ('c', name, 'W')          won (cleared)
    ('c', name, 'D')          dead (jammed)
    ('s', Lstate, Rstate)     series
    ('p', Lstate, Rstate)     parallel
'win prunes siblings' and 'jam strands downstream' are handled implicitly by
status()/frontier(), so no explicit surgery on the state is needed.
"""
import sys
from networks import net_info
sys.setrecursionlimit(1_000_000)

WON, DEAD, LIVE = 'WON', 'DEAD', 'LIVE'

def init_state(G):
    if G[0] == 'ctrl': return ('c', G[1], 0)
    if G[0] == 'ser':  return ('s', init_state(G[1]), init_state(G[2]))
    return ('p', init_state(G[1]), init_state(G[2]))

def status(s):
    t = s[0]
    if t == 'c':
        return WON if s[2] == 'W' else DEAD if s[2] == 'D' else LIVE
    L, R = status(s[1]), status(s[2])
    if t == 's':
        if L == DEAD: return DEAD
        if L == WON:  return R
        return LIVE
    # parallel (OR-race)
    if L == WON or R == WON: return WON
    if L == DEAD and R == DEAD: return DEAD
    return LIVE

def frontier(s):
    """List of (control_name, k) currently attackable."""
    if status(s) != LIVE: return []
    t = s[0]
    if t == 'c': return [(s[1], s[2])]
    if t == 's':
        return frontier(s[2]) if status(s[1]) == WON else frontier(s[1])
    return frontier(s[1]) + frontier(s[2])

def attack(s, name, outcome, info):
    t = s[0]
    if t == 'c':
        if s[1] != name: return s
        if outcome == 'success': return ('c', name, 'W')
        q = len(info[name][1]); k = s[2] + 1
        return ('c', name, 'D') if k >= q else ('c', name, k)
    a = attack(s[1], name, outcome, info); b = attack(s[2], name, outcome, info)
    return (t, a, b)

def optimal_value(G):
    info = net_info(G); cache = {}; counter = [0]
    def V(s):
        st = status(s)
        if st == WON: return 1.0
        if st == DEAD: return 0.0
        if s in cache: return cache[s]
        counter[0] += 1
        best = 0.0
        for (name, k) in frontier(s):
            beta, ps = info[name]; p = ps[k]
            q = beta * (p * V(attack(s, name, 'success', info)) + (1 - p) * V(attack(s, name, 'fail', info)))
            if q > best: best = q
        cache[s] = best; return best
    val = V(init_state(G))
    return val, counter[0]            # value, number of distinct states expanded

def policy_value(G, policy):
    """Exact expected discounted value of a fixed policy.
    policy(state, frontier_dict) -> control name to attack, or None to stop."""
    info = net_info(G); cache = {}
    def V(s):
        st = status(s)
        if st == WON: return 1.0
        if st == DEAD: return 0.0
        if s in cache: return cache[s]
        fr = dict(frontier(s))
        name = policy(s, fr)
        if name is None: cache[s] = 0.0; return 0.0
        k = fr[name]; beta, ps = info[name]; p = ps[k]
        v = beta * (p * V(attack(s, name, 'success', info)) + (1 - p) * V(attack(s, name, 'fail', info)))
        cache[s] = v; return v
    return V(init_state(G))

if __name__ == "__main__":
    from networks import worked_example
    import sp_index
    G = worked_example()
    val, n = optimal_value(G)
    print("MDP optimal value :", round(val, 6), " states:", n)
    print("index game value  :", round(sp_index.game_value(G), 6))

# ---------- additions: capped MDP + Monte-Carlo policy evaluation ----------
class CapExceeded(Exception):
    pass

def optimal_value_capped(G, state_cap=300_000):
    """Like optimal_value but raises CapExceeded once state_cap states are seen."""
    info = net_info(G); cache = {}; counter = [0]
    def V(s):
        st = status(s)
        if st == WON: return 1.0
        if st == DEAD: return 0.0
        if s in cache: return cache[s]
        counter[0] += 1
        if counter[0] > state_cap: raise CapExceeded()
        best = 0.0
        for (name, k) in frontier(s):
            beta, ps = info[name]; p = ps[k]
            q = beta * (p * V(attack(s, name, 'success', info)) + (1 - p) * V(attack(s, name, 'fail', info)))
            if q > best: best = q
        cache[s] = best; return best
    val = V(init_state(G)); return val, counter[0]

def mc_policy_value(G, policy, n_samples=20000, rng=None):
    """Monte-Carlo estimate of a fixed policy's value: scales to any network
    size (independent of state count). Discounted payoff = product of betas to
    reaching the target, else 0."""
    import random as _r
    if rng is None: rng = _r.Random(0)
    info = net_info(G); total = 0.0
    for _ in range(n_samples):
        s = init_state(G); disc = 1.0
        while True:
            st = status(s)
            if st == WON: total += disc; break
            if st == DEAD: break
            fr = dict(frontier(s)); name = policy(s, fr)
            if name is None: break
            beta, ps = info[name]; disc *= beta
            outcome = 'success' if rng.random() < ps[fr[name]] else 'fail'
            s = attack(s, name, outcome, info)
    return total / n_samples
