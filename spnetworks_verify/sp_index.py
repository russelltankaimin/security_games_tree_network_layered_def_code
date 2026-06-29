"""
Exact index algorithm (Algorithm 1): convex PWL profiles + index threading.
Operates on the network tuples from networks.py. See sp_index in the note.
"""
import bisect

class PWL:
    def __init__(self, xs, ys):
        nx, ny = [], []
        for x, y in zip(xs, ys):
            if nx and abs(x - nx[-1]) < 1e-15: ny[-1] = y
            else: nx.append(x); ny.append(y)
        self.xs, self.ys = nx, ny
    def __call__(self, x):
        xs, ys = self.xs, self.ys
        if x <= xs[0]:  return ys[0] if len(xs) == 1 else ys[0] + (x - xs[0]) / (xs[1] - xs[0]) * (ys[1] - ys[0])
        if x >= xs[-1]: return ys[-1] + (x - xs[-1]) / (xs[-1] - xs[-2]) * (ys[-1] - ys[-2])
        i = bisect.bisect_right(xs, x) - 1
        return ys[i] + (x - xs[i]) / (xs[i+1] - xs[i]) * (ys[i+1] - ys[i])

IDENT  = PWL([0.0, 1.0], [0.0, 1.0])
CONST1 = PWL([0.0, 1.0], [1.0, 1.0])

def lincomb(f, g, cf, cg):
    grid = sorted(set(f.xs) | set(g.xs))
    return PWL(grid, [cf * f(x) + cg * g(x) for x in grid])

def pwl_max(f, g):
    grid = sorted(set(f.xs) | set(g.xs)); xs, ys = [], []
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i+1]
        da, db = f(a) - g(a), f(b) - g(b)
        xs.append(a); ys.append(max(f(a), g(a)))
        if da * db < 0:
            t = da / (da - db); xs.append(a + t * (b - a)); ys.append(f(a) + t * (f(b) - f(a)))
    xs.append(grid[-1]); ys.append(max(f(grid[-1]), g(grid[-1])))
    return PWL(xs, ys)

def par_combine(f1, f2):
    grid = sorted(set(f1.xs) | set(f2.xs)); ys = [0.0] * len(grid); ys[-1] = 1.0
    for i in range(len(grid) - 2, -1, -1):
        a, b = grid[i], grid[i+1]
        s1 = (f1(b) - f1(a)) / (b - a); s2 = (f2(b) - f2(a)) / (b - a)
        ys[i] = ys[i+1] - s1 * s2 * (b - a)
    return PWL(grid, ys)

def compose(R, P):
    cand = set(R.xs)
    for i in range(len(R.xs) - 1):
        ax, bx, ay, by = R.xs[i], R.xs[i+1], R.ys[i], R.ys[i+1]
        b = (by - ay) / (bx - ax); a = ay - b * ax
        for u in P.xs:
            denom = 1 - u * b
            if abs(denom) > 1e-15:
                g = u * a / denom
                if ax - 1e-12 <= g <= bx + 1e-12: cand.add(min(max(g, ax), bx))
    cand = sorted(c for c in cand if -1e-9 <= c <= 1 + 1e-9)
    xs, ys = [], []
    for g in cand:
        Rg = R(g); arg = min(max(g / Rg, 0.0), 1.0) if Rg > 1e-15 else 0.0
        xs.append(g); ys.append(Rg * P(arg))
    return PWL(xs, ys)

def crossing(f):
    d = lincomb(f, IDENT, 1.0, -1.0); xs, ys = d.xs, d.ys
    for i in range(len(xs)):
        if ys[i] <= 1e-12:
            if i == 0: return xs[0]
            y0, y1, x0, x1 = ys[i-1], ys[i], xs[i-1], xs[i]
            return x0 if y0 == y1 else x0 + y0 / (y0 - y1) * (x1 - x0)
    return xs[-1]

# ---------- Pass 1: profiles ----------
def profile(G):
    if G[0] == 'ctrl':
        _, _, beta, ps = G; Psi = IDENT
        for k in range(len(ps) - 1, -1, -1):
            Psi = pwl_max(IDENT, lincomb(Psi, CONST1, beta * (1 - ps[k]), beta * ps[k]))
        return Psi
    if G[0] == 'par': return par_combine(profile(G[1]), profile(G[2]))
    return compose(profile(G[2]), profile(G[1]))

# ---------- Pass 2: indices ----------
def indices(G, R, table):
    if G[0] == 'ctrl':
        _, name, beta, ps = G; Psi = IDENT
        for k in range(len(ps) - 1, -1, -1):
            Psi = pwl_max(IDENT, lincomb(Psi, R, beta * (1 - ps[k]), beta * ps[k]))
            table[(name, k)] = crossing(Psi)
        return
    if G[0] == 'ser':
        indices(G[2], R, table); indices(G[1], compose(R, profile(G[2])), table); return
    indices(G[1], R, table); indices(G[2], R, table)

# ---------- public helpers ----------
def game_value(G):
    """Optimal expected discounted reward of the true game (gamma=0, reward 1)."""
    return profile(G)(0.0)

def index_table(G):
    """Return {(control_name, count k): alpha(v,k)} for the whole network."""
    t = {}; indices(G, CONST1, t); return t

if __name__ == "__main__":
    from networks import worked_example
    N = worked_example()
    print("game value:", round(game_value(N), 6))
    for k, v in sorted(index_table(N).items()):
        print(k, round(v, 6))
