"""
Exact index algorithm for series-parallel layered-defence networks.
Implements Algorithm 1 (Profile / Indices / online ranking) from the note.

Core object: a convex piecewise-linear (PWL) function on [0,1], stored by
breakpoints (xs, ys).  Every value-profile in the fold is such a function.
"""
import bisect

# ---------- PWL primitives ----------
class PWL:
    def __init__(self, xs, ys):
        # collapse duplicate x
        nx, ny = [], []
        for x, y in zip(xs, ys):
            if nx and abs(x - nx[-1]) < 1e-15:
                ny[-1] = y
            else:
                nx.append(x); ny.append(y)
        self.xs, self.ys = nx, ny
    def __call__(self, x):
        xs, ys = self.xs, self.ys
        if x <= xs[0]:  return ys[0] if len(xs)==1 else ys[0]+(x-xs[0])/(xs[1]-xs[0])*(ys[1]-ys[0])
        if x >= xs[-1]: return ys[-1]+(x-xs[-1])/(xs[-1]-xs[-2])*(ys[-1]-ys[-2])
        i = bisect.bisect_right(xs, x) - 1
        return ys[i] + (x-xs[i])/(xs[i+1]-xs[i])*(ys[i+1]-ys[i])

IDENT = PWL([0.0, 1.0], [0.0, 1.0])              # s |-> s
CONST1 = PWL([0.0, 1.0], [1.0, 1.0])             # gamma |-> 1

def lincomb(f, g, cf, cg):                        # cf*f + cg*g  (PWL, exact)
    grid = sorted(set(f.xs) | set(g.xs))
    return PWL(grid, [cf*f(x) + cg*g(x) for x in grid])

def pwl_max(f, g):                                # pointwise max (adds crossings)
    grid = sorted(set(f.xs) | set(g.xs))
    xs, ys = [], []
    for i in range(len(grid)-1):
        a, b = grid[i], grid[i+1]
        da, db = f(a)-g(a), f(b)-g(b)
        xs.append(a); ys.append(max(f(a), g(a)))
        if da*db < 0:                             # strict crossing inside
            t = da/(da-db); xc = a+t*(b-a)
            xs.append(xc); ys.append(f(a)+t*(f(b)-f(a)))
    xs.append(grid[-1]); ys.append(max(f(grid[-1]), g(grid[-1])))
    return PWL(xs, ys)

def par_combine(f1, f2):                           # 1 - \int_s^1 f1' f2'  (Theorem 19 / eq:par)
    grid = sorted(set(f1.xs) | set(f2.xs))
    ys = [0.0]*len(grid)
    ys[-1] = 1.0
    for i in range(len(grid)-2, -1, -1):
        a, b = grid[i], grid[i+1]
        s1 = (f1(b)-f1(a))/(b-a); s2 = (f2(b)-f2(a))/(b-a)
        ys[i] = ys[i+1] - s1*s2*(b-a)             # subtract slope-product * width
    return PWL(grid, ys)

def compose(R, P):                                 # gamma |-> R(gamma) * P(gamma / R(gamma))
    cand = set(R.xs)                               # (series substitution + homogeneity)
    for i in range(len(R.xs)-1):
        ax, bx, ay, by = R.xs[i], R.xs[i+1], R.ys[i], R.ys[i+1]
        b = (by-ay)/(bx-ax); a = ay - b*ax         # R = a + b*gamma on this piece
        for u in P.xs:                             # solve gamma/R(gamma) = u
            denom = 1 - u*b
            if abs(denom) > 1e-15:
                g = u*a/denom
                if ax-1e-12 <= g <= bx+1e-12:
                    cand.add(min(max(g, ax), bx))
    cand = sorted(c for c in cand if -1e-9 <= c <= 1+1e-9)
    xs, ys = [], []
    for g in cand:
        Rg = R(g); arg = min(max(g/Rg, 0.0), 1.0) if Rg > 1e-15 else 0.0
        xs.append(g); ys.append(Rg*P(arg))
    return PWL(xs, ys)

def crossing(f):                                   # min{s : f(s)=s}  = the index
    d = lincomb(f, IDENT, 1.0, -1.0)               # f(s)-s, nonincreasing, >=0 then 0
    xs, ys = d.xs, d.ys
    for i in range(len(xs)):
        if ys[i] <= 1e-12:
            if i == 0: return xs[0]
            y0, y1, x0, x1 = ys[i-1], ys[i], xs[i-1], xs[i]
            return x0 if y0 == y1 else x0 + y0/(y0-y1)*(x1-x0)
    return xs[-1]

# ---------- network nodes ----------
def Ctrl(name, beta, ps): return ('ctrl', name, beta, ps)   # ps = [p(0),...,p(q-1)]
def Ser(a, b):            return ('ser', a, b)
def Par(a, b):            return ('par', a, b)

# ---------- Pass 1: intrinsic profiles ----------
def profile(G):
    if G[0] == 'ctrl':
        _, _, beta, ps = G
        Psi = IDENT                                # count q: jammed -> value = buyout
        for k in range(len(ps)-1, -1, -1):         # backward induction over counts
            inner = lincomb(Psi, CONST1, beta*(1-ps[k]), beta*ps[k])  # b[(1-p)Psi + p*1]
            Psi = pwl_max(IDENT, inner)
        return Psi
    if G[0] == 'par':
        return par_combine(profile(G[1]), profile(G[2]))
    return compose(profile(G[2]), profile(G[1]))    # ser: prize of G1 is value of G2

# ---------- Pass 2: per-control indices ----------
def indices(G, R, table):
    if G[0] == 'ctrl':
        _, name, beta, ps = G
        Psi = IDENT
        for k in range(len(ps)-1, -1, -1):
            inner = lincomb(Psi, R, beta*(1-ps[k]), beta*ps[k])       # b[(1-p)Psi + p*R(gamma)]
            Psi = pwl_max(IDENT, inner)
            table[(name, k)] = crossing(Psi)
        return
    if G[0] == 'ser':
        indices(G[2], R, table)                     # right child inherits reward R
        indices(G[1], compose(R, profile(G[2])), table)  # left child: prize = value of G2
        return
    indices(G[1], R, table); indices(G[2], R, table) # par: both inherit R

# ---------- run on the worked example ----------
A = Ctrl('A', 0.9, [0.5, 0.3])
B = Ctrl('B', 0.9, [0.8, 0.5])
C = Ctrl('C', 0.9, [0.6, 0.4])
N = Par(Ser(A, B), C)

print("Game value  V_N =", round(profile(N)(0.0), 6), " (expected 0.758591)")
tbl = {}
indices(N, CONST1, tbl)
expected = {('C',0):0.84375,('C',1):0.782609,('B',0):0.878049,('B',1):0.818182,
            ('A',0):0.701868,('A',1):0.621227}
print("\n  control,count : computed   expected")
for key in [('A',0),('A',1),('B',0),('B',1),('C',0),('C',1)]:
    print(f"   alpha{key} = {tbl[key]:.6f}   {expected[key]:.6f}")