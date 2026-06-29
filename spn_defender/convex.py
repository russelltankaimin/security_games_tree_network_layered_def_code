"""Convexity check for V*(l) on the simplex.

Samples random chords on the simplex and verifies the convexity inequality
V*(z) <= lam*V*(x)+(1-lam)*V*(y) for z=lam*x+(1-lam)*y, across several nested
instances and discount rates. Confirms Proposition A2 numerically.
"""
import math, random
from verify import controls, Vstar_fold, brute_Vstar

def Vstar(G, ell, rho):
    ctl=controls(G); beta={v:math.exp(-rho*ell[v]) for v in ctl}
    return Vstar_fold(G,beta)

def simplex_point(ctl, rng):
    x=[rng.random() for _ in ctl]; s=sum(x); return {v:xi/s for v,xi in zip(ctl,x)}

random.seed(0)
insts={
 "Par(a,b)": ('par',('ctrl','a',1,[0.5]),('ctrl','b',1,[0.5])),
 "Ser(a,b)": ('ser',('ctrl','a',2,[0.8,0.6]),('ctrl','b',1,[0.7])),
 "Par(Ser(a,b),c)": ('par',('ser',('ctrl','a',1,[0.6]),('ctrl','b',2,[0.5,0.4])),('ctrl','c',1,[0.55])),
 "Par(Ser(a,Par(b,c)),d)": ('par',('ser',('ctrl','a',2,[0.6,0.5]),('par',('ctrl','b',1,[0.7]),('ctrl','c',1,[0.5]))),('ctrl','d',2,[0.4,0.3])),
}
for rho in (0.5,1.0,2.0):
  for name,G in insts.items():
    ctl=controls(G); worst=0.0; ntest=4000
    for _ in range(ntest):
        x=simplex_point(ctl,random.Random(random.random()))
        y=simplex_point(ctl,random.Random(random.random()))
        lam=random.random()
        z={v:lam*x[v]+(1-lam)*y[v] for v in ctl}
        lhs=Vstar(G,z,rho)
        rhs=lam*Vstar(G,x,rho)+(1-lam)*Vstar(G,y,rho)
        # convexity: lhs <= rhs ; violation = lhs-rhs
        worst=max(worst, lhs-rhs)
    print(f"rho={rho}  {name:24s}  max convexity violation (lhs-rhs) over {ntest} chords = {worst:.2e}")
print("\n(convex  <=>  violation <= ~0; tiny positive = numerical noise)")
