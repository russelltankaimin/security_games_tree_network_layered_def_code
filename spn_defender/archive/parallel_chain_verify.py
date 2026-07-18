"""Our method on the AAAI-26 PARALLEL-CHAINS topology.

N = Par(Series(a1,a2), Series(b1,b2)) is exactly their n=2, m=2 parallel-chains
model. Shows fold V* == brute-force MDP V*, and our reverse-mode subgradient ==
central finite differences of V*_l (the same acceptance check the AAAI authors
use: 'gradients match Euler forward method').

NOTE: this applies OUR fold/reverse-mode to their topology and cross-checks it
against the brute-force MDP + finite differences. It is NOT a reimplementation
of the authors' own stage/product-distribution gradient algorithm.
"""
import math
from verify import controls, Vstar_fold, brute_Vstar, grad_reverse
# AAAI parallel-chains instance: n=2 chains, each a Series of 2 controls (m=2)
# N = Par( Series(a1,a2), Series(b1,b2) ) -- exactly their topology
G=('par', ('ser',('ctrl','a1',2,[0.6,0.5]),('ctrl','a2',1,[0.55])),
          ('ser',('ctrl','b1',1,[0.5]),  ('ctrl','b2',2,[0.45,0.4])))
rho=1.0
ell={'a1':0.30,'a2':0.20,'b1':0.25,'b2':0.25}   # on the simplex
beta={v:math.exp(-rho*ell[v]) for v in controls(G)}
pd={'a1':[0.6,0.5],'a2':[0.55],'b1':[0.5],'b2':[0.45,0.4]}
qd={'a1':2,'a2':1,'b1':1,'b2':2}
Vf=Vstar_fold(G,beta); Vb=brute_Vstar(G,beta,pd,qd)
gr=grad_reverse(G,beta,rho)
print(f"parallel-chains  V*: fold={Vf:.8f}  brute-MDP={Vb:.8f}  |diff|={abs(Vf-Vb):.1e}")
h=1e-6
print("subgradient (our reverse-mode)  vs  finite-diff of V*_l  (= the AAAI check):")
for v in controls(G):
    ep=dict(ell); ep[v]+=h; em=dict(ell); em[v]-=h
    bp={k:math.exp(-rho*ep[k]) for k in ell}; bm={k:math.exp(-rho*em[k]) for k in ell}
    fd=(brute_Vstar(G,bp,pd,qd)-brute_Vstar(G,bm,pd,qd))/(2*h)
    print(f"   dV*/dl_{v:3s}: reverse={gr[v]:+.7f}   FD={fd:+.7f}   |diff|={abs(gr[v]-fd):.1e}")
