"""Dumps the backward-pass weight lists at every node of Par(Ser(a,b),c).

Used to source the exact numbers shown in the interactive walkthrough: the
seed L_N={(0,1)}, the per-node weight lists, and the final gradient.
"""
import math
from verify import fold, grad_reverse, controls, Vstar_fold, cumulative, step_to_list

# Clean single-shot instance: Par(Ser(a,b), c)
G=('par',('ser',('ctrl','a',1,[0.6]),('ctrl','b',1,[0.5])),('ctrl','c',1,[0.55]))
ell={'a':0.30,'b':0.30,'c':0.40}
rho=1.0
beta={v:math.exp(-rho*ell[v]) for v in controls(G)}

# instrument grad_reverse to record weight lists per node
ann={}
fold(G,beta,ann)
records=[]
def go(H,L,label):
    a=ann[id(H)]
    records.append((label, a[0], [(round(s,3),round(c,3)) for s,c in L]))
    if a[0]=='ctrl':
        return
    elif a[0]=='par':
        _,P,P1,P2,G1,G2=a
        C=cumulative(L)
        b1=set(P2.xs)|set(s for s,_ in L); b2=set(P1.xs)|set(s for s,_ in L)
        W1=lambda z:P2.rslope(z)*C(z); W2=lambda z:P1.rslope(z)*C(z)
        L1=step_to_list(W1,b1); L2=step_to_list(W2,b2)
        go(G1,L1,label+"→L"); go(G2,L2,label+"→R")
    else:
        _,P,P1,P2,G1,G2=a
        def t(s):
            p2=P2(s); return s/p2 if p2>1e-15 else 0.0
        L1=[(t(s),c*P2(s)) for s,c in L]
        L2=[(s,c*(P1(t(s))-t(s)*P1.rslope(t(s)))) for s,c in L]
        go(G1,L1,label+"→L"); go(G2,L2,label+"→R")
go(G,[(0.0,1.0)],"N")
print("V* =", round(Vstar_fold(G,beta),5))
for lab,typ,L in records:
    print(f"{lab:8s} [{typ:4s}]  L = {L}")
print("grad =", {k:round(v,5) for k,v in grad_reverse(G,beta,rho).items()})
