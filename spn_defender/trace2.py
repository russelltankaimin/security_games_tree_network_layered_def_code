"""Full numerical trace (Tutorial 2): N = Par(Series(a,Par(b,c)), Series(d,e)).

Larger instance with three nesting levels and a multi-failure control (q_a=2),
so the leaf D^0 recursion has two rungs and the buried Par has a non-trivial
cumulative weight. Prints all intermediates + brute-force/finite-diff checks.
"""
import math
from verify import (PWL, ctrl_profile, ser_profile, par_profile, fold,
                    brute_Vstar, cumulative, step_to_list)

rho=1.0
ell={'a':0.20,'b':0.10,'c':0.15,'d':0.30,'e':0.25}
plist={'a':[0.5,0.4],'b':[0.6],'c':[0.55],'d':[0.5],'e':[0.45]}
qd={'a':2,'b':1,'c':1,'d':1,'e':1}
beta={v:math.exp(-rho*ell[v]) for v in ell}
def R(x,n=5): return round(x,n)

print("PARAMETERS (rho=1)")
for v in ['a','b','c','d','e']:
    print(f"  {v}: l={ell[v]}, q={qd[v]}, p={plist[v]}, beta={R(beta[v])}")
print()

# leaf profiles
P={}; PSI={}; THR={}
for v in ['a','b','c','d','e']:
    p,psi,thr=ctrl_profile(beta[v],plist[v],qd[v])
    P[v]=p; PSI[v]=psi; THR[v]=thr
print("LEAF PROFILES")
for v in ['a','b','c','d','e']:
    print(f"  {v}: breakpoints/values:", [(R(x,4),R(y,4)) for x,y in zip(P[v].xs,P[v].ys)],
          " thresholds:", {k:R(THR[v][k],4) for k in sorted(THR[v])})
print()

# forward
PM=par_profile(P['b'],P['c'])          # M = Par(b,c)
PL=ser_profile(P['a'],PM)              # L = Series(a, M)
PR=ser_profile(P['d'],P['e'])         # R = Series(d, e)
PN=par_profile(PL,PR)                  # N = Par(L, R)
Vstar=PN(0.0)
def pieces(name,Pp):
    print(f"  {name}: bkpts/vals:", [(R(x,4),R(y,4)) for x,y in zip(Pp.xs,Pp.ys)])
print("FORWARD FOLD")
pieces("M=Par(b,c)",PM)
pieces("L=Ser(a,M)",PL)
pieces("R=Ser(d,e)",PR)
pieces("N=Par(L,R)",PN)
print(f"  V* = Psi_N(0) = {R(Vstar)}")
print()

# backward, instrumented
print("BACKWARD PASS")
def cum(L): return cumulative(L)
LN=[(0.0,1.0)]
print(f"  L_N = {[(R(s,4),R(c,4)) for s,c in LN]}")

# Par @ N: children L (=branch1, sibling R) and R (=branch2, sibling L)
C=cum(LN)
WL=lambda z: PR.rslope(z)*C(z)   # branch L modulated by sibling R's slope
WR=lambda z: PL.rslope(z)*C(z)
L_L=step_to_list(WL,set(PR.xs)|{s for s,_ in LN})
L_R=step_to_list(WR,set(PL.xs)|{s for s,_ in LN})
print(f"  [Par@N]  L_L = {[(R(s,4),R(c,4)) for s,c in L_L]}")
print(f"           L_R = {[(R(s,4),R(c,4)) for s,c in L_R]}")

# Series @ L: upstream a, downstream M
def tser(Pdown,s):
    d=Pdown(s); return s/d if d>1e-15 else 0.0
L_a=[]; L_M=[]
print("  [Ser@L] upstream a <- (t(s), c*Psi_M(s));  downstream M <- (s, c*kappa), kappa=Psi_a(t)-t*g_a(t)")
for s,c in L_L:
    t=tser(PM,s); pm=PM(s); kap=P['a'](t)-t*P['a'].rslope(t)
    L_a.append((t,c*pm)); L_M.append((s,c*kap))
    print(f"     s={R(s,4)},c={R(c,4)}: t={R(t,4)},Psi_M(s)={R(pm,4)},kappa={R(kap,4)} -> a:({R(t,4)},{R(c*pm,4)}) M:({R(s,4)},{R(c*kap,4)})")
print(f"     L_a = {[(R(s,4),R(c,4)) for s,c in L_a]}")
print(f"     L_M = {[(R(s,4),R(c,4)) for s,c in L_M]}")

# Series @ R: upstream d, downstream e
L_d=[]; L_e=[]
print("  [Ser@R] upstream d, downstream e")
for s,c in L_R:
    t=tser(P['e'],s); pe=P['e'](s); kap=P['d'](t)-t*P['d'].rslope(t)
    L_d.append((t,c*pe)); L_e.append((s,c*kap))
    print(f"     s={R(s,4)},c={R(c,4)}: t={R(t,4)},Psi_e(s)={R(pe,4)},kappa={R(kap,4)} -> d:({R(t,4)},{R(c*pe,4)}) e:({R(s,4)},{R(c*kap,4)})")
print(f"     L_d = {[(R(s,4),R(c,4)) for s,c in L_d]}")
print(f"     L_e = {[(R(s,4),R(c,4)) for s,c in L_e]}")

# Par @ M: children b (sibling c), c (sibling b), incoming L_M
CM=cum(L_M)
Wb=lambda z: P['c'].rslope(z)*CM(z)
Wc=lambda z: P['b'].rslope(z)*CM(z)
L_b=step_to_list(Wb,set(P['c'].xs)|{s for s,_ in L_M})
L_c=step_to_list(Wc,set(P['b'].xs)|{s for s,_ in L_M})
print(f"  [Par@M] C_M(z)=cumsum(L_M);  L_b=StepToList(g_c*C_M), L_c=StepToList(g_b*C_M)")
print(f"          L_b = {[(R(s,4),R(c,4)) for s,c in L_b]}")
print(f"          L_c = {[(R(s,4),R(c,4)) for s,c in L_c]}")
print()

# leaf cashout with D0 recursion
def D0(v,s):
    q=qd[v]; pl=plist[v]; b=beta[v]; psi=PSI[v]; thr=THR[v]
    D=0.0
    rungs=[]
    for k in range(q-1,-1,-1):
        if s<thr[k]-1e-12:
            val=(pl[k]+(1-pl[k])*psi[k+1](s)) + b*(1-pl[k])*D
        else:
            val=0.0
        rungs.append((k,val))
        D=val
    return D,rungs

print("LEAF CASH-OUT:  dV*/dl_v = -rho*beta_v * sum_j c_j * D0_v(s_j)")
grad={}
for v,L in [('a',L_a),('b',L_b),('c',L_c),('d',L_d),('e',L_e)]:
    tot=0.0; parts=[]
    for s,c in L:
        d,_=D0(v,s); tot+=c*d
        parts.append(f"{R(c,4)}*{R(d,4)}={R(c*d,5)}")
    g=-rho*beta[v]*tot; grad[v]=g
    print(f"  {v}: sum = "+" + ".join(parts)+f" = {R(tot,5)};  dV*/dl_{v} = -{R(beta[v])}*{R(tot,5)} = {R(g,5)}")
# show a's D0 recursion explicitly at its query points
print("  --- control a (q=2) D0 recursion at its query points ---")
for s,c in L_a:
    d,rungs=D0('a',s)
    print(f"     s={R(s,4)}: "+"; ".join(f"D^({k})={R(val,4)}" for k,val in rungs)+f"  -> D0={R(d,4)}")
print()

# verify
print("VERIFY")
G=('par',('ser',('ctrl','a',2,[0.5,0.4]),('par',('ctrl','b',1,[0.6]),('ctrl','c',1,[0.55]))),
         ('ser',('ctrl','d',1,[0.5]),('ctrl','e',1,[0.45])))
pd={v:plist[v] for v in plist}
Vb=brute_Vstar(G,beta,pd,qd)
print(f"  V*: fold={R(Vstar)}  brute={R(Vb)}  |diff|={abs(Vstar-Vb):.2e}")
h=1e-6
for v in ['a','b','c','d','e']:
    ep=dict(ell); ep[v]+=h; em=dict(ell); em[v]-=h
    bp={k:math.exp(-rho*ep[k]) for k in ell}; bm={k:math.exp(-rho*em[k]) for k in ell}
    fd=(brute_Vstar(G,bp,pd,qd)-brute_Vstar(G,bm,pd,qd))/(2*h)
    print(f"  dV*/dl_{v}: trace={R(grad[v],5)}  FD={R(fd,5)}  |diff|={abs(grad[v]-fd):.2e}")
