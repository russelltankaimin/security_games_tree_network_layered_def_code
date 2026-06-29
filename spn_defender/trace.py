"""Full numerical trace (Tutorial 1): N = Par(Series(a,b), c), single-shot.

Prints every intermediate of the forward fold (leaf profiles, series fold,
parallel integral) and the backward gradient pass (seed, parallel rule,
series rule, leaf cash-out), then verifies V* and grad against the brute-force
MDP and central finite differences.
"""
import math
from verify import (PWL, ctrl_profile, ser_profile, par_profile, fold,
                    Vstar_fold, brute_Vstar, controls, cumulative, step_to_list,
                    grad_reverse)

# ---- instance: Par( Series(a,b), c ), single-shot controls ----
rho=1.0
ell={'a':0.20,'b':0.30,'c':0.50}
p={'a':0.60,'b':0.50,'c':0.55}
beta={v:math.exp(-rho*ell[v]) for v in ell}
def R(x,n=5): return round(x,n)

print("PARAMETERS")
for v in ['a','b','c']:
    print(f"  {v}: l={ell[v]}, p={p[v]}, beta=e^-{rho*ell[v]}={R(beta[v])}")
print()

# ---- leaf profiles (single shot q=1) ----
def leaf_info(v):
    b=beta[v]; pv=p[v]
    slope=b*(1-pv); icpt=b*pv
    alpha=icpt/(1-slope)
    return slope,icpt,alpha
print("STEP 1 - LEAF PROFILES  Psi_v(s)=max{s, beta_v*p_v + beta_v*(1-p_v)*s}")
prof={}
for v in ['a','b','c']:
    slope,icpt,alpha=leaf_info(v)
    P,_,_=ctrl_profile(beta[v],[p[v]],1)
    prof[v]=P
    print(f"  {v}: attack line = {R(icpt)} + {R(slope)}*s ;  threshold a_hat={R(alpha)}")
    print(f"      Psi_{v}: = attack line on [0,{R(alpha)}],  = s on [{R(alpha)},1]")
    print(f"      g_{v}(s) = {R(slope)} on [0,{R(alpha)}),  = 1 on [{R(alpha)},1]")
print()

# ---- series fold ab ----
print("STEP 2 - SERIES FOLD  Psi_ab(s)=Psi_b(s)*Psi_a( s/Psi_b(s) )")
Pa,Pb,Pc=prof['a'],prof['b'],prof['c']
Pab=ser_profile(Pa,Pb)
print("  breakpoints of Psi_ab and values:")
for x,y in zip(Pab.xs,Pab.ys):
    print(f"      s={R(x)}  Psi_ab={R(y)}")
print("  slopes g_ab per piece:")
for i in range(len(Pab.xs)-1):
    print(f"      [{R(Pab.xs[i])},{R(Pab.xs[i+1])}): g_ab={R((Pab.ys[i+1]-Pab.ys[i])/(Pab.xs[i+1]-Pab.xs[i]))}")
# threshold of ab
ab_thr=None
for i in range(len(Pab.xs)-1):
    if abs(Pab.ys[i]-Pab.xs[i])<1e-9:
        ab_thr=Pab.xs[i]; break
print(f"  a_hat_ab (where Psi_ab leaves the diagonal) = {R(ab_thr) if ab_thr is not None else 'n/a'}")
print()

# ---- par fold ----
print("STEP 3 - PARALLEL FOLD  Psi_N(s)=1 - integral_s^1 g_ab(z) g_c(z) dz")
slope_c,icpt_c,alpha_c=leaf_info('c')
grid=sorted(set(Pab.xs)|set(Pc.xs))
print("  integration pieces (interval | g_ab | g_c | g_ab*g_c | width | contribution):")
total=0
for i in range(len(grid)-1):
    z0,z1=grid[i],grid[i+1]; mid=0.5*(z0+z1)
    gab=Pab.rslope(mid); gc=Pc.rslope(mid); prod=gab*gc; w=z1-z0; c=prod*w
    total+=c
    print(f"      [{R(z0)},{R(z1)}] | {R(gab)} | {R(gc)} | {R(prod)} | {R(w)} | {R(c)}")
PN=par_profile(Pab,Pc)
Vstar=PN(0.0)
print(f"  integral_0^1 g_ab g_c = {R(total)}")
print(f"  V* = Psi_N(0) = 1 - {R(total)} = {R(Vstar)}")
print(f"  (check Vstar_fold={R(Vstar_fold(fold_=None) if False else PN(0.0))})")
print()

# ---- BACKWARD ----
print("STEP 4 - SEED ROOT:  L_N = {(0, 1)}")
print()

print("STEP 5 - PARALLEL RULE at N (children: ab=branch1, c=branch2)")
LN=[(0.0,1.0)]
C=cumulative(LN)
print(f"  cumulative C(z) = sum of weights with s_j<=z = 1 for all z in [0,1]")
# child ab gets StepToList(g_c * C); child c gets StepToList(g_ab * C)
Wc=lambda z: Pc.rslope(z)*C(z)
Wab=lambda z: Pab.rslope(z)*C(z)
L_ab=step_to_list(Wc, set(Pc.xs)|set(s for s,_ in LN))
L_c =step_to_list(Wab,set(Pab.xs)|set(s for s,_ in LN))
print(f"  branch ab <- StepToList(g_c * C):  L_ab = {[(R(s,4),R(c,4)) for s,c in L_ab]}")
print(f"     (g_c = {R(slope_c)} then 1 at a_hat_c={R(alpha_c)};  point at 0 carries {R(slope_c)}, jump {R(1-slope_c)} at {R(alpha_c)})")
print(f"  branch c  <- StepToList(g_ab * C):  L_c  = {[(R(s,4),R(c,4)) for s,c in L_c]}")
print()

print("STEP 6 - SERIES RULE at ab (a=upstream, b=downstream), incoming L_ab")
def t(s):
    p2=Pb(s); return s/p2 if p2>1e-15 else 0.0
print("  for each (s,c) in L_ab:  a gets (t(s), c*Psi_b(s));  b gets (s, c*kappa(s))")
print("    where t(s)=s/Psi_b(s),  kappa(s)=Psi_a(t(s)) - t(s)*g_a(t(s))")
L_a=[]; L_b=[]
for s,c in L_ab:
    ts=t(s); psib=Pb(s); kap=Pa(ts)-ts*Pa.rslope(ts)
    L_a.append((ts,c*psib)); L_b.append((s,c*kap))
    print(f"    s={R(s,4)}, c={R(c,4)}:  t(s)={R(ts,4)}, Psi_b(s)={R(psib,4)}, kappa={R(kap,4)}")
    print(f"        -> a:({R(ts,4)}, {R(c*psib,4)})    b:({R(s,4)}, {R(c*kap,4)})")
print(f"  L_a = {[(R(s,4),R(c,4)) for s,c in L_a]}")
print(f"  L_b = {[(R(s,4),R(c,4)) for s,c in L_b]}")
print()

print("STEP 7 - LEAF CASH-OUT.  D0_v(s)=p_v+(1-p_v)*s for s<a_hat_v else 0;  dV/dl_v = -rho*beta_v*sum c*D0(s)")
def D0(v,s):
    sl,ic,al=leaf_info(v)
    return (p[v]+(1-p[v])*s) if s<al-1e-12 else 0.0
grad={}
for v,L in [('a',L_a),('b',L_b),('c',L_c)]:
    sl,ic,al=leaf_info(v)
    terms=[]
    tot=0.0
    for s,c in L:
        d=D0(v,s); tot+=c*d
        terms.append(f"{R(c,4)}*D0({R(s,4)})={R(c,4)}*{R(d,4)}={R(c*d,5)}")
    dbeta=tot
    g=-rho*beta[v]*dbeta
    grad[v]=g
    print(f"  {v}: a_hat={R(al)};  sum = " + " + ".join(terms) + f" = {R(dbeta,5)}")
    print(f"      dV*/dbeta_{v} = {R(dbeta,5)};  dV*/dl_{v} = -{rho}*{R(beta[v])}*{R(dbeta,5)} = {R(g,5)}")
print()

print("STEP 8 - VERIFY against independent brute-force MDP + finite differences")
pdict={'a':[p['a']],'b':[p['b']],'c':[p['c']]}; qdict={'a':1,'b':1,'c':1}
G=('par',('ser',('ctrl','a',1,[p['a']]),('ctrl','b',1,[p['b']])),('ctrl','c',1,[p['c']]))
Vb=brute_Vstar(G,beta,pdict,qdict)
print(f"  V*: fold={R(Vstar)}  brute-MDP={R(Vb)}  |diff|={abs(Vstar-Vb):.2e}")
h=1e-6
for v in ['a','b','c']:
    ep=dict(ell); ep[v]+=h; em=dict(ell); em[v]-=h
    bp={k:math.exp(-rho*ep[k]) for k in ell}; bm={k:math.exp(-rho*em[k]) for k in ell}
    fd=(brute_Vstar(G,bp,pdict,qdict)-brute_Vstar(G,bm,pdict,qdict))/(2*h)
    print(f"  dV*/dl_{v}: trace={R(grad[v],5)}  finite-diff={R(fd,5)}  |diff|={abs(grad[v]-fd):.2e}")
