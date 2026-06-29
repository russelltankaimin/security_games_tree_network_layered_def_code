"""Core library + self-test for the SP layered-defence gradient work.

Provides: a piecewise-linear series-parallel VALUE FOLD (forward pass), an
independent BRUTE-FORCE MDP solver (backward induction over the full joint
state), the REVERSE-MODE weight-list GRADIENT (backward pass), and a
FINITE-DIFFERENCE oracle. Importing this module is silent; run it directly
(`python3 verify.py`) for the full self-test over six parallel/series/nested
instances plus the symmetric-tie subgradient diagnostic.

Conventions: tree nodes are ('ctrl',name,q,p_list) | ('ser',G1,G2) | ('par',G1,G2);
beta_v = exp(-rho*l_v); V* = Psi_N(0).
"""
import math, itertools
from functools import lru_cache

# ---------------------------------------------------------------
# PWL profiles on [0,1], assumed convex, xs[0]=0, xs[-1]=1.
# ---------------------------------------------------------------
TOL=1e-12
class PWL:
    def __init__(self, xs, ys):
        # clean duplicate x
        nx,ny=[],[]
        for x,y in zip(xs,ys):
            if nx and abs(x-nx[-1])<1e-14:
                ny[-1]=y
            else:
                nx.append(x); ny.append(y)
        self.xs=nx; self.ys=ny
    def __call__(self,s):
        xs,ys=self.xs,self.ys
        if s<=xs[0]: 
            # extrapolate first piece
            if len(xs)>1:
                m=(ys[1]-ys[0])/(xs[1]-xs[0]); return ys[0]+m*(s-xs[0])
            return ys[0]
        if s>=xs[-1]:
            if len(xs)>1:
                m=(ys[-1]-ys[-2])/(xs[-1]-xs[-2]); return ys[-1]+m*(s-xs[-1])
            return ys[-1]
        import bisect
        i=bisect.bisect_right(xs,s)-1
        i=max(0,min(i,len(xs)-2))
        m=(ys[i+1]-ys[i])/(xs[i+1]-xs[i])
        return ys[i]+m*(s-xs[i])
    def rslope(self,s):
        # right derivative
        xs,ys=self.xs,self.ys
        if s>=xs[-1]:
            i=len(xs)-2
        else:
            import bisect
            i=bisect.bisect_right(xs,s)-1
            i=max(0,min(i,len(xs)-2))
        return (ys[i+1]-ys[i])/(xs[i+1]-xs[i])
    def breakpoints(self):
        return list(self.xs)

IDENT=PWL([0,1],[0,1])

def affine_of_pwl(P, a, b):  # returns s-> a + b*P(s), keeps breakpoints
    return PWL(list(P.xs), [a+b*y for y in P.ys])

def max_pwl(f,g):
    # pointwise max of two convex PWL -> PWL (exact)
    pts=set(f.xs)|set(g.xs)
    # add crossings within each interval
    grid=sorted(pts)
    allx=sorted(set(grid))
    cross=[]
    for i in range(len(allx)-1):
        x0,x1=allx[i],allx[i+1]
        f0,f1=f(x0),f(x1); g0,g1=g(x0),g(x1)
        d0=f0-g0; d1=f1-g1
        if d0==0 or d1==0: continue
        if (d0<0)!=(d1<0):
            # crossing
            t=d0/(d0-d1)
            xc=x0+t*(x1-x0)
            cross.append(xc)
    allx=sorted(set(allx)|set(cross))
    ys=[max(f(x),g(x)) for x in allx]
    return PWL(allx,ys)

def ctrl_profile(beta,p_list,q):
    # backward recursion; cache psi^{(k)} and thresholds
    psi={q:IDENT}
    thr={}  # threshold alpha-hat(v,k): smallest s with psi^{(k)}(s)=s
    for k in range(q-1,-1,-1):
        p=p_list[k]
        A=affine_of_pwl(psi[k+1], beta*p, beta*(1-p))  # beta*(p + (1-p)*psi)
        pk=max_pwl(IDENT,A)
        psi[k]=pk
        # threshold: smallest s where pk(s)==s  i.e. A(s)<=s
        # A(s)-s changes sign once; find crossing
        th=1.0
        xs=pk.xs
        for i in range(len(xs)-1):
            x0,x1=xs[i],xs[i+1]
            d0=A(x0)-x0; d1=A(x1)-x1
            if d0>1e-15 and d1<=1e-15:
                t=d0/(d0-d1); th=x0+t*(x1-x0); break
            if d0<=1e-15:
                th=x0; break
        thr[k]=th
    return psi[0], psi, thr

def par_profile(P1,P2):
    g1pts=P1.xs; g2pts=P2.xs
    grid=sorted(set(g1pts)|set(g2pts))
    # slopes constant on each subinterval
    ys=[None]*len(grid)
    ys[-1]=1.0
    for i in range(len(grid)-2,-1,-1):
        x0,x1=grid[i],grid[i+1]
        mid=0.5*(x0+x1)
        gg=P1.rslope(mid)*P2.rslope(mid)
        ys[i]=ys[i+1]-gg*(x1-x0)
    return PWL(grid,ys)

def ser_profile(P1,P2):
    # Psi(s)=P2(s)*P1(s/P2(s))
    bps=set([0.0,1.0])|set(P2.xs)
    # solve s/P2(s)=xb for xb in P1 breakpoints, per P2 linear piece
    for pi in range(len(P2.xs)-1):
        a0,a1=P2.xs[pi],P2.xs[pi+1]
        c=P2.ys[pi]; d=(P2.ys[pi+1]-P2.ys[pi])/(a1-a0)  # P2(s)=c+d*(s-a0)
        # P2(s)=(c-d*a0)+d*s = C0 + D*s
        C0=c-d*a0; D=d
        for xb in P1.xs:
            denom=(1-xb*D)
            if abs(denom)<1e-15: continue
            s=xb*C0/denom
            if a0-1e-12<=s<=a1+1e-12 and 0<=s<=1:
                bps.add(min(1,max(0,s)))
    grid=sorted(bps)
    ys=[]
    for s in grid:
        p2=P2(s)
        if p2<=1e-15:
            ys.append(0.0)
        else:
            ys.append(p2*P1(s/p2))
    return PWL(grid,ys)

# ---------------------------------------------------------------
# Tree structures
# ('ctrl', name, q, p_list) | ('ser',G1,G2) | ('par',G1,G2)
# ---------------------------------------------------------------
def controls(G):
    if G[0]=='ctrl': return [G[1]]
    return controls(G[1])+controls(G[2])

def fold(G, beta, ann=None):
    # returns profile; if ann is dict, store node annotations
    if G[0]=='ctrl':
        _,name,q,pl=G
        P,psi,thr=ctrl_profile(beta[name],pl,q)
        if ann is not None: ann[id(G)]=('ctrl',name,P,psi,thr)
        return P
    elif G[0]=='ser':
        P1=fold(G[1],beta,ann); P2=fold(G[2],beta,ann)
        P=ser_profile(P1,P2)
        if ann is not None: ann[id(G)]=('ser',P,P1,P2,G[1],G[2])
        return P
    else:
        P1=fold(G[1],beta,ann); P2=fold(G[2],beta,ann)
        P=par_profile(P1,P2)
        if ann is not None: ann[id(G)]=('par',P,P1,P2,G[1],G[2])
        return P

def Vstar_fold(G,beta):
    return fold(G,beta)(0.0)

# ---------------------------------------------------------------
# Independent brute-force MDP value (gamma=0)
# control state: 0..q-1 pending, 'J' jam, 'P' pass
# ---------------------------------------------------------------
def brute_Vstar(G, beta, pdict, qdict):
    ctl=controls(G)
    def done(H,st):
        if H[0]=='ctrl': return st[H[1]]=='P'
        if H[0]=='ser': return done(H[1],st) and done(H[2],st)
        return done(H[1],st) or done(H[2],st)
    def frontier(H,st):
        if H[0]=='ctrl':
            v=H[1]; s=st[v]
            return [v] if isinstance(s,int) else []
        if H[0]=='ser':
            return frontier(H[1],st) if not done(H[1],st) else frontier(H[2],st)
        # par
        if done(H[1],st) or done(H[2],st): return []
        return frontier(H[1],st)+frontier(H[2],st)
    @lru_cache(maxsize=None)
    def V(stt):
        st=dict(stt)
        if done(G,st): return 1.0
        f=frontier(G,st)
        if not f: return 0.0
        best=0.0
        for v in f:
            k=st[v]; p=pdict[v][k]; b=beta[v]; q=qdict[v]
            stp=dict(st); stp[v]='P'
            stf=dict(st); stf[v]=('J' if k+1==q else k+1)
            val=b*(p*V(tuple(sorted(stp.items()))) + (1-p)*V(tuple(sorted(stf.items()))))
            best=max(best,val)
        return best
    init=tuple(sorted({v:0 for v in ctl}.items()))
    return V(init)

# ---------------------------------------------------------------
# Reverse-mode gradient via B2-B4
# weight list: list of (s, c)
# ---------------------------------------------------------------
def cumulative(L):
    # returns function C(z)=sum c_j for s_j<=z
    pts=sorted(L)
    def C(z):
        tot=0.0
        for s,c in pts:
            if s<=z+1e-15: tot+=c
        return tot
    return C

def step_to_list(W, bps):
    # W: callable; bps: sorted interior breakpoints in (0,1)
    out=[]
    val0=W(1e-13)  # right limit at 0
    out.append((0.0,val0))
    prev=val0
    for b in sorted(bps):
        if b<=1e-12 or b>=1-1e-12: 
            continue
        valR=W(b+1e-13)
        jump=valR-prev
        if abs(jump)>1e-14:
            out.append((b,jump))
        prev=valR
    return out

def leaf_sens_D0(name, s, ann_ctrl, beta, G):
    # D0(s) via recursion using cached psi and thr
    _,nm,P,psi,thr=ann_ctrl
    # need q, p_list from G ctrl node
    q=G[2]; pl=G[3]; b=beta[name]
    D=0.0
    for k in range(q-1,-1,-1):
        if s < thr[k]-1e-13:   # attack branch
            p=pl[k]
            D = (p + (1-p)*psi[k+1](s)) + b*(1-p)*D
        else:
            D=0.0
    return D

def grad_reverse(G, beta, rho):
    ann={}
    fold(G,beta,ann)
    grad={}
    def go(H, L):
        a=ann[id(H)]
        if a[0]=='ctrl':
            _,name,P,psi,thr=a
            gb=sum(c*leaf_sens_D0(name,s,a,beta,H) for s,c in L)
            grad[name]= -rho*beta[name]*gb
        elif a[0]=='par':
            _,P,P1,P2,G1,G2=a
            C=cumulative(L)
            bpsP1=set(P1.xs); bpsP2=set(P2.xs); bpsL=set(s for s,_ in L)
            # child1 (G1): W1 = g2 * C
            W1=lambda z: P2.rslope(z)*C(z)
            L1=step_to_list(W1, bpsP2|bpsL)
            # child2 (G2): W2 = g1 * C
            W2=lambda z: P1.rslope(z)*C(z)
            L2=step_to_list(W2, bpsP1|bpsL)
            go(G1,L1); go(G2,L2)
        else: # ser
            _,P,P1,P2,G1,G2=a
            def t(s):
                p2=P2(s); return s/p2 if p2>1e-15 else 0.0
            L1=[(t(s), c*P2(s)) for s,c in L]
            L2=[(s, c*(P1(t(s)) - t(s)*P1.rslope(t(s)))) for s,c in L]
            go(G1,L1); go(G2,L2)
    go(G,[(0.0,1.0)])
    return grad

def grad_fd(G, beta_of_l, rho, ell, h=1e-6):
    # finite difference of brute Vstar wrt ell_v (free partials)
    ctl=list(ell.keys())
    pdict={}; qdict={}
    def collect(H):
        if H[0]=='ctrl':
            pdict[H[1]]=H[3]; qdict[H[1]]=H[2]
        else:
            collect(H[1]); collect(H[2])
    collect(G)
    g={}
    for v in ctl:
        ep=dict(ell); ep[v]+=h; em=dict(ell); em[v]-=h
        bp={k:math.exp(-rho*ep[k]) for k in ctl}
        bm={k:math.exp(-rho*em[k]) for k in ctl}
        Vp=brute_Vstar(G,bp,pdict,qdict)
        Vm=brute_Vstar(G,bm,pdict,qdict)
        g[v]=(Vp-Vm)/(2*h)
    return g

# ---------------------------------------------------------------
# Test instances
# ---------------------------------------------------------------
def run(name, G, ell, rho):
    ctl=controls(G)
    beta={v:math.exp(-rho*ell[v]) for v in ctl}
    pdict={}; qdict={}
    def collect(H):
        if H[0]=='ctrl': pdict[H[1]]=H[3]; qdict[H[1]]=H[2]
        else: collect(H[1]); collect(H[2])
    collect(G)
    Vf=Vstar_fold(G,beta)
    Vb=brute_Vstar(G,beta,pdict,qdict)
    gr=grad_reverse(G,beta,rho)
    gf=grad_fd(G,None,rho,ell)
    print(f"=== {name} ===")
    print(f"  V* fold   = {Vf:.10f}")
    print(f"  V* brute  = {Vb:.10f}   |diff|={abs(Vf-Vb):.2e}")
    maxg=0.0
    for v in ctl:
        d=abs(gr[v]-gf[v]); maxg=max(maxg,d)
        print(f"  d V*/d l_{v}: reverse={gr[v]:+.8f}  FD={gf[v]:+.8f}  |diff|={d:.2e}")
    print(f"  --> value match {abs(Vf-Vb)<1e-7},  grad match {maxg<1e-5}")
    print()
    return abs(Vf-Vb)<1e-7 and maxg<1e-5

if __name__ == "__main__":
    rho=1.0
    oks=[]
    # 1. Par of two single-shot
    G1=('par',('ctrl','a',1,[0.5]),('ctrl','b',1,[0.5]))
    oks.append(run("Par(a,b) single-shot symmetric", G1, {'a':0.5,'b':0.5}, rho))
    oks.append(run("Par(a,b) single-shot asymmetric", G1, {'a':0.3,'b':0.7}, rho))

    # 2. Series of two
    G2=('ser',('ctrl','a',2,[0.8,0.6]),('ctrl','b',1,[0.7]))
    oks.append(run("Ser(a,b)", G2, {'a':0.4,'b':0.6}, rho))

    # 3. Nested: Par(Ser(a,b), c)
    G3=('par',('ser',('ctrl','a',1,[0.6]),('ctrl','b',2,[0.5,0.4])),('ctrl','c',1,[0.55]))
    oks.append(run("Par(Ser(a,b),c)", G3, {'a':0.25,'b':0.35,'c':0.40}, rho))

    # 4. Nested other way: Ser(a, Par(b,c))
    G4=('ser',('ctrl','a',2,[0.7,0.5]),('par',('ctrl','b',1,[0.6]),('ctrl','c',1,[0.45])))
    oks.append(run("Ser(a,Par(b,c))", G4, {'a':0.3,'b':0.3,'c':0.4}, rho))

    # 5. deeper nest, different rho
    rho2=2.0
    G5=('par',('ser',('ctrl','a',2,[0.6,0.5]),('par',('ctrl','b',1,[0.7]),('ctrl','c',1,[0.5]))),('ctrl','d',2,[0.4,0.3]))
    oks.append(run("Par(Ser(a,Par(b,c)),d) rho=2", G5, {'a':0.2,'b':0.25,'c':0.25,'d':0.3}, rho2))

    print("ALL TESTS PASS:", all(oks))

    print("="*60)
    print("DIAGNOSTIC: the symmetric Par(a,b) tie point l_a=l_b=0.5")
    print("="*60)
    G=('par',('ctrl','a',1,[0.5]),('ctrl','b',1,[0.5]))
    pdict={'a':[0.5],'b':[0.5]}; qdict={'a':1,'b':1}
    rho=1.0
    def Vb_at(la,lb):
        beta={'a':math.exp(-rho*la),'b':math.exp(-rho*lb)}
        return brute_Vstar(G,beta,pdict,qdict)
    h=1e-6
    la=lb=0.5
    # one-sided partials in l_a (holding l_b)
    right=(Vb_at(la+h,lb)-Vb_at(la,lb))/h
    left =(Vb_at(la,lb)-Vb_at(la-h,lb))/h
    central=(Vb_at(la+h,lb)-Vb_at(la-h,lb))/(2*h)
    beta={'a':math.exp(-rho*0.5),'b':math.exp(-rho*0.5)}
    gr=grad_reverse(G,beta,rho)
    print(f"  reverse-mode  d/dl_a = {gr['a']:+.6f}")
    print(f"  RIGHT deriv          = {right:+.6f}")
    print(f"  LEFT  deriv          = {left:+.6f}")
    print(f"  CENTRAL (FD)         = {central:+.6f}  (= average of one-sided, the FD the test used)")
    lo,hi=min(left,right),max(left,right)
    print(f"  subdifferential interval for d/dl_a : [{lo:+.6f}, {hi:+.6f}]")
    print(f"  reverse-mode value in subdifferential? {lo-1e-9<=gr['a']<=hi+1e-9}")
    print()
    # confirm: nudge OFF the tie -> differentiable -> reverse == FD again
    print("  Just off the tie (l_a=0.49, l_b=0.51): should be differentiable")
    beta2={'a':math.exp(-rho*0.49),'b':math.exp(-rho*0.51)}
    gr2=grad_reverse(G,beta2,rho)
    cf=(Vb_at(0.49+h,0.51)-Vb_at(0.49-h,0.51))/(2*h)
    print(f"    reverse d/dl_a={gr2['a']:+.6f}  central FD={cf:+.6f}  |diff|={abs(gr2['a']-cf):.2e}")
    beta3={'a':math.exp(-rho*0.51),'b':math.exp(-rho*0.49)}
    gr3=grad_reverse(G,beta3,rho)
    cf3=(Vb_at(0.51+h,0.49)-Vb_at(0.51-h,0.49))/(2*h)
    print(f"    (other side l_a=0.51): reverse d/dl_a={gr3['a']:+.6f}  central FD={cf3:+.6f}  |diff|={abs(gr3['a']-cf3):.2e}")

