"""
Certify a returned (sub)gradient by the DEFINITION, using the independent
brute-force MDP value as the reference convex function f = V*_l.

Two tests, both valid at kinks (where finite differences are not a criterion):

  (A) Subgradient inequality (global):  h in d f(l)  iff
          f(y) >= f(l) + <h, y-l>   for all y.
      We sample y around l and report the MINIMUM slack
          slack(y) = f(y) - f(l) - <h, y-l>.
      A valid subgradient has slack >= 0 everywhere; a clearly negative
      minimum is a certificate that h is NOT a subgradient.

  (B) Directional-derivative test (local):  h in d f(l)  iff
          <h, d> <= f'(l; d)   for all directions d,
      where f'(l;d) is the one-sided directional derivative. For convex f the
      forward difference [f(l+t d)-f(l)]/t overestimates f'(l;d) as t->0, so
          <h,d> - forward_diff(t)   must be <= ~0.
      We report the MAXIMUM violation over random directions.
"""
import math, random
from sp_gradients2 import (Control, Series, Parallel, value_and_gradient,
                         brute_force_value, to_binary_tree, collect_controls)

def _f(binary_tree, names, alloc, rho):
    return brute_force_value(binary_tree, {n: math.exp(-rho*alloc[n]) for n in names})

def subgradient_inequality_min_slack(tree, alloc, rho, h, n=4000, radius=0.2, seed=0):
    bt = to_binary_tree(tree); names = [c.name for c in collect_controls(bt)]
    rng = random.Random(seed)
    f_l = _f(bt, names, alloc, rho)
    worst = math.inf
    for _ in range(n):
        y = {nm: max(0.0, alloc[nm] + rng.uniform(-radius, radius)) for nm in names}
        slack = _f(bt, names, y, rho) - f_l - sum(h[nm]*(y[nm]-alloc[nm]) for nm in names)
        worst = min(worst, slack)
    return worst

def directional_max_violation(tree, alloc, rho, h, n=2000, t=1e-6, seed=1):
    bt = to_binary_tree(tree); names = [c.name for c in collect_controls(bt)]
    rng = random.Random(seed)
    f_l = _f(bt, names, alloc, rho)
    worst = -math.inf
    for _ in range(n):
        d = {nm: rng.gauss(0, 1) for nm in names}
        norm = math.sqrt(sum(v*v for v in d.values())) or 1.0
        d = {nm: v/norm for nm, v in d.items()}
        y = {nm: alloc[nm] + t*d[nm] for nm in names}
        forward = (_f(bt, names, y, rho) - f_l)/t
        worst = max(worst, sum(h[nm]*d[nm] for nm in names) - forward)
    return worst

def certify(label, tree, alloc, rho, h=None):
    if h is None:
        _, h = value_and_gradient(tree, alloc, rho)
    slack = subgradient_inequality_min_slack(tree, alloc, rho, h)
    viol  = directional_max_violation(tree, alloc, rho, h)
    ok = (slack >= -1e-7) and (viol <= 1e-5)
    print(f"{label}")
    print(f"    (A) min subgradient-inequality slack = {slack:+.3e}   (need >= ~0)")
    print(f"    (B) max directional violation        = {viol:+.3e}   (need <= ~0)")
    print(f"    => {'VALID subgradient' if ok else 'NOT a subgradient'}\n")
    return ok

if __name__ == "__main__":
    print("="*64)
    print("1) Smooth instance -- our gradient should certify cleanly")
    print("="*64)
    nested = Parallel(
        Series(Control("a",2,[0.5,0.4]), Parallel(Control("b",1,0.6), Control("c",1,0.55))),
        Series(Control("d",1,0.5), Control("e",1,0.45)))
    certify("nested tree, l=(.2,.1,.15,.3,.25)", nested,
            {"a":.2,"b":.1,"c":.15,"d":.3,"e":.25}, 1.0)

    print("="*64)
    print("2) KINK instance: symmetric Par(a,b) at l_a=l_b=0.5 (an index tie)")
    print("="*64)
    tie = Parallel(Control("a",1,0.5), Control("b",1,0.5))
    _, h_ours = value_and_gradient(tie, {"a":0.5,"b":0.5}, 1.0)
    print(f"   our returned subgradient: {{a:{h_ours['a']:.4f}, b:{h_ours['b']:.4f}}}")
    certify("   ours (one-sided derivative)", tie, {"a":0.5,"b":0.5}, 1.0, h_ours)

    print("   Now test a DELIBERATELY WRONG vector (push a-component outside the subdifferential):")
    h_bad = dict(h_ours); h_bad["a"] = h_ours["a"] - 0.30
    print(f"   wrong vector: {{a:{h_bad['a']:.4f}, b:{h_bad['b']:.4f}}}")
    certify("   wrong vector", tie, {"a":0.5,"b":0.5}, 1.0, h_bad)