"""
estimate_nary_runtime.py
========================

ANALYTICAL estimate (no fold execution, no re-running) of the exact-gradient
runtime if the defender used the NATIVE n-ary fold instead of the binary
right-fold. It rescales the ALREADY-MEASURED binary runtimes in exp2/runtimes.csv
by a structure-only cost ratio.

Why this is a sound estimate
----------------------------
1. Native n-ary computes the IDENTICAL value and gradient as the binary fold
   (verified to machine precision), so it drives the SAME regret-matching
   trajectory and converges in the SAME number of iterations. Total runtime
   therefore scales purely with the PER-ITERATION fold cost.
2. A fold's per-iteration cost is ~ the sum over internal nodes of the node's
   subtree breakpoint count (each merge reprocesses its children's accumulated
   breakpoints, bounded by the subtree's total lockout Q). That equals

        work = sum over internal nodes of subtree_Q(node)
             = sum over leaves of  q_leaf * depth(leaf).

   Flattening associative Series/Series and Par/Par chains into n-ary nodes
   REDUCES leaf depth, so nary_work <= binary_work, and both are computable from
   the structure string + per-control lockouts alone.

Estimate:  est_nary_runtime = measured_binary_runtime * (nary_work / binary_work)
           est_speedup      = binary_work / nary_work.

Output CSV columns:
  source, network_id, n_controls, total_Q, binary_work, nary_work,
  work_speedup, runtime_exact_s, iters_exact, est_nary_runtime_s
"""

from __future__ import annotations

import argparse
import csv
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


# --- parse a binary "Ser(...)/Par(...)/name" structure into a nested tuple ---
# node := ("c", name) | ("S", [child, child]) | ("P", [child, child])
def parse_structure(text):
    pos = 0

    def parse():
        nonlocal pos
        head = text[pos:pos + 4]
        if head in ("Ser(", "Par("):
            op = "S" if head[:3] == "Ser" else "P"
            pos += 4
            left = parse()
            assert text[pos] == ","
            pos += 1
            right = parse()
            assert text[pos] == ")"
            pos += 1
            return (op, [left, right])
        start = pos
        while pos < len(text) and text[pos] not in ",)":
            pos += 1
        return ("c", text[start:pos])

    tree = parse()
    assert pos == len(text), f"trailing input: {text[pos:]!r}"
    return tree


def flatten(node):
    """Collapse nested same-op nodes into n-ary Series/Par (associativity)."""
    if node[0] == "c":
        return node
    op, children = node
    merged = []
    for child in (flatten(c) for c in children):
        if child[0] == op:
            merged.extend(child[1])
        else:
            merged.append(child)
    return (op, merged)


def fold_work(node, lockout):
    """Return (work, subtree_Q). `work` = sum over internal nodes of subtree_Q --
    the per-iteration fold cost proxy (breakpoint reprocessing)."""
    if node[0] == "c":
        return 0, lockout[node[1]]
    work = 0
    subtree_q = 0
    for child in node[1]:
        child_work, child_q = fold_work(child, lockout)
        work += child_work
        subtree_q += child_q
    work += subtree_q                 # this node's merge reprocesses subtree_Q breakpoints
    return work, subtree_q


def load_structures(csv_path):
    """{network_id: (structure, {control: lockout})}."""
    out = {}
    for r in csv.DictReader(open(csv_path, newline="")):
        probs = json.loads(r["probs"]) if r.get("probs") else {}
        lockout = {name: len(ps) for name, ps in probs.items()}
        out[r["network_id"]] = (r["structure"], lockout)
    return out


OUT_FIELDS = ["source", "network_id", "n_controls", "total_Q",
              "binary_work", "nary_work", "work_speedup",
              "runtime_exact_s", "iters_exact", "est_nary_runtime_s"]


def run(args):
    synth = load_structures(os.path.join(_ROOT, "exp1", "synth_nets_random_ell.csv"))
    real = load_structures(os.path.join(_ROOT, "exp1", "real_world_random_ell.csv"))

    with open(args.runtimes, newline="") as handle:
        rt_rows = list(csv.DictReader(handle))

    work_cache = {}   # (dataset, network_id) -> (binary_work, nary_work)

    def works(dataset, nid):
        key = (dataset, nid)
        if key not in work_cache:
            table = real if dataset == "real_world" else synth
            structure, lockout = table[nid]
            binary_tree = parse_structure(structure)
            bw, _ = fold_work(binary_tree, lockout)
            nw, _ = fold_work(flatten(binary_tree), lockout)
            work_cache[key] = (bw, nw)
        return work_cache[key]

    out_rows = []
    for r in rt_rows:
        src = r["source"]
        dataset = "real_world" if src.startswith("rw_") else "synthetic"
        nid = r["network_id"]
        table = real if dataset == "real_world" else synth
        if nid not in table:
            continue
        bw, nw = works(dataset, nid)
        rt_ex = (r.get("runtime_exact_s") or "").strip()
        est = f"{float(rt_ex) * nw / bw:.4f}" if rt_ex and bw else ""
        out_rows.append({
            "source": src, "network_id": nid,
            "n_controls": r.get("n_controls", ""), "total_Q": r.get("total_Q", ""),
            "binary_work": bw, "nary_work": nw,
            "work_speedup": f"{bw / nw:.4f}" if nw else "",
            "runtime_exact_s": rt_ex,
            "iters_exact": r.get("iters_exact", ""),
            "est_nary_runtime_s": est,
        })

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    # brief console summary of where n-ary is projected to help
    speedups = sorted(((float(r["work_speedup"]), r) for r in out_rows if r["work_speedup"]),
                      key=lambda x: -x[0])
    print(f"wrote {args.out}  ({len(out_rows)} rows)")
    if speedups:
        import statistics as st
        vals = [s for s, _ in speedups]
        print(f"work_speedup (binary/nary): median={st.median(vals):.3f}  max={max(vals):.3f}")
        print("top structural speedups:")
        seen = set()
        for s, r in speedups:
            k = (r["source"].startswith("rw_"), r["network_id"])
            if k in seen:
                continue
            seen.add(k)
            print(f"  {r['source']:<11} net{r['network_id']:>3} n={r['n_controls']:>3} "
                  f"Q={r['total_Q']:>3}  speedup={s:.2f}x  "
                  f"binary_rt={r['runtime_exact_s']}s -> est_nary={r['est_nary_runtime_s']}s")
            if len(seen) >= 6:
                break


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runtimes", default=os.path.join(_HERE, "runtimes.csv"),
                   help="measured binary runtimes (exp2/runtimes.csv from collect_runtimes.py)")
    p.add_argument("--out", default=os.path.join(_HERE, "nary_runtime_estimates.csv"),
                   help="output CSV path")
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
