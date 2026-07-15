#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Exp1 (SA/TA vs RRT). Paired within each (sota,map,N,scen,rep) cell on final delay.
 log-ratio l = 100*ln((delay_a+1)/(delay_b+1));  l<0 => arm a shorter delay => a >= b.
Reports median l, %-a-better, and a one-sided sign-test p (normal approx, exact for small n).
Usage: python3 agg_sata.py [results_dir=results/sata]"""
import json, glob, math, statistics as st, sys

root = sys.argv[1] if len(sys.argv) > 1 else "results/sata"
D = {}; nok = nfail = 0
for f in glob.glob(root + "/**/*.json", recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("status") != "ok": nfail += 1; continue
    dl = (d.get("metrics") or {}).get("delay")
    if dl is None: continue
    nok += 1
    D.setdefault((d["sota"], d["map"], d["N"], d["scen"], d.get("rep", 0)), {})[d["arm"]] = dl

def sign_p(pos, neg):
    """one-sided sign-test p that a is better (more negatives=a-shorter than positives)."""
    n = pos + neg
    if n == 0: return float("nan")
    k = min(pos, neg)                       # smaller tail
    if n < 60:                              # exact binomial two-tail-ish (report one side)
        p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    else:
        z = (k - n / 2.0) / math.sqrt(n / 4.0)
        p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return p

def cmp(a, b, host=None):
    xs = [100 * math.log((v[a] + 1) / (v[b] + 1)) for k, v in D.items()
          if a in v and b in v and v[b] > 0 and v[a] > 0 and (host is None or k[0] == host)]
    n = len(xs)
    if not n: return None
    med = st.median(xs)
    a_better = sum(1 for x in xs if x < 0); a_worse = sum(1 for x in xs if x > 0)
    p = sign_p(a_better, a_worse)           # a_better = a-shorter = negatives
    return dict(n=n, med=med, pct=100 * (math.exp(med / 100) - 1),
                a_better=a_better, a_worse=a_worse, p=p)

def line(tag, r):
    if r is None: print(f"  {tag:22s} n=0"); return
    print(f"  {tag:22s} n={r['n']:5d}  median l={r['med']:+6.2f} ({r['pct']:+5.1f}%)  "
          f"a-shorter={r['a_better']:5d}/{r['n']} ({100*r['a_better']/r['n']:3.0f}%)  sign-p={r['p']:.2e}")

print(f"ok runs={nok}  failed={nfail}  paired cells={len(D)}\n")
print("HEADLINE  (a vs b; a-shorter% high & median<0 => a >= b on delay):")
line("rr5_v5(RRT) vs sa", cmp("rr5_v5", "sa"))
line("rr5_v5(RRT) vs ta", cmp("rr5_v5", "ta"))
line("full(CARE)  vs sa", cmp("full", "sa"))
line("full(CARE)  vs ta", cmp("full", "ta"))
line("full(CARE)  vs rr5_v5", cmp("full", "rr5_v5"))
line("sa          vs ta", cmp("sa", "ta"))
print("\nT0/tau0 ROBUSTNESS  (RRT vs tuned SA/TA; RRT-shorter% => not a default-T0 artifact):")
for arm in ("sa5", "sa50", "ta5", "ta50"):
    line(f"rr5_v5(RRT) vs {arm}", cmp("rr5_v5", arm))
print("\nPER-HOST  rr5_v5(RRT) vs sa:")
for s in ["lns2", "balance", "address", "tackle", "tackle_alns"]:
    line(s, cmp("rr5_v5", "sa", host=s))
print("\nPER-HOST  rr5_v5(RRT) vs ta:")
for s in ["lns2", "balance", "address", "tackle", "tackle_alns"]:
    line(s, cmp("rr5_v5", "ta", host=s))
