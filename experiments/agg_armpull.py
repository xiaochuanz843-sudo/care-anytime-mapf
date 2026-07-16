#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Exp2 (bandit arm-pull frequency). Reads metrics.repair_arms=[n0..n4] captured from
the REPAIR_ARMS stdout line of every REPAIR=v5 run.
 arms: 0=random(anchor) 1=longest-haul(EXCLUDED from screened set) 2=shortest 3=most-delayed 4=least-delayed
Reports the per-run-normalised pull share the bandit sends to the random anchor (arm 0) vs the
three delay/haul-ordered arms (2,3,4), overall and per host, so v5 is shown NOT equivalent to a
fixed rule that merely drops the harmful longest-haul arm.
Usage: python3 agg_armpull.py [results_dir=results/sata]"""
import json, glob, statistics as st, sys

root = sys.argv[1] if len(sys.argv) > 1 else "results/sata"
rows = []     # (host, arm, frac0, frac_useful, frac_longest, total_pulls)
for f in glob.glob(root + "/**/*.json", recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("status") != "ok": continue
    ra = (d.get("metrics") or {}).get("repair_arms")
    if not ra or len(ra) != 5: continue
    tot = sum(ra)
    if tot <= 0: continue
    rows.append((d["sota"], d["arm"], ra[0] / tot, (ra[2] + ra[3] + ra[4]) / tot, ra[1] / tot, tot))

def report(sub, label):
    if not sub: print(f"  {label:24s} n=0"); return
    f0 = st.mean(r[2] for r in sub); fu = st.mean(r[3] for r in sub); fl = st.mean(r[4] for r in sub)
    tp = st.median(r[5] for r in sub)
    print(f"  {label:24s} runs={len(sub):5d}  random(arm0)={100*f0:4.1f}%  "
          f"useful(2/3/4)={100*fu:4.1f}%  longest(arm1)={100*fl:4.1f}%  median_pulls={tp:.0f}")

print(f"arm-pull captured runs={len(rows)}\n")
report(rows, "ALL")
print("\nby host:")
for s in ["lns2", "balance", "address", "tackle", "tackle_alns"]:
    report([r for r in rows if r[0] == s], s)
print("\nby arm (acceptance mode; bandit is the same v5 repair under each):")
for a in ["full", "rr5_v5", "sa", "ta"]:
    report([r for r in rows if r[1] == a], a)
