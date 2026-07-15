#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Exp3 (incumbent-AUC anytime profile). RE-parses the preserved .traj files with the
correct t_end=60 (the runner's inline parse assumes t_end=300). Pairs full(CARE) vs stock within
each (sota,map,N,scen,rep) on:
   auc_delay  = area under the incumbent-DELAY step curve over [first_t, 60]  (lower=better anytime)
   delay@t    = incumbent delay at t in {10,30,60}                            (anytime checkpoints)
 log-ratio l = 100*ln((x_full+1)/(x_stock+1));  l<0 => CARE better (smaller area / smaller delay).
Usage: python3 agg_auc.py [results_dir=results/auc] [traj_dir=results/traj]"""
import json, glob, math, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_util

root = sys.argv[1] if len(sys.argv) > 1 else "results/auc"
tdir = sys.argv[2] if len(sys.argv) > 2 else "results/traj"
CKPT = (10, 30, 60); T_END = 60.0

D = {}; nok = ntraj = 0
for f in glob.glob(root + "/**/*.json", recursive=True):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("status") != "ok": continue
    m = d.get("metrics") or {}
    sd = m.get("sum_dist")
    if sd is None: continue
    nok += 1
    tf = os.path.join(tdir, d["run_id"] + ".traj")
    if not os.path.isfile(tf): continue
    r = traj_util.parse_traj(tf, sd, checkpoints=CKPT, t_end=T_END)
    if not r: continue
    ntraj += 1
    rec = {"auc": r["auc_delay"], "final": r["final_delay"]}
    for c in CKPT: rec[f"d{c}"] = r["delay_at"].get(c)
    D.setdefault((d["sota"], d["map"], d["N"], d["scen"], d.get("rep", 0)), {})[d["arm"]] = rec

def paired(field, host=None):
    xs = []
    for k, v in D.items():
        if "full" not in v or "stock" not in v: continue
        if host and k[0] != host: continue
        a, b = v["full"].get(field), v["stock"].get(field)
        if a is None or b is None or b <= 0 or a <= 0: continue
        xs.append(100 * math.log((a + 1) / (b + 1)))
    if not xs: return None
    med = st.median(xs); better = sum(1 for x in xs if x < 0)
    return dict(n=len(xs), med=med, pct=100*(math.exp(med/100)-1), better=better)

def line(tag, r):
    if r is None: print(f"  {tag:20s} n=0"); return
    print(f"  {tag:20s} n={r['n']:4d}  median l={r['med']:+6.2f} ({r['pct']:+5.1f}%)  "
          f"CARE-better={r['better']:4d}/{r['n']} ({100*r['better']/r['n']:3.0f}%)")

print(f"ok runs={nok}  with-traj={ntraj}  paired cells={len(D)}\n")
print("full(CARE) vs stock  (l<0 => CARE smaller => better anytime):")
line("incumbent-AUC[0,60]", paired("auc"))
for c in CKPT: line(f"delay@{c}s", paired(f"d{c}"))
line("final delay(=60s)", paired("final"))
print("\nincumbent-AUC per host:")
for s in ["lns2", "balance", "address", "tackle", "tackle_alns"]:
    line(s, paired("auc", host=s))
