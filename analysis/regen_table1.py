#!/usr/bin/env python3
"""Regenerate main Table 1 (hierarchical bootstrap) and the global non-inferiority
lower bound from run-level data.

Inputs : runlevel_all.csv.gz  (146,660-row run-level table; uses batch=A, arms
         stock/full, status=ok)
         percell CSV or A_full_permap.json for the per-cell ub95 flags
         (non-inferiority lower bound); pass --percell to point at it.

Table 1 statistic (per host and pooled):
  point   : median of the pooled scenario-paired log effects
            l = 100*ln((delay_CARE+1)/(delay_stock+1)) over all complete pairs
            (6,914 official pairs; pairing unit (host,map,N,scen)).
  interval: hierarchical bootstrap, B=2000, seed=20260716. Each replicate
            resamples (i) the 33 official maps with replacement, (ii) within
            each sampled map its cells (map,N combinations), (iii) within each
            cell its scenario pairs; the replicate statistic is the median of
            the pooled resampled l. Percentile 2.5/97.5.
  display : bt = (1 - exp(l/100)) * 100 (true % reduction).

Non-inferiority lower bound:
  proportion of the 285 official cells with ub95(l) <= 100*ln(1.03) (per-cell
  BCa upper bounds precomputed in the released per-cell table); cluster
  bootstrap resamples MAPS (B=5000, seed=20260716), one-sided 95% lower bound
  = 5th percentile of the resampled proportion.

Usage: python3 regen_table1.py path/to/runlevel_all.csv.gz path/to/A_full_permap.json
"""
import csv
import gzip
import json
import math
import sys
from collections import defaultdict

import numpy as np

SEED = 20260716
B_TABLE = 2000
B_MARGIN = 5000
MARGIN_L = 100.0 * math.log(1.03)
HOSTS = ["lns2", "balance", "address", "tackle", "tackle_alns"]
LABEL = {"lns2": "MAPF-LNS2", "balance": "BALANCE", "address": "ADDRESS",
         "tackle": "TACKLE", "tackle_alns": "MAPF-LNS (ALNS)"}


def bt(l):
    return (1.0 - math.exp(l / 100.0)) * 100.0


def load_pairs(runlevel_path):
    """host -> map -> cell(N) -> np.array of paired l (scenario level)."""
    raw = defaultdict(dict)
    with gzip.open(runlevel_path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["batch"] != "A" or r["arm"] not in ("stock", "full"):
                continue
            if r["status"] != "ok" or not r["delay"]:
                continue
            raw[(r["host"], r["map"], r["N"], r["scen"])][r["arm"]] = float(r["delay"])
    tree = {h: defaultdict(lambda: defaultdict(list)) for h in HOSTS}
    for (h, mp, N, sc), d in raw.items():
        if "stock" in d and "full" in d:
            l = 100.0 * math.log((d["full"] + 1.0) / (d["stock"] + 1.0))
            tree[h][mp][N].append(l)
    return {h: {mp: {N: np.asarray(v) for N, v in cells.items()}
                for mp, cells in maps.items()} for h, maps in tree.items()}


def hier_boot(maps_dict, rng, B):
    """maps_dict: map -> cell -> np.array(l). Returns (point, lo, hi) of the pooled
    scenario-pair median under the map->cell->pair hierarchical bootstrap."""
    map_names = sorted(maps_dict)
    pooled = np.concatenate([maps_dict[m][c] for m in map_names for c in maps_dict[m]])
    point = float(np.median(pooled))
    reps = np.empty(B)
    for b in range(B):
        chunks = []
        for m in rng.choice(map_names, size=len(map_names), replace=True):
            cells = sorted(maps_dict[m])
            for c in rng.choice(len(cells), size=len(cells), replace=True):
                arr = maps_dict[m][cells[c]]
                chunks.append(arr[rng.integers(0, len(arr), size=len(arr))])
        reps[b] = np.median(np.concatenate(chunks))
    return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def margin_bound(percell_path, rng):
    cells = json.load(open(percell_path, encoding="utf-8"))
    per_map = defaultdict(list)   # map -> [pass(0/1) per cell, across hosts]
    n = ok = 0
    for h in HOSTS:
        for c in cells[h]:
            if c.get("ub95_l") is None:
                continue
            p = 1 if c["ub95_l"] <= MARGIN_L else 0
            per_map[c["map"]].append(p)
            n += 1
            ok += p
    maps = sorted(per_map)
    reps = np.empty(B_MARGIN)
    for b in range(B_MARGIN):
        sel = rng.choice(maps, size=len(maps), replace=True)
        flags = [f for m in sel for f in per_map[m]]
        reps[b] = sum(flags) / len(flags)
    return ok, n, ok / n, float(np.percentile(reps, 5.0))


def main():
    runlevel = sys.argv[1] if len(sys.argv) > 1 else "runlevel_all.csv.gz"
    percell = sys.argv[2] if len(sys.argv) > 2 else "A_full_permap.json"
    rng = np.random.default_rng(SEED)
    tree = load_pairs(runlevel)

    print(f"{'Host':16s} {'median l':>9s} {'95% CI (l)':>20s} {'bt':>6s} {'CI (bt)':>16s}")
    all_maps = defaultdict(lambda: defaultdict(list))
    rows = []
    for h in HOSTS:
        pt, lo, hi = hier_boot(tree[h], rng, B_TABLE)
        rows.append((LABEL[h], pt, lo, hi))
        print(f"{LABEL[h]:16s} {pt:9.2f} [{lo:8.2f},{hi:8.2f}] {bt(pt):5.1f}% "
              f"[{bt(hi):5.1f},{bt(lo):5.1f}]%")
        for mp, cells in tree[h].items():
            for N, arr in cells.items():
                all_maps[mp][(h, N)] = arr
    pooled_tree = {mp: {c: np.asarray(v) for c, v in cells.items()}
                   for mp, cells in all_maps.items()}
    pt, lo, hi = hier_boot(pooled_tree, rng, B_TABLE)
    print(f"{'pooled':16s} {pt:9.2f} [{lo:8.2f},{hi:8.2f}] {bt(pt):5.1f}% "
          f"[{bt(hi):5.1f},{bt(lo):5.1f}]%")

    ok, n, prop, lb = margin_bound(percell, rng)
    print(f"\nnon-inferiority (3% margin): {ok}/{n} = {prop:.1%}; "
          f"map-clustered one-sided 95% lower bound = {lb:.1%} (B={B_MARGIN})")


if __name__ == "__main__":
    main()
