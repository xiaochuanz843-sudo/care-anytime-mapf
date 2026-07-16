#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the bridge experiment (results/bridge): per (host, map, N, arm) paired-vs-stock
delay effects. Stdlib only. Effect: l = 100*ln((delay_arm+1)/(delay_stock+1)) per scenario,
cell summary = median l, Hodges-Lehmann pseudomedian, exact two-sided sign test, W/T/L.

  python3 experiments/agg_bridge.py            # prints table + writes results/bridge_agg.json
"""
import os, re, json, math, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "bridge")
pat = re.compile(r"^BR\.([a-z_0-9]+)\.(.+)\.N(\d+)\.s(\d+)\.r(\d+)\.([a-z0-9_]+)\.t\d+\.json$")

cells = {}  # (host, map, N) -> scen -> arm -> delay
n_ok = n_fail = 0
for fn in os.listdir(RES):
    m = pat.match(fn)
    if not m:
        continue
    host, mp, N, scen, rep, arm = m.groups()
    try:
        j = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
    except Exception:
        continue
    if j.get("status") != "ok":
        n_fail += 1
        continue
    d = (j.get("metrics") or {}).get("delay")
    if d is None:
        continue
    cells.setdefault((host, mp, int(N)), {}).setdefault(int(scen), {})[arm] = d
    n_ok += 1
print(f"loaded ok={n_ok} fail={n_fail}")

def hl(x):
    w = [(x[i] + x[j]) / 2 for i in range(len(x)) for j in range(i, len(x))]
    w.sort()
    n = len(w)
    return w[n // 2] if n % 2 else (w[n // 2 - 1] + w[n // 2]) / 2

def sign_p(x):
    pos = sum(1 for v in x if v > 0); neg = sum(1 for v in x if v < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

ARMS = ["gce0", "rr5_rand", "rr5_long", "rr5_short", "rr5_mdel", "rr5_ldel",
        "rr5_unif", "rr5_v5", "full"]
out = {}
hdr = f"{'cell':44s}" + "".join(f"{a:>10s}" for a in ARMS)
print(hdr); print("-" * len(hdr))
for key in sorted(cells):
    host, mp, N = key
    scens = cells[key]
    row = {}
    for a in ARMS:
        ls = []
        for sc, arms in scens.items():
            if "stock" in arms and a in arms:
                ls.append(100 * math.log((arms[a] + 1) / (arms["stock"] + 1)))
        if len(ls) >= 5:
            row[a] = {"n": len(ls), "med_l": round(statistics.median(ls), 2),
                      "hl_l": round(hl(ls), 2), "p_sign": round(sign_p(ls), 4),
                      "wins": sum(1 for v in ls if v < 0), "losses": sum(1 for v in ls if v > 0)}
    out[f"{host}.{mp}.N{N}"] = row
    print(f"{host + '.' + mp + '.N' + str(N):44s}" +
          "".join(f"{(str(row[a]['med_l']) if a in row else '--'):>10s}" for a in ARMS))
dst = os.path.join(ROOT, "results", "bridge_agg.json")
json.dump(out, open(dst, "w"), indent=1)
print(f"\nwrote {dst}   (med_l shown; <0 = better than stock; full detail in JSON)")
