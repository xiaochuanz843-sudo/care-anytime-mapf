#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feasibility N-probe for generated maps (halving ladder, preregistered rule, no exclusions).

Feasibility is HOST-DEPENDENT (smoke evidence: on perfect mazes lns2 initializes where
address/tackle do not), so a cell is feasible only if ALL FOUR host binaries produce an
initial solution within the battery budget (t=60, scenario 1, the battery's scen-1 seed).
CLI templates are imported from experiments/gen_config_aaai.py (single source of truth);
--maxIterations=1 makes a feasible probe cost ~init-time only. "Feasible" = the CSV's
"solution cost" column > 0 — on init failure the hosts still write a CSV row with
solution cost=0, so runtime>0 is NOT a valid success check.

If infeasible, halve N (floor 10, the floor itself is probed) and re-probe; write the final
feasible (N_low, N_high) per map to experiments/gen_n_final.json. Deterministic rule, applied
uniformly to every map BEFORE any layer arm runs (this ladders the CELL, never drops a map).
Run ONCE, on machine 1; copy the output to machine 2 (see README).

Usage:  python3 setup/03_nprobe.py [--jobs 24] [--t 60]
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PKG, "experiments"))
from gen_config_aaai import CLI, BIN_REL   # frozen CLI templates + binary paths

PROBE_SOTAS = ("lns2", "balance", "address", "tackle")   # tackle_alns = same binary/init
MAPS = os.path.join(PKG, "data", "maps")
SCEN = os.path.join(PKG, "data", "scen-random")
NFLOOR = 10
SEED = 101          # battery scen-1 seed (SEED_BASE + scen + 1000*rep = 100 + 1 + 0)

def one_host_ok(sota, mp, N, t, workdir):
    out = os.path.join(workdir, f"probe_{sota}_{mp}_{N}")
    subst = {"bin": os.path.join(PKG, BIN_REL[sota]),
             "map": os.path.join(MAPS, mp + ".map"),
             "scen": os.path.join(SCEN, mp + "-random-1.scen"),
             "N": str(N), "t": str(t), "out": out, "seed": str(SEED),
             "k": str(min(32, N)), "maxiter": "1"}
    cmd = [tok.format(**subst) for tok in CLI[sota]]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("TK_", "AD_", "BL_", "L2_"))}
    env["OMP_NUM_THREADS"] = "1"
    try:
        subprocess.run(cmd, env=env, timeout=t * 4 + 30, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False
    csv = out + "-LNS.csv"
    okf = False
    if os.path.isfile(csv):
        try:
            rows = open(csv).read().strip().splitlines()
            if len(rows) >= 2:
                hdr = [h.strip().lower() for h in rows[0].split(",")]
                okf = float(rows[-1].split(",")[hdr.index("solution cost")]) > 0
        except (ValueError, IndexError, OSError):
            okf = False
    for suf in ("-LNS.csv", "-initLNS.csv"):
        try: os.remove(out + suf)
        except OSError: pass
    return okf

def feasible(mp, N, t, workdir):
    return all(one_host_ok(s, mp, N, t, workdir) for s in PROBE_SOTAS)

def ladder(mp, N0, t, workdir):
    """Halving ladder that PROBES the floor itself: 15 -> 10 (not 15 -> 7 -> dead)."""
    N = N0
    while True:
        if feasible(mp, N, t, workdir):
            return N
        if N <= NFLOOR:
            return None
        N = max(NFLOOR, N // 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--t", type=int, default=60)
    args = ap.parse_args()
    man = json.load(open(os.path.join(PKG, "data", "gen_maps", "MANIFEST.json")))["maps"]
    workdir = os.path.join(PKG, "work", "nprobe"); os.makedirs(workdir, exist_ok=True)
    for s in PROBE_SOTAS:
        b = os.path.join(PKG, BIN_REL[s])
        if not os.access(b, os.X_OK):
            print(f"FATAL: {b} not built (run setup/01_build_all.sh)", file=sys.stderr)
            sys.exit(1)
    # installation preflight: a missing map/scen must be a FATAL, not a silent "infeasible"
    # exclusion (preregistered no-exclusions)
    miss = [mp for mp in man
            if not (os.path.isfile(os.path.join(MAPS, mp + ".map"))
                    and os.path.isfile(os.path.join(SCEN, mp + "-random-1.scen")))]
    if miss:
        print(f"FATAL: {len(miss)} generated maps not installed into data/maps "
              f"(run setup/02_gen_maps.py --install), e.g. {miss[:5]}", file=sys.stderr)
        sys.exit(1)
    def one(item):
        mp, info = item
        hi = ladder(mp, int(info["N_high"]), args.t, workdir)
        if hi is None:
            return mp, None
        lo0 = min(int(info["N_low"]), hi)
        lo = ladder(mp, lo0, args.t, workdir) or hi
        return mp, {"N_low": min(lo, hi), "N_high": hi,
                    "laddered": hi != int(info["N_high"]) or lo != int(info["N_low"])}
    res, dead = {}, []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for mp, r in ex.map(one, sorted(man.items())):
            if r is None: dead.append(mp)
            else: res[mp] = r
            n = len(res) + len(dead)
            if n % 20 == 0: print(f"  {n}/{len(man)}", flush=True)
    out = os.path.join(PKG, "experiments", "gen_n_final.json")
    tmp = out + ".tmp"
    with open(tmp, "w", newline="\n") as f: json.dump(res, f, indent=1, sort_keys=True)
    os.replace(tmp, out)
    lad = sum(1 for r in res.values() if r["laddered"])
    print(f"[nprobe] OK maps={len(res)} laddered={lad} infeasible_at_floor={dead} -> {out}",
          flush=True)
    if dead:
        print("[nprobe] NOTE: infeasible maps stay OUT of gen_n_final.json and the config "
              "generator will not schedule them; this is disclosed, not silent.", flush=True)

if __name__ == "__main__":
    main()
