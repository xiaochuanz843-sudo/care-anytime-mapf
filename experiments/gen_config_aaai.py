#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified AAAI battery config generator (frozen design; see PREREGISTRATION.md).

Design constants are FROZEN here on purpose (preregistration): editing this file = a design change
and must be logged in PREREGISTRATION.md §7.

Emits (into experiments/configs/):
  lock.json          Batch L  : decontention LOCK, tackle only, LOW concurrency (run FIRST, alone)
  m1.json, m2.json   Batches E0/A/B/G : the fused main battery, split across two machines by a
                     deterministic weighted hash of block_id (both arms of a pair are ALWAYS in the
                     same block => same machine, same worker slot).
  *.json.md5         sidecars for upload integrity (md5sum -c).

All paths inside configs are RELATIVE to the package root; runner.py resolves them against
--root (default: parent of the config file's directory).

Batches (phase field, runner --only selects):
  L  : tackle x 8 protocol cells x {stock, full} x 25 scen x 3 reps, t=60  (lock.json, core_cap=8)
  E0 : noise-floor calibration: 5 hosts x spine-8 @ N* x stock x 25 scen x reps{1,2}, t=60
  A  : official maps: (spine-8 x fracs{.4,.6,.8,1}xN* + breadth-25 @ N*) x 5 hosts x 8 arms x 25
  B  : anytime t=300+traj: protocol-5 x {.6,1}xN* x 5 hosts x 6 traj arms x 25 scen
  G  : generated maps (source=gen:<family>): all maps in data/gen_maps x {N_lo,N_hi} x 5 hosts
       x 6 arms x 8 scen, t=60

Usage:  python3 gen_config_aaai.py [--m1-cores 80] [--m2-cores 64] [--single-machine]
"""
import argparse, hashlib, json, os, sys, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

T_LIMIT = 60
T300 = 300
T300_RUN_TIMEOUT = 660          # runner kill margin for the t=300 layer
RUN_TIMEOUT = 360               # runner kill margin for t=60 runs
SCENS_OFFICIAL = list(range(1, 26))   # the 25 standard MovingAI random scenarios
SCENS_GEN = list(range(1, 9))         # 8 deterministic generated scenarios
SEED_BASE = 100                 # seed = SEED_BASE + scen + 1000*rep
# 1e8: must NEVER bind before the wall-clock budget does. Review finding: at 1e6 fast cells
# (empty-8-8, t=300 random-32) can hit the cap, and greedy stock hits it FIRST (its early-break
# makes iterations cheaper) — a silent bias IN FAVOR of the layer. Audit: any run with
# metrics.iterations >= MAXITER means the cap bound; aggregate flags this.
MAXITER = "100000000"

# ---- official 33-map atlas: map -> N* (feasibility-proven scen-capped ladder, nstar.json) -----
NSTAR = {
    "Berlin_1_256": 1000, "Boston_0_256": 1000, "Paris_1_256": 1000, "brc202d": 700,
    "den312d": 450, "den520d": 1000, "empty-16-16": 128, "empty-32-32": 500,
    "empty-48-48": 1000, "empty-8-8": 32, "ht_chantry": 700, "ht_mansion_n": 700,
    "lak303d": 700, "lt_gallowstemplar_n": 450, "maze-128-128-1": 125,
    "maze-128-128-10": 1000, "maze-128-128-2": 250, "maze-32-32-2": 135,
    "maze-32-32-4": 158, "orz900d": 450, "ost003d": 700, "random-32-32-10": 450,
    "random-32-32-20": 280, "random-64-64-10": 1000, "random-64-64-20": 700,
    "room-32-32-4": 210, "room-64-64-16": 450, "room-64-64-8": 450,
    "w_woundedcoast": 700, "warehouse-10-20-10-2-1": 700, "warehouse-10-20-10-2-2": 1000,
    "warehouse-20-40-10-2-1": 700, "warehouse-20-40-10-2-2": 1000,
}
# bandwidth-token level per map (from the calibrated Phase-B model; used for display/telemetry
# only when token_budget=999 = pure core-cap scheduling)
LEVEL = {m: ("XL" if m == "orz900d" else
             "HEAVY" if m in ("den520d", "warehouse-20-40-10-2-2", "brc202d", "w_woundedcoast",
                              "Boston_0_256", "Berlin_1_256", "Paris_1_256",
                              "warehouse-20-40-10-2-1") else
             "MID" if m in ("lak303d", "ht_mansion_n", "ht_chantry", "maze-128-128-10",
                            "ost003d", "lt_gallowstemplar_n", "maze-128-128-2",
                            "warehouse-10-20-10-2-2", "maze-128-128-1",
                            "warehouse-10-20-10-2-1") else "CACHE")
         for m in NSTAR}
TOKENS = {"CACHE": 0, "MID": 1, "HEAVY": 2, "XL": 4}

PROTOCOL5 = ["random-32-32-20", "warehouse-20-40-10-2-2", "ost003d", "den520d", "Paris_1_256"]
SPINE8 = PROTOCOL5 + ["room-32-32-4", "room-64-64-8", "maze-32-32-4"]
SPINE_FRACS = (0.4, 0.6, 0.8, 1.0)
B_FRACS = (0.6, 1.0)

# ---- LOCK batch: exact decontention cells (comparable to published TACKLE + prior lock run) ---
LOCK_CELLS = [("random-32-32-20", 200), ("random-32-32-20", 280),
              ("warehouse-20-40-10-2-2", 600), ("warehouse-20-40-10-2-2", 1000),
              ("den520d", 600), ("den520d", 1000),
              ("Paris_1_256", 1000), ("ost003d", 700)]
LOCK_REPS = (0, 1, 2)

# ---- hosts ------------------------------------------------------------------------------------
SOTAS = ["lns2", "balance", "address", "tackle", "tackle_alns"]
PFX = {"lns2": "L2_", "balance": "BL_", "address": "AD_", "tackle": "TK_", "tackle_alns": "TK_"}
BIN_REL = {"lns2": "src/lns2/lns", "balance": "src/balance/balance",
           "address": "src/address/address", "tackle": "src/tackle/tackle",
           "tackle_alns": "src/tackle/tackle"}
# CLI templates; placeholders: {bin} {map} {scen} {N} {t} {out} {seed} {maxiter}
CLI = {
    "lns2":    ["{bin}", "-m", "{map}", "-a", "{scen}", "-k", "{N}", "-t", "{t}", "-o", "{out}",
                "--maxIterations={maxiter}", "--neighborSize=8", "--destoryStrategy=Adaptive",
                "--initAlgo=PP", "--seed={seed}", "--screen=0"],
    "balance": ["{bin}", "-m", "{map}", "-a", "{scen}", "-k", "{N}", "-t", "{t}", "-o", "{out}",
                "--maxIterations={maxiter}", "--banditAlgo=Thompson", "--neighborCandidateSizes=5",
                "--screen=0", "--seed={seed}"],
    # ADDRESS: README-exact recipe. --destroyStrategy=RandomWalk is LOAD-BEARING: with the
    # default (Adaptive) + default --b=canonical, the LNS ctor OVERWRITES algorithm to
    # "canonical" and the bernoulie top-K mechanism never runs (LNS.cpp ctor, review finding).
    # Short -k = agentNum; long --k = ADDRESS's top-K (two distinct boost options).
    "address": ["{bin}", "-m", "{map}", "-a", "{scen}", "-k", "{N}", "-t", "{t}", "-o", "{out}",
                "--maxIterations={maxiter}", "--screen=0", "--destroyStrategy=RandomWalk",
                "--algorithm=bernoulie", "--k", "32", "--seed={seed}"],   # K=32 = ADDRESS paper default
    # {k}: runner substitutes min(32, N) — tackle_small() fills topAgents with -1 when N < k
    # and then writes frequency[-1] (heap overflow, review finding).
    "tackle":  ["{bin}", "-m", "{map}", "-a", "{scen}", "--agentNum={N}", "-t", "{t}", "-o", "{out}",
                "--maxIterations={maxiter}", "--destroyStrategy=RandomWalk",
                "--algorithm=tackle-roulette", "--k={k}", "--seed={seed}"],
    # canonical MAPF-LNS ALNS = adaptive roulette over destroy heuristics; the binary's
    # --banditAlgo DEFAULT is "Random" (uniform pick, weights never updated) — must be
    # explicit or the "canonical ALNS" label is wrong (review finding).
    "tackle_alns": ["{bin}", "-m", "{map}", "-a", "{scen}", "--agentNum={N}", "-t", "{t}",
                    "-o", "{out}", "--maxIterations={maxiter}", "--destroyStrategy=Adaptive",
                    "--algorithm=canonical", "--banditAlgo=Roulette", "--seed={seed}"],
}

# ---- arms (fused ablation; see PREREGISTRATION.md §2) ------------------------------------------
def arms_for(sota):
    p = PFX[sota]
    base = {
        "stock":     {},
        "full":      {p + "ACCEPT": "spsa", p + "REPAIR": "v5"},   # THE confirmatory arm
        "spsa_only": {p + "ACCEPT": "spsa"},
        "v5_only":   {p + "REPAIR": "v5"},
        "rr5_v5":    {p + "ACCEPT": "rr", p + "RR_DELTA": "5", p + "REPAIR": "v5"},
        "spsa_20":   {p + "ACCEPT": "spsa", p + "REPAIR": "20"},
        "cart_v5":   {p + "ACCEPT": "cart", p + "CART_D0": "20", p + "REPAIR": "v5"},
        "rr5_20":    {p + "ACCEPT": "rr", p + "RR_DELTA": "5", p + "REPAIR": "20"},
        "rr20_v5":   {p + "ACCEPT": "rr", p + "RR_DELTA": "20", p + "REPAIR": "v5"},
        "rr50_v5":   {p + "ACCEPT": "rr", p + "RR_DELTA": "50", p + "REPAIR": "v5"},
    }
    if sota == "lns2":   # telemetry-only extra CSV column (env-gated in the binary, all arms)
        for env in base.values():
            env["L2_MKSPAN"] = "1"
    return base

ARMS_A = ["stock", "full", "spsa_only", "v5_only", "rr5_v5", "spsa_20", "cart_v5", "rr5_20"]
ARMS_B = ["stock", "full", "rr5_v5", "rr20_v5", "rr50_v5", "spsa_20"]      # all with TRAJ
ARMS_G = ["stock", "full", "spsa_only", "v5_only", "rr5_v5", "spsa_20"]

def frac_n(nstar, f):
    return int(round(f * nstar))

def mk_run(block_id, sota, mp, N, scen, arm, rep=0, t=None, env=None, timeout=None,
           source="official", traj=False):
    seed = SEED_BASE + scen + 1000 * rep
    # PHASE PREFIX in run_id (= result filename): LOCK_CELLS overlap A's protocol maps at
    # the same (sota,map,N,scen,rep,arm) and t=60, so without the phase prefix an L run and an
    # A run share a filename. The full battery keeps them in separate results dirs (results/lock
    # vs results/main), but the prefix makes filenames globally unique so any same-dir merge
    # (e.g. a scaled pre-experiment) stays collision-free. aggregate keys on phase regardless.
    phase = block_id.split(".")[0]
    rid = f"{phase}.{sota}.{mp}.N{N}.s{scen}.r{rep}.{arm}" + (f".t{t}" if t else "")
    run = {"run_id": rid, "block_id": block_id, "sota": sota, "map": mp, "N": N,
           "scen": scen, "rep": rep, "arm": arm, "seed": seed, "source": source,
           "env": arms_for(sota)[arm] if env is None else env}
    if t: run["t"] = t
    if timeout: run["timeout"] = timeout
    if traj: run["traj_env"] = PFX[sota] + "TRAJ"
    return run

def load_gen_maps():
    """Generated-map atlas: MANIFEST (design N) overridden by the feasibility probe output."""
    man_path = os.path.join(PKG, "data", "gen_maps", "MANIFEST.json")
    if not os.path.isfile(man_path):
        print(f"[gen_config] WARN: {man_path} missing -> G batch empty "
              f"(run setup/02_gen_maps.py first)", file=sys.stderr)
        return {}
    man = json.load(open(man_path))["maps"]
    probe_path = os.path.join(HERE, "gen_n_final.json")
    probe = json.load(open(probe_path)) if os.path.isfile(probe_path) else {}
    if os.path.isfile(probe_path) and not probe:
        print("[gen_config] FATAL-ish: gen_n_final.json exists but is EMPTY (probe judged "
              "every map infeasible?) — treating as missing; investigate setup/03_nprobe.py",
              file=sys.stderr)
    if not probe:
        print("[gen_config] WARN: gen_n_final.json missing -> using MANIFEST design N "
              "(run setup/03_nprobe.py on machine 1 for feasibility-laddered N)", file=sys.stderr)
    atlas = {}
    for mp, info in man.items():
        if probe and mp not in probe:
            # probe ran and judged this map infeasible at NFLOOR: disclosed skip, not silent
            print(f"[gen_config] NOTE: {mp} infeasible at probe floor -> excluded from G",
                  file=sys.stderr)
            continue
        n_hi = int(probe.get(mp, {}).get("N_high", info["N_high"]))
        n_lo = int(probe.get(mp, {}).get("N_low", info["N_low"]))
        if n_lo >= n_hi:            # ladder collapsed the contrast: keep hi only
            ns = [n_hi]
        else:
            ns = [n_lo, n_hi]
        atlas[mp] = {"family": info["family"], "Ns": ns}
    return atlas

def mk_blocks():
    blocks = []
    def add(phase, mp, N, scen, runs, level, machine=None, suffix=""):
        blocks.append({"block_id": f"{phase}.{mp}.N{N}.s{scen}{suffix}",
                       "phase": phase, "map": mp, "N": N, "scen": scen,
                       "level": level, "machine": machine, "runs": runs})
    # ---- L lock (tackle, decontention; its own config) ----
    for mp, N in LOCK_CELLS:
        for rep in LOCK_REPS:
            for scen in SCENS_OFFICIAL:
                bid = f"L.{mp}.N{N}.s{scen}.r{rep}"
                runs = [mk_run(bid, "tackle", mp, N, scen, a, rep=rep) for a in ("stock", "full")]
                add("L", mp, N, scen, runs, LEVEL[mp], machine="lock", suffix=f".r{rep}")
    # ---- E0 noise floor: stock replicates over ALL spine maps (protocol-5 + room/maze),
    # so every map that carries a do-no-harm claim gets an own-regime eps (not global fallback) ----
    for mp in SPINE8:
        N = NSTAR[mp]
        for rep in (1, 2):
            for scen in SCENS_OFFICIAL:
                bid = f"E0.{mp}.N{N}.s{scen}.r{rep}"
                runs = [mk_run(bid, s, mp, N, scen, "stock", rep=rep) for s in SOTAS]
                add("E0", mp, N, scen, runs, LEVEL[mp], suffix=f".r{rep}")
    # ---- A official main battery ----
    cells = []
    for mp in SPINE8:
        for f in SPINE_FRACS:
            cells.append((mp, frac_n(NSTAR[mp], f)))
    for mp in NSTAR:
        if mp not in SPINE8:
            cells.append((mp, NSTAR[mp]))
    for mp, N in cells:
        for scen in SCENS_OFFICIAL:
            bid = f"A.{mp}.N{N}.s{scen}"
            runs = [mk_run(bid, s, mp, N, scen, a) for s in SOTAS for a in ARMS_A]
            add("A", mp, N, scen, runs, LEVEL[mp])
    # ---- B anytime t=300 (+traj) ----
    for mp in PROTOCOL5:
        for f in B_FRACS:
            N = frac_n(NSTAR[mp], f)
            for scen in SCENS_OFFICIAL:
                bid = f"B.{mp}.N{N}.s{scen}.t{T300}"
                runs = [mk_run(bid, s, mp, N, scen, a, t=T300, timeout=T300_RUN_TIMEOUT, traj=True)
                        for s in SOTAS for a in ARMS_B]
                add("B", mp, N, scen, runs, LEVEL[mp], suffix=f".t{T300}")
    # ---- G generated maps (fused into the same battery; source column marks provenance) ----
    for mp, info in sorted(load_gen_maps().items()):
        for N in info["Ns"]:
            for scen in SCENS_GEN:
                bid = f"G.{mp}.N{N}.s{scen}"
                runs = [mk_run(bid, s, mp, N, scen, a, source="gen:" + info["family"])
                        for s in SOTAS for a in ARMS_G]
                add("G", mp, N, scen, runs, "MID" if "128" in mp else "CACHE")
    return blocks

def machine_of(block_id, w1):
    """Deterministic weighted split: both arms of a pair share the block => same machine."""
    return "m1" if (zlib.crc32(block_id.encode()) % 10000) < w1 * 10000 else "m2"

def emit(path, cfg):
    with open(path, "w", newline="\n") as f:
        json.dump(cfg, f, indent=1)
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    with open(path + ".md5", "w", newline="\n") as f:
        f.write(f"{digest}  {os.path.basename(path)}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1-cores", type=int, default=80)
    ap.add_argument("--m2-cores", type=int, default=64)
    ap.add_argument("--single-machine", action="store_true",
                    help="emit everything into m1.json (plus lock.json)")
    args = ap.parse_args()
    outdir = os.path.join(HERE, "configs"); os.makedirs(outdir, exist_ok=True)
    blocks = mk_blocks()
    w1 = 1.0 if args.single_machine else args.m1_cores / float(args.m1_cores + args.m2_cores)

    lock_blocks = [b for b in blocks if b["machine"] == "lock"]
    rest = [b for b in blocks if b["machine"] != "lock"]
    for b in rest:
        b["machine"] = machine_of(b["block_id"], w1)

    # NOTE: setup/02_gen_maps.py copies generated maps/scens INTO data/maps + data/scen-random
    # (no name collisions: generated families are g-prefixed); provenance stays in data/gen_maps/.
    common = {"t_limit": T_LIMIT, "maxiter": MAXITER, "tokens": TOKENS, "cli": CLI,
              "bin": BIN_REL, "mapdir": "data/maps", "scendir": "data/scen-random",
              "run_timeout": RUN_TIMEOUT, "heartbeat_s": 60, "keep_csv_cells": [],
              "traj_dir": "results/traj"}
    # LOCK: dedicated low-concurrency config (run alone on machine 1 FIRST)
    emit(os.path.join(outdir, "lock.json"),
         {"machine": "lock", "core_cap": 8, "token_budget": 999,
          "results_dir": "results/lock", "workdir": "work/lock", **common,
          "blocks": lock_blocks})
    caps = {"m1": max(4, args.m1_cores - 8), "m2": max(4, args.m2_cores - 8)}
    for mach in (("m1",) if args.single_machine else ("m1", "m2")):
        mine = [b for b in rest if b["machine"] == mach]
        emit(os.path.join(outdir, f"{mach}.json"),
             {"machine": mach, "core_cap": caps[mach], "token_budget": 999,
              "results_dir": "results/main", "workdir": f"work/{mach}", **common,
              "blocks": mine})
        nr = sum(len(b["runs"]) for b in mine)
        ch = sum((r.get("t") or T_LIMIT) + 6 for b in mine for r in b["runs"]) / 3600.0
        print(f"{mach}: blocks={len(mine)} runs={nr} ~core-h={ch:.0f}")
    nl = sum(len(b["runs"]) for b in lock_blocks)
    print(f"lock: blocks={len(lock_blocks)} runs={nl}")
    tot = sum(len(b["runs"]) for b in blocks)
    by_phase = {}
    for b in blocks:
        by_phase[b["phase"]] = by_phase.get(b["phase"], 0) + len(b["runs"])
    print(f"TOTAL runs={tot}  by_phase={by_phase}")

if __name__ == "__main__":
    main()
