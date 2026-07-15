#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-battery config: tiny but FULL-COVERAGE end-to-end check. MUST pass before any battery.

Covers: all 5 hosts x all 10 arms (arm->env toggles), one official easy cell + one official
structured cell, a t=300 traj cell, and (if generated maps are present) one generated cell
per available family. Blocks are split PER HOST (review finding: runs inside a block execute
sequentially in one worker slot, so a single 50-run block would serialize ~50 min; per-host
blocks keep each slot ~10 min and pairing stays within-block). ~35-40 min on 16 cores.

Emits experiments/configs/smoke.json. Validate with:  python3 analysis/check_smoke.py
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_config_aaai import (SOTAS, PFX, CLI, BIN_REL, TOKENS, MAXITER,
                             arms_for, mk_run, load_gen_maps)

SMOKE_CELLS = [("random-32-32-20", 150), ("warehouse-20-40-10-2-2", 600)]
SMOKE_SCENS = (1, 2)
ALL_ARMS = ["stock", "full", "spsa_only", "v5_only", "rr5_v5", "spsa_20",
            "cart_v5", "rr5_20", "rr20_v5", "rr50_v5"]

def main():
    outdir = os.path.join(HERE, "configs"); os.makedirs(outdir, exist_ok=True)
    blocks = []
    def add(phase, bid, mp, N, scen, runs):
        blocks.append({"block_id": bid, "phase": phase, "map": mp, "N": N, "scen": scen,
                       "level": "CACHE", "machine": "smoke", "runs": runs})
    # S1: all hosts x all arms x 2 cells x 2 scens, t=60 — one block per (cell, scen, HOST)
    for mp, N in SMOKE_CELLS:
        for scen in SMOKE_SCENS:
            for s in SOTAS:
                bid = f"S1.{mp}.N{N}.s{scen}.{s}"
                add("S1", bid, mp, N, scen, [mk_run(bid, s, mp, N, scen, a) for a in ALL_ARMS])
    # S2: t=300 traj cell — one block per HOST (3 arms x ~302 s each)
    mp, N = "warehouse-20-40-10-2-2", 600
    for s in SOTAS:
        bid = f"S2.{mp}.N{N}.s1.t300.{s}"
        add("S2", bid, mp, N, 1, [mk_run(bid, s, mp, N, 1, a, t=300, timeout=660, traj=True)
                                  for a in ("stock", "full", "rr50_v5")])
    # S3: one generated cell per family, stock+full — one block per (family, HOST).
    # Cell choice: SMALLEST size first, at N_lo (review finding: N_high of a 128-size map is
    # the hardest cell of the family; the gate should exercise coverage, not gamble on it).
    fams = {}
    for gmp, info in sorted(load_gen_maps().items(), key=lambda kv: (len(kv[0]), kv[0])):
        fams.setdefault(info["family"], (gmp, info["Ns"][0]))
    for fam, (gmp, N) in sorted(fams.items()):
        for s in SOTAS:
            bid = f"S3.{gmp}.N{N}.s1.{s}"
            add("S3", bid, gmp, N, 1, [mk_run(bid, s, gmp, N, 1, a, source="gen:" + fam)
                                       for a in ("stock", "full")])
    cfg = {"machine": "smoke", "core_cap": 16, "token_budget": 999,
           "results_dir": "results/smoke", "workdir": "work/smoke",
           "t_limit": 60, "maxiter": MAXITER, "tokens": TOKENS, "cli": CLI, "bin": BIN_REL,
           "mapdir": "data/maps", "scendir": "data/scen-random",
           "run_timeout": 360, "heartbeat_s": 30, "keep_csv_cells": [],
           "traj_dir": "results/smoke_traj", "blocks": blocks}
    path = os.path.join(outdir, "smoke.json")
    with open(path, "w", newline="\n") as f: json.dump(cfg, f, indent=1)
    with open(path, "rb") as f: digest = hashlib.md5(f.read()).hexdigest()
    with open(path + ".md5", "w", newline="\n") as f:
        f.write(f"{digest}  smoke.json\n")
    nr = sum(len(b["runs"]) for b in blocks)
    print(f"smoke: blocks={len(blocks)} runs={nr} -> {path}")

if __name__ == "__main__":
    main()
