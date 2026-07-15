#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplement configs for the AAAI rebuttal experiments. All three REUSE the frozen main-battery
design (gen_config_aaai.py): same hosts, CLI, seeds, NSTAR, arm envs, official maps — only the
cell/arm subset changes, so results splice cleanly onto the main tables.

  sata.json : Exp1 (SA/TA vs record-to-record RRT) + Exp2 (bandit arm-pull, via capture_stdout).
              cells  = all 33 official maps @ N*  (+ protocol-5 x {0.6,1.0} T0-robustness sweep)
              hosts  = 5 ;  arms = {full, rr5_v5, sa, ta}  (all REPAIR=v5 -> all emit REPAIR_ARMS)
              t=60, 25 scen.  full=CARE, rr5_v5=pure RRT, sa=Metropolis, ta=threshold-accepting.
  auc.json  : Exp3 (incumbent-AUC anytime profile) — protocol-5 x {0.6,1.0}, {full, stock}+traj,
              t=60, 25 scen.  Unbiased anytime area from the .traj incumbent curve (not the CSV col).

Emitted into experiments/configs/ so runner.py resolves data/ + src/ against the package root
(root = parent of the config file's dir's dir).  Run:  python3 experiments/gen_supp.py
"""
import json, os
import gen_config_aaai as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "configs"); os.makedirs(OUT, exist_ok=True)

COMMON = {"t_limit": G.T_LIMIT, "maxiter": G.MAXITER, "tokens": G.TOKENS, "cli": G.CLI,
          "bin": G.BIN_REL, "mapdir": "data/maps", "scendir": "data/scen-random",
          "run_timeout": G.RUN_TIMEOUT, "heartbeat_s": 60, "keep_csv_cells": [],
          "traj_dir": "results/traj"}

def env_sata(sota, accept, t0=None):
    """rr5_v5's host-correct base (keeps REPAIR=v5 + host-mode/telemetry env, e.g. L2_MKSPAN),
    swap ACCEPT rr->sa/ta, drop RR_DELTA. t0 sets SA_T0 (sa) or TA_TAU0 (ta) for the sweep."""
    e = dict(G.arms_for(sota)["rr5_v5"]); p = G.PFX[sota]
    e[p + "ACCEPT"] = accept
    e.pop(p + "RR_DELTA", None)
    if t0 is not None:
        e[(p + "SA_T0") if accept == "sa" else (p + "TA_TAU0")] = str(t0)
    return e

def block(phase, mp, N, scen, runs, suffix=""):
    return {"block_id": f"{phase}.{mp}.N{N}.s{scen}{suffix}", "phase": phase, "map": mp,
            "N": N, "scen": scen, "level": G.LEVEL[mp], "machine": "supp", "runs": runs}

def emit(name, blocks, extra=None):
    cfg = {"machine": "supp", "core_cap": 62, "token_budget": 999,
           "results_dir": f"results/{name}", "workdir": f"work/{name}", **COMMON, "blocks": blocks}
    if extra: cfg.update(extra)
    path = os.path.join(OUT, name + ".json")
    with open(path, "w", newline="\n") as f: json.dump(cfg, f, indent=1)
    nr = sum(len(b["runs"]) for b in blocks)
    print(f"{name}.json: blocks={len(blocks)} runs={nr}  (results/{name})")
    return nr

# ================= Exp1 (SA/TA) + Exp2 (arm-pull) =================
SATA_ARMS = ["full", "rr5_v5", "sa", "ta"]                 # full=CARE, rr5_v5=pure RRT
T0_SWEEP  = [("sa", 5), ("sa", 50), ("ta", 5), ("ta", 50)]  # RRT must not be a tuned-SA/tau0 artifact
sata = []
for mp in G.NSTAR:                                         # all 33 official maps @ N* (breadth)
    N = G.NSTAR[mp]
    for scen in G.SCENS_OFFICIAL:
        runs = []
        for s in G.SOTAS:
            for a in SATA_ARMS:
                env = None if a in ("full", "rr5_v5") else env_sata(s, a)   # full/rr5_v5 from arms_for
                runs.append(G.mk_run(f"SA.{mp}.N{N}.s{scen}", s, mp, N, scen, a, t=60, env=env))
        sata.append(block("SA", mp, N, scen, runs))
for mp in G.PROTOCOL5:                                     # T0/tau0 robustness sweep (bounded)
    for f in (0.6, 1.0):
        N = G.frac_n(G.NSTAR[mp], f)
        for scen in G.SCENS_OFFICIAL:
            runs = []
            for s in G.SOTAS:
                # rr5_v5 anchor so the sweep arms have a pairing partner at BOTH fracs; its run_id
                # is suffix-independent, so at frac 1.0 it dedups against core sata (skipped), and
                # at frac 0.6 (not in core) it runs once — no double execution either way.
                runs.append(G.mk_run(f"SA.{mp}.N{N}.s{scen}.t0sw", s, mp, N, scen, "rr5_v5", t=60))
                for (acc, t0) in T0_SWEEP:
                    runs.append(G.mk_run(f"SA.{mp}.N{N}.s{scen}.t0sw", s, mp, N, scen,
                                         f"{acc}{t0}", t=60, env=env_sata(s, acc, t0)))
            sata.append(block("SA", mp, N, scen, runs, suffix=".t0sw"))
n_sata = emit("sata", sata, extra={"capture_stdout": True})

# ================= Exp3 (incumbent-AUC) =================
# t=60 to match the A-batch regime the review flagged; phase "B" so runner PRESERVES the .traj
# into traj_dir (results/traj) — agg_auc.py re-parses with the correct t_end=60 (the runner's
# built-in parse assumes t_end=300, which would bury the early-anytime signal under a flat tail).
auc = []
for mp in G.PROTOCOL5:
    for f in (0.6, 1.0):
        N = G.frac_n(G.NSTAR[mp], f)
        for scen in G.SCENS_OFFICIAL:
            runs = [G.mk_run(f"B.{mp}.N{N}.s{scen}", s, mp, N, scen, a, t=60, traj=True)
                    for s in G.SOTAS for a in ("full", "stock")]
            auc.append(block("B", mp, N, scen, runs))
n_auc = emit("auc", auc)

print(f"TOTAL supplement runs = {n_sata + n_auc}  (sata {n_sata} + auc {n_auc})")
