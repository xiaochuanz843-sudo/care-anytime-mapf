#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge-experiment config (final binaries, zero code changes): the factorial control
matrix that separates (i) closing the greedy early break, (ii) the acceptance rule, and
(iii) the replanning-order policy — the attribution controls absent from the main battery.

  bridge.json : 4 maps x 2 densities x 5 hosts x 25 scenarios x 10 arms = 10,000 runs @ 60 s.
                maps = warehouse-20-40-10-2-2, Paris_1_256, den520d, random-32-32-20
                densities = {0.6, 1.0} x N*   (same frozen NSTAR ladder as the battery)

Arms (all pure env combinations already supported by every host binary):
  stock      host default (greedy acceptance + early break + random order)
  gce0       ACCEPT=rr, RR_DELTA=0        greedy-complete bridge: closes the early break,
                                          admits only cost-neutral ties (decision-invariance check)
  rr5_rand   ACCEPT=rr, RR_DELTA=5        fixed small band, stock random order
  rr5_long   + REPAIR=20, RB_ARMS=1       fixed longest-haul-first  (the excluded ordering)
  rr5_short  + REPAIR=20, RB_ARMS=2       fixed shortest-first
  rr5_mdel   + REPAIR=20, RB_ARMS=3       fixed most-delayed-first
  rr5_ldel   + REPAIR=20, RB_ARMS=4       fixed least-delayed-first
  rr5_unif   + REPAIR=20, RB_ARMS=0,2,3,4, RB_EPS=1.0   uniform mixture, learning disabled
  rr5_v5     battery arm (fixed band + Thompson portfolio)
  full       battery arm (= CARE, spsa + v5)

Contrasts this buys:  gce0-stock (early-break/tie channel), rr5_rand-gce0 (acceptance rule),
best-fixed vs rr5_unif vs rr5_v5 (is adaptive allocation necessary?), full-rr5_v5 (learned band).

Run on the rented box (low concurrency preferred; results splice via the standard aggregator):
  python3 experiments/gen_bridge.py
  python3 experiments/runner.py experiments/configs/bridge.json
  python3 experiments/agg_bridge.py            # paired-vs-stock table per cell/arm
"""
import json, os
import gen_config_aaai as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "configs"); os.makedirs(OUT, exist_ok=True)

MAPS = ["warehouse-20-40-10-2-2", "Paris_1_256", "den520d", "random-32-32-20"]
FRACS = (0.6, 1.0)

def bridge_arms(sota):
    """Env dicts per bridge arm, derived from the frozen battery arm envs so host-mode and
    telemetry variables (e.g. L2_MKSPAN, ADDRESS RandomWalk) stay identical to the battery."""
    p = G.PFX[sota]
    base = G.arms_for(sota)
    rr5 = dict(base["rr5_v5"]); rr5.pop(p + "REPAIR", None)      # ACCEPT=rr + RR_DELTA=5 + host env
    def with_repair(extra):
        e = dict(rr5); e[p + "REPAIR"] = "20"; e.update({p + k: v for k, v in extra.items()})
        return e
    gce0 = dict(rr5); gce0[p + "RR_DELTA"] = "0"
    return {
        "stock":     base["stock"],
        "gce0":      gce0,
        "rr5_rand":  dict(rr5),
        "rr5_long":  with_repair({"RB_ARMS": "1",       "RB_EPS": "0"}),
        "rr5_short": with_repair({"RB_ARMS": "2",       "RB_EPS": "0"}),
        "rr5_mdel":  with_repair({"RB_ARMS": "3",       "RB_EPS": "0"}),
        "rr5_ldel":  with_repair({"RB_ARMS": "4",       "RB_EPS": "0"}),
        "rr5_unif":  with_repair({"RB_ARMS": "0,2,3,4", "RB_EPS": "1.0"}),
        "rr5_v5":    base["rr5_v5"],
        "full":      base["full"],
    }

COMMON = {"t_limit": G.T_LIMIT, "maxiter": G.MAXITER, "tokens": G.TOKENS, "cli": G.CLI,
          "bin": G.BIN_REL, "mapdir": "data/maps", "scendir": "data/scen-random",
          "run_timeout": G.RUN_TIMEOUT, "heartbeat_s": 60, "keep_csv_cells": [],
          "traj_dir": "results/traj"}

blocks = []
for mp in MAPS:
    for f in FRACS:
        N = G.frac_n(G.NSTAR[mp], f)
        for scen in G.SCENS_OFFICIAL:
            runs = []
            for s in G.SOTAS:
                arms = bridge_arms(s)
                for a, env in arms.items():
                    runs.append(G.mk_run(f"BR.{mp}.N{N}.s{scen}", s, mp, N, scen, a, t=60, env=env))
            blocks.append({"block_id": f"BR.{mp}.N{N}.s{scen}", "phase": "BR", "map": mp,
                           "N": N, "scen": scen, "level": G.LEVEL[mp], "machine": "bridge",
                           "runs": runs})

cfg = {"machine": "bridge", "core_cap": 62, "token_budget": 999,
       "results_dir": "results/bridge", "workdir": "work/bridge",
       "capture_stdout": True, **COMMON, "blocks": blocks}
path = os.path.join(OUT, "bridge.json")
with open(path, "w", newline="\n") as fo:
    json.dump(cfg, fo, indent=1)
nr = sum(len(b["runs"]) for b in blocks)
print(f"bridge.json: blocks={len(blocks)} runs={nr}  (expect 4*2*25=200 blocks, 200*50=10000 runs)")
