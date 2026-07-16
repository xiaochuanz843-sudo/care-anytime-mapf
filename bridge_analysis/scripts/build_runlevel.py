#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flatten bridge per-run JSONs -> bridge_runlevel.csv.gz, mirroring the column
convention of CARE_handoff/runlevel_all.csv.gz (+ pull_* columns like
runlevel_supplement.csv.gz).  Usage:  py build_runlevel.py <results_dir> <out.csv.gz>"""
import csv
import glob
import gzip
import io
import json
import os
import sys

COLS = ["run_id", "block_id", "machine", "host", "arm", "map", "source", "N", "scen",
        "rep", "seed", "status", "reason", "delay", "soc", "sum_dist_lb", "init_soc",
        "iterations", "auc", "makespan", "runtime", "wall_s", "bin_md5",
        "pull_random", "pull_longest", "pull_shortest", "pull_mdel", "pull_ldel",
        "loadavg_pre", "rss_kb", "t_limit", "env", "finished_at"]


def main(res_dir, out_path):
    files = sorted(glob.glob(os.path.join(res_dir, "*.json")))
    rows, unreadable = [], 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            unreadable += 1
            print(f"[warn] unreadable: {f}", file=sys.stderr)
            continue
        m = r.get("metrics") or {}
        arms = m.get("repair_arms") or [None] * 5
        if len(arms) != 5:
            arms = (list(arms) + [None] * 5)[:5]
        rows.append({
            "run_id": r.get("run_id"), "block_id": r.get("block_id"),
            "machine": r.get("machine"), "host": r.get("sota") or r.get("host"),
            "arm": r.get("arm"), "map": r.get("map"),
            "source": r.get("source", "official"), "N": r.get("N"),
            "scen": r.get("scen"), "rep": r.get("rep"), "seed": r.get("seed"),
            "status": r.get("status"), "reason": r.get("reason"),
            "delay": m.get("delay"), "soc": m.get("soc"),
            "sum_dist_lb": m.get("sum_dist"), "init_soc": m.get("init_soc"),
            "iterations": m.get("iterations"), "auc": m.get("auc"),
            "makespan": m.get("makespan"), "runtime": m.get("runtime"),
            "wall_s": r.get("wall_s"), "bin_md5": r.get("bin_md5"),
            "pull_random": arms[0], "pull_longest": arms[1], "pull_shortest": arms[2],
            "pull_mdel": arms[3], "pull_ldel": arms[4],
            "loadavg_pre": r.get("loadavg_pre"), "rss_kb": r.get("rss_kb"),
            "t_limit": r.get("t_limit"),
            "env": json.dumps(r.get("env") or {}, sort_keys=True),
            "finished_at": r.get("finished_at"),
        })
    rows.sort(key=lambda x: (str(x["host"]), str(x["map"]), int(x["N"] or 0),
                             int(x["scen"] or 0), str(x["arm"])))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as fo:
        fo.write(buf.getvalue())
    print(f"wrote {out_path}: {len(rows)} rows ({unreadable} unreadable skipped)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
