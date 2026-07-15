#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke-battery validator — the mandatory gate before ANY real battery (run_all.sh stage 1).

Reads results/smoke (relative to the package root; --results overrides; --root has the
same semantics as runner.py: default = the package root, here the parent of analysis/).

Checks (HARD checks gate the exit code; WARN checks are printed but never block):
  0  HARD  completeness: every run_id scheduled in configs/smoke.json has a record
           (a partially-executed smoke must not pass the gate)
  1  HARD  every run has status=ok (failures listed with reason)
  2  HARD  env census: every (sota, arm) combo of the smoke design has >=1 ok record AND
           its recorded env == gen_config_aaai.arms_for(sota)[arm] exactly
  3  HARD  metric sanity on every ok run: delay >= 0, iterations > 0,
           runtime <= t_limit * 1.2
  4  WARN  layer direction sanity: on warehouse-20-40-10-2-2 N600 (t=60), per host,
           median(full delay) <= median(stock delay)  (weak directional check only)
  5  HARD  S2 t=300 traj: for stock/full/rr50_v5 on every host: metrics.traj present,
           delay_at carries all 5 checkpoints {60,120,180,240,300} (JSON stringifies the
           keys — handled), |traj.final_delay - metrics.delay| <= 1;
           a None delay_at value (no incumbent yet at that checkpoint) is a WARN
  6  HARD  S3 generated cells (if any): per family, stock+full ok on every host;
           no S3 records at all => SKIP note (generated maps not installed)
  7  WARN  zero-overhead sanity: per S1 cell, median(full iterations) /
           median(stock iterations) within [0.5, 2]
  8  INFO  per-host stock delays printed for eyeball comparison against history

Exit 0 iff all HARD checks pass, else 1.
Stdlib only + numpy-free (medians are computed with statistics.median).
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DEFAULT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG_DEFAULT, "experiments"))
try:
    from gen_config_aaai import SOTAS, arms_for              # the frozen design source
    from gen_config_smoke import ALL_ARMS
except ImportError as e:                                     # keep the gate usable anyway
    print(f"[check_smoke] FATAL: cannot import the frozen design "
          f"(experiments/gen_config_aaai.py / gen_config_smoke.py): {e}", file=sys.stderr)
    sys.exit(1)

CHECKPOINTS = ("60", "120", "180", "240", "300")   # JSON round-trip makes delay_at keys str
SANITY_CELL = ("warehouse-20-40-10-2-2", 600)      # check-4 cell (S1 structured cell)
ITER_RATIO_LO, ITER_RATIO_HI = 0.5, 2.0

RESULTS = []                                        # (level, name, detail)


def report(level, name, detail=""):
    RESULTS.append((level, name, detail))
    print(f"[{level}] {name}" + (f": {detail}" if detail else ""))


def phase_of(rec):
    return str(rec.get("block_id", "")).split(".")[0]


def is_ok(rec):
    return rec.get("status") == "ok" and bool(rec.get("metrics"))


def med(vals):
    return statistics.median(vals) if vals else None


def load(results_dir):
    recs, unreadable = [], 0
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                recs.append(json.load(fh))
        except (OSError, ValueError):
            unreadable += 1
            print(f"[check_smoke] unreadable: {f}", file=sys.stderr)
    return recs, unreadable


def check0_completeness(recs, root):
    """HARD: every run_id scheduled in configs/smoke.json has a record. Without this a
    partially-executed smoke (whole cells missing) still printed PASS (review finding)."""
    cfg_path = os.path.join(root, "experiments", "configs", "smoke.json")
    if not os.path.isfile(cfg_path):
        report("FAIL", "0 completeness", f"{cfg_path} missing (generate it first)")
        return False
    try:
        cfg = json.load(open(cfg_path))
    except (ValueError, OSError) as e:
        report("FAIL", "0 completeness", f"{cfg_path} unreadable: {e}")
        return False
    want = {r["run_id"] for b in cfg["blocks"] for r in b["runs"]}
    have = {r.get("run_id") for r in recs}
    missing = sorted(want - have)
    extra = sorted(have - want)
    if missing:
        report("FAIL", f"0 completeness ({len(missing)}/{len(want)} scheduled runs missing)",
               "; ".join(missing[:8]) + (f" (+{len(missing)-8} more)" if len(missing) > 8 else ""))
        return False
    if extra:
        report("WARN", "0 completeness", f"{len(extra)} records not in smoke.json "
               f"(stale results dir?): {extra[:5]}")
    report("PASS", f"0 completeness ({len(want)} scheduled runs all present)")
    return True


def check1_all_ok(recs):
    bad = [r for r in recs if not is_ok(r)]
    if bad:
        detail = "; ".join(f"{r.get('run_id')} reason={r.get('reason')}" for r in bad[:10])
        more = f" (+{len(bad) - 10} more)" if len(bad) > 10 else ""
        report("FAIL", f"1 all-runs-ok ({len(bad)}/{len(recs)} not ok)", detail + more)
        return False
    report("PASS", f"1 all-runs-ok ({len(recs)} runs)")
    return True


def check2_env_census(recs):
    """Every (sota, arm) of the smoke design covered by >=1 ok record whose recorded env
    equals the frozen arms_for() env exactly (the runner stores the ACTUAL injected env,
    B7) — catches silent arm/env drift between config generations."""
    seen = {}                                       # (sota, arm) -> ok rec
    mismatches = []
    for r in recs:
        if not is_ok(r):
            continue
        key = (r.get("sota"), r.get("arm"))
        seen.setdefault(key, r)
        try:
            expected = arms_for(r["sota"])[r["arm"]]
        except KeyError:
            mismatches.append(f"{r.get('run_id')}: unknown (sota,arm)={key}")
            continue
        if r.get("env") != expected:
            mismatches.append(f"{r.get('run_id')}: env={r.get('env')} != expected={expected}")
    missing = [(s, a) for s in SOTAS for a in ALL_ARMS if (s, a) not in seen]
    ok = not missing and not mismatches
    if missing:
        report("FAIL", f"2 env-census: {len(missing)} (sota,arm) combos with no ok record",
               ", ".join(f"{s}/{a}" for s, a in missing[:15]))
    if mismatches:
        report("FAIL", f"2 env-census: {len(mismatches)} env mismatches",
               "; ".join(mismatches[:5]))
    if ok:
        report("PASS", f"2 env-census ({len(SOTAS) * len(ALL_ARMS)} combos, env exact-match)")
    return ok


def check3_metric_sanity(recs):
    viol = []
    for r in recs:
        if not is_ok(r):
            continue
        m = r["metrics"]
        tl = float(r.get("t_limit") or 60)
        if not (m.get("delay") is not None and m["delay"] >= 0):
            viol.append(f"{r.get('run_id')}: delay={m.get('delay')}")
        if not (m.get("iterations") is not None and m["iterations"] > 0):
            viol.append(f"{r.get('run_id')}: iterations={m.get('iterations')}")
        if not (m.get("runtime") is not None and m["runtime"] <= tl * 1.2):
            viol.append(f"{r.get('run_id')}: runtime={m.get('runtime')} > {tl}*1.2")
    if viol:
        report("FAIL", f"3 metric-sanity ({len(viol)} violations)", "; ".join(viol[:8]))
        return False
    report("PASS", "3 metric-sanity (delay>=0, iterations>0, runtime<=1.2*t)")
    return True


def check4_layer_direction(recs):
    """Weak directional sanity on the structured S1 cell — WARN only (2 scens is far too
    small for inference; a red flag here means 'look before launching', not 'broken')."""
    mp, N = SANITY_CELL
    by = defaultdict(list)                          # (sota, arm) -> delays
    for r in recs:
        if (is_ok(r) and phase_of(r) == "S1" and r.get("map") == mp
                and int(r.get("N", -1)) == N):
            by[(r["sota"], r["arm"])].append(float(r["metrics"]["delay"]))
    checked = warned = 0
    for sota in SOTAS:
        ms, mf = med(by.get((sota, "stock"), [])), med(by.get((sota, "full"), []))
        if ms is None or mf is None:
            continue
        checked += 1
        if mf > ms:
            warned += 1
            report("WARN", f"4 layer-direction {sota}@{mp}.N{N}",
                   f"median full delay {mf:.0f} > stock {ms:.0f} (weak check, not blocking)")
    if checked and not warned:
        report("PASS", f"4 layer-direction (full<=stock median on {mp}.N{N}, "
                       f"{checked} hosts) [warn-only check]")
    elif not checked:
        report("WARN", "4 layer-direction", f"no stock/full pairs found on {mp}.N{N}")
    return True                                     # never blocks


def check5_traj(recs):
    ok = True
    s2 = [r for r in recs if phase_of(r) == "S2"]
    if not s2:
        report("FAIL", "5 S2-traj", "no S2 (t=300 traj) records found")
        return False
    want = {(s, a) for s in SOTAS for a in ("stock", "full", "rr50_v5")}
    seen = set()
    for r in s2:
        key = (r.get("sota"), r.get("arm"))
        seen.add(key)
        if not is_ok(r):
            report("FAIL", f"5 S2-traj {r.get('run_id')}", f"status={r.get('status')} "
                   f"reason={r.get('reason')}")
            ok = False
            continue
        tr = (r["metrics"] or {}).get("traj")
        if not tr or not tr.get("delay_at"):
            report("FAIL", f"5 S2-traj {r.get('run_id')}", "metrics.traj missing/empty")
            ok = False
            continue
        da = {str(k): v for k, v in tr["delay_at"].items()}
        missing = [cp for cp in CHECKPOINTS if cp not in da]
        if missing:
            report("FAIL", f"5 S2-traj {r.get('run_id')}",
                   f"delay_at missing checkpoints {missing}")
            ok = False
        nones = [cp for cp in CHECKPOINTS if cp in da and da.get(cp) is None]
        if nones:
            report("WARN", f"5 S2-traj {r.get('run_id')}",
                   f"no incumbent yet at t={nones} (slow init under load?)")
        fd, d = tr.get("final_delay"), r["metrics"].get("delay")
        if fd is None or d is None or abs(float(fd) - float(d)) > 1.0:
            report("FAIL", f"5 S2-traj {r.get('run_id')}",
                   f"final_delay={fd} vs metrics.delay={d} (|diff|>1)")
            ok = False
    miss = sorted(want - seen)
    if miss:
        report("FAIL", "5 S2-traj coverage", f"missing combos: {miss}")
        ok = False
    if ok:
        report("PASS", f"5 S2-traj ({len(s2)} runs: traj present, 5 checkpoints, "
                       f"final_delay==delay +-1)")
    return ok


def check6_gen(recs):
    s3 = [r for r in recs if phase_of(r) == "S3"]
    if not s3:
        report("PASS", "6 S3-generated", "SKIP - no S3 records (generated maps not "
                                         "installed); G battery must stay empty too")
        return True
    ok = True
    fams = defaultdict(dict)                        # family -> (sota, arm) -> ok?
    for r in s3:
        fam = str(r.get("source", ""))[4:] or "?"
        fams[fam][(r.get("sota"), r.get("arm"))] = is_ok(r)
    for fam, combos in sorted(fams.items()):
        bad = [f"{s}/{a}" for s in SOTAS for a in ("stock", "full")
               if not combos.get((s, a), False)]
        if bad:
            report("FAIL", f"6 S3-generated family={fam}",
                   f"missing/failed: {', '.join(bad)}")
            ok = False
    if ok:
        report("PASS", f"6 S3-generated ({len(fams)} families x {len(SOTAS)} hosts, "
                       f"stock+full all ok)")
    return ok


def check7_iter_ratio(recs):
    """Zero-overhead sanity: the layer must not silently halve/double the iteration
    throughput. WARN only — smoke cells are tiny."""
    by = defaultdict(list)                          # (sota,map,N,arm) -> iters
    for r in recs:
        if is_ok(r) and phase_of(r) == "S1":
            by[(r["sota"], r["map"], int(r["N"]), r["arm"])].append(
                float(r["metrics"]["iterations"]))
    warned = checked = 0
    for (sota, mp, N) in sorted({k[:3] for k in by}):
        ms = med(by.get((sota, mp, N, "stock"), []))
        mf = med(by.get((sota, mp, N, "full"), []))
        if not ms or mf is None:                    # `not ms` also guards ms==0 division
            continue
        checked += 1
        ratio = mf / ms
        if not (ITER_RATIO_LO <= ratio <= ITER_RATIO_HI):
            warned += 1
            report("WARN", f"7 iter-ratio {sota}@{mp}.N{N}",
                   f"full/stock iterations = {ratio:.2f} outside "
                   f"[{ITER_RATIO_LO},{ITER_RATIO_HI}] (not blocking)")
    if checked and not warned:
        report("PASS", f"7 iter-ratio (full/stock within [{ITER_RATIO_LO},"
                       f"{ITER_RATIO_HI}] on {checked} cells) [warn-only check]")
    elif not checked:
        report("WARN", "7 iter-ratio", "no stock/full iteration pairs found")
    return True                                     # never blocks


def check8_print_stock(recs):
    print("\n[INFO] 8 stock delays per host (eyeball vs history):")
    by = defaultdict(dict)                          # (phase,map,N,t) -> sota -> {scen: delay}
    for r in recs:
        if is_ok(r) and r.get("arm") == "stock":
            by[(phase_of(r), r["map"], int(r["N"]), int(r.get("t_limit") or 60))] \
                .setdefault(r["sota"], {})[int(r["scen"])] = float(r["metrics"]["delay"])
    for (ph, mp, N, t), hosts in sorted(by.items()):
        print(f"  {ph} {mp} N{N} t{t}:")
        for sota in SOTAS:
            if sota in hosts:
                vals = [v for _, v in sorted(hosts[sota].items())]
                print(f"    {sota:<12} " +
                      " ".join(f"s{s}={v:.0f}" for s, v in sorted(hosts[sota].items())) +
                      f"   median={med(vals):.0f}")
    print()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=PKG_DEFAULT,
                    help="package root (runner --root semantics); default = parent of "
                         "this script's directory")
    ap.add_argument("--results", default="",
                    help="smoke results dir; default <root>/results/smoke")
    args = ap.parse_args()
    rdir = args.results or os.path.join(args.root, "results", "smoke")
    if not os.path.isdir(rdir):
        print(f"[check_smoke] FATAL: results dir not found: {rdir}", file=sys.stderr)
        sys.exit(1)
    recs, unreadable = load(rdir)
    if not recs:
        print(f"[check_smoke] FATAL: no result JSONs in {rdir}", file=sys.stderr)
        sys.exit(1)
    print(f"[check_smoke] {len(recs)} records from {rdir} "
          f"({unreadable} unreadable)\n")
    hard_ok = True
    hard_ok &= check0_completeness(recs, args.root)
    # checks 1-8 judge ONLY the scheduled set: stale records from an older smoke config
    # (e.g. re-laddered generated-map N) are disclosed by check0 as extras, not re-judged
    cfg_path = os.path.join(args.root, "experiments", "configs", "smoke.json")
    try:
        if os.path.isfile(cfg_path):
            want = {r["run_id"] for bl in json.load(open(cfg_path))["blocks"] for r in bl["runs"]}
            recs = [r for r in recs if r.get("run_id") in want]
    except (ValueError, OSError):
        pass   # check0 already FAILed on an unreadable config; keep all recs for the rest
    hard_ok &= check1_all_ok(recs)
    hard_ok &= check2_env_census(recs)
    hard_ok &= check3_metric_sanity(recs)
    check4_layer_direction(recs)                    # WARN only
    hard_ok &= check5_traj(recs)
    hard_ok &= check6_gen(recs)
    check7_iter_ratio(recs)                         # WARN only
    check8_print_stock(recs)
    if unreadable:
        report("FAIL", "0 readability", f"{unreadable} unreadable result files")
        hard_ok = False
    n_warn = sum(1 for lv, _, _ in RESULTS if lv == "WARN")
    verdict = "PASS" if hard_ok else "FAIL"
    print(f"\n[check_smoke] SMOKE {verdict} "
          f"({sum(1 for lv, _, _ in RESULTS if lv == 'PASS')} pass, "
          f"{sum(1 for lv, _, _ in RESULTS if lv == 'FAIL')} fail, {n_warn} warn)")
    sys.exit(0 if hard_ok else 1)


if __name__ == "__main__":
    main()
