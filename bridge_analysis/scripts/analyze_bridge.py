#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge-experiment analyzer — same statistical protocol as the main battery
(mapf_aaai/analysis/aggregate_aaai.py; its stats core is copied VERBATIM below with
provenance notes, so every number here is produced by the audited machinery).

Design under analysis (experiments/gen_bridge.py):
  4 maps x {0.6,1.0}xN* x 5 hosts x 25 scens x 10 arms = 10,000 runs @ t=60, rep=0.
  Arms: stock, gce0, rr5_rand, rr5_long, rr5_short, rr5_mdel, rr5_ldel, rr5_unif,
        rr5_v5, full.

Outputs (work/):
  integrity.json    file/dedup/completeness/pairing/md5/wall/loadavg audit
  cells.json        every arm vs stock per (host,map,N): delay primary
                    (+ iterations descriptive, + csv_auc descriptive-flagged)
  contrasts.json    C1 gce0-stock (equivalence framing), C2 acceptance rule,
                    C3 ordering policies, C4 learned band, headline full-stock
  pulls.json        repair_arms telemetry: fixed-arm sanity, uniform mixer check,
                    Thompson concentration, full-vs-rr5_v5 allocation distance
  throughput.json   per-arm iteration medians + paired iteration ratios vs stock
  replication.json  bridge full/rr5_v5 vs battery A-batch same-(host,map,N) cells
  summary.txt       printable digest

Usage:  py analyze_bridge.py --results <dir> --out <workdir>
        [--battery <A_allarms.json from the main-battery aggregates>]
"""
import argparse
import glob
import json
import math
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- frozen constants
T_MAIN = 60
BASE_ARM = "stock"
N_RESAMPLES = 9999
SEED_RNG = 20260709          # same rng base as the battery aggregator
Q_FDR = 0.05
EPS_SECONDARY_L = math.log(1.03) * 100.0     # +3% ratio margin (~2.956 log-points)
ARMS = ["stock", "gce0", "rr5_rand", "rr5_long", "rr5_short", "rr5_mdel", "rr5_ldel",
        "rr5_unif", "rr5_v5", "full"]
FIXED_ARMS = ["rr5_long", "rr5_short", "rr5_mdel", "rr5_ldel"]
HOSTS = ["lns2", "balance", "address", "tackle", "tackle_alns"]
MAPS = ["warehouse-20-40-10-2-2", "Paris_1_256", "den520d", "random-32-32-20"]
N_SCENS = 25
PULL_NAMES = ["random", "longest", "shortest", "mdel", "ldel"]
# regime + measured eps95_l from the battery E0 (aggregates/epsilon_table.json)
REGIME_OF = {"warehouse-20-40-10-2-2": "warehouse", "Paris_1_256": "city",
             "den520d": "game", "random-32-32-20": "dispersed"}
EPS_REGIME_L = {"warehouse": 349.7972, "city": 227.4917, "game": 62.5052,
                "dispersed": 17.4998}


# ---------------------------------------------------------------- stats core
# (verbatim from mapf_aaai/analysis/aggregate_aaai.py — the audited battery machinery)
def lr100(y_arm, y_base):
    return (math.log(float(y_arm) + 1.0) - math.log(float(y_base) + 1.0)) * 100.0


def bt(l):
    return None if l is None else (1.0 - math.exp(float(l) / 100.0)) * 100.0


def r4(x):
    if x is None:
        return None
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


def hl(x, axis=-1):
    x = np.asarray(x)
    i, j = np.triu_indices(x.shape[-1])
    return np.median((x[..., i] + x[..., j]) / 2.0, axis=axis)


def wilcoxon_less(l, exact_below=26):
    l = np.asarray(l, dtype=float)
    if l.size < 2 or np.allclose(l, 0.0):
        return 1.0, "degenerate"
    if l.size < exact_below:
        try:
            return float(stats.wilcoxon(l, alternative="less", zero_method="pratt",
                                        method="exact").pvalue), "exact"
        except (ValueError, TypeError):
            pass
    return float(stats.wilcoxon(l, alternative="less", zero_method="pratt",
                                method="approx").pvalue), "approx"


def wilcoxon_two(l, exact_below=26):
    """Two-sided variant (same pratt/exact/fallback conventions) — used for the
    equivalence-framed C1 contrast where no direction is hypothesized."""
    l = np.asarray(l, dtype=float)
    if l.size < 2 or np.allclose(l, 0.0):
        return 1.0, "degenerate"
    if l.size < exact_below:
        try:
            return float(stats.wilcoxon(l, alternative="two-sided", zero_method="pratt",
                                        method="exact").pvalue), "exact"
        except (ValueError, TypeError):
            pass
    return float(stats.wilcoxon(l, alternative="two-sided", zero_method="pratt",
                                method="approx").pvalue), "approx"


def bca_ci(l, tag, level=0.95):
    l = np.asarray(l, dtype=float)
    if l.size == 0:
        return None, None, "empty"
    if l.size == 1 or np.all(l == l[0]):
        v = float(hl(l))
        return v, v, "percentile_fallback"
    rng = np.random.default_rng([SEED_RNG, zlib.crc32(tag.encode())])
    try:
        bs = stats.bootstrap((l,), hl, confidence_level=level, method="BCa",
                             n_resamples=N_RESAMPLES, vectorized=True, random_state=rng)
        lo = float(bs.confidence_interval.low)
        hi = float(bs.confidence_interval.high)
        if math.isfinite(lo) and math.isfinite(hi):
            return lo, hi, ""
    except Exception:
        pass
    rng = np.random.default_rng([SEED_RNG + 1, zlib.crc32(tag.encode())])
    try:
        bs = stats.bootstrap((l,), hl, confidence_level=level, method="percentile",
                             n_resamples=N_RESAMPLES, vectorized=True, random_state=rng)
        lo = float(bs.confidence_interval.low)
        hi = float(bs.confidence_interval.high)
    except Exception:
        lo = hi = float("nan")
    if not (math.isfinite(lo) and math.isfinite(hi)):
        v = float(hl(l))
        lo = hi = v
    return lo, hi, "percentile_fallback"


def ub95(l, tag):
    _, hi, flag = bca_ci(l, tag + ".ub", level=0.90)
    return hi, flag


def bh_fdr(pvals, q=Q_FDR):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return [], []
    order = np.argsort(p)
    adj = p[order] * m / (np.arange(m) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return [float(x) for x in out], [bool(x <= q) for x in out]


def cell_stats(l, tag, with_tests=True, two_sided=False):
    """Per-cell paired stats on the l-scale — same fields as the battery cell_stats
    (aggregate_aaai.py), with an optional two-sided test for equivalence contrasts."""
    n = int(l.size)
    d = {"n_eff": n, "hl_l": None, "ci_l": [None, None], "ub95_l": None,
         "bt_hl": None, "bt_ci": [None, None],
         "wins": 0, "ties": 0, "losses": 0, "direction": None,
         "p_sup": None, "wilcoxon_method": None, "flags": []}
    if n == 0:
        d["flags"] = ["empty"]
        return d
    hl_l = float(hl(l))
    lo, hi, f1 = bca_ci(l, tag, 0.95)
    ub, f2 = ub95(l, tag)
    flags = sorted({f for f in (f1, f2) if f})
    if n < 5:
        flags.append("low_n")
    d.update({"hl_l": r4(hl_l), "ci_l": [r4(lo), r4(hi)], "ub95_l": r4(ub),
              "bt_hl": r4(bt(hl_l)), "bt_ci": [r4(bt(hi)), r4(bt(lo))],
              "wins": int((l < 0).sum()), "ties": int((l == 0).sum()),
              "losses": int((l > 0).sum()),
              "direction": "improved" if hl_l < 0 else ("worse" if hl_l > 0 else "zero"),
              "flags": flags})
    if with_tests and n >= 5:
        p, meth = (wilcoxon_two(l) if two_sided else wilcoxon_less(l))
        d["p_sup"] = float(p)
        d["wilcoxon_method"] = meth
    return d


# ---------------------------------------------------------------- loading (battery dedup rules)
def load_runs(res_dir):
    groups = defaultdict(list)
    n_files = unreadable = 0
    for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
        n_files += 1
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            print(f"[warn] unreadable result: {f}", file=sys.stderr)
            unreadable += 1
            continue
        try:
            key = (r["sota"], r["map"], int(r["N"]), int(r["scen"]),
                   int(r["rep"]), r["arm"])
        except (KeyError, TypeError, ValueError):
            print(f"[warn] malformed result: {f}", file=sys.stderr)
            unreadable += 1
            continue
        try:
            mt = os.path.getmtime(f)
        except OSError:
            mt = 0.0
        groups[key].append((mt, f, r))

    def is_ok(r):
        return r.get("status") == "ok" and bool(r.get("metrics"))

    def sig(r):
        return json.dumps({"seed": r.get("seed"), "metrics": r.get("metrics")},
                          sort_keys=True)

    records, conflicts = [], []
    benign = superseded = 0
    for key, lst in sorted(groups.items()):
        oks = [(m, p, r) for (m, p, r) in lst if is_ok(r)]
        fls = [(m, p, r) for (m, p, r) in lst if not is_ok(r)]
        if len(oks) >= 2:
            s0 = sig(oks[0][2])
            if all(sig(r) == s0 for _, _, r in oks[1:]):
                benign += len(oks) - 1
            else:
                conflicts.append((key, [(r.get("run_id"), p) for _, p, r in oks]))
                continue
        if oks:
            records.append(oks[0][2])
            superseded += len(fls)
        elif fls:
            records.append(max(fls, key=lambda x: x[0])[2])
    if conflicts:
        print(f"[FATAL] {len(conflicts)} pairing-key conflicts:", file=sys.stderr)
        for key, pairs in conflicts:
            print("  " + ".".join(map(str, key)) + " -> " + str(pairs), file=sys.stderr)
        sys.exit(1)
    report = {"files_seen": n_files, "unreadable": unreadable,
              "records_kept": len(records), "benign_ok_duplicates": benign,
              "superseded_fail_records": superseded}
    return records, report


def index_cells(records):
    cells = defaultdict(lambda: defaultdict(lambda: {"ok": {}, "fail": {}}))
    for r in records:
        key = (r["sota"], r["map"], int(r["N"]))
        slot = ("ok" if (r.get("status") == "ok" and r.get("metrics")) else "fail")
        cells[key][r["arm"]][slot][int(r["scen"])] = r
    return cells


def paired_ls(cell_arms, arm_a, arm_b, metric="delay"):
    """l_i = lr100(y_a, y_b) over scens where BOTH ok. (arm_b is the baseline.)"""
    a = cell_arms[arm_a]["ok"] if arm_a in cell_arms else {}
    b = cell_arms[arm_b]["ok"] if arm_b in cell_arms else {}
    common = sorted(set(a) & set(b))
    dropped = len(set(a) | set(b)) - len(common)
    l = np.array([lr100(a[k]["metrics"][metric], b[k]["metrics"][metric])
                  for k in common
                  if a[k]["metrics"].get(metric) is not None
                  and b[k]["metrics"].get(metric) is not None], dtype=float)
    return l, common, dropped


# ---------------------------------------------------------------- integrity
def build_integrity(records, load_report, cells):
    md5 = defaultdict(set)
    seeds = defaultdict(dict)            # (host,map,N,scen) -> arm -> seed
    wall = defaultdict(list)
    loadavg = []
    fails = []
    per_slot = defaultdict(int)          # (host,map,N,arm) -> ok count
    for r in records:
        h = r["sota"]
        md5[h].add(r.get("bin_md5"))
        seeds[(h, r["map"], int(r["N"]), int(r["scen"]))][r["arm"]] = r.get("seed")
        if r.get("wall_s") is not None:
            wall[h].append(float(r["wall_s"]))
        la = r.get("loadavg_pre")
        if isinstance(la, (list, tuple)) and la:
            loadavg.append(float(la[0]))
        elif isinstance(la, (int, float)):
            loadavg.append(float(la))
        if r.get("status") != "ok":
            fails.append({"run_id": r.get("run_id"), "host": h, "map": r["map"],
                          "N": r["N"], "scen": r["scen"], "arm": r["arm"],
                          "reason": r.get("reason")})
        else:
            per_slot[(h, r["map"], int(r["N"]), r["arm"])] += 1
    seed_viol = []
    for k, arm_seeds in sorted(seeds.items()):
        if len(set(arm_seeds.values())) > 1:
            seed_viol.append({"key": list(k), "seeds": arm_seeds})
    incomplete = {f"{h}.{mp}.N{N}.{arm}": c
                  for (h, mp, N, arm), c in sorted(per_slot.items()) if c != N_SCENS}
    exp_cells = len(HOSTS) * len(MAPS) * 2
    return {
        "load": load_report,
        "expected": {"runs": exp_cells * len(ARMS) * N_SCENS, "cells": exp_cells,
                     "arms": len(ARMS), "scens_per_cell": N_SCENS},
        "n_fail": len(fails), "fail_ledger": fails,
        "bin_md5_per_host": {h: sorted(x for x in s if x) for h, s in sorted(md5.items())},
        "md5_unique_per_host": all(len(s) == 1 for s in md5.values()),
        "seed_pairing_violations": seed_viol,
        "cells_seen": len(cells),
        "incomplete_arm_slots": incomplete,
        "wall_s": {h: {"median": r4(np.median(v)), "p95": r4(np.percentile(v, 95)),
                       "max": r4(max(v))} for h, v in sorted(wall.items())},
        "loadavg_pre": ({"median": r4(np.median(loadavg)), "p95": r4(np.percentile(loadavg, 95)),
                         "max": r4(max(loadavg))} if loadavg else None),
    }


# ---------------------------------------------------------------- cells (all arms vs stock)
def build_cells(cells):
    rows = []
    for key, arms in sorted(cells.items()):
        host, mp, N = key
        regime = REGIME_OF.get(mp, "other")
        for arm in ARMS:
            if arm == BASE_ARM or arm not in arms:
                continue
            for metric, note in (("delay", "primary"),
                                 ("iterations", "descriptive_mechanism"),
                                 ("auc", "descriptive_csv_auc_biased_vs_uphill_arms")):
                l, common, dropped = paired_ls(arms, arm, BASE_ARM, metric)
                tag = f"BR.{host}.{mp}.N{N}.{arm}.{metric}"
                st = cell_stats(l, tag, with_tests=(metric == "delay"))
                rows.append({"host": host, "map": mp, "N": N, "regime": regime,
                             "arm": arm, "baseline": BASE_ARM, "metric": metric,
                             "metric_role": note, "dropped_pairs": dropped, **st})
    # BH-FDR per (host, metric=delay) family, descriptive star counts
    fam = defaultdict(list)
    for r in rows:
        if r["metric"] == "delay" and r["p_sup"] is not None:
            fam[r["host"]].append(r)
    for host, frows in sorted(fam.items()):
        padj, rej = bh_fdr([r["p_sup"] for r in frows])
        for r, pa, rj in zip(frows, padj, rej):
            r["p_bh"], r["sig_bh"] = r4(pa), bool(rj)
    return rows


# ---------------------------------------------------------------- contrasts C1-C4
CONTRASTS = (
    # name, arm_a, arm_b, hypothesis/framing, two_sided
    ("C1_earlybreak_channel", "gce0", "stock",
     "closing the greedy early break + admitting cost-neutral ties; Prop-1 predicts ~0 "
     "(equivalence vs +/-3% and +/-eps_regime)", True),
    ("C2_acceptance_rule", "rr5_rand", "gce0",
     "fixed return-band delta=5 on top of the closed break (the pure acceptance-rule "
     "effect, order held at stock random)", False),
    ("C2b_acceptance_vs_stock", "rr5_rand", "stock",
     "fixed band vs untouched stock (the deployable single-knob acceptance arm)", False),
    ("C3_long_vs_unif", "rr5_long", "rr5_unif", "excluded fixed ordering vs uniform mixture", False),
    ("C3_short_vs_unif", "rr5_short", "rr5_unif", "fixed shortest-first vs uniform mixture", False),
    ("C3_mdel_vs_unif", "rr5_mdel", "rr5_unif", "fixed most-delayed-first vs uniform mixture", False),
    ("C3_ldel_vs_unif", "rr5_ldel", "rr5_unif", "fixed least-delayed-first vs uniform mixture", False),
    ("C3_unif_vs_v5", "rr5_unif", "rr5_v5",
     "uniform mixture vs Thompson portfolio (is ADAPTIVE allocation necessary, "
     "beyond mere mixing?)", False),
    ("C3_long_vs_v5", "rr5_long", "rr5_v5", "excluded ordering vs Thompson", False),
    ("C3_short_vs_v5", "rr5_short", "rr5_v5", "fixed shortest vs Thompson", False),
    ("C3_mdel_vs_v5", "rr5_mdel", "rr5_v5", "fixed most-delayed vs Thompson", False),
    ("C3_ldel_vs_v5", "rr5_ldel", "rr5_v5", "fixed least-delayed vs Thompson", False),
    ("C3_v5_gain_on_rr5", "rr5_v5", "rr5_rand",
     "Thompson repair portfolio added on top of the fixed band (repair-side gain, "
     "acceptance held fixed)", False),
    ("C4_learned_band", "full", "rr5_v5",
     "spsa-learned band vs fixed delta=5 (repair held at v5) — battery "
     "learned_vs_fixed_accept replication on final binaries", False),
    ("H_full_vs_stock", "full", "stock", "headline CARE vs stock replication", False),
    ("H_rr5v5_vs_stock", "rr5_v5", "stock", "fixed-band + Thompson vs stock", False),
)


def build_contrasts(cells):
    out = {}
    for name, arm_a, arm_b, hyp, two in CONTRASTS:
        crows = []
        for key, arms in sorted(cells.items()):
            host, mp, N = key
            if arm_a not in arms or arm_b not in arms:
                continue
            l, common, dropped = paired_ls(arms, arm_a, arm_b, "delay")
            tag = f"ctr.{name}.{host}.{mp}.N{N}"
            st = cell_stats(l, tag, with_tests=True, two_sided=two)
            row = {"host": host, "map": mp, "N": N, "regime": REGIME_OF.get(mp, "other"),
                   "contrast": f"{arm_a} vs {arm_b}", "dropped_pairs": dropped, **st}
            if name.startswith("C1"):
                eps = EPS_REGIME_L.get(REGIME_OF.get(mp, ""), None)
                ci = st["ci_l"]
                row["equiv_pm3pct"] = (ci[0] is not None and ci[0] >= -EPS_SECONDARY_L
                                       and ci[1] <= EPS_SECONDARY_L)
                row["equiv_pm_eps_regime"] = (eps is not None and ci[0] is not None
                                              and ci[0] >= -eps and ci[1] <= eps)
                row["eps_regime_l"] = eps
            crows.append(row)
        # rollups
        def rollup(rows_sel):
            hls = [r["hl_l"] for r in rows_sel if r["hl_l"] is not None]
            if not hls:
                return None
            better = sum(1 for r in rows_sel if r["hl_l"] is not None and r["hl_l"] < 0)
            worse = sum(1 for r in rows_sel if r["hl_l"] is not None and r["hl_l"] > 0)
            p = (float(stats.binomtest(better, better + worse, 0.5,
                                       alternative="two-sided").pvalue)
                 if better + worse > 0 else None)
            return {"n_cells": len(rows_sel), "median_hl_l": r4(np.median(hls)),
                    "median_bt": r4(bt(float(np.median(hls)))),
                    "cells_better": better, "cells_worse": worse,
                    "cells_zero": len(rows_sel) - better - worse,
                    "binom_p_two_sided": r4(p)}
        by_host = {h: rollup([r for r in crows if r["host"] == h]) for h in HOSTS}
        by_map = {m: rollup([r for r in crows if r["map"] == m]) for m in MAPS}
        by_frac = {}
        ns_by_map = defaultdict(set)
        for r in crows:
            ns_by_map[r["map"]].add(r["N"])
        for label, pick in (("0.6xN*", min), ("1.0xN*", max)):
            sel = [r for r in crows if r["N"] == pick(ns_by_map[r["map"]])]
            by_frac[label] = rollup(sel)
        out[name] = {"hypothesis": hyp, "two_sided": two, "cells": crows,
                     "overall": rollup(crows), "by_host": by_host, "by_map": by_map,
                     "by_frac": by_frac}
    # oracle best-fixed vs rr5_v5: per cell pick the fixed arm with the LOWEST hl vs
    # rr5_v5 (an ORACLE — selection uses the same data, disclosed; upper bound on any
    # fixed-order policy)
    oracle_rows = []
    for key, arms in sorted(cells.items()):
        host, mp, N = key
        best = None
        for fa in FIXED_ARMS:
            if fa not in arms or "rr5_v5" not in arms:
                continue
            l, _, _ = paired_ls(arms, fa, "rr5_v5", "delay")
            if l.size == 0:
                continue
            h_val = float(hl(l))
            if best is None or h_val < best[1]:
                best = (fa, h_val, l)
        if best is None:
            continue
        fa, h_val, l = best
        st = cell_stats(l, f"ctr.oracle.{host}.{mp}.N{N}", with_tests=True)
        oracle_rows.append({"host": host, "map": mp, "N": N,
                            "regime": REGIME_OF.get(mp, "other"),
                            "picked_arm": fa, "contrast": f"{fa} vs rr5_v5", **st})
    hls = [r["hl_l"] for r in oracle_rows if r["hl_l"] is not None]
    out["C3_oracle_fixed_vs_v5"] = {
        "hypothesis": "per-cell ORACLE best fixed ordering vs Thompson (selection on the "
                      "same data; optimistic upper bound for any fixed-order policy)",
        "two_sided": False, "cells": oracle_rows,
        "overall": {"n_cells": len(oracle_rows),
                    "median_hl_l": r4(np.median(hls)) if hls else None,
                    "median_bt": r4(bt(float(np.median(hls)))) if hls else None,
                    "cells_better": sum(1 for v in hls if v < 0),
                    "cells_worse": sum(1 for v in hls if v > 0)},
        "picked_counts": {fa: sum(1 for r in oracle_rows if r["picked_arm"] == fa)
                          for fa in FIXED_ARMS}}
    return out


# ---------------------------------------------------------------- pulls (repair_arms)
def build_pulls(cells):
    per_cell = []
    fixed_expect = {"rr5_long": 1, "rr5_short": 2, "rr5_mdel": 3, "rr5_ldel": 4}
    sanity_viol = []
    for key, arms in sorted(cells.items()):
        host, mp, N = key
        for arm in ("rr5_long", "rr5_short", "rr5_mdel", "rr5_ldel", "rr5_unif",
                    "rr5_v5", "full"):
            if arm not in arms:
                continue
            shares, tops = [], []
            n_with = 0
            for rec in arms[arm]["ok"].values():
                pa = (rec["metrics"] or {}).get("repair_arms")
                if not pa or sum(pa) <= 0:
                    continue
                n_with += 1
                s = np.array(pa, dtype=float) / sum(pa)
                shares.append(s)
                tops.append(int(np.argmax(s)))
                if arm in fixed_expect and s[fixed_expect[arm]] < 0.999:
                    sanity_viol.append({"host": host, "map": mp, "N": N, "arm": arm,
                                        "run_id": rec.get("run_id"),
                                        "shares": [r4(x) for x in s]})
            if not shares:
                continue
            mean_s = np.mean(np.vstack(shares), axis=0)
            row = {"host": host, "map": mp, "N": N, "arm": arm, "n_runs": n_with,
                   "mean_share": {PULL_NAMES[i]: r4(mean_s[i]) for i in range(5)},
                   "top_arm": PULL_NAMES[int(np.argmax(mean_s))],
                   "top_share": r4(float(np.max(mean_s)))}
            if arm == "rr5_unif":
                dev = np.abs(mean_s[[0, 2, 3, 4]] - 0.25)
                row["unif_max_dev_from_025"] = r4(float(dev.max()))
                row["arm1_leak"] = r4(float(mean_s[1]))
            per_cell.append(row)
    # full vs rr5_v5 allocation L1 distance per cell (does the learned band change WHERE
    # repair budget goes?)
    l1 = []
    by = {(r["host"], r["map"], r["N"], r["arm"]): r for r in per_cell}
    for key in sorted(cells):
        host, mp, N = key
        a = by.get((host, mp, N, "full"))
        b = by.get((host, mp, N, "rr5_v5"))
        if a and b:
            va = np.array([a["mean_share"][p] or 0 for p in PULL_NAMES])
            vb = np.array([b["mean_share"][p] or 0 for p in PULL_NAMES])
            l1.append({"host": host, "map": mp, "N": N,
                       "l1_dist": r4(float(np.abs(va - vb).sum())),
                       "top_full": a["top_arm"], "top_rr5v5": b["top_arm"],
                       "same_top": a["top_arm"] == b["top_arm"]})
    return {"per_cell": per_cell, "fixed_arm_sanity_violations": sanity_viol,
            "full_vs_rr5v5_allocation": l1}


# ---------------------------------------------------------------- throughput
def build_throughput(cells):
    rows = []
    for key, arms in sorted(cells.items()):
        host, mp, N = key
        base = arms.get(BASE_ARM, {}).get("ok", {})
        for arm in ARMS:
            if arm not in arms:
                continue
            oks = arms[arm]["ok"]
            iters = [float(r["metrics"]["iterations"]) for r in oks.values()
                     if (r["metrics"] or {}).get("iterations") is not None]
            row = {"host": host, "map": mp, "N": N, "arm": arm,
                   "median_iters": r4(np.median(iters)) if iters else None}
            if arm != BASE_ARM:
                common = sorted(set(base) & set(oks))
                ratios = [float(oks[k]["metrics"]["iterations"])
                          / float(base[k]["metrics"]["iterations"])
                          for k in common
                          if (oks[k]["metrics"] or {}).get("iterations")
                          and (base[k]["metrics"] or {}).get("iterations")]
                row["iter_ratio_vs_stock_median"] = (r4(np.median(ratios))
                                                     if ratios else None)
            rows.append(row)
    # per-arm geometric-median rollup of the ratio
    per_arm = {}
    for arm in ARMS:
        if arm == BASE_ARM:
            continue
        v = [r["iter_ratio_vs_stock_median"] for r in rows
             if r["arm"] == arm and r.get("iter_ratio_vs_stock_median") is not None]
        per_arm[arm] = {"n_cells": len(v), "median_of_cell_medians": r4(np.median(v))
                        if v else None, "min": r4(min(v)) if v else None,
                        "max": r4(max(v)) if v else None}
    return {"per_cell_arm": rows, "per_arm_ratio_rollup": per_arm}


# ---------------------------------------------------------------- replication vs battery
def build_replication(cells, battery_path):
    if not battery_path or not os.path.isfile(battery_path):
        return {"note": "battery A_allarms.json not found; replication skipped"}
    ga = json.load(open(battery_path, encoding="utf-8"))
    rows = []
    for key, arms in sorted(cells.items()):
        host, mp, N = key
        cell_key = f"{mp}.N{N}"
        bat = (ga.get(host) or {}).get(cell_key)
        if not bat:
            continue
        for arm in ("full", "rr5_v5"):
            if arm not in arms or arm not in (bat.get("arms") or {}):
                continue
            l, _, _ = paired_ls(arms, arm, BASE_ARM, "delay")
            if l.size == 0:
                continue
            hl_br = float(hl(l))
            hl_bat = bat["arms"][arm].get("hl_l")
            rows.append({"host": host, "map": mp, "N": N, "arm": arm,
                         "hl_l_bridge": r4(hl_br), "hl_l_battery": hl_bat,
                         "bt_bridge": r4(bt(hl_br)), "bt_battery": r4(bt(hl_bat)),
                         "diff_l": r4(hl_br - hl_bat) if hl_bat is not None else None,
                         "same_sign": (hl_bat is not None
                                       and (hl_br < 0) == (hl_bat < 0))})
    diffs = [abs(r["diff_l"]) for r in rows if r["diff_l"] is not None]
    return {"note": "bridge ran on NSCC 64-core final binaries; battery A cells ran on "
                    "m1/m2 — same (host,map,N,scen,seed) design, so per-cell HL should "
                    "reproduce up to machine noise",
            "cells": rows,
            "n_cells": len(rows),
            "same_sign": sum(1 for r in rows if r["same_sign"]),
            "median_abs_diff_l": r4(np.median(diffs)) if diffs else None,
            "max_abs_diff_l": r4(max(diffs)) if diffs else None}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--battery",
                    default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    records, load_report = load_runs(args.results)
    cells = index_cells(records)
    print(f"records={len(records)} cells={len(cells)}")

    integ = build_integrity(records, load_report, cells)
    dump(args.out, "integrity.json", integ)
    cell_rows = build_cells(cells)
    dump(args.out, "cells.json", cell_rows)
    contrasts = build_contrasts(cells)
    dump(args.out, "contrasts.json", contrasts)
    pulls = build_pulls(cells)
    dump(args.out, "pulls.json", pulls)
    thr = build_throughput(cells)
    dump(args.out, "throughput.json", thr)
    repl = build_replication(cells, args.battery)
    dump(args.out, "replication.json", repl)

    # ---- summary.txt
    lines = []
    lines.append(f"bridge runs kept={load_report['records_kept']} "
                 f"files={load_report['files_seen']} fail={integ['n_fail']} "
                 f"md5_unique={integ['md5_unique_per_host']} "
                 f"seed_violations={len(integ['seed_pairing_violations'])} "
                 f"incomplete_slots={len(integ['incomplete_arm_slots'])}")
    for name, _, _, _, _ in CONTRASTS:
        o = contrasts[name]["overall"]
        if o and o.get("median_hl_l") is not None:
            lines.append(f"{name:28s} med_hl={o['median_hl_l']:+8.2f} lp "
                         f"(bt {o['median_bt']:+6.2f}%)  cells {o['cells_better']}W/"
                         f"{o['cells_worse']}L/{o['cells_zero']}T  p_binom={o['binom_p_two_sided']}")
    o = contrasts["C3_oracle_fixed_vs_v5"]["overall"]
    if o.get("median_hl_l") is not None:
        lines.append(f"{'C3_oracle_fixed_vs_v5':28s} med_hl={o['median_hl_l']:+8.2f} lp "
                     f"(bt {o['median_bt']:+6.2f}%)  cells {o['cells_better']}W/{o['cells_worse']}L "
                     f" picks={contrasts['C3_oracle_fixed_vs_v5']['picked_counts']}")
    txt = "\n".join(lines)
    open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


def dump(out, name, obj):
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print(f"wrote {os.path.join(out, name)}")


if __name__ == "__main__":
    main()
