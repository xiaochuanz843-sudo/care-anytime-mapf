#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AAAI battery aggregator — implements PREREGISTRATION.md §6 exactly.

Independent successor of the audited phaseb/aggregate.py (NOT imported); its BCa /
Wilcoxon cores are copied here with the audit defects fixed:

  * nan guard        : a BCa CI that comes back nan (bias/acceleration undefined), or a
                       degenerate all-identical sample, falls back to a percentile CI and
                       is flagged "percentile_fallback". nan NEVER leaks into any output.
  * pairing hard-err : two DIFFERING ok records for one pairing key abort with exit(1),
                       printing the conflicting run_id/path pairs — silent overwrite is
                       forbidden (audit rule). Byte-identical duplicates (re-synced dirs)
                       are deduped with a counted warning. A fail record superseded by a
                       successful retry is dropped and never double-counted (audit trap).
  * n_eff downgrade  : do-no-harm cells with n_eff<5 are skipped (disclosed); 5<=n_eff<10
                       are "cannot_certify" (never counted as pass).

Pairing unit (PREREGISTRATION: "paired within (host, map, N, scenario), same batch"):
    (phase, sota, map, N, scen, rep, t_limit)
  phase (block_id prefix L/E0/A/B/G) MUST be part of the key: the LOCK batch re-runs
  A-batch cells under the SAME run_id namespace at low concurrency; merging L with A
  would pair across concurrency regimes (and collide on run_id).

Effect scale:  l = ln((y_arm+1)/(y_stock+1)) * 100      ("log-points"; l<0 = arm better)
Display scale: bt(l) = (1 - exp(l/100)) * 100           (TRUE % reduction; bt>0 = better)
  Session-36 calibration: raw log-ratios must never be labelled "%"; every table carries
  both l and bt.

Analysis conventions frozen here (each also documented at the implementing function):
  * tests (one-sided exact Wilcoxon, pratt zeros, exact for n<26) are computed for the
    confirmatory arm `full` at t=60 only; ALL other arms/budgets are effect+CI only.
  * BH-FDR (q=.05) per (sota, metric) family over phase-A full cells (descriptive
    star counts); host headline = per-host exact TWO-SIDED binomial over cell directions,
    never pooled across hosts, never a cross-map median.
  * do-no-harm is judged on the l-scale: ub95_l <= eps_measured(regime)  AND
    ub95_l <= ln(1.03)*100 (= +3% ratio, the substantive secondary margin).
  * anytime drift call ("rr worse than stock, certified"): the 95% BCa CI of the
    paired log-ratio lies entirely above 0 (ci_lo > 0). Under the benefit = -l
    convention this is exactly "benefit CI upper bound < 0" as worded in the design.
  * anytime AUC validity: a pair counts as init-VALID only if both sides' initial SoC
    are equal (metrics.init_soc, falling back to the raw .traj first incumbent when
    --traj is given). Unequal -> excluded + counted (auc_invalid_pairs). Unverifiable
    (no init on either side) -> EXCLUDED from the primary AUC statistic + counted
    (auc_unverified_pairs, flagged), per PREREGISTRATION ("valid only where both arms
    share the initial solution").

Inputs : --results DIR (repeatable; per-run JSONs from experiments/runner.py, multi-box
         merge), --traj DIR (optional; raw B-batch .traj files, used only to backfill
         records whose metrics.traj is missing and to verify shared inits), --out DIR.
Outputs: cells.json headline.json dnh.json epsilon.json ablation.json anytime.json
         gen.json recon.json summary.txt   (all under --out)

Requires: numpy, scipy>=1.9 (stats.bootstrap method="BCa", wilcoxon method=...).
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
import scipy
from scipy import stats

# ---------------------------------------------------------------- frozen protocol constants
T_MAIN = 60                                # confirmatory budget (s); B batch is t=300
CHECKPOINTS = (60, 120, 180, 240, 300)     # anytime checkpoints (DESIGN §6)
METRICS = ("delay",)   # ONLY delay is compared per-cell in phase A/L/G. The -LNS.csv
# "area under curve" column integrates the WORKING solution (iteration_stats), which for
# uphill-accepting layer arms (spsa/rr/cart) != the best-so-far incumbent, so a stock-vs-arm
# AUC from it is apples-to-oranges (biased against the layer). The valid anytime AUC is the
# incumbent AUC from phase-B .traj (build_anytime -> anytime.json). CSV auc stays in each
# run record for provenance but is never a headline/confirmatory comparison.
CONF_ARM = "full"                          # THE single confirmatory arm (preregistered)
BASE_ARM = "stock"
Q_FDR = 0.05                               # BH-FDR level (descriptive star counts)
N_RESAMPLES = 9999                         # bootstrap resamples (as in the audited parent)
SEED_RNG = 20260709                        # rng base; per-cell stream derived via crc32(tag)
EPS_SECONDARY_L = math.log(1.03) * 100.0   # +3% ratio on the l-scale (~2.956 log-points)
DNH_N_CERTIFY = 10                         # n_eff >= 10 required to certify (prereg H2)
DNH_N_SKIP = 5                             # n_eff < 5 -> skip (disclosed)
KNOWN_PHASES = ("L", "E0", "A", "B", "G")
DRIFT_ARMS = ("rr20_v5", "rr50_v5")        # fixed-threshold smoking-gun arms (B batch)
SCENS_OFFICIAL = 25                        # MovingAI standard random scens per map
SCENS_GEN = 8                              # frozen generated scens per map
REPS_BY_PHASE = {"L": (0, 1, 2)}           # every other paired phase runs rep 0 only
# mechanism-decomposition contrasts (DESIGN §2 ablation-extraction table); l<0 = full better
ABLATION_CONTRASTS = (
    ("accept_side_gain",        "full", "v5_only"),   # spsa accept added on locked v5 repair
    ("repair_side_gain",        "full", "spsa_only"), # v5 repair added on locked spsa accept
    ("learned_vs_fixed_accept", "full", "rr5_v5"),
    ("learned_vs_fixed_repair", "full", "spsa_20"),
    ("accept_selection_cart",   "full", "cart_v5"),
    ("legacy_config",           "full", "rr5_20"),
)
# regime classification for epsilon (DESIGN §6): official map-name prefixes; gen -> source
REGIME_PREFIX = (
    ("dispersed", ("random", "empty")),
    ("room",      ("room",)),
    ("maze",      ("maze",)),
    ("warehouse", ("warehouse",)),
    ("city",      ("Paris", "Berlin", "Boston")),
    ("game",      ("den", "ost", "orz", "brc", "lak", "ht_", "lt_", "w_")),
)


# ---------------------------------------------------------------- tiny pure helpers
def phase_of(rec):
    """Batch phase = block_id prefix (L/E0/A/B/G; smoke S1..S3 / pilot P are ignored)."""
    return str(rec.get("block_id", "")).split(".")[0]


def lr100(y_arm, y_base):
    """Paired effect l = ln((y_arm+1)/(y_base+1)) * 100 (log-points; l<0 = arm better).
    +1 offset per protocol so delay=0 is representable. DESIGN §6 / PREREGISTRATION."""
    return (math.log(float(y_arm) + 1.0) - math.log(float(y_base) + 1.0)) * 100.0


def bt(l):
    """Display back-transform bt(l) = (1 - exp(l/100)) * 100 = TRUE percent reduction
    (bt>0 = improvement). Monotone DECREASING in l, so CI [lo,hi] maps to [bt(hi), bt(lo)].
    Session-36 calibration fix: log-ratios are never labelled '%'; bt is."""
    return None if l is None else (1.0 - math.exp(float(l) / 100.0)) * 100.0


def r4(x):
    """Round for JSON readability; never passes nan through (audit nan guard)."""
    if x is None:
        return None
    x = float(x)
    return round(x, 4) if math.isfinite(x) else None


def classify_regime(map_name, source="official"):
    """Regime for the epsilon table: gen maps by source family ('gen:<family>'); official
    maps by name prefix (DESIGN §6): dispersed={random,empty}, room, maze, warehouse,
    city={Paris,Berlin,Boston}, game={den,ost,orz,brc,lak,ht_,lt_,w_}. Unknown -> 'other'
    (falls back to the global epsilon downstream)."""
    if source and str(source).startswith("gen:"):
        return str(source)
    for name, prefixes in REGIME_PREFIX:
        if any(map_name.startswith(p) for p in prefixes):
            return name
    return "other"


# ---------------------------------------------------------------- statistics core
def hl(x, axis=-1):
    """Hodges–Lehmann pseudomedian: median of the Walsh averages (x_i+x_j)/2, i<=j.
    Vectorized over the last axis so scipy.stats.bootstrap(vectorized=True) can call it.
    Copied verbatim from the audited phaseb aggregator."""
    x = np.asarray(x)
    i, j = np.triu_indices(x.shape[-1])
    return np.median((x[..., i] + x[..., j]) / 2.0, axis=axis)


def wilcoxon_less(l, exact_below=26):
    """One-sided Wilcoxon signed-rank, H1: location(l) < 0 (arm improves).
    zero_method='pratt' (zero differences keep their rank in the |.| ranking, then drop
    from the signed sum) per protocol; method='exact' when n < 26, else the normal
    approximation. All-zero (or n<2) samples are degenerate -> p=1.0.
    Returns (pvalue, method). Provenance: phaseb aggregate.wilcoxon_less + the n<26
    exact rule from the AAAI spec; the exact->approx fallback (scipy refuses exact for
    some tie/zero patterns) is kept from the audited parent."""
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


def bca_ci(l, tag, level=0.95):
    """Two-sided bootstrap CI of the HL pseudomedian (N_RESAMPLES, BCa), rng derived from
    the cell tag => reproducible and independent of iteration order.
    Degenerate-sample guard (audit fix): all-identical values, or a BCa interval that
    contains nan (bias/acceleration undefined on zero-variance resamples), fall back to a
    percentile CI with flag='percentile_fallback' (for a constant sample the percentile
    CI degenerates to the point itself). nan never leaks. Returns (lo, hi, flag) on the
    SAME scale as l."""
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
    except Exception:                      # scipy raises on some degenerate resample sets
        pass
    rng = np.random.default_rng([SEED_RNG + 1, zlib.crc32(tag.encode())])
    try:
        bs = stats.bootstrap((l,), hl, confidence_level=level, method="percentile",
                             n_resamples=N_RESAMPLES, vectorized=True, random_state=rng)
        lo = float(bs.confidence_interval.low)
        hi = float(bs.confidence_interval.high)
    except Exception:
        lo = hi = float("nan")
    if not (math.isfinite(lo) and math.isfinite(hi)):   # ultimate guard: point interval
        v = float(hl(l))
        lo = hi = v
    return lo, hi, "percentile_fallback"


def ub95(l, tag):
    """One-sided 95% upper confidence bound of the HL pseudomedian = the upper endpoint
    of the two-sided 90% BCa interval (as in the audited parent). Same fallback/flag
    semantics as bca_ci. Returns (ub, flag)."""
    _, hi, flag = bca_ci(l, tag + ".ub", level=0.90)
    return hi, flag


def bh_fdr(pvals, q=Q_FDR):
    """Benjamini–Hochberg step-up FDR: adjusted p_i = min_k>=rank(i) { p_(k) * m / k },
    clipped to 1. Returns (p_adjusted list, rejected(bool at level q) list).
    Used for DESCRIPTIVE per-cell significance counts within a (sota, metric) family and
    within anytime checkpoint families (DESIGN §6)."""
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


# ---------------------------------------------------------------- loading / dedup / indexing
def load_runs(dirs):
    """Load per-run JSONs from all --results dirs and dedupe on the FULL key
    (pairing key + arm) with the audit rules:
      * >=2 ok records, IDENTICAL (seed, metrics)  -> keep one (benign duplicate, counted)
      * >=2 ok records, DIFFERING                  -> CONFLICT: print all run_id/path
                                                      pairs and exit(1) — silent overwrite
                                                      is forbidden (audit hard error)
      * ok + fail(s)                               -> ok wins (a successful retry
                                                      supersedes; the fail is dropped and
                                                      never double-counted — audit trap)
      * only fails                                 -> keep the mtime-newest fail
    Records with a phase outside KNOWN_PHASES (smoke S*, pilot P) are set aside and
    counted, not analyzed. Returns (records, report_dict)."""
    groups = defaultdict(list)     # full key -> [(mtime, path, rec)]
    n_files = unreadable = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            n_files += 1
            try:
                with open(f, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (OSError, ValueError):
                print(f"[warn] unreadable result: {f}", file=sys.stderr)
                unreadable += 1
                continue
            try:
                key = (phase_of(r), r["sota"], r["map"], int(r["N"]), int(r["scen"]),
                       int(r["rep"]), int(r.get("t_limit") or T_MAIN), r["arm"])
            except (KeyError, TypeError, ValueError):
                print(f"[warn] malformed result (missing key fields): {f}", file=sys.stderr)
                unreadable += 1
                continue
            try:
                mt = os.path.getmtime(f)
            except OSError:
                mt = 0.0
            groups[key].append((mt, f, r))

    def is_ok(r):
        return r.get("status") == "ok" and bool(r.get("metrics"))

    def sig(r):    # content signature for benign-duplicate detection
        return json.dumps({"seed": r.get("seed"), "metrics": r.get("metrics")}, sort_keys=True)

    records, conflicts = [], []
    benign = superseded = fail_dups = 0
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
            if len(fls) > 1:
                fail_dups += len(fls) - 1
                print(f"[warn] {len(fls)} fail records for key={key}: keeping mtime-newest "
                      f"(all are failures; analysis unaffected)", file=sys.stderr)
            records.append(max(fls, key=lambda x: x[0])[2])
    if conflicts:
        print(f"[FATAL] {len(conflicts)} pairing-key conflicts — two DIFFERING ok records "
              f"share one (phase,sota,map,N,scen,rep,t,arm) key. Silent overwrite is "
              f"forbidden (audit rule). Conflicts:", file=sys.stderr)
        for key, pairs in conflicts:
            print("  key=" + ".".join(map(str, key)) + ":", file=sys.stderr)
            for rid, p in pairs:
                print(f"    {rid} @ {p}", file=sys.stderr)
        sys.exit(1)
    kept, ignored = [], defaultdict(int)
    for r in records:
        ph = phase_of(r)
        if ph in KNOWN_PHASES:
            kept.append(r)
        else:
            ignored[ph] += 1
    report = {"dirs": list(dirs), "files_seen": n_files, "unreadable": unreadable,
              "records_kept": len(kept), "benign_ok_duplicates": benign,
              "superseded_fail_records": superseded, "fail_duplicates": fail_dups,
              "ignored_phases": dict(sorted(ignored.items()))}
    return kept, report


def index_cells(records):
    """cells[(phase, sota, map, N, t)][arm] = {"ok": {(scen,rep): rec},
                                               "fail": {(scen,rep): rec}}
    One record per slot is guaranteed by load_runs' dedup."""
    cells = defaultdict(lambda: defaultdict(lambda: {"ok": {}, "fail": {}}))
    for r in records:
        key = (phase_of(r), r["sota"], r["map"], int(r["N"]),
               int(r.get("t_limit") or T_MAIN))
        ik = (int(r["scen"]), int(r["rep"]))
        slot = "ok" if (r.get("status") == "ok" and r.get("metrics")) else "fail"
        cells[key][r["arm"]][slot][ik] = r
    return cells


def cell_source(cell_arms):
    """The run 'source' field of a cell (identical across its runs by construction)."""
    for arm in cell_arms.values():
        for slot in ("ok", "fail"):
            for rec in arm[slot].values():
                return rec.get("source", "official")
    return "official"


def paired_ls(cell_arms, arm, metric):
    """Within-instance paired log-ratios l_i = lr100(y_arm, y_stock) over the instances
    (scen,rep) where BOTH arms are status=ok. Returns (np.array l, common keys,
    dropped_pairs = |union| - |intersection| of the two arms' ok instance sets)."""
    base = cell_arms[BASE_ARM]["ok"]
    a = cell_arms[arm]["ok"]
    common = sorted(set(base) & set(a))
    dropped = len(set(base) | set(a)) - len(common)
    l = np.array([lr100(a[k]["metrics"][metric], base[k]["metrics"][metric])
                  for k in common], dtype=float)
    return l, common, dropped


# ---------------------------------------------------------------- epsilon (solver noise floor)
def compute_epsilon(records):
    """eps_measured (DESIGN §6 / H2): the solver-noise floor from stock-vs-stock replicate
    pairs. Sources: phase E0 stock (rep 1,2) + phase A stock (rep 0), same
    (sota,map,N,scen) at t=60 — a triple of replicates per instance (different seeds:
    seed = 100 + scen + 1000*rep). For each instance with the rep set R, all C(|R|,2)
    unordered pairs contribute |l| = |lr100(y_r1, y_r2)| (delay metric, the do-no-harm
    metric). eps(regime) = 95th percentile of the pooled |l| within the regime;
    eps_global = 95th percentile over ALL pairs. A cell whose regime has no calibration
    pairs falls back to eps_global (flagged by the consumer).
    Field note (disclosed): E0 and A blocks may land on different machines (block-hash
    sharding), so some pairs straddle machines — this only INFLATES eps (conservative)."""
    tri = defaultdict(dict)      # (sota,map,N,scen) -> rep -> (delay, source)
    for r in records:
        if r.get("arm") != BASE_ARM or phase_of(r) not in ("E0", "A"):
            continue
        if int(r.get("t_limit") or T_MAIN) != T_MAIN:
            continue
        if r.get("status") != "ok" or not r.get("metrics"):
            continue
        tri[(r["sota"], r["map"], int(r["N"]), int(r["scen"]))][int(r["rep"])] = (
            float(r["metrics"]["delay"]), r.get("source", "official"))
    per_regime = defaultdict(list)
    per_regime_maps = defaultdict(set)
    all_l = []
    for (sota, mp, N, scen), reps in sorted(tri.items()):
        rs = sorted(reps)
        if len(rs) < 2:
            continue
        reg = classify_regime(mp, reps[rs[0]][1])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                al = abs(lr100(reps[rs[i]][0], reps[rs[j]][0]))
                per_regime[reg].append(al)
                per_regime_maps[reg].add(mp)
                all_l.append(al)
    def pct(v):
        return {"eps95_l": r4(np.percentile(v, 95)) if v else None,
                "median_l": r4(np.median(v)) if v else None,
                "n_pairs": len(v)}
    out = {"protocol": "eps95 = 95th pct of |lr100| over stock replicate pairs "
                       "(E0 reps 1,2 + A rep 0, t=60, delay); per regime, "
                       "global fallback for uncovered regimes",
           "global": pct(all_l),
           "regimes": {reg: {**pct(v), "maps": sorted(per_regime_maps[reg])}
                       for reg, v in sorted(per_regime.items())}}
    return out


def eps_for(epsilon, regime):
    """(eps_l, fallback_used) for a regime; regimes without calibration pairs (e.g. room,
    maze, gen:* — E0 only covers protocol-5 maps) fall back to the global eps95."""
    reg = epsilon["regimes"].get(regime)
    if reg and reg.get("eps95_l") is not None:
        return reg["eps95_l"], False
    return epsilon["global"].get("eps95_l"), True


# ---------------------------------------------------------------- per-cell statistics
def cell_stats(l, tag, with_tests):
    """All per-cell paired statistics on the l-scale (DESIGN §6):
      HL pseudomedian, 95% two-sided BCa CI, one-sided 95% upper bound (ub95, from the
      90% two-sided BCa), bt display transforms, win/tie/loss (sign of l; l<0 = win),
      direction, and — for the confirmatory arm at t=60 only — the one-sided exact
      Wilcoxon p (pratt zeros, exact for n<26). Never emits nan; degenerate CIs are
      flagged 'percentile_fallback'; n_eff<5 flagged 'low_n' (excluded from all
      inference downstream)."""
    n = int(l.size)
    d = {"n_eff": n, "hl_l": None, "ci_l": [None, None], "ub95_l": None,
         "bt_hl": None, "bt_ci": [None, None],
         "wins": 0, "ties": 0, "losses": 0, "direction": None,
         "p_sup": None, "wilcoxon_method": None, "p_bh": None, "sig_bh": None,
         "flags": []}
    if n == 0:
        d["flags"] = ["empty"]                 # dead-coverage audit: cell stays visible
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
        p, meth = wilcoxon_less(l)
        d["p_sup"] = float(p)                  # full precision (BH consumes this)
        d["wilcoxon_method"] = meth
    return d


def build_cells(cells):
    """cells.json: every (phase,sota,map,N,t,arm!=stock,metric) cell vs stock.
    Tests only for arm=full at t=60 (preregistration: single confirmatory arm; every
    other arm and every off-protocol budget = effect+CI, no stars). BH-FDR (q=.05) per
    (sota, metric) family over phase-A full cells with n_eff>=5 — the DESCRIPTIVE
    significant-cell counts; L/G full cells carry a raw p only (their claims run through
    lock/gen sections), B cells run through anytime.json."""
    rows = []
    fam = defaultdict(list)          # (sota, metric) -> phase-A full rows for BH
    for key, arms in sorted(cells.items()):
        phase, sota, mp, N, t = key
        if BASE_ARM not in arms:
            continue
        src = cell_source(arms)
        regime = classify_regime(mp, src)
        for arm in sorted(a for a in arms if a != BASE_ARM):
            for metric in METRICS:
                l, common, dropped = paired_ls(arms, arm, metric)
                tag = f"{phase}.{sota}.{mp}.N{N}.t{t}.{arm}.{metric}"
                conf = (arm == CONF_ARM and t == T_MAIN)
                st = cell_stats(l, tag, with_tests=conf)
                row = {"phase": phase, "sota": sota, "map": mp, "N": N, "t": t,
                       "arm": arm, "baseline": BASE_ARM, "metric": metric,
                       "source": src, "regime": regime, "dropped_pairs": dropped,
                       "confirmatory": bool(conf), **st}
                rows.append(row)
                if conf and phase == "A" and st["n_eff"] >= 5:
                    fam[(sota, metric)].append(row)
    for (sota, metric), frows in sorted(fam.items()):
        padj, rej = bh_fdr([r["p_sup"] for r in frows], Q_FDR)
        for r, pa, rj in zip(frows, padj, rej):
            r["p_bh"], r["sig_bh"] = r4(pa), bool(rj)
    return rows


# ---------------------------------------------------------------- headline (H1 + H3 lock)
def build_headline(cell_rows, cells):
    """headline.json (H1): per host, delay only, over phase-A t=60 FULL cells (n_eff>=5):
    exact TWO-SIDED binomial on per-cell HL direction (improved = hl_l<0; exact-zero
    cells excluded from the trial count, disclosed), bt effect median/range, BH-sig
    count. NO cross-host pooling, NO cross-map medians as claims (protocol).
    H3 (lock_decontention): phase-L absolute per-cell median delays for stock and full
    (3 reps x 25 scens) — the against-published-values comparison is done in the paper,
    not here."""
    out = {"protocol": "per-host exact two-sided binomial over full-cell directions "
                       "(phase A, t=60, n_eff>=5); ties excluded and disclosed; "
                       "bt = true % reduction", "hosts": {}, "lock_decontention": []}
    sel = [r for r in cell_rows
           if r["phase"] == "A" and r["arm"] == CONF_ARM and r["t"] == T_MAIN
           and r["n_eff"] >= 5]
    hosts = sorted({r["sota"] for r in sel})
    for sota in hosts:
        out["hosts"][sota] = {}
        for metric in METRICS:
            rows = [r for r in sel if r["sota"] == sota and r["metric"] == metric]
            if not rows:
                continue
            imp = sum(1 for r in rows if r["direction"] == "improved")
            wor = sum(1 for r in rows if r["direction"] == "worse")
            zer = len(rows) - imp - wor
            p = (float(stats.binomtest(imp, imp + wor, 0.5,
                                       alternative="two-sided").pvalue)
                 if imp + wor > 0 else None)
            bts = [r["bt_hl"] for r in rows if r["bt_hl"] is not None]
            out["hosts"][sota][metric] = {
                "cells": len(rows), "improved": imp, "worse": wor, "zero": zer,
                "binom_p_two_sided": r4(p),
                "bt_hl_median": r4(np.median(bts)) if bts else None,
                "bt_hl_range": [r4(min(bts)), r4(max(bts))] if bts else [None, None],
                "bh_sig_cells": sum(1 for r in rows if r.get("sig_bh")),
                "low_n_excluded_note": "cells with n_eff<5 excluded (see cells.json flags)"}
    for key, arms in sorted(cells.items()):
        phase, sota, mp, N, t = key
        if phase != "L":
            continue
        row = {"sota": sota, "map": mp, "N": N, "t": t}
        for arm in (BASE_ARM, CONF_ARM):
            oks = arms.get(arm, {}).get("ok", {})
            vals = [float(r["metrics"]["delay"]) for r in oks.values()]
            row[arm] = {"n_ok": len(vals),
                        "median_delay": r4(np.median(vals)) if vals else None}
        l, common, dropped = (paired_ls(arms, CONF_ARM, "delay")
                              if CONF_ARM in arms and BASE_ARM in arms
                              else (np.array([]), [], 0))
        st = cell_stats(l, f"L.{sota}.{mp}.N{N}.t{t}.full.delay", with_tests=True)
        row["paired_full_vs_stock"] = {k: st[k] for k in
                                       ("n_eff", "hl_l", "ci_l", "bt_hl", "bt_ci",
                                        "p_sup", "wilcoxon_method", "flags")}
        out["lock_decontention"].append(row)
    return out


# ---------------------------------------------------------------- do-no-harm (H2)
def build_dnh(cell_rows, epsilon):
    """dnh.json (H2, confirmatory arm `full` ONLY, metric=delay — the preregistered
    do-no-harm metric; AUC envelopes live in cells.json as descriptive columns).
    Worst-cell criterion per cell, on the l-scale:
        pass  <=>  ub95_l <= eps_measured(regime)   (primary)
              AND  ub95_l <= ln(1.03)*100 (~2.956)  (secondary, substantive +3% margin)
    n_eff gates (audit fix):  n_eff < 5  -> status='skip' (disclosed, never judged);
    5 <= n_eff < 10 -> status='cannot_certify' (NOT a pass, listed separately).
    Every full/delay/t=60 cell (phases A, L, G) is listed — nothing silently vanishes.
    Two overall verdicts are reported honestly:
      all_evaluable_pass  = no cell with n_eff>=10 fails;
      all_certified_pass  = all_evaluable_pass AND no cannot_certify AND no skip."""
    rows = []
    for r in cell_rows:
        if r["arm"] != CONF_ARM or r["metric"] != "delay" or r["t"] != T_MAIN:
            continue
        if r["phase"] not in ("A", "L", "G"):
            continue
        eps, fb = eps_for(epsilon, r["regime"])
        row = {"phase": r["phase"], "sota": r["sota"], "map": r["map"], "N": r["N"],
               "regime": r["regime"], "n_eff": r["n_eff"], "hl_l": r["hl_l"],
               "ub95_l": r["ub95_l"], "bt_hl": r["bt_hl"],
               "eps_l": eps, "eps_global_fallback": bool(fb),
               "eps_secondary_l": r4(EPS_SECONDARY_L), "flags": r["flags"]}
        if r["n_eff"] < DNH_N_SKIP:
            row["status"] = "skip"
        elif r["n_eff"] < DNH_N_CERTIFY:
            row["status"] = "cannot_certify"
        elif r["ub95_l"] is None or eps is None:
            row["status"] = "cannot_certify"
            row["flags"] = sorted(set(row["flags"]) | {"missing_bound_or_eps"})
        else:
            ok_primary = r["ub95_l"] <= eps
            ok_secondary = r["ub95_l"] <= EPS_SECONDARY_L
            row["pass_primary"], row["pass_secondary"] = bool(ok_primary), bool(ok_secondary)
            row["status"] = "pass" if (ok_primary and ok_secondary) else "fail"
        rows.append(row)
    per_host, overall = {}, {"pass": 0, "fail": 0, "cannot_certify": 0, "skip": 0}
    for sota in sorted({r["sota"] for r in rows}):
        hr = [r for r in rows if r["sota"] == sota]
        cnt = {s: sum(1 for r in hr if r["status"] == s)
               for s in ("pass", "fail", "cannot_certify", "skip")}
        judged = [r for r in hr if r["status"] in ("pass", "fail")]
        worst = max(judged, key=lambda r: r["ub95_l"]) if judged else None
        per_host[sota] = {
            **cnt,
            "worst_cell": ({"phase": worst["phase"], "map": worst["map"], "N": worst["N"],
                            "ub95_l": worst["ub95_l"], "eps_l": worst["eps_l"],
                            "status": worst["status"]} if worst else None),
            "fail_cells": [f"{r['phase']}.{r['map']}.N{r['N']}" for r in hr
                           if r["status"] == "fail"],
            "cannot_certify_cells": [f"{r['phase']}.{r['map']}.N{r['N']}" for r in hr
                                     if r["status"] == "cannot_certify"],
            "skipped_cells": [f"{r['phase']}.{r['map']}.N{r['N']}" for r in hr
                              if r["status"] == "skip"],
            "all_evaluable_pass": cnt["fail"] == 0,
            "all_certified_pass": (cnt["fail"] == 0 and cnt["cannot_certify"] == 0
                                   and cnt["skip"] == 0)}
        for s, c in cnt.items():
            overall[s] += c
    return {"criterion": "ub95_l <= eps_measured(regime) AND ub95_l <= ln(1.03)*100; "
                         "n_eff<5 skip, 5<=n_eff<10 cannot_certify (prereg H2)",
            "cells": rows, "per_host": per_host,
            "overall": {**overall,
                        "all_evaluable_pass": overall["fail"] == 0,
                        "all_certified_pass": (overall["fail"] == 0
                                               and overall["cannot_certify"] == 0
                                               and overall["skip"] == 0)}}


# ---------------------------------------------------------------- ablation (mechanism decomposition)
def build_ablation(cells):
    """ablation.json: mechanism decompositions extracted from the SAME battery by
    within-instance pairing (DESIGN §2 extraction table) — all EXPLORATORY: HL + BCa CI
    only, no tests, no stars (preregistration: none may be promoted to a headline).
    Contrasts (l = lr100(y_full, y_other); l<0 = full better than the comparator):
      accept_side_gain=full vs v5_only; repair_side_gain=full vs spsa_only;
      learned_vs_fixed_accept=full vs rr5_v5; learned_vs_fixed_repair=full vs spsa_20;
      accept_selection_cart=full vs cart_v5; legacy_config=full vs rr5_20.
    Orthogonality: per instance with ALL of {stock, full, spsa_only, v5_only} ok,
      interaction = (l_spsa_only + l_v5_only) - l_full   (each l_* vs stock);
      interaction > 0 <=> full beats the additive prediction (synergy), < 0 sub-additive.
    Scope: phases A and G at t=60 (the fused main battery)."""
    contrasts = {name: [] for name, _, _ in ABLATION_CONTRASTS}
    ortho = []
    for key, arms in sorted(cells.items()):
        phase, sota, mp, N, t = key
        if phase not in ("A", "G") or t != T_MAIN:
            continue
        src = cell_source(arms)
        for name, arm_a, arm_b in ABLATION_CONTRASTS:
            if arm_a not in arms or arm_b not in arms:
                continue
            a_ok, b_ok = arms[arm_a]["ok"], arms[arm_b]["ok"]
            common = sorted(set(a_ok) & set(b_ok))
            for metric in METRICS:
                l = np.array([lr100(a_ok[k]["metrics"][metric],
                                    b_ok[k]["metrics"][metric]) for k in common],
                             dtype=float)
                tag = f"abl.{name}.{phase}.{sota}.{mp}.N{N}.{metric}"
                st = cell_stats(l, tag, with_tests=False)
                contrasts[name].append(
                    {"phase": phase, "sota": sota, "map": mp, "N": N, "metric": metric,
                     "source": src, "contrast": f"{arm_a} vs {arm_b}",
                     **{k: st[k] for k in ("n_eff", "hl_l", "ci_l", "bt_hl", "bt_ci",
                                           "wins", "ties", "losses", "flags")}})
        need = (BASE_ARM, "full", "spsa_only", "v5_only")
        if all(a in arms for a in need):
            oks = {a: arms[a]["ok"] for a in need}
            common = sorted(set.intersection(*(set(v) for v in oks.values())))
            for metric in METRICS:
                li = []
                for k in common:
                    ls = lr100(oks["spsa_only"][k]["metrics"][metric],
                               oks[BASE_ARM][k]["metrics"][metric])
                    lv = lr100(oks["v5_only"][k]["metrics"][metric],
                               oks[BASE_ARM][k]["metrics"][metric])
                    lf = lr100(oks["full"][k]["metrics"][metric],
                               oks[BASE_ARM][k]["metrics"][metric])
                    li.append((ls + lv) - lf)
                l = np.array(li, dtype=float)
                tag = f"abl.ortho.{phase}.{sota}.{mp}.N{N}.{metric}"
                st = cell_stats(l, tag, with_tests=False)
                ortho.append({"phase": phase, "sota": sota, "map": mp, "N": N,
                              "metric": metric, "source": src,
                              "definition": "(l_spsa_only + l_v5_only) - l_full; "
                                            ">0 = synergy, <0 = sub-additive",
                              **{k: st[k] for k in ("n_eff", "hl_l", "ci_l", "flags")}})
    summary = {}
    for name, rows in contrasts.items():
        for metric in METRICS:
            sel = [r for r in rows if r["metric"] == metric and r["hl_l"] is not None
                   and r["n_eff"] >= 5]
            summary[f"{name}.{metric}"] = {
                "n_cells": len(sel),
                "median_hl_l": r4(np.median([r["hl_l"] for r in sel])) if sel else None,
                "median_bt": r4(np.median([r["bt_hl"] for r in sel])) if sel else None,
                "cells_full_better": sum(1 for r in sel if r["hl_l"] < 0)}
    return {"role": "exploratory (effect+CI only; no tests, no stars — preregistered)",
            "contrasts": contrasts, "orthogonality": ortho, "summary": summary}


# ---------------------------------------------------------------- anytime (H4, B batch)
def _import_traj_util():
    """experiments/traj_util.py (sibling of analysis/) for raw .traj re-parsing; the
    aggregator stays runnable without it (metrics.traj from the runner is the primary
    source)."""
    exp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "experiments")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    try:
        import traj_util
        return traj_util
    except Exception as e:
        print(f"[warn] traj_util unavailable ({type(e).__name__}: {e}); "
              f"--traj raw re-parse disabled", file=sys.stderr)
        return None


def get_traj(rec, trajdir, tu):
    """Anytime observables for one B-batch run: metrics.traj (runner-parsed; note JSON
    round-trip turns delay_at keys into STRINGS — normalized to int here), else re-parse
    the raw <run_id>.traj from --traj via traj_util. Returns
    {"delay_at": {int t: delay|None}, "auc_delay": float|None} or None."""
    m = rec.get("metrics") or {}
    tr = m.get("traj")
    if tr and tr.get("delay_at"):
        return {"delay_at": {int(k): v for k, v in tr["delay_at"].items()},
                "auc_delay": tr.get("auc_delay")}
    if trajdir and tu is not None and m.get("sum_dist") is not None:
        p = os.path.join(trajdir, str(rec.get("run_id")) + ".traj")
        if os.path.isfile(p):
            r = tu.parse_traj(p, float(m["sum_dist"]), checkpoints=CHECKPOINTS,
                              t_end=float(rec.get("t_limit") or CHECKPOINTS[-1]))
            if r:
                return {"delay_at": {int(k): v for k, v in r["delay_at"].items()},
                        "auc_delay": r.get("auc_delay")}
    return None


def get_init_soc(rec, trajdir, tu):
    """Initial-solution SoC for the shared-init AUC check: metrics.init_soc when the
    binary emits it; else the FIRST incumbent of the raw .traj (the PP init) when --traj
    is given; else None (unverifiable)."""
    m = rec.get("metrics") or {}
    if m.get("init_soc") is not None:
        return float(m["init_soc"])
    if trajdir and tu is not None:
        p = os.path.join(trajdir, str(rec.get("run_id")) + ".traj")
        if os.path.isfile(p):
            ts, socs = tu.load_traj(p)
            if socs:
                return float(socs[0])
    return None


def build_anytime(cells, trajdir):
    """anytime.json (H4, phase B, t=300+traj). Per (sota,map,N) x arm-vs-stock:
      * per checkpoint t in {60,120,180,240,300}: paired Delta(t) = delay_arm(t) -
        delay_stock(t) (HL, descriptive raw units) AND paired log-ratio l(t) =
        lr100(delay_arm(t), delay_stock(t)) with HL + 95% BCa CI + ub95 + one-sided
        Wilcoxon (improvement); BH-FDR across the 5-checkpoint family of each
        (cell, arm) (within-checkpoint-family BH). Pairs where either side has no incumbent
        yet at t (delay_at[t] is None) are excluded and counted (n_no_incumbent).
      * AUC (traj.auc_delay, the PRIMARY anytime metric — single test per cell, no
        family correction): same paired treatment, with the shared-init partition —
        valid (init equal) / auc_invalid_pairs (init differs; EXCLUDED, disclosed) /
        auc_unverified_pairs (init unknown; EXCLUDED, disclosed + flagged) — see module
        docstring for the rationale.
      * drift_table (smoking gun): every (cell, t) where a DRIFT arm (rr20_v5/rr50_v5)
        is CERTIFIED worse than stock: log-ratio 95% CI entirely above 0 (ci_lo > 0 ==
        benefit CI upper bound < 0 in the design's wording).
    NEVER infer from overlap of two absolute curves' CIs — only these paired stats."""
    tu = _import_traj_util() if trajdir else None
    out_cells, drift = [], []
    n_missing_traj = 0
    for key, arms in sorted(cells.items()):
        phase, sota, mp, N, t = key
        if phase != "B" or BASE_ARM not in arms:
            continue
        base_ok = arms[BASE_ARM]["ok"]
        base_traj, base_init = {}, {}
        for ik, rec in base_ok.items():
            tr = get_traj(rec, trajdir, tu)
            if tr is None:
                n_missing_traj += 1
            else:
                base_traj[ik] = tr
            base_init[ik] = get_init_soc(rec, trajdir, tu)
        for arm in sorted(a for a in arms if a != BASE_ARM):
            arm_ok = arms[arm]["ok"]
            arm_traj, arm_init = {}, {}
            for ik, rec in arm_ok.items():
                tr = get_traj(rec, trajdir, tu)
                if tr is None:
                    n_missing_traj += 1
                else:
                    arm_traj[ik] = tr
                arm_init[ik] = get_init_soc(rec, trajdir, tu)
            common = sorted(set(base_traj) & set(arm_traj))
            cell = {"sota": sota, "map": mp, "N": N, "t_budget": t, "arm": arm,
                    "n_traj_pairs": len(common), "checkpoints": {}, "auc": None}
            ps = []
            for cp in CHECKPOINTS:
                deltas, ls = [], []
                miss = 0
                for ik in common:
                    ds = base_traj[ik]["delay_at"].get(cp)
                    da = arm_traj[ik]["delay_at"].get(cp)
                    if ds is None or da is None:
                        miss += 1
                        continue
                    deltas.append(float(da) - float(ds))
                    ls.append(lr100(da, ds))
                l = np.array(ls, dtype=float)
                tag = f"B.{sota}.{mp}.N{N}.{arm}.t{cp}"
                st = cell_stats(l, tag, with_tests=True)
                entry = {"n_pairs": int(l.size), "n_no_incumbent": miss,
                         "delta_hl": r4(hl(np.asarray(deltas))) if deltas else None,
                         **{k: st[k] for k in ("hl_l", "ci_l", "ub95_l", "bt_hl",
                                               "p_sup", "wilcoxon_method", "flags")}}
                cell["checkpoints"][str(cp)] = entry
                ps.append(entry)
                if (arm in DRIFT_ARMS and entry["ci_l"][0] is not None
                        and entry["ci_l"][0] > 0):
                    drift.append({"sota": sota, "map": mp, "N": N, "arm": arm, "t": cp,
                                  "hl_l": entry["hl_l"], "ci_l": entry["ci_l"],
                                  "bt_hl": entry["bt_hl"],
                                  "certified": "worse_than_stock (log-ratio CI > 0)"})
            with_p = [e for e in ps if e["p_sup"] is not None]
            padj, rej = bh_fdr([e["p_sup"] for e in with_p], Q_FDR)
            for e, pa, rj in zip(with_p, padj, rej):
                e["p_bh"], e["sig_bh"] = r4(pa), bool(rj)
            # ---- AUC (primary anytime metric) with shared-init verification ----
            # PREREGISTRATION: AUC "valid only where both arms share the initial solution;
            # verified via equal initial delay" => unverifiable pairs are EXCLUDED from the
            # primary statistic (counted + disclosed), like invalid pairs (review fix).
            ls, n_invalid, n_unverified = [], 0, 0
            for ik in common:
                as_, aa = base_traj[ik].get("auc_delay"), arm_traj[ik].get("auc_delay")
                if as_ is None or aa is None:
                    continue
                is_, ia = base_init.get(ik), arm_init.get(ik)
                if is_ is None or ia is None:
                    n_unverified += 1
                    continue
                if abs(is_ - ia) > 1e-6:
                    n_invalid += 1
                    continue
                ls.append(lr100(aa, as_))
            l = np.array(ls, dtype=float)
            st = cell_stats(l, f"B.{sota}.{mp}.N{N}.{arm}.auc", with_tests=True)
            flags = list(st["flags"])
            if n_unverified:
                flags.append("init_unverified_pairs_excluded")
            cell["auc"] = {"n_pairs": int(l.size), "auc_invalid_pairs": n_invalid,
                           "auc_unverified_pairs": n_unverified,
                           **{k: st[k] for k in ("hl_l", "ci_l", "ub95_l", "bt_hl",
                                                 "p_sup", "wilcoxon_method")},
                           "flags": flags}
            out_cells.append(cell)
    # H4 late-time dominance: the real anytime story is NOT "fixed high-delta rr drifts BELOW
    # stock" (drift_table is usually empty — rr stays above stock) but "spsa (full) keeps
    # improving to t=300 while fixed high-delta rr plateaus". Per cell, compare full's t=300
    # vs-stock improvement against rr20/rr50's. More-negative hl_l = more improvement; a positive
    # advantage = spsa pulled ahead late.
    by_cell = defaultdict(dict)
    for c in out_cells:
        e = c["checkpoints"].get("300", {})
        if e.get("hl_l") is not None:
            by_cell[(c["sota"], c["map"], c["N"])][c["arm"]] = e["hl_l"]
    late = []
    for (sota, mp, N), arm_hl in sorted(by_cell.items()):
        f = arm_hl.get("full")
        if f is None:
            continue
        row = {"sota": sota, "map": mp, "N": N, "full_t300_hl": r4(f)}
        for rr in ("rr20_v5", "rr50_v5"):
            if rr in arm_hl:
                row[f"{rr}_t300_hl"] = r4(arm_hl[rr])
                row[f"full_adv_over_{rr}"] = r4(arm_hl[rr] - f)   # >0 => full improved more
        late.append(row)
    return {"checkpoints": list(CHECKPOINTS),
            "protocol": "paired Delta(t)/log-ratio per checkpoint, BH within each "
                        "(cell,arm) checkpoint family; AUC primary (init-verified); "
                        "drift = log-ratio CI entirely above 0; no absolute-curve "
                        "CI-overlap inference. late_dominance = full's t=300 improvement "
                        "vs fixed high-delta rr's (the H4 story: spsa keeps improving late)",
            "n_runs_missing_traj": n_missing_traj,
            "cells": out_cells, "drift_table": drift, "late_dominance": late}


# ---------------------------------------------------------------- generated maps (H5)
def build_gen(cells, cell_rows):
    """gen.json (H5, phase G, source='gen:<family>'): per (family, host), full-vs-stock
    on delay: map-level HL (all paired l of that map pooled across its N levels — the
    per-family ECDF data points, on both scales), direction binomial over maps
    (one-sided 'greater than chance improvement' — H5 is directional), worst cell by
    ub95_l (from cells.json rows), and n. gmaze carries null_control=true (preregistered
    perfect-maze null: expected effect ~0 — built-in mechanism evidence, not a failure).
    No cross-family pooling; no post-hoc exclusion of any map/seed/scenario."""
    per = defaultdict(lambda: defaultdict(list))    # (family, sota) -> map -> [l...]
    for key, arms in sorted(cells.items()):
        phase, sota, mp, N, t = key
        if phase != "G" or BASE_ARM not in arms or CONF_ARM not in arms:
            continue
        src = cell_source(arms)
        if not str(src).startswith("gen:"):
            continue
        fam = str(src)[4:]
        l, common, _ = paired_ls(arms, CONF_ARM, "delay")
        per[(fam, sota)][mp].extend(float(x) for x in l)
    out = {}
    for (fam, sota), maps in sorted(per.items()):
        map_hl = {mp: float(hl(np.asarray(v))) for mp, v in sorted(maps.items()) if v}
        vals = sorted(map_hl.values())
        imp = sum(1 for v in vals if v < 0)
        wor = sum(1 for v in vals if v > 0)
        p = (float(stats.binomtest(imp, imp + wor, 0.5, alternative="greater").pvalue)
             if imp + wor > 0 else None)
        wrows = [r for r in cell_rows
                 if r["phase"] == "G" and r["sota"] == sota and r["arm"] == CONF_ARM
                 and r["metric"] == "delay" and r["source"] == "gen:" + fam
                 and r["ub95_l"] is not None]
        worst = max(wrows, key=lambda r: r["ub95_l"]) if wrows else None
        out.setdefault(fam, {"null_control": fam == "gmaze", "per_host": {}})
        out[fam]["per_host"][sota] = {
            "n_maps": len(map_hl), "n_cells": len(wrows),
            "n_pairs": int(sum(len(v) for v in maps.values())),
            "maps_improved": imp, "maps_worse": wor,
            "maps_zero": len(vals) - imp - wor,
            "binom_p_one_sided": r4(p),
            "ecdf_map_hl_l": [r4(v) for v in vals],
            "ecdf_map_bt": [r4(bt(v)) for v in sorted(vals, reverse=True)],
            "worst_cell": ({"map": worst["map"], "N": worst["N"],
                            "ub95_l": worst["ub95_l"], "hl_l": worst["hl_l"],
                            "n_eff": worst["n_eff"]} if worst else None)}
    return {"protocol": "per-family per-host: map-level HL ECDF + one-sided direction "
                        "binomial over maps; gmaze = preregistered null control; "
                        "zero-shot OOD (params frozen on official maps)",
            "families": out}


# ---------------------------------------------------------------- reconciliation
def build_recon(cells, load_report):
    """recon.json: per paired cell (phases L/A/B/G) and non-stock arm the accounting
        scheduled = n_scen(source) * n_reps(phase)      (official 25, gen 8; L reps 3)
        scheduled =?= n_eff + dropped(single-side FAILED, split by side) + both_failed
    'missing' = a scheduled run with NO record at all (battery incomplete) breaks the
    equation on purpose -> HARD WARNING, listed. Records outside the expected
    scen/rep grid are 'unexpected' -> HARD WARNING, listed. Fail records superseded by
    a successful retry were already dropped at load time (never double-counted).
    E0 (stock-only, unpaired) gets its own coverage table (feeds epsilon)."""
    rows, warnings, e0 = [], [], []
    for key, arms in sorted(cells.items()):
        phase, sota, mp, N, t = key
        src = cell_source(arms)
        n_scen = SCENS_GEN if str(src).startswith("gen:") else SCENS_OFFICIAL
        if phase == "E0":
            for rep in (1, 2):
                oks = sum(1 for (s, rp) in arms.get(BASE_ARM, {}).get("ok", {}) if rp == rep)
                fls = sum(1 for (s, rp) in arms.get(BASE_ARM, {}).get("fail", {}) if rp == rep)
                e0.append({"sota": sota, "map": mp, "N": N, "rep": rep, "ok": oks,
                           "fail": fls, "missing": n_scen - oks - fls})
                if oks + fls != n_scen:
                    warnings.append(f"E0.{sota}.{mp}.N{N}.r{rep}: ok({oks})+fail({fls})"
                                    f" != {n_scen} (missing runs)")
            continue
        if phase not in ("L", "A", "B", "G"):
            continue
        reps = REPS_BY_PHASE.get(phase, (0,))
        expected = {(s, rp) for s in range(1, n_scen + 1) for rp in reps}
        scheduled = len(expected)
        base = arms.get(BASE_ARM, {"ok": {}, "fail": {}})
        for arm in sorted(a for a in arms if a != BASE_ARM):
            am = arms[arm]
            n_eff = drop_arm = drop_stock = both_failed = missing_pairs = 0
            for ik in expected:
                s_ok, a_ok = ik in base["ok"], ik in am["ok"]
                s_f, a_f = ik in base["fail"], ik in am["fail"]
                if s_ok and a_ok:
                    n_eff += 1
                elif s_ok and a_f:
                    drop_arm += 1
                elif a_ok and s_f:
                    drop_stock += 1
                elif s_f and a_f:
                    both_failed += 1
                else:                       # at least one side has NO record at all
                    missing_pairs += 1
            unexpected = sorted(set(list(base["ok"]) + list(base["fail"])
                                    + list(am["ok"]) + list(am["fail"])) - expected)
            balanced = (n_eff + drop_arm + drop_stock + both_failed == scheduled)
            row = {"phase": phase, "sota": sota, "map": mp, "N": N, "t": t, "arm": arm,
                   "source": src, "scheduled": scheduled, "n_eff": n_eff,
                   "dropped_arm_failed": drop_arm, "dropped_stock_failed": drop_stock,
                   "both_failed": both_failed, "missing_pairs": missing_pairs,
                   "unexpected_records": len(unexpected), "balanced": bool(balanced)}
            rows.append(row)
            if not balanced or unexpected:
                w = (f"{phase}.{sota}.{mp}.N{N}.t{t}.{arm}: n_eff({n_eff})+dropped"
                     f"({drop_arm}+{drop_stock})+both_failed({both_failed}) != "
                     f"scheduled({scheduled}); missing_pairs={missing_pairs}"
                     + (f"; unexpected={unexpected[:5]}" if unexpected else ""))
                warnings.append(w)
    for w in warnings:
        print(f"[recon-WARN] {w}", file=sys.stderr)
    return {"note": "scheduled = n_scen x reps (official 25 / gen 8; L reps 0-2, others "
                    "rep 0); recon enumerates cells with >=1 record — a cell that never "
                    "produced ANY record is only catchable against the config, out of "
                    "scope here (disclosed limitation)",
            "cells": rows, "e0_coverage": e0, "warnings": warnings,
            "load_report": load_report}


# ---------------------------------------------------------------- summary + io
def _fmt(x, nd=2, unit=""):
    return "na" if x is None else f"{x:+.{nd}f}{unit}"


def write_summary(outdir, load_report, epsilon, headline, dnh, ablation, anytime, gen,
                  recon, cell_rows):
    """summary.txt — the human-readable digest of the JSON outputs (paper tables must be
    built from the JSONs, not from this file)."""
    L = []
    L.append("AAAI battery aggregation summary (see *.json for the auditable numbers)")
    L.append(f"scipy=={scipy.__version__} numpy=={np.__version__}")
    L.append(f"loaded: {load_report['records_kept']} records "
             f"({load_report['files_seen']} files, {load_report['unreadable']} unreadable, "
             f"{load_report['benign_ok_duplicates']} benign dups, "
             f"{load_report['superseded_fail_records']} superseded fails, "
             f"ignored phases {load_report['ignored_phases']})")
    L.append("")
    L.append("[EPSILON] solver-noise floor (95th pct |lr100|, stock replicate pairs)")
    g = epsilon["global"]
    L.append(f"  global : eps95={_fmt(g['eps95_l'])} l-pts  (n_pairs={g['n_pairs']})")
    for reg, v in epsilon["regimes"].items():
        L.append(f"  {reg:<12}: eps95={_fmt(v['eps95_l'])} l-pts  (n_pairs={v['n_pairs']})")
    L.append("")
    L.append("[HEADLINE H1] per-host two-sided binomial over full-cell directions (phase A)")
    for sota, mets in headline["hosts"].items():
        for metric, h in mets.items():
            L.append(f"  {sota}.{metric}: improved {h['improved']}/{h['improved']+h['worse']}"
                     f" (+{h['zero']} zero)  p={h['binom_p_two_sided']}  "
                     f"bt median {_fmt(h['bt_hl_median'])}% "
                     f"range [{_fmt(h['bt_hl_range'][0])},{_fmt(h['bt_hl_range'][1])}]%  "
                     f"BH-sig {h['bh_sig_cells']}/{h['cells']}")
    if headline["lock_decontention"]:
        L.append("")
        L.append("[LOCK H3] absolute median delay (stock vs full), 3 reps x 25 scens")
        for r in headline["lock_decontention"]:
            st, fu = r.get(BASE_ARM, {}), r.get(CONF_ARM, {})
            L.append(f"  {r['map']}.N{r['N']}: stock {st.get('median_delay')} "
                     f"-> full {fu.get('median_delay')}  "
                     f"(paired bt {_fmt(r['paired_full_vs_stock']['bt_hl'])}%, "
                     f"n={r['paired_full_vs_stock']['n_eff']})")
    L.append("")
    L.append("[DO-NO-HARM H2] full arm, delay, ub95_l <= eps(regime) AND <= ln(1.03)*100")
    for sota, h in dnh["per_host"].items():
        wc = h["worst_cell"]
        L.append(f"  {sota}: pass {h['pass']}  fail {h['fail']}  "
                 f"cannot_certify {h['cannot_certify']}  skip {h['skip']}  "
                 f"worst {wc['map'] + '.N' + str(wc['N']) if wc else 'na'} "
                 f"ub95={_fmt(wc['ub95_l']) if wc else 'na'} l-pts  "
                 f"evaluable_pass={h['all_evaluable_pass']} "
                 f"certified={h['all_certified_pass']}")
        if h["fail_cells"]:
            L.append(f"    FAIL cells: {h['fail_cells']}")
        if h["cannot_certify_cells"]:
            L.append(f"    cannot_certify: {h['cannot_certify_cells']}")
    o = dnh["overall"]
    L.append(f"  OVERALL: evaluable_pass={o['all_evaluable_pass']} "
             f"certified={o['all_certified_pass']} "
             f"(pass {o['pass']} / fail {o['fail']} / cannot {o['cannot_certify']} / "
             f"skip {o['skip']})")
    L.append("")
    L.append("[ABLATION] exploratory medians across cells (n_eff>=5), l<0 = full better")
    for k, s in ablation["summary"].items():
        if k.endswith(".delay"):
            L.append(f"  {k:<38} n={s['n_cells']:<4} median_hl_l={_fmt(s['median_hl_l'])} "
                     f"(bt {_fmt(s['median_bt'])}%)  full-better cells "
                     f"{s['cells_full_better']}/{s['n_cells']}")
    L.append("")
    dr = anytime["drift_table"]
    L.append(f"[ANYTIME H4] B batch: {len(anytime['cells'])} (cell,arm) trajectories; "
             f"drift smoking gun: {len(dr)} certified-worse (cell,t) entries "
             f"(rr20/rr50 vs stock)")
    for d in dr[:20]:
        L.append(f"  DRIFT {d['sota']}.{d['map']}.N{d['N']}.{d['arm']}@t={d['t']}: "
                 f"hl_l={_fmt(d['hl_l'])} ci={d['ci_l']}")
    if len(dr) > 20:
        L.append(f"  ... {len(dr) - 20} more in anytime.json")
    late = anytime.get("late_dominance", [])
    advs = [r["full_adv_over_rr50_v5"] for r in late if r.get("full_adv_over_rr50_v5") is not None]
    if advs:
        wins = sum(1 for a in advs if a > 0)
        L.append(f"  late dominance (t=300, full vs fixed rr50): full improved MORE in "
                 f"{wins}/{len(advs)} cells; median advantage {_fmt(float(np.median(advs)))} l-pts "
                 f"(>0 = spsa keeps improving while high-delta rr plateaus — the real H4 signal)")
        for r in late[:8]:
            if r.get("full_adv_over_rr50_v5") is not None:
                L.append(f"    {r['sota']}.{r['map']}.N{r['N']}: full t300={_fmt(r['full_t300_hl'])} "
                         f"rr50={_fmt(r.get('rr50_v5_t300_hl'))} adv={_fmt(r['full_adv_over_rr50_v5'])}")
    L.append("")
    L.append("[GENERATED H5] per family x host: maps improved (one-sided binomial)")
    for fam, f in gen["families"].items():
        tag = " [NULL CONTROL: expected ~0]" if f["null_control"] else ""
        for sota, h in f["per_host"].items():
            L.append(f"  {fam}.{sota}: {h['maps_improved']}/{h['n_maps']} maps improved "
                     f"p={h['binom_p_one_sided']}{tag}")
    L.append("")
    nfb = sum(1 for r in cell_rows if "percentile_fallback" in r["flags"])
    nlow = sum(1 for r in cell_rows if "low_n" in r["flags"])
    L.append(f"[FLAGS] percentile_fallback cells: {nfb}; low_n (<5) cells: {nlow}; "
             f"recon warnings: {len(recon['warnings'])} (recon.json)")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump(outdir, name, obj):
    with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)


ITER_CAP = 100000000   # MAXITER in gen_config_aaai.py: iterations reaching it means the cap
                       # (not the wall-clock budget) bound the run -> biases greedy vs layer
def audit_iteration_cap(records):
    """PREREGISTRATION/DESIGN promise: flag any run whose iteration count reaches the cap.
    A non-empty list means the iteration cap bound before the time budget (greedy hits it
    first, biasing the comparison) — raise MAXITER and re-run those cells."""
    hits = [{"run_id": r.get("run_id"), "sota": r.get("sota"), "map": r.get("map"),
             "N": r.get("N"), "arm": r.get("arm"), "iterations": r["metrics"]["iterations"]}
            for r in records
            if r.get("status") == "ok" and r.get("metrics")
            and r["metrics"].get("iterations") is not None
            and r["metrics"]["iterations"] >= ITER_CAP]
    return {"cap": ITER_CAP, "n_at_cap": len(hits), "runs": hits[:200],
            "note": "empty = wall-clock budget always bound first (good); non-empty = "
                    "iteration cap bound, biasing greedy-vs-layer -> raise MAXITER"}

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", action="append", required=True,
                    help="per-run JSON dir (repeatable; multi-box merge)")
    ap.add_argument("--traj", default=None,
                    help="raw B-batch .traj dir (optional; backfill + init verification)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for _d in args.results:
        if not os.path.isdir(_d):
            print(f"FATAL: --results dir not found: {_d}", file=sys.stderr)
            sys.exit(1)
    records, load_report = load_runs(args.results)          # exits 1 on pairing conflicts
    if not records:
        print("[FATAL] no analyzable records found", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    cells = index_cells(records)
    epsilon = compute_epsilon(records)
    cell_rows = build_cells(cells)
    headline = build_headline(cell_rows, cells)
    dnh = build_dnh(cell_rows, epsilon)
    ablation = build_ablation(cells)
    anytime = build_anytime(cells, args.traj)
    gen = build_gen(cells, cell_rows)
    recon = build_recon(cells, load_report)
    recon["iteration_cap_audit"] = audit_iteration_cap(records)
    dump(args.out, "cells.json", cell_rows)
    dump(args.out, "epsilon.json", epsilon)
    dump(args.out, "headline.json", headline)
    dump(args.out, "dnh.json", dnh)
    dump(args.out, "ablation.json", ablation)
    dump(args.out, "anytime.json", anytime)
    dump(args.out, "gen.json", gen)
    dump(args.out, "recon.json", recon)
    write_summary(args.out, load_report, epsilon, headline, dnh, ablation, anytime,
                  gen, recon, cell_rows)


if __name__ == "__main__":
    main()
