#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anytime-trajectory reconstruction for the CART+TS zoo battery.

The solver (add_traj.py, L2_TRAJ=<path>) dumps a 2-column CSV "runtime,incumbent": the best-so-far
SoC each time it strictly improves, plus one final row at the terminal runtime. This module turns
ONE t=T run's trajectory into the anytime observables the aggregate needs, with NO extra runs:

  parse_traj(traj_file, sum_dist, checkpoints, t_end)
      -> {"delay_at": {t: delay(incumbent@t)},        # step-interpolated (last incumbent <= t)
          "soc_at":   {t: incumbent_soc@t},
          "auc_delay": AUC of delay over [t0, t_end],   # trapezoid on the step function
          "final_delay", "final_soc", "n_points", "first_t"}

Anytime comparators (baseline vs arm), each from the two parsed dicts:
  time_to_match(arm_traj, target_soc)  -> first runtime the arm's incumbent reaches <= target
                                          (e.g. the baseline's FINAL soc => "time to match stock")
  crossing_time(base_traj, arm_traj)   -> first t where arm incumbent <= base incumbent (step)

Design notes:
  * incumbent is a step function: value(t) = incumbent of the last dumped row with runtime <= t.
    Before the first row, value is undefined (solver had no incumbent yet) -> we clamp to the
    initial incumbent (first row) for t >= first_t, and report None for checkpoints < first_t.
  * delay = incumbent_soc - sum_dist (sum_dist from the run's -LNS.csv "sum of distance").
  * AUC integrates DELAY (not raw SoC) so it is comparable across N; matches the solver's own
    "area under curve" column semantics (which also subtracts sum_of_distances), letting stock
    cross-check (stock: working==incumbent => our AUC ~= the CSV AUC column).
"""
import bisect


def load_traj(path):
    """Read a runtime,incumbent CSV -> (ts, socs) sorted by runtime, deduped to the running min
    (defensive: the dump is already monotone, but a duplicate final row or float jitter is fine).
    Returns ([], []) if the file is missing/empty/garbage."""
    rows = []
    try:
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln[0].isalpha():      # skip header / blank
                    continue
                a, _, b = ln.partition(",")
                try:
                    rows.append((float(a), float(b)))
                except ValueError:
                    continue
    except OSError:
        return [], []
    rows.sort()
    ts, socs, best = [], [], float("inf")
    for t, c in rows:
        if c < best:                              # enforce non-increasing incumbent
            best = c
        if ts and t == ts[-1]:                    # same timestamp -> keep the better
            socs[-1] = min(socs[-1], best)
        else:
            ts.append(t); socs.append(best)
    return ts, socs


def _step_soc(ts, socs, t):
    """incumbent SoC at time t (step: last row with runtime <= t). None if t precedes first row."""
    if not ts:
        return None
    if t < ts[0]:
        return None
    i = bisect.bisect_right(ts, t) - 1
    return socs[i]


def parse_traj(traj_file, sum_dist, checkpoints=(60, 120, 180, 240, 300), t_end=300.0):
    """Reconstruct anytime observables for ONE run. sum_dist = "sum of distance" from -LNS.csv.
    Returns None if the trajectory is unusable (no rows)."""
    ts, socs = load_traj(traj_file)
    if not ts:
        return None
    first_t = ts[0]
    delay_at, soc_at = {}, {}
    for cp in checkpoints:
        soc = _step_soc(ts, socs, cp)
        if soc is None:                           # checkpoint before the first incumbent existed
            delay_at[cp] = None; soc_at[cp] = None
        else:
            soc_at[cp] = soc
            delay_at[cp] = soc - sum_dist
    # AUC of DELAY over [first_t, t_end] on the step function (trapezoid == rectangles for a step).
    # Anchor the left edge at first_t (before that there is no incumbent). Extend the last incumbent
    # flat to t_end (the solver's final row already sits at ~t_end, so this is a no-op in practice).
    t0 = first_t
    auc = 0.0
    prev_t = t0
    prev_delay = socs[0] - sum_dist
    for t, c in zip(ts[1:], socs[1:]):
        if t <= prev_t:
            continue
        seg_end = min(t, t_end)
        if seg_end > prev_t:
            auc += prev_delay * (seg_end - prev_t)
        prev_t = seg_end
        prev_delay = c - sum_dist
        if prev_t >= t_end:
            break
    if prev_t < t_end:                            # flat tail to t_end
        auc += prev_delay * (t_end - prev_t)
    return {"delay_at": delay_at, "soc_at": soc_at, "auc_delay": auc,
            "final_soc": socs[-1], "final_delay": socs[-1] - sum_dist,
            "n_points": len(ts), "first_t": first_t}


def time_to_match(ts, socs, target_soc):
    """First runtime at which the incumbent reaches <= target_soc. None if never (or empty)."""
    for t, c in zip(ts, socs):
        if c <= target_soc:
            return t
    return None


def crossing_time(base_ts, base_socs, arm_ts, arm_socs, grid=None):
    """First time the ARM incumbent pulls STRICTLY ahead of the BASE incumbent (arm < base), on a
    shared step grid. Strict because stock and arm share the same PP init => they start EQUAL, so
    a <= b is trivially true at t0 and would report a meaningless crossing of 0. grid defaults to
    the union of both series' timestamps. None if the arm never gets strictly ahead."""
    if not arm_ts or not base_ts:
        return None
    if grid is None:
        grid = sorted(set(base_ts) | set(arm_ts))
    for t in grid:
        b = _step_soc(base_ts, base_socs, t)
        a = _step_soc(arm_ts, arm_socs, t)
        if a is not None and b is not None and a < b:
            return t
    return None


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    import sys, tempfile, os
    if len(sys.argv) >= 3:
        # real check: parse_traj on an actual traj + its -LNS.csv, cross-check final delay vs CSV.
        traj, csv = sys.argv[1], sys.argv[2]
        with open(csv) as f:
            rows = [r.strip() for r in f if r.strip()]
        hdr = [h.strip().lower() for h in rows[0].split(",")]
        last = rows[-1].split(",")
        def col(nm):
            return float(last[hdr.index(nm)]) if nm in hdr else None
        soc, sd = col("solution cost"), col("sum of distance")
        auc_csv = col("area under curve"); ms = col("makespan"); rt = col("runtime")
        r = parse_traj(traj, sd, t_end=max(rt or 300.0, 300.0))
        print(f"CSV : solution_cost={soc} sum_dist={sd} delay={soc-sd:.1f} "
              f"auc_col={auc_csv} makespan={ms} runtime={rt}")
        print(f"TRAJ: n_points={r['n_points']} first_t={r['first_t']:.3f} "
              f"final_soc={r['final_soc']} final_delay={r['final_delay']:.1f} "
              f"auc_delay={r['auc_delay']:.1f}")
        print(f"      delay_at={ {k: (round(v,1) if v is not None else None) for k,v in r['delay_at'].items()} }")
        d_final_match = abs(r["final_soc"] - soc) < 1e-6
        # for stock/greedy, working==incumbent => final soc must equal CSV solution cost exactly.
        print(f"CHECK final_soc==CSV solution_cost : {d_final_match} "
              f"(diff={r['final_soc']-soc:+.1f}; exact only for best-return/greedy arms)")
        if auc_csv:
            print(f"CHECK auc_delay vs CSV auc_col ratio: {r['auc_delay']/auc_csv:.4f} "
                  f"(≈1 for stock where working==incumbent)")
        sys.exit(0)
    # synthetic unit test (no box needed): a hand-built step trajectory with known answers.
    fd, p = tempfile.mkstemp(suffix=".traj"); os.close(fd)
    with open(p, "w") as f:
        f.write("runtime,incumbent\n")
        for t, c in [(1.0, 1000), (10.0, 900), (10.0, 900), (100.0, 850), (250.0, 800), (300.0, 800)]:
            f.write(f"{t},{c}\n")
    sd = 500.0
    r = parse_traj(p, sd, checkpoints=(0.5, 5, 60, 120, 300), t_end=300.0)
    ts, socs = load_traj(p)
    assert r["delay_at"][0.5] is None, "before first incumbent -> None"
    assert r["soc_at"][5] == 1000, "step holds first incumbent at t=5"
    assert r["soc_at"][60] == 900 and r["soc_at"][120] == 850, r["soc_at"]
    assert r["soc_at"][300] == 800 and r["final_delay"] == 300, r
    # AUC(delay) over [1,300]: 500*(10-1)+400*(100-10)+350*(250-100)+300*(300-250)
    exp = 500*9 + 400*90 + 350*150 + 300*50
    assert abs(r["auc_delay"] - exp) < 1e-6, (r["auc_delay"], exp)
    assert abs(time_to_match(ts, socs, 850) - 100.0) < 1e-9, "reaches 850 at t=100"
    assert time_to_match(ts, socs, 700) is None, "never reaches 700"
    # crossing: both start at 1000@t=1 (shared init); arm drops to 900 at t=10 while base holds
    # 1000 until t=250 -> arm is STRICTLY ahead from t=10 (not t=1, the equal start).
    with open(p + ".b", "w") as f:
        f.write("runtime,incumbent\n1.0,1000\n250.0,850\n300.0,850\n")
    bts, bsocs = load_traj(p + ".b")
    assert crossing_time(bts, bsocs, ts, socs) == 10.0, crossing_time(bts, bsocs, ts, socs)
    # symmetric sanity: arm never ahead of a base that dominates it -> None
    assert crossing_time(ts, socs, bts, bsocs) is None, "base(arg2) never beats the faster arm(arg1)"
    os.remove(p); os.remove(p + ".b")
    print("traj_util self-test PASS: step-interp, AUC, time-to-match, crossing all correct")
