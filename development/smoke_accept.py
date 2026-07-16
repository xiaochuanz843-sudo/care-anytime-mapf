#!/usr/bin/env python3
"""Accept-lever smoke battery — ISO-WALL-CLOCK, single-thread, paired.

The direction-v2 workflow verdict: selective exact repair is DEAD as a magnitude engine at
iso-wall-clock (confirmed: ties at N100, loses +2-3% at N150). The REAL magnitude engine is
NON-GREEDY ACCEPTANCE (rr = record-to-record): a one-line accept condition with ZERO per-iteration
cost, so iso-iteration ~= iso-wall-clock for it (unlike EECBS). This battery tests it cleanly.

Modes:
  --mode magnitude : Smoke-0. greedy vs rr{2,5,10} on one (map,N). Does rr win iso-wall-clock, and
                     does it actually FIRE (different iteration/accept behavior, not silently greedy)?
  --mode sweep     : Smoke-1. greedy vs rr5 across a congestion sweep (pass several --agents). Confirm
                     the sign is MONOTONE in congestion gap (win low/mid N, regress on the congested tail).
  --mode switch    : Smoke-2 (THE ML-decider). rr5 with an ORACLE time-switch AMOR_SWITCH_FRAC in
                     {0,.25,.5,.75,1}: rr for the first f of wall-clock, greedy after. If some INTERIOR
                     f beats BOTH endpoints (f=0 pure-greedy, f=1 pure-rr) beyond seed noise, within-run
                     non-stationarity is REAL -> online bandit has signal -> Form A (strong ML paper).
                     If best f is an endpoint -> no online signal -> ship zero-ML rr-port (Form B).

Usage:
  python3 experiments/smoke_accept.py --mode magnitude --map room-32-32-4 --agents 200 --scens 1,2,3,4,5 --times 15,30
  python3 experiments/smoke_accept.py --mode sweep --map room-32-32-4 --agents 150,200,300 --scens 1,2,3 --times 30
  python3 experiments/smoke_accept.py --mode switch --map room-32-32-4 --agents 200 --scens 1,2,3 --times 30
"""
from __future__ import annotations
import argparse, math, os, re, statistics, subprocess, sys, tempfile, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def find_binary():
    for c in ["MAPF-LNS2/lns", "MAPF-LNS2/build/lns", "MAPF-LNS2/build/lns.exe", "MAPF-LNS2/lns.exe"]:
        p = ROOT / c
        if p.exists(): return str(p)
    sys.exit("lns binary not found — build first.")

def map_path(m):
    for c in [f"movingai/maps/{m}.map", f"movingai/{m}.map", f"data/maps/{m}.map"]:
        if (ROOT / c).exists(): return str(ROOT / c)
    sys.exit(f"map not found: {m}")

def scen_path(m, n):
    for c in [f"movingai/scen-random/{m}-random-{n}.scen", f"data/scen/{m}-random-{n}.scen"]:
        if (ROOT / c).exists(): return str(ROOT / c)
    return ""

def parse_soc(pf):
    if not os.path.exists(pf): return None
    soc = n = 0
    with open(pf, encoding="utf-8", errors="ignore") as f:
        for line in f:
            mo = re.match(r"Agent\s+\d+:(.*)", line.strip(), re.IGNORECASE)
            if mo:
                c = re.findall(r"\(\d+,\s*\d+\)", mo.group(1))
                if c: soc += len(c) - 1; n += 1
    return (soc, n) if n else (None, 0)

MAXITER = "1000000000"
ITERSTATS = None
def run_one(binary, mapf, scenf, k, seed, t, env_add, tag=""):
    """Returns a dict: soc/iters/lb/best (ints or None), diag (CUCB/REPAIR_ARMS line), status.
    status: ok | timeout | initfail | crash | badpaths — audit fix: silent Nones hid WHY cells dropped,
    so scale batteries conditioned on 'instances easy enough for both arms' with no audit trail."""
    fd, pf = tempfile.mkstemp(suffix=".txt"); os.close(fd)
    cmd = [binary, "-m", mapf, "-a", scenf, "-k", str(k), "-t", str(t), "--seed", str(seed),
           "--maxIterations", MAXITER, "--destoryStrategy", "RandomWalk",
           "--initAlgo", "PP", "--replanAlgo", "PP", "--outputPaths", pf]
    if ITERSTATS:   # PI(T) anytime metric: per-iteration incumbent CSV (binary --stats)
        cmd += ["--stats", os.path.join(ITERSTATS, f"iter_{tag}.csv")]
    env = dict(os.environ); env.update(env_add)
    rec = {"soc": None, "iters": None, "lb": None, "best": None, "diag": None, "status": "ok"}
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=t + 120, env=env, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        # iterations MUST come from the final `LNS(...)` line — a bare first-match regex grabs the
        # InitLNS line and produced the bogus time-invariant "iters~77/79/81" artifact in old logs.
        lns_line = next((l for l in reversed(out.splitlines()) if l.startswith("LNS(")), "")
        mi = re.search(r"iterations = (\d+)", lns_line)
        arms = re.search(r"(?:REPAIR_ARMS|CUCB_ARMS):([^\n]+)", out)   # bandit convergence counts, when present
        if arms:
            rec["diag"] = arms.group(1).strip()
            print(f"      REPAIR_ARMS {rec['diag']}")
        lbm = re.search(r"lb = (\d+)", lns_line)
        bim = re.search(r"best_incumbent = (\d+)", lns_line)   # estimand audit: final-accepted vs best (differs under rr)
        soc, nag = parse_soc(pf)
        if "Failed to find an initial solution" in out: rec["status"] = "initfail"
        elif r.returncode != 0: rec["status"] = "crash"
        elif soc is not None and nag != k: soc = None; rec["status"] = "badpaths"   # partial paths file != real SOC
        rec.update(soc=soc, iters=(int(mi.group(1)) if mi else None), lb=(int(lbm.group(1)) if lbm else None),
                   best=(int(bim.group(1)) if bim else None))
        return rec
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"; return rec
    finally:
        try: os.remove(pf)
        except OSError: pass

def wilcoxon_p(diffs):
    nz = [d for d in diffs if d != 0]; n = len(nz)
    if n < 1: return 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i])); rank = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and abs(nz[order[j+1]]) == abs(nz[order[i]]): j += 1
        avg = (i+j)/2.0+1
        for k in range(i, j+1): rank[order[k]] = avg
        i = j+1
    wp = sum(rank[i] for i in range(n) if nz[i] > 0)
    if n <= 12:
        # EXACT two-sided signed-rank p by enumeration (audit: normal approx is anti-conservative at
        # small n — n=5 all-same-sign gives 0.043 under the approx but the exact minimum is 0.0625).
        ranks = [rank[i] for i in range(n)]
        from itertools import product
        sums = [sum(r for r, b in zip(ranks, bits) if b) for bits in product((0, 1), repeat=n)]
        mean = n*(n+1)/4.0
        dev = abs(wp - mean)
        p = sum(1 for s in sums if abs(s - mean) >= dev - 1e-9) / float(len(sums))
        return min(1.0, p)
    mean = n*(n+1)/4.0; sd = (n*(n+1)*(2*n+1)/24.0)**0.5
    if sd == 0: return 1.0
    z = (wp-mean)/sd
    return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))

def build_arms(mode):
    if mode == "sens":
        # KNOB SENSITIVITY + GUARD ABLATION (panel item 10): vary one frozen knob at a time around
        # the V5 learner; nofloor ~1% floor (beta=0.01 clamps period to 100), nowarm W=1,
        # nolock gamma=9999 (gate opens at first check -> LOCK disabled).
        iso = {"AMOR_REPAIR_REPLANONLY": "1"}
        v5 = {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "ts", "AMOR_RB_REWARD": "rawdelta", "AMOR_CU_DMIN": "0.15", **iso}
        return {"rand": {}, "v5": dict(v5),
                "b10": {**v5, "AMOR_CU_BETA": "0.10"}, "b30": {**v5, "AMOR_CU_BETA": "0.30"},
                "w50": {**v5, "AMOR_CU_WARMUP": "50"}, "w200": {**v5, "AMOR_CU_WARMUP": "200"},
                "g20": {**v5, "AMOR_CU_GAMMA": "0.20"}, "g40": {**v5, "AMOR_CU_GAMMA": "0.40"},
                "nofloor": {**v5, "AMOR_CU_BETA": "0.01"},
                "nowarm": {**v5, "AMOR_CU_WARMUP": "1"},
                "nolock": {**v5, "AMOR_CU_GAMMA": "9999"}}
    if mode == "magnitude":
        return {"greedy": {}, "rr2": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "2"},
                "rr5": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5"},
                "rr10": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "10"}}
    if mode == "sweep":
        return {"greedy": {}, "rr5": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5"}}
    if mode == "switch":
        base = {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5"}
        return {f"f{int(f*100):03d}": {**base, "AMOR_SWITCH_FRAC": str(f)}
                for f in (0.0, 0.25, 0.5, 0.75, 1.0)}
    if mode == "obandit":
        # THE LEARNING SMOKE: does the in-binary repair-order bandit (AMOR_REPAIR=20, eps-greedy over
        # 5 order rules) learn the per-map best order ONLINE, WITHOUT labels? GO iff bandit within ~1%
        # of the per-map best fixed arm on EVERY map (maze->longest, random->random) AND REPAIR_ARMS
        # shows it converged to the right arm. All arms replan-only for clean isolation.
        iso = {"AMOR_REPAIR_REPLANONLY": "1"}
        return {"rand": {}, "long": {"AMOR_REPAIR": "1", **iso},
                "mdel": {"AMOR_REPAIR": "3", **iso},
                "bandit": {"AMOR_REPAIR": "20", **iso},                                # 5-arm (B1-fixed)
                "bandit3": {"AMOR_REPAIR": "20", "AMOR_RB_ARMS": "0,1,3", **iso}}      # 3-arm mask (drop dominated)
    if mode == "cucb":
        # SMOKE B (pre-registered): AMOR-CUCB vs stock / fixed arms / naive eps-greedy.
        # ldel = single-arm bandit mask -> always least-delayed (no fixed mode exists for arm 4).
        iso = {"AMOR_REPAIR_REPLANONLY": "1"}
        return {"rand": {}, "long": {"AMOR_REPAIR": "1", **iso},
                "ldel": {"AMOR_REPAIR": "20", "AMOR_RB_ARMS": "4", **iso},
                "eps3": {"AMOR_REPAIR": "20", "AMOR_RB_ARMS": "0,1,4", **iso},
                "cucb": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "breaker", "AMOR_RB_ALPHA_BUDGET": "1.0", **iso},   # rereview C1: alpha=1 -> breaker trip unsatisfiable -> TRUE unguarded UCB-V (mode-1 path has no floor/warmup); gate-off necessity reference
                "cucbS": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "breaker", "AMOR_RB_REWARD": "slackfrac", **iso},  # SOC-aligned reward + UCB-V (fixes the random-map surrogate misalignment)
                "mixer": {"AMOR_REPAIR": "20", "AMOR_RB_ARMS": "0,1,4", "AMOR_RB_EPS": "1.0", **iso},   # NECESSITY control: uniform mixer, NO learning (eps=1 = always explore)
                "bgse": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "budget", "AMOR_CU_DMIN": "0.15", **iso},
                "bgts": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "ts", "AMOR_CU_DMIN": "0.15", **iso},   # BG-TS (pre-registered V3): budget-gated Thompson + anchor floor
                "roul": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "roulette", **iso},   # Ropke-Pisinger'06 roulette-wheel (ALNS necessity-table row, unguarded)
                "cts": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "cts", "AMOR_CU_DMIN": "0.15", **iso},   # V4 (pre-reg 2026-07-03): confidence-gated departure, SECONDARY endpoint
                "v5": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "ts", "AMOR_RB_REWARD": "rawdelta", "AMOR_CU_DMIN": "0.15", **iso},   # V5: magnitude-aware Bernoulli(rawdelta/16) channel (Phase-0-validated ranking)
                "etc50": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "etc", "AMOR_RB_REWARD": "rawdelta", **iso},   # FF-10 necessity: identify-then-commit, same channel as v5
                "mdel": {"AMOR_REPAIR": "20", "AMOR_RB_ARMS": "3", **iso},   # FF-10 necessity: fixed most-delayed
                "v5rr": {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "ts", "AMOR_RB_REWARD": "rawdelta", "AMOR_CU_DMIN": "0.15",
                         "AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5", **iso}}   # FF-10: learner + rr acceptance composition
    if mode == "stack21":
        # U1 STACKING GATE: does our repair-order learner add value ON TOP of ADDRESS-style destroy
        # at large N? AMOR_SEED=20 = in-binary ADDRESS destroy (orthogonal to AMOR_REPAIR=repair order).
        # The paper's key comparison is addr_v5 vs addr_stock (does v5 help on top of SOTA destroy).
        v5  = {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "ts", "AMOR_RB_REWARD": "rawdelta", "AMOR_CU_DMIN": "0.15", "AMOR_REPAIR_REPLANONLY": "1"}
        v3  = {"AMOR_REPAIR": "21", "AMOR_CU_MODE": "ts", "AMOR_CU_DMIN": "0.15", "AMOR_REPAIR_REPLANONLY": "1"}   # completion channel
        return {"rw_stock": {}, "rw_v5": dict(v5),                                    # RandomWalk destroy: stock vs our learner (standalone)
                "addr_stock": {"AMOR_SEED": "20"},                                    # ADDRESS destroy + stock repair order (SOTA baseline)
                "addr_v5":  {"AMOR_SEED": "20", **v5},                                # ADDRESS + our learner (THE additive-value test)
                "addr_bgts": {"AMOR_SEED": "20", **v3},                               # ADDRESS + V3 (secondary)
                "tack_stock": {"AMOR_SEED": "30"},                                    # TACKLE (mode 30 MABUC) destroy + stock order (2nd stack substrate, plan §4.2)
                "tack_v5":  {"AMOR_SEED": "30", **v5}}                                # TACKLE + our learner (does order compose with TACKLE seed-bandit too?)
    if mode == "couple":
        # THE "WE IMPROVE THE KING" TEST: do our best FIXED orthogonal criteria — record-to-record
        # ACCEPT (rr5, the proven -5.5% zero-overhead win) and longest-first REPAIR order — ENHANCE the
        # strongest SOTA destroy (ADDRESS mode20 / TACKLE mode30) vs SOTA-destroy-alone? Sub-additivity
        # (better destroy -> less stagnation -> smaller rr gain) is the risk; measured, not assumed.
        rr = {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5"}
        lo = {"AMOR_REPAIR": "1", "AMOR_REPAIR_REPLANONLY": "1"}    # longest-haul repair order (LNS-loop only)
        arms = {"rw_base": {},                                      # stock LNS2 (RandomWalk + random repair + greedy)
                "rw_rr":   dict(rr),                                 # stock destroy + our accept (baseline for "does SOTA destroy add?")
                "rw_ord":  dict(lo),                                 # stock destroy + our repair order
                "rw_both": {**lo, **rr}}                             # stock destroy + both
        for sd, nm in (("20", "addr"), ("30", "tack")):
            b = {"AMOR_SEED": sd}
            arms[f"{nm}_base"] = dict(b)                            # SOTA destroy alone = the baseline to beat
            arms[f"{nm}_rr"]   = {**b, **rr}                        # SOTA destroy + our ACCEPT
            arms[f"{nm}_ord"]  = {**b, **lo}                        # SOTA destroy + our REPAIR order
            arms[f"{nm}_both"] = {**b, **lo, **rr}                  # SOTA destroy + BOTH
        return arms
    if mode == "orderiso":
        # Isolate INIT-order vs REPLAN-order: is the longest-first win from a better initial solution
        # or from the LNS-loop repair order? rand=stock, long_comb=longest for init+replan,
        # long_replan=longest ONLY in replan (random init). If long_replan ~= long_comb -> genuine
        # repair-order effect; if long_replan collapses to rand -> it was an init-ordering artifact.
        return {"rand": {}, "long_comb": {"AMOR_REPAIR": "1"},
                "long_replan": {"AMOR_REPAIR": "1", "AMOR_REPAIR_REPLANONLY": "1"}}
    if mode == "order":
        # Angle-4 oracle-order test: does a fixed PP repair ORDER beat stock random at iso-wall-clock,
        # and is the best order MAP-SPECIFIC (=> a contextual order selector has residual to learn)?
        # AMOR_REPAIR: 0=random(stock) 1=longest-haul 2=shortest 3=most-delayed. Zero-overhead (a sort).
        return {"ord_rand": {}, "ord_long": {"AMOR_REPAIR": "1"},
                "ord_short": {"AMOR_REPAIR": "2"}, "ord_delay": {"AMOR_REPAIR": "3"}}
    if mode == "stack":
        # LOAD-BEARING 2x2 accept x destroy. A1 stock, A2 accept-only, A3 destroy-only(=SOTA), A4 stack.
        # Estimand = interaction: does rr still help ON TOP of a bandit destroy (orthogonal) or get absorbed?
        rr = {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5"}
        return {"A1_rw_greedy": {}, "A2_rw_rr": {**rr},
                "A3_bandit_greedy": {"AMOR_SEED": "20"}, "A4_bandit_rr": {"AMOR_SEED": "20", **rr}}
    if mode == "budget":
        # Mechanism proof: sweep budget B (via N or T at fixed map) -> rr-gain should FLIP sign
        # (loss when budget-starved, win when budget-rich). rr5_best tests if best-return fixes the loss side.
        return {"greedy": {}, "rr5": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5"},
                "rr5_best": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5", "AMOR_BEST_RETURN": "1"}}
    if mode == "gd":
        # Attribution ablation (avoid V12): does best-incumbent-RETURN fix rr's tail loss, and does the
        # great-deluge SCHEDULE add anything beyond best-return? A=greedy B=rr C=rr+best-return D=gd+best-return.
        return {"greedy": {}, "rr5": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5"},
                "rr5_best": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": "5", "AMOR_BEST_RETURN": "1"},
                "gd20_best": {"AMOR_ACCEPT": "gd", "AMOR_GD_D0": "20", "AMOR_BEST_RETURN": "1"},
                "gd40_best": {"AMOR_ACCEPT": "gd", "AMOR_GD_D0": "40", "AMOR_BEST_RETURN": "1"}}
    if mode == "delta":
        # Angle-2 oracle-delta test: sweep the rr threshold; is the best delta* MAP-SPECIFIC
        # (=> contextual delta=f(congestion) has residual over a single global delta)?
        return {"greedy": {}, **{f"d{d}": {"AMOR_ACCEPT": "rr", "AMOR_RR_DELTA": str(d)}
                                 for d in (2, 5, 10, 20, 40)}}
    sys.exit(f"bad mode {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["magnitude", "sweep", "switch", "order", "delta", "gd", "budget", "stack", "orderiso", "obandit", "cucb", "stack21", "sens", "couple"])
    ap.add_argument("--map", required=True)
    ap.add_argument("--agents", required=True, help="one N, or comma list for --mode sweep")
    ap.add_argument("--scens", default="1,2,3,4,5")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--times", default="30")
    ap.add_argument("--workers", type=int, default=1, help="arms-per-scen wave parallelism (<= physical cores)")
    ap.add_argument("--only", default=None, help="comma list: run only these arms of the mode (e.g. rand,ldel,mixer,cucb,bgts)")
    ap.add_argument("--maxiter", default="1000000000", help="cap iterations (iso-ITERATION ablation: throughput-confound decomposition)")
    ap.add_argument("--iterstats", default=None, help="dir for per-iteration incumbent CSVs (--stats passthrough; enables PI(T) anytime metric)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try: sys.stdout.reconfigure(line_buffering=True)  # so background runs are monitorable in the log
    except Exception: pass
    try:   # inline keepawake: ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED. DISPLAY_REQUIRED added
        # after Modern-Standby ("Idle Timeout" S0) froze a battery for 3.5h THROUGH plain SYSTEM_REQUIRED;
        # belt-and-braces with `powercfg /change standby-timeout-ac 0`. Lid close still suspends — keep it open.
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    except Exception: pass
    global MAXITER, ITERSTATS; MAXITER = args.maxiter
    if args.iterstats:
        ITERSTATS = args.iterstats; Path(ITERSTATS).mkdir(parents=True, exist_ok=True)
    binary = find_binary(); mapf = map_path(args.map)
    Ns = [int(x) for x in args.agents.split(",") if x]
    scens = [int(x) for x in args.scens.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    times = [float(x) for x in args.times.split(",") if x]
    arms = build_arms(args.mode)
    if args.only:
        want = [a for a in args.only.split(",") if a]
        miss = [a for a in want if a not in arms]
        if miss: sys.exit(f"--only unknown arms {miss}; available: {list(arms)}")
        arms = {a: arms[a] for a in want}
    ref = "greedy" if "greedy" in arms else "f100"  # switch mode baseline handled below

    print(f"# ACCEPT smoke mode={args.mode} | map={args.map} N={Ns} | binary={binary}")
    print(f"# arms={list(arms)} | scens={scens} seeds={seeds} times={times} | ISO-WALL-CLOCK serial\n")

    summary = {}
    raw = []       # full evidence chain (audit: CUCB diagnostics/raw SOC existed only in scrollback)
    dropped = {}   # arm -> {status: count} for cells excluded from pairing
    if args.workers > 1 and len(arms) > args.workers:
        # audit fix: ThreadPoolExecutor silently STAGGERS when arms > workers -> deterministic
        # arm-ordered contention skew (the same contamination class as the false 52%-capture read).
        sys.exit(f"wave mode needs workers >= #arms ({len(arms)}); got {args.workers}. Use --workers 1 (serial protocol) or trim --only.")
    for N in Ns:
        insts = [(sc, scen_path(args.map, sc), sd) for sc in scens for sd in seeds if scen_path(args.map, sc)]
        res = {t: {a: {} for a in arms} for t in times}
        itr = {t: {a: [] for a in arms} for t in times}
        lbs = {}   # (scen,seed) -> LB (sum of free-flow distances); delay = SOC - LB = the field's headline unit
        for (sc, sp, sd) in insts:
            for t in times:
                # WAVE-PARALLEL: all arms of the SAME scen run concurrently (one wave, #jobs<=workers)
                # -> shared contention hits every arm equally -> paired deltas stay fair. NOTE (calibration
                # 2026-07-02): absolute iteration counts drop 17% at k=2 already on the 155H laptop, so
                # HEADLINE/absolute numbers (learner capture etc.) must use --workers 1; waves are for
                # paired-delta screening only.
                if args.workers > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        futs = {a: ex.submit(run_one, binary, mapf, sp, N, sd, t, arms[a], f"{args.map}_N{N}_s{sc}_d{sd}_t{int(t)}_{a}") for a in arms}
                    outs = {a: futs[a].result() for a in arms}
                else:
                    outs = {a: run_one(binary, mapf, sp, N, sd, t, arms[a], f"{args.map}_N{N}_s{sc}_d{sd}_t{int(t)}_{a}") for a in arms}
                cells = []
                for a in arms:
                    rec = outs[a]
                    res[t][a][(sc, sd)] = rec["soc"]
                    if rec["lb"] is not None: lbs[(sc, sd)] = rec["lb"]
                    if rec["iters"] is not None: itr[t][a].append(rec["iters"])
                    if rec["status"] != "ok": dropped.setdefault(a, {}).setdefault(rec["status"], 0); dropped[a][rec["status"]] += 1
                    raw.append({"N": N, "scen": sc, "seed": sd, "t": t, "arm": a, **rec})
                    cells.append(f"{a}={rec['soc']}")
                print(f"  N{N} scen{sc} seed{sd} t{int(t)}s | " + "  ".join(cells) + f" | LB={lbs.get((sc, sd))}")
        for t in times:
            print(f"\n--- N={N} iso-wall-clock t={int(t)}s ---")
            base_arm = next((b for b in ("greedy", "ord_rand", "f000") if b in arms), list(arms)[0])
            bmap = res[t][base_arm]
            gi = statistics.mean(itr[t][base_arm]) if itr[t][base_arm] else 0
            for a in arms:
                keys = [k for k in res[t][a] if res[t][a][k] is not None and bmap.get(k) is not None]
                if not keys: print(f"  {a:<8} (no data)"); continue
                base = statistics.mean(bmap[k] for k in keys)
                m = statistics.mean(res[t][a][k] for k in keys)
                diffs = [res[t][a][k] - bmap[k] for k in keys]
                rel = 100*(m-base)/base if base else 0
                # DELAY units (delay = SOC - LB): the field's headline unit (BALANCE/ADDRESS/TACKLE all report delay)
                dkeys = [k for k in keys if lbs.get(k)]
                ddel = (100*statistics.mean((res[t][a][k]-lbs[k]) - (bmap[k]-lbs[k]) for k in dkeys)
                        / max(1, statistics.mean(bmap[k]-lbs[k] for k in dkeys))) if dkeys else None   # audit: 0.0 faked a null result when LB missing
                win = sum(1 for d in diffs if d < 0)
                p = wilcoxon_p(diffs)
                n_eff = sum(1 for d in diffs if d != 0)   # audit: p is computed on nonzero diffs; report both
                ai = statistics.mean(itr[t][a]) if itr[t][a] else 0
                summary[f"N{N}_t{int(t)}_{a}"] = {"rel%": round(rel,2), "delay%": (round(ddel,2) if ddel is not None else None),
                                                  "win": win, "n": len(keys), "n_eff": n_eff, "p": round(p,4), "iters": round(ai)}
                dstr = f"{ddel:+7.2f}%" if ddel is not None else "     NA"
                print(f"  {a:<8} SOC={m:9.1f} delta={rel:+6.2f}% DELAY={dstr} vs {base_arm}  win={win}/{len(keys)} p={p:.4f} iters~{ai:.0f}(x{(ai/gi if gi else 0):.2f})")
    if args.mode == "magnitude":
        print("\n# GO iff some rr wins >=3% at p<0.05 AND iters ~= greedy (zero-overhead). NO-GO => magnitude gone.")
    elif args.mode == "sweep":
        print("\n# GO iff rr5 sign is congestion-ordered: win at low/mid N, regress on the congested tail.")
    elif args.mode == "switch":
        print("\n# GO (Form A, strong ML) iff some interior f (f025/f050/f075) beats BOTH f000(greedy) and f100(rr) > noise.")
        print("# NO-GO => best is an endpoint => no within-run signal => ship zero-ML rr-port (Form B).")
    out = args.out or str(ROOT / f"results/accept_{args.mode}_{args.map}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"args": vars(args), "summary": summary,
                                     "dropped": dropped, "raw": raw}, indent=2))   # full evidence chain for paper tables
    if dropped: print(f"  DROPPED cells: {dropped}")
    # FF-7: symmetric-drop invariant check — init failures must hit ALL arms of an instance identically
    # (init is byte-identical across arms by construction: AMOR_REPAIR_REPLANONLY=1). Asymmetry = red flag.
    by_inst = {}
    for r in raw:
        if r["status"] == "initfail": by_inst.setdefault((r["N"], r["scen"], r["seed"], r["t"]), set()).add(r["arm"])
    asym = {str(k): sorted(v) for k, v in by_inst.items() if len(v) != len(arms)}
    if asym: print(f"  WARNING FF-7 ASYMMETRIC INIT-FAILURES (conditioning invariant violated): {asym}")
    print(f"\nsaved: {out}")

if __name__ == "__main__":
    main()
