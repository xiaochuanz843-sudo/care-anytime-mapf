#!/usr/bin/env python3
"""Phase-B battery runner (v1.3). Runs ON a box, fully autonomous (survives SSH loss), stdlib only.

Guarantees (per the preregistered design, PREREGISTRATION.md §5/§7):
  * paired blocks: all arms of one (map,N,scen[,rep]) run SEQUENTIALLY in one worker slot,
    arm order shuffled per block (crc32-keyed => stable across resumes; hash() is salted);
    blocks globally shuffled; dual-pool scheduling (B8): non-CACHE blocks on a
    min(token_budget, core_cap)-worker executor, CACHE blocks fill the remaining cores
    on a second executor, both submitted concurrently.
  * bandwidth-token weighted semaphore + hard core cap (asserted feasible at init).
  * startup preflight (B2): every binary executable + every map/scen file present, else exit(1).
  * circuit breaker (B3): fail>=30 with done==0, or fail>100 and fail>30% => ABORT + stop.
  * atomic per-run result files (pid-unique tmp + os.replace) => idempotent resume.
  * failures recorded, never silent; per-run telemetry (wall, RSS, iterations, loadavg, host),
    plus the actual arm env dict and the per-SOTA binary md5 computed once at startup (B7).
  * single-instance pidfile mutex (O_EXCL + liveness probe), triple-hardened (B1):
    live pid => REFUSED; dead pid, corrupt/empty pidfile (re-read once), or POSIX PID reuse
    (/proc/<pid>/cmdline without "runner.py") => stale takeover. Removed after ALL-DONE/DRY-DONE.
  * timed-out runs killed as a whole process group and reaped (no orphan solvers);
    run-level "timeout" field overrides cfg run_timeout (B6, t=300 layer uses 660).
  * SIGTERM and SIGINT both request a clean stop (B7); stale results/*.tmp cleaned after the
    mutex is held (B7).
  * arm env is injected into a sanitized environment (all TK_/AD_/BL_ vars stripped first).
  * CSV disposition: P-block CSVs moved to workdir/pilot_csv/ (never cleaned, B4); A/B stock+full
    -LNS.csv of cfg["keep_csv_cells"] cells moved to workdir/curves_csv/ (B5); the rest deleted.

Usage:  python3 runner.py --config configs/m1.json [--root PKG] [--dry] [--only A,B] [--shuffle-seed 7]
"""
import argparse, glob, hashlib, json, os, random, re, shlex, shutil, signal, socket, subprocess, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def pid_alive(pid_s):
    """True=alive, False=surely dead, None=unknown/unparsable (caller treats None as alive)."""
    try:
        pid = int(str(pid_s).strip())
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if os.name == "posix":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True
    import ctypes                                    # Windows (local tests only)
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, 0, pid)              # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return code.value == 259                 # STILL_ACTIVE
        return None
    finally:
        k32.CloseHandle(h)

def pid_parsable(pid_s):
    """Is the pidfile content a plausible pid? (B1: corrupt/empty => stale after one re-read)."""
    try:
        return int(str(pid_s).strip()) > 0
    except (TypeError, ValueError):
        return False

def pid_cmdline(pid_s):
    """POSIX PID-reuse probe (B1): /proc/<pid>/cmdline as one string, None if unavailable
    (non-POSIX, no /proc, or the process vanished)."""
    try:
        pid = int(str(pid_s).strip())
    except (TypeError, ValueError):
        return None
    if os.name != "posix" or pid <= 0:
        return None
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return None

class WeightedSem:
    def __init__(self, budget):
        self.free = budget; self.cv = threading.Condition()
    @contextmanager
    def take(self, w):
        if w <= 0:
            yield; return
        with self.cv:
            while self.free < w: self.cv.wait()
            self.free -= w
        try:
            yield
        finally:
            with self.cv:
                self.free += w; self.cv.notify_all()

class Runner:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        # AAAI package: config paths are RELATIVE to the package root (portable across boxes).
        # Default root: configs live at <root>/experiments/configs/<x>.json => three levels up.
        root = os.path.abspath(args.root) if args.root else \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(args.config))))
        def resolve(p):
            return p if (p is None or os.path.isabs(p)) else os.path.join(root, p)
        for key in ("results_dir", "workdir", "mapdir", "scendir", "traj_dir"):
            if cfg.get(key): cfg[key] = resolve(cfg[key])
        cfg["bin"] = {k: resolve(v) for k, v in cfg["bin"].items()}
        self.rdir = cfg["results_dir"]; os.makedirs(self.rdir, exist_ok=True)
        self.wdir = cfg["workdir"]; os.makedirs(self.wdir, exist_ok=True)
        self.pilot_dir = os.path.join(self.wdir, "pilot_csv")    # B4: P-block CSVs, never cleaned
        self.curves_dir = os.path.join(self.wdir, "curves_csv")  # B5: headline convergence curves
        os.makedirs(self.pilot_dir, exist_ok=True); os.makedirs(self.curves_dir, exist_ok=True)
        self.traj_dir = cfg.get("traj_dir")   # anytime .traj kept here (ZT/T3); None => delete all
        if self.traj_dir:
            os.makedirs(self.traj_dir, exist_ok=True)
        # B5: whitelist cells whose A/B stock/full -LNS.csv are preserved (config-driven)
        self.keep_csv = {(c[0], c[1], int(c[2])) for c in cfg.get("keep_csv_cells", [])}
        self.tokens = cfg["tokens"]
        self.cap_out = bool(cfg.get("capture_stdout"))   # opt-in: parse REPAIR_ARMS from stdout
        self.sem = WeightedSem(cfg["token_budget"])
        self.log_path = os.path.join(self.wdir, "progress.log")
        self.stop = threading.Event()
        self.aborted = False            # set by the circuit breaker => process exits 2
        self.lock = threading.RLock()   # RE-ENTRANT: log() is called under the counter lock (fix R-1)
        self.done = 0; self.failed = 0; self.skipped = 0
        self.total_runs = sum(len(b["runs"]) for b in self.blocks())
        maxw = max((self.tokens.get(b["level"], 0) for b in self.blocks()), default=0)
        assert maxw <= cfg["token_budget"], \
            f"max block weight {maxw} > token_budget {cfg['token_budget']}: semaphore would deadlock"
        self.time_ok = self._probe_time()
        self.host = socket.gethostname()
        # B7: per-SOTA binary md5, computed ONCE at startup; recorded in every result JSON
        self.bin_md5 = {}
        for s in sorted({r["sota"] for b in self.blocks() for r in b["runs"]}):
            try:
                h = hashlib.md5()
                with open(cfg["bin"][s], "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                self.bin_md5[s] = h.hexdigest()
            except (OSError, KeyError):
                self.bin_md5[s] = None                 # preflight (B2) aborts real runs anyway
        self.t0 = time.time()

    def blocks(self):
        bl = self.cfg["blocks"]
        if self.args.only:
            phases = set(self.args.only.split(","))
            bl = [b for b in bl if b["phase"] in phases]
        return bl

    def log(self, msg):
        line = f"[{now()}] {msg}\n"
        with self.lock:
            try:
                with open(self.log_path, "a") as f: f.write(line)
            except OSError:
                pass                       # logging must never kill the battery

    # ---------- per-run ----------
    def rpath(self, rid):   return os.path.join(self.rdir, rid + ".json")
    def is_done(self, rid): return os.path.exists(self.rpath(rid))

    def save(self, rid, obj):
        tmp = self.rpath(rid) + f".{os.getpid()}.tmp"   # pid-unique: no cross-process clobber
        with open(tmp, "w") as f: json.dump(obj, f)
        os.replace(tmp, self.rpath(rid))   # atomic on POSIX

    def _probe_time(self):
        """Is /usr/bin/time -v usable? Decided ONCE at startup."""
        try:
            return subprocess.run(["/usr/bin/time", "-v", "true"], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=10).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def preflight(self):
        """B2: EVERY referenced binary must be executable and EVERY map/scen file must exist,
        else exit(1) before the battery touches anything."""
        missing = []
        sotas = sorted({r["sota"] for b in self.blocks() for r in b["runs"]})
        for s in sotas:
            p = self.cfg["bin"].get(s)
            if not (p and os.path.isfile(p) and os.access(p, os.X_OK)):
                missing.append(f"bin:{s}:{p}")
        files = set()
        for b in self.blocks():
            for r in b["runs"]:
                files.add(os.path.join(self.cfg["mapdir"], r["map"] + ".map"))
                files.add(os.path.join(self.cfg["scendir"], f"{r['map']}-random-{r['scen']}.scen"))
        missing += [p for p in sorted(files) if not os.path.isfile(p)]
        # scen-capacity check: every run's N must fit its scenario file's agent count
        cap = {}
        for b in self.blocks():
            for r in b["runs"]:
                sf = os.path.join(self.cfg["scendir"], f"{r['map']}-random-{r['scen']}.scen")
                if sf not in cap and os.path.isfile(sf):
                    try:
                        with open(sf) as fh:
                            cap[sf] = sum(1 for ln in fh if ln.strip()) - 1   # minus header line
                    except OSError:
                        cap[sf] = None
                if cap.get(sf) is not None and r["N"] > cap[sf]:
                    missing.append(f"scen_capacity:{os.path.basename(sf)}:N{r['N']}>cap{cap[sf]}")
        if missing:
            self.log(f"PREFLIGHT-FAIL missing={len(missing)} first={missing[:5]}")
            print(f"[runner] PREFLIGHT-FAIL: {len(missing)} missing artifacts, e.g. "
                  f"{missing[:5]} -- run the setup/ steps first (see README.md).", file=sys.stderr)
            sys.exit(1)
        self.log(f"PREFLIGHT-OK sotas={len(sotas)} files={len(files)}")

    def _run_reaped(self, argv, env, capture_err=False, timeout=None, capture_out=False):
        """Run argv in its own session/process-group; on timeout SIGKILL the WHOLE group and
        reap the zombie (communicate). Returns (returncode, stderr_text|None, stdout_text|None);
        re-raises TimeoutExpired after cleanup so callers record fail/timeout.
        timeout: run-level override (B6), falls back to cfg run_timeout.
        capture_out (opt-in, cfg capture_stdout): pipe solver stdout so callers can parse the
        REPAIR_ARMS n: line (bandit arm-pull telemetry). /usr/bin/time writes to stderr, so
        stdout stays the solver's cout only."""
        kw = dict(env=env, stdout=subprocess.PIPE if capture_out else subprocess.DEVNULL,
                  stderr=subprocess.PIPE if capture_err else subprocess.DEVNULL)
        if capture_err or capture_out:
            kw["text"] = True
        if os.name == "posix":
            kw["start_new_session"] = True             # own pgid => killpg reaches grandchildren
        p = subprocess.Popen(argv, **kw)
        try:
            out, err = p.communicate(timeout=timeout or self.cfg["run_timeout"])
            return p.returncode, err, out
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                else:
                    p.kill()
            except OSError:
                try: p.kill()
                except OSError: pass
            try:
                p.communicate(timeout=30)              # reap + drain pipes (no orphans/zombies)
            except (subprocess.TimeoutExpired, OSError):
                pass
            raise

    def build_cmd(self, run):
        c = self.cfg
        mapf = os.path.join(c["mapdir"], run["map"] + ".map")
        scenf = os.path.join(c["scendir"], f"{run['map']}-random-{run['scen']}.scen")
        out = os.path.join(self.wdir, "tmp_out", run["run_id"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subst = {"bin": c["bin"][run["sota"]], "map": mapf, "scen": scenf, "N": str(run["N"]),
                 "t": str(run.get("t") or c["t_limit"]),   # run-level budget override
                 "out": out, "seed": str(run["seed"]),
                 "k": str(min(32, int(run["N"]))),  # tackle top-k: k>N overflows tackle_small()
                 "maxiter": c["maxiter"]}
        argv = [tok.format(**subst) for tok in c["cli"][run["sota"]]]
        return argv, out

    def parse_csv(self, out):
        f = out + "-LNS.csv"
        if not os.path.exists(f): return None, "no_csv"
        try:
            with open(f) as fh: rows = [r.rstrip("\n") for r in fh]
        except OSError: return None, "csv_read_err"
        if len(rows) < 2: return None, "csv_short"
        hdr = [h.strip().lower() for h in rows[0].split(",")]
        last = rows[-1].split(",")
        def col(name):
            try: return float(last[hdr.index(name)])
            except (ValueError, IndexError): return None
        soc = col("solution cost"); sd = col("sum of distance")
        iters = col("iterations"); auc = col("area under curve"); rt = col("runtime")
        ms = col("makespan")     # lns2-only (L2_MKSPAN); other hosts have no such column -> None
        init_soc = col("initial solution cost")   # AUC shared-init check (aggregate_aaai)
        if soc is None or sd is None or soc <= 0 or sd <= 0: return None, "csv_bad_vals"
        if auc is None or iters is None or rt is None: return None, "csv_missing_cols"
        delay = soc - sd
        if delay < 0: return None, "negative_delay"
        return {"delay": delay, "soc": soc, "sum_dist": sd, "init_soc": init_soc,
                "iterations": iters, "auc": auc, "runtime": rt, "makespan": ms}, None

    def exec_run(self, run):
        rid = run["run_id"]
        if self.is_done(rid):
            with self.lock: self.skipped += 1
            return
        argv, out = self.build_cmd(run)
        for ext in ("-LNS.csv", "-initLNS.csv", ".traj"):   # stale-output hygiene before running
            try: os.remove(out + ext)
            except OSError: pass
        env = {k: v for k, v in os.environ.items()      # arm hygiene: no inherited arm vars (B P0-2)
               if not k.startswith(("TK_", "AD_", "BL_", "L2_"))}
        env.update(run["env"]); env["OMP_NUM_THREADS"] = "1"
        traj_path = None                                 # anytime trajectory dump (ZT/T3, traj-hosts)
        if run.get("traj_env"):
            traj_path = out + ".traj"
            env[run["traj_env"]] = traj_path             # e.g. L2_TRAJ=/.../<rid>.traj
        try: load1 = open("/proc/loadavg").read().split()[0]
        except OSError: load1 = ""
        cmd = (["/usr/bin/time", "-v"] + argv) if self.time_ok else argv
        rto = run.get("timeout") or self.cfg["run_timeout"]      # B6: run-level timeout override
        t0 = time.time(); status, reason, rss_kb = "ok", None, None
        stdout_txt = None                                # opt-in solver stdout (arm-pull capture)
        try:
            rc, err, stdout_txt = self._run_reaped(cmd, env, capture_err=self.time_ok,
                                                   timeout=rto, capture_out=self.cap_out)
            if self.time_ok:
                m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err or "")
                if m: rss_kb = int(m.group(1))
            if rc != 0: status, reason = "fail", f"rc={rc}"
        except subprocess.TimeoutExpired:
            status, reason = "fail", "timeout"
        except FileNotFoundError:
            if self.time_ok:                            # time vanished mid-battery: degrade to bare
                try:
                    rc, _, stdout_txt = self._run_reaped(argv, env, timeout=rto,
                                                         capture_out=self.cap_out)
                    if rc != 0: status, reason = "fail", f"rc={rc}"
                except subprocess.TimeoutExpired:
                    status, reason = "fail", "timeout"
                except OSError as e:
                    status, reason = "fail", f"exec_err:{type(e).__name__}"
            else:
                status, reason = "fail", "exec_err:FileNotFoundError"
        except OSError as e:                            # spawn failure (ENOMEM/EACCES/...)
            status, reason = "fail", f"exec_err:{type(e).__name__}"
        wall = time.time() - t0
        metrics = None
        if status == "ok":
            metrics, perr = self.parse_csv(out)
            if metrics is None: status, reason = "fail", perr
            elif traj_path and os.path.exists(traj_path):   # enrich from the anytime trajectory
                try:
                    import traj_util
                    tr = traj_util.parse_traj(traj_path, metrics["sum_dist"])
                    if tr:
                        metrics["traj"] = {"delay_at": tr["delay_at"], "auc_delay": tr["auc_delay"],
                                           "final_delay": tr["final_delay"], "n_points": tr["n_points"]}
                except Exception as e:
                    self.log(f"TRAJ-PARSE-ERR {rid} {type(e).__name__}: {e}")
            if self.cap_out and stdout_txt and metrics is not None:   # bandit arm-pull telemetry
                m2 = re.search(r"REPAIR_ARMS n:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", stdout_txt)
                if m2:                                # arms: random longest shortest most-del least-del
                    metrics["repair_arms"] = [int(m2.group(i)) for i in range(1, 6)]
        rec = {**{k: run[k] for k in ("run_id","block_id","sota","map","N","scen","rep","arm","seed")},
               "source": run.get("source", "official"),
               "status": status, "reason": reason, "wall_s": round(wall, 2),
               "rss_kb": rss_kb, "loadavg_pre": load1, "machine": self.cfg["machine"],
               "host": self.host,
               "t_limit": run.get("t") or self.cfg["t_limit"],   # effective budget (run-level t wins)
               "run_timeout": rto,                                # B6: effective runner timeout
               "env": run["env"],                                 # B7: actual arm env injected
               "bin_md5": self.bin_md5.get(run["sota"]),          # B7: startup binary fingerprint
               "finished_at": now(),
               "metrics": metrics}
        self.save(rid, rec)
        phase = run["block_id"].split(".")[0]
        if traj_path and os.path.exists(traj_path):     # .traj disposition: keep anytime, else delete
            if phase in ("B", "S2") and self.traj_dir:  # S2: smoke exercises the archive path too
                try: os.replace(traj_path, os.path.join(self.traj_dir, os.path.basename(traj_path)))
                except OSError: pass
            else:
                try: os.remove(traj_path)
                except OSError: pass
        if phase == "P":
            # B4: pilot CSVs live in workdir/pilot_csv/ (AUC/variance audit; NEVER cleaned)
            for ext in ("-LNS.csv", "-initLNS.csv"):
                src = out + ext
                if os.path.exists(src):
                    try: os.replace(src, os.path.join(self.pilot_dir, os.path.basename(src)))
                    except OSError: pass
        else:
            # B5: headline convergence curves — whitelisted A/B stock/full -LNS.csv preserved
            if (phase in ("A", "B") and run["arm"] in ("stock", "full")
                    and (run["sota"], run["map"], run["N"]) in self.keep_csv):
                src = out + "-LNS.csv"
                if os.path.exists(src):
                    try: os.replace(src, os.path.join(self.curves_dir, os.path.basename(src)))
                    except OSError: pass
            for ext in ("-LNS.csv", "-initLNS.csv"):    # keep disk bounded (JSON has the numbers)
                try: os.remove(out + ext)
                except OSError: pass
        with self.lock:
            if status == "ok": self.done += 1
            else: self.failed += 1
            d, f, s = self.done, self.failed, self.skipped
        if status != "ok":
            self.log(f"FAIL {rid} {reason}")
            self._check_breaker(d, s, f)

    def _check_breaker(self, d, s, f):
        # B3: circuit breaker — systemic failure must not burn the whole battery.
        # BOTH rules count skipped (disk-resident successes): on a resume/retry pass only the
        # failed runs are pending, so excluding skipped would make e.g. 150 known-hard retries
        # look like a 100% failure rate and falsely abort a nearly-complete battery.
        if (f >= 30 and d + s == 0) or (f > 100 and f > 0.30 * (d + s + f)):
            if not self.stop.is_set():
                self.log(f"ABORT circuit-breaker: done={d} skip={s} fail={f} "
                         f"(rule: fail>=30&done+skip==0 | fail>100&fail>30%-of-attempted) -- stopping")
                self.aborted = True
                self.stop.set()

    # ---------- block ----------
    def exec_block(self, block):
        if self.stop.is_set(): return
        w = self.tokens.get(block["level"], 0)
        with self.sem.take(w):
            runs = list(block["runs"])
            random.Random(zlib.crc32(block["block_id"].encode())).shuffle(runs)  # stable arm order
            for run in runs:
                if self.stop.is_set(): return
                try:
                    self.exec_run(run)
                except Exception as e:      # one crashed run must not kill the battery; log-only,
                    with self.lock:         # NO result file => the run is retried on resume
                        self.failed += 1
                        d, s, f = self.done, self.skipped, self.failed
                    self.log(f"CRASH {run['run_id']} {type(e).__name__}: {e}")
                    self._check_breaker(d, s, f)   # a crash storm must trip the breaker too

    # ---------- main ----------
    def heartbeat(self):
        while not self.stop.wait(self.cfg.get("heartbeat_s", 60)):
            with self.lock:
                d, f, s = self.done, self.failed, self.skipped
            el = time.time() - self.t0
            self.log(f"HB done={d} fail={f} skip={s} of {self.total_runs} "
                     f"elapsed={el/60:.1f}m free_tokens={self.sem.free}")

    def _read_pidfile(self, pid_path):
        try:
            with open(pid_path) as f: return f.read()
        except OSError:
            return ""

    def _acquire_pidfile(self, pid_path):
        """O_EXCL mutex, triple-hardened (B1):
        - live pid running runner.py            => REFUSE dual start (exit 1)
        - dead pid                              => stale takeover
        - corrupt/empty pidfile (re-read once)  => stale takeover
        - POSIX PID reuse (/proc cmdline without "runner.py") => stale takeover."""
        while True:
            try:
                fd = os.open(pid_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return
            except FileExistsError:
                time.sleep(0.2)                        # let a just-starting winner write its pid
                other = self._read_pidfile(pid_path)
                if not pid_parsable(other):            # corrupt/empty: ONE re-read, then stale
                    time.sleep(0.5)
                    other = self._read_pidfile(pid_path)
                    if not pid_parsable(other):
                        self.log(f"PIDFILE stale takeover (corrupt/empty content {other[:32]!r})")
                        try: os.remove(pid_path)
                        except OSError: pass
                        continue
                stale = pid_alive(other) is False      # surely-dead pid
                if not stale:
                    cmd = pid_cmdline(other)           # POSIX: PID reused by unrelated process?
                    if cmd is not None and "runner.py" not in cmd:
                        stale = True
                if stale:
                    self.log(f"PIDFILE stale takeover (pid {other.strip()})")
                    try: os.remove(pid_path)           # stale (crashed/killed runner): take over
                    except OSError: pass
                    continue
                print(f"[runner] REFUSED: pidfile {pid_path} held by live pid "
                      f"{other.strip() or '?'} -- a second runner would corrupt pairing. "
                      f"Stop it (or remove a stale pidfile) first.", file=sys.stderr)
                sys.exit(1)

    def run(self):
        self.preflight()                               # B2: refuse to start on missing artifacts
        pid_path = os.path.join(self.wdir, "runner.pid")
        self._acquire_pidfile(pid_path)
        try:
            # safe only under the mutex: no other runner is using tmp_out / writing results
            shutil.rmtree(os.path.join(self.wdir, "tmp_out"), ignore_errors=True)
            for f in glob.glob(os.path.join(self.rdir, "*.tmp")):   # B7: stale tmp hygiene
                try: os.remove(f)
                except OSError: pass
            blocks = self.blocks()
            pend = [b for b in blocks
                    if any(not self.is_done(r["run_id"]) for r in b["runs"])]
            rng = random.Random(self.args.shuffle_seed)
            noncache = [b for b in pend if self.tokens.get(b["level"], 0) > 0]
            cache    = [b for b in pend if self.tokens.get(b["level"], 0) == 0]
            rng.shuffle(noncache); rng.shuffle(cache)
            ordered = noncache + cache                       # (dry listing order only)
            # B8 dual-pool split: bandwidth lane gets min(budget, cores) workers, CACHE lane
            # fills the remaining cores; both lanes are saturated from t=0.
            cap = self.cfg["core_cap"]
            n_nc = max(1, min(self.cfg["token_budget"], cap))
            n_ca = 0 if n_nc >= cap else max(1, cap - n_nc)   # budget unbound (999) => ONE pool
            self.log(f"START machine={self.cfg['machine']} blocks={len(ordered)} "
                     f"(noncache={len(noncache)} cache={len(cache)}) total_runs={self.total_runs} "
                     f"budget={self.cfg['token_budget']} cores={self.cfg['core_cap']} "
                     f"pools=nc:{n_nc}+cache:{n_ca} dry={self.args.dry}")
            if not self.time_ok:
                self.log("WARN /usr/bin/time -v unavailable -> RSS telemetry degraded (rss_kb=None)")
            if self.args.dry:
                seen = set()                                 # cover EVERY (sota,arm) combo incl. env
                for b in blocks:
                    for r in b["runs"]:
                        key = (r["sota"], r["arm"])
                        if key in seen: continue
                        seen.add(key)
                        argv, _ = self.build_cmd(r)
                        self.log(f"DRY {b['block_id']} lvl={b['level']} sota={r['sota']} "
                                 f"arm={r['arm']} env={json.dumps(r['env'])} :: {shlex.join(argv)}")
                self.log(f"DRY-DONE combos={len(seen)}")
                return
            for sig in (signal.SIGTERM, signal.SIGINT):           # B7: SIGINT == SIGTERM
                try: signal.signal(sig, lambda *a: self.stop.set())
                except (OSError, ValueError): pass
            hb = threading.Thread(target=self.heartbeat, daemon=True); hb.start()
            # B8: two executors submitted CONCURRENTLY; every future collected (per-run
            # exceptions are already swallowed inside exec_block, this catches pool-level ones)
            with ThreadPoolExecutor(max_workers=n_nc) as ex_nc, \
                 ThreadPoolExecutor(max_workers=max(1, n_ca)) as ex_ca:
                if n_ca == 0:     # single-pool: cache blocks share the full-width executor
                    futs = [ex_nc.submit(self.exec_block, b) for b in noncache + cache]
                else:
                    futs = [ex_nc.submit(self.exec_block, b) for b in noncache]
                    futs += [ex_ca.submit(self.exec_block, b) for b in cache]
                for fu in futs:
                    try:
                        fu.result()
                    except Exception as e:
                        self.log(f"CRASH block-future {type(e).__name__}: {e}")
            self.stop.set()
            with self.lock:
                self.log(f"ALL-DONE done={self.done} fail={self.failed} skip={self.skipped} "
                         f"of {self.total_runs} wall={(time.time()-self.t0)/3600:.2f}h"
                         + (" [ABORTED by circuit breaker]" if self.aborted else ""))
        finally:
            try: os.remove(pid_path)                         # battery over -> release the mutex
            except OSError: pass
        if self.aborted:
            # orchestration must SEE the truncation (review finding: exit 0 after ABORT let
            # run_all.sh march on through every later phase and print DONE)
            print("[runner] ABORTED by circuit breaker -- see progress.log; exiting 2",
                  file=sys.stderr)
            sys.exit(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="", help="package root; default = parent of config dir")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="", help="comma list of phases, e.g. E0 or A,B,G")
    ap.add_argument("--shuffle-seed", type=int, default=7)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    Runner(cfg, args).run()

if __name__ == "__main__":
    main()
