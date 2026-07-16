# CARE — An Acceptance and Repair Layer for Anytime MAPF-LNS

Research code and data for CARE, a small, environment-gated **acceptance + repair layer** added
to five generations of MAPF-LNS solvers, leaving the destroy operator unchanged and using no
per-instance tuning. We evaluate it with within-instance pairing, a preregistered non-inferiority
("do-no-harm") protocol calibrated to the measured solver-noise floor, an anytime (t=300 s)
study, and a zero-shot generalization suite on 178 procedurally generated maps (a preregistered
study of 146,660 runs).

> **Status:** work in progress, being prepared for submission — not peer-reviewed, and the
> findings should be read as preliminary.

- **Frozen experimental design / preregistration:** [`PREREGISTRATION.md`](PREREGISTRATION.md)
- **Aggregated per-cell results:** [`results/`](results/)
- **License:** MIT for the CARE layer + harness; vendored upstream solvers keep their own
  licenses — see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).

## Hosts (source included, layer env-gated, default = vanilla)

| host | paper | binary |
|---|---|---|
| `lns2` | MAPF-LNS2 (AAAI'22) | `src/lns2/lns` |
| `balance` | BALANCE (AAAI'24) | `src/balance/balance` |
| `address` | ADDRESS (AAAI'25) | `src/address/address` |
| `tackle` | TACKLE (AAAI'26) | `src/tackle/tackle` |
| `tackle_alns` | MAPF-LNS canonical ALNS | same binary, `--algorithm=canonical` |

With **no** `L2_/BL_/AD_/TK_` environment variable set, every binary reproduces its upstream
vanilla behavior. Every layer change to a host source is marked with a `*_PORT` / `AMOR_*`
comment, so diffing against the upstream repository yields exactly the layer (see `NOTICE.md`).
The only extra in `stock` runs is an env-gated telemetry `makespan` CSV column on `lns2`,
enabled by the runner for all arms symmetrically.

## The layer (all arms available on all five hosts)

- **Accept** (`<PFX>ACCEPT`): `spsa` (state-conditioned record-to-record threshold, online
  sign-SPSA-tuned), `rr` (fixed threshold δ), `cart` (budget-decayed threshold), `sa`/`ta`
  (textbook simulated-annealing / threshold-accepting baselines). All anchored on the best
  incumbent, all with a universal best-return that makes non-greedy acceptance anytime-safe.
- **Repair** (`<PFX>REPAIR`): `v5` (Thompson-sampling safe-arm replan-priority bandit), `20`
  (ε-greedy variant).
- **Confirmatory arm (preregistered, exactly one):** `full` = `ACCEPT=spsa REPAIR=v5` (= CARE).
  Everything else (`spsa_only`, `v5_only`, `rr5_v5`, `spsa_20`, `cart_v5`, `rr5_20`, `rr20_v5`,
  `rr50_v5`) is an exploratory mechanism decomposition **fused into the same battery** — every
  ablation is extracted from the same paired scenarios, never from separate small runs.

## Repository layout

```
src/               four host codebases (vendored upstream + env-gated layer; see NOTICE.md)
experiments/       config generators + battery runner (checkpoint / atomic / paired blocks)
analysis/          statistics stack (paired HL + BCa CIs, do-no-harm, anytime) + smoke validator
setup/             dataset download / build / generated-map pipeline / N-probe
data/gen_maps/     MANIFEST + generator for the 178 generated maps (the maps themselves are a
                   Release asset / regenerated locally; see "Datasets")
results/           aggregated per-cell tables underlying our reported results
development/       pre-preregistration screening: fixed replanning orders, a
                   no-learning uniform-mixture control, bandit variants (see its README)
PREREGISTRATION.md frozen experimental design
LICENSE, NOTICE.md MIT (layer + harness) + third-party attributions
```

## Requirements

- **C++ build:** `cmake >= 3.10`, `g++ >= 9`, `libboost-all-dev` (program_options, system,
  filesystem), `libeigen3-dev`.
- **Python:** `python3 >= 3.8` (runner and generators are stdlib-only); `numpy` + `scipy` for
  `analysis/aggregate_aaai.py`.

## Datasets

- **Official** (`source=official`): 33 MovingAI maps + the standard 25 `random` scenarios,
  downloaded by `setup/00_download_movingai.sh`. Spine-8 maps are swept over 4 congestion
  levels ({0.4,0.6,0.8,1.0}×N*).
- **Generated** (`source=gen:<family>`): 8 families frozen by a per-file md5 in
  `data/gen_maps/MANIFEST.json` — `grandom`, `gmaze` (perfect maze, **preregistered null
  control**), `gmazeb` (braided maze), `groom`, `gware`, `gtiles` (real-city 64×64 tiles),
  `gpogr`/`gpogm` (POGEMA-generated anchors). To obtain them, either download the
  `gen_maps.zip` Release asset and extract into `data/gen_maps/`, **or** regenerate byte-identically:
  ```bash
  python3 setup/02_gen_maps.py            # regenerate from the frozen MANIFEST
  python3 setup/02_gen_maps.py --check    # verify byte-identical to MANIFEST md5s
  ```

## Reproduction

```bash
# 1. deps (see Requirements), then:
bash setup/00_download_movingai.sh          # 33 official maps + 25 scenarios each
bash setup/01_build_all.sh                  # build the 4 host binaries
python3 setup/02_gen_maps.py                # generated maps (or extract gen_maps.zip)
python3 setup/03_nprobe.py                  # machine 1 ONLY (~1 h feasibility); m2 copies output
./run_all.sh smoke                          # MANDATORY gate (~30-40 min): all hosts x all arms

# machine 1 (80 cores) — configs are generated ONCE here:
nohup ./run_all.sh m1 > run_m1.log 2>&1 &   # LOCK first, then machine 1's shard
# copy experiments/gen_n_final.json + experiments/configs/ to machine 2 (same paths)

# machine 2 (64 cores) — uses the COPIED, md5-verified configs (generating its own would
# desynchronize the deterministic shard split):
nohup ./run_all.sh m2 > run_m2.log 2>&1 &

# aggregate (after copying machine 2's results/main/ into machine 1):
python3 analysis/aggregate_aaai.py
```

Everything is checkpointed and resumable; re-running the same command continues where it
stopped. Batches run in information-density order: **L** (decontention lock, low concurrency) →
**E0** (solver-noise ε calibration) → **A** (official 33-map fused battery) → **B** (t=300 s
anytime trajectories) → **G** (generated maps). A single-machine mode is available via
`run_all.sh` (see the script header).

## Results

Per-run JSON in `results/` (metrics + arm env + binary md5 + telemetry), aggregated by
`analysis/aggregate_aaai.py` into per-cell effects with BCa 95% CIs, per-host exact binomials, a
do-no-harm ledger against the measured per-regime noise floor, the fused ablation extraction,
anytime Δ(t) tables, and generated-map per-family ECDFs. The aggregated tables underlying our
reported results are provided under [`results/`](results/) with a per-file guide in
[`results/README.md`](results/README.md).

## License

The CARE layer and the entire harness (`experiments/`, `analysis/`, `setup/`, top-level files)
are released under the MIT License ([`LICENSE`](LICENSE)). The vendored upstream solvers under
`src/` retain their own licenses; see [`NOTICE.md`](NOTICE.md).
