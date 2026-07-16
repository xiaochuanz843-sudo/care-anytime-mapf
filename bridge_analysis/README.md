# Bridge experiment — final-binary attribution controls

The factorial control matrix that separates (i) closing the greedy early break,
(ii) the acceptance rule, and (iii) the replanning-order policy — the attribution
controls that the main battery does not contain. Everything runs on the frozen final
binaries via env knobs only (`experiments/gen_bridge.py`; zero code changes).

**Design.** 4 official maps (warehouse-20-40-10-2-2, Paris_1_256, den520d,
random-32-32-20) x {0.6, 1.0}xN* x 5 hosts x 25 scenarios x 10 arms = **10,000 runs**
@ t=60 s. Within every (host, map, N, scenario) block all ten arms share one seed, so
every contrast is scenario-paired. Executed on a 64-core node (62-way concurrency),
wall clock 3.47 h, **10,000/10,000 ok — zero failures, zero skips, zero incomplete
cells**; per-run `bin_md5` recorded (four builds across five hosts; the canonical-ALNS
host is the TACKLE binary in its second mode), zero drift.

**Statistics.** Identical to the audited main-battery machinery
(`analysis/aggregate_aaai.py`): paired log-ratio l = 100*ln((y_a+1)/(y_b+1)) on delay,
HL pseudomedian, BCa (9999 resamples, rng = seed 20260709 + crc32(cell tag) — bit
reproducible), Wilcoxon zero_method=pratt (exact below n=26, recorded fallback).

**Headline.** `full`-`stock` = -21.43 lp (bt +19.3%), 40/40 cells improved, and the
chain is near-additive on the median-of-cells scale:

| contrast | median l (lp) | cells W/L | reading |
|---|---|---|---|
| gce0 - stock | **-11.12** | 39/1 | early-break/tie channel is a real gain (delta=0 accepts no uphill; iteration ratio 1.016 excludes throughput) |
| rr5 - gce0 | -1.00 | 23/17 | the return band adds little at t=60 |
| rr5_v5 - rr5 | **-9.27** | 27/13 | the repair portfolio is the second driver |
| full - rr5_v5 | +0.14 | 19/21 | learned band == fixed band (honest negative) |
| fixed longest / least-delayed vs uniform | +124 / +126 | 0/40 each | catastrophic fixed orderings; regime-dependent, not knowable a priori |
| uniform vs Thompson | +4.48 | 13/27 | adaptive allocation beats blind mixing (net effect concentrated on structured maps) |

## Contents

- `raw/bridge_results.tgz` — all 10,000 per-run JSONs + `bridge_agg.json` (on-box
  aggregate) + `bridge.json` (the exact config) + runner logs + `bin_md5.txt`
- `bridge_runlevel.csv.gz` — 10,000-row flat table (same column convention as the
  battery run-level tables, plus `pull_*` repair-arm telemetry columns)
- `aggregates/` — integrity, per-cell effects, contrasts (C1–C4 + ordering matrix +
  oracle), pull telemetry, throughput, cross-machine replication vs the main battery
  (80/80 same-sign), and `summary.txt`
- `supp_tables/` — booktabs LaTeX tables (attribution chain, ordering matrix,
  40-cell per-cell table) + protocol paragraph
- `scripts/` — the full pipeline

## Reproduce

```
cd bridge_analysis
tar xzf raw/bridge_results.tgz -C raw          # -> raw/bridge/*.json
python scripts/build_runlevel.py raw/bridge bridge_runlevel.csv.gz
python scripts/analyze_bridge.py --results raw/bridge --out aggregates
python scripts/gen_supp_tables.py              # -> supp_tables/*.tex
```

Requires numpy, scipy>=1.9, matplotlib not needed for the tables. To re-run the
10,000 runs themselves: `python experiments/gen_bridge.py` then
`python experiments/runner.py --config experiments/configs/bridge.json` on a built
tree (see the top-level README for the build).
