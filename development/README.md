# Development-stage screening (pre-preregistration)

This folder archives the **development experiments that predate the frozen
preregistration** and that motivated three design decisions of the shipped layer:
the **screened ordering set** (the longest-haul ordering is excluded), the
**random anchor arm**, and **adaptive allocation** over the set rather than any
single fixed ordering or a uniform mixture. It is provided for transparency and
for the fixed-ordering / no-learning baselines discussed in the paper's
supplement; it is **not** part of the confirmatory battery.

**Honest scope.** These runs use the development fork of MAPF-LNS2
(`dev_MAPF-LNS2_LNS.cpp`, env prefix `AMOR_*`), *not* the battery binaries; each
cell has only n = 6 paired (scenario, seed) instances; maps overlap the official
benchmark (evaluation-frozen, not held out from development — the files named
`heldout_*` were held out *within* the development phase only). Replanning-order
arms run under `AMOR_REPAIR_REPLANONLY=1` so the ordering lever is isolated from
the acceptance lever. Read these numbers as screening evidence, not as
confirmatory effect sizes.

## Harness

`smoke_accept.py` — iso-wall-clock harness: all arms of the same scenario run in
one wave; `delay%` is the mean paired delay change versus the `rand` base arm
(delay = SoC − sum of single-agent shortest paths), `p` is a paired Wilcoxon on
the (scenario, seed) differences, `win`/`n` are paired win counts.

## Arm key

| key | meaning |
|---|---|
| `rand` | stock random replanning order (base arm) |
| `long` | fixed longest-haul-first order (`AMOR_REPAIR=1`) |
| `mdel` | fixed most-delayed-first order (`AMOR_REPAIR=3`) |
| `ldel` | always least-delayed-first (single-arm bandit mask `AMOR_RB_ARMS=4`) |
| `eps3` | epsilon-greedy allocation over {random, longest, least-delayed} |
| `mixer` | **uniform mixture over the same arms with learning disabled** (epsilon = 1) — the no-learning necessity control |
| `cucb`, `cucbS`, `bgse`, `bgts`, `cts` | development bandit variants (UCB / SoC-aligned reward / budget-gated successive elimination / budget-gated Thompson / confidence-gated) |
| `v5` | the predecessor of the shipped Beta-Thompson allocator |
| `roul` | roulette-wheel baseline |

## Key readings

- **No single fixed ordering is safe across maps** (`accept_cucb_*.json`,
  6 maps; `heldout_*.json`, 5 more): `ldel` is the best rule on maze-32-32-2
  (−31.5% delay vs. `rand`) but strongly adverse on warehouse-10-20-10-2-1 (+77.7%),
  maze-128-128-10 (+46.5%), and held-out Paris_1_256 (+1403%); `long` is
  harmful on most maps (+80.9% warehouse, +39.2% maze-128) and best on none —
  this is the arm excluded from the shipped screened set.
- **A uniform, non-learning mixture is also unsafe** (`mixer`: +28.2%
  warehouse, +17.9% maze-128, +48.2% held-out Paris), so the safety of the
  shipped layer is not explained by mixing alone.
- **Budget-gated adaptive allocation was the only variant never worse than
  +1.1% anywhere in the screening** (`bgse`/`bgts`; e.g. −0.1%/−1.3% on the
  maps where every fixed rule and the mixer regressed by +18–81%), which
  motivated the shipped design: anchor + screened set + adaptive allocation.
- **Adaptive vs. best-guess fixed** (`winhunt_*.json`): the `v5` allocator is
  safer and usually better than the single best-guess fixed rule (`ldel`):
  maze-128 +8.0% vs. +59.5%, random-32 −4.6% vs. +2.2%, room-32 −4.3% vs. +1.0%;
  on a map matched to a fixed rule the fixed rule can win locally
  (`nsweep_maze322.json`, N175–200).
- **Hyperparameter sensitivity** (`sens_*.json`): the shipped allocator's
  constants varied one at a time on two maps.

## Files

| file(s) | contents |
|---|---|
| `results/accept_cucb_<map>.json` (6) | main screening grid: fixed rules vs. mixer vs. bandit variants |
| `results/accept_cucb_serial_verdict.json` | 9-scenario serial re-check on maze-32-32-2 |
| `results/heldout_*.json` (5) | development hold-outs: `rand`/`ldel`/`mixer`/`bgts`/`roul` |
| `results/winhunt_*.json` (3) | adaptive (`v5`) vs. best-guess fixed (`ldel`) vs. `rand` |
| `results/v5val_*.json` (2), `results/nsweep_maze322.json`, `results/isoiter_maze322.json`, `results/sanity_postfix.json` | `v5` validation, N-sweep, iso-iteration and post-fix sanity checks |
| `results/sens_*.json` (2) | allocator hyperparameter sensitivity |
| `smoke_accept.py` | the harness that produced all of the above |
| `dev_MAPF-LNS2_LNS.cpp` | the development fork's `LNS.cpp` (MAPF-LNS2 base; same upstream license as `src/lns2`) |
