# Preregistration — Unified Accept+Repair Layer Battery

Frozen before the full battery is launched. Any deviation will be disclosed in the paper.

## Hosts (5 configurations, 4 independent codebases)

MAPF-LNS2 (AAAI'22), BALANCE (AAAI'24), ADDRESS (AAAI'25), TACKLE (AAAI'26),
and MAPF-LNS canonical ALNS (same binary as TACKLE, `--algorithm=canonical
--banditAlgo=Roulette`, i.e. the adaptive roulette destroy selection of MAPF-LNS —
the binary's default banditAlgo "Random" is NOT canonical ALNS). ADDRESS runs its
README-exact configuration (`--destroyStrategy=RandomWalk --algorithm=bernoulie
--k 64`); with the binary's default destroyStrategy=Adaptive the constructor
overwrites the algorithm to "canonical" and ADDRESS's mechanism never executes.
All layer code is environment-variable gated; with no variables set, each binary
reproduces its upstream vanilla behavior (verified before launch). The iteration
cap (1e8) is set so the wall-clock budget always binds first; aggregation flags
any run whose iteration count reaches the cap.

## Confirmatory arm (exactly one)

`full` = `<PFX>_ACCEPT=spsa` + `<PFX>_REPAIR=v5`
(state-conditioned record-to-record acceptance with online SPSA-tuned threshold;
Thompson-sampling safe-arm repair bandit). All hyperparameters at code defaults,
identical across hosts, frozen on official maps; zero per-map tuning anywhere,
zero tuning on generated maps (strict zero-shot OOD).

All other arms (`spsa_only`, `v5_only`, `rr5_v5`, `spsa_20`, `cart_v5`, `rr5_20`,
`rr20_v5`, `rr50_v5`) are exploratory mechanism decompositions: effect sizes with
CIs only, no significance claims, and none will be promoted to a headline claim
post hoc.

## Hypotheses

- **H1 (stacking):** `full` reduces sum-of-delays vs `stock` on the majority of
  cells within every host (per-host exact binomial over cells; no cross-host
  pooling; no cross-map medians as headline).
- **H2 (do-no-harm):** on every cell, the one-sided 95% bootstrap upper bound of
  the paired log-ratio ≤ ε, with ε = measured solver-noise floor per regime
  (Batch E0), and secondarily ≤ +3%. Cells with n_eff < 10: "cannot certify"
  (not counted as pass). Known preregistered boundary regimes: maze-class maps
  near the solvability phase transition; LNS2 makespan may worsen on ~1/3 of
  instances (known cost of optimizing delay).
- **H3 (decontention):** under low concurrency (8 parallel jobs, dedicated
  machine), official-protocol TACKLE + `full` achieves absolute sum-of-delays
  below TACKLE's published values on the 5-map protocol.
- **H4 (anytime stability):** fixed-threshold acceptance with large δ (rr20/rr50)
  degrades late in a 300 s budget (paired Δ(t) vs `stock` crosses zero), while
  `full` remains non-inferior at every checkpoint t ∈ {60,120,180,240,300} s.
- **H5 (OOD transfer):** on generated maps (parameters frozen from official
  maps), `full` vs `stock` direction is positive per family (per-family binomial,
  ECDFs reported); the perfect-maze family `gmaze` is a preregistered null
  control (expected effect ≈ 0, zero detour freedom).

## Statistics

Paired within (host, map, N, scenario), same batch, same machine.
Effect ℓ = ln((y_arm+1)/(y_stock+1)); tables display the back-transform
(1−e^ℓ) as true relative reduction. Per cell: Hodges–Lehmann pseudomedian,
BCa 95% CI, one-sided exact Wilcoxon (Pratt). Descriptive per-cell significance
counts corrected with Benjamini–Hochberg within host family. Primary and only
headline/confirmatory quality metric: sum-of-delays at budget end. The primary
anytime metric is the AUC of the incumbent delay-vs-time curve, computed ONLY
from the phase-B trajectory logs (monotone best-so-far incumbent), valid only
where both arms share the initial solution (verified via equal initial cost).
The per-run CSV "area under curve" column integrates the working (non-incumbent)
solution and is NOT used in any comparison — for uphill-accepting arms it is not
the same anytime quantity as for the greedy baseline.

Baseline authenticity: each host runs its published algorithm — MAPF-LNS2
(Adaptive ALNS, PP init, seed-controlled), ADDRESS (RandomWalk + bernoulie
top-K Thompson, K=32 = the paper default), BALANCE (Thompson bandit), TACKLE
(tackle-roulette, top-k with k=min(32,N) — identical to the published k=32 for
N≥32 and the only valid choice for N<32), and MAPF-LNS canonical ALNS (roulette
destroy selection). The injected acceptance layer is disclosed as a
state-conditioned threshold whose online SPSA tuning is largely inert (it defaults
to a near-greedy fixed threshold); the contribution is the acceptance+repair
lever and its do-no-harm envelope, not "learning".

## Data hygiene

MovingAI: standard 25 `random` scenarios per map, no synthesis.
Generated maps: fixed seed ranges (frozen in `data/gen_maps/MANIFEST.json`
with per-file md5), byte-identical regeneration checked; **no post-hoc
exclusion of any seed, map, scenario, or cell.** Failed runs are reported in a
reconciliation table (n_eff + dropped + both_failed = scheduled).

## Compute protocol

t=60 s budget for Batches L/E0/A/G; t=300 s with trajectory logging for
Batch B. Batch L runs at low concurrency (8 workers, dedicated machine);
other batches at high concurrency with job-order shuffling (fixed-seed
deterministic shuffle) to decorrelate load drift from arm identity; both arms of
every pair always run in the same batch on the same machine.
