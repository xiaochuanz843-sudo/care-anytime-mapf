# DEVIATIONS.md — Preregistration deviation disclosure

This document discloses every deviation between `PREREGISTRATION.md` (frozen before the
main battery; see Timestamp record below) and the analysis reported in the paper.
Convention: ℓ = 100·ln((y_arm+1)/(y_stock+1)) ("log-points", lp); bt = (1−e^{ℓ/100})·100%
(true percent reduction). Each row: planned → executed → reason → impact.

| # | Item | Preregistered plan | Actually executed | Reason | Impact |
|---|------|--------------------|-------------------|--------|--------|
| 1 | **H2 criterion hierarchy** | Primary do-no-harm criterion: per-cell ub95(ℓ) ≤ ε_regime (measured E0 stock-replicate noise floor); secondary: fixed +3% ratio margin (ub95 ≤ 2.956 lp). | Primary criterion passes **285/285** official full-arm cells (largest ub95 = 32.79 lp < smallest binding ε, e.g. city ε = 227.5 lp). The main text voluntarily headlines the **stricter secondary 3% margin (274/285)** instead. | The measured ε landed wide (17.5–349.8 lp per regime), so the preregistered primary margin has little discriminative power; reporting the stricter margin is more informative and harder to pass. | **Favorable deviation**: the headline criterion is strictly harder than the preregistered primary endpoint; the preregistered endpoint is satisfied in full and reported alongside. |
| 2 | **H3 wording (absolute anchor)** | Compare CARE's absolute delay against TACKLE's published values (cross-paper, cross-hardware). | Headline changed to same-machine paired comparison at published-protocol instances (LOCK batch, low concurrency); no cross-hardware ranking is claimed. Supplementary fact: the preregistered target is nonetheless met at both anchors — LOCK warehouse-20-40-10-2-2.N1000 CARE median delay **245** < published 411.1±37, and Paris_1_256.N1000 **973** < published 1180.8±117 — presented only as scale context in the appendix. | Cross-hardware absolute comparisons are not methodologically defensible as a headline (different machines, wall-clock-bound solvers). | Conservative deviation: the claim was weakened, not strengthened; the original target is still met and disclosed as context. |
| 3 | **H5 gmaze control** | gmaze (perfect maze) preregistered as a **null control**: "expected effect ≈ 0". | Reported as an **attenuation control**: measured family map-level median is bt ≈ **+1.5%** (ℓ = −1.50 lp; per-host medians −0.4 to −2.1 lp) — strongly attenuated versus the +8–9% official-map effect, but not exactly zero. | Perfect mazes remove alternative paths but not all plateau/ordering slack, so a small residual effect is mechanistically plausible; calling an observed +1.5% "null confirmed" would overstate. | Honest relabeling; the mechanism inference (effect requires path multiplicity; braided mazes restore it) is unchanged and supported by the gmaze→gmazeb contrast. |
| 4 | **Terminology ("safe-arm")** | Development notes and early drafts called the screened repair-arm portfolio the "safe-arm" set. | All paper-facing text uses "**screened arm set**" (the exclusion of the longest-first ordering was an empirical development-phase screening decision, not a formal safety guarantee). | "Safe" suggests a verified property; the exclusion is evidence-based screening (later corroborated: the excluded ordering is 0W/40L catastrophic in the final-binary bridge experiment). | Wording-only; no analysis change. |
| 5 | **Timestamp chain** | (Implicit) preregistration document itself timestamped before execution. | `PREREGISTRATION.md` has **no standalone OTS stamp of its own file**. Its freeze evidence is a chain: (a) OpenTimestamps stamps dated **2026-07-04 UTC** covering the development artifacts whose SHA-256 manifest it summarizes (PREREGISTRATION_cucb / FULL_EXPERIMENT_PLAN / SERVER_BATTERY_SPEC / dev LNS.cpp / smoke_accept, in `prereg_timestamp/manifest.json`); (b) battery execution dates 2026-07-10..12 (run-level `finished_at`); (c) repository commit 2026-07-15 (after execution; not usable alone as freeze evidence). This file and `PREREGISTRATION.md` are both OTS-stamped **today** (see record below) to fix the disclosure itself in time. | The 07-04 stamping covered the spec/manifest artifacts rather than the assembled PREREGISTRATION.md file. | The freeze evidence is a documented chain rather than a single stamp; disclosed here in full. The new stamps bound the *disclosure*, they do not retroactively strengthen the *freeze*. |

## Timestamp record

Method: the SHA-256 digest of each file is submitted directly to three public
OpenTimestamps calendar servers (`a.pool.opentimestamps.org`,
`b.pool.opentimestamps.org`, `alice.btc.calendar.opentimestamps.org`); each server's
raw signed response is stored under `timestamps/`. This is the same calendar-response
scheme as the 2026-07-04 stamps in `prereg_timestamp/` (the Windows `ots` CLI v0.7.2
fails on this machine — a documented python-bitcoinlib ctypes issue — so proofs are
kept in calendar-response form; the permanent Bitcoin attestation for these digests
remains retrievable from the calendars, and the responses can be assembled into
standard `.ots` files on any machine with a working client).

- Stamped (UTC): **2026-07-16T15:08Z** (exact per-file times in `timestamps/manifest.json`)
- `PREREGISTRATION.md` SHA-256:
  `083b4821c32decf11bb7602e6d6bab19e8c28ce74fd310d5c06cbefe316cbf0f`
  proofs: `timestamps/PREREGISTRATION.md.{a,b,alice}.response.bin`
- `DEVIATIONS.md` (this file): its own SHA-256 cannot be embedded here without
  changing itself; it is recorded in the sidecar **`timestamps/TIMESTAMP_RECORD.txt`**
  (which also repeats the hash above), and the file as stamped is byte-identical to
  the committed version whose digest appears there.
  proofs: `timestamps/DEVIATIONS.md.{a,b,alice}.response.bin`
