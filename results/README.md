# Aggregated results

Per-cell aggregated tables underlying our reported results. All effects
use the log-ratio `ℓ = 100·ln((y_arm+1)/(y_stock+1))` (`ℓ<0` = improvement) and its true-reduction
transform `bt = (1−e^{ℓ/100})×100` (`bt>0` = improvement). A *cell* is one `(host, map, N)`; the
pairing unit is `(host, map, N, scen)`. `full` = `spsa+v5` = CARE (the preregistered confirmatory
arm); `stock` = each host's vanilla baseline. These are produced from the 146,660 per-run JSONs by
the analysis pipeline (`analysis/aggregate_aaai.py` and the per-cell/contrast reducers).

| File | Backs | Contents |
|---|---|---|
| `headline_tables.json` | H1, H3 | Per-host exact-binomial improvement counts; LOCK decontention (8 cells, absolute stock→full). |
| `A_full_permap.json` | H1 | Per-cell `full` vs `stock` for each host's 57 cells: `bt_hl`, BCa CI, `ub95_l`, `n_eff`, W/T/L. |
| `A_allarms.json` | H1, ablation | Full A-batch 8-arm grid per cell (per-arm `bt_hl`, `ub95_l`, wins/losses). |
| `contrasts_tables.json` | Ablation | The six paired contrasts (accept-side, repair-side, learned-vs-fixed accept/repair, cart selection, legacy) with per-host/per-regime/per-source aggregates + best/worst-20 cells. |
| `orthogonality.json` | Ablation | Accept×repair super/sub-additivity `(ℓ_spsa_only+ℓ_v5_only)−ℓ_full` + extreme cells. |
| `dnh_tables.json` | H2 | Do-no-harm ledger: per-host pass/fail and the exception cells, criterion `ub95 ≤ min(ε_regime, 2.96)`. |
| `epsilon_table.json` | H2 | E0 per-regime solver-noise floor `ε` (95th pct of stock-vs-stock \|ℓ\|). |
| `anytime_tables.json` | H4 | B-batch checkpoints (50 cells × arms × {60,120,180,240,300}s), the certified-worse drift table, and late-dominance. |
| `B_curves.json` | H4 | Per-cell incumbent-delay trajectories `med_delay_at{60..300}` and `med_auc` for each arm. |
| `gen_tables.json` | H5 | Per-family generated-map results (`maps_improved`, binomial p, per-map HL ECDF). |
| `G_allarms.json` | H5, ablation | G-batch 6-arm grid per cell, tagged by family. |
| `absolutes.json` | H1, H3 | Absolute median delays (`stock`→arm) and median iterations per cell. |
| `addenda.json` | Honesty checks | Spine-vs-breadth stratification, generated-map per-arm medians, B-batch `full`-vs-`rr5` pairing, iteration-ratio audit. |
| `integrity.json`, `recon_summary.json` | Integrity | Completeness (97.6%), failure classification, machine sharding, binary md5s, iteration-cap audit (`n_at_cap=0`). |
| `supplement_sata_armpull.txt` | Supplement | SA/TA acceptance comparison and bandit arm-pull frequencies. **Run separately** from the preregistered study (which carries no SA/TA arm and does not log per-arm pull counts). |
| `supplement_auc.txt` | Supplement | Incumbent-AUC (`full` vs `stock`), t=60 protocol-5. Separate supplementary battery; consistent with the main B-batch AUC. |

The two `supplement_*.txt` files come from a separate supplementary battery on different hardware;
all headline and absolute-magnitude claims rest on the main preregistered study above.
