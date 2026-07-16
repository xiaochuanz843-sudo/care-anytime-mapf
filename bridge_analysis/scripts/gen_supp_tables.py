#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge supplement tables (booktabs, \\input-ready) from work/*.json.
Row statistics are HIERARCHICAL over the 40 (host,map,N) cells: each cell contributes
its scenario-paired HL(ℓ) (n=25 upstream), and the table row reports the HL pseudomedian
over the 40 cell values with a BCa-9999 CI over cells + two-sided Wilcoxon (Pratt) over
cells — the same convention as the paper's hierarchical Table 1, using the identical
audited stats core (imported from analyze_bridge.py).

  py gen_supp_tables.py     -> supp_tables/{t_r1_attribution,t_r2_ordering,t_r3_percell,protocol}.tex
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_bridge as ab  # noqa: E402  (stats core: hl, bca_ci, wilcoxon_two, bt)

WORK = os.path.join(ROOT, "aggregates")
OUT = os.path.join(ROOT, "supp_tables")
os.makedirs(OUT, exist_ok=True)
contrasts = json.load(open(os.path.join(WORK, "contrasts.json"), encoding="utf-8"))
thr = json.load(open(os.path.join(WORK, "throughput.json"), encoding="utf-8"))

HOSTS = ["lns2", "balance", "address", "tackle", "tackle_alns"]
MAPS = ["warehouse-20-40-10-2-2", "Paris_1_256", "den520d", "random-32-32-20"]
MAP_TEX = {"warehouse-20-40-10-2-2": "wh-20-40", "Paris_1_256": "Paris",
           "den520d": "den520d", "random-32-32-20": "rnd-32"}
HOST_TEX = {"lns2": "LNS2", "balance": "BALANCE", "address": "ADDRESS",
            "tackle": "TACKLE", "tackle_alns": "TACKLE\\textsubscript{ALNS}"}


def ptex(p):
    if p is None:
        return "--"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\!\\times\\!10^{{{int(e)}}}$"
    return f"${p:.3f}$"


def row_stats(cell_hls, tag):
    """Hierarchical row: HL over cell-HLs, BCa95 over cells, W/L cells, two-sided
    Wilcoxon-Pratt over cells."""
    x = np.asarray([v for v in cell_hls if v is not None], dtype=float)
    h = float(ab.hl(x))
    lo, hi, flag = ab.bca_ci(x, tag, 0.95)
    p, meth = ab.wilcoxon_two(x)
    w = int((x < 0).sum())
    l = int((x > 0).sum())
    return h, lo, hi, w, l, p, meth, ab.bt(h)


def contrast_row(name, label):
    cells = contrasts[name]["cells"]
    return (label,) + row_stats([r["hl_l"] for r in cells], "supp." + name)


def fmt_row(label, h, lo, hi, w, l, p, meth, btv, note=""):
    return (f"{label} & ${h:+.2f}$ & $[{lo:+.2f},\\,{hi:+.2f}]$ & ${btv:+.1f}\\%$ "
            f"& {w}/{l} & {ptex(p)}{note} \\\\")


# ---------------- T-R1 attribution chain ----------------
R1 = [
    ("C1_earlybreak_channel", "\\texttt{gce0}$-$\\texttt{stock} (early-break/tie channel)"),
    ("C2_acceptance_rule", "\\texttt{rr5}$-$\\texttt{gce0} (return band $\\delta{=}5$)"),
    ("C3_v5_gain_on_rr5", "\\texttt{rr5\\_v5}$-$\\texttt{rr5} (repair portfolio)"),
    ("C4_learned_band", "\\texttt{full}$-$\\texttt{rr5\\_v5} (learned band)"),
    ("H_full_vs_stock", "\\texttt{full}$-$\\texttt{stock} (total)"),
]
lines = [
    "% T-R1: bridge attribution chain (hierarchical over 40 cells; stats core identical",
    "% to the main battery: HL pseudomedian, BCa 9999, Wilcoxon zero_method=pratt).",
    "\\begin{table}[t]\\centering\\small",
    "\\caption{Final-binary bridge experiment: attribution chain. Each row summarizes",
    "the 40 $(\\text{host},\\text{map},N)$ cells; per cell the effect is the",
    "scenario-paired ($n{=}25$, shared seeds) Hodges--Lehmann log-ratio",
    "$\\ell=100\\ln\\!\\frac{y_a+1}{y_b+1}$ on delay ($\\ell<0$: first arm better).",
    "Row entries: HL over cell effects, BCa$_{95}$ over cells, back-transform",
    "$\\mathrm{bt}=(1-e^{\\ell/100})\\cdot100\\%$, cells improved/worsened, two-sided",
    "Wilcoxon (Pratt) over cells. On the plain median-of-cells scale the chain is",
    "near-additive: $-11.12-1.00-9.27+0.14=-21.25$ vs.\\ measured $-21.43$ (the table",
    "columns report the HL pseudomedian over cells, which is not exactly additive).}",
    "\\label{stab:bridge-attribution}",
    "\\begin{tabular}{@{}lrrrrr@{}}",
    "\\toprule",
    "Contrast & HL $\\ell$ & BCa$_{95}$ & bt & W/L & $p$ \\\\",
    "\\midrule",
]
for name, label in R1:
    lines.append(fmt_row(*contrast_row(name, label)))
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(OUT, "t_r1_attribution.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---------------- T-R2 ordering policies ----------------
R2 = [
    ("C3_long_vs_unif", "\\texttt{rr5\\_long}$-$\\texttt{rr5\\_unif} (fixed longest-haul)"),
    ("C3_ldel_vs_unif", "\\texttt{rr5\\_ldel}$-$\\texttt{rr5\\_unif} (fixed least-delayed)"),
    ("C3_short_vs_unif", "\\texttt{rr5\\_short}$-$\\texttt{rr5\\_unif} (fixed shortest)"),
    ("C3_mdel_vs_unif", "\\texttt{rr5\\_mdel}$-$\\texttt{rr5\\_unif} (fixed most-delayed)"),
    ("C3_unif_vs_v5", "\\texttt{rr5\\_unif}$-$\\texttt{rr5\\_v5} (uniform vs.\\ Thompson)"),
]
lines = [
    "% T-R2: replanning-order policies (same hierarchical convention as T-R1).",
    "\\begin{table}[t]\\centering\\small",
    "\\caption{Bridge experiment: replanning-order control matrix. Fixed single",
    "orderings vs.\\ the uniform mixture, the mixture vs.\\ the Thompson portfolio,",
    "and the per-cell \\emph{oracle} best fixed ordering vs.\\ Thompson (the oracle",
    "picks per cell on the same data --- an optimistic upper bound for any fixed-order",
    "policy; picks: most-delayed 21, shortest 16, longest 3, least-delayed 0).}",
    "\\label{stab:bridge-ordering}",
    "\\begin{tabular}{@{}lrrrrr@{}}",
    "\\toprule",
    "Contrast & HL $\\ell$ & BCa$_{95}$ & bt & W/L & $p$ \\\\",
    "\\midrule",
]
for name, label in R2:
    lines.append(fmt_row(*contrast_row(name, label)))
orc = contrasts["C3_oracle_fixed_vs_v5"]["cells"]
lines.append(fmt_row("oracle fixed$-$\\texttt{rr5\\_v5}",
                     *row_stats([r["hl_l"] for r in orc], "supp.oracle"),
                     note="$^{\\dagger}$"))
lines += ["\\bottomrule", "\\end{tabular}",
          "\\par\\smallskip\\raggedright\\footnotesize $^{\\dagger}$Selection bias in the",
          "oracle's favor (per-cell pick uses the evaluation data).",
          "\\end{table}"]
open(os.path.join(OUT, "t_r2_ordering.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---------------- T-R3 per-cell table ----------------
hf = {(r["host"], r["map"], r["N"]): r for r in contrasts["H_full_vs_stock"]["cells"]}
tr = {(r["host"], r["map"], r["N"]): r.get("iter_ratio_vs_stock_median")
      for r in thr["per_cell_arm"] if r["arm"] == "full"}
lines = [
    "% T-R3: all 40 bridge cells, full vs stock (delay), with the full/stock iteration",
    "% ratio (throughput channel; ratios near 1 exclude a throughput explanation).",
    "\\begin{table}[t]\\centering\\footnotesize",
    "\\caption{Bridge experiment, all 40 cells: \\texttt{full} vs.\\ \\texttt{stock}",
    "scenario-paired HL $\\ell$ on delay with BCa$_{95}$ ($n{=}25$ shared-seed pairs per",
    "cell), and the paired \\texttt{full}/\\texttt{stock} iteration ratio (cell median).}",
    "\\label{stab:bridge-percell}",
    "\\begin{tabular}{@{}llrrrr@{}}",
    "\\toprule",
    "Host & Cell & HL $\\ell$ & BCa$_{95}$ & W/L & iter.\\ ratio \\\\",
    "\\midrule",
]
for hi_, h in enumerate(HOSTS):
    for m in MAPS:
        Ns = sorted({k[2] for k in hf if k[0] == h and k[1] == m})
        for N in Ns:
            r = hf[(h, m, N)]
            ratio = tr.get((h, m, N))
            lines.append(
                f"{HOST_TEX[h] if (m == MAPS[0] and N == Ns[0]) else ''} & "
                f"{MAP_TEX[m]}$\\cdot$N{N} & ${r['hl_l']:+.2f}$ & "
                f"$[{r['ci_l'][0]:+.2f},\\,{r['ci_l'][1]:+.2f}]$ & "
                f"{r['wins']}/{r['losses']} & {ratio:.3f} \\\\")
    if hi_ < len(HOSTS) - 1:
        lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(OUT, "t_r3_percell.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---------------- protocol paragraph ----------------
proto = r"""% Bridge experiment: protocol paragraph (5 lines)
The bridge experiment ran on a rented 64-core node (62-way concurrency), wall clock
3.47\,h, completing 10{,}000/10{,}000 runs with zero failures, zero skips, and zero
incomplete cells. Within every $(\text{host},\text{map},N,\text{scenario})$ block all
ten arms share one seed, so every contrast is scenario-paired; binaries are the frozen
final ones (four builds across five hosts --- the canonical-ALNS host is the TACKLE
binary in its second mode --- with per-run \texttt{bin\_md5} recorded and zero drift).
"""
open(os.path.join(OUT, "protocol.tex"), "w", encoding="utf-8").write(proto)

for f in ("t_r1_attribution.tex", "t_r2_ordering.tex", "t_r3_percell.tex", "protocol.tex"):
    print("=" * 20, f, "=" * 20)
    print(open(os.path.join(OUT, f), encoding="utf-8").read())
