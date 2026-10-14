"""Figures and tables for the standard-benchmark validation and the ablation.

Consumes the artefacts written by benchmark_study.py and ablation_study.py
(benchmark_summary.json, benchmark_curves.npz, ablation_summary.json,
ablation_curves.npz) and emits:
  fig_benchmarks.png     HV-vs-evaluation convergence on DTLZ2 / DTLZ4 / C2-DTLZ2
  fig_ablation.png       ablation ladder on the FinFET twin (curves + bars)
  table_benchmarks.{tex,md}   HV and IGD+ with Wilcoxon tests
  table_ablation.{tex,md}     component ladder (HV@early, HV@final, AUC)

Run:  python -m hera_twin.bench_report
"""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

plt.rcParams.update({
    "font.size": 14, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.dpi": 120,
})

BENCH_METHODS = ["HERA", "qEHVI", "qParEGO", "NSGA-II", "Random"]
METHOD_COLORS = {"HERA": "#d6271e", "qEHVI": "#1f77b4", "qParEGO": "#2ca02c",
                 "NSGA-II": "#9467bd", "Random": "#7f7f7f"}
BENCH_PROBLEMS = ["DTLZ2", "DTLZ4", "C2-DTLZ2"]

ABL_VARIANTS = ["q-EHVI (pool)", "+ NSGA-II acq. opt.", "+ RL warm-start (hier.)",
                "Full HERA (+ gate)", "RL warm-start (flat)"]
ABL_COLORS = {"q-EHVI (pool)": "#7f7f7f", "+ NSGA-II acq. opt.": "#1f77b4",
              "+ RL warm-start (hier.)": "#ff7f0e", "Full HERA (+ gate)": "#d6271e",
              "RL warm-start (flat)": "#2ca02c"}


def _p(name):
    return os.path.join(C.RESULTS_DIR, name)


def _load(name):
    with open(_p(name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------
def fig_benchmarks(path="fig_benchmarks.png"):
    curves = np.load(_p("benchmark_curves.npz"))
    summ = _load("benchmark_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for ax, prob in zip(axes, BENCH_PROBLEMS):
        for m in BENCH_METHODS:
            Y = curves[f"{prob}__{m}"]
            x = np.arange(1, Y.shape[1] + 1)
            mean = Y.mean(0)
            ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
            lw = 3.0 if m == "HERA" else 1.8
            ax.plot(x, mean, color=METHOD_COLORS[m], lw=lw, label=m,
                    zorder=5 if m == "HERA" else 3)
            ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[m], alpha=0.13)
        n0 = C.HP["n0"]
        ax.axvline(n0, ls="--", color="k", alpha=0.45, lw=1.1)
        ax.set_title(prob)
        ax.set_xlabel("expensive evaluations  $N$")
    axes[0].set_ylabel("normalised hypervolume  HV")
    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.suptitle("Standard-benchmark convergence (mean $\\pm$ 95% CI over seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(_p(path), bbox_inches="tight")
    plt.close(fig)
    return _p(path)


def fig_ablation(path="fig_ablation.png"):
    curves = np.load(_p("ablation_curves.npz"))
    summ = _load("ablation_summary.json")
    early = summ["early"]
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(14.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    # (a) convergence curves for the cumulative ladder + flat contrast
    for v in ABL_VARIANTS:
        Y = curves[v.replace(" ", "_")]
        x = np.arange(1, Y.shape[1] + 1)
        mean = Y.mean(0)
        ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
        ls = "--" if v == "RL warm-start (flat)" else "-"
        lw = 3.0 if v == "Full HERA (+ gate)" else 1.9
        axc.plot(x, mean, color=ABL_COLORS[v], lw=lw, ls=ls, label=v,
                 zorder=5 if v.startswith("Full") else 3)
        axc.fill_between(x, mean - ci, mean + ci, color=ABL_COLORS[v], alpha=0.10)
    axc.axvline(early, ls=":", color="k", alpha=0.5, lw=1.1)
    axc.text(early + 0.5, axc.get_ylim()[0], f" N={early}", fontsize=11, rotation=90, va="bottom", alpha=0.7)
    axc.axvline(C.HP["n0"], ls="--", color="k", alpha=0.4, lw=1.0)
    axc.set_xlabel("expensive evaluations  $N$")
    axc.set_ylabel("feasible Pareto hypervolume  HV")
    axc.set_title("(a) Cumulative component ladder")
    axc.legend(loc="lower right", framealpha=0.9, fontsize=11)
    # (b) grouped bars HV@early and HV@final
    order = ABL_VARIANTS
    hv_e = [summ["stats"][v]["hv_early_mean"] for v in order]
    hv_e_sd = [summ["stats"][v]["hv_early_sd"] for v in order]
    hv_f = [summ["stats"][v]["hv_final_mean"] for v in order]
    hv_f_sd = [summ["stats"][v]["hv_final_sd"] for v in order]
    y = np.arange(len(order))
    h = 0.38
    axb.barh(y + h / 2, hv_e, height=h, xerr=hv_e_sd, color="#8c9bb5",
             label=f"HV @ {early} evals", capsize=3)
    axb.barh(y - h / 2, hv_f, height=h, xerr=hv_f_sd,
             color=[ABL_COLORS[v] for v in order], label="HV @ 64 evals", capsize=3)
    axb.set_yticks(y)
    axb.set_yticklabels([v.replace(" warm-start", "").replace("acq. opt.", "acq.") for v in order],
                        fontsize=11)
    axb.invert_yaxis()
    axb.set_xlabel("feasible Pareto hypervolume  HV")
    axb.set_title("(b) Early vs final hypervolume")
    axb.legend(loc="lower right", fontsize=11)
    axb.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(_p(path), bbox_inches="tight")
    plt.close(fig)
    return _p(path)


# --------------------------------------------------------------------------
def table_benchmarks():
    summ = _load("benchmark_summary.json")
    ns = summ[BENCH_PROBLEMS[0]]["n_seeds"]
    bud = summ[BENCH_PROBLEMS[0]]["budget"]
    lines = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Standard multi-objective benchmarks (d=6, M=3), "
         + f"{ns} seeds at a budget of {bud} evaluations. HV: normalised "
         r"hypervolume (higher is better); IGD$^+$: modified inverted generational "
         r"distance to the analytic Pareto front (lower is better). $p$: Wilcoxon "
         r"signed-rank test of HERA vs.\ the baseline on final HV.}"),
        r"\label{tab:bench}", r"\small",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Problem & Method & HV $\uparrow$ & IGD$^+$ $\downarrow$ & $p$ & HERA wins \\",
        r"\midrule",
    ]
    md = ["| Problem | Method | HV ↑ | IGD+ ↓ | p | wins |", "|---|---|---|---|---|---|"]
    for prob in BENCH_PROBLEMS:
        st = summ[prob]["stats"]
        sig = summ[prob]["sig"]
        for i, m in enumerate(BENCH_METHODS):
            name = prob if i == 0 else ""
            hv = f"{st[m]['hv_mean']:.4f}$\\pm${st[m]['hv_sd']:.4f}"
            igd = f"{st[m]['igd_mean']:.4f}$\\pm${st[m]['igd_sd']:.4f}"
            if m == "HERA":
                row = f"{name} & \\textbf{{HERA}} & \\textbf{{{hv}}} & \\textbf{{{igd}}} & --- & --- \\\\"
                mdrow = f"| {name} | **HERA** | {st[m]['hv_mean']:.4f}±{st[m]['hv_sd']:.4f} | {st[m]['igd_mean']:.4f}±{st[m]['igd_sd']:.4f} | — | — |"
            else:
                p = sig[m]["p_value"]
                row = f"{name} & {m} & {hv} & {igd} & {p:.1e} & {sig[m]['wins']}/{ns} \\\\"
                mdrow = f"| {name} | {m} | {st[m]['hv_mean']:.4f} | {st[m]['igd_mean']:.4f} | {p:.1e} | {sig[m]['wins']}/{ns} |"
            lines.append(row)
            md.append(mdrow)
        if prob != BENCH_PROBLEMS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(_p("table_benchmarks.tex"), "w", encoding="utf-8").write("\n".join(lines))
    open(_p("table_benchmarks.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return summ


def table_ablation():
    summ = _load("ablation_summary.json")
    e, bud = summ["early"], summ["budget"]
    st, sig = summ["stats"], summ["sig"]
    lines = [
        r"\begin{table}[H]", r"\centering",
        (r"\caption{Component ablation of HERA on the FinFET case study "
         + f"({summ['n_seeds']} seeds, budget {bud}). Each row of the upper block adds "
         r"one component to the previous. HV@" + str(e) + r" probes the early "
         r"(sample-efficiency) regime; HV@" + str(bud) + r" is the converged value; "
         r"AUC is the mean hypervolume over the run. $\Delta$ and $p$ compare each "
         r"configuration to the full HERA (Wilcoxon signed-rank on final HV).}"),
        r"\label{tab:ablation}", r"\small",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Configuration & HV@" + str(e) + r" & HV@" + str(bud) + r" & AUC & $\Delta$HV$_{\mathrm{final}}$ ($p$) \\",
        r"\midrule",
    ]
    md = [f"| Configuration | HV@{e} | HV@{bud} | AUC | ΔHV_final (p) |",
          "|---|---|---|---|---|"]
    ladder = ABL_VARIANTS[:4]
    for v in ladder:
        s = st[v]
        cell = f"{s['hv_early_mean']:.4f} & {s['hv_final_mean']:.4f} & {s['auc_mean']:.4f}"
        if v == "Full HERA (+ gate)":
            lines.append(f"\\textbf{{{v}}} & \\textbf{{{s['hv_early_mean']:.4f}}} & "
                         f"\\textbf{{{s['hv_final_mean']:.4f}}} & \\textbf{{{s['auc_mean']:.4f}}} & --- \\\\")
            md.append(f"| **{v}** | {s['hv_early_mean']:.4f} | {s['hv_final_mean']:.4f} | {s['auc_mean']:.4f} | — |")
        else:
            d = -sig[v]["hv_gain"]
            lines.append(f"{v} & {cell} & ${d:+.4f}$ (${sig[v]['p_value']:.1e}$) \\\\")
            md.append(f"| {v} | {s['hv_early_mean']:.4f} | {s['hv_final_mean']:.4f} | {s['auc_mean']:.4f} | {d:+.4f} ({sig[v]['p_value']:.1e}) |")
    lines.append(r"\midrule")
    md.append("| *hierarchy contrast* | | | | |")
    v = "RL warm-start (flat)"
    s = st[v]
    d = -sig[v]["hv_gain"]
    lines.append(r"\textit{RL warm-start (flat, no hierarchy)} & "
                 f"{s['hv_early_mean']:.4f} & {s['hv_final_mean']:.4f} & {s['auc_mean']:.4f} & "
                 f"${d:+.4f}$ (${sig[v]['p_value']:.1e}$) \\\\")
    md.append(f"| RL warm-start (flat) | {s['hv_early_mean']:.4f} | {s['hv_final_mean']:.4f} | {s['auc_mean']:.4f} | {d:+.4f} ({sig[v]['p_value']:.1e}) |")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    open(_p("table_ablation.tex"), "w", encoding="utf-8").write("\n".join(lines))
    open(_p("table_ablation.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return summ


def main():
    b = table_benchmarks()
    print("benchmark table written")
    a = table_ablation()
    print("ablation table written")
    print("fig:", fig_benchmarks())
    print("fig:", fig_ablation())


if __name__ == "__main__":
    main()
