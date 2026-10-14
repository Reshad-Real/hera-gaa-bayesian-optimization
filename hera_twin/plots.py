"""Publication-ready figures for the HERA study."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from . import config as C
from . import finfet_oracle as O

# Larger, publication-scale typography across all figures.
plt.rcParams.update({
    "font.size": 15, "axes.grid": True, "grid.alpha": 0.3,
    "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "figure.dpi": 120,
})

METHOD_COLORS = {
    "HERA": "#d6271e", "qEHVI": "#1f77b4", "qParEGO": "#2ca02c",
    "NSGA-II": "#9467bd", "Random": "#7f7f7f",
}
METHOD_ORDER = ["HERA", "qEHVI", "qParEGO", "NSGA-II", "Random"]


def _p(name):
    return os.path.join(C.RESULTS_DIR, name)


def plot_convergence(curves, n_init, path="fig1_convergence.png"):
    """Mean +- 95% CI hypervolume vs number of expensive evaluations."""
    fig, ax = plt.subplots(figsize=(10, 6.6))
    for m in METHOD_ORDER:
        if m not in curves:
            continue
        Y = np.asarray(curves[m])              # (seeds, budget)
        x = np.arange(1, Y.shape[1] + 1)
        mean = Y.mean(0)
        ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
        lw = 3.0 if m == "HERA" else 2.0
        ax.plot(x, mean, label=m, color=METHOD_COLORS[m], lw=lw,
                zorder=(5 if m == "HERA" else 3))
        ax.fill_between(x, mean - ci, mean + ci, color=METHOD_COLORS[m], alpha=0.15)
    ax.axvline(n_init, ls="--", color="k", alpha=0.5, lw=1.2)
    ax.text(n_init + 0.6, ax.get_ylim()[0] + 0.01, " end of initial DoE",
            fontsize=12, rotation=90, va="bottom", alpha=0.7)
    ax.set_xlabel("number of expensive evaluations  $N$")
    ax.set_ylabel("feasible Pareto hypervolume  HV($\\mathcal{P}$; r)")
    ax.set_title("Hypervolume convergence (mean $\\pm$ 95% CI over seeds)")
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(_p(path), bbox_inches="tight"); plt.close(fig)
    return _p(path)


def plot_final_hv_box(final, sig, path="fig3_final_hv_box.png"):
    """Boxplot of final HV per method with significance vs HERA."""
    fig, ax = plt.subplots(figsize=(10, 6.4))
    data = [np.asarray(final[m]) for m in METHOD_ORDER if m in final]
    labels = [m for m in METHOD_ORDER if m in final]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="k",
                                                   markeredgecolor="k", markersize=7))
    for patch, m in zip(bp["boxes"], labels):
        patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(0.55)
    # significance stars vs HERA
    ymax = max(d.max() for d in data)
    for i, m in enumerate(labels):
        if m == "HERA" or m not in sig:
            continue
        p = sig[m]["p_value"]
        star = ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s.")
        ax.text(i + 1, ymax * 1.012, star, ha="center", fontsize=15)
    ax.set_ylim(top=ymax * 1.07)
    ax.set_ylabel("final feasible Pareto hypervolume")
    ax.set_title("Final hypervolume across seeds (♦ = mean; stars: Wilcoxon vs HERA)")
    ax.tick_params(axis="x", labelsize=14)
    fig.tight_layout(); fig.savefig(_p(path), bbox_inches="tight"); plt.close(fig)
    return _p(path)


def plot_pareto(method_paretos, chosen_Y=None, path="fig2_pareto_fronts.png"):
    """3-D scatter + 2-D projections of feasible Pareto fronts (raw objectives)."""
    fig = plt.figure(figsize=(18, 5.8))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    pairs = [(0, 2), (0, 1)]
    axp = [fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
    labels = ["$y_1=\\log_{10}(I_{on}/I_{off})$", "$y_2=-SS$ [V/dec]",
              "$y_3=\\log_{10}(\\mathrm{SP})$"]
    for m, Y in method_paretos.items():
        Y = np.atleast_2d(Y)
        if Y.size == 0:
            continue
        c = METHOD_COLORS.get(m, None)
        s = 46 if m == "HERA" else 22
        a = 0.95 if m == "HERA" else 0.5
        ax3d.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=s, color=c, alpha=a, label=m,
                     depthshade=False)
        for k, (i, j) in enumerate(pairs):
            axp[k].scatter(Y[:, i], Y[:, j], s=s, color=c, alpha=a, label=m)
    if chosen_Y is not None:
        ax3d.scatter(*chosen_Y, s=240, marker="*", color="gold", edgecolor="k",
                     zorder=10, label="chosen")
        for k, (i, j) in enumerate(pairs):
            axp[k].scatter(chosen_Y[i], chosen_Y[j], s=300, marker="*",
                           color="gold", edgecolor="k", zorder=10)
    # 3D axis labels: moderate size + padding so they don't collide with ticks
    ax3d.set_xlabel(labels[0], fontsize=12, labelpad=10)
    ax3d.set_ylabel(labels[1], fontsize=12, labelpad=10)
    ax3d.set_zlabel(labels[2], fontsize=12, labelpad=6)
    ax3d.tick_params(labelsize=10)
    ax3d.set_title("(a) Pareto fronts in objective space", fontsize=15)
    ax3d.view_init(22, -52)
    ax3d.legend(fontsize=11, loc="upper left")
    for k, (i, j) in enumerate(pairs):
        axp[k].set_xlabel(labels[i], fontsize=14); axp[k].set_ylabel(labels[j], fontsize=14)
        axp[k].set_title(f"({'bc'[k]}) projection", fontsize=15)
        axp[k].legend(fontsize=11)
    fig.suptitle("Final feasible Pareto fronts", y=1.02, fontsize=17)
    fig.tight_layout(); fig.savefig(_p(path), bbox_inches="tight"); plt.close(fig)
    return _p(path)


def plot_idvg(devsim_metrics, params, path="fig6_devsim_idvg.png"):
    """DEVSIM Id-Vg (TCAD) vs the oracle compact subthreshold/on-current model."""
    dm = devsim_metrics
    wf = dm.get("gate_wf_offset", 0.0)
    vgs = np.asarray(dm["vgs"]) + wf      # align DEVSIM to the oracle by the metal-gate WF
    r = O.evaluate(params)
    # oracle compact-model Id-Vg (per um): subthreshold blended into on-current
    Weff_um = r.Weff * 1e-3
    SSv = r.SS / 1000.0
    ion_um = r.Ion / Weff_um
    sub = O._I0_ST * 10 ** ((vgs - r.Vth_sat) / SSv)             # A/um subthreshold
    on = ion_um * np.clip((vgs - r.Vth_sat) / (C.VDD - r.Vth_sat), 0, None)
    oracle_sat = np.maximum(np.minimum(sub, np.where(on > 0, on, sub)), 1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    for ax, scale in zip(axes, ["log", "linear"]):
        ax.plot(vgs, dm["Id_sat"], "o-", color="#d6271e", ms=5,
                label=f"DEVSIM TCAD, $V_{{DS}}$={dm['vd_high']:.2f} V")
        ax.plot(vgs, dm["Id_lin"], "s-", color="#1f77b4", ms=5,
                label=f"DEVSIM TCAD, $V_{{DS}}$={dm['vd_low']:.2f} V")
        ax.plot(vgs, oracle_sat, "--", color="k", lw=2.0, alpha=0.8,
                label="oracle compact model")
        ax.set_yscale(scale)
        ax.set_xlabel("$V_{GS}$ [V]  (work-function aligned)")
        ax.set_ylabel("$I_D$ [A/$\\mu$m]")
        ax.set_title(f"({'ab'[0 if scale=='log' else 1]}) Id–Vg ({scale} scale)")
        ax.legend(fontsize=12)
    # annotate extracted metrics (SS, DIBL are the work-function-independent checks)
    txt = (f"DEVSIM: SS={dm['SS']:.1f} mV/dec, DIBL={dm['DIBL']:.1f} mV/V\n"
           f"Oracle: SS={r.SS:.1f} mV/dec, DIBL={r.DIBL:.1f} mV/V\n"
           f"($V_{{GS}}$ shifted by WF offset {wf:+.2f} V)")
    axes[0].text(0.03, 0.97, txt, transform=axes[0].transAxes, fontsize=11.5,
                 va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    fig.suptitle("DEVSIM TCAD drift-diffusion vs digital-twin oracle", y=1.0, fontsize=17)
    fig.tight_layout(); fig.savefig(_p(path), bbox_inches="tight"); plt.close(fig)
    return _p(path)


def plot_oracle_vs_devsim_bars(params, devsim_metrics, path="fig7_oracle_devsim_bars.png"):
    """Per-metric panels comparing width-independent FoM (Vth in mV, SS, DIBL).

    Each figure of merit has its own panel/axis (and an (a)/(b)/(c) label) so the
    very different scales all read clearly.
    """
    r = O.evaluate(params)
    dm = devsim_metrics
    wf = dm.get("gate_wf_offset", 0.0)
    # Vth in mV, work-function-aligned; SS and DIBL are the WF-independent checks
    panels = [
        ("$V_{th}$ (sat, WF-aligned)", r.Vth_sat * 1e3, (dm["Vth_sat"] + wf) * 1e3, "mV", "%.0f"),
        ("Subthreshold swing", r.SS, dm["SS"], "mV/dec", "%.1f"),
        ("DIBL", r.DIBL, dm["DIBL"], "mV/V", "%.1f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    for idx, (ax, (name, ov, dv, unit, fmt)) in enumerate(zip(axes, panels)):
        b1 = ax.bar([0], [ov], 0.62, color="#d6271e", alpha=0.85, label="Digital-twin oracle")
        b2 = ax.bar([1], [dv], 0.62, color="#1f77b4", alpha=0.85, label="DEVSIM TCAD")
        ax.bar_label(b1, fmt=fmt, fontsize=14, padding=3)
        ax.bar_label(b2, fmt=fmt, fontsize=14, padding=3)
        err = 100.0 * (dv - ov) / abs(ov) if ov else float("nan")
        ax.set_title(f"({chr(97 + idx)}) {name}\n[{unit}]   (Δ = {err:+.1f}%)",
                     fontsize=15)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["oracle", "DEVSIM"], fontsize=14)
        ax.set_ylim(0, max(ov, dv) * 1.24)
        ax.margins(x=0.25)
    # single figure-level legend below the panels (never overlaps the bars)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=13,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Width-independent figures of merit: digital-twin oracle vs DEVSIM TCAD",
                 fontsize=16.5, y=1.04)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(_p(path), bbox_inches="tight"); plt.close(fig)
    return _p(path)
