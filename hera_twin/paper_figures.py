"""Print-size versions of the data figures (Figs. 3-12) for the elsarticle 5p two-column
layout of the Elsevier template.

    py -3.14 -m hera_twin.paper_figures            # all figures
    py -3.14 -m hera_twin.paper_figures pareto     # one figure

Every figure is drawn at the width it is printed at -- the full text width of the
two-column page (\\textwidth = 522 pt = 7.22 in) or one column (252 pt = 3.49 in) --
so that LaTeX includes it at scale 1 and the text stays at 8-8.5 pt. Only stored
results are read; no table or JSON output is rewritten.
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import numpy as np

from . import caee_report as R          # data helpers, colours, labels
from . import config as C
from . import finfet_oracle as O
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter

TW = 522 / 72.27                        # elsarticle 5p \textwidth (both columns) in inches
CW = 252 / 72.27                        # one column of the 5p layout in inches
FIG_DIR = R.FIG_DIR
plt.rcParams.update({
    "font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 3,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "legend.framealpha": 0.92, "legend.borderpad": 0.3, "legend.labelspacing": 0.25,
    "legend.handlelength": 1.6, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.3, "legend.edgecolor": "0.75",
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03, "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
    "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Colour-blind-safe palette (Okabe & Ito) plus distinct line styles, so that the
# methods remain distinguishable with impaired colour vision and in grey scale.
COL = {"HERA": "#D55E00", "qEHVI-LBFGS": "#56B4E9", "qEHVI": "#0072B2", "qParEGO": "#009E73",
       "NSGA-II": "#CC79A7", "Random": "#7F7F7F"}
LS = {"HERA": "-", "qEHVI-LBFGS": (0, (6, 2)), "qEHVI": "-", "qParEGO": (0, (5, 1.5, 1.5, 1.5)),
      "NSGA-II": (0, (1.5, 1.5)), "Random": (0, (1, 2.5))}
LAB = dict(R.METHOD_LABEL, **{"qEHVI-LBFGS": "q-EHVI (L-BFGS-B)"})
ORDER = ["HERA", "qEHVI-LBFGS", "qEHVI", "qParEGO", "NSGA-II", "Random"]
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.5, alpha=0.92)


def _methods(data):
    """Methods of ORDER that are present in the study results."""
    s0 = data["seeds"][0]
    return [m for m in ORDER if m in data["results"][s0]]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))     # no bbox cropping: exact size
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {name}  ({w:.2f} x {h:.2f} in)")


def _ci(Y):
    return Y.mean(0), 1.96 * Y.std(0, ddof=1) / np.sqrt(Y.shape[0])


# ---------------------------------------------------------------------------
def calibration_parity():
    D = np.load(os.path.join(C.RESULTS_DIR, "devsim_calib.npz"))
    meta = json.load(open(os.path.join(C.RESULTS_DIR, "twin_params.json")))
    ok = D["ok"].astype(bool)
    X = D["X"][ok]
    tr, te = np.array(meta["train_idx"]), np.array(meta["test_idx"])
    d = O.intrinsic(X)
    panels = [
        ("(a) SS (mV/dec)", D["SS"][ok], d["SS"], False),
        ("(b) DIBL (mV/V)", D["DIBL"][ok], d["DIBL"], False),
        ("(c) $V_{th,sat}$ (V)", D["Vth_sat"][ok], d["Vth_sat"], False),
        ("(d) $I_{on}$ (A/µm)", D["Ion"][ok], d["Ion_um"], True),
        ("(e) $I_{off}$ (A/µm)", D["Ioff"][ok], d["Ioff_um"], True),
        ("(f) $\\Delta Q_g$ (µC/cm²)", D["dQg_lin"][ok] * 1e6, d["dQg"] * 1e6, False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(TW, 4.6))
    for ax, (title, y, yh, logs) in zip(axes.ravel(), panels):
        h1 = ax.scatter(y[tr], yh[tr], s=10, facecolors="none", edgecolors="#555555", lw=0.6)
        h2 = ax.scatter(y[te], yh[te], s=11, color="#c0392b", alpha=0.9, lw=0)
        lo = min(y.min(), yh.min()); hi = max(y.max(), yh.max())
        if logs:
            ax.set_xscale("log"); ax.set_yscale("log")
            lo, hi = lo / 1.6, hi * 1.6
            r2 = R._r2(np.log10(y[te]), np.log10(yh[te]))
            tag = "test $R^2$ (log) = %.3f" % r2
            for a in (ax.xaxis, ax.yaxis):
                a.set_major_locator(LogLocator(numticks=4)); a.set_minor_formatter(NullFormatter())
        else:
            pad = 0.06 * (hi - lo); lo, hi = lo - pad, hi + pad
            r2 = R._r2(y[te], yh[te])
            tag = "test $R^2$ = %.3f" % r2
            ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("TCAD"); ax.set_ylabel("digital twin")
        ax.set_title(title)
        ax.text(0.05, 0.95, tag, transform=ax.transAxes, va="top", fontsize=7.5, bbox=BOX)
    fig.legend([h1, h2], [f"training (n = {len(tr)})", f"held-out test (n = {len(te)})"],
               loc="outside upper center", ncol=2)
    save(fig, "fig_calibration_parity")


def calibration_idvg():
    D = np.load(os.path.join(C.RESULTS_DIR, "devsim_calib.npz"))
    meta = json.load(open(os.path.join(C.RESULTS_DIR, "twin_params.json")))
    ok = np.where(D["ok"].astype(bool))[0]
    te = ok[np.array(meta["test_idx"])]
    dibl = D["DIBL"][te]
    pick = [te[np.argmin(dibl)], te[np.argsort(dibl)[len(dibl) // 2]], te[np.argmax(dibl)]]
    vgs = D["vgs"]
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.75), sharey=True)
    for ax, i, tag in zip(axes, pick, ["(a) low DIBL", "(b) median DIBL", "(c) high DIBL"]):
        x = D["X"][i]
        ax.semilogy(vgs, D["Id_sat"][i], "o", ms=2.6, color="#c0392b", label="TCAD, $V_{DS}$ = 0.70 V")
        ax.semilogy(vgs, D["Id_lin"][i], "s", ms=2.4, color="#1f77b4", label="TCAD, $V_{DS}$ = 0.05 V")
        ax.semilogy(vgs, O.idvg_um(x, vgs, C.VDS_HIGH), "-", color="#c0392b", lw=1.1, label="twin, 0.70 V")
        sub = O.idvg_um(x, vgs, C.VDS_LOW)
        m = sub < 3e-6
        ax.semilogy(vgs[m], sub[m], "-", color="#1f77b4", lw=1.0, label="twin, 0.05 V (subthreshold)")
        r = O.evaluate(x)
        ax.set_title(tag)
        ax.text(0.97, 0.04, "$L_g$ %.1f, $W$ %.1f, $L_{sp}$ %.1f nm\nTCAD: SS %.1f, DIBL %.0f\ntwin:   SS %.1f, DIBL %.0f"
                % (x[0], x[1], x[4], D["SS"][i], D["DIBL"][i], r.SS, r.DIBL), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, bbox=BOX, linespacing=1.25)
        ax.set_xlabel("$V_{GS}$ (V)")
        ax.set_ylim(1e-14, 2e-3)
        ax.yaxis.set_major_locator(LogLocator(numticks=7)); ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("$I_D$ (A/µm)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=7.5)
    save(fig, "fig_calibration_idvg")


# ---------------------------------------------------------------------------
def convergence():
    data, _ = R.load_study()
    n0 = C.HP["n0"]
    fig, ax = plt.subplots(figsize=(CW, 2.7))
    for m in _methods(data):
        Y = np.array([R._pad(data["results"][s][m]["hv_history"], data["budget"]) for s in data["seeds"]])
        x = np.arange(1, Y.shape[1] + 1)
        mu, ci = _ci(Y)
        ax.plot(x, mu, color=COL[m], ls=LS[m], lw=1.8 if m == "HERA" else 1.1, label=LAB[m],
                zorder=5 if m == "HERA" else 3)
        ax.fill_between(x, mu - ci, mu + ci, color=COL[m], alpha=0.15, lw=0)
    ax.axvline(n0, ls="--", color="k", lw=0.8, alpha=0.6)
    ax.text(n0 + 0.8, 0.03, "end of shared initial design", rotation=90, fontsize=7, alpha=0.75, va="bottom")
    ax.set_xlabel("number of expensive evaluations $N$")
    ax.set_ylabel("feasible Pareto hypervolume")
    ax.legend(ncol=2, loc="lower right", fontsize=7, handlelength=2.2)
    ax.set_ylim(bottom=0); ax.set_xlim(1, data["budget"])
    save(fig, "fig_convergence")


def final_box():
    data, summary = R.load_study()
    order = _methods(data)
    final = {m: np.array([data["results"][s][m]["hv_history"][-1] for s in data["seeds"]]) for m in order}
    short = dict(LAB, **{"qEHVI-LBFGS": "q-EHVI\n(L-BFGS-B)"})
    fig, ax = plt.subplots(figsize=(CW, 2.7))
    bp = ax.boxplot([final[m] for m in order], tick_labels=[short[m] for m in order], patch_artist=True,
                    widths=0.55, showmeans=True, flierprops=dict(markersize=3),
                    medianprops=dict(color="k", lw=0.9), boxprops=dict(lw=0.7),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                    meanprops=dict(marker="D", markerfacecolor="k", markeredgecolor="k", markersize=3))
    for b, m in zip(bp["boxes"], order):
        b.set_facecolor(COL[m]); b.set_alpha(0.6)
    top = max(v.max() for v in final.values())
    for i, m in enumerate(order):
        if m == "HERA":
            continue
        p = summary["significance"][m]["p_value"]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        if summary["significance"][m]["mean_improvement"] < 0 and star != "n.s.":
            star += "†"                  # dagger: this baseline is higher than HERA
        ax.text(i + 1, top * 1.022, star, ha="center", fontsize=8 if star == "n.s." else 9)
    ax.set_ylim(top=top * 1.07)
    ax.set_ylabel("final feasible Pareto hypervolume")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_final_hv_box")


# ---------------------------------------------------------------------------
def benchmarks():
    curves = np.load(os.path.join(C.RESULTS_DIR, "benchmark_curves.npz"))
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.55))
    for ax, prob, tag in zip(axes, ["DTLZ2", "DTLZ4", "C2-DTLZ2"], "abc"):
        for m in [k for k in ["HERA", "qEHVI-LBFGS", "qEHVI", "qParEGO", "NSGA-II", "Random"]
                  if f"{prob}__{k}" in curves.files]:
            Y = curves[f"{prob}__{m}"]
            x = np.arange(1, Y.shape[1] + 1)
            mu, ci = _ci(Y)
            ax.plot(x, mu, color=COL[m], ls=LS[m], lw=1.6 if m == "HERA" else 1.0, label=LAB[m],
                    zorder=5 if m == "HERA" else 3)
            ax.fill_between(x, mu - ci, mu + ci, color=COL[m], alpha=0.13, lw=0)
        ax.axvline(C.HP["n0"], ls="--", color="k", lw=0.8, alpha=0.5)
        ax.set_title(f"({tag}) {prob}")
        ax.set_xlabel("evaluations $N$")
        ax.set_xlim(1, 64); ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("normalized hypervolume")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(h), handlelength=2.2)
    save(fig, "fig_benchmarks")


def ablation():
    curves = np.load(os.path.join(C.RESULTS_DIR, "ablation_curves.npz"))
    summ = json.load(open(os.path.join(C.RESULTS_DIR, "ablation_summary.json")))
    order, labels = list(R.ABL_ORDER), dict(R.ABL_LABEL)
    if "q-EHVI (L-BFGS-B)" in summ["stats"]:          # added by ablation_extra.py
        order.insert(1, "q-EHVI (L-BFGS-B)")
        labels["q-EHVI (L-BFGS-B)"] = "q-EHVI + L-BFGS-B acq. optimizer"
        labels["+ NSGA-II acq. opt."] = "q-EHVI + NSGA-II acq. optimizer"
    # colour-blind-safe colours (Okabe & Ito) and line styles
    cols = {"q-EHVI (pool)": "#0072B2", "q-EHVI (L-BFGS-B)": "#56B4E9", "+ NSGA-II acq. opt.": "#009E73",
            "+ RL warm-start (hier.)": "#CC79A7", "Full HERA (+ gate)": "#D55E00",
            "RL warm-start (flat)": "#7F7F7F"}
    styles = {"q-EHVI (pool)": "-", "q-EHVI (L-BFGS-B)": (0, (6, 1.5, 1.5, 1.5)), "+ NSGA-II acq. opt.": (0, (5, 1.5)),
              "+ RL warm-start (hier.)": (0, (1.5, 1.5)), "Full HERA (+ gate)": "-",
              "RL warm-start (flat)": (0, (4, 2))}
    fig = plt.figure(figsize=(TW, 3.2))
    sa, sb = fig.subfigures(1, 2, width_ratios=[0.95, 1.15], wspace=0.02)
    a, b = sa.subplots(), sb.subplots()
    for v in order:
        Y = curves[v.replace(" ", "_")]
        x = np.arange(1, Y.shape[1] + 1)
        mu, ci = _ci(Y)
        a.plot(x, mu, color=cols[v], lw=1.7 if v.startswith("Full") else 1.1,
               ls=styles[v], label=labels[v])
        a.fill_between(x, mu - ci, mu + ci, color=cols[v], alpha=0.1, lw=0)
    a.axvline(C.HP["n0"], ls="--", color="k", lw=0.8, alpha=0.5)
    a.axvline(summ["early"], ls=":", color="k", lw=0.9, alpha=0.6)
    a.set_xlabel("number of expensive evaluations $N$"); a.set_ylabel("feasible Pareto HV")
    a.set_title("(a) Convergence", loc="left"); a.legend(loc="lower right", fontsize=7.5)
    a.set_xlim(1, summ["budget"])
    yy = np.arange(len(order))
    e = [summ["stats"][v]["hv_early_mean"] for v in order]
    es = [summ["stats"][v]["hv_early_sd"] for v in order]
    f = [summ["stats"][v]["hv_final_mean"] for v in order]
    fs = [summ["stats"][v]["hv_final_sd"] for v in order]
    ek = dict(elinewidth=0.7, capsize=1.5, capthick=0.7)
    b.barh(yy + 0.2, e, 0.38, xerr=es, color="#9ecae1", error_kw=ek, label=f"after $N$ = {summ['early']}")
    b.barh(yy - 0.2, f, 0.38, xerr=fs, color="#c0392b", alpha=0.8, error_kw=ek,
           label=f"after $N$ = {summ['budget']}")
    b.set_yticks(yy); b.set_yticklabels([labels[v] for v in order])
    b.invert_yaxis()
    b.set_xlim(min(e) - 0.03, max(np.add(f, fs)) + 0.008)
    b.set_xlabel("feasible Pareto HV (mean ± s.d.)")
    b.set_title("(b) After 32 and 64 evaluations", loc="left")
    b.grid(axis="y", visible=False)
    # legend below the axis, clear of the bars
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    save(fig, "fig_ablation")


# ---------------------------------------------------------------------------
def pareto():
    data, _ = R.load_study()
    ch = json.load(open(os.path.join(C.RESULTS_DIR, "chosen_design.json")))
    seed, cy = ch["seed"], np.array(ch["y"])
    lab_y1 = "$y_1=\\log_{10}(I_{on}/I_{off})$"
    lab_y3 = "$y_3=\\log_{10}$SP ($\\mathrm{s}^{-1}$)"
    # manual layout (one row): constrained layout does not see the 3-D tick labels
    fig = plt.figure(figsize=(TW, 2.95), layout="none")
    ax3 = fig.add_axes([0.0, 0.0, 0.37, 0.90], projection="3d")
    ax3.set_box_aspect(None, zoom=0.86)
    b = fig.add_axes([0.455, 0.16, 0.235, 0.66])
    c = fig.add_axes([0.76, 0.16, 0.235, 0.66])
    hs = []
    for m in ["Random", "NSGA-II", "qEHVI", "HERA"]:
        Y = np.atleast_2d(data["results"][seed][m]["pareto_Y"])
        if Y.size == 0:
            continue
        ss = -1000 * Y[:, 1]
        big = m == "HERA"
        kw = dict(s=14 if big else 7, color=COL[m], alpha=0.95 if big else 0.6, label=LAB[m],
                  edgecolors="k" if big else "none", linewidths=0.3)
        hs.append(ax3.scatter(Y[:, 0], ss, Y[:, 2], depthshade=False, **kw))
        b.scatter(Y[:, 0], Y[:, 2], **kw)
        c.scatter(Y[:, 0], ss, **kw)
    star = dict(marker="*", color="gold", edgecolors="k", linewidths=0.5, zorder=10)
    hs.append(ax3.scatter(cy[0], -1000 * cy[1], cy[2], s=90, label="selected design P$^{\\star}$", **star))
    b.scatter(cy[0], cy[2], s=100, **star)
    c.scatter(cy[0], -1000 * cy[1], s=100, **star)
    ax3.set_xlabel("$y_1$", labelpad=-4); ax3.set_ylabel("SS (mV/dec)", labelpad=-2)
    ax3.set_zlabel("")
    fig.text(0.322, 0.70, "$y_3$", fontsize=8.5)    # z label above the z ticks, clear of panel (b)
    ax3.tick_params(axis="both", which="major", labelsize=7, pad=-2)
    ax3.tick_params(axis="z", pad=1)
    ax3.view_init(22, -55)
    for a3 in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        a3.set_major_locator(MaxNLocator(4))
    fig.text(0.01, 0.84, "(a)", fontsize=8.5)
    b.set_xlabel(lab_y1); b.set_ylabel(lab_y3); b.set_title("(b)", loc="left")
    c.set_xlabel(lab_y1); c.set_ylabel("SS (mV/dec)"); c.set_title("(c)", loc="left")
    c.invert_yaxis()
    fig.legend(hs, [h.get_label() for h in hs], loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, fontsize=7.5)
    save(fig, "fig_pareto")


def heatmaps():
    from matplotlib.colors import LogNorm
    x0 = np.array(json.load(open(os.path.join(C.RESULTS_DIR, "chosen_design.json")))["x"])
    N = 160
    Lg = np.linspace(C.LB[0], C.UB[0], N)
    W = np.linspace(C.LB[1], C.UB[1], N)
    LG, WW = np.meshgrid(Lg, W)
    X = np.tile(x0, (LG.size, 1))
    X[:, 0] = LG.ravel(); X[:, 1] = WW.ravel()
    d = O.evaluate_array(X)
    feas = O.is_feasible(X).reshape(LG.shape)
    panels = [("(a) $I_{on}$ (µA/µm)", (d["Ion_um"] * 1e6).reshape(LG.shape), "viridis", None),
              ("(b) $I_{off}$ (A/µm)", d["Ioff_um"].reshape(LG.shape), "plasma", "log"),
              ("(c) $\\log_{10}(I_{on}/I_{off})$", d["y1"].reshape(LG.shape), "cividis", None)]
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.45), sharey=True)
    for ax, (t, Z, cm, sc) in zip(axes, panels):
        norm = LogNorm(vmin=Z.min(), vmax=Z.max()) if sc == "log" else None
        pc = ax.pcolormesh(LG, WW, Z, cmap=cm, norm=norm, shading="gouraud", rasterized=True)
        cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.06, aspect=22)
        cb.ax.tick_params(labelsize=7, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)
        if sc == "log":
            cb.ax.yaxis.set_major_locator(LogLocator(numticks=5)); cb.ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            cb.ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.contourf(LG, WW, (~feas).astype(float), levels=[0.5, 1.5], colors="white", alpha=0.45)
        ax.contour(LG, WW, feas.astype(float), levels=[0.5], colors="k", linestyles="--", linewidths=0.9)
        ax.plot(x0[0], x0[1], marker="*", ms=9, color="gold", mec="k", mew=0.5)
        ax.set_xlabel("$L_g$ (nm)")
        ax.set_title(t); ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("$W$ (nm)")
    save(fig, "fig_heatmaps")


# ---------------------------------------------------------------------------
def pareto_tcad_set():
    """Unique final-Pareto designs of the methods in ORDER (30-seed twin study) with their
    TCAD figures of merit: DEVSIM results from devsim_pareto.npz, or from the TCAD cache of
    tcad_loop for designs simulated later (q-EHVI with L-BFGS-B)."""
    from . import tcad_loop as TL
    data, _ = R.load_study()
    P = np.load(os.path.join(C.RESULTS_DIR, "devsim_pareto.npz"))
    keys = ["Ion", "Ioff", "SS", "DIBL", "dQg_lin"]
    by_x = {tuple(np.round(x, 10)): i for i, x in enumerate(P["X"])}
    cache = TL._load(TL.CACHE, {})
    rows, seen = [], set()
    for s in data["seeds"]:
        for m in ORDER:
            for x in np.atleast_2d(data["results"][s][m]["pareto_X"]):
                k = tuple(np.round(x, 10))
                if k in seen:
                    continue
                seen.add(k)
                if k in by_x:
                    i = by_x[k]
                    rec = {q: float(P[q][i]) for q in keys}
                    rec["ok"] = bool(P["ok"][i])
                else:
                    rec = cache[TL.tcad_key(x)]
                rows.append((x, rec))
    X = np.array([r[0] for r in rows])
    ok = np.array([r[1]["ok"] for r in rows])
    M = {q: np.array([r[1][q] if r[1]["ok"] else 1.0 for r in rows]) for q in keys}
    return X, M, ok


def tcad_parity():
    Xu, M, ok = pareto_tcad_set()
    Yt, ct = R.tcad_objectives(Xu, M)
    Yo = O.objectives(Xu)
    # accuracy summary for the text (Section 6.5)
    feas_twin = O.is_feasible(Xu)
    feas_tcad = ok & np.all(ct <= 0, axis=1)
    e = Yo[ok] - Yt[ok]
    summ = dict(n=int(len(Xu)), n_ok=int(ok.sum()),
                y1_MAE_dec=float(np.mean(np.abs(e[:, 0]))), SS_MAE_mVdec=float(1000 * np.mean(np.abs(e[:, 1]))),
                y3_MAE_dec=float(np.mean(np.abs(e[:, 2]))),
                R2=[float(R._r2(Yt[ok, j], Yo[ok, j])) for j in range(3)],
                twin_feasible_tcad_infeasible=float(np.mean(~feas_tcad[feas_twin])))
    json.dump(summ, open(os.path.join(C.RESULTS_DIR, "tcad_parity_summary.json"), "w"), indent=1)
    print("TCAD parity:", summ)
    titles = ["(a) $y_1=\\log_{10}(I_{on}/I_{off})$", "(b) SS (mV/dec)", "(c) $y_3=\\log_{10}$SP ($\\mathrm{s}^{-1}$)"]
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.5))
    for j, ax in enumerate(axes):
        a, b = Yt[ok, j], Yo[ok, j]
        if j == 1:
            a, b = -1000 * a, -1000 * b
        ax.scatter(a, b, s=4, alpha=0.4, color="#c0392b", lw=0, rasterized=True)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        pad = 0.04 * (hi - lo); lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("TCAD"); ax.set_ylabel("digital twin")
        ax.set_title(titles[j], fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.text(0.05, 0.95, f"$R^2$ = {R._r2(a, b):.3f}", transform=ax.transAxes, va="top",
                fontsize=7.5, bbox=BOX)
    save(fig, "fig_tcad_pareto_parity")


def chosen_idvg():
    T = np.load(os.path.join(C.RESULTS_DIR, "devsim_chosen.npz"))
    x = T["X"][0]
    vgs = T["vgs"]
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.6))
    for ax, sc in zip(axes, ["log", "linear"]):
        ax.set_yscale(sc)
        ax.plot(vgs, T["Id_sat"][0] * 1e6, "o", ms=2.8, color="#c0392b", label="TCAD, $V_{DS}$ = 0.70 V")
        ax.plot(vgs, T["Id_lin"][0] * 1e6, "s", ms=2.6, color="#1f77b4", label="TCAD, $V_{DS}$ = 0.05 V")
        ax.plot(vgs, O.idvg_um(x, vgs, C.VDS_HIGH) * 1e6, "-", color="#c0392b", lw=1.1, label="twin, 0.70 V")
        if sc == "log":
            sub = O.idvg_um(x, vgs, C.VDS_LOW) * 1e6
            m = sub < 3
            ax.plot(vgs[m], sub[m], "-", color="#1f77b4", lw=1.0, label="twin, 0.05 V (subthreshold)")
            ax.axhline(C.IOFF_MAX_UM * 1e6, color="grey", ls=":", lw=1.0)
            ax.text(0.33, C.IOFF_MAX_UM * 1e6 * 2.0, "IRDS HD $I_{off}$ limit", fontsize=7.5, color="0.35")
            ax.yaxis.set_major_locator(LogLocator(numticks=7)); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("$V_{GS}$ (V)"); ax.set_ylabel("$I_D$ (µA/µm)")
        ax.set_title("(a) Logarithmic scale" if sc == "log" else "(b) Linear scale")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=7.5)
    save(fig, "fig_chosen_idvg")


ALL = {"calib_parity": calibration_parity, "calib_idvg": calibration_idvg, "convergence": convergence,
       "box": final_box, "bench": benchmarks, "ablation": ablation, "pareto": pareto,
       "heatmaps": heatmaps, "tcad_parity": tcad_parity, "chosen_idvg": chosen_idvg}

if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL):
        ALL[k]()
