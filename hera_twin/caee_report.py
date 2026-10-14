"""Figures and tables for the CAEE manuscript (TCAD-calibrated GAA case study).

    python -m hera_twin.caee_report calib      # calibration parity + Id-Vg figures
    python -m hera_twin.caee_report study      # convergence, box, tables from the 30-seed study
    python -m hera_twin.caee_report showcase   # showcase run, Pareto figure, chosen design, heat maps
    python -m hera_twin.caee_report tcad       # TCAD re-scoring of every method's final Pareto set

All figures are written to results/figures/ at 300 dpi (PNG) plus vector PDF.
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from . import finfet_oracle as O
from . import hypervolume as H

FIG_DIR = os.path.join(C.RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "legend.fontsize": 10, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.dpi": 300, "font.family": "DejaVu Sans",
})
METHOD_ORDER = ["HERA", "qEHVI", "qParEGO", "NSGA-II", "Random"]
METHOD_LABEL = {"HERA": "HERA", "qEHVI": "q-EHVI", "qParEGO": "ParEGO",
                "NSGA-II": "NSGA-II", "Random": "Random"}
METHOD_COLORS = {"HERA": "#c0392b", "qEHVI": "#1f77b4", "qParEGO": "#2ca02c",
                 "NSGA-II": "#9467bd", "Random": "#7f7f7f"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _r2(y, yh):
    return 1 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------------------
# Calibration figures
# ---------------------------------------------------------------------------
def fig_calibration_parity():
    D = np.load(os.path.join(C.RESULTS_DIR, "devsim_calib.npz"))
    meta = json.load(open(os.path.join(C.RESULTS_DIR, "twin_params.json")))
    ok = D["ok"].astype(bool)
    X = D["X"][ok]
    tr, te = np.array(meta["train_idx"]), np.array(meta["test_idx"])
    d = O.intrinsic(X)
    panels = [
        ("(a) Subthreshold swing", "SS [mV/dec]", D["SS"][ok], d["SS"], False),
        ("(b) DIBL", "DIBL [mV/V]", D["DIBL"][ok], d["DIBL"], False),
        ("(c) Threshold voltage", "$V_{th,sat}$ [V]", D["Vth_sat"][ok], d["Vth_sat"], False),
        ("(d) On-current", "$I_{on}$ [A/µm]", D["Ion"][ok], d["Ion_um"], True),
        ("(e) Off-current", "$I_{off}$ [A/µm]", D["Ioff"][ok], d["Ioff_um"], True),
        ("(f) Switched gate charge", "$\\Delta Q_g$ [µC/cm²]", D["dQg_lin"][ok] * 1e6, d["dQg"] * 1e6, False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for ax, (title, lab, y, yh, logs) in zip(axes.ravel(), panels):
        ax.scatter(y[tr], yh[tr], s=22, facecolors="none", edgecolors="#555555", lw=0.9,
                   label=f"training (n={len(tr)})")
        ax.scatter(y[te], yh[te], s=30, color="#c0392b", alpha=0.9,
                   label=f"held-out test (n={len(te)})")
        lo = min(y.min(), yh.min()); hi = max(y.max(), yh.max())
        if logs:
            ax.set_xscale("log"); ax.set_yscale("log")
            lo, hi = lo / 1.5, hi * 1.5
            r2 = _r2(np.log10(y[te]), np.log10(yh[te]))
            tag = "$R^2_{test}$ (log) = %.3f" % r2
        else:
            pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
            r2 = _r2(y[te], yh[te])
            tag = "$R^2_{test}$ = %.3f" % r2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("DEVSIM TCAD  " + lab); ax.set_ylabel("digital twin  " + lab)
        ax.set_title(title)
        ax.text(0.04, 0.95, tag, transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_calibration_parity")


def fig_calibration_idvg():
    """Twin vs DEVSIM Id-Vg for three held-out designs spanning the SCE range."""
    D = np.load(os.path.join(C.RESULTS_DIR, "devsim_calib.npz"))
    meta = json.load(open(os.path.join(C.RESULTS_DIR, "twin_params.json")))
    ok = np.where(D["ok"].astype(bool))[0]
    te = ok[np.array(meta["test_idx"])]
    dibl = D["DIBL"][te]
    pick = [te[np.argmin(dibl)], te[np.argsort(dibl)[len(dibl) // 2]], te[np.argmax(dibl)]]
    vgs = D["vgs"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, i, tag in zip(axes, pick, ["(a) low DIBL", "(b) median DIBL", "(c) high DIBL"]):
        x = D["X"][i]
        ax.semilogy(vgs, D["Id_sat"][i], "o", ms=4.5, color="#c0392b", label="TCAD, $V_{DS}$=0.70 V")
        ax.semilogy(vgs, D["Id_lin"][i], "s", ms=4, color="#1f77b4", label="TCAD, $V_{DS}$=0.05 V")
        ax.semilogy(vgs, O.idvg_um(x, vgs, C.VDS_HIGH), "-", color="#c0392b", lw=1.8, label="twin, 0.70 V")
        sub = O.idvg_um(x, vgs, C.VDS_LOW)
        m = sub < 3e-6
        ax.semilogy(vgs[m], sub[m], "-", color="#1f77b4", lw=1.6, label="twin, 0.05 V (subthr.)")
        r = O.evaluate(x)
        ax.set_title(f"{tag}: $L_g$={x[0]:.1f}, $W$={x[1]:.1f}, $L_{{sp}}$={x[4]:.1f} nm", fontsize=11)
        ax.text(0.97, 0.05, "TCAD: SS=%.1f, DIBL=%.0f\ntwin:  SS=%.1f, DIBL=%.0f"
                % (D["SS"][i], D["DIBL"][i], r.SS, r.DIBL), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_xlabel("$V_{GS}$ [V]"); ax.set_ylabel("$I_D$ [A/µm of gated perimeter]")
        ax.set_ylim(1e-14, 2e-3)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_calibration_idvg")


def calib():
    fig_calibration_parity()
    fig_calibration_idvg()


# ---------------------------------------------------------------------------
# Main-study figures/tables
# ---------------------------------------------------------------------------
def load_study():
    with open(os.path.join(C.RESULTS_DIR, "study_results.pkl"), "rb") as f:
        data = pickle.load(f)
    summary = json.load(open(os.path.join(C.RESULTS_DIR, "study_summary.json")))
    return data, summary


def _pad(h, n):
    h = np.asarray(h, float)
    return h[:n] if len(h) >= n else np.concatenate([h, np.full(n - len(h), h[-1])])


def fig_convergence(data, n0=None):
    n0 = n0 or C.HP["n0"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for m in METHOD_ORDER:
        Y = np.array([_pad(data["results"][s][m]["hv_history"], data["budget"]) for s in data["seeds"]])
        x = np.arange(1, Y.shape[1] + 1)
        mu = Y.mean(0)
        ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(len(Y))
        ax.plot(x, mu, color=METHOD_COLORS[m], lw=2.6 if m == "HERA" else 1.6,
                label=METHOD_LABEL[m], zorder=5 if m == "HERA" else 3)
        ax.fill_between(x, mu - ci, mu + ci, color=METHOD_COLORS[m], alpha=0.15)
    ax.axvline(n0, ls="--", color="k", lw=1, alpha=0.6)
    ax.text(n0 + 0.8, 0.02, "end of shared initial design", rotation=90, fontsize=9, alpha=0.7)
    ax.set_xlabel("number of expensive evaluations $N$")
    ax.set_ylabel("feasible Pareto hypervolume (normalised)")
    ax.legend(ncol=2, loc="lower right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig_convergence")


def fig_final_box(data, summary):
    final = {m: np.array([data["results"][s][m]["hv_history"][-1] for s in data["seeds"]])
             for m in METHOD_ORDER}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bp = ax.boxplot([final[m] for m in METHOD_ORDER],
                    tick_labels=[METHOD_LABEL[m] for m in METHOD_ORDER],
                    patch_artist=True, widths=0.6, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="k", markeredgecolor="k", markersize=5))
    for b, m in zip(bp["boxes"], METHOD_ORDER):
        b.set_facecolor(METHOD_COLORS[m])
        b.set_alpha(0.55)
    top = max(v.max() for v in final.values())
    for i, m in enumerate(METHOD_ORDER):
        if m == "HERA":
            continue
        p = summary["significance"][m]["p_value"]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        ax.text(i + 1, top * 1.015, star, ha="center", fontsize=12)
    ax.set_ylim(top=top * 1.08)
    ax.set_ylabel("final feasible Pareto hypervolume")
    fig.tight_layout()
    save(fig, "fig_final_hv_box")


def table_comparison(summary):
    rows = ["| Method | HV (mean ± s.d.) | 95% CI | ΔHV vs HERA | p (Wilcoxon) | HERA wins | CPU time / run (s) |",
            "|---|---|---|---|---|---|---|"]
    for m in METHOD_ORDER:
        s = summary["stats"][m]
        t = summary["cpu_time_per_run"][m]
        if m == "HERA":
            rows.append(f"| **HERA** | **{s['mean']:.4f} ± {s['sd']:.4f}** | ±{s['ci95']:.4f} | — | — | — | "
                        f"{t['mean_s']:.1f} ± {t['sd_s']:.1f} |")
        else:
            g = summary["significance"][m]
            n = g["wins"] + g["losses"] + g["ties"]
            rows.append(f"| {METHOD_LABEL[m]} | {s['mean']:.4f} ± {s['sd']:.4f} | ±{s['ci95']:.4f} | "
                        f"{-g['mean_improvement']:+.4f} | {g['p_value']:.1e} | {g['wins']}/{n} | "
                        f"{t['mean_s']:.1f} ± {t['sd_s']:.1f} |")
    open(os.path.join(C.RESULTS_DIR, "table_comparison.md"), "w", encoding="utf-8").write("\n".join(rows))
    print("\n".join(rows))


def study():
    data, summary = load_study()
    fig_convergence(data)
    fig_final_box(data, summary)
    table_comparison(summary)


# ---------------------------------------------------------------------------
# TCAD re-scoring of every method's final Pareto set
# ---------------------------------------------------------------------------
def tcad_objectives(X, M):
    """Objective vector and constraints recomputed from DEVSIM per-perimeter metrics."""
    X = np.atleast_2d(X)
    Lg, W, Hf, Lsp = X[:, 0], X[:, 1], X[:, 2], X[:, 4]
    Weff_nm = 2.0 * (Hf + W)
    Ion_fin = M["Ion"] * Weff_nm * 1e-3
    Cint = M["dQg_lin"] * 1e4 / C.VDD * (Lg * 1e-9) * (Weff_nm * 1e-9)
    Cf = 2.0 * (2.0 * O.EPS_SP * C.EPS0 / np.pi) * np.log1p(O.H_GATE / Lsp) * (Weff_nm * 1e-9)
    SP = Ion_fin / ((Cint + Cf + O.C_LOAD) * C.VDD)
    Y = np.column_stack([np.log10(M["Ion"] / M["Ioff"]), -M["SS"] / 1000.0, np.log10(SP)])
    g = O.geometric_constraints(X)
    c = np.column_stack([g, (M["DIBL"] - C.DIBL_MAX) / C.DIBL_MAX,
                         np.log10(M["Ioff"] / C.IOFF_MAX_UM)])
    return Y, c


def tcad_collect():
    """Write the unique final-Pareto designs of all methods/seeds for DEVSIM."""
    data, _ = load_study()
    rows, index = [], []
    for s in data["seeds"]:
        for m in METHOD_ORDER:
            P = np.atleast_2d(data["results"][s][m]["pareto_X"])
            for k in range(len(P) if P.size else 0):
                rows.append(P[k])
                index.append((s, m, k))
    X = np.array(rows)
    Xu, inv = np.unique(np.round(X, 10), axis=0, return_inverse=True)
    np.save(os.path.join(C.RESULTS_DIR, "pareto_for_tcad.npy"), Xu)
    with open(os.path.join(C.RESULTS_DIR, "pareto_for_tcad_index.pkl"), "wb") as f:
        pickle.dump(dict(index=index, inv=np.ravel(inv)), f)
    print(f"{len(X)} Pareto designs, {len(Xu)} unique -> pareto_for_tcad.npy")


def tcad_rescore():
    from scipy.stats import wilcoxon
    data, _ = load_study()
    T = np.load(os.path.join(C.RESULTS_DIR, "devsim_pareto.npz"))
    idx = pickle.load(open(os.path.join(C.RESULTS_DIR, "pareto_for_tcad_index.pkl"), "rb"))
    Xu = T["X"]
    M = {k: T[k] for k in ["Ion", "Ioff", "SS", "DIBL", "dQg_lin", "Vth_sat"]}
    ok = T["ok"].astype(bool)
    Yt, ct = tcad_objectives(Xu, M)
    feas_t = ok & np.all(ct <= 0, axis=1)
    Yo = O.objectives(Xu)
    index = idx["index"]
    inv = idx["inv"]
    hv = {m: [] for m in METHOD_ORDER}
    keep = {m: [] for m in METHOD_ORDER}
    for s in data["seeds"]:
        for m in METHOD_ORDER:
            sel = [j for j, (ss, mm, k) in enumerate(index) if ss == s and mm == m]
            u = inv[sel]
            f = feas_t[u]
            keep[m].append(f.mean() if len(f) else np.nan)
            hv[m].append(H.hypervolume(C.normalize_y(Yt[u][f]), C.REF_POINT_NORM) if f.any() else 0.0)
    hv = {m: np.array(v) for m, v in hv.items()}
    out = dict(stats={}, sig={}, retained_feasible={})
    for m in METHOD_ORDER:
        v = hv[m]
        out["stats"][m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                               ci95=float(1.96 * v.std(ddof=1) / np.sqrt(len(v))))
        out["retained_feasible"][m] = float(np.nanmean(keep[m]))
        if m != "HERA":
            d = hv["HERA"] - v
            try:
                p = float(wilcoxon(hv["HERA"], v).pvalue)
            except ValueError:
                p = 1.0
            out["sig"][m] = dict(p_value=p, wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                                 mean_improvement=float(d.mean()))
    e = Yo[ok] - Yt[ok]
    out["objective_error"] = dict(
        n=int(ok.sum()), n_total=int(len(ok)),
        y1_MAE_dec=float(np.mean(np.abs(e[:, 0]))),
        SS_MAE_mVdec=float(1000 * np.mean(np.abs(e[:, 1]))),
        y3_MAE_dec=float(np.mean(np.abs(e[:, 2]))),
        DIBL_MAE=float(np.mean(np.abs(O.evaluate_array(Xu[ok])["DIBL"] - M["DIBL"][ok]))),
        R2_y1=float(_r2(Yt[ok, 0], Yo[ok, 0])), R2_y2=float(_r2(Yt[ok, 1], Yo[ok, 1])),
        R2_y3=float(_r2(Yt[ok, 2], Yo[ok, 2])),
        frac_tcad_infeasible=float(np.mean(~feas_t[ok])))
    out["hv_per_seed"] = {m: hv[m].tolist() for m in METHOD_ORDER}
    json.dump(out, open(os.path.join(C.RESULTS_DIR, "tcad_rescore_summary.json"), "w"), indent=1)
    rows = ["| Method | TCAD-scored HV (mean ± s.d.) | ΔHV vs HERA | p (Wilcoxon) | HERA wins | Pareto designs TCAD-feasible |",
            "|---|---|---|---|---|---|"]
    for m in METHOD_ORDER:
        s = out["stats"][m]
        r = out["retained_feasible"][m]
        if m == "HERA":
            rows.append(f"| **HERA** | **{s['mean']:.4f} ± {s['sd']:.4f}** | — | — | — | {100 * r:.0f}% |")
        else:
            g = out["sig"][m]
            rows.append(f"| {METHOD_LABEL[m]} | {s['mean']:.4f} ± {s['sd']:.4f} | {-g['mean_improvement']:+.4f} | "
                        f"{g['p_value']:.1e} | {g['wins']}/{len(hv[m])} | {100 * r:.0f}% |")
    open(os.path.join(C.RESULTS_DIR, "table_tcad_rescore.md"), "w", encoding="utf-8").write("\n".join(rows))
    print("\n".join(rows))
    print(json.dumps(out["objective_error"], indent=1))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    labs = ["$y_1=\\log_{10}(I_{on}/I_{off})$", "SS [mV/dec]", "$y_3=\\log_{10}$SP [s$^{-1}$]"]
    for j, ax in enumerate(axes):
        a, b = Yt[ok, j], Yo[ok, j]
        if j == 1:
            a, b = -1000 * a, -1000 * b
        ax.scatter(a, b, s=10, alpha=0.5, color="#c0392b")
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("DEVSIM TCAD " + labs[j])
        ax.set_ylabel("digital twin " + labs[j])
        ax.set_title(f"({'abc'[j]})  $R^2$ = {_r2(a, b):.3f}")
    fig.tight_layout()
    save(fig, "fig_tcad_pareto_parity")


# ---------------------------------------------------------------------------
# Standard benchmarks (DTLZ) and ablation figures
# ---------------------------------------------------------------------------
def fig_benchmarks():
    curves = np.load(os.path.join(C.RESULTS_DIR, "benchmark_curves.npz"))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, prob, tag in zip(axes, ["DTLZ2", "DTLZ4", "C2-DTLZ2"], "abc"):
        for m in ["HERA", "qEHVI", "qParEGO", "NSGA-II", "Random"]:
            Y = curves[f"{prob}__{m}"]
            x = np.arange(1, Y.shape[1] + 1)
            mu = Y.mean(0)
            ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
            ax.plot(x, mu, color=METHOD_COLORS[m], lw=2.4 if m == "HERA" else 1.5, label=METHOD_LABEL[m],
                    zorder=5 if m == "HERA" else 3)
            ax.fill_between(x, mu - ci, mu + ci, color=METHOD_COLORS[m], alpha=0.13)
        ax.axvline(C.HP["n0"], ls="--", color="k", lw=1, alpha=0.5)
        ax.set_title(f"({tag}) {prob}")
        ax.set_xlabel("number of evaluations $N$")
    axes[0].set_ylabel("normalised hypervolume")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig_benchmarks")


ABL_ORDER = ["q-EHVI (pool)", "+ NSGA-II acq. opt.", "+ RL warm-start (hier.)", "Full HERA (+ gate)",
             "RL warm-start (flat)"]
ABL_LABEL = {"q-EHVI (pool)": "q-EHVI (fixed pool)", "+ NSGA-II acq. opt.": "+ NSGA-II acquisition optimizer",
             "+ RL warm-start (hier.)": "+ RL warm-start (hierarchical)", "Full HERA (+ gate)": "+ gate = full HERA",
             "RL warm-start (flat)": "NSGA-II + RL (flat policy)"}
ABL_COL = {"q-EHVI (pool)": "#1f77b4", "+ NSGA-II acq. opt.": "#2ca02c", "+ RL warm-start (hier.)": "#9467bd",
           "Full HERA (+ gate)": "#c0392b", "RL warm-start (flat)": "#7f7f7f"}


def ablation():
    curves = np.load(os.path.join(C.RESULTS_DIR, "ablation_curves.npz"))
    summ = json.load(open(os.path.join(C.RESULTS_DIR, "ablation_summary.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for v in ABL_ORDER:
        Y = curves[v.replace(" ", "_")]
        x = np.arange(1, Y.shape[1] + 1)
        mu = Y.mean(0)
        ci = 1.96 * Y.std(0, ddof=1) / np.sqrt(Y.shape[0])
        a.plot(x, mu, color=ABL_COL[v], lw=2.4 if v.startswith("Full") else 1.6,
               ls="--" if "flat" in v else "-", label=ABL_LABEL[v])
        a.fill_between(x, mu - ci, mu + ci, color=ABL_COL[v], alpha=0.1)
    a.axvline(C.HP["n0"], ls="--", color="k", lw=1, alpha=0.5)
    a.axvline(summ["early"], ls=":", color="k", lw=1, alpha=0.5)
    a.set_xlabel("number of expensive evaluations $N$"); a.set_ylabel("feasible Pareto hypervolume")
    a.set_title("(a) convergence"); a.legend(loc="lower right", fontsize=9)
    yy = np.arange(len(ABL_ORDER))
    e = [summ["stats"][v]["hv_early_mean"] for v in ABL_ORDER]
    es = [summ["stats"][v]["hv_early_sd"] for v in ABL_ORDER]
    f = [summ["stats"][v]["hv_final_mean"] for v in ABL_ORDER]
    fs = [summ["stats"][v]["hv_final_sd"] for v in ABL_ORDER]
    b.barh(yy + 0.2, e, 0.38, xerr=es, color="#9ecae1", label=f"after {summ['early']} evaluations")
    b.barh(yy - 0.2, f, 0.38, xerr=fs, color="#c0392b", alpha=0.8, label=f"after {summ['budget']} evaluations")
    b.set_yticks(yy); b.set_yticklabels([ABL_LABEL[v] for v in ABL_ORDER], fontsize=9.5)
    b.invert_yaxis()
    lo = min(e) - 0.03
    b.set_xlim(lo, max(f) + 0.02)
    b.set_xlabel("feasible Pareto hypervolume"); b.set_title("(b) hypervolume (mean ± s.d.)")
    b.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_ablation")
    rows = ["| Configuration | HV@32 | HV@64 | AUC | ΔHV@64 vs full HERA (p) |", "|---|---|---|---|---|"]
    for v in ABL_ORDER:
        s = summ["stats"][v]
        g = summ["sig"].get(v)
        sig = "—" if g is None or v.startswith("Full") else (
            f"{-g['hv_gain']:+.4f} ({g['p_value']:.1e})" if g["p_value"] == g["p_value"] else f"{-g['hv_gain']:+.4f} (identical)")
        rows.append(f"| {ABL_LABEL[v]} | {s['hv_early_mean']:.4f} ± {s['hv_early_sd']:.4f} | "
                    f"{s['hv_final_mean']:.4f} ± {s['hv_final_sd']:.4f} | {s['auc_mean']:.4f} | {sig} |")
    open(os.path.join(C.RESULTS_DIR, "table_ablation_caee.md"), "w", encoding="utf-8").write("\n".join(rows))
    print("\n".join(rows))


# ---------------------------------------------------------------------------
# Sensitivity of the comparison to the hypervolume reference frame
# ---------------------------------------------------------------------------
def hv_frame_sensitivity():
    from scipy.stats import wilcoxon
    data, _ = load_study()
    allY = np.vstack([np.asarray(data["results"][s][m]["Y"])[np.asarray(data["results"][s][m]["feas"], bool)]
                      for s in data["seeds"] for m in METHOD_ORDER])
    base_lo, base_hi = C.OBJ_MIN.copy(), C.OBJ_MAX.copy()
    span = base_hi - base_lo
    frames = {
        "a priori (main)": (base_lo, base_hi),
        "data-driven (union of all feasible evaluations)": (allY.min(0), allY.max(0)),
        "a priori, enlarged by 10% per side": (base_lo - 0.1 * span, base_hi + 0.1 * span),
        "a priori, tightened nadir (y1>=7, SS<=70 mV/dec, y3>=11.7)": (np.array([7.0, -0.070, 11.7]), base_hi),
    }
    rows = ["| Reference frame | HERA | q-EHVI | ParEGO | NSGA-II | Random | HERA vs q-EHVI: wins, p |",
            "|---|---|---|---|---|---|---|"]
    out = {}
    for name, (lo, hi) in frames.items():
        final = {}
        for m in METHOD_ORDER:
            v = []
            for s in data["seeds"]:
                r = data["results"][s][m]
                Y = np.asarray(r["Y"])[np.asarray(r["feas"], bool)]
                Yn = (Y - lo) / (hi - lo)
                v.append(H.hypervolume(Yn, np.zeros(3)) if len(Yn) else 0.0)
            final[m] = np.array(v)
        d = final["HERA"] - final["qEHVI"]
        p = float(wilcoxon(final["HERA"], final["qEHVI"]).pvalue)
        others_beaten = all((final["HERA"] > final[m]).sum() == len(d) for m in METHOD_ORDER if m != "HERA")
        out[name] = dict(means={m: float(final[m].mean()) for m in METHOD_ORDER}, p_vs_qehvi=p,
                         wins_vs_qehvi=int((d > 0).sum()), wins_all_baselines_all_seeds=bool(others_beaten))
        rows.append(f"| {name} | " + " | ".join(f"{final[m].mean():.3f}" for m in METHOD_ORDER)
                    + f" | {(d > 0).sum()}/{len(d)}, {p:.1e} |")
    json.dump(out, open(os.path.join(C.RESULTS_DIR, "hv_frame_sensitivity.json"), "w"), indent=1)
    open(os.path.join(C.RESULTS_DIR, "table_hv_frame_sensitivity.md"), "w", encoding="utf-8").write("\n".join(rows))
    print("\n".join(rows))
    print({k: v["wins_all_baselines_all_seeds"] for k, v in out.items()})


# ---------------------------------------------------------------------------
# Representative seed: Pareto figure, chosen design, heat maps, tables
# ---------------------------------------------------------------------------
def representative_seed(data):
    finals = {s: data["results"][s]["HERA"]["hv_history"][-1] for s in data["seeds"]}
    med = np.median(list(finals.values()))
    return min(data["seeds"], key=lambda s: abs(finals[s] - med))


def pick_knee(PX, PY):
    """Balanced design: maximise the worst normalised objective (max-min knee)."""
    Yn = C.normalize_y(PY)
    return int(np.argmax(Yn.min(axis=1)))


def _device_table(X, labels, name, chosen_row=None):
    d = O.evaluate_array(X)
    hdr = ("| ID | $L_g$ (nm) | $W$ (nm) | $H$ (nm) | EOT (nm) | $L_{sp}$ (nm) | $N_{ch}$ (cm$^{-3}$) | "
           "$I_{on}$ (µA/µm) | $I_{off}$ (pA/µm) | SS (mV/dec) | DIBL (mV/V) | $y_1$ | $y_3$ | feasible |")
    rows = [hdr, "|" + "---|" * 14]
    feas = O.is_feasible(X)
    for i in range(len(X)):
        x = X[i]
        rows.append(f"| {labels[i]} | {x[0]:.2f} | {x[1]:.2f} | {x[2]:.1f} | {x[3]:.3f} | {x[4]:.2f} | "
                    f"{x[5]:.2e} | {d['Ion_um'][i]*1e6:.0f} | {d['Ioff_um'][i]*1e12:.1f} | {d['SS'][i]:.1f} | "
                    f"{d['DIBL'][i]:.1f} | {d['y1'][i]:.2f} | {d['y3'][i]:.3f} | {'yes' if feas[i] else 'no'} |")
    open(os.path.join(C.RESULTS_DIR, name), "w", encoding="utf-8").write("\n".join(rows))


def fig_pareto(data, seed, chosen_y):
    labs = ["$y_1=\\log_{10}(I_{on}/I_{off})$", "SS [mV/dec]", "$y_3=\\log_{10}$SP [s$^{-1}$]"]
    fig = plt.figure(figsize=(17.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.38)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    a2 = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    for m in ["Random", "NSGA-II", "qEHVI", "HERA"]:
        Y = np.atleast_2d(data["results"][seed][m]["pareto_Y"])
        if Y.size == 0:
            continue
        ss = -1000 * Y[:, 1]
        big = m == "HERA"
        kw = dict(s=40 if big else 20, color=METHOD_COLORS[m], alpha=0.95 if big else 0.6,
                  label=METHOD_LABEL[m], edgecolors="k" if big else "none", linewidths=0.4)
        ax3.scatter(Y[:, 0], ss, Y[:, 2], depthshade=False, **kw)
        a2[0].scatter(Y[:, 0], Y[:, 2], **kw)
        a2[1].scatter(Y[:, 0], ss, **kw)
    cy = chosen_y
    ax3.scatter(cy[0], -1000 * cy[1], cy[2], s=260, marker="*", color="gold", edgecolors="k", zorder=10,
                label="selected design")
    a2[0].scatter(cy[0], cy[2], s=300, marker="*", color="gold", edgecolors="k", zorder=10)
    a2[1].scatter(cy[0], -1000 * cy[1], s=300, marker="*", color="gold", edgecolors="k", zorder=10)
    ax3.set_xlabel(labs[0], labelpad=8); ax3.set_ylabel(labs[1], labelpad=8); ax3.set_zlabel(labs[2], labelpad=8)
    ax3.view_init(22, -55); ax3.set_title("(a) objective space")
    ax3.legend(fontsize=8.5, loc="upper left")
    a2[0].set_xlabel(labs[0]); a2[0].set_ylabel(labs[2]); a2[0].set_title("(b) on/off ratio vs switching speed")
    a2[1].set_xlabel(labs[0]); a2[1].set_ylabel(labs[1]); a2[1].set_title("(c) on/off ratio vs swing")
    a2[1].invert_yaxis()
    ax3.set_zlabel("$y_3$", labelpad=4)
    save(fig, "fig_pareto")


def fig_heatmaps(x0):
    from matplotlib.colors import LogNorm
    N = 160
    Lg = np.linspace(C.LB[0], C.UB[0], N)
    W = np.linspace(C.LB[1], C.UB[1], N)
    LG, WW = np.meshgrid(Lg, W)
    X = np.tile(x0, (LG.size, 1))
    X[:, 0] = LG.ravel(); X[:, 1] = WW.ravel()
    d = O.evaluate_array(X)
    feas = O.is_feasible(X).reshape(LG.shape)
    panels = [("(a) $I_{on}$ [µA/µm]", (d["Ion_um"] * 1e6).reshape(LG.shape), "viridis", None),
              ("(b) $I_{off}$ [A/µm]", d["Ioff_um"].reshape(LG.shape), "plasma", "log"),
              ("(c) $\\log_{10}(I_{on}/I_{off})$", d["y1"].reshape(LG.shape), "turbo", None)]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (t, Z, cm, sc) in zip(axes, panels):
        norm = LogNorm(vmin=Z.min(), vmax=Z.max()) if sc == "log" else None
        pc = ax.pcolormesh(LG, WW, Z, cmap=cm, norm=norm, shading="gouraud")
        fig.colorbar(pc, ax=ax)
        ax.contourf(LG, WW, (~feas).astype(float), levels=[0.5, 1.5], colors="white", alpha=0.45)
        ax.contour(LG, WW, feas.astype(float), levels=[0.5], colors="k", linestyles="--", linewidths=1.3)
        ax.plot(x0[0], x0[1], marker="*", ms=17, color="gold", mec="k")
        ax.set_xlabel("gate length $L_g$ [nm]"); ax.set_ylabel("body thickness $W$ [nm]")
        ax.set_title(t); ax.grid(False)
    fig.tight_layout()
    save(fig, "fig_heatmaps")


def showcase():
    data, _ = load_study()
    seed = representative_seed(data)
    r = data["results"][seed]["HERA"]
    PX, PY = np.atleast_2d(r["pareto_X"]), np.atleast_2d(r["pareto_Y"])
    j = pick_knee(PX, PY)
    x_ch, y_ch = PX[j], PY[j]
    print(f"representative seed {seed}: {len(PX)} HERA Pareto designs; chosen {np.round(x_ch, 3)}")
    np.save(os.path.join(C.RESULTS_DIR, "chosen_design.npy"), x_ch[None])
    json.dump(dict(seed=int(seed), x=x_ch.tolist(), y=y_ch.tolist(), n_pareto=int(len(PX))),
              open(os.path.join(C.RESULTS_DIR, "chosen_design.json"), "w"), indent=1)
    Xi = np.atleast_2d(r["init_X"])
    _device_table(Xi, [f"D{i+1}" for i in range(len(Xi))], "table_initial_designs.md")
    order = np.argsort(-PY[:, 0])
    _device_table(PX[order], [("P*" if k == j else f"P{n+1}") for n, k in enumerate(order)],
                  "table_pareto_designs.md")
    fig_pareto(data, seed, y_ch)
    fig_heatmaps(x_ch)


def chosen_tcad():
    """Twin vs DEVSIM for the selected design (needs devsim_chosen.npz)."""
    T = np.load(os.path.join(C.RESULTS_DIR, "devsim_chosen.npz"))
    x = T["X"][0]
    r = O.evaluate(x)
    M = {k: T[k][0] for k in ["SS", "DIBL", "Vth_sat", "Vth_lin", "Ion", "Ioff", "log10_ratio", "dQg_lin"]}
    Yt, _ = tcad_objectives(T["X"], {k: T[k] for k in ["Ion", "Ioff", "SS", "DIBL", "dQg_lin"]})
    rows = ["| Quantity | Digital twin | DEVSIM TCAD | Difference |", "|---|---|---|---|"]
    def add(n, a, b, fmt, rel=True):
        diff = f"{100 * (a - b) / abs(b):+.1f}%" if rel else f"{a - b:+.3g}"
        rows.append(f"| {n} | {fmt % a} | {fmt % b} | {diff} |")
    add("SS (mV/dec)", r.SS, M["SS"], "%.1f")
    add("DIBL (mV/V)", r.DIBL, M["DIBL"], "%.1f")
    add("V_th,sat (V)", r.Vth_sat, M["Vth_sat"], "%.3f", rel=False)
    add("I_on (µA/µm)", r.Ion_um * 1e6, M["Ion"] * 1e6, "%.0f")
    add("I_off (pA/µm)", r.Ioff_um * 1e12, M["Ioff"] * 1e12, "%.2f")
    add("log10(I_on/I_off)", r.y1, M["log10_ratio"], "%.2f", rel=False)
    add("ΔQ_g (µC/cm²)", r.dQg * 1e6, M["dQg_lin"] * 1e6, "%.3f")
    add("log10 SP (s⁻¹)", r.y3, Yt[0, 2], "%.3f", rel=False)
    open(os.path.join(C.RESULTS_DIR, "table_chosen_tcad.md"), "w", encoding="utf-8").write("\n".join(rows))
    print("\n".join(rows))
    vgs = T["vgs"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, sc in zip(axes, ["log", "linear"]):
        ax.plot(vgs, T["Id_sat"][0] * 1e6, "o", ms=4.5, color="#c0392b", label="TCAD, $V_{DS}$=0.70 V")
        ax.plot(vgs, T["Id_lin"][0] * 1e6, "s", ms=4, color="#1f77b4", label="TCAD, $V_{DS}$=0.05 V")
        ax.plot(vgs, O.idvg_um(x, vgs, C.VDS_HIGH) * 1e6, "-", color="#c0392b", lw=1.8, label="twin, 0.70 V")
        if sc == "log":
            sub = O.idvg_um(x, vgs, C.VDS_LOW) * 1e6
            m = sub < 3
            ax.plot(vgs[m], sub[m], "-", color="#1f77b4", lw=1.6, label="twin, 0.05 V (subthr.)")
            ax.axhline(C.IOFF_MAX_UM * 1e6, color="grey", ls=":", lw=1.2)
            ax.text(0.35, C.IOFF_MAX_UM * 1e6 * 1.6, "IRDS HD $I_{off}$ limit", fontsize=9, color="grey")
        ax.set_yscale(sc)
        ax.set_xlabel("$V_{GS}$ [V]"); ax.set_ylabel("$I_D$ [µA/µm]")
        ax.set_title("(a) logarithmic scale" if sc == "log" else "(b) linear scale")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_chosen_idvg")


if __name__ == "__main__":
    {"calib": calib, "study": study, "tcad_collect": tcad_collect,
     "tcad_rescore": tcad_rescore, "showcase": showcase,
     "chosen_tcad": chosen_tcad, "hv_frame": hv_frame_sensitivity, "bench": fig_benchmarks, "ablation": ablation}[sys.argv[1] if len(sys.argv) > 1 else "calib"]()
