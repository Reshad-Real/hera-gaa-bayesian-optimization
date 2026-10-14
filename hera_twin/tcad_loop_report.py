"""Statistics, table and figure for the TCAD-in-the-loop study (tcad_loop.py).

    py -3.14 -m hera_twin.tcad_loop_report

Writes results/tcad_loop_summary.json, results/table_tcad_loop.md and
results/figures/fig_tcad_loop.{png,pdf}.

Reference "twin-only": HERA run on the digital twin (64 twin evaluations, 30-seed
study) whose final Pareto set is re-simulated in TCAD (caee_report.tcad_rescore);
its TCAD cost is one simulation per Pareto design, on top of the 128 calibration
and validation simulations of the twin.
"""
from __future__ import annotations

import os
import json
import pickle
import numpy as np
from scipy.stats import wilcoxon

from . import config as C
from . import paper_figures as PF          # print-size style and palette
import matplotlib.pyplot as plt

OUT = os.path.join(C.RESULTS_DIR, "tcad_loop_results.pkl")
ORDER = ["HERA-TID", "HERA-D", "HERA-TI", "HERA", "qEHVI-LBFGS-D", "qEHVI-LBFGS", "qEHVI-TI", "qEHVI", "Random"]
# The variants are one optimizer run with different prior knowledge, not separate algorithms:
# METHOD names the optimizer, PRIOR the prior knowledge given to its surrogates.
METHOD = {"HERA-TID": "HERA", "HERA-D": "HERA", "HERA-TI": "HERA", "HERA": "HERA",
          "qEHVI-LBFGS-D": "q-EHVI (L-BFGS-B)", "qEHVI-LBFGS": "q-EHVI (L-BFGS-B)",
          "qEHVI-TI": "q-EHVI", "qEHVI": "q-EHVI", "Random": "Random"}
PRIOR = {"HERA-TID": "twin prior + calibration data", "HERA-D": "calibration data", "HERA-TI": "twin prior",
         "HERA": "none", "qEHVI-LBFGS-D": "calibration data", "qEHVI-LBFGS": "none", "qEHVI-TI": "twin prior",
         "qEHVI": "none", "Random": "none"}
LABEL = {m: (METHOD[m] if PRIOR[m] == "none" else f"{METHOD[m]}, {PRIOR[m]}") for m in METHOD}
COLOR = {"HERA-TID": "#D55E00", "HERA-D": "#009E73", "HERA-TI": "#CC79A7", "HERA": "#000000",
         "qEHVI-LBFGS-D": "#56B4E9", "qEHVI-LBFGS": "#56B4E9", "qEHVI-TI": "#0072B2", "qEHVI": "#0072B2",
         "Random": "#7F7F7F"}
STYLE = {"HERA-TID": "-", "HERA-D": "-", "HERA-TI": "-", "HERA": (0, (5, 2)), "qEHVI-LBFGS-D": "-",
         "qEHVI-LBFGS": (0, (5, 2)), "qEHVI-TI": "-", "qEHVI": (0, (5, 2)), "Random": (0, (1, 2.5))}
CHECKPOINTS = [12, 16, 24, 40, 64]
PAIRS = [("HERA", "qEHVI"), ("HERA", "qEHVI-LBFGS"), ("HERA-D", "qEHVI-LBFGS-D"), ("qEHVI-LBFGS-D", "qEHVI-LBFGS"), ("HERA-TI", "HERA"), ("HERA-D", "HERA"), ("HERA-TID", "HERA"),
         ("HERA-D", "HERA-TI"), ("HERA-TID", "HERA-D"), ("HERA-TID", "HERA-TI"), ("qEHVI-TI", "qEHVI")]


def load():
    with open(OUT, "rb") as f:
        out = pickle.load(f)
    runs = out["runs"]
    methods = [m for m in ORDER if any(k[1] == m for k in runs)]
    seeds = sorted({k[0] for k in runs})
    seeds = [s for s in seeds if all((s, m) in runs for m in methods)]
    return out["budget"], runs, methods, seeds


def twin_only_reference(seeds):
    """TCAD-scored HV and TCAD cost of the twin-optimized HERA fronts for these seeds."""
    data = pickle.load(open(os.path.join(C.RESULTS_DIR, "study_results.pkl"), "rb"))
    resc = json.load(open(os.path.join(C.RESULTS_DIR, "tcad_rescore_summary.json")))
    idx = {s: i for i, s in enumerate(data["seeds"])}
    hv = np.array([resc["hv_per_seed"]["HERA"][idx[s]] for s in seeds])
    n_tcad = np.array([len(np.atleast_2d(data["results"][s]["HERA"]["pareto_X"])) for s in seeds])
    return hv, n_tcad


def _w(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def summarize(budget, runs, methods, seeds):
    curves = {m: np.array([runs[(s, m)]["result"]["hv_history"][:budget] for s in seeds]) for m in methods}
    ref_hv, ref_n = twin_only_reference(seeds)
    summ = dict(budget=budget, seeds=seeds, n_seeds=len(seeds), methods=methods,
                twin_only=dict(hv_mean=float(ref_hv.mean()), hv_sd=float(ref_hv.std(ddof=1)),
                               n_tcad_mean=float(ref_n.mean()), n_tcad_min=int(ref_n.min()),
                               n_tcad_max=int(ref_n.max()), hv=ref_hv.tolist(), n_tcad=ref_n.tolist()),
                stats={}, checkpoints={}, reach_twin_only={}, pairs={}, vs_twin_only={})
    for m in methods:
        r = runs
        feas = np.mean([np.mean(r[(s, m)]["result"]["feas"][10:]) for s in seeds])
        nsim = np.sum([r[(s, m)]["n_sim"] for s in seeds])
        summ["stats"][m] = dict(
            hv_mean=float(curves[m][:, -1].mean()), hv_sd=float(curves[m][:, -1].std(ddof=1)),
            feasible_after_init=float(feas), n_sim=int(nsim),
            tcad_s_per_sim=float(np.sum([r[(s, m)]["sim_time"] for s in seeds]) / max(1, nsim)),
            n_fail=int(np.sum([r[(s, m)]["n_fail"] for s in seeds])),
            wall_s_per_run=float(np.mean([r[(s, m)]["wall_s"] for s in seeds])))
        summ["checkpoints"][m] = {str(n): dict(mean=float(curves[m][:, n - 1].mean()),
                                               sd=float(curves[m][:, n - 1].std(ddof=1)))
                                  for n in CHECKPOINTS if n <= budget}
        need = []
        for k in range(len(seeds)):
            hit = np.where(curves[m][k] >= ref_hv[k])[0]
            need.append(int(hit[0]) + 1 if len(hit) else None)
        got = [n for n in need if n is not None]
        summ["reach_twin_only"][m] = dict(n_reached=len(got), per_seed=need,
                                          median=float(np.median(got)) if got else None,
                                          min=int(min(got)) if got else None, max=int(max(got)) if got else None)
        d = curves[m][:, -1] - ref_hv
        summ["vs_twin_only"][m] = dict(mean_diff=float(d.mean()), p_value=_w(curves[m][:, -1], ref_hv),
                                       wins=int((d > 0).sum()))
    for a, b in PAIRS:
        if a in methods and b in methods:
            summ["pairs"][f"{a} vs {b}"] = {
                str(n): dict(mean_diff=float((curves[a][:, n - 1] - curves[b][:, n - 1]).mean()),
                             p_value=_w(curves[a][:, n - 1], curves[b][:, n - 1]),
                             wins=int((curves[a][:, n - 1] > curves[b][:, n - 1]).sum()))
                for n in (16, 24, 64)}
    return summ, curves, ref_hv, ref_n


def write_table(summ):
    rows = ["| Method | Prior knowledge | HV after 16 | HV after 24 | HV after 64 | Evaluations to reach twin-only HV |",
            "|---|---|---|---|---|---|"]
    best = max(summ["methods"], key=lambda m: summ["stats"][m]["hv_mean"])
    for m in summ["methods"]:
        cp = summ["checkpoints"][m]
        cells = [f"{cp[str(n)]['mean']:.3f} ± {cp[str(n)]['sd']:.3f}" for n in (16, 24, 64)]
        r = summ["reach_twin_only"][m]
        reach = (f"{r['median']:.0f} ({r['min']}–{r['max']})" + ("" if r["n_reached"] == summ["n_seeds"]
                 else f"; {r['n_reached']}/{summ['n_seeds']} seeds") if r["n_reached"] else "not reached")
        name = METHOD[m]
        if m == best:
            cells = [f"**{c}**" for c in cells]
        rows.append(f"| {name} | {PRIOR[m]} | " + " | ".join(cells) + f" | {reach} |")
    t = summ["twin_only"]
    rows.append(f"| HERA on the twin, TCAD-verified | twin as oracle | – | – | {t['hv_mean']:.3f} ± {t['hv_sd']:.3f} | "
                f"{t['n_tcad_mean']:.0f} verification simulations |")
    path = os.path.join(C.RESULTS_DIR, "table_tcad_loop.md")
    open(path, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print("\n".join(rows))


def figure(summ, curves, ref_hv, ref_n):
    fig = plt.figure(figsize=(PF.TW, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (a, b):
        for m in summ["methods"]:
            Y = curves[m]
            x = np.arange(1, Y.shape[1] + 1)
            mu, ci = PF._ci(Y)
            ax.plot(x, mu, color=COLOR[m], ls=STYLE[m], lw=1.7 if m == "HERA-TID" else 1.1,
                    label=LABEL[m], zorder=5)
            ax.fill_between(x, mu - ci, mu + ci, color=COLOR[m], alpha=0.12, lw=0)
        ax.errorbar([ref_n.mean()], [ref_hv.mean()], xerr=[[ref_n.mean() - ref_n.min()], [ref_n.max() - ref_n.mean()]],
                    yerr=[1.96 * ref_hv.std(ddof=1) / np.sqrt(len(ref_hv))], fmt="*", ms=9, color="k",
                    mfc="gold", mec="k", mew=0.5, elinewidth=0.7, capsize=2, zorder=6,
                    label="twin-only workflow (TCAD-verified)")
        ax.axhline(ref_hv.mean(), color="0.4", lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    a.axvline(10, ls="--", color="k", lw=0.8, alpha=0.5)
    a.set_xlabel("number of TCAD evaluations $N$"); a.set_ylabel("TCAD feasible Pareto hypervolume")
    a.set_xlim(1, summ["budget"]); a.set_ylim(0.35, 0.71)
    a.set_title("(a) Optimization with TCAD in the loop", loc="left")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, fontsize=7, handlelength=2.4, columnspacing=1.0)
    b.set_xlim(12, summ["budget"]); b.set_ylim(0.58, 0.70)
    b.set_xlabel("number of TCAD evaluations $N$")
    b.set_title("(b) Detail of (a)", loc="left")
    PF.save(fig, "fig_tcad_loop")


def main():
    budget, runs, methods, seeds = load()
    summ, curves, ref_hv, ref_n = summarize(budget, runs, methods, seeds)
    json.dump(summ, open(os.path.join(C.RESULTS_DIR, "tcad_loop_summary.json"), "w"), indent=1)
    for m in methods:
        st, r, v = summ["stats"][m], summ["reach_twin_only"][m], summ["vs_twin_only"][m]
        print(f"{m:9s} HV64={st['hv_mean']:.4f}+-{st['hv_sd']:.4f} feas={st['feasible_after_init']:.2f} "
              f"tcad/sim={st['tcad_s_per_sim']:.1f}s fail={st['n_fail']} wall/run={st['wall_s_per_run']:.0f}s "
              f"reach twin-only {r['n_reached']}/{len(seeds)} med N={r['median']} | vs twin-only "
              f"{v['mean_diff']:+.4f} p={v['p_value']:.2e} W={v['wins']}")
    for k, v in summ["pairs"].items():
        print(f"{k:22s} " + "  ".join(f"N{n}: {d['mean_diff']:+.4f} (p={d['p_value']:.3g}, W={d['wins']})"
                                         for n, d in v.items()))
    t = summ["twin_only"]
    print(f"twin-only: HV={t['hv_mean']:.4f}+-{t['hv_sd']:.4f}  TCAD sims={t['n_tcad_mean']:.1f} "
          f"({t['n_tcad_min']}-{t['n_tcad_max']})")
    write_table(summ)
    figure(summ, curves, ref_hv, ref_n)
    write_curves(curves, methods)


# column names of the mean-curve CSV used by the graphical abstract (pgfplots)
CSV_COL = {"HERA-TID": "HERA_twinprior_calib", "HERA-D": "HERA_calib", "HERA-TI": "HERA_twinprior", "HERA": "HERA",
           "qEHVI-LBFGS-D": "qEHVI_LBFGS_calib", "qEHVI-LBFGS": "qEHVI_LBFGS", "qEHVI-TI": "qEHVI_twinprior",
           "qEHVI": "qEHVI", "Random": "Random"}


def write_curves(curves, methods):
    ms = [m for m in ORDER if m in methods]
    mean = {m: curves[m].mean(axis=0) for m in ms}
    lines = ["N," + ",".join(CSV_COL[m] for m in ms)]
    for n in range(len(mean[ms[0]])):
        lines.append(f"{n + 1}," + ",".join(f"{mean[m][n]:.5f}" for m in ms))
    path = os.path.join(PF.FIG_DIR, "tcad_loop_mean_curves.csv")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
