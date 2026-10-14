"""Markdown rows of the result tables (Tables 6-9 and 11) from the stored results.

    py -3.14 -m hera_twin.paper_tables

Prints each table so it can be pasted into manuscript/sec6.md, and writes
results/table_*_paper.md. Includes q-EHVI with the L-BFGS-B acquisition optimizer.
"""
from __future__ import annotations

import os
import json
import pickle
import numpy as np
from scipy.stats import wilcoxon

from . import config as C
from . import hypervolume as H

R = C.RESULTS_DIR
ORDER = ["HERA", "qEHVI-LBFGS", "qEHVI", "qParEGO", "NSGA-II", "Random"]
LAB = {"HERA": "HERA", "qEHVI-LBFGS": "q-EHVI (L-BFGS-B)", "qEHVI": "q-EHVI", "qParEGO": "ParEGO",
       "NSGA-II": "NSGA-II", "Random": "Random"}


def fp(p):
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.01:
        return f"{p:.3f}"
    a, b = f"{p:.1e}".split("e")
    return f"${a}\\times10^{{{int(b)}}}$"


def _w(a, b):
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def table6():
    S = json.load(open(os.path.join(R, "benchmark_summary.json")))
    rows = ["| Problem | Method | HV | IGD$^{+}$ | $p$ | HERA wins |", "|---|---|---|---|---|---|"]
    for prob in ["DTLZ2", "DTLZ4", "C2-DTLZ2"]:
        st, sg = S[prob]["stats"], S[prob]["sig"]
        ms = [m for m in ["HERA", "qEHVI-LBFGS", "qEHVI", "qParEGO", "NSGA-II", "Random"] if m in st]
        best_hv = max(st[m]["hv_mean"] for m in ms)
        best_igd = min(st[m]["igd_mean"] for m in ms)
        for i, m in enumerate(ms):
            hv = f"{st[m]['hv_mean']:.3f} ± {st[m]['hv_sd']:.3f}"
            igd = f"{st[m]['igd_mean']:.3f} ± {st[m]['igd_sd']:.3f}"
            if abs(st[m]["hv_mean"] - best_hv) < 5e-4:
                hv = f"**{hv}**"
            if abs(st[m]["igd_mean"] - best_igd) < 5e-4:
                igd = f"**{igd}**"
            name = f"**{LAB[m]}**" if m == "HERA" else LAB[m]
            tail = "– | –" if m == "HERA" else f"{fp(sg[m]['p_value'])} | {sg[m]['wins']}/{S[prob]['n_seeds']}"
            rows.append(f"| {prob if i == 0 else ''} | {name} | {hv} | {igd} | {tail} |")
    return "\n".join(rows)


def table7():
    S = json.load(open(os.path.join(R, "ablation_summary.json")))
    order = ["q-EHVI (pool)", "q-EHVI (L-BFGS-B)", "+ NSGA-II acq. opt.", "+ RL warm-start (hier.)",
             "Full HERA (+ gate)", "RL warm-start (flat)"]
    lab = {"q-EHVI (pool)": "q-EHVI (fixed pool)", "q-EHVI (L-BFGS-B)": "q-EHVI + L-BFGS-B acquisition optimizer",
           "+ NSGA-II acq. opt.": "q-EHVI + NSGA-II acquisition optimizer",
           "+ RL warm-start (hier.)": "+ RL warm-start (hierarchical)", "Full HERA (+ gate)": "+ gate = full HERA",
           "RL warm-start (flat)": "NSGA-II + RL (flat policy)"}
    rows = ["| Configuration | HV after 32 | HV after 64 | AUC | $\\Delta$HV vs full HERA ($p$) |",
            "|---|---|---|---|---|"]
    for v in order:
        if v not in S["stats"]:
            continue
        st = S["stats"][v]
        g = S["sig"].get(v)
        sig = "–" if g is None or v.startswith("Full") else f"{-g['hv_gain']:+.3f} ({fp(g['p_value'])})"
        rows.append(f"| {lab[v]} | {st['hv_early_mean']:.3f} ± {st['hv_early_sd']:.3f} | "
                    f"{st['hv_final_mean']:.3f} ± {st['hv_final_sd']:.3f} | {st['auc_mean']:.3f} | {sig} |")
    return "\n".join(rows)


def table8():
    S = json.load(open(os.path.join(R, "study_summary.json")))
    rows = ["| Method | HV (mean ± s.d.) | 95% CI | $\\Delta$HV vs HERA | $p$ | HERA wins | CPU time / run (s) |",
            "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        st, t = S["stats"][m], S["cpu_time_per_run"][m]
        cpu = f"{t['mean_s']:.1f} ± {t['sd_s']:.1f}" if t["mean_s"] >= 1 else f"{t['mean_s']:.1f}"
        if m == "HERA":
            rows.append(f"| **HERA** | **{st['mean']:.3f} ± {st['sd']:.3f}** | ±{st['ci95']:.3f} | – | – | – | {cpu} |")
        else:
            g = S["significance"][m]
            rows.append(f"| {LAB[m]} | {st['mean']:.3f} ± {st['sd']:.3f} | ±{st['ci95']:.3f} | "
                        f"{-g['mean_improvement']:+.3f} | {fp(g['p_value'])} | {g['wins']}/30 | {cpu} |")
    return "\n".join(rows)


def table9():
    data = pickle.load(open(os.path.join(R, "study_results.pkl"), "rb"))
    allY = np.vstack([np.asarray(data["results"][s][m]["Y"])[np.asarray(data["results"][s][m]["feas"], bool)]
                      for s in data["seeds"] for m in ORDER])
    lo0, hi0 = C.OBJ_MIN.copy(), C.OBJ_MAX.copy()
    span = hi0 - lo0
    frames = [("A priori (main)", lo0, hi0),
              ("Data-driven (all feasible evaluations)", allY.min(0), allY.max(0)),
              ("A priori, enlarged 10% per side", lo0 - 0.1 * span, hi0 + 0.1 * span),
              ("Tightened nadir ($y_1\\ge7$, $\\mathrm{SS}\\le70$ mV/dec, $y_3\\ge11.7$)", np.array([7.0, -0.070, 11.7]), hi0)]
    cols = ["HERA", "qEHVI-LBFGS", "qEHVI", "qParEGO", "NSGA-II", "Random"]
    head = ["Reference frame", "HERA", "q-EHVI (L-BFGS-B)", "q-EHVI", "ParEGO", "NSGA-II", "Random",
            "HERA vs q-EHVI (L-BFGS-B): wins, $p$", "HERA vs q-EHVI: wins, $p$"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out = {}
    for name, lo, hi in frames:
        final = {}
        for m in cols:
            v = []
            for s in data["seeds"]:
                r = data["results"][s][m]
                Y = np.asarray(r["Y"])[np.asarray(r["feas"], bool)]
                v.append(H.hypervolume((Y - lo) / (hi - lo), np.zeros(3)) if len(Y) else 0.0)
            final[m] = np.array(v)
        cmp_ = []
        for b in ("qEHVI-LBFGS", "qEHVI"):
            d = final["HERA"] - final[b]
            cmp_.append(f"{(d > 0).sum()}/{len(d)}, {fp(_w(final['HERA'], final[b]))}")
        out[name] = {m: float(final[m].mean()) for m in cols}
        rows.append(f"| {name} | " + " | ".join(f"{final[m].mean():.3f}" for m in cols) + " | " + " | ".join(cmp_) + " |")
    json.dump(out, open(os.path.join(R, "hv_frame_sensitivity_paper.json"), "w"), indent=1)
    return "\n".join(rows)


def table11():
    S = json.load(open(os.path.join(R, "tcad_rescore_summary.json")))
    rows = ["| Method | TCAD-scored HV (mean ± s.d.) | $\\Delta$HV vs HERA | $p$ | HERA wins | Pareto designs TCAD-feasible |",
            "|---|---|---|---|---|---|"]
    for m in ORDER:
        if m not in S["stats"]:
            continue
        st, r = S["stats"][m], S["retained_feasible"][m]
        if m == "HERA":
            rows.append(f"| **HERA** | **{st['mean']:.3f} ± {st['sd']:.3f}** | – | – | – | {100 * r:.0f}% |")
        else:
            g = S["sig"][m]
            rows.append(f"| {LAB[m]} | {st['mean']:.3f} ± {st['sd']:.3f} | {-g['mean_improvement']:+.3f} | "
                        f"{fp(g['p_value'])} | {g['wins']}/30 | {100 * r:.0f}% |")
    return "\n".join(rows)


def main():
    for name, fn in [("6", table6), ("7", table7), ("8", table8), ("9", table9), ("11", table11)]:
        try:
            t = fn()
        except Exception as e:                       # a result file not yet complete
            t = f"(not available: {e!r})"
        open(os.path.join(R, f"table_{name}_paper.md"), "w", encoding="utf-8").write(t + "\n")
        print(f"=== Table {name}\n{t}\n")


if __name__ == "__main__":
    main()
