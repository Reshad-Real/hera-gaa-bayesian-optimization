"""TCAD re-scoring of the final Pareto sets of q-EHVI (L-BFGS-B) in the 30-seed twin study.

    py -3.14 -m hera_twin.tcad_rescore_extra --workers 17

Simulates (with DEVSIM, through tcad_loop.simulate and its cache) every design of the
qEHVI-LBFGS final Pareto sets, re-scores the fronts exactly as caee_report.tcad_rescore,
and merges the result into results/tcad_rescore_summary.json and table_tcad_rescore.md.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import wilcoxon

from . import config as C
from . import hypervolume as H
from . import tcad_loop as T
from .caee_report import tcad_objectives

NEW = "qEHVI-LBFGS"
SUMMARY = os.path.join(C.RESULTS_DIR, "tcad_rescore_summary.json")
ORDER = ["HERA", "qEHVI-LBFGS", "qEHVI", "qParEGO", "NSGA-II", "Random"]
LABEL = {"HERA": "HERA", "qEHVI-LBFGS": "q-EHVI (L-BFGS-B)", "qEHVI": "q-EHVI", "qParEGO": "ParEGO",
         "NSGA-II": "NSGA-II", "Random": "Random"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=17)
    a = ap.parse_args()
    data = pickle.load(open(os.path.join(C.RESULTS_DIR, "study_results.pkl"), "rb"))
    cache = T._load(T.CACHE, {})
    fronts = {s: np.atleast_2d(data["results"][s][NEW]["pareto_X"]) for s in data["seeds"]}
    todo = {}
    for P in fronts.values():
        for x in P:
            k = T.tcad_key(x)
            if k not in cache:
                todo[k] = x
    print(f"{sum(len(P) for P in fronts.values())} Pareto designs, {len(todo)} to simulate", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(T._simulate_now, x): k for k, x in todo.items()}
        for i, fu in enumerate(as_completed(futs), 1):
            cache[futs[fu]] = fu.result()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    T._save(T.CACHE, cache)

    hv, keep = [], []
    for s in data["seeds"]:
        X = fronts[s]
        rows = [cache[T.tcad_key(x)] for x in X]
        ok = np.array([r["ok"] for r in rows])
        M = {k: np.array([r[k] if r["ok"] else 1.0 for r in rows]) for k in ["Ion", "Ioff", "SS", "DIBL", "dQg_lin"]}
        Y, c = tcad_objectives(X, M)
        f = ok & np.all(c <= 0, axis=1)
        keep.append(f.mean() if len(f) else np.nan)
        hv.append(H.hypervolume(C.normalize_y(Y[f]), C.REF_POINT_NORM) if f.any() else 0.0)
    hv = np.array(hv)
    summ = json.load(open(SUMMARY))
    hera = np.array(summ["hv_per_seed"]["HERA"])
    summ["stats"][NEW] = dict(mean=float(hv.mean()), sd=float(hv.std(ddof=1)),
                              ci95=float(1.96 * hv.std(ddof=1) / np.sqrt(len(hv))))
    summ["retained_feasible"][NEW] = float(np.nanmean(keep))
    d = hera - hv
    summ["sig"][NEW] = dict(p_value=float(wilcoxon(hera, hv).pvalue), wins=int((d > 0).sum()),
                            losses=int((d < 0).sum()), mean_improvement=float(d.mean()))
    summ["hv_per_seed"][NEW] = hv.tolist()
    json.dump(summ, open(SUMMARY, "w"), indent=1)
    rows = ["| Method | TCAD-scored HV (mean ± s.d.) | ΔHV vs HERA | p (Wilcoxon) | HERA wins | Pareto designs TCAD-feasible |",
            "|---|---|---|---|---|---|"]
    for m in ORDER:
        s_ = summ["stats"][m]
        r = summ["retained_feasible"][m]
        if m == "HERA":
            rows.append(f"| **HERA** | **{s_['mean']:.4f} ± {s_['sd']:.4f}** | — | — | — | {100 * r:.0f}% |")
        else:
            g = summ["sig"][m]
            rows.append(f"| {LABEL[m]} | {s_['mean']:.4f} ± {s_['sd']:.4f} | {-g['mean_improvement']:+.4f} | "
                        f"{g['p_value']:.1e} | {g['wins']}/{len(hv)} | {100 * r:.0f}% |")
    open(os.path.join(C.RESULTS_DIR, "table_tcad_rescore.md"), "w", encoding="utf-8").write("\n".join(rows))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
