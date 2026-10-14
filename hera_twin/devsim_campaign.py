"""Parallel DEVSIM campaigns: calibration/validation design sets and re-simulation.

    python -m hera_twin.devsim_campaign calib  --n 128 --workers 18
    python -m hera_twin.devsim_campaign points --file results/pareto_for_tcad.npy --tag pareto

Each design is simulated by devsim_sim.run_idvg (Id-Vg and gate charge at low and
high drain bias).  Results (inputs, extracted metrics, curves, wall-clock time)
are written to results/devsim_<tag>.npz and a JSON summary.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import qmc

from . import config as C

KEYS = ["Vth_lin", "Vth_sat", "DIBL", "SS", "SS_lin", "Ion", "Ioff", "log10_ratio",
        "dQg_lin", "dQg_sat"]


def _worker(x):
    import contextlib, io
    from . import devsim_sim as D
    import devsim as ds
    ds.set_parameter(name="debug_level", value="error")
    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            m = D.run_idvg(np.asarray(x, float))
        ok = all(np.isfinite(m[k]) for k in KEYS)
        out = {k: float(m[k]) for k in KEYS}
        out.update(ok=bool(ok), Id_lin=m["Id_lin"], Id_sat=m["Id_sat"],
                   Qg_lin=m["Qg_lin"], Qg_sat=m["Qg_sat"], vgs=m["vgs"])
    except Exception as e:                      # convergence failure etc.
        out = {k: float("nan") for k in KEYS}
        out.update(ok=False, err=str(e)[:200])
    out["time_s"] = time.time() - t0
    return out


def calibration_designs(n, seed=2026):
    """Scrambled Sobol over the full design box (Hfin fixed: 2-D DEVSIM is per-perimeter)."""
    z = qmc.Sobol(d=6, seed=seed).random(n)
    X = C.denormalize_x(z)
    X[:, 2] = 40.0          # Hfin does not enter the per-perimeter 2-D quantities
    return X


def run_set(X, tag, workers):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_worker, list(X)))
    wall = time.time() - t0
    ok = np.array([r["ok"] for r in res])
    M = {k: np.array([r[k] for r in res]) for k in KEYS}
    t = np.array([r["time_s"] for r in res])
    curves = {}
    for name in ("Id_lin", "Id_sat", "Qg_lin", "Qg_sat"):
        curves[name] = np.array([r.get(name, np.full(29, np.nan)) for r in res], dtype=float)
    vgs = next((r["vgs"] for r in res if "vgs" in r), np.arange(29) * 0.025)
    np.savez(os.path.join(C.RESULTS_DIR, f"devsim_{tag}.npz"), X=X, ok=ok, time_s=t,
             vgs=vgs, **M, **curves)
    summ = dict(tag=tag, n=int(len(X)), n_ok=int(ok.sum()), wall_s=wall,
                time_per_design_s=dict(mean=float(t.mean()), median=float(np.median(t)),
                                       max=float(t.max())),
                failures=[dict(x=list(map(float, X[i])), err=res[i].get("err", ""))
                          for i in np.where(~ok)[0]])
    with open(os.path.join(C.RESULTS_DIR, f"devsim_{tag}.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "failures"}, indent=1))
    print("failures:", len(summ["failures"]))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "points"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--file", default="")
    ap.add_argument("--tag", default="calib")
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()
    if a.mode == "calib":
        run_set(calibration_designs(a.n), "calib", a.workers)
    else:
        X = np.load(a.file)
        run_set(np.atleast_2d(X), a.tag, a.workers)


if __name__ == "__main__":
    main()
