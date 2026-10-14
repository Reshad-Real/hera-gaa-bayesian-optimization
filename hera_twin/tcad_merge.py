"""Bring the TCAD re-simulation set up to date with the current final Pareto sets.

After `caee_report tcad_collect` has rewritten pareto_for_tcad.npy (e.g. after
the NSGA-II re-run), this script
  1. finds designs that are not yet in results/devsim_pareto.npz,
  2. writes them to results/pareto_missing.npy (simulate with
     `devsim_campaign points --file results/pareto_missing.npy --tag pareto_missing`),
  3. with `merge`, rebuilds devsim_pareto.npz in the order of pareto_for_tcad.npy.

    python -m hera_twin.tcad_merge missing
    python -m hera_twin.tcad_merge merge
"""
import os
import sys
import numpy as np
from . import config as C

KEYS = ["ok", "time_s", "Vth_lin", "Vth_sat", "DIBL", "SS", "SS_lin", "Ion", "Ioff", "log10_ratio",
        "dQg_lin", "dQg_sat", "Id_lin", "Id_sat", "Qg_lin", "Qg_sat"]


def _key(X):
    return [tuple(np.round(x, 9)) for x in np.atleast_2d(X)]


def _load(tag):
    p = os.path.join(C.RESULTS_DIR, f"devsim_{tag}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as f:                 # close the file (Windows rename safety)
        return {k: f[k] for k in f.files}


def missing():
    target = np.load(os.path.join(C.RESULTS_DIR, "pareto_for_tcad.npy"))
    have = set()
    for tag in ("pareto_raw", "pareto", "pareto_missing"):
        D = _load(tag)
        if D is not None:
            have |= set(_key(D["X"]))
    miss = np.array([x for x, k in zip(target, _key(target)) if k not in have])
    np.save(os.path.join(C.RESULTS_DIR, "pareto_missing.npy"), miss)
    print(f"{len(target)} target designs, {len(target) - len(miss)} already simulated, {len(miss)} missing")


def merge():
    target = np.load(os.path.join(C.RESULTS_DIR, "pareto_for_tcad.npy"))
    pool = {}
    vgs = None
    for tag in ("pareto_raw", "pareto", "pareto_missing"):
        D = _load(tag)
        if D is None:
            continue
        vgs = D["vgs"]
        for i, k in enumerate(_key(D["X"])):
            pool[k] = {kk: D[kk][i] for kk in KEYS}
    rows = [pool[k] for k in _key(target)]
    out = {kk: np.array([r[kk] for r in rows]) for kk in KEYS}
    old = os.path.join(C.RESULTS_DIR, "devsim_pareto.npz")
    raw = os.path.join(C.RESULTS_DIR, "devsim_pareto_raw.npz")
    if os.path.exists(old) and not os.path.exists(raw):
        os.replace(old, raw)          # keep the original campaign output
    np.savez(old, X=target, vgs=vgs, **out)
    print(f"merged {len(target)} designs; converged {int(out['ok'].sum())}")


if __name__ == "__main__":
    {"missing": missing, "merge": merge}[sys.argv[1]]()
