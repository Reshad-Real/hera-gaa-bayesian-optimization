"""TCAD-in-the-loop study: the optimisers query DEVSIM TCAD directly.

    py -3.14 -m hera_twin.tcad_loop init   --seeds 10 --workers 18
    py -3.14 -m hera_twin.tcad_loop run    --seeds 10 --budget 64 --workers 18
    py -3.14 -m hera_twin.tcad_loop status

The expensive oracle is the 2-D DEVSIM drift-diffusion model of devsim_sim (the
reference the digital twin was calibrated on); objectives and constraints are
computed from the TCAD figures of merit exactly as in the post-hoc re-scoring
(caee_report.tcad_objectives). All methods of a seed start from the same n0 = 10
designs (baselines.initial_design), which are simulated once ("init") and shared.

Methods
  Random            Sobol search on TCAD
  qEHVI, HERA       plain GP surrogates of the TCAD response
  HERA-D            HERA whose GPs are also trained on the 96 TCAD designs used to
                    calibrate the twin (same information as the twin, as raw data;
                    gp_surrogate.EXTRA)
  qEHVI-TI, HERA-TI twin-informed surrogates: the calibrated digital twin is the
                    prior mean and the GPs model the TCAD - twin residual
                    (gp_surrogate.PRIOR)
  HERA-TID          both: twin prior mean, and residual GPs trained also on the
                    96 calibration designs
  qEHVI-LBFGS(-D)   q-EHVI with the multi-start L-BFGS-B acquisition optimiser,
                    plain GPs (or also trained on the 96 calibration designs)

TCAD results are cached by (Lg, W, EOT, Lsp, log10 Nch): the 2-D simulation is per
micrometre of gated perimeter, so it does not depend on the body height H, which
enters the objectives analytically through W_eff = 2(H + W).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import config as C

CACHE = os.path.join(C.RESULTS_DIR, "tcad_loop_cache.pkl")
OUT = os.path.join(C.RESULTS_DIR, "tcad_loop_results.pkl")
KEYS = ["Ion", "Ioff", "SS", "DIBL", "dQg_lin", "Vth_sat"]
METHODS = ["HERA-TI", "qEHVI-TI", "HERA", "qEHVI", "Random"]


def tcad_key(x):
    x = np.asarray(x, float)
    return tuple(np.round([x[0], x[1], x[3], x[4], np.log10(x[5])], 6).tolist())


SIM_TIMEOUT = 180.0     # s; the slowest of 3,160 earlier simulations took 23 s
_EXEC = None


def _simulate_now(x):
    from .devsim_campaign import _worker
    r = _worker(np.asarray(x, float))
    rec = {k: float(r.get(k, np.nan)) for k in KEYS}
    rec["ok"] = bool(r.get("ok", False)) and all(np.isfinite(rec[k]) for k in KEYS)
    rec["time_s"] = float(r.get("time_s", np.nan))
    return rec


def simulate(x):
    """One DEVSIM simulation -> dict of the figures of merit (+ ok, time_s).

    The simulation runs in a child process with a time limit, so that a solve
    that hangs inside DEVSIM is recorded as a failed (infeasible) evaluation
    instead of blocking the optimisation run.
    """
    global _EXEC
    from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
    if _EXEC is None:
        _EXEC = ProcessPoolExecutor(max_workers=1)
    fut = _EXEC.submit(_simulate_now, np.asarray(x, float))
    try:
        return fut.result(timeout=SIM_TIMEOUT)
    except FutTimeout:
        for p in list(_EXEC._processes.values()):
            p.terminate()
        _EXEC.shutdown(wait=False, cancel_futures=True)
        _EXEC = None
        rec = {k: float("nan") for k in KEYS}
        rec.update(ok=False, time_s=SIM_TIMEOUT, timeout=True)
        return rec


class TCADOracle:
    """objectives / constraints / is_feasible backed by DEVSIM, with a design cache."""

    def __init__(self, cache):
        from . import finfet_oracle as O
        self.cache = dict(cache)
        self.new = {}
        self.n_sim = 0
        self.n_fail = 0
        self.sim_time = 0.0
        self.twin_obj, self.twin_con = O.objectives, O.constraints

    def _metrics(self, X):
        rows = []
        for x in X:
            k = tcad_key(x)
            if k not in self.cache:
                rec = simulate(x)
                self.n_sim += 1
                self.sim_time += rec["time_s"]
                self.n_fail += int(not rec["ok"])
                self.cache[k] = rec
                self.new[k] = rec
            rows.append(self.cache[k])
        return rows

    def evaluate(self, X):
        from .caee_report import tcad_objectives
        X = np.atleast_2d(np.asarray(X, float))
        rows = self._metrics(X)
        ok = np.array([r["ok"] for r in rows])
        M = {k: np.array([r[k] if r["ok"] else 1.0 for r in rows]) for k in KEYS}
        Y, c = tcad_objectives(X, M)
        if (~ok).any():                    # failed simulation: twin objectives, infeasible
            Y[~ok] = self.twin_obj(X[~ok])
            c[~ok, 3:] = 1.0
        return Y, c

    def objectives(self, X):
        return self.evaluate(X)[0]

    def constraints(self, X):
        return self.evaluate(X)[1]

    def is_feasible(self, X, tol=0.0):
        return np.all(self.constraints(X) <= tol, axis=1)


def calibration_data():
    """The 96 TCAD training designs of the twin calibration, as extra GP data (HERA-D)."""
    from .caee_report import tcad_objectives
    D = np.load(os.path.join(C.RESULTS_DIR, "devsim_calib.npz"))
    meta = json.load(open(os.path.join(C.RESULTS_DIR, "twin_params.json")))
    ok = D["ok"].astype(bool)
    tr = np.array(meta["train_idx"])
    X = D["X"][ok][tr]
    M = {k: D[k][ok][tr] for k in ["Ion", "Ioff", "SS", "DIBL", "dQg_lin"]}
    Y, c = tcad_objectives(X, M)
    return C.normalize_x(X), Y, c


def make_twin_prior(twin_obj, twin_con):
    """Prior mean for the twin-informed surrogate (normalised objectives, constraints)."""
    def prior(Xn):
        X = C.denormalize_x(np.atleast_2d(Xn))
        return C.normalize_y(twin_obj(X)), twin_con(X)
    return prior


def _lite(res):
    return dict(name=res.name, hv_history=np.asarray(res.hv_history, float),
                X=np.asarray(res.X, float), Y=np.asarray(res.Y, float),
                feas=np.asarray(res.feas, bool), pareto_X=np.asarray(res.pareto_X, float),
                pareto_Y=np.asarray(res.pareto_Y, float), init_X=np.asarray(res.init_X, float))


def run_one(args):
    seed, method, budget, cache = args
    from . import finfet_oracle as O
    from . import gp_surrogate as G
    from . import baselines as B
    from .hera import HERA
    orc = TCADOracle(cache)
    prior = make_twin_prior(orc.twin_obj, orc.twin_con)
    O.objectives, O.constraints, O.is_feasible = orc.objectives, orc.constraints, orc.is_feasible
    G.PRIOR = prior if method in ("HERA-TI", "qEHVI-TI", "HERA-TID") else None
    G.EXTRA = calibration_data() if method in ("HERA-D", "HERA-TID", "qEHVI-LBFGS-D") else None
    Xinit = B.initial_design(seed)
    t0, c0 = time.time(), time.process_time()
    if method.startswith("HERA"):
        res = HERA(budget=budget, seed=seed, init_X=Xinit).run()
    elif method.startswith("qEHVI-LBFGS"):
        res = B.qehvi_lbfgs(budget, seed, Xinit=Xinit)
    elif method.startswith("qEHVI"):
        res = B.qehvi(budget, seed, Xinit=Xinit)
    elif method == "Random":
        res = B.random_search(budget, seed, Xinit=Xinit)
    else:
        raise ValueError(method)
    res.name = method
    return dict(seed=seed, method=method, result=_lite(res), new_cache=orc.new,
                n_sim=orc.n_sim, sim_time=orc.sim_time, n_fail=orc.n_fail,
                wall_s=time.time() - t0, cpu_s=time.process_time() - c0 - orc.sim_time)


def _load(path, default):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return default


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def study_seeds(n):
    with open(os.path.join(C.RESULTS_DIR, "study_results.pkl"), "rb") as f:
        return list(pickle.load(f)["seeds"][:n])


def cmd_init(n_seeds, workers):
    from . import baselines as B
    cache = _load(CACHE, {})
    todo = {}
    for s in study_seeds(n_seeds):
        for x in B.initial_design(s):
            k = tcad_key(x)
            if k not in cache:
                todo[k] = x
    print(f"{len(todo)} initial designs to simulate")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(simulate, x): k for k, x in todo.items()}
        for i, fu in enumerate(as_completed(futs), 1):
            cache[futs[fu]] = fu.result()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")
    _save(CACHE, cache)
    print("cache size", len(cache), "failures", sum(not r["ok"] for r in cache.values()))


def cmd_run(n_seeds, budget, workers, methods):
    cache = _load(CACHE, {})
    out = _load(OUT, dict(budget=budget, runs={}))
    tasks = [(s, m) for m in methods for s in study_seeds(n_seeds) if (s, m) not in out["runs"]]
    print(f"{len(tasks)} runs to do; cache {len(cache)}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, (s, m, budget, cache)): (s, m) for s, m in tasks}
        for fu in as_completed(futs):
            s, m = futs[fu]
            try:
                r = fu.result()
            except Exception as e:                       # keep going; report at the end
                print(f"FAILED seed {s} {m}: {e!r}", flush=True)
                continue
            cache.update(r.pop("new_cache"))
            out["runs"][(s, m)] = r
            _save(OUT, out)
            _save(CACHE, cache)
            hv = r["result"]["hv_history"][-1]
            print(f"[{time.time() - t0:7.0f}s] seed {s:3d} {m:9s} HV={hv:.4f} sims={r['n_sim']} "
                  f"fail={r['n_fail']} wall={r['wall_s']:.0f}s cpu(opt)={r['cpu_s']:.0f}s", flush=True)
    print("done", len(out["runs"]), "runs")


def cmd_status():
    out = _load(OUT, dict(runs={}))
    by = {}
    for (s, m), r in out["runs"].items():
        by.setdefault(m, []).append(r["result"]["hv_history"][-1])
    for m, v in by.items():
        print(f"{m:9s} n={len(v):2d} HV={np.mean(v):.4f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["init", "run", "status"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    if a.mode == "init":
        cmd_init(a.seeds, a.workers)
    elif a.mode == "run":
        cmd_run(a.seeds, a.budget, a.workers, a.methods.split(","))
    else:
        cmd_status()


if __name__ == "__main__":
    main()
