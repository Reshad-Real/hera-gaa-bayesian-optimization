"""Add q-EHVI with a multi-start L-BFGS-B acquisition optimiser to the DTLZ benchmark
study (same problems, seeds, initial designs and budget), then recompute the summary.

    py -3.14 -m hera_twin.benchmark_extra --workers 18
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import shutil
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from hera_twin import config as C
from hera_twin import benchmarks as BM
from hera_twin import benchmark_study as BS

NEW = "qEHVI-LBFGS"
PKL = os.path.join(C.RESULTS_DIR, "benchmark_results.pkl")


def run_one(task):
    p_idx, seed, budget = task
    problem = BM.make_problems()[p_idx]
    BM._set_frame(problem)
    Xinit = BM.sobol_feasible(problem, C.HP["n0"], np.random.default_rng(seed))
    r = BM.run_qehvi_lbfgs(problem, budget, seed, Xinit=Xinit)
    return p_idx, seed, dict(final_hv=r["final_hv"], final_igd=r["final_igd"],
                             hv_history=np.asarray(r["hv_history"], float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()
    data = pickle.load(open(PKL, "rb"))
    budget = data["budget"]
    tasks = [(pi, s, budget) for pi in data["results"] for s in data["results"][pi]
             if NEW not in data["results"][pi][s]]
    print(f"{len(tasks)} runs", flush=True)
    if tasks and not os.path.exists(PKL + ".before_lbfgs"):
        shutil.copy(PKL, PKL + ".before_lbfgs")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fu in as_completed(futs):
            pi, s, out = fu.result()
            data["results"][pi][s][NEW] = out
            with open(PKL, "wb") as f:
                pickle.dump(data, f)
            print(f"[{time.time() - t0:6.0f}s] {data['problems'][pi]} seed {s}: HV={out['final_hv']:.4f}",
                  flush=True)
    BS.METHODS = ["HERA", NEW, "qEHVI", "qParEGO", "NSGA-II", "Random"]
    aggs = {data["problems"][pi]: BS.aggregate(data["problems"][pi], data["results"][pi], budget)
            for pi in data["results"]}
    np.savez(os.path.join(C.RESULTS_DIR, "benchmark_curves.npz"),
             **{f"{name}__{m}": aggs[name]["curves"][m] for name in aggs for m in BS.METHODS})
    summary = {name: dict(stats=aggs[name]["stats"], sig=aggs[name]["sig"],
                          n_seeds=len(aggs[name]["seeds"]), budget=budget) for name in aggs}
    json.dump(summary, open(os.path.join(C.RESULTS_DIR, "benchmark_summary.json"), "w"), indent=2)
    for name in aggs:
        print(name)
        for m in BS.METHODS:
            st = aggs[name]["stats"][m]
            sg = aggs[name]["sig"].get(m)
            tag = f"  p={sg['p_value']:.2e} W/L={sg['wins']}/{sg['losses']}" if sg else ""
            print(f"  {m:12s} HV={st['hv_mean']:.4f}+-{st['hv_sd']:.4f} IGD+={st['igd_mean']:.4f}{tag}")


if __name__ == "__main__":
    main()
