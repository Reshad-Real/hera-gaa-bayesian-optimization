"""Standard-benchmark validation study: HERA vs general-purpose baselines.

Runs the identical HERA algorithm and four baselines on the DTLZ1, DTLZ2 and
C2-DTLZ2 problems (d=6, M=3) over many seeds, from a shared feasible initial
design per seed.  Records the hypervolume-vs-evaluation history, the final
normalised hypervolume (HV, higher better) and the modified inverted generational
distance (IGD+, lower better) to the analytic Pareto front, with paired Wilcoxon
signed-rank tests of HERA against each baseline.

Run:  python -m hera_twin.benchmark_study --seeds 12 --budget 64 --workers 10
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
from concurrent.futures import ProcessPoolExecutor

from hera_twin import config as C
from hera_twin import benchmarks as BM

METHODS = ["HERA", "qEHVI", "qParEGO", "NSGA-II", "Random"]
_RUN = {"HERA": BM.run_hera, "qEHVI": BM.run_qehvi, "qParEGO": BM.run_qparego,
        "NSGA-II": BM.run_nsga2, "Random": BM.run_random}


def run_one(task):
    p_idx, seed, budget = task
    problem = BM.make_problems()[p_idx]
    BM._set_frame(problem)
    Xinit = BM.sobol_feasible(problem, C.HP["n0"], np.random.default_rng(seed))
    out = {}
    for m in METHODS:
        r = _RUN[m](problem, budget, seed, Xinit=Xinit)
        out[m] = dict(final_hv=r["final_hv"], final_igd=r["final_igd"],
                      hv_history=np.asarray(r["hv_history"], float))
    return p_idx, seed, out


def _pad(h, n):
    h = np.asarray(h, float)
    return h[:n] if len(h) >= n else np.concatenate([h, np.full(n - len(h), h[-1])])


def aggregate(problem_name, per_seed, budget):
    from scipy.stats import wilcoxon
    seeds = sorted(per_seed.keys())
    stats, curves = {}, {}
    for m in METHODS:
        hv = np.array([per_seed[s][m]["final_hv"] for s in seeds])
        igd = np.array([per_seed[s][m]["final_igd"] for s in seeds])
        curves[m] = np.array([_pad(per_seed[s][m]["hv_history"], budget) for s in seeds])
        n = len(seeds)
        stats[m] = dict(
            hv_mean=float(hv.mean()), hv_sd=float(hv.std(ddof=1)),
            igd_mean=float(np.nanmean(igd)), igd_sd=float(np.nanstd(igd, ddof=1)))
    sig = {}
    hvH = np.array([per_seed[s]["HERA"]["final_hv"] for s in seeds])
    for m in METHODS:
        if m == "HERA":
            continue
        hvm = np.array([per_seed[s][m]["final_hv"] for s in seeds])
        try:
            _, p = wilcoxon(hvH, hvm)
        except ValueError:
            p = 1.0
        wins = int(np.sum(hvH > hvm))
        sig[m] = dict(p_value=float(p), wins=wins, losses=int(np.sum(hvH < hvm)),
                      hv_gain=float((hvH - hvm).mean()))
    return dict(problem=problem_name, seeds=seeds, stats=stats, sig=sig, curves=curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    problems = BM.make_problems()
    tasks = [(pi, s, args.budget) for pi in range(len(problems)) for s in range(args.seeds)]
    print(f"Benchmark study: {len(problems)} problems x {args.seeds} seeds x "
          f"{len(METHODS)} methods, budget={args.budget}, workers={args.workers}", flush=True)

    results = {pi: {} for pi in range(len(problems))}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (pi, seed, out) in enumerate(ex.map(run_one, tasks), 1):
            results[pi][seed] = out
            print(f"  [{i}/{len(tasks)}] {problems[pi].name} seed {seed}  "
                  f"HERA HV={out['HERA']['final_hv']:.4f} IGD+={out['HERA']['final_igd']:.4f}",
                  flush=True)
    print(f"Finished in {(time.time()-t0)/60:.1f} min", flush=True)

    aggs = {problems[pi].name: aggregate(problems[pi].name, results[pi], args.budget)
            for pi in range(len(problems))}

    with open(os.path.join(C.RESULTS_DIR, "benchmark_results.pkl"), "wb") as f:
        pickle.dump(dict(results=results, budget=args.budget, seeds=args.seeds,
                         problems=[p.name for p in problems]), f)
    # curves for the convergence figure
    np.savez(os.path.join(C.RESULTS_DIR, "benchmark_curves.npz"),
             **{f"{name}__{m}": aggs[name]["curves"][m]
                for name in aggs for m in METHODS})
    summary = {name: dict(stats=aggs[name]["stats"], sig=aggs[name]["sig"],
                          n_seeds=args.seeds, budget=args.budget) for name in aggs}
    with open(os.path.join(C.RESULTS_DIR, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Benchmark summary (final HV mean+-sd | IGD+ mean+-sd) ===")
    for name in aggs:
        print(f"\n{name}:")
        for m in METHODS:
            st = aggs[name]["stats"][m]
            tag = ""
            if m != "HERA":
                s = aggs[name]["sig"][m]
                tag = f"   p={s['p_value']:.2e}  W/L={s['wins']}/{s['losses']}"
            print(f"  {m:9s} HV={st['hv_mean']:.4f}+-{st['hv_sd']:.4f}  "
                  f"IGD+={st['igd_mean']:.4f}+-{st['igd_sd']:.4f}{tag}")


if __name__ == "__main__":
    main()
