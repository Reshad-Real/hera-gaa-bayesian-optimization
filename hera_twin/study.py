"""Multi-seed benchmark study: HERA vs literature baselines.

Runs all methods from a shared feasible initial design for >=30 seeds, records
the hypervolume-vs-evaluation history, final hypervolume, and final Pareto sets,
then computes mean +- s.d., 95% confidence intervals, and paired significance
tests (Wilcoxon signed-rank, HERA vs each baseline).  Parallelised across seeds.

Run:  python -m hera_twin.study --seeds 30 --budget 70 --workers 16
"""
from __future__ import annotations

import os
# limit per-worker math threads to avoid oversubscription under multiprocessing
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
from hera_twin import baselines as B
from hera_twin.hera import HERA

METHOD_ORDER = ["HERA", "qEHVI", "qParEGO", "NSGA-II", "Random"]


def _lite(res):
    return dict(
        name=res.name,
        hv_history=np.asarray(res.hv_history, dtype=float),
        X=np.asarray(res.X, dtype=float),
        Y=np.asarray(res.Y, dtype=float),
        feas=np.asarray(res.feas, dtype=bool),
        pareto_X=np.asarray(res.pareto_X, dtype=float),
        pareto_Y=np.asarray(res.pareto_Y, dtype=float),
        init_X=np.asarray(res.init_X, dtype=float),
        init_Y=np.asarray(res.init_Y, dtype=float),
    )


def run_one_seed(args):
    seed, budget = args
    Xinit = B.initial_design(seed)
    out = {}
    runners = [
        ("HERA", lambda: HERA(budget=budget, seed=seed, init_X=Xinit).run()),
        ("qEHVI", lambda: B.qehvi(budget, seed, Xinit=Xinit)),
        ("qParEGO", lambda: B.qparego(budget, seed, Xinit=Xinit)),
        ("NSGA-II", lambda: B.nsga2_baseline(budget, seed, Xinit=Xinit)),
        ("Random", lambda: B.random_search(budget, seed, Xinit=Xinit)),
    ]
    for name, fn in runners:
        t0 = time.process_time()
        res = fn()
        out[name] = _lite(res)
        out[name]["cpu_s"] = time.process_time() - t0      # optimiser + oracle CPU time
    return seed, out


def _pad(hist, n):
    hist = np.asarray(hist, dtype=float)
    if len(hist) >= n:
        return hist[:n]
    return np.concatenate([hist, np.full(n - len(hist), hist[-1])])


def aggregate(results, budget):
    """Compute per-method HV curves and final-HV statistics."""
    from scipy.stats import wilcoxon
    seeds = sorted(results.keys())
    curves = {m: np.array([_pad(results[s][m]["hv_history"], budget) for s in seeds])
              for m in METHOD_ORDER}
    final = {m: np.array([results[s][m]["hv_history"][-1] for s in seeds])
             for m in METHOD_ORDER}

    stats = {}
    n = len(seeds)
    for m in METHOD_ORDER:
        v = final[m]
        ci = 1.96 * v.std(ddof=1) / np.sqrt(n)
        stats[m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                        ci95=float(ci), median=float(np.median(v)),
                        min=float(v.min()), max=float(v.max()))
    # paired significance: HERA vs each baseline
    sig = {}
    for m in METHOD_ORDER:
        if m == "HERA":
            continue
        diff = final["HERA"] - final[m]
        try:
            stat, p = wilcoxon(final["HERA"], final[m])
        except ValueError:
            stat, p = np.nan, 1.0
        # rank-biserial effect size
        wins = int(np.sum(diff > 0)); losses = int(np.sum(diff < 0))
        sig[m] = dict(p_value=float(p), stat=float(stat),
                      mean_improvement=float(diff.mean()),
                      median_improvement=float(np.median(diff)),
                      wins=wins, losses=losses, ties=int(n - wins - losses))
    return dict(seeds=seeds, curves=curves, final=final, stats=stats, sig=sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=int, default=70)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    tasks = [(s, args.budget) for s in seeds]
    t0 = time.time()
    results = {}
    print(f"Running {len(seeds)} seeds x {len(METHOD_ORDER)} methods, "
          f"budget={args.budget}, workers={args.workers} ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (seed, out) in enumerate(ex.map(run_one_seed, tasks), 1):
            results[seed] = out
            print(f"  [{i}/{len(seeds)}] seed {seed} done  "
                  f"(HERA finalHV={out['HERA']['hv_history'][-1]:.4f})", flush=True)
    dt = time.time() - t0
    print(f"Study finished in {dt/60:.1f} min", flush=True)

    agg = aggregate(results, args.budget)

    # save full + summary
    with open(os.path.join(C.RESULTS_DIR, "study_results.pkl"), "wb") as f:
        pickle.dump(dict(results=results, budget=args.budget, seeds=seeds), f)
    np.savez(os.path.join(C.RESULTS_DIR, "study_curves.npz"),
             **{f"curve_{m}": agg["curves"][m] for m in METHOD_ORDER},
             **{f"final_{m}": agg["final"][m] for m in METHOD_ORDER})
    cpu = {m: [results[s][m]["cpu_s"] for s in seeds] for m in METHOD_ORDER}
    timing = {m: dict(mean_s=float(np.mean(v)), sd_s=float(np.std(v, ddof=1)))
              for m, v in cpu.items()}
    summary = dict(budget=args.budget, n_seeds=len(seeds), wall_clock_min=dt / 60,
                   workers=args.workers, cpu_time_per_run=timing,
                   stats=agg["stats"], significance=agg["sig"])
    with open(os.path.join(C.RESULTS_DIR, "study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final hypervolume (mean +- s.d.) ===", flush=True)
    for m in METHOD_ORDER:
        s = agg["stats"][m]
        print(f"  {m:9s}: {s['mean']:.4f} +- {s['sd']:.4f}   "
              f"[95% CI +-{s['ci95']:.4f}]")
    print("\n=== HERA vs baseline (Wilcoxon signed-rank) ===", flush=True)
    for m, s in agg["sig"].items():
        print(f"  HERA vs {m:9s}: dHV={s['mean_improvement']:+.4f}  "
              f"p={s['p_value']:.2e}  W/L/T={s['wins']}/{s['losses']}/{s['ties']}")


if __name__ == "__main__":
    main()
