"""Add q-EHVI with a gradient-based acquisition optimiser (multi-start L-BFGS-B) to the
30-seed digital-twin study, with the same seeds, initial designs and budget.

    py -3.14 -m hera_twin.extra_study --workers 18

The new runs are merged into results/study_results.pkl under "qEHVI-LBFGS" (a backup
of the previous file is kept) and results/study_summary.json is recomputed for all
methods (final-HV statistics, Wilcoxon tests against HERA, CPU time per run).
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
from hera_twin import baselines as B
from hera_twin import study as S

NEW = "qEHVI-LBFGS"
ORDER = ["HERA", "qEHVI-LBFGS", "qEHVI", "qParEGO", "NSGA-II", "Random"]
PKL = os.path.join(C.RESULTS_DIR, "study_results.pkl")
SUMMARY = os.path.join(C.RESULTS_DIR, "study_summary.json")


def run_one(args):
    seed, budget = args
    t0 = time.process_time()
    res = B.qehvi_lbfgs(budget, seed, Xinit=B.initial_design(seed))
    out = S._lite(res)
    out["cpu_s"] = time.process_time() - t0
    return seed, out


def summarize(data):
    S.METHOD_ORDER = ORDER                   # aggregate() iterates over METHOD_ORDER
    agg = S.aggregate(data["results"], data["budget"])
    summ = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    cpu = {m: [data["results"][s][m]["cpu_s"] for s in data["seeds"]] for m in ORDER}
    summ.update(cpu_time_per_run={m: dict(mean_s=float(np.mean(v)), sd_s=float(np.std(v, ddof=1)))
                                  for m, v in cpu.items()},
                stats=agg["stats"], significance=agg["sig"], method_order=ORDER)
    json.dump(summ, open(SUMMARY, "w"), indent=2)
    np.savez(os.path.join(C.RESULTS_DIR, "study_curves.npz"),
             **{f"curve_{m}": agg["curves"][m] for m in ORDER},
             **{f"final_{m}": agg["final"][m] for m in ORDER})
    for m in ORDER:
        s = agg["stats"][m]
        g = agg["sig"].get(m)
        extra = f"  dHV={-g['mean_improvement']:+.4f} p={g['p_value']:.2e} W/L={g['wins']}/{g['losses']}" if g else ""
        print(f"{m:12s} {s['mean']:.4f} +- {s['sd']:.4f}  cpu={np.mean(cpu[m]):7.1f}s{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    data = pickle.load(open(PKL, "rb"))
    if not a.summary_only:
        todo = [s for s in data["seeds"] if NEW not in data["results"][s]]
        print(f"{len(todo)} seeds to run", flush=True)
        if todo and not os.path.exists(PKL + ".before_lbfgs"):
            shutil.copy(PKL, PKL + ".before_lbfgs")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_one, (s, data["budget"])) for s in todo]
            for fu in as_completed(futs):
                s, out = fu.result()
                data["results"][s][NEW] = out
                with open(PKL, "wb") as f:
                    pickle.dump(data, f)
                print(f"[{time.time() - t0:6.0f}s] seed {s}: HV={out['hv_history'][-1]:.4f} "
                      f"cpu={out['cpu_s']:.0f}s", flush=True)
    summarize(data)


if __name__ == "__main__":
    main()
