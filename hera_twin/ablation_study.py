"""Component ablation of HERA on the FinFET twin.

Isolates the contribution of each architectural addition by building HERA up as a
cumulative ladder from the pool-based q-EHVI baseline, and separately contrasts a
flat vs a hierarchical (tier-factored) RL policy.  All variants share the same
feasible initial design per seed as the main study (baselines.initial_design), so
differences are attributable to the toggled component alone.

Ladder (each row adds one component):
  q-EHVI (pool)            pool-based constrained q-EHVI            [baselines.qehvi]
  + NSGA-II acq. optimiser HERA, use_rl=False, use_gate=False
  + RL warm-start (hier.)  HERA, use_gate=False
  + cost-aware gate = HERA HERA, full
Side contrast:
  RL warm-start (flat)     HERA, use_hierarchy=False, use_gate=False

Reported: HV after 32 evals (early / sample-efficiency regime), HV after the full
budget, and the area under the HV-vs-evaluation curve (AUC).

Run:  python -m hera_twin.ablation_study --seeds 15 --budget 64 --workers 12
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from hera_twin import config as C
from hera_twin import baselines as B
from hera_twin.hera import HERA

VARIANTS = ["q-EHVI (pool)", "+ NSGA-II acq. opt.", "+ RL warm-start (hier.)",
            "Full HERA (+ gate)", "RL warm-start (flat)"]


def run_one(task):
    seed, budget = task
    Xinit = B.initial_design(seed)
    hv = {}
    hv["q-EHVI (pool)"] = np.asarray(B.qehvi(budget, seed, Xinit=Xinit).hv_history, float)
    hv["+ NSGA-II acq. opt."] = HERA(budget=budget, seed=seed, init_X=Xinit,
                                     use_rl=False, use_gate=False).run().hv_history
    hv["+ RL warm-start (hier.)"] = HERA(budget=budget, seed=seed, init_X=Xinit,
                                         use_gate=False).run().hv_history
    hv["Full HERA (+ gate)"] = HERA(budget=budget, seed=seed, init_X=Xinit).run().hv_history
    hv["RL warm-start (flat)"] = HERA(budget=budget, seed=seed, init_X=Xinit,
                                      use_hierarchy=False, use_gate=False).run().hv_history
    return seed, {k: np.asarray(v, float) for k, v in hv.items()}


def _pad(h, n):
    h = np.asarray(h, float)
    return h[:n] if len(h) >= n else np.concatenate([h, np.full(n - len(h), h[-1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--early", type=int, default=32)
    args = ap.parse_args()

    tasks = [(s, args.budget) for s in range(args.seeds)]
    print(f"Ablation: {args.seeds} seeds x {len(VARIANTS)} variants, budget={args.budget}",
          flush=True)
    per_seed = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (seed, hv) in enumerate(ex.map(run_one, tasks), 1):
            per_seed[seed] = hv
            print(f"  [{i}/{len(tasks)}] seed {seed} "
                  f"full={hv['Full HERA (+ gate)'][-1]:.4f}", flush=True)
    print(f"Finished in {(time.time()-t0)/60:.1f} min", flush=True)

    seeds = sorted(per_seed.keys())
    B_ = args.budget
    e = args.early
    stats, curves = {}, {}
    for v in VARIANTS:
        H = np.array([_pad(per_seed[s][v], B_) for s in seeds])       # (nseed, budget)
        curves[v] = H
        hv_early = H[:, e - 1]
        hv_final = H[:, -1]
        auc = H.mean(axis=1)
        stats[v] = dict(
            hv_early_mean=float(hv_early.mean()), hv_early_sd=float(hv_early.std(ddof=1)),
            hv_final_mean=float(hv_final.mean()), hv_final_sd=float(hv_final.std(ddof=1)),
            auc_mean=float(auc.mean()), auc_sd=float(auc.std(ddof=1)))

    # paired Wilcoxon: Full HERA vs each other variant (final HV)
    from scipy.stats import wilcoxon
    full = np.array([per_seed[s]["Full HERA (+ gate)"][-1] for s in seeds])
    sig = {}
    for v in VARIANTS:
        if v == "Full HERA (+ gate)":
            continue
        other = np.array([_pad(per_seed[s][v], B_)[-1] for s in seeds])
        try:
            _, p = wilcoxon(full, other)
        except ValueError:
            p = 1.0
        sig[v] = dict(p_value=float(p), hv_gain=float((full - other).mean()),
                      wins=int(np.sum(full > other)), losses=int(np.sum(full < other)))

    np.savez(os.path.join(C.RESULTS_DIR, "ablation_curves.npz"),
             **{v.replace(" ", "_"): curves[v] for v in VARIANTS})
    out = dict(n_seeds=args.seeds, budget=B_, early=e, stats=stats, sig=sig)
    with open(os.path.join(C.RESULTS_DIR, "ablation_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== Ablation (HV@{e} | HV@{B_} | AUC), mean+-sd over {len(seeds)} seeds ===")
    for v in VARIANTS:
        st = stats[v]
        tag = ""
        if v in sig:
            tag = f"   dHV_final vs full={-sig[v]['hv_gain']:+.4f} p={sig[v]['p_value']:.2e}"
        print(f"  {v:26s} {st['hv_early_mean']:.4f}  {st['hv_final_mean']:.4f}  "
              f"{st['auc_mean']:.4f}{tag}")


if __name__ == "__main__":
    main()
