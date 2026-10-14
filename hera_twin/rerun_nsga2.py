"""Re-run the NSGA-II baseline after fixing the re-evaluation of the shared
initial designs (they were logged a second time, so NSGA-II effectively had
only 54 new evaluations).  All other methods' results are unchanged and reused.

    python -m hera_twin.rerun_nsga2 device     # 30-seed device study
    python -m hera_twin.rerun_nsga2 dtlz       # 12-seed DTLZ benchmarks
"""
import os
import sys
import json
import time
import pickle
import shutil
import numpy as np

from . import config as C
from . import baselines as B


def device():
    from .study import _lite, aggregate, METHOD_ORDER
    p = os.path.join(C.RESULTS_DIR, "study_results.pkl")
    bak = os.path.join(C.RESULTS_DIR, "study_results_before_nsga2_fix.pkl")
    if not os.path.exists(bak):
        shutil.copy(p, bak)
    data = pickle.load(open(bak, "rb"))
    for s in data["seeds"]:
        Xinit = B.initial_design(s)
        assert np.allclose(Xinit, data["results"][s]["HERA"]["init_X"])
        t0 = time.process_time()
        r = B.nsga2_baseline(data["budget"], s, Xinit=Xinit)
        out = _lite(r)
        out["cpu_s"] = time.process_time() - t0
        assert len(out["X"]) == data["budget"]
        # the shared initial designs must not be logged a second time
        assert not np.allclose(out["X"][10:20], out["X"][:10])
        data["results"][s]["NSGA-II"] = out
    pickle.dump(data, open(p, "wb"))
    agg = aggregate(data["results"], data["budget"])
    summ_p = os.path.join(C.RESULTS_DIR, "study_summary.json")
    summ = json.load(open(summ_p))
    v = [data["results"][s]["NSGA-II"]["cpu_s"] for s in data["seeds"]]
    summ["cpu_time_per_run"]["NSGA-II"] = dict(mean_s=float(np.mean(v)), sd_s=float(np.std(v, ddof=1)))
    summ["stats"], summ["significance"] = agg["stats"], agg["sig"]
    summ["note"] = "NSGA-II re-run after fixing re-evaluation of the shared initial designs"
    json.dump(summ, open(summ_p, "w"), indent=2)
    np.savez(os.path.join(C.RESULTS_DIR, "study_curves.npz"),
             **{f"curve_{m}": agg["curves"][m] for m in METHOD_ORDER},
             **{f"final_{m}": agg["final"][m] for m in METHOD_ORDER})
    for m in METHOD_ORDER:
        st = agg["stats"][m]
        g = agg["sig"].get(m)
        print(f"{m:8s} {st['mean']:.4f} +- {st['sd']:.4f}" +
              (f"  p={g['p_value']:.2e} W/L={g['wins']}/{g['losses']}" if g else ""))


def dtlz():
    from . import benchmarks as BM
    from .benchmark_study import aggregate, METHODS
    src = os.path.join(os.path.dirname(C.RESULTS_DIR), "..", "results", "benchmark_results.pkl")
    data = pickle.load(open(os.path.abspath(src), "rb"))
    problems = BM.make_problems()
    for pi, prob in enumerate(problems):
        BM._set_frame(prob)
        for s in range(data["seeds"]):
            BM._set_frame(prob)
            Xinit = BM.sobol_feasible(prob, C.HP["n0"], np.random.default_rng(s))
            r = BM.run_nsga2(prob, data["budget"], s, Xinit=Xinit)
            data["results"][pi][s]["NSGA-II"] = dict(final_hv=r["final_hv"], final_igd=r["final_igd"],
                                                     hv_history=np.asarray(r["hv_history"], float))
    pickle.dump(data, open(os.path.join(C.RESULTS_DIR, "benchmark_results.pkl"), "wb"))
    aggs = {problems[pi].name: aggregate(problems[pi].name, data["results"][pi], data["budget"])
            for pi in range(len(problems))}
    np.savez(os.path.join(C.RESULTS_DIR, "benchmark_curves.npz"),
             **{f"{n}__{m}": aggs[n]["curves"][m] for n in aggs for m in METHODS})
    summary = {n: dict(stats=aggs[n]["stats"], sig=aggs[n]["sig"], n_seeds=data["seeds"],
                       budget=data["budget"]) for n in aggs}
    json.dump(summary, open(os.path.join(C.RESULTS_DIR, "benchmark_summary.json"), "w"), indent=2)
    for n in aggs:
        print(n)
        for m in METHODS:
            st = aggs[n]["stats"][m]
            g = aggs[n]["sig"].get(m)
            print(f"  {m:8s} HV {st['hv_mean']:.4f}+-{st['hv_sd']:.4f} IGD+ {st['igd_mean']:.4f}+-{st['igd_sd']:.4f}"
                  + (f"  p={g['p_value']:.1e} W/L={g['wins']}/{g['losses']}" if g else ""))


if __name__ == "__main__":
    {"device": device, "dtlz": dtlz}[sys.argv[1]]()
