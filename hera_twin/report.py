"""Orchestrate the full HERA digital-twin deliverable.

Loads the multi-seed benchmark, runs a higher-budget showcase HERA optimisation
to obtain a rich final Pareto set, selects a balanced ("knee") optimised FinFET,
reconstructs its 3-D / 2-D models (validated), recreates it in DEVSIM, and
emits every publication figure and table.
"""
from __future__ import annotations

import os
import json
import pickle
import numpy as np

from . import config as C
from . import finfet_oracle as O
from . import hypervolume as H
from . import baselines as B
from .hera import HERA
from . import tables as T
from . import plots as P
from . import finfet_geometry as Gm

SHOWCASE_BUDGET = 96


def load_study():
    with open(os.path.join(C.RESULTS_DIR, "study_results.pkl"), "rb") as f:
        data = pickle.load(f)
    with open(os.path.join(C.RESULTS_DIR, "study_summary.json")) as f:
        summary = json.load(f)
    return data, summary


def representative_seed(data, summary):
    """Seed whose HERA final HV is closest to the median (a typical run)."""
    seeds = data["seeds"]
    finals = {s: data["results"][s]["HERA"]["hv_history"][-1] for s in seeds}
    med = np.median(list(finals.values()))
    return min(seeds, key=lambda s: abs(finals[s] - med))


def pick_knee(pareto_X, pareto_Y):
    """Pick a representative balanced, manufacturable Pareto-optimal FinFET.

    The optimiser drives the whole front to ultra-thin bodies (best
    electrostatics).  For the device that is reconstructed and re-simulated we
    take the widest-body designs on the front (most manufacturable / robustly
    simulatable in 2-D DEVSIM) and, among those, the most balanced one
    (max-min normalised objective).  Every dimension is still <= 3 nm.
    """
    Yn = np.clip(C.normalize_y(pareto_Y), 0, None)
    score = Yn.min(axis=1)            # worst normalised objective (balance)
    wfin = pareto_X[:, 1]
    near_widest = wfin >= (wfin.max() - 0.12)
    idx = np.where(near_widest)[0]
    j = int(idx[np.argmax(score[idx])])
    return j, pareto_X[j], pareto_Y[j]


def run_showcase(seed, budget=SHOWCASE_BUDGET):
    """Run all methods for one seed at the showcase budget (for the Pareto figure)."""
    Xinit = B.initial_design(seed)
    res = {}
    res["HERA"] = HERA(budget=budget, seed=seed, init_X=Xinit).run()
    res["qEHVI"] = B.qehvi(budget, seed, Xinit=Xinit)
    res["NSGA-II"] = B.nsga2_baseline(budget, seed, Xinit=Xinit)
    res["Random"] = B.random_search(budget, seed, Xinit=Xinit)
    return res, Xinit


def generate_all(devsim_metrics_path=None):
    data, summary = load_study()
    seed = representative_seed(data, summary)
    print(f"Representative seed: {seed}")

    # ---- showcase optimisation (rich Pareto) ----
    res, Xinit = run_showcase(seed)
    hera = res["HERA"]
    print(f"Showcase HERA: {len(hera.pareto_X)} Pareto designs, "
          f"final HV={hera.hv_history[-1]:.4f}")

    # ---- choose a balanced optimised FinFET ----
    j, x_chosen, y_chosen = pick_knee(hera.pareto_X, hera.pareto_Y)
    print(f"Chosen design (knee): {np.round(x_chosen,3)}  y={np.round(y_chosen,3)}")

    # ---- tables ----
    T.table_initial_designs(hera.init_X)
    T.table_pareto_designs(hera.pareto_X)
    T.table_method_comparison(summary)
    T.table_chosen_design(x_chosen)

    # ---- geometry (validated) ----
    rep = Gm.validate_geometry(x_chosen)
    Gm.print_validation(rep)
    Gm.render_3d(x_chosen, os.path.join(C.RESULTS_DIR, "fig4_finfet_3d.png"))
    Gm.render_2d_cross_sections(x_chosen, os.path.join(C.RESULTS_DIR, "fig5_finfet_2d.png"))

    # ---- benchmark figures ----
    curves = {m: np.array([B_pad(data["results"][s][m]["hv_history"], data["budget"])
                           for s in data["seeds"]]) for m in P.METHOD_ORDER}
    final = {m: np.array([data["results"][s][m]["hv_history"][-1] for s in data["seeds"]])
             for m in P.METHOD_ORDER}
    P.plot_convergence(curves, C.HP["n0"])
    P.plot_final_hv_box(final, summary["significance"])
    method_paretos = {m: res[m].pareto_Y for m in ["HERA", "qEHVI", "NSGA-II", "Random"]}
    P.plot_pareto(method_paretos, chosen_Y=y_chosen)

    # save chosen design for the DEVSIM step
    np.save(os.path.join(C.RESULTS_DIR, "chosen_design.npy"), x_chosen)
    with open(os.path.join(C.RESULTS_DIR, "chosen_design.json"), "w") as f:
        json.dump(dict(x=x_chosen.tolist(), y=y_chosen.tolist(), seed=int(seed)), f, indent=2)
    print("Tables + figures (benchmark, geometry, Pareto) written to results/.")
    return x_chosen, y_chosen


def B_pad(hist, n):
    hist = np.asarray(hist, dtype=float)
    if len(hist) >= n:
        return hist[:n]
    return np.concatenate([hist, np.full(n - len(hist), hist[-1])])


if __name__ == "__main__":
    generate_all()
