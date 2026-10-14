"""Add "q-EHVI + L-BFGS-B acquisition optimizer" to the component ablation.

    py -3.14 -m hera_twin.ablation_extra

Uses the qEHVI-LBFGS runs of the 30-seed study for seeds 0-14, which share the
initial designs, budget and seeds of the ablation ladder (ablation_study.py), and
computes the same statistics (HV after 32 and 64 evaluations, AUC, paired Wilcoxon
test against the full HERA).
"""
from __future__ import annotations

import os
import json
import pickle
import numpy as np
from scipy.stats import wilcoxon

from . import config as C

KEY = "q-EHVI (L-BFGS-B)"
CURVES = os.path.join(C.RESULTS_DIR, "ablation_curves.npz")
SUMMARY = os.path.join(C.RESULTS_DIR, "ablation_summary.json")


def _pad(h, n):
    h = np.asarray(h, float)
    return h[:n] if len(h) >= n else np.concatenate([h, np.full(n - len(h), h[-1])])


def main():
    summ = json.load(open(SUMMARY))
    curves = dict(np.load(CURVES))
    n, B, e = summ["n_seeds"], summ["budget"], summ["early"]
    data = pickle.load(open(os.path.join(C.RESULTS_DIR, "study_results.pkl"), "rb"))
    Hh = np.array([_pad(data["results"][s]["qEHVI-LBFGS"]["hv_history"], B) for s in range(n)])
    curves[KEY.replace(" ", "_")] = Hh
    np.savez(CURVES, **curves)
    full = curves["Full_HERA_(+_gate)"][:, -1]
    summ["stats"][KEY] = dict(
        hv_early_mean=float(Hh[:, e - 1].mean()), hv_early_sd=float(Hh[:, e - 1].std(ddof=1)),
        hv_final_mean=float(Hh[:, -1].mean()), hv_final_sd=float(Hh[:, -1].std(ddof=1)),
        auc_mean=float(Hh.mean(axis=1).mean()), auc_sd=float(Hh.mean(axis=1).std(ddof=1)))
    summ["sig"][KEY] = dict(p_value=float(wilcoxon(full, Hh[:, -1]).pvalue),
                            hv_gain=float((full - Hh[:, -1]).mean()),
                            wins=int((full > Hh[:, -1]).sum()), losses=int((full < Hh[:, -1]).sum()))
    # consistency check: the ablation's full HERA equals the study's HERA for these seeds
    hs = np.array([data["results"][s]["HERA"]["hv_history"][-1] for s in range(n)])
    summ["check_full_equals_study_HERA"] = bool(np.allclose(hs, full))
    json.dump(summ, open(SUMMARY, "w"), indent=2)
    st, g = summ["stats"][KEY], summ["sig"][KEY]
    print(f"{KEY}: HV@{e}={st['hv_early_mean']:.4f}+-{st['hv_early_sd']:.4f} "
          f"HV@{B}={st['hv_final_mean']:.4f}+-{st['hv_final_sd']:.4f} AUC={st['auc_mean']:.4f} "
          f"dHV vs full={-g['hv_gain']:+.4f} p={g['p_value']:.2e} W/L={g['wins']}/{g['losses']}; "
          f"full==study HERA: {summ['check_full_equals_study_HERA']}")


if __name__ == "__main__":
    main()
