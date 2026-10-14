"""Calibrate the compact digital twin against DEVSIM drift-diffusion TCAD.

    python -m hera_twin.calibrate_twin

Reads results/devsim_calib.npz (from devsim_campaign calib), splits the converged
designs into a training set (75 %) and a held-out test set (25 %), selects the
scale-length form by 5-fold cross-validation on the training set only, fits the
free parameters by robust nonlinear least squares, and reports accuracy metrics
(R^2, RMSE, MAE, MAPE) on both sets.  Writes results/twin_params.json and
results/twin_calibration_metrics.json (+ table_calibration.md).
"""
from __future__ import annotations

import json
import os
import numpy as np
from scipy.optimize import least_squares

from . import config as C
from . import finfet_oracle as O

FIT_NAMES = ["eta_s", "eta_d", "g_u", "a_ss", "a_d", "a_r", "V0", "c_N", "c_N2",
             "C_inv", "mu_s", "v_inj", "r0", "r_u", "g_q", "d0", "b_n"]
X0 = dict(eta_s=2.0, eta_d=2.5, g_u=0.45, a_ss=0.7, a_d=0.3, a_r=0.35, V0=0.33, c_N=1.0,
          c_N2=1.0, C_inv=0.06, mu_s=250.0, v_inj=1.0e7, r0=50.0, r_u=40.0, g_q=0.3,
          d0=8.0, b_n=0.5)
LO = dict(eta_s=0.3, eta_d=0.8, g_u=0.0, a_ss=0.0, a_d=0.0, a_r=-2.0, V0=-1.0, c_N=0.0,
          c_N2=0.0, C_inv=0.005, mu_s=20.0, v_inj=2e6, r0=0.0, r_u=0.0, g_q=-1.0,
          d0=0.0, b_n=0.0)
HI = dict(eta_s=6.0, eta_d=6.0, g_u=1.5, a_ss=20.0, a_d=20.0, a_r=5.0, V0=1.5, c_N=5.0,
          c_N2=5.0, C_inv=1.0, mu_s=2000.0, v_inj=5e7, r0=2000.0, r_u=2000.0, g_q=1.0,
          d0=60.0, b_n=20.0)
SCALE = dict(eta_s=1, eta_d=1, g_u=1, a_ss=1, a_d=1, a_r=1, V0=1, c_N=1, c_N2=1,
             C_inv=0.1, mu_s=100, v_inj=1e7, r0=100, r_u=100, g_q=1, d0=10, b_n=1)


def _unpack(v, lam_form):
    p = {k: float(v[i]) * SCALE[k] for i, k in enumerate(FIT_NAMES)}
    p["lam_form"] = lam_form
    p["k_f"] = 1.0
    return p


def _residuals(v, X, T, lam_form):
    p = _unpack(v, lam_form)
    d = O.intrinsic(X, p)
    r = [
        (d["SS"] - T["SS"]) / (0.02 * T["SS"]),
        (d["Vth_lin"] - T["Vth_lin"]) / 0.010,
        (d["Vth_sat"] - T["Vth_sat"]) / 0.010,
        (np.log10(d["Ion_um"]) - np.log10(T["Ion"])) / 0.03,
        (d["dQg"] - T["dQg_lin"]) / (0.05 * np.abs(T["dQg_lin"])),
        (np.log(d["DIBL"]) - np.log(T["DIBL"])) / 0.10,
    ]
    return np.concatenate(r)


def fit(X, T, lam_form):
    v0 = np.array([X0[k] / SCALE[k] for k in FIT_NAMES])
    lo = np.array([LO[k] / SCALE[k] for k in FIT_NAMES])
    hi = np.array([HI[k] / SCALE[k] for k in FIT_NAMES])
    best = None
    rng = np.random.default_rng(0)
    for trial in range(8):                       # multi-start for robustness
        start = v0 if trial == 0 else np.clip(v0 * np.exp(0.4 * rng.standard_normal(len(v0))), lo, hi)
        r = least_squares(_residuals, start, bounds=(lo, hi), args=(X, T, lam_form),
                          loss="soft_l1", f_scale=2.0, max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    return _unpack(best.x, lam_form), best


def predictions(X, p):
    d = O.intrinsic(X, p)
    return dict(SS=d["SS"], DIBL=d["DIBL"], Vth_sat=d["Vth_sat"], Vth_lin=d["Vth_lin"],
                logIon=np.log10(d["Ion_um"]), logIoff=np.log10(d["Ioff_um"]),
                logratio=np.log10(d["Ion_um"] / d["Ioff_um"]), dQg=d["dQg"])


def targets(T):
    return dict(SS=T["SS"], DIBL=T["DIBL"], Vth_sat=T["Vth_sat"], Vth_lin=T["Vth_lin"],
                logIon=np.log10(T["Ion"]), logIoff=np.log10(T["Ioff"]),
                logratio=T["log10_ratio"], dQg=T["dQg_lin"])


UNITS = dict(SS="mV/dec", DIBL="mV/V", Vth_sat="V", Vth_lin="V", logIon="dec",
             logIoff="dec", logratio="dec", dQg="C/cm^2")
REL = {"SS", "DIBL", "dQg"}           # report MAPE for these (and Ion via logIon)


def metrics(pred, tgt):
    out = {}
    for k in pred:
        y, yh = np.asarray(tgt[k]), np.asarray(pred[k])
        e = yh - y
        ss_res = np.sum(e ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        m = dict(R2=float(1 - ss_res / ss_tot), RMSE=float(np.sqrt(np.mean(e ** 2))),
                 MAE=float(np.mean(np.abs(e))))
        if k in REL:
            m["MAPE_%"] = float(100 * np.mean(np.abs(e / y)))
        if k == "logIon":
            m["MAPE_Ion_%"] = float(100 * np.mean(np.abs(10 ** e - 1)))
        out[k] = m
    return out


def main():
    D = np.load(os.path.join(C.RESULTS_DIR, "devsim_calib.npz"))
    ok = D["ok"].astype(bool)
    X = D["X"][ok]
    T = {k: D[k][ok] for k in ["SS", "DIBL", "Vth_sat", "Vth_lin", "Ion", "Ioff",
                               "log10_ratio", "dQg_lin", "dQg_sat"]}
    n = len(X)
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    ntr = int(round(0.75 * n))
    tr, te = perm[:ntr], perm[ntr:]
    sub = lambda T, idx: {k: v[idx] for k, v in T.items()}

    # model selection (scale-length form) by 5-fold CV on the training set only
    cv = {}
    folds = np.array_split(rng.permutation(tr), 5)
    for form in ("gaa", "dg"):
        errs = []
        for f in range(5):
            va = folds[f]; fi = np.concatenate([folds[j] for j in range(5) if j != f])
            p, _ = fit(X[fi], sub(T, fi), form)
            r = _residuals(np.array([p[k] / SCALE[k] for k in FIT_NAMES]), X[va], sub(T, va), form)
            errs.append(np.mean(r ** 2))
        cv[form] = float(np.mean(errs))
    form = min(cv, key=cv.get)
    print("CV mean-square normalised residual:", cv, "-> selected", form)

    p, res = fit(X[tr], sub(T, tr), form)
    m_tr = metrics(predictions(X[tr], p), targets(sub(T, tr)))
    m_te = metrics(predictions(X[te], p), targets(sub(T, te)))

    with open(os.path.join(C.RESULTS_DIR, "twin_params.json"), "w") as f:
        json.dump(dict(params=p, cv=cv, n_train=int(len(tr)), n_test=int(len(te)),
                       train_idx=tr.tolist(), test_idx=te.tolist()), f, indent=1)
    with open(os.path.join(C.RESULTS_DIR, "twin_calibration_metrics.json"), "w") as f:
        json.dump(dict(train=m_tr, test=m_te, lam_form=form, cv=cv), f, indent=1)

    lines = ["| Quantity | Unit | R² (train) | R² (test) | RMSE (test) | MAPE (test) |",
             "|---|---|---|---|---|---|"]
    for k in ["SS", "DIBL", "Vth_sat", "logIon", "logIoff", "logratio", "dQg"]:
        mt, me = m_tr[k], m_te[k]
        mape = me.get("MAPE_%", me.get("MAPE_Ion_%", float("nan")))
        lines.append(f"| {k} | {UNITS[k]} | {mt['R2']:.3f} | {me['R2']:.3f} | "
                     f"{me['RMSE']:.3g} | {mape:.2f}% |")
    open(os.path.join(C.RESULTS_DIR, "table_calibration.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print("params:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}))


if __name__ == "__main__":
    main()
