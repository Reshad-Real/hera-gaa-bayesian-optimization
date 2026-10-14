"""Publication-ready tables (markdown + CSV + LaTeX) for the HERA study."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from . import config as C
from . import finfet_oracle as O


def _manual_md(df, index=False):
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=index, name=None)]
    return "\n".join([header, sep] + rows)


def _save(df: pd.DataFrame, name: str, float_fmt="%.4g", index=False):
    base = os.path.join(C.RESULTS_DIR, name)
    df.to_csv(base + ".csv", index=index)
    with open(base + ".md", "w") as f:
        try:
            f.write(df.to_markdown(index=index, floatfmt=".4g"))
        except Exception:
            f.write(_manual_md(df, index=index))
    try:
        with open(base + ".tex", "w") as f:
            f.write(df.to_latex(index=index, float_format=lambda x: float_fmt % x))
    except Exception:
        pass
    return df


def _device_rows(X):
    """Return a DataFrame of design parameters + device figures of merit."""
    X = np.atleast_2d(X)
    d = O.evaluate_array(X)
    rows = []
    for i in range(len(X)):
        rows.append(dict(
            Lg_nm=X[i, 0], Wfin_nm=X[i, 1], Hfin_nm=X[i, 2], Tox_nm=X[i, 3],
            Lsp_nm=X[i, 4], Nch_cm3=X[i, 5],
            Ion_uA=d["Ion"][i] * 1e6, Ioff_pA=d["Ioff"][i] * 1e12,
            SS_mVdec=d["SS"][i], DIBL_mVV=d["DIBL"][i], Vth_V=d["Vth_sat"][i],
            Ion_Ioff_log10=d["y1"][i], PE_log10=d["y3"][i],
        ))
    return pd.DataFrame(rows)


def table_initial_designs(init_X):
    df = _device_rows(init_X)
    df.insert(0, "Design", [f"D{i+1}" for i in range(len(df))])
    return _save(df, "table1_initial_designs")


def table_pareto_designs(pareto_X, max_rows=None):
    df = _device_rows(pareto_X)
    # order by Ion/Ioff for readability
    df = df.sort_values("Ion_Ioff_log10", ascending=False).reset_index(drop=True)
    if max_rows:
        df = df.head(max_rows)
    df.insert(0, "Pareto pt", [f"P{i+1}" for i in range(len(df))])
    return _save(df, "table2_pareto_designs")


def table_method_comparison(summary):
    stats = summary["stats"]
    sig = summary["significance"]
    rows = []
    for m, s in stats.items():
        row = dict(Method=m, HV_mean=s["mean"], HV_sd=s["sd"], CI95=s["ci95"],
                   HV_median=s["median"], HV_min=s["min"], HV_max=s["max"])
        if m in sig:
            row["dHV_vs_HERA"] = -sig[m]["mean_improvement"]
            row["p_value"] = sig[m]["p_value"]
            row["HERA_wins"] = f"{sig[m]['wins']}/{sig[m]['wins']+sig[m]['losses']+sig[m]['ties']}"
        else:
            row["dHV_vs_HERA"] = 0.0
            row["p_value"] = np.nan
            row["HERA_wins"] = "-"
        rows.append(row)
    df = pd.DataFrame(rows)
    return _save(df, "table3_method_comparison")


def table_chosen_design(params, label="chosen"):
    Lg, Wfin, Hfin, Tox, Lsp, Nch = [float(v) for v in np.ravel(params)[:6]]
    r = O.evaluate(params)
    rows = [
        ("Gate length  Lg", f"{Lg:.2f}", "nm"),
        ("Fin width  Wfin", f"{Wfin:.2f}", "nm"),
        ("Fin height  Hfin", f"{Hfin:.2f}", "nm"),
        ("Gate-oxide EOT  Tox", f"{Tox:.3f}", "nm"),
        ("Spacer width  Lsp", f"{Lsp:.2f}", "nm"),
        ("Channel doping  Nch", f"{Nch:.3e}", "cm^-3"),
        ("Natural length  lambda_n", f"{r.lambda_n:.3f}", "nm"),
        ("Aspect ratio  Hfin/Wfin", f"{Hfin/Wfin:.2f}", "-"),
        ("Effective width  Weff", f"{r.Weff:.2f}", "nm"),
        ("On-current  Ion", f"{r.Ion*1e6:.3f}", "uA/fin"),
        ("Off-current  Ioff", f"{r.Ioff*1e12:.4f}", "pA/fin"),
        ("Subthreshold swing  SS", f"{r.SS:.2f}", "mV/dec"),
        ("DIBL", f"{r.DIBL:.2f}", "mV/V"),
        ("Threshold  Vth(sat)", f"{r.Vth_sat:.4f}", "V"),
        ("log10(Ion/Ioff)", f"{r.y1:.3f}", "-"),
        ("-SS  (objective y2)", f"{r.y2:.5f}", "V/dec"),
        ("log10(PE)  (objective y3)", f"{r.y3:.3f}", "-"),
    ]
    df = pd.DataFrame(rows, columns=["Parameter / FoM", "Value", "Unit"])
    return _save(df, f"table4_{label}_design")


def table_devsim_comparison(params, devsim_metrics):
    """Compare the oracle and DEVSIM for the chosen device.

    SS and DIBL are work-function-independent short-channel metrics and form the
    primary cross-check.  Vth differs by a constant metal-gate work-function
    offset (DEVSIM uses an ideal gate for robust convergence); that rigid VGS
    shift does not affect SS or DIBL.
    """
    r = O.evaluate(params)
    dm = devsim_metrics
    wf = dm.get("gate_wf_offset", 0.0)
    rows = []

    def add(name, oracle, devsim, unit, note=""):
        if oracle not in (0, None) and np.isfinite(oracle) and not note:
            note = f"{100.0 * (devsim - oracle) / abs(oracle):+.1f}%"
        rows.append(dict(Metric=name, Oracle=oracle, DEVSIM=devsim,
                         Note=note, Unit=unit))

    # primary work-function-independent validations
    add("Subthreshold swing SS", r.SS, dm["SS"], "mV/dec")
    add("DIBL", r.DIBL, dm["DIBL"], "mV/V")
    # threshold: rigid metal-gate work-function offset
    add("Vth (sat), ideal gate", r.Vth_sat, dm["Vth_sat"], "V",
        note=f"WF offset {wf:+.3f} V")
    add("Vth (sat), WF-aligned", r.Vth_sat, dm["Vth_sat"] + wf, "V",
        note="matched by 1 WF param")
    # currents per um of gate width (oracle per-fin -> per-um)
    add("Ion (per um width)", r.Ion / (r.Weff * 1e-3), dm["Ion"], "A/um")
    df = pd.DataFrame(rows)
    return _save(df, "table5_devsim_comparison")
