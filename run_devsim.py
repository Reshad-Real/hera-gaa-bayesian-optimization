"""Recreate the chosen optimised FinFET in DEVSIM and compare with the oracle.

Reads results/chosen_design.npy (written by report.generate_all), runs the 2-D
double-gate drift-diffusion Id-Vg sweep, extracts Vth/SS/DIBL/Ion/Ioff, writes
the comparison table and the Id-Vg / bar figures.
"""
import os
import json
import numpy as np

from hera_twin import config as C
from hera_twin import devsim_sim as D     # sets up MKL + imports devsim
from hera_twin import tables as T
from hera_twin import plots as P
import devsim as ds

ds.set_parameter(name="debug_level", value="error")


VDS_HIGH = 0.5    # reduced drain bias (sub-3nm-appropriate supply; aids DD convergence)


def main():
    from hera_twin import finfet_oracle as O
    x = np.load(os.path.join(C.RESULTS_DIR, "chosen_design.npy"))
    print("Chosen design:", np.round(x, 4))
    vth_oracle = O.evaluate(x).Vth_sat

    # Ideal metal gate (WF=0) for robust DD convergence at sub-3nm.  The absolute
    # Vth then differs from the oracle by a constant metal-gate work-function
    # offset; this rigid VGS shift does NOT affect SS or DIBL, which are the
    # work-function-independent short-channel metrics being cross-checked.
    D.GATE_WF = 0.0
    m = D.run_idvg(x, vd_high=VDS_HIGH)
    m["gate_wf_offset"] = float(vth_oracle - m["Vth_sat"])
    print(f"Implied metal-gate work-function offset: {m['gate_wf_offset']:+.3f} V")
    # persist metrics (+ curves)
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in m.items() if k != "params"}
    with open(os.path.join(C.RESULTS_DIR, "devsim_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\nDEVSIM extracted:  Vth_sat=%.4f  SS=%.2f  DIBL=%.2f  "
          "Ion=%.3e  Ioff=%.3e  log10(Ion/Ioff)=%.3f"
          % (m["Vth_sat"], m["SS"], m["DIBL"], m["Ion"], m["Ioff"], m["log10_ratio"]))

    T.table_devsim_comparison(x, m)
    P.plot_idvg(m, x)
    P.plot_oracle_vs_devsim_bars(x, m)
    print("DEVSIM comparison table + figures written to results/.")


if __name__ == "__main__":
    main()
