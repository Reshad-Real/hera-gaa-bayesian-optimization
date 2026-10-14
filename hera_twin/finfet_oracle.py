"""TCAD-calibrated compact digital twin  E(x)  of a 3 nm/2 nm-class GAA transistor.

The twin is a deterministic, physics-based semi-analytical (compact) model -- it
is NOT a data-driven surrogate.  Its functional forms are standard compact-model
relations; a small set of free parameters is fitted to drift-diffusion TCAD
(DEVSIM, see devsim_sim.py) over a space-filling calibration set, and its accuracy
is reported on held-out TCAD designs (calibrate_twin.py, Appendix A).

Model (per micrometre of gated perimeter unless stated):
  * Scale length  lambda (Eq. 8 form, selectable), effective channel length
        L_eff = Lg + 2 g_u (Lsp - x_j),  x_j = sigma_sd sqrt(ln(N_SD/N_ch))
    (x_j is where the Gaussian S/D extension doping meets the channel doping:
    Lsp < x_j -> gate overlap, Lsp > x_j -> underlap), and the short-channel
    factor  theta = exp(-L_eff / (2 eta lambda)).
  * SS = SS0 / (1 - a_ss theta);  DIBL = a_d theta;  Vth roll-off a_r theta.
  * Vth_lin/sat = V0 + c_N q N W/(2 C_ox) - a_r theta - DIBL * VDS  (constant-current
    definition at I_crit = 100 nA/um, the same criterion as the TCAD extraction).
  * On-current (virtual-source form):  Q = C_eff n phi_t ln(1+exp((VDD-Vth_sat)/(n phi_t))),
    v = v_d / (1 + v_d / v_inj),  v_d = mu_eff k_f (VDD - Vth_sat)/L_eff,  I_on = Q v.
  * Off-current:  I_off = I_crit 10^(-Vth_sat/SS)  (exactly consistent with the
    constant-current Vth).
  * Switched gate charge  dQ = C_eff n phi_t [ln(1+e^((VDD-Vth_lin)/(n phi_t))) -
    ln(1+e^((-Vth_lin)/(n phi_t)))]  (validated against TCAD gate charge).
  * Per-fin quantities use the gated perimeter W_eff = 2(H_fin + W_fin); the
    switched capacitance adds the outer-fringe (spacer) capacitance
    C_f = 2 (2 eps_sp/pi) ln(1 + H_g/L_sp) W_eff and a fixed per-fin load C_L.
  * Switching-speed FoM  SP = I_on / ((C_int + C_f + C_L) VDD)  [1/s].
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
import numpy as np

from . import config as C

PHI_T = C.PHIT                               # thermal voltage [V]
SS0 = 1000.0 * np.log(10.0) * PHI_T          # ideal swing [mV/dec] (~59.5 at 300 K)
I_CRIT = 1e-7                                # constant-current Vth criterion [A/um]

# Process inputs shared with the TCAD reference (not fitted)
N_SD = 2.0e20                                # peak S/D doping [cm^-3]
SIGMA_SD = 3.0                               # lateral S/D Gaussian straggle [nm]
MU_MAX, MU_MIN, N_REF, ALPHA = 1417.0, 52.2, 9.68e16, 0.68   # Masetti/Arora electrons

# Parasitics (assumed, not calibrated)
EPS_SP = 3.9                                 # spacer relative permittivity (low-k)
H_GATE = 30.0                                # gate electrode height above the body [nm]
C_LOAD = 2.0e-17                             # fixed per-fin load capacitance [F] (0.02 fF)

# Default calibrated parameters (overwritten by results/twin_params.json)
DEFAULT_PARAMS = dict(
    lam_form="gaa", eta_s=2.0, eta_d=2.5, g_u=0.45, a_ss=0.7, a_d=0.3, a_r=0.35,
    V0=0.33, c_N=1.0, c_N2=1.0, C_inv=0.06, mu_s=250.0, v_inj=1.0e7, k_f=1.0,
    r0=50.0, r_u=40.0, g_q=0.3, d0=8.0, b_n=0.5,
)
_PARAM_FILE = os.path.join(C.RESULTS_DIR, "twin_params.json")


def load_params():
    p = dict(DEFAULT_PARAMS)
    if os.path.exists(_PARAM_FILE):
        with open(_PARAM_FILE) as f:
            p.update(json.load(f).get("params", {}))
    return p


PARAMS = load_params()


@dataclass
class DeviceResult:
    Lg: float
    Wfin: float
    Hfin: float
    Tox: float
    Lsp: float
    Nch: float
    lambda_n: float       # scale length [nm]
    L_eff: float          # effective electrostatic channel length [nm]
    Vth_lin: float
    Vth_sat: float
    SS: float             # [mV/dec]
    DIBL: float           # [mV/V]
    Ion_um: float         # [A/um]
    Ioff_um: float        # [A/um]
    Ion: float            # [A/fin]
    Ioff: float           # [A/fin]
    dQg: float            # switched intrinsic gate charge [C/cm^2]
    PE: float             # switching-speed FoM SP [1/s] (field name kept for compatibility)
    Weff: float           # gated perimeter [nm]
    Cox: float            # [F/m^2]
    Ctot: float           # [F/fin]
    y1: float
    y2: float
    y3: float

    def objectives(self):
        return np.array([self.y1, self.y2, self.y3])

    def as_dict(self):
        return asdict(self)


def natural_length_nm(Wfin, Tox, form="gaa"):
    """Electrostatic scale length [nm].  'gaa': Eq. (8); 'dg': double-gate (Suzuki)."""
    Wfin = np.asarray(Wfin, float); Tox = np.asarray(Tox, float)
    if form == "dg":
        return np.sqrt(C.EPS_SI * Wfin * Tox / (2.0 * C.EPS_OX))
    num = 2.0 * C.EPS_SI * Wfin * Tox + C.EPS_OX * Wfin ** 2
    return np.sqrt(num / (16.0 * C.EPS_OX))


def junction_offset_nm(Nch):
    return SIGMA_SD * np.sqrt(np.log(N_SD / np.asarray(Nch, float)))


def mu_bulk(N):
    return MU_MIN + (MU_MAX - MU_MIN) / (1.0 + (np.asarray(N, float) / N_REF) ** ALPHA)


def _softplus_ov(V, n):
    a = V / (n * PHI_T)
    return n * PHI_T * np.where(a > 30, a, np.log1p(np.exp(np.minimum(a, 30))))


def _softplus_nm(u, s=1.0):
    """Smooth max(u, 0) [nm] with a 1 nm transition."""
    return s * np.log1p(np.exp(np.clip(u / s, -30, 30)))


def intrinsic(X, p=None):
    """Per-perimeter (width-independent) quantities for designs X (n,6)."""
    p = PARAMS if p is None else p
    X = np.atleast_2d(np.asarray(X, float))
    Lg, Wfin, Hfin, Tox, Lsp, Nch = (X[:, i] for i in range(6))
    lam = natural_length_nm(Wfin, Tox, p["lam_form"])
    xj = junction_offset_nm(Nch)
    u = Lsp - xj                                                     # underlap (>0) / overlap (<0) [nm]
    Leff = np.maximum(Lg + 2.0 * p["g_u"] * u, 0.4 * Lg)
    theta_s = np.exp(-Leff / (2.0 * p["eta_s"] * lam))
    theta_d = np.exp(-Leff / (2.0 * p["eta_d"] * lam))
    theta = theta_s
    Cox = C.EPS_OX * C.EPS0 / (Tox * 1e-9)                            # F/m^2
    qNW = C.Q * (Nch * 1e6) * (Wfin * 1e-9)                           # C/m^2
    # body-charge degradation of the subthreshold ideality (b_n) and
    # short-channel degradation through the scale-length factor theta_s
    SS = SS0 * (1.0 + p["b_n"] * qNW / (2.0 * Cox)) / np.clip(1.0 - p["a_ss"] * theta_s, 0.2, None)
    # DIBL: short-channel term + long-channel floor.  The floor has an analytic
    # part from the constant-current Vth extraction (the low-VDS current is
    # reduced by 1 - exp(-VDS_low/phi_t)) plus a fitted residual d0.
    dibl_cc = (SS / 1000.0) * np.log10(1.0 / (1.0 - np.exp(-C.VDS_LOW / PHI_T)))         / (C.VDS_HIGH - C.VDS_LOW) * 1000.0
    DIBL = 1000.0 * p["a_d"] * theta_d + p["d0"] + dibl_cc            # mV/V
    Vdop = (p["c_N"] * qNW / (2.0 * Cox)
            + p["c_N2"] * qNW * (Wfin * 1e-9) / (8.0 * C.EPS_SI * C.EPS0))   # V
    Vlong = p["V0"] + Vdop - p["a_r"] * theta_s
    Vth_lin = Vlong - DIBL / 1000.0 * C.VDS_LOW
    Vth_sat = Vlong - DIBL / 1000.0 * C.VDS_HIGH
    n = SS / SS0
    Ceff = 1.0 / (1.0 / Cox + 1.0 / p["C_inv"])                       # F/m^2
    Vov = _softplus_ov(C.VDD - Vth_sat, n)
    Q = Ceff * Vov                                                    # C/m^2
    mu = 1.0 / (1.0 / mu_bulk(Nch) + 1.0 / p["mu_s"]) * 1e-4          # m^2/Vs
    vd = mu * p["k_f"] * np.maximum(C.VDD - Vth_sat, 0.02) / (Leff * 1e-9)
    vinj = p["v_inj"] * 1e-2                                          # m/s
    v = vd / (1.0 + vd / vinj)
    I_int = Q * v * 1e-6                                              # A/um (intrinsic)
    # source/drain series resistance [Ohm um]: contact/extension + underlap term
    Rs = p["r0"] + p["r_u"] * _softplus_nm(u)
    Ion_um = I_int / (1.0 + I_int * Rs / np.maximum(Vov, 0.05))       # A/um
    Ioff_um = I_CRIT * 10.0 ** (-Vth_sat / (SS / 1000.0))             # A/um
    # switched gate charge per gated area; overlap adds, underlap removes charge
    Lq = np.clip(1.0 - 2.0 * p["g_q"] * u / Lg, 0.3, 2.0)
    dQ = Ceff * Lq * (_softplus_ov(C.VDD - Vth_lin, n) - _softplus_ov(-Vth_lin, n)) * 1e-4  # C/cm^2
    return dict(lam=lam, Leff=Leff, theta=theta, SS=SS, DIBL=DIBL, Vth_lin=Vth_lin,
                Vth_sat=Vth_sat, Ion_um=Ion_um, Ioff_um=Ioff_um, dQg=dQ, Cox=Cox)


def evaluate_array(X):
    X = np.atleast_2d(np.asarray(X, float))
    Lg, Wfin, Hfin, Tox, Lsp, Nch = (X[:, i] for i in range(6))
    d = intrinsic(X)
    Weff_nm = 2.0 * (Hfin + Wfin)
    Weff_um = Weff_nm * 1e-3
    Ion = d["Ion_um"] * Weff_um
    Ioff = d["Ioff_um"] * Weff_um
    Cint = d["dQg"] * 1e4 / C.VDD * (Lg * 1e-9) * (Weff_nm * 1e-9)          # F/fin
    Cf = 2.0 * (2.0 * EPS_SP * C.EPS0 / np.pi) * np.log1p(H_GATE / Lsp) * (Weff_nm * 1e-9)
    Ctot = Cint + Cf + C_LOAD
    SP = Ion / (Ctot * C.VDD)
    y1 = np.log10(Ion / np.maximum(Ioff, 1e-30))
    y2 = -d["SS"] / 1000.0
    y3 = np.log10(SP)
    return dict(Lg=Lg, Wfin=Wfin, Hfin=Hfin, Tox=Tox, Lsp=Lsp, Nch=Nch,
                lambda_n=d["lam"], L_eff=d["Leff"], Vth_lin=d["Vth_lin"], Vth_sat=d["Vth_sat"],
                SS=d["SS"], DIBL=d["DIBL"], Ion_um=d["Ion_um"], Ioff_um=d["Ioff_um"],
                Ion=Ion, Ioff=Ioff, dQg=d["dQg"], PE=SP, Weff=Weff_nm, Cox=d["Cox"],
                Ctot=Ctot, y1=y1, y2=y2, y3=y3)


def evaluate(x):
    d = evaluate_array(np.atleast_2d(x))
    return DeviceResult(**{k: float(np.ravel(v)[0]) for k, v in d.items()})


def objectives(X):
    d = evaluate_array(X)
    return np.column_stack([d["y1"], d["y2"], d["y3"]])


# --------------------------------------------------------------------------
# Constraints c(x) <= 0 (Eqs. 6-10).  c1-c3 are analytic (geometry/process);
# c4-c5 require a device evaluation.
# --------------------------------------------------------------------------
def geometric_constraints(X):
    X = np.atleast_2d(np.asarray(X, float))
    Lg, Wfin, Hfin, Tox = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    c1 = C.TOX_MIN - Tox
    c2 = Hfin / Wfin - C.AR_MAX
    c3 = C.KAPPA * natural_length_nm(Wfin, Tox, "gaa") - Lg
    return np.column_stack([c1, c2, c3])


def constraints(X):
    X = np.atleast_2d(np.asarray(X, float))
    d = evaluate_array(X)
    g = geometric_constraints(X)
    c4 = (d["DIBL"] - C.DIBL_MAX) / C.DIBL_MAX                       # scaled
    c5 = np.log10(d["Ioff_um"] / C.IOFF_MAX_UM)                       # log-scaled
    return np.column_stack([g, c4, c5])


def is_feasible(X, tol=0.0):
    return np.all(constraints(X) <= tol, axis=1)


def total_violation(X):
    return np.sum(np.clip(constraints(X), 0.0, None), axis=1)


CONSTRAINT_NAMES = [
    "c1: Tox >= Tox_min",
    "c2: Hfin/Wfin <= ARmax",
    "c3: Lg >= kappa*lambda_GAA",
    "c4: DIBL <= DIBLmax",
    "c5: Ioff <= Ioff_max",
]


def idvg_um(x, vgs, vds):
    """Continuous twin I_D(V_GS) [A/um] at drain bias vds, for plotting only.

    Harmonic blend of the subthreshold law I_crit 10^((V_GS-V_th)/SS) and the
    virtual-source on-state law evaluated at V_GS; it reproduces the twin's
    I_off (V_GS = 0) and I_on (V_GS = V_DD, V_DS = V_DD) exactly in the limits.
    """
    p = PARAMS
    x = np.atleast_2d(np.asarray(x, float))
    d = intrinsic(x, p)
    vgs = np.asarray(vgs, float)
    Vlong = d["Vth_lin"][0] + d["DIBL"][0] / 1000.0 * C.VDS_LOW
    Vth = Vlong - d["DIBL"][0] / 1000.0 * vds
    SSv = d["SS"][0] / 1000.0
    n = d["SS"][0] / SS0
    I_sub = I_CRIT * 10.0 ** ((vgs - Vth) / SSv)
    Lg, Wfin, Hfin, Tox, Lsp, Nch = x[0]
    Cox = C.EPS_OX * C.EPS0 / (Tox * 1e-9)
    Ceff = 1.0 / (1.0 / Cox + 1.0 / p["C_inv"])
    Vov = _softplus_ov(vgs - Vth, n)
    mu = 1.0 / (1.0 / mu_bulk(Nch) + 1.0 / p["mu_s"]) * 1e-4
    Leff = d["Leff"][0]
    # linear-region drain field for small vds; saturation field otherwise
    vdrive = np.minimum(np.maximum(vgs - Vth, 0.02), vds) if vds < 0.2 else np.maximum(vgs - Vth, 0.02)
    vd = mu * vdrive / (Leff * 1e-9)
    vinj = p["v_inj"] * 1e-2
    v = vd / (1.0 + vd / vinj)
    I_int = Ceff * Vov * v * 1e-6
    u = Lsp - junction_offset_nm(Nch)
    Rs = p["r0"] + p["r_u"] * _softplus_nm(u)
    I_on = I_int / (1.0 + I_int * Rs / np.maximum(Vov, 0.05))
    return 1.0 / (1.0 / I_sub + 1.0 / I_on)
