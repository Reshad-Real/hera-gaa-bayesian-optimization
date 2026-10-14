"""Problem definition and global settings for the HERA FinFET digital twin.

Everything that is node- or hyper-parameter specific lives here so the rest of
the package stays node-agnostic, exactly as Section 2 of the algorithm spec asks.
The bounds below instantiate the symbolic search box for a 3 nm/2 nm-class
gate-all-around (GAA) logic transistor.  NOTE: node names ("3 nm", "2 nm") are
marketing labels, not physical dimensions; the physical gate length of these
nodes is ~12-16 nm (IRDS More Moore), which is the range used here.
"""
from __future__ import annotations

import os
import numpy as np

# --------------------------------------------------------------------------
# Design variables  x = [Lg, Wfin, Hfin, Tox, Lsp, Nch]      (Eq. 1)
# Lengths in nm, doping in cm^-3.  Doping is searched on a log10 scale.
# --------------------------------------------------------------------------
VAR_NAMES = ["Lg", "Wfin", "Hfin", "Tox", "Lsp", "Nch"]
VAR_UNITS = ["nm", "nm", "nm", "nm", "nm", "cm^-3"]
VAR_LABELS = [
    r"$L_g$", r"$W_{fin}$", r"$H_{fin}$", r"$T_{ox}$", r"$L_{sp}$", r"$N_{ch}$",
]

# Physical lower/upper bounds of the search box X  (Eq. 2): 3 nm/2 nm-class GAA.
#   Lg   gate length                12-20 nm   (IRDS: 16 nm @3nm, 14 nm @2nm, 12 nm floor)
#   Wfin body (sheet/fin) thickness  3-8 nm
#   Hfin fin / stacked-sheet height  20-60 nm
#   Tox  gate EOT                    0.6-1.2 nm
#   Lsp  spacer length               4-10 nm  (sets S/D overlap vs underlap)
#   Nch  channel doping              1e15-5e18 cm^-3 (log scale)
LB = np.array([12.0, 3.0, 20.0, 0.60, 4.0, 1e15])
UB = np.array([20.0, 8.0, 60.0, 1.20, 10.0, 5e18])

# Which variables are handled on a log10 scale for normalisation/search.
LOG_VARS = np.array([False, False, False, False, False, True])

# Hierarchical parameter decomposition into 3 physically ordered tiers (Eq. 15).
#   tier 1 (structural): Lg, Wfin   -> indices 0,1
#   tier 2 (vertical):   Hfin, Tox  -> indices 2,3
#   tier 3 (process):    Lsp, Nch   -> indices 4,5
TIERS = [(0, 1), (2, 3), (4, 5)]

# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------
Q = 1.602176634e-19         # C
KB = 1.380649e-23           # J/K
T_LATTICE = 300.0           # K
PHIT = KB * T_LATTICE / Q   # thermal voltage ~0.02585 V
EPS0 = 8.8541878128e-12     # F/m
EPS_SI = 11.7
EPS_OX = 3.9                # SiO2-equivalent (EOT basis)
NI_SI = 1.0e10              # cm^-3, intrinsic carrier conc. at 300 K

VDD = 0.70                  # supply voltage [V] (3 nm/2 nm-class logic)
VDS_LOW = 0.05              # low drain bias for DIBL/linear extraction [V]
VDS_HIGH = VDD              # high drain bias [V]

# --------------------------------------------------------------------------
# Constraint specifications  c(x) <= 0   (Eqs. 6-10)
# --------------------------------------------------------------------------
TOX_MIN = 0.70              # EOT floor [nm] (gate-leakage / reliability) -> c1
AR_MAX = 10.0              # max manufacturable Hfin/Wfin               -> c2
KAPPA = 6.0                # Lg >= kappa * lambda_GAA (scale-length rule) -> c3
DIBL_MAX = 45.0            # short-channel spec [mV/V] (production GAA: 20-42) -> c4
IOFF_MAX_UM = 1.0e-10      # leakage spec [A/um]: IRDS high-density (HD) Ioff -> c5
N_CONSTRAINTS = 5

# --------------------------------------------------------------------------
# HERA hyper-parameters  (Table 1)
# --------------------------------------------------------------------------
HP = dict(
    ensemble_size=5,            # E
    batch_size=4,               # q
    mc_samples=128,             # S (qEHVI)
    nsga_pop=128,               # NSGA-II population
    nsga_gen=60,                # NSGA-II generations
    ppo_clip=0.2,               # epsilon
    ppo_entropy=0.01,           # omega
    gae_lambda=0.95,            # lambda
    discount=0.99,              # gamma
    reward_feas=0.2,            # eta_f
    reward_viol=0.5,            # eta_c
    conv_window=50,             # W
    eps_hv=1e-3,                # eps_H
    eps_alpha=1e-3,             # eps_alpha
    eps_sigma=0.02,             # eps_sigma (tuned)
    n0=10,                      # initial feasible designs
)

# --------------------------------------------------------------------------
# Objective-space normalisation (fixed reference frame for hypervolume).
# Fixed a priori from the specification envelope and physical limits -- no
# oracle pre-sampling is used:
#   y1 = log10(Ion/Ioff) : 6 decades (nadir) .. 9 decades (ideal)
#   y2 = -SS [V/dec]     : 80 mV/dec (nadir) .. 59.5 mV/dec (300 K thermionic limit)
#   y3 = log10(SP [1/s]) : 11.5 (nadir) .. 12.4 (ideal)
# --------------------------------------------------------------------------
OBJ_MIN = np.array([6.00, -0.0800, 11.50])   # nadir -> normalised 0 (HV reference)
OBJ_MAX = np.array([9.00, -0.0595, 12.40])   # ideal -> normalised 1
REF_POINT_NORM = np.zeros(3)                 # r = 0 in normalised space
M_OBJ = 3


def normalize_x(x: np.ndarray) -> np.ndarray:
    """Map physical design vector(s) to the unit cube [0,1]^6."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    z = np.empty_like(x)
    for i in range(6):
        lo, hi = LB[i], UB[i]
        if LOG_VARS[i]:
            z[:, i] = (np.log10(x[:, i]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        else:
            z[:, i] = (x[:, i] - lo) / (hi - lo)
    return z


def denormalize_x(z: np.ndarray) -> np.ndarray:
    """Map unit-cube coordinate(s) back to physical design vector(s)."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    x = np.empty_like(z)
    for i in range(6):
        lo, hi = LB[i], UB[i]
        if LOG_VARS[i]:
            x[:, i] = 10 ** (np.log10(lo) + z[:, i] * (np.log10(hi) - np.log10(lo)))
        else:
            x[:, i] = lo + z[:, i] * (hi - lo)
    return x


def normalize_y(y: np.ndarray) -> np.ndarray:
    """Map objective vector(s) to a normalised maximisation space (~[0,1])."""
    y = np.atleast_2d(np.asarray(y, dtype=float))
    return (y - OBJ_MIN) / (OBJ_MAX - OBJ_MIN)


def set_reference(obj_min: np.ndarray, obj_max: np.ndarray) -> None:
    """Install the global objective normalisation frame."""
    global OBJ_MIN, OBJ_MAX
    OBJ_MIN = np.asarray(obj_min, dtype=float)
    OBJ_MAX = np.asarray(obj_max, dtype=float)


# --------------------------------------------------------------------------
# DEVSIM math-library bootstrap (Intel MKL + PARDISO).  Must run before the
# first `import devsim`.  Located dynamically so the path is machine-portable.
# --------------------------------------------------------------------------
def enable_devsim_math_libs() -> str:
    """Configure environment so `import devsim` finds a BLAS/LAPACK provider.

    Returns the resolved mkl_rt path (or '' if not found).
    """
    import glob
    import sysconfig

    candidates = []
    # Typical locations of the pip-installed Intel MKL runtime.
    roots = [
        os.path.join(sysconfig.get_path("data"), "Library", "bin"),
        os.path.join(os.path.dirname(os.__file__), "..", "Library", "bin"),
    ]
    site = sysconfig.get_paths().get("purelib", "")
    if site:
        roots.append(os.path.join(os.path.dirname(site), "Library", "bin"))
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isdir(root):
            for name in ("mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.1.dll", "mkl_rt.dll"):
                hits = glob.glob(os.path.join(root, name))
                if hits:
                    candidates.append((root, hits[0]))
    if not candidates:
        return ""
    bindir, mkl = candidates[0]
    os.environ["DEVSIM_MATH_LIBS"] = mkl
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(bindir)
    except (AttributeError, FileNotFoundError, OSError):
        pass
    return mkl


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
