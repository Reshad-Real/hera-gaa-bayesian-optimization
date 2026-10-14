"""DEVSIM drift-diffusion reference simulator for the GAA design case study.

Each design is rebuilt as a 2-D double-gate cross-section in the transport
plane (the dominant sidewall-gated slice of the gate-all-around body):

    x = source contact -> spacer -> gate -> spacer -> drain contact
        (total length 2*Lsp + Lg; the contacts sit at the outer spacer edges),
    y = across the body (thickness = Wfin) with a gate oxide of EOT Tox and a
        gate electrode on each face (the double gate) over the gate length.

Physics (all standard TCAD models, see the manuscript, Appendix A):
  * Poisson + electron/hole continuity, Scharfetter-Gummel discretisation,
    Shockley-Read-Hall recombination (DEVSIM simple_physics).
  * Source/drain doping: Gaussian lateral profile decaying from each contact
    edge into the spacer region,  N_D(x) = N_SD exp(-(x/sigma)^2) + mirror,
    so a short spacer gives gate overlap and a long spacer gives underlap
    (the classical spacer/underlap trade-off).
  * Electron mobility: doping-dependent Caughey-Thomas (Masetti/Arora
    parameters) combined by Matthiessen's rule with a thin-body/surface
    scattering limit, and a Caughey-Thomas high-field (velocity-saturation)
    model driven by the field component along the transport direction.

Extracted figures of merit (per micrometre of *gated perimeter*: the 2-D
current per unit depth of the double-gated slab is divided by 2, one per gated
face):  constant-current Vth (lin/sat), SS, DIBL, Ion, Ioff, log10(Ion/Ioff),
and the switched gate charge dQg = Qg(VGS=VDD) - Qg(VGS=0).

All lengths internally in cm (DEVSIM convention); 1 nm = 1e-7 cm.
"""
from __future__ import annotations

import numpy as np

from . import config as C
C.enable_devsim_math_libs()   # must precede `import devsim`

import devsim as ds
from devsim.python_packages import simple_physics as sp
from devsim.python_packages.model_create import (
    CreateNodeModel, CreateSolution, CreateEdgeModel)
from devsim.python_packages.simple_dd import CreateElectronCurrent, CreateHoleCurrent

NM = 1e-7                     # cm per nm
NSD = 2.0e20                  # peak source/drain donor doping [cm^-3]
SIGMA_SD = 3.0                # lateral S/D Gaussian straggle [nm]
GATE_WF = -0.05               # gate work-function offset w.r.t. midgap [V]; one
                              # technology-wide value (not tuned per design)

# Electron mobility model parameters
MU_MAX, MU_MIN, N_REF, ALPHA = 1417.0, 52.2, 9.68e16, 0.68   # Masetti/Arora (electrons)
MU_SR = 250.0                 # thin-body / surface-roughness limit [cm^2/Vs]
VSAT_N = 1.07e7               # electron saturation velocity [cm/s] (Canali)
MU_P = 100.0                  # hole mobility (minority carriers; irrelevant for nFET)

PHYS = ["bulk", "oxtop", "oxbot", "source", "drain",
        "gate_top", "gate_bot", "si_oxtop", "si_oxbot"]
_PIDX = {n: i for i, n in enumerate(PHYS)}


def _segmented_grid(points, spacings):
    arr = [np.array([points[0]])]
    for (a, b), s in zip(zip(points[:-1], points[1:]), spacings):
        n = max(1, int(round((b - a) / s)))
        arr.append(np.linspace(a, b, n + 1)[1:])
    g = np.concatenate(arr)
    return np.unique(np.round(g, 13))


def build_mesh_data(params):
    Lg, Wfin, Hfin, Tox, Lsp, Nch = [float(v) for v in np.ravel(params)[:6]]
    tox, tsi, lg, lsp = Tox * NM, Wfin * NM, Lg * NM, Lsp * NM
    xg1, xg2 = lsp, lsp + lg               # gate edges
    Lx = 2 * lsp + lg

    ox_sp = max(tox / 3.0, 0.1 * NM)
    si_sp = max(tsi / 10.0, 0.25 * NM)
    ch_sp = 0.5 * NM
    sp_sp = 0.4 * NM

    xs = _segmented_grid([0.0, xg1, xg2, Lx], [sp_sp, ch_sp, sp_sp])
    ys = _segmented_grid([-tox, 0.0, tsi, tsi + tox], [ox_sp, si_sp, ox_sp])
    nx, ny = len(xs), len(ys)

    def nid(i, j):
        return j * nx + i

    coords = []
    for j in range(ny):
        for i in range(nx):
            coords.extend([float(xs[i]), float(ys[j]), 0.0])

    j0 = int(np.argmin(np.abs(ys - 0.0)))
    jt = int(np.argmin(np.abs(ys - tsi)))
    ix1 = int(np.argmin(np.abs(xs - xg1)))
    ix2 = int(np.argmin(np.abs(xs - xg2)))

    elements = []

    def add_el(etype, pname, nodes):
        elements.extend([etype, _PIDX[pname]] + list(nodes))

    for j in range(ny - 1):
        yc = 0.5 * (ys[j] + ys[j + 1])
        reg = "oxbot" if yc < 0 else ("bulk" if yc < tsi else "oxtop")
        for i in range(nx - 1):
            n00, n10 = nid(i, j), nid(i + 1, j)
            n01, n11 = nid(i, j + 1), nid(i + 1, j + 1)
            add_el(2, reg, [n00, n10, n11])
            add_el(2, reg, [n00, n11, n01])

    for j in range(j0, jt):
        add_el(1, "source", [nid(0, j), nid(0, j + 1)])
        add_el(1, "drain", [nid(nx - 1, j), nid(nx - 1, j + 1)])
    for i in range(ix1, ix2):
        add_el(1, "gate_top", [nid(i, ny - 1), nid(i + 1, ny - 1)])
        add_el(1, "gate_bot", [nid(i, 0), nid(i + 1, 0)])
    for i in range(nx - 1):
        add_el(1, "si_oxtop", [nid(i, jt), nid(i + 1, jt)])
        add_el(1, "si_oxbot", [nid(i, j0), nid(i + 1, j0)])

    info = dict(Lx=Lx, xg1=xg1, xg2=xg2, tsi=tsi, tox=tox, Nch=Nch, Lsp=lsp,
                nx=nx, ny=ny, params=(Lg, Wfin, Hfin, Tox, Lsp, Nch))
    return coords, elements, list(PHYS), info


def build_device(params, device="gaa", mesh="gaa_mesh"):
    for fn, kw in ((ds.delete_device, dict(device=device)), (ds.delete_mesh, dict(mesh=mesh))):
        try:
            fn(**kw)
        except Exception:
            pass
    coords, elements, physnames, info = build_mesh_data(params)
    ds.create_gmsh_mesh(mesh=mesh, coordinates=coords, elements=elements,
                        physical_names=physnames)
    ds.add_gmsh_region(mesh=mesh, gmsh_name="bulk", region="bulk", material="Silicon")
    ds.add_gmsh_region(mesh=mesh, gmsh_name="oxtop", region="oxtop", material="Oxide")
    ds.add_gmsh_region(mesh=mesh, gmsh_name="oxbot", region="oxbot", material="Oxide")
    for c in ("source", "drain"):
        ds.add_gmsh_contact(mesh=mesh, gmsh_name=c, name=c, region="bulk", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="gate_top", name="gate_top",
                        region="oxtop", material="metal")
    ds.add_gmsh_contact(mesh=mesh, gmsh_name="gate_bot", name="gate_bot",
                        region="oxbot", material="metal")
    ds.add_gmsh_interface(mesh=mesh, gmsh_name="si_oxtop", name="si_oxtop",
                          region0="bulk", region1="oxtop")
    ds.add_gmsh_interface(mesh=mesh, gmsh_name="si_oxbot", name="si_oxbot",
                          region0="bulk", region1="oxbot")
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=device)
    info["device"] = device
    return info


def _set_doping(device, region, info):
    """Gaussian lateral S/D doping from each contact edge; uniform p-type channel."""
    Lx, Nch = info["Lx"], info["Nch"]
    s = SIGMA_SD * NM
    # (pow() is used because DEVSIM parses -(a)^2 as (-(a))^2)
    nd = ("{N}*(exp(-pow(x/{s}, 2)) + exp(-pow(({Lx}-x)/{s}, 2)))"
          .format(N=NSD, s=s, Lx=Lx))
    CreateNodeModel(device, region, "Donors", nd)
    CreateNodeModel(device, region, "Acceptors", "{0}".format(Nch))
    CreateNodeModel(device, region, "NetDoping", "Donors - Acceptors")
    CreateNodeModel(device, region, "TotalDoping", "Donors + Acceptors")


def _create_mobility(device, region):
    """Edge low-field electron mobility (doping CT + Matthiessen surface limit)."""
    ds.edge_from_node_model(device=device, region=region, node_model="TotalDoping")
    mu_ct = ("({mn} + ({mx}-{mn})/(1 + pow(0.5*(TotalDoping@n0+TotalDoping@n1)/{nr}, {a})))"
             .format(mn=MU_MIN, mx=MU_MAX, nr=N_REF, a=ALPHA))
    CreateEdgeModel(device, region, "mu_lf_n", "1/(1/{0} + 1/{1})".format(mu_ct, MU_SR))
    ds.set_parameter(device=device, region=region, name="vsat_n", value=VSAT_N)
    # field-dependent mobility (Caughey-Thomas, beta=2) on the x-directed field
    # component of each edge; written inline so Newton gets exact derivatives.
    return ("(mu_lf_n/pow(1 + pow(mu_lf_n*(Potential@n0-Potential@n1)*EdgeInverseLength*unitx"
            "/vsat_n, 2), 0.5))")


def setup_physics(info, T=300.0):
    device = info["device"]
    silicon, oxides = "bulk", ["oxtop", "oxbot"]
    sp.SetSiliconParameters(device, silicon, T)
    ds.set_parameter(device=device, region=silicon, name="mu_p", value=MU_P)
    for ox in oxides:
        sp.SetOxideParameters(device, ox, T)
    _set_doping(device, silicon, info)

    CreateSolution(device, silicon, "Potential")
    init = ("ifelse(NetDoping > 0, V_t*log({c}/n_i), -V_t*log({h}/n_i))"
            .format(c=sp.celec_model, h=sp.chole_model))
    CreateNodeModel(device, silicon, "PotInit", init)
    ds.set_node_values(device=device, region=silicon, name="Potential", init_from="PotInit")

    sp.CreateSiliconPotentialOnly(device, silicon)
    for ox in oxides:
        CreateSolution(device, ox, "Potential")
        sp.CreateOxidePotentialOnly(device, ox, "log_damp")
    for c in ("source", "drain"):
        ds.set_parameter(device=device, name=sp.GetContactBiasName(c), value=0.0)
        sp.CreateSiliconPotentialOnlyContact(device, silicon, c)
    for g in ("gate_top", "gate_bot"):
        ds.set_parameter(device=device, name=sp.GetContactBiasName(g), value=0.0)
        _create_gate_contact(device, "oxtop" if g == "gate_top" else "oxbot", g)
    sp.CreateSiliconOxideInterface(device, "si_oxtop")
    sp.CreateSiliconOxideInterface(device, "si_oxbot")


def _create_gate_contact(device, region, contact):
    bias = sp.GetContactBiasName(contact)
    cmn = sp.GetContactNodeModelName(contact)
    eq = "Potential - {0} + {1}".format(bias, GATE_WF)
    sp.CreateContactNodeModel(device, contact, cmn, eq)
    sp.CreateContactNodeModelDerivative(device, contact, cmn, eq, "Potential")
    if not sp.InEdgeModelList(device, region, sp.contactcharge_edge):
        sp.CreateEdgeModel(device, region, sp.contactcharge_edge, "Permittivity*ElectricField")
        sp.CreateEdgeModelDerivatives(device, region, sp.contactcharge_edge,
                                      "Permittivity*ElectricField", "Potential")
    ds.contact_equation(device=device, contact=contact, name="PotentialEquation",
                        node_model=cmn, edge_charge_model=sp.contactcharge_edge)


def add_drift_diffusion(info):
    device, silicon = info["device"], "bulk"
    CreateSolution(device, silicon, "Electrons")
    CreateSolution(device, silicon, "Holes")
    ds.set_node_values(device=device, region=silicon, name="Electrons", init_from="IntrinsicElectrons")
    ds.set_node_values(device=device, region=silicon, name="Holes", init_from="IntrinsicHoles")
    mu_n_expr = _create_mobility(device, silicon)
    # assemble drift-diffusion with the field-dependent electron mobility
    sp.CreatePE(device, silicon)
    sp.CreateBernoulli(device, silicon)
    sp.CreateSRH(device, silicon)
    CreateElectronCurrent(device, silicon, mu_n_expr)
    CreateNodeModel(device, silicon, "NCharge", "-ElectronCharge * Electrons")
    sp.CreateNodeModelDerivative(device, silicon, "NCharge", "-ElectronCharge * Electrons", "Electrons")
    ds.equation(device=device, region=silicon, name="ElectronContinuityEquation",
                variable_name="Electrons", time_node_model="NCharge",
                edge_model="ElectronCurrent", variable_update="positive",
                node_model="ElectronGeneration")
    sp.CreateHCE(device, silicon, "mu_p")
    for c in ("source", "drain"):
        sp.CreateSiliconDriftDiffusionAtContact(device, silicon, c)


# --------------------------------------------------------------------------
def _solve(**kw):
    p = dict(type="dc", solver_type="direct", relative_error=1e-8,
             absolute_error=1e10, maximum_iterations=60)
    p.update(kw)
    ds.solve(**p)


def _ramp(setter, cur, target, base_step=0.05, min_step=2e-4):
    step = base_step
    while abs(target - cur) > 1e-9:
        s = min(step, abs(target - cur)) * (1.0 if target > cur else -1.0)
        trial = cur + s
        setter(trial)
        try:
            _solve()
            cur = trial
            step = min(base_step, step * 1.5)
        except Exception:
            setter(cur)
            step *= 0.5
            if step < min_step:
                raise RuntimeError(f"bias ramp stalled at {cur:.4f} -> {target}")
    return cur


def drain_current(device):
    e = ds.get_contact_current(device=device, contact="drain", equation=sp.ece_name)
    h = ds.get_contact_current(device=device, contact="drain", equation=sp.hce_name)
    return e + h                               # A per cm of depth (2-D)


def gate_charge(device):
    return sum(ds.get_contact_charge(device=device, contact=g, equation="PotentialEquation")
               for g in ("gate_top", "gate_bot"))   # C per cm of depth


def _sweep_one_vds(params, vgs, vd):
    info = build_device(params)
    device = info["device"]
    setup_physics(info)
    _solve()
    add_drift_diffusion(info)
    _solve()
    dname = sp.GetContactBiasName("drain")
    _ramp(lambda v: ds.set_parameter(device=device, name=dname, value=v), 0.0, vd)

    def gset(v):
        for g in ("gate_top", "gate_bot"):
            ds.set_parameter(device=device, name=sp.GetContactBiasName(g), value=v)

    Id, Qg, cur = [], [], 0.0
    for vg in vgs:
        cur = _ramp(gset, cur, float(vg))
        Id.append(abs(drain_current(device)))
        Qg.append(gate_charge(device))
    return np.array(Id), np.array(Qg), info


def run_idvg(params, vgs=None, vd_low=None, vd_high=None):
    """Id-Vg (+ gate charge) at low and high VDS. Per-um-of-gated-perimeter units."""
    if vgs is None:
        vgs = np.round(np.arange(0.0, C.VDD + 1e-9, 0.025), 4)
    vd_low = C.VDS_LOW if vd_low is None else vd_low
    vd_high = C.VDS_HIGH if vd_high is None else vd_high
    Id_lin, Qg_lin, info = _sweep_one_vds(params, vgs, vd_low)
    Id_sat, Qg_sat, _ = _sweep_one_vds(params, vgs, vd_high)
    # A/cm (depth) of the double-gated slab -> A/um of gated perimeter (2 faces)
    to_perim = 1.0 / (2.0 * 1e4)
    m = extract_metrics(np.array(vgs), Id_lin * to_perim, Id_sat * to_perim, vd_low, vd_high)
    # switched gate charge per unit gated area [C/cm^2]: per-cm-depth charge
    # divided by the 2 gated faces and the gate length
    lg_cm = info["xg2"] - info["xg1"]
    dq = lambda Q: (Q[-1] - Q[0]) / (2.0 * lg_cm)
    m.update(dQg_lin=float(dq(Qg_lin)), dQg_sat=float(dq(Qg_sat)),
             vgs=np.array(vgs), Id_lin=Id_lin * to_perim, Id_sat=Id_sat * to_perim,
             Qg_lin=Qg_lin, Qg_sat=Qg_sat, vd_low=vd_low, vd_high=vd_high,
             params=info["params"])
    return m


# --------------------------------------------------------------------------
I_CRIT = 1e-7   # constant-current threshold criterion [A/um of gated perimeter]


def _vth_constant_current(vgs, Id, icrit=I_CRIT):
    Id = np.asarray(Id)
    above = np.where(Id >= icrit)[0]
    if len(above) == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float("nan")
    x0, x1 = vgs[i - 1], vgs[i]
    y0, y1 = np.log10(Id[i - 1]), np.log10(Id[i])
    return float(x0 + (x1 - x0) * (np.log10(icrit) - y0) / (y1 - y0))


def _subthreshold_swing(vgs, Id, icrit=I_CRIT):
    """Mean swing over the decade-and-a-half below the threshold current."""
    Id = np.asarray(Id)
    logI = np.log10(np.maximum(Id, 1e-30))
    lo, hi = np.log10(icrit) - 2.0, np.log10(icrit) - 0.25
    m = (logI >= lo) & (logI <= hi)
    if m.sum() >= 2:
        slope = np.polyfit(vgs[m], logI[m], 1)[0]
    else:   # fall back to the steepest finite-difference slope
        s = np.diff(logI) / np.diff(vgs)
        slope = np.max(s)
    return float(1000.0 / slope)


def extract_metrics(vgs, Id_lin, Id_sat, vd_low, vd_high):
    vth_lin = _vth_constant_current(vgs, Id_lin)
    vth_sat = _vth_constant_current(vgs, Id_sat)
    dibl = (vth_lin - vth_sat) / (vd_high - vd_low) * 1000.0
    ss_sat = _subthreshold_swing(vgs, Id_sat)
    ss_lin = _subthreshold_swing(vgs, Id_lin)
    ion = float(Id_sat[np.argmin(np.abs(vgs - C.VDD))])
    ioff = float(Id_sat[np.argmin(np.abs(vgs - 0.0))])
    return dict(Vth_lin=vth_lin, Vth_sat=vth_sat, DIBL=dibl, SS=ss_sat, SS_lin=ss_lin,
                Ion=ion, Ioff=ioff, log10_ratio=float(np.log10(ion / max(ioff, 1e-30))))
