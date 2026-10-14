"""3-D and 2-D geometric reconstruction of an optimised FinFET.

Given a design vector x = [Lg, Wfin, Hfin, Tox, Lsp, Nch] the module builds an
exact layered model of a tri-gate (3 nm-node) FinFET and renders:
  * a publication-quality 3-D view (substrate/BOX, fin, conformal gate oxide,
    wrap-around gate, spacers, raised source/drain),
  * three labelled 2-D cross-sections (longitudinal, transverse, top),
and validates that every drawn dimension matches the parameter exactly.

Coordinate convention (nm):
  x = along channel (source -> drain / transport direction)
  y = fin-width direction
  z = fin height (vertical), fin base at z = 0.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- fixed visualisation constants (not design variables) ------------------
L_SD = 20.0       # source/drain length each side [nm]
T_GATE = 9.0      # gate-metal visual thickness over the oxide [nm]
H_SUB = 28.0      # buried-oxide / substrate height shown [nm]
Y_MARGIN = 22.0   # substrate half-width beyond the fin [nm]
EPI_OVER = 3.0    # raised S/D epi overgrowth beyond fin faces [nm]

# layer colours
C_SUB = "#c9ccd1"
C_FIN = "#5b8fb9"
C_FIN_SD = "#b8860b"
C_OX = "#e34f4f"
C_GATE = "#7a7f87"
C_SPACER = "#a7d3a6"
C_EPI = "#d4a017"


@dataclass
class Geometry:
    Lg: float; Wfin: float; Hfin: float; Tox: float; Lsp: float; Nch: float
    L_tot: float
    x_src0: float; x_src1: float      # source extent (x)
    x_sp_l0: float; x_sp_l1: float    # left spacer
    x_g0: float; x_g1: float          # gate (channel) extent -> length Lg
    x_sp_r0: float; x_sp_r1: float    # right spacer
    x_drn0: float; x_drn1: float      # drain extent
    y_fin0: float; y_fin1: float      # fin width extent (y)
    z_fin0: float; z_fin1: float      # fin height extent (z)
    # derived gate-stack envelope (transverse)
    y_ox0: float; y_ox1: float
    z_ox1: float
    y_g0: float; y_g1: float
    z_g1: float
    # adaptive (device-scaled) cosmetic dimensions
    L_sd: float; T_gate: float; H_sub: float; Y_margin: float


def compute_geometry(x) -> Geometry:
    Lg, Wfin, Hfin, Tox, Lsp, Nch = [float(v) for v in np.ravel(x)[:6]]
    # adaptive cosmetic scales so the picture is well-proportioned at any node
    T_gate = float(np.clip(1.8 * Lg, 2.0, 9.0))     # gate-metal visual thickness
    L_sd = float(np.clip(2.2 * Lg, 5.0, 20.0))      # source/drain length each side
    H_sub = float(np.clip(0.45 * Hfin, 8.0, 28.0))  # substrate height shown
    # x layout: [source][L-spacer][gate=Lg][R-spacer][drain], gate centred at 0
    x_g0, x_g1 = -Lg / 2.0, Lg / 2.0
    x_sp_l1, x_sp_l0 = x_g0, x_g0 - Lsp
    x_sp_r0, x_sp_r1 = x_g1, x_g1 + Lsp
    x_src1, x_src0 = x_sp_l0, x_sp_l0 - L_sd
    x_drn0, x_drn1 = x_sp_r1, x_sp_r1 + L_sd
    L_tot = x_drn1 - x_src0
    # fin cross-section
    y_fin0, y_fin1 = -Wfin / 2.0, Wfin / 2.0
    z_fin0, z_fin1 = 0.0, Hfin
    # conformal gate oxide envelope (Tox on top + both sides)
    y_ox0, y_ox1 = y_fin0 - Tox, y_fin1 + Tox
    z_ox1 = z_fin1 + Tox
    # wrap-around gate envelope (T_gate over the oxide)
    y_g0, y_g1 = y_ox0 - T_gate, y_ox1 + T_gate
    z_g1 = z_ox1 + T_gate
    Y_margin = float(y_g1 + max(8.0, 0.6 * Wfin + 8.0))   # substrate half-width
    return Geometry(
        Lg, Wfin, Hfin, Tox, Lsp, Nch, L_tot,
        x_src0, x_src1, x_sp_l0, x_sp_l1, x_g0, x_g1, x_sp_r0, x_sp_r1,
        x_drn0, x_drn1, y_fin0, y_fin1, z_fin0, z_fin1,
        y_ox0, y_ox1, z_ox1, y_g0, y_g1, z_g1,
        L_sd, T_gate, H_sub, Y_margin)


# --------------------------------------------------------------------------
# 3-D rendering
# --------------------------------------------------------------------------
def _box_faces(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]


def _add_box(ax, x0, x1, y0, y1, z0, z1, color, alpha=1.0, ec="k", lw=0.3, zorder=1):
    faces = _box_faces(x0, x1, y0, y1, z0, z1)
    pc = Poly3DCollection(faces, facecolors=color, edgecolors=ec,
                          linewidths=lw, alpha=alpha)
    pc.set_zorder(zorder)
    ax.add_collection3d(pc)


def _saddle(ax, xa, xb, g, color, alpha, zorder):
    """Draw a tri-gate saddle (top cap + two side slabs) straddling the fin/oxide."""
    # top cap above the oxide
    _add_box(ax, xa, xb, g.y_g0, g.y_g1, g.z_ox1, g.z_g1, color, alpha, zorder=zorder)
    # left side slab (gate-envelope outer to oxide outer)
    _add_box(ax, xa, xb, g.y_g0, g.y_ox0, 0, g.z_g1, color, alpha, zorder=zorder)
    # right side slab
    _add_box(ax, xa, xb, g.y_ox1, g.y_g1, 0, g.z_g1, color, alpha, zorder=zorder)


def render_3d(x, path=None, title=None, show_gate=True):
    g = compute_geometry(x)
    fig = plt.figure(figsize=(12.5, 9))
    ax = fig.add_subplot(111, projection="3d")

    # substrate / buried oxide
    _add_box(ax, g.x_src0, g.x_drn1, -g.Y_margin, g.Y_margin, -g.H_sub, 0.0,
             C_SUB, alpha=0.45, lw=0.2, zorder=0)

    # fin: source/drain doped segments (gold) + channel segment (blue), solid
    _add_box(ax, g.x_src0, g.x_sp_l0, g.y_fin0, g.y_fin1, 0, g.z_fin1,
             C_FIN_SD, alpha=1.0, zorder=2)            # source
    _add_box(ax, g.x_drn0, g.x_drn1, g.y_fin0, g.y_fin1, 0, g.z_fin1,
             C_FIN_SD, alpha=1.0, zorder=2)            # drain
    _add_box(ax, g.x_sp_l0, g.x_sp_r1, g.y_fin0, g.y_fin1, 0, g.z_fin1,
             C_FIN, alpha=1.0, zorder=3)               # channel + under-spacer fin

    # conformal gate oxide shell over the channel (top cap + two side slabs)
    _add_box(ax, g.x_g0, g.x_g1, g.y_ox0, g.y_ox1, g.z_fin1, g.z_ox1,
             C_OX, alpha=0.9, lw=0.15, zorder=4)        # top
    _add_box(ax, g.x_g0, g.x_g1, g.y_ox0, g.y_fin0, 0, g.z_fin1,
             C_OX, alpha=0.9, lw=0.15, zorder=4)        # left
    _add_box(ax, g.x_g0, g.x_g1, g.y_fin1, g.y_ox1, 0, g.z_fin1,
             C_OX, alpha=0.9, lw=0.15, zorder=4)        # right

    # spacers as saddles flanking the gate (translucent so the fin shows through)
    for (xa, xb) in [(g.x_sp_l0, g.x_sp_l1), (g.x_sp_r0, g.x_sp_r1)]:
        _saddle(ax, xa, xb, g, C_SPACER, 0.40, zorder=5)

    # wrap-around gate metal saddle over the channel
    if show_gate:
        _saddle(ax, g.x_g0, g.x_g1, g, C_GATE, 0.52, zorder=6)

    # cosmetics
    ax.set_xlabel("\nx  [nm]  (source → drain)", fontsize=13, labelpad=12)
    ax.set_ylabel("\ny  [nm]  (fin width)", fontsize=13, labelpad=12)
    ax.set_zlabel("z  [nm]  (height)", fontsize=13, labelpad=8)
    ax.tick_params(labelsize=11)
    ax.set_box_aspect((g.L_tot, 2 * g.Y_margin, g.z_g1 + g.H_sub))
    ax.view_init(elev=24, azim=-60)
    ax.set_zlim(-g.H_sub, g.z_g1 + 4)
    if title is None:
        title = (f"Optimised FinFET 3-D model   "
                 f"$L_g$={g.Lg:.1f}, $W_{{fin}}$={g.Wfin:.2f}, "
                 f"$H_{{fin}}$={g.Hfin:.1f}, $T_{{ox}}$={g.Tox:.2f} nm")
    ax.set_title(title, fontsize=15, pad=16)

    # legend proxies
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C_FIN, label="Si fin (channel)"),
               Patch(facecolor=C_FIN_SD, label="S/D (doped)"),
               Patch(facecolor=C_OX, label=f"gate oxide ($T_{{ox}}$={g.Tox:.2f} nm)"),
               Patch(facecolor=C_GATE, label="wrap-around gate"),
               Patch(facecolor=C_SPACER, label=f"spacer ($L_{{sp}}$={g.Lsp:.1f} nm)"),
               Patch(facecolor=C_SUB, label="substrate / BOX")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.97),
              fontsize=12, framealpha=0.9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# 2-D cross-sections
# --------------------------------------------------------------------------
def _dim_arrow(ax, p0, p1, text, offset=(0, 0), color="k", fs=12):
    a = FancyArrowPatch(p0, p1, arrowstyle="<|-|>", mutation_scale=8,
                        color=color, lw=1.1, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    mx, my = (p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1]
    ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _leader(ax, xy, xytext, text, color="k", fs=12, ha="left"):
    """Label a small feature with a leader line (robust at any scale)."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=color, ha=ha,
                va="center", arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def render_2d_cross_sections(x, path=None):
    g = compute_geometry(x)
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.2))
    for _ax in axes:
        _ax.tick_params(labelsize=12)

    # ---- (A) longitudinal: x-z plane through fin centre ----
    ax = axes[0]
    ax.add_patch(Rectangle((g.x_src0, -g.H_sub), g.L_tot, g.H_sub, fc=C_SUB, ec="k", lw=0.5))
    # fin (side view): S/D doped + channel
    ax.add_patch(Rectangle((g.x_src0, 0), g.x_sp_l0 - g.x_src0, g.Hfin, fc=C_FIN_SD, ec="k", lw=0.5))
    ax.add_patch(Rectangle((g.x_drn0, 0), g.x_drn1 - g.x_drn0, g.Hfin, fc=C_FIN_SD, ec="k", lw=0.5))
    ax.add_patch(Rectangle((g.x_sp_l0, 0), g.x_sp_r1 - g.x_sp_l0, g.Hfin, fc=C_FIN, ec="k", lw=0.5))
    # spacers flank the gate, sitting on the fin top (drawn first, behind gate)
    for (xa, xb) in [(g.x_sp_l0, g.x_sp_l1), (g.x_sp_r0, g.x_sp_r1)]:
        ax.add_patch(Rectangle((xa, g.Hfin), xb - xa, g.z_g1 - g.Hfin,
                               fc=C_SPACER, ec="k", lw=0.4, alpha=0.85))
    # thin conformal gate-oxide cap on top of the silicon channel (Tox)
    ax.add_patch(Rectangle((g.x_g0, g.Hfin), g.Lg, g.Tox, fc=C_OX, ec="darkred", lw=0.5))
    # gate metal cap above the oxide
    ax.add_patch(Rectangle((g.x_g0, g.z_ox1), g.Lg, g.z_g1 - g.z_ox1,
                           fc=C_GATE, ec="k", lw=0.6))
    ax.text(0, g.z_ox1 + (g.z_g1 - g.z_ox1) / 2, "Gate", ha="center", va="center",
            fontsize=12, color="white")
    ax.text(0, g.Hfin / 2, "channel", ha="center", va="center", fontsize=11,
            color="white", rotation=90)
    _leader(ax, (0, g.z_g1), (g.x_g1 + 7, g.z_g1 + 5), f"$L_g$={g.Lg:.2f} nm")
    _dim_arrow(ax, (g.x_sp_r0, -6), (g.x_sp_r1, -6), f"$L_{{sp}}$={g.Lsp:.1f}", offset=(0, -3.5), fs=11)
    _dim_arrow(ax, (g.x_drn1 + 4, 0), (g.x_drn1 + 4, g.Hfin), f"$H_{{fin}}$={g.Hfin:.1f}", offset=(6, 0))
    ax.text(g.x_src0 + (g.x_sp_l0 - g.x_src0) / 2, g.Hfin / 2, "Source", ha="center", va="center", fontsize=12, rotation=90)
    ax.text(g.x_drn0 + (g.x_drn1 - g.x_drn0) / 2, g.Hfin / 2, "Drain", ha="center", va="center", fontsize=12, rotation=90)
    ax.set_title("(A) Longitudinal section (x–z, through fin centre)", fontsize=14)
    ax.set_xlabel("x [nm]", fontsize=13); ax.set_ylabel("z [nm]", fontsize=13)
    ax.set_xlim(g.x_src0 - 8, g.x_drn1 + 18); ax.set_ylim(-g.H_sub - 12, g.z_g1 + 16)
    ax.set_aspect("equal")

    # ---- (B) transverse: y-z plane in the gate region (the wrap-around) ----
    ax = axes[1]
    ax.add_patch(Rectangle((-g.Y_margin, -g.H_sub), 2 * g.Y_margin, g.H_sub, fc=C_SUB, ec="k", lw=0.5))
    # gate metal envelope
    ax.add_patch(Rectangle((g.y_g0, 0), g.y_g1 - g.y_g0, g.z_g1, fc=C_GATE, ec="k", lw=0.6, alpha=0.7))
    # gate oxide envelope
    ax.add_patch(Rectangle((g.y_ox0, 0), g.y_ox1 - g.y_ox0, g.z_ox1, fc=C_OX, ec="k", lw=0.5, alpha=0.95))
    # fin
    ax.add_patch(Rectangle((g.y_fin0, 0), g.Wfin, g.Hfin, fc=C_FIN, ec="k", lw=0.6))
    _leader(ax, (0, g.Hfin * 0.86), (g.y_g0 - 6, g.Hfin * 0.96),
            f"$W_{{fin}}$={g.Wfin:.2f} nm", ha="right")
    _dim_arrow(ax, (g.y_g1 + 5, 0), (g.y_g1 + 5, g.Hfin), f"$H_{{fin}}$={g.Hfin:.1f}", offset=(7, 0))
    # Tox callout (points at the conformal oxide on the fin sidewall)
    ax.annotate(f"$T_{{ox}}$={g.Tox:.2f} nm", xy=(g.y_ox1, g.Hfin * 0.55),
                xytext=(g.y_ox1 + 10, g.Hfin * 0.74), fontsize=12, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.1))
    ax.set_title("(B) Transverse section (y–z, gate wrap-around)", fontsize=14)
    ax.set_xlabel("y [nm]", fontsize=13); ax.set_ylabel("z [nm]", fontsize=13)
    ax.set_xlim(g.y_g0 - 8, g.y_g1 + 20); ax.set_ylim(-g.H_sub - 8, g.z_g1 + 20)
    ax.set_aspect("equal")

    # ---- (C) top view: x-y plane ----
    ax = axes[2]
    ax.add_patch(Rectangle((g.x_src0, -g.Y_margin), g.L_tot, 2 * g.Y_margin, fc=C_SUB, ec="k", lw=0.5))
    # fin (full length)
    ax.add_patch(Rectangle((g.x_src0, g.y_fin0), g.x_sp_l0 - g.x_src0, g.Wfin, fc=C_FIN_SD, ec="k", lw=0.5))
    ax.add_patch(Rectangle((g.x_drn0, g.y_fin0), g.x_drn1 - g.x_drn0, g.Wfin, fc=C_FIN_SD, ec="k", lw=0.5))
    ax.add_patch(Rectangle((g.x_sp_l0, g.y_fin0), g.x_sp_r1 - g.x_sp_l0, g.Wfin, fc=C_FIN, ec="k", lw=0.5))
    # gate crossing the fin (full width incl. wrap)
    ax.add_patch(Rectangle((g.x_g0, g.y_g0), g.Lg, g.y_g1 - g.y_g0, fc=C_GATE, ec="k", lw=0.6, alpha=0.5))
    for (xa, xb) in [(g.x_sp_l0, g.x_sp_l1), (g.x_sp_r0, g.x_sp_r1)]:
        ax.add_patch(Rectangle((xa, g.y_g0), xb - xa, g.y_g1 - g.y_g0, fc=C_SPACER, ec="k", lw=0.4, alpha=0.7))
    _leader(ax, (0, g.y_g1), (g.x_g1 + 4, g.y_g1 + 7), f"$L_g$={g.Lg:.2f} nm")
    _leader(ax, (g.x_drn0, 0), (g.x_drn1 + 3, g.y_fin1 + 7),
            f"$W_{{fin}}$={g.Wfin:.2f} nm", fs=11)
    xs_c = (g.x_src0 + g.x_sp_l0) / 2.0
    xd_c = (g.x_drn0 + g.x_drn1) / 2.0
    ax.text(xs_c, g.y_g1 + 3, "Source", ha="center", fontsize=12)
    ax.text(xd_c, g.y_g1 + 3, "Drain", ha="center", fontsize=12)
    ax.set_title("(C) Top view (x–y plane)", fontsize=14)
    ax.set_xlabel("x [nm]", fontsize=13); ax.set_ylabel("y [nm]", fontsize=13)
    ax.set_xlim(g.x_src0 - 8, g.x_drn1 + 22); ax.set_ylim(-g.Y_margin - 6, g.Y_margin + 14)
    ax.set_aspect("equal")

    fig.suptitle(
        f"Optimised FinFET 2-D cross-sections  "
        f"($L_g$={g.Lg:.2f}, $W_{{fin}}$={g.Wfin:.2f}, $H_{{fin}}$={g.Hfin:.2f}, "
        f"$T_{{ox}}$={g.Tox:.3f}, $L_{{sp}}$={g.Lsp:.2f} nm, "
        f"$N_{{ch}}$={g.Nch:.2e} cm$^{{-3}}$)", fontsize=15.5, y=1.03)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------
# Validation: every drawn dimension must equal the parameter.
# --------------------------------------------------------------------------
def validate_geometry(x, tol=1e-9):
    Lg, Wfin, Hfin, Tox, Lsp, Nch = [float(v) for v in np.ravel(x)[:6]]
    g = compute_geometry(x)
    checks = []

    def chk(name, measured, expected):
        ok = abs(measured - expected) <= tol + 1e-6 * abs(expected)
        checks.append((name, measured, expected, ok))

    chk("gate length Lg (x_g1-x_g0)", g.x_g1 - g.x_g0, Lg)
    chk("fin width Wfin (y_fin1-y_fin0)", g.y_fin1 - g.y_fin0, Wfin)
    chk("fin height Hfin (z_fin1-z_fin0)", g.z_fin1 - g.z_fin0, Hfin)
    chk("oxide thickness Tox (top)", g.z_ox1 - g.z_fin1, Tox)
    chk("oxide thickness Tox (right side)", g.y_ox1 - g.y_fin1, Tox)
    chk("oxide thickness Tox (left side)", g.y_fin0 - g.y_ox0, Tox)
    chk("left spacer width Lsp", g.x_sp_l1 - g.x_sp_l0, Lsp)
    chk("right spacer width Lsp", g.x_sp_r1 - g.x_sp_r0, Lsp)
    chk("aspect ratio Hfin/Wfin", (g.z_fin1 - g.z_fin0) / (g.y_fin1 - g.y_fin0), Hfin / Wfin)
    # symmetry / sanity
    chk("fin centred in y", g.y_fin0 + g.y_fin1, 0.0)
    chk("gate centred in x", g.x_g0 + g.x_g1, 0.0)
    chk("source length = drain length", (g.x_src1 - g.x_src0) - (g.x_drn1 - g.x_drn0), 0.0)
    chk("oxide conformal (sides==top)", (g.y_ox1 - g.y_fin1) - (g.z_ox1 - g.z_fin1), 0.0)
    # no overlap of gate region with S/D
    chk("gate-source gap = Lsp", g.x_g0 - g.x_sp_l0, Lsp)

    all_ok = all(c[3] for c in checks)
    return dict(all_ok=all_ok, checks=checks, geometry=g)


def print_validation(report):
    print(f"{'check':42s} {'measured':>12s} {'expected':>12s}  ok")
    print("-" * 74)
    for name, meas, exp, ok in report["checks"]:
        print(f"{name:42s} {meas:12.6f} {exp:12.6f}  {'PASS' if ok else 'FAIL'}")
    print("-" * 74)
    print("ALL CHECKS PASSED" if report["all_ok"] else "*** VALIDATION FAILED ***")
