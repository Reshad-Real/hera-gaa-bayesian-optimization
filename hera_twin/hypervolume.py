"""Hypervolume indicator and exact Expected-Hypervolume-Improvement (EHVI).

All objectives are *maximisation* and live in the normalised objective space
(see config.normalize_y), with the reference point r = 0.

Implements:
* pareto_front          - non-dominated subset (Eq. 11-12)
* hypervolume           - HV indicator by dimension sweep (Eq. 13), M=3
* box_decomposition     - exact axis-aligned cell partition of the non-dominated
                          region above the front (the K cells of Eq. 25)
* ehvi                  - closed-form Expected HVI (Eqs. 25-26), vectorised
* greedy_batch          - submodular greedy batch via single-point cEHVI marginal
                          gains (Algorithm 2 / Prop. 1, Eq. 31)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr   # standard-normal CDF (vectorised, fast)

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard-normal PDF."""
    return np.exp(-0.5 * x * x) / _SQRT2PI


def pareto_front(Y: np.ndarray, return_mask: bool = False):
    """Return the non-dominated subset of Y (maximisation).

    A point i is dominated if some j has Y[j] >= Y[i] in all objectives and
    strictly greater in at least one.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n = Y.shape[0]
    if n == 0:
        return (np.zeros(0, bool) if return_mask else Y)
    # Sort by first objective descending for an efficient sweep.
    order = np.argsort(-Y[:, 0], kind="stable")
    Ys = Y[order]
    nd = np.ones(n, bool)
    for i in range(n):
        if not nd[i]:
            continue
        # any later point (smaller/equal y0) that dominates i?
        for j in range(n):
            if i == j or not nd[j]:
                continue
            if np.all(Ys[j] >= Ys[i]) and np.any(Ys[j] > Ys[i]):
                nd[i] = False
                break
    mask = np.zeros(n, bool)
    mask[order[nd]] = True
    if return_mask:
        return mask
    return Y[mask]


def hypervolume(Y: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume of the points Y dominating reference ref (maximisation).

    Supports M = 1, 2, 3 objectives.  For M = 3 uses a slab sweep on the third
    objective with an exact 2-D staircase area per slab (Eq. 13).
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    ref = np.asarray(ref, dtype=float)
    M = Y.shape[1]
    # keep only points strictly above the reference in every objective
    Y = Y[np.all(Y > ref, axis=1)]
    if Y.shape[0] == 0:
        return 0.0
    Y = pareto_front(Y)
    if M == 1:
        return float(np.max(Y[:, 0]) - ref[0])
    if M == 2:
        return _hv2d(Y, ref)
    if M == 3:
        # sort by third objective descending; sweep slabs
        order = np.argsort(-Y[:, 2])
        Ys = Y[order]
        zlevels = Ys[:, 2]
        vol = 0.0
        for k in range(len(Ys)):
            z_high = zlevels[k]
            z_low = zlevels[k + 1] if k + 1 < len(Ys) else ref[2]
            if z_high <= z_low:
                continue
            active = Ys[: k + 1, :2]               # points with y3 >= z_high
            area = _hv2d(active, ref[:2])
            vol += area * (z_high - z_low)
        return float(vol)
    raise NotImplementedError("hypervolume implemented for M<=3")


def _hv2d(Y2: np.ndarray, ref2: np.ndarray) -> float:
    """2-D staircase hypervolume (maximisation), rectangles anchored at ref2."""
    Y2 = Y2[np.all(Y2 > ref2, axis=1)]
    if Y2.shape[0] == 0:
        return 0.0
    # non-dominated in 2-D: sort by y1 descending, keep strictly increasing y2
    order = np.argsort(-Y2[:, 0], kind="stable")
    Ys = Y2[order]
    area = 0.0
    prev_x = Ys[0, 0]            # not used directly; staircase below
    best_y2 = ref2[1]
    x_right = Ys[0, 0]
    # Build the upper staircase: iterate points by descending y1, track max y2.
    # Width of each step = (current y1) - (next y1), height = (max y2 so far)-ref.
    nd_pts = []
    cur_max_y2 = -np.inf
    for i in range(len(Ys)):
        if Ys[i, 1] > cur_max_y2:
            nd_pts.append(Ys[i])
            cur_max_y2 = Ys[i, 1]
    nd = np.array(nd_pts)        # sorted y1 desc, y2 asc
    x_prev = ref2[0]
    # integrate as union of rectangles
    # sort nd by y1 ascending for a clean sweep
    nd = nd[np.argsort(nd[:, 0])]
    prev_y1 = ref2[0]
    # sweep from high y2/low y1 ... use the standard formula:
    # sort by y1 descending; area = sum (y1_i - y1_{i+1})*(y2_i - ref_y2)
    nd_desc = nd[np.argsort(-nd[:, 0])]
    for i in range(len(nd_desc)):
        x_i = nd_desc[i, 0]
        x_next = nd_desc[i + 1, 0] if i + 1 < len(nd_desc) else ref2[0]
        area += (x_i - x_next) * (nd_desc[i, 1] - ref2[1])
    return float(area)


_ACQ_FRONT_CAP = 18   # max front points used for EHVI cell decomposition (acquisition only)


def _subsample_front(P, cap):
    """Keep at most `cap` front points for cell decomposition (acquisition only).

    Always retains the per-objective extreme points; fills the rest with a
    crowding-spread subsample.  Only affects the EHVI acquisition, never the
    reported hypervolume (hypervolume() uses the full front)."""
    n, M = P.shape
    if n <= cap:
        return P
    keep = set(int(np.argmax(P[:, m])) for m in range(M))
    keep |= set(int(np.argmin(P[:, m])) for m in range(M))
    remaining = [i for i in range(n) if i not in keep]
    rng = np.random.default_rng(0)
    extra = cap - len(keep)
    if extra > 0 and remaining:
        # spread by distance along the first objective
        order = sorted(remaining, key=lambda i: P[i, 0])
        idx = np.linspace(0, len(order) - 1, min(extra, len(order))).astype(int)
        keep |= set(order[j] for j in idx)
    return P[sorted(keep)]


def box_decomposition(P: np.ndarray, ref: np.ndarray, upper: np.ndarray):
    """Exact axis-aligned cell partition of the non-dominated region.

    Returns (L, U): arrays of shape (K, M) giving the lower and upper corners of
    the K cells that tile the region in [ref, upper] not yet dominated by the
    current front P.  These are the cells over which EHVI integrates (Eq. 25).

    For tractability the front is capped at _ACQ_FRONT_CAP points (cells grow as
    (|P|+1)^M); this affects only the acquisition surface, not the reported HV.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    ref = np.asarray(ref, dtype=float)
    upper = np.asarray(upper, dtype=float)
    M = ref.shape[0]
    if P.shape[0] == 0:
        return ref.reshape(1, M).copy(), upper.reshape(1, M).copy()
    P = pareto_front(P)
    P = _subsample_front(P, _ACQ_FRONT_CAP)

    # per-axis grid breakpoints: ref, each front coord, upper
    grids = []
    for m in range(M):
        g = np.unique(np.concatenate([[ref[m]], P[:, m], [upper[m]]]))
        g = g[(g >= ref[m] - 1e-12) & (g <= upper[m] + 1e-12)]
        grids.append(g)

    # enumerate cells via meshgrid of interval indices
    idx_ranges = [np.arange(len(g) - 1) for g in grids]
    mesh = np.meshgrid(*idx_ranges, indexing="ij")
    cells_idx = np.stack([m.ravel() for m in mesh], axis=1)   # (K, M)

    L = np.empty((cells_idx.shape[0], M))
    U = np.empty((cells_idx.shape[0], M))
    for m in range(M):
        L[:, m] = grids[m][cells_idx[:, m]]
        U[:, m] = grids[m][cells_idx[:, m] + 1]

    # keep cells NOT dominated by any front point: a cell [l,u] is already
    # dominated iff some p in P has p >= u (componentwise).  Vectorised test.
    # dominated_k = exists p : all_m P[p,m] >= U[k,m]
    # compute via broadcasting in chunks to bound memory
    keep = np.ones(L.shape[0], bool)
    Pp = P
    chunk = 4096
    for s in range(0, L.shape[0], chunk):
        u = U[s:s + chunk]                                  # (c, M)
        # (c, nP): is u dominated by p?
        dom = np.all(Pp[None, :, :] >= u[:, None, :] - 1e-12, axis=2).any(axis=1)
        keep[s:s + chunk] = ~dom
    return L[keep], U[keep]


def _cell_integral(l, u, mu, sigma):
    """One-dimensional cell integral I(l,u,mu,sigma)  (Eq. 26).

    I = E[(min(u,t) - l)_+] for t ~ N(mu, sigma^2).  All inputs broadcast.
    """
    sigma = np.maximum(sigma, 1e-9)
    alpha = (u - mu) / sigma
    beta = (l - mu) / sigma
    return ((mu - l) * (ndtr(alpha) - ndtr(beta))
            + sigma * (_phi(beta) - _phi(alpha))
            + (u - l) * (1.0 - ndtr(alpha)))


def ehvi(mu: np.ndarray, sigma: np.ndarray, L: np.ndarray, U: np.ndarray,
         chunk: int = 64) -> np.ndarray:
    """Closed-form Expected Hypervolume Improvement (Eqs. 25-26), vectorised.

    Parameters
    ----------
    mu, sigma : (B, M) posterior mean / std of candidate objective vectors.
    L, U      : (K, M) cell lower/upper corners from box_decomposition.

    Returns
    -------
    (B,) array of EHVI values.
    """
    mu = np.atleast_2d(mu)
    sigma = np.atleast_2d(sigma)
    B, M = mu.shape
    out = np.empty(B)
    for s in range(0, B, chunk):
        m = mu[s:s + chunk][:, None, :]          # (c,1,M)
        sg = sigma[s:s + chunk][:, None, :]       # (c,1,M)
        I = _cell_integral(L[None], U[None], m, sg)   # (c,K,M)
        contrib = np.prod(I, axis=2)              # (c,K)
        out[s:s + chunk] = contrib.sum(axis=1)
    return out


def greedy_batch(cand_mu, cand_sigma, cand_pfeas, P, ref, upper, q):
    """Submodular greedy batch selection (Algorithm 2, Prop. 1, Eq. 31).

    Each marginal gain is a single-point constrained-EHVI evaluation against the
    front augmented with the already-selected points' posterior means.

    Returns the indices (into the candidate pool) of the selected batch.
    """
    cand_mu = np.atleast_2d(cand_mu)
    cand_sigma = np.atleast_2d(cand_sigma)
    cand_pfeas = np.asarray(cand_pfeas, dtype=float)
    n = cand_mu.shape[0]
    selected = []
    Pcur = np.atleast_2d(P).copy() if len(P) else np.zeros((0, ref.shape[0]))

    for _ in range(min(q, n)):
        L, U = box_decomposition(Pcur, ref, upper)
        vals = ehvi(cand_mu, cand_sigma, L, U) * cand_pfeas
        vals[selected] = -np.inf
        j = int(np.argmax(vals))
        if not np.isfinite(vals[j]) or vals[j] <= 0:
            # no further positive marginal gain; still fill batch with best raw
            remaining = [k for k in range(n) if k not in selected]
            if not remaining:
                break
            j = remaining[int(np.argmax(cand_pfeas[remaining]))]
        selected.append(j)
        # augment front with this candidate's predicted mean
        Pcur = np.vstack([Pcur, cand_mu[j]]) if Pcur.size else cand_mu[j:j + 1]
        Pcur = pareto_front(Pcur)
    return selected


def single_cehvi(mu, sigma, pfeas, P, ref, upper):
    """Single-point constrained EHVI alpha_cEHVI({x}|P)  (Eq. 30)."""
    L, U = box_decomposition(np.atleast_2d(P), ref, upper)
    return ehvi(mu, sigma, L, U) * np.asarray(pfeas, dtype=float)
