"""NSGA-II (Deb et al. 2002).

Used two ways:
* HERA inner acquisition optimiser (Section 7): evolves on the surrogate
  posterior means with constraint-domination, warm-started by the RL controller,
  returning a diverse non-dominated candidate pool.
* Standalone baseline: evolves directly on the true oracle objectives.

All search happens in the normalised unit cube [0,1]^6.  Objectives are passed
as *maximisation* values; the sort negates internally.  Constraint handling uses
Deb's constrained-domination (feasible >- infeasible; among infeasible smaller
total violation dominates).
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def _domination_matrix(F: np.ndarray, viol: np.ndarray) -> np.ndarray:
    """Vectorised constrained-domination matrix dom[p,q] = 'p dominates q'.

    Minimisation of F with Deb constrained-domination:
      feasible dominates infeasible; among infeasible smaller violation
      dominates; among feasible standard Pareto domination.
    """
    feas = viol <= 0
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)   # p<=q all objectives
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)    # p<q some objective
    pareto = le & lt
    both_feas = feas[:, None] & feas[None, :]
    p_feas_q_inf = feas[:, None] & (~feas[None, :])
    both_inf = (~feas[:, None]) & (~feas[None, :])
    viol_less = viol[:, None] < viol[None, :]
    dom = (both_feas & pareto) | p_feas_q_inf | (both_inf & viol_less)
    np.fill_diagonal(dom, False)
    return dom


def _fast_nondominated_sort(F: np.ndarray, viol: np.ndarray):
    """Constrained non-dominated sort (minimisation of F). Returns list of fronts."""
    n = F.shape[0]
    dom = _domination_matrix(F, viol)
    ndom = dom.sum(axis=0).astype(int)        # how many points dominate each q
    assigned = np.zeros(n, bool)
    fronts = []
    cur = np.where(ndom == 0)[0]
    while cur.size:
        fronts.append(list(cur))
        assigned[cur] = True
        ndom = ndom - dom[cur, :].sum(axis=0)
        cur = np.where((ndom == 0) & (~assigned))[0]
    return fronts


def _crowding_distance(F: np.ndarray, front: list) -> np.ndarray:
    """Crowding distance within one front (minimisation F), vectorised over M."""
    l = len(front)
    dist = np.zeros(l)
    if l <= 2:
        dist[:] = np.inf
        return dist
    Ff = F[front]
    for m in range(F.shape[1]):
        order = np.argsort(Ff[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        rng = Ff[order[-1], m] - Ff[order[0], m]
        if rng <= 0:
            continue
        dist[order[1:-1]] += (Ff[order[2:], m] - Ff[order[:-2], m]) / rng
    return dist


def _sbx_crossover(p1, p2, rng, eta=15.0, pc=0.9):
    d = p1.shape[0]
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() <= pc:
        for i in range(d):
            if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-12:
                x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                u = rng.random()
                beta = 1.0 + 2.0 * (x1 - 0.0) / (x2 - x1)
                alpha = 2.0 - beta ** -(eta + 1)
                bq = ((u * alpha) ** (1.0 / (eta + 1)) if u <= 1.0 / alpha
                      else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1)))
                c1[i] = 0.5 * ((x1 + x2) - bq * (x2 - x1))
                beta = 1.0 + 2.0 * (1.0 - x2) / (x2 - x1)
                alpha = 2.0 - beta ** -(eta + 1)
                bq = ((u * alpha) ** (1.0 / (eta + 1)) if u <= 1.0 / alpha
                      else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1)))
                c2[i] = 0.5 * ((x1 + x2) + bq * (x2 - x1))
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def _poly_mutation(x, rng, eta=20.0, pm=None):
    d = x.shape[0]
    if pm is None:
        pm = 1.0 / d
    y = x.copy()
    for i in range(d):
        if rng.random() <= pm:
            u = rng.random()
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta + 1)) - 1.0
            else:
                delta = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))
            y[i] = np.clip(y[i] + delta, 0, 1)
    return y


def _tournament(rank, crowd, rng):
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] < rank[b]:
        return a
    if rank[b] < rank[a]:
        return b
    return a if crowd[a] >= crowd[b] else b


def nsga2(eval_fn: Callable[[np.ndarray], tuple], d: int, pop_size: int,
          n_gen: int, rng: np.random.Generator, init_pop: np.ndarray = None,
          return_pool: int = None):
    """Run NSGA-II.

    Parameters
    ----------
    eval_fn : Z (k,d)->(F (k,M) maximise, viol (k,)) total constraint violation>=0.
    init_pop : optional (m,d) seed individuals (rest filled randomly).
    return_pool : if set, return up to this many best individuals as the pool C.

    Returns
    -------
    dict with 'X' (pop), 'F' (maximise), 'viol', and 'pool' (rank-sorted subset).
    """
    # initial population
    Z = rng.random((pop_size, d))
    if init_pop is not None and len(init_pop):
        k = min(len(init_pop), pop_size)
        Z[:k] = np.clip(init_pop[:k], 0, 1)

    Fmax, viol = eval_fn(Z)
    F = -np.asarray(Fmax, dtype=float)        # internal minimisation
    viol = np.asarray(viol, dtype=float)

    for _ in range(n_gen):
        # ranks + crowding for selection
        fronts = _fast_nondominated_sort(F, viol)
        rank = np.empty(len(Z), dtype=int)
        crowd = np.zeros(len(Z))
        for r, fr in enumerate(fronts):
            for idx in fr:
                rank[idx] = r
            cd = _crowding_distance(F, fr)
            for j, idx in enumerate(fr):
                crowd[idx] = cd[j]
        # offspring
        off = np.empty_like(Z)
        for i in range(0, pop_size, 2):
            p1 = _tournament(rank, crowd, rng)
            p2 = _tournament(rank, crowd, rng)
            c1, c2 = _sbx_crossover(Z[p1], Z[p2], rng)
            off[i] = _poly_mutation(c1, rng)
            if i + 1 < pop_size:
                off[i + 1] = _poly_mutation(c2, rng)
        Foff_max, voff = eval_fn(off)
        Foff = -np.asarray(Foff_max, dtype=float)
        voff = np.asarray(voff, dtype=float)

        # combine + environmental selection (mu+lambda elitism)
        Zc = np.vstack([Z, off])
        Fc = np.vstack([F, Foff])
        vc = np.concatenate([viol, voff])
        fronts = _fast_nondominated_sort(Fc, vc)
        newidx = []
        for fr in fronts:
            if len(newidx) + len(fr) <= pop_size:
                newidx.extend(fr)
            else:
                cd = _crowding_distance(Fc, fr)
                order = np.argsort(-cd)
                need = pop_size - len(newidx)
                newidx.extend([fr[o] for o in order[:need]])
                break
        newidx = np.array(newidx)
        Z, F, viol = Zc[newidx], Fc[newidx], vc[newidx]

    # final ranking for the returned pool
    fronts = _fast_nondominated_sort(F, viol)
    ordered = []
    for fr in fronts:
        cd = _crowding_distance(F, fr)
        order = np.argsort(-cd)
        ordered.extend([fr[o] for o in order])
    ordered = np.array(ordered)
    pool = ordered if return_pool is None else ordered[:return_pool]
    return dict(X=Z, F=-F, viol=viol, pool_idx=pool, pool_X=Z[pool])
