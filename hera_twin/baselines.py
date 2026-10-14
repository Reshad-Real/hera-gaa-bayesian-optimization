"""Literature baselines for the comparative study.

All methods share the same n0=10 feasible initial design per seed and the same
oracle, so the comparison isolates the optimiser.  Each returns an OptResult with
an HV-vs-evaluation history in the normalised objective space.

Methods
-------
random  : quasi-random (Sobol) space-filling search.
nsga2   : NSGA-II run directly on the oracle (Deb et al. 2002).
qparego : random-scalarisation Bayesian optimisation (ParEGO, Knowles 2006).
qehvi   : hypervolume Bayesian optimisation with EHVI over a candidate pool
          (Daulton et al. 2020/21) - the GP+EHVI core without HERA's RL
          controller, hierarchy, or NSGA-II inner optimiser.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc
from scipy.special import ndtr

from . import config as C
from . import finfet_oracle as O
from . import hypervolume as H
from . import gp_surrogate as G
from . import nsga2
from .hera import OptResult, REF, UPPER, sobol_feasible

_SQRT2PI = np.sqrt(2 * np.pi)


def _phi(x):
    return np.exp(-0.5 * x * x) / _SQRT2PI


def initial_design(seed: int, n0: int = None):
    """Shared n0 feasible Sobol initial designs for a given seed."""
    n0 = n0 or C.HP["n0"]
    return sobol_feasible(n0, np.random.default_rng(seed))


def _hv_feas(Yraw, feas):
    Yraw = np.asarray(Yraw); feas = np.asarray(feas)
    if not feas.any():
        return 0.0
    return H.hypervolume(C.normalize_y(Yraw[feas]), REF)


def _finalize(name, X, Yraw, feas, hv_hist, n0, Xinit):
    X = np.array(X); Yraw = np.array(Yraw); feas = np.array(feas)
    if feas.any():
        pm = H.pareto_front(C.normalize_y(Yraw[feas]), return_mask=True)
        pX, pY = X[feas][pm], Yraw[feas][pm]
    else:
        pX, pY = np.zeros((0, 6)), np.zeros((0, 3))
    return OptResult(name=name, X=X, Y=Yraw, feas=feas, hv_history=np.array(hv_hist),
                     n_init=n0, pareto_X=pX, pareto_Y=pY,
                     init_X=Xinit, init_Y=O.objectives(Xinit))


def _candidate_pool(seed, rng, best_Xn, n_sobol=2048, n_pert=512):
    """Sobol cover of the box plus local perturbations of the best designs."""
    sob = qmc.Sobol(d=6, seed=int(rng.integers(1 << 30)))
    Z = sob.random(n_sobol)
    if best_Xn is not None and len(best_Xn):
        reps = int(np.ceil(n_pert / len(best_Xn)))
        base = np.tile(best_Xn, (reps, 1))[:n_pert]
        pert = np.clip(base + 0.05 * rng.standard_normal(base.shape), 0, 1)
        Z = np.vstack([Z, pert])
    return Z


def _init_state(seed, n0, Xinit):
    X = list(Xinit)
    Yraw = list(O.objectives(Xinit))
    Ccon = list(O.constraints(Xinit))
    feas = list(O.is_feasible(Xinit))
    hv = [_hv_feas(np.array(Yraw[:i + 1]), np.array(feas[:i + 1])) for i in range(len(X))]
    return X, Yraw, Ccon, feas, hv


# --------------------------------------------------------------------------
def random_search(budget, seed, Xinit=None):
    n0 = C.HP["n0"]
    rng = np.random.default_rng(seed + 1000)
    if Xinit is None:
        Xinit = initial_design(seed)
    X, Yraw, Ccon, feas, hv = _init_state(seed, n0, Xinit)
    sob = qmc.Sobol(d=6, seed=int(rng.integers(1 << 30)))
    while len(X) < budget:
        z = sob.random(1)
        xp = C.denormalize_x(z)[0]
        X.append(xp)
        Yraw.append(O.objectives(xp[None])[0])
        feas.append(bool(O.is_feasible(xp[None])[0]))
        hv.append(_hv_feas(np.array(Yraw), np.array(feas)))
    return _finalize("Random", X, Yraw, feas, hv, n0, Xinit)


# --------------------------------------------------------------------------
def nsga2_baseline(budget, seed, Xinit=None, pop=20):
    """Evaluation-budgeted NSGA-II on the true oracle; HV logged per eval."""
    n0 = C.HP["n0"]
    rng = np.random.default_rng(seed + 2000)
    if Xinit is None:
        Xinit = initial_design(seed)
    X, Yraw, Ccon, feas, hv = _init_state(seed, n0, Xinit)

    # custom generational loop so we can log HV after each oracle evaluation
    Zinit = C.normalize_x(Xinit)
    Z = rng.random((pop, 6))
    Z[:min(pop, len(Zinit))] = Zinit[:pop]

    def evaluate(Zb, skip=0):
        # the first `skip` rows are the shared initial designs, already evaluated
        # and logged in X: reuse them without spending budget again.
        Xp = C.denormalize_x(Zb)
        Yn = C.normalize_y(O.objectives(Xp))
        viol = O.total_violation(Xp)
        # log each new evaluation
        for k in range(skip, len(Zb)):
            if len(X) >= budget:
                break
            X.append(Xp[k]); Yraw.append(O.objectives(Xp[k][None])[0])
            feas.append(bool(viol[k] <= 0))
            hv.append(_hv_feas(np.array(Yraw), np.array(feas)))
        return Yn, viol

    Fmax, viol = evaluate(Z, skip=min(pop, len(Zinit)))
    F = -Fmax
    while len(X) < budget:
        fronts = nsga2._fast_nondominated_sort(F, viol)
        rank = np.empty(len(Z), int); crowd = np.zeros(len(Z))
        for r, fr in enumerate(fronts):
            for idx in fr:
                rank[idx] = r
            cd = nsga2._crowding_distance(F, fr)
            for j, idx in enumerate(fr):
                crowd[idx] = cd[j]
        off = np.empty_like(Z)
        for i in range(0, pop, 2):
            p1 = nsga2._tournament(rank, crowd, rng)
            p2 = nsga2._tournament(rank, crowd, rng)
            c1, c2 = nsga2._sbx_crossover(Z[p1], Z[p2], rng)
            off[i] = nsga2._poly_mutation(c1, rng)
            if i + 1 < pop:
                off[i + 1] = nsga2._poly_mutation(c2, rng)
        Foff_max, voff = evaluate(off)
        Zc = np.vstack([Z, off]); Fc = np.vstack([F, -Foff_max])
        vc = np.concatenate([viol, voff])
        fronts = nsga2._fast_nondominated_sort(Fc, vc)
        newidx = []
        for fr in fronts:
            if len(newidx) + len(fr) <= pop:
                newidx.extend(fr)
            else:
                cd = nsga2._crowding_distance(Fc, fr)
                order = np.argsort(-cd)
                newidx.extend([fr[o] for o in order[:pop - len(newidx)]])
                break
        newidx = np.array(newidx)
        Z, F, viol = Zc[newidx], Fc[newidx], vc[newidx]
    return _finalize("NSGA-II", X[:budget], Yraw[:budget], feas[:budget], hv[:budget], n0, Xinit)


# --------------------------------------------------------------------------
def _fit_scalar_gp(Xn, y):
    return G._GPSingle(seed=0).fit(Xn, y)


def _constrained_ei(mu, sigma, fbest, pfeas, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - fbest - xi) / sigma
    ei = (mu - fbest - xi) * ndtr(z) + sigma * _phi(z)
    return np.maximum(ei, 0.0) * pfeas


def qparego(budget, seed, Xinit=None):
    """ParEGO: random augmented-Chebyshev scalarisation + EI (sequential)."""
    n0 = C.HP["n0"]
    rng = np.random.default_rng(seed + 3000)
    if Xinit is None:
        Xinit = initial_design(seed)
    X, Yraw, Ccon, feas, hv = _init_state(seed, n0, Xinit)
    while len(X) < budget:
        Xn = C.normalize_x(np.array(X))
        Yn = C.normalize_y(np.array(Yraw))
        w = rng.random(3); w /= w.sum()
        # augmented Tchebycheff (maximisation)
        scal = np.min(w * Yn, axis=1) + 0.05 * np.sum(w * Yn, axis=1)
        gp = _fit_scalar_gp(Xn, scal)
        con = G.Surrogate(E=1, seed=seed)  # constraint GPs only
        con.con_models = [G._GPSingle(seed + l).fit(Xn, np.array(Ccon)[:, l])
                          for l in range(C.N_CONSTRAINTS)]
        best_feas_Xn = Xn[np.array(feas)] if np.array(feas).any() else Xn
        pool = _candidate_pool(seed, rng, best_feas_Xn)
        mu, sg = gp.predict(pool)
        pf, _, _ = con.predict_feasibility(pool)
        fbest = scal.max()
        ei = _constrained_ei(mu, sg, fbest, pf)
        j = int(np.argmax(ei))
        xp = C.denormalize_x(pool[j][None])[0]
        X.append(xp); Yraw.append(O.objectives(xp[None])[0])
        Ccon.append(O.constraints(xp[None])[0])
        feas.append(bool(O.is_feasible(xp[None])[0]))
        hv.append(_hv_feas(np.array(Yraw), np.array(feas)))
    return _finalize("qParEGO", X, Yraw, feas, hv, n0, Xinit)


# --------------------------------------------------------------------------
def qehvi(budget, seed, Xinit=None, q=4):
    """Hypervolume BO with EHVI over a candidate pool (no RL / hierarchy / NSGA)."""
    n0 = C.HP["n0"]
    rng = np.random.default_rng(seed + 4000)
    if Xinit is None:
        Xinit = initial_design(seed)
    X, Yraw, Ccon, feas, hv = _init_state(seed, n0, Xinit)
    while len(X) < budget:
        Xn = C.normalize_x(np.array(X))
        surr = G.Surrogate(E=C.HP["ensemble_size"], seed=seed).fit(
            Xn, np.array(Yraw), np.array(Ccon))
        Yn = C.normalize_y(np.array(Yraw))
        f = np.array(feas)
        P = H.pareto_front(Yn[f]) if f.any() else np.zeros((0, 3))
        best_feas_Xn = Xn[f] if f.any() else Xn
        pool = _candidate_pool(seed, rng, best_feas_Xn, n_sobol=512, n_pert=128)
        mu, sg, _ = surr.predict_objectives(pool)
        pf, _, _ = surr.predict_feasibility(pool)
        sel = H.greedy_batch(mu, sg, pf, P, REF, UPPER, q)
        for j in sel:
            if len(X) >= budget:
                break
            xp = C.denormalize_x(pool[j][None])[0]
            X.append(xp); Yraw.append(O.objectives(xp[None])[0])
            Ccon.append(O.constraints(xp[None])[0])
            feas.append(bool(O.is_feasible(xp[None])[0]))
            hv.append(_hv_feas(np.array(Yraw), np.array(feas)))
    return _finalize("qEHVI", X, Yraw, feas, hv, n0, Xinit)


# --------------------------------------------------------------------------
def qehvi_lbfgs(budget, seed, Xinit=None, q=4, n_raw=512, n_restarts=10, maxiter=50):
    """q-EHVI with a gradient-based acquisition optimiser (BoTorch-style).

    Same surrogate, constrained EHVI and Kriging-believer sequential-greedy batch
    as `qehvi`, but each batch point maximises the single-point cEHVI over the
    continuous box with multi-start L-BFGS-B (finite-difference gradients),
    started from the `n_restarts` best of `n_raw` raw samples (Sobol cover plus
    perturbations of the feasible designs). As in the pool-based version, a design
    is not selected twice: candidates closer than `min_dist` (normalised units) to
    an evaluated or already selected design are skipped, and the next-best local
    optimum (or raw sample) is taken instead.
    """
    n0 = C.HP["n0"]
    rng = np.random.default_rng(seed + 6000)
    if Xinit is None:
        Xinit = initial_design(seed)
    X, Yraw, Ccon, feas, hv = _init_state(seed, n0, Xinit)
    while len(X) < budget:
        Xn = C.normalize_x(np.array(X))
        surr = G.Surrogate(E=C.HP["ensemble_size"], seed=seed).fit(
            Xn, np.array(Yraw), np.array(Ccon))
        Yn = C.normalize_y(np.array(Yraw))
        f = np.array(feas)
        P = H.pareto_front(Yn[f]) if f.any() else np.zeros((0, 3))
        best_feas_Xn = Xn[f] if f.any() else Xn
        raw = _candidate_pool(seed, rng, best_feas_Xn, n_sobol=n_raw, n_pert=128)
        chosen = lbfgs_greedy_batch(surr, P, raw, Xn, q, n_restarts, maxiter)
        for z in chosen:
            if len(X) >= budget:
                break
            xp = C.denormalize_x(z[None])[0]
            X.append(xp); Yraw.append(O.objectives(xp[None])[0])
            Ccon.append(O.constraints(xp[None])[0])
            feas.append(bool(O.is_feasible(xp[None])[0]))
            hv.append(_hv_feas(np.array(Yraw), np.array(feas)))
    return _finalize("qEHVI-LBFGS", X, Yraw, feas, hv, n0, Xinit)


def lbfgs_greedy_batch(surr, P, raw, Xn_eval, q, n_restarts=10, maxiter=50, min_dist=0.02):
    """Sequential-greedy cEHVI batch with a multi-start L-BFGS-B acquisition optimiser.

    For each batch slot, the single-point cEHVI against the current (believer-augmented)
    front is maximised from the `n_restarts` best raw samples. The best local optimum
    that is at least `min_dist` away from every evaluated and already selected design
    is chosen; if none qualifies, the best qualifying raw sample is used.
    """
    from scipy.optimize import minimize
    d = raw.shape[1]
    bounds = [(0.0, 1.0)] * d
    Pcur = np.atleast_2d(P).copy() if len(P) else np.zeros((0, REF.shape[0]))
    taken = [np.atleast_2d(Xn_eval)]
    chosen = []
    for _ in range(q):
        L, U = H.box_decomposition(Pcur, REF, UPPER)

        def acq(Z):
            mu, sg, _ = surr.predict_objectives(Z)
            pf, _, _ = surr.predict_feasibility(Z)
            return H.ehvi(mu, sg, L, U) * pf

        vals = acq(raw)
        order = np.argsort(-vals)
        cands = []
        for z0 in raw[order[:n_restarts]]:
            r = minimize(lambda z: -acq(z[None])[0], z0, method="L-BFGS-B",
                         bounds=bounds, options=dict(maxiter=maxiter))
            if np.isfinite(r.fun):
                cands.append((-float(r.fun), np.clip(r.x, 0.0, 1.0)))
        cands.sort(key=lambda t: -t[0])
        cands += [(float(vals[j]), raw[j]) for j in order]        # fallback: raw samples
        T = np.vstack(taken + ([np.array(chosen)] if chosen else []))
        pick = next((z for _, z in cands if np.min(np.linalg.norm(T - z, axis=1)) >= min_dist),
                    cands[0][1])
        chosen.append(pick)
        mu_b, _, _ = surr.predict_objectives(pick[None])
        Pcur = H.pareto_front(np.vstack([Pcur, mu_b])) if Pcur.size else mu_b
    return chosen


METHODS = {
    "HERA": None,   # handled by hera.HERA
    "qEHVI": qehvi,
    "qParEGO": qparego,
    "NSGA-II": nsga2_baseline,
    "Random": random_search,
}
