"""Standard multi-objective benchmark problems and a self-contained harness.

The FinFET oracle is a bespoke problem, so a reviewer cannot separate "is the
optimiser good?" from "is that particular oracle easy?".  This module removes
that ambiguity: it runs the *identical* HERA algorithm (Algorithm 1) and the
general-purpose baselines on community-standard problems whose Pareto fronts are
known in closed form, and reports two metrics anyone can reproduce.

Problems (DTLZ family; Deb, Thiele, Laumanns & Zitzler 2005, and the constrained
C2-DTLZ2 of Jain & Deb 2014) are posed at d = 6 variables, M = 3 objectives on
the unit cube - the same structural shape as the FinFET case study - so every
component (GP-ensemble surrogate, constrained q-EHVI acquisition, tier-factored
PPO controller, NSGA-II inner optimiser) runs unchanged.

Conventions
-----------
* Objectives are returned as *maximisation* (the DTLZ minimisation objectives
  negated), matching the rest of the package.
* Constraints are <= 0 -feasible.  Unconstrained problems carry one smooth,
  always-satisfied constraint so the surrogate/feasibility path is exercised
  identically to the FinFET problem (no special-casing in the optimiser).
* Metrics: normalised hypervolume HV (reference point r = 0, front nadir at 0,
  ideal at 1) and modified inverted generational distance IGD+ (Ishibuchi 2015)
  to the analytic Pareto front, computed in the raw minimisation space.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from . import config as C
from . import hypervolume as H
from . import gp_surrogate as G
from . import nsga2
from . import rl_controller as RL

DIM = 6
M = 3
K = DIM - M + 1                      # 4 distance variables
REF = np.zeros(M)                    # r = 0 in the normalised objective space
UPPER = np.full(M, 1.15)            # slight headroom above the ideal for EHVI cells


# ==========================================================================
# DTLZ objective functions (minimisation), M = 3, d = 6.
# ==========================================================================
def _g_dtlz1(dist):
    return 100.0 * (dist.shape[1] + np.sum((dist - 0.5) ** 2
                    - np.cos(20.0 * np.pi * (dist - 0.5)), axis=1))


def _g_dtlz2(dist):
    return np.sum((dist - 0.5) ** 2, axis=1)


def dtlz1_f(X):
    X = np.atleast_2d(X)
    pos, dist = X[:, :M - 1], X[:, M - 1:]
    g = _g_dtlz1(dist)
    f = np.empty((X.shape[0], M))
    f[:, 0] = 0.5 * (1 + g) * pos[:, 0] * pos[:, 1]
    f[:, 1] = 0.5 * (1 + g) * pos[:, 0] * (1 - pos[:, 1])
    f[:, 2] = 0.5 * (1 + g) * (1 - pos[:, 0])
    return f


def dtlz2_f(X):
    X = np.atleast_2d(X)
    pos, dist = X[:, :M - 1], X[:, M - 1:]
    g = _g_dtlz2(dist)
    t = pos * (np.pi / 2.0)
    f = np.empty((X.shape[0], M))
    f[:, 0] = (1 + g) * np.cos(t[:, 0]) * np.cos(t[:, 1])
    f[:, 1] = (1 + g) * np.cos(t[:, 0]) * np.sin(t[:, 1])
    f[:, 2] = (1 + g) * np.sin(t[:, 0])
    return f


def dtlz4_f(X, alpha=100.0):
    """DTLZ4: same spherical Pareto front as DTLZ2 but with a biased position
    mapping x -> x^alpha that clusters solutions, stressing spread/diversity."""
    X = np.atleast_2d(X)
    pos, dist = X[:, :M - 1] ** alpha, X[:, M - 1:]
    g = _g_dtlz2(dist)
    t = pos * (np.pi / 2.0)
    f = np.empty((X.shape[0], M))
    f[:, 0] = (1 + g) * np.cos(t[:, 0]) * np.cos(t[:, 1])
    f[:, 1] = (1 + g) * np.cos(t[:, 0]) * np.sin(t[:, 1])
    f[:, 2] = (1 + g) * np.sin(t[:, 0])
    return f


def _c2_constraint(F, r=0.4):
    """C2-DTLZ2 feasibility functional (Jain & Deb 2014); <= 0 is feasible."""
    F = np.atleast_2d(F)
    sumsq = np.sum(F ** 2, axis=1, keepdims=True)
    term_i = (F - 1.0) ** 2 + (sumsq - F ** 2) - r ** 2      # (n, M)
    t1 = np.min(term_i, axis=1)
    t2 = np.sum((F - 1.0 / np.sqrt(M)) ** 2, axis=1) - r ** 2
    return np.minimum(t1, t2)                                 # <= 0 feasible


class Problem:
    """A d=6, M=3 benchmark posed on the unit cube (maximisation objectives)."""

    def __init__(self, name, f_fn, obj_min, obj_max, constrained=False, r=0.4):
        self.name = name
        self.f_fn = f_fn
        self.dim = DIM
        self.n_obj = M
        self.constrained = constrained
        self.r = r
        self.obj_min = np.asarray(obj_min, float)
        self.obj_max = np.asarray(obj_max, float)
        self.n_con = 1                       # always one constraint column

    # -- oracle interface (mirrors finfet_oracle) -------------------------
    def objectives(self, X01):
        return -self.f_fn(X01)               # maximisation

    def constraints(self, X01):
        X01 = np.atleast_2d(X01)
        if self.constrained:
            c = _c2_constraint(self.f_fn(X01), self.r)
        else:
            # smooth, always-satisfied constraint (keeps the feasibility path
            # identical to the constrained case; contributes no active bound).
            c = 0.5 * np.sum((X01 - 0.5) ** 2, axis=1) - 100.0
        return c[:, None]

    def is_feasible(self, X01):
        return np.all(self.constraints(X01) <= 0, axis=1)

    def total_violation(self, X01):
        return np.sum(np.clip(self.constraints(X01), 0.0, None), axis=1)

    # -- analytic Pareto front (raw minimisation space) for IGD+ -----------
    def pareto_reference(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        if self.f_fn is dtlz1_f:
            w = rng.dirichlet(np.ones(M), size=n) * 0.5      # simplex, sum = 0.5
            return w
        # DTLZ2 / C2-DTLZ2: positive unit sphere
        v = np.abs(rng.standard_normal((n * 3, M)))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        if self.constrained:
            v = v[_c2_constraint(v, self.r) <= 1e-9]         # keep feasible arc
        return v[:n]


def make_problems():
    return [
        Problem("DTLZ2", dtlz2_f, obj_min=[-1.0] * M, obj_max=[0.0] * M),
        Problem("DTLZ4", dtlz4_f, obj_min=[-1.0] * M, obj_max=[0.0] * M),
        Problem("C2-DTLZ2", dtlz2_f, obj_min=[-1.0] * M, obj_max=[0.0] * M,
                constrained=True, r=0.4),
    ]


# ==========================================================================
# Metrics.
# ==========================================================================
def igd_plus(ref, A):
    """Modified IGD+ (Ishibuchi 2015), minimisation; ref, A in raw f-space."""
    ref = np.atleast_2d(ref)
    A = np.atleast_2d(A)
    if A.shape[0] == 0:
        return float("nan")
    d = 0.0
    for z in ref:
        gap = np.maximum(A - z, 0.0)                         # (|A|, M)
        d += np.min(np.sqrt(np.sum(gap ** 2, axis=1)))
    return float(d / len(ref))


def _hv_norm(Yraw_feas):
    if len(Yraw_feas) == 0:
        return 0.0
    return H.hypervolume(C.normalize_y(np.atleast_2d(Yraw_feas)), REF)


# ==========================================================================
# Shared harness.
# ==========================================================================
def _set_frame(problem):
    """Install the problem's objective-normalisation frame (HV reference)."""
    C.set_reference(problem.obj_min, problem.obj_max)


def sobol_feasible(problem, n0, rng, max_tries=200000):
    sob = qmc.Sobol(d=problem.dim, seed=int(rng.integers(1 << 30)))
    X, tries = [], 0
    while len(X) < n0 and tries < max_tries:
        Z = sob.random(256)
        feas = problem.is_feasible(Z)
        for k in np.where(feas)[0]:
            X.append(Z[k])
            if len(X) >= n0:
                break
        tries += 256
    return np.array(X[:n0])


def _pareto_feas(X, Yraw, feas):
    X, Yraw, feas = np.array(X), np.array(Yraw), np.array(feas)
    if not feas.any():
        return np.zeros((0, DIM)), np.zeros((0, M))
    pm = H.pareto_front(C.normalize_y(Yraw[feas]), return_mask=True)
    return X[feas][pm], Yraw[feas][pm]


def _record(problem, X, Yraw, feas, hv_hist):
    pX, pY = _pareto_feas(X, Yraw, feas)
    igd = igd_plus(problem.pareto_reference(), -pY) if len(pY) else float("nan")
    return dict(name=problem.name, hv_history=np.asarray(hv_hist, float),
                final_hv=float(hv_hist[-1]), final_igd=igd,
                pareto_Y=pY, n_eval=len(X))


def _init_state(problem, Xinit):
    X = list(Xinit)
    Yraw = list(problem.objectives(Xinit))
    Ccon = list(problem.constraints(Xinit))
    feas = list(problem.is_feasible(Xinit))
    hv = [_hv_norm(np.array(Yraw[:i + 1])[np.array(feas[:i + 1])]) for i in range(len(X))]
    return X, Yraw, Ccon, feas, hv


def _candidate_pool(rng, best, n_sobol=2048, n_pert=512):
    sob = qmc.Sobol(d=DIM, seed=int(rng.integers(1 << 30)))
    Z = sob.random(n_sobol)
    if best is not None and len(best):
        reps = int(np.ceil(n_pert / len(best)))
        base = np.tile(best, (reps, 1))[:n_pert]
        Z = np.vstack([Z, np.clip(base + 0.05 * rng.standard_normal(base.shape), 0, 1)])
    return Z


# ==========================================================================
# HERA (Algorithm 1) on a benchmark problem.  Ablation flags mirror hera.HERA.
# ==========================================================================
def run_hera(problem, budget, seed, use_rl=True, use_hierarchy=True, use_gate=True, Xinit=None):
    _set_frame(problem)
    hp = C.HP
    q = hp["batch_size"]
    rng = np.random.default_rng(seed)
    Xinit = sobol_feasible(problem, hp["n0"], rng) if Xinit is None else np.asarray(Xinit)
    X, Yraw, Ccon, feas, hv_hist = _init_state(problem, Xinit)

    surr = G.Surrogate(E=hp["ensemble_size"], seed=seed).fit(
        np.array(X), np.array(Yraw), np.array(Ccon))
    tiers = None if use_hierarchy else [tuple(range(DIM))]
    policy = RL.PPOController(seed=seed, tiers=tiers)

    def Yn_feas():
        Y, f = np.array(Yraw), np.array(feas)
        return C.normalize_y(Y[f]) if f.any() else np.zeros((0, M))

    P = H.pareto_front(Yn_feas()) if np.array(feas).any() else np.zeros((0, M))
    acq_history = []
    Tmax = max(budget, 1)
    t = 0
    while len(X) < budget:
        t += 1
        yn = Yn_feas()
        mean_obj = yn.mean(axis=0) if len(yn) else np.full(M, 0.5)
        s = RL.state_features(_hv_norm(np.array(Yraw)[np.array(feas)]) if np.array(feas).any() else 0.0,
                              len(P), len(X) / budget, mean_obj)
        if use_rl:
            Zrl, _ = policy.propose(s, n=hp["nsga_pop"] // 2)
        else:
            Zrl = rng.random((hp["nsga_pop"] // 2, DIM))

        def inner_eval(Z):
            mu, _, _ = surr.predict_objectives(Z)
            _, mc, _ = surr.predict_feasibility(Z)
            viol = np.sum(np.clip(mc, 0, None), axis=1) if problem.constrained else np.zeros(len(Z))
            return mu, viol

        seeds = [Zrl]
        Xa, ff = np.array(X), np.array(feas)
        if ff.any():
            best = Xa[ff][:min(20, ff.sum())]
            seeds.append(np.clip(best + 0.03 * rng.standard_normal(best.shape), 0, 1))
        res = nsga2.nsga2(inner_eval, d=DIM, pop_size=hp["nsga_pop"], n_gen=hp["nsga_gen"],
                          rng=rng, init_pop=np.vstack(seeds), return_pool=hp["nsga_pop"])
        pool = res["pool_X"]
        mu_p, sg_p, _ = surr.predict_objectives(pool)
        pf_p, _, _ = surr.predict_feasibility(pool)
        sel = H.greedy_batch(mu_p, sg_p, pf_p, P, REF, UPPER, q)
        batch_Z = pool[sel]

        tau_t = 0.02 * (1.0 - t / (Tmax + 1)) * (np.median(acq_history) if acq_history else 0.0)
        transitions, spent = [], 0
        for bi, z in enumerate(batch_Z):
            if len(X) >= budget:
                break
            mu1, sg1 = mu_p[sel[bi]:sel[bi] + 1], sg_p[sel[bi]:sel[bi] + 1]
            pf1 = pf_p[sel[bi]]
            acq = H.single_cehvi(mu1, sg1, pf1, P, REF, UPPER)[0]
            U = surr.epistemic_U(z[None])[0]
            if use_gate:
                gate = (acq >= tau_t) or (U > hp["eps_sigma"]) or (spent == 0 and bi == len(batch_Z) - 1)
            else:
                gate = True
            if not gate:
                continue
            yraw = problem.objectives(z[None])[0]
            ccon = problem.constraints(z[None])[0]
            fz = bool(np.all(ccon <= 0))
            X.append(z); Yraw.append(yraw); Ccon.append(ccon); feas.append(fz)
            spent += 1
            acq_history.append(acq)
            hv_before = _hv_norm(np.array(Yraw[:-1])[np.array(feas[:-1])]) if any(feas[:-1]) else 0.0
            P = H.pareto_front(Yn_feas()) if np.array(feas).any() else P
            hv_after = _hv_norm(np.array(Yraw)[np.array(feas)])
            gain = max(hv_after - hv_before, 0.0)
            reward = (gain / max(hv_after, 1e-6) + hp["reward_feas"] * (1.0 if fz else 0.0)
                      - hp["reward_viol"] * float(np.sum(np.clip(ccon, 0, None))))
            hv_hist.append(hv_after)
            if use_rl:
                transitions.append(dict(s=s, means=policy._means(s),
                                        z_tier=[z[list(tr)][None] for tr in policy.tiers],
                                        reward=reward))
        if use_rl:
            policy.update(transitions)
        surr.fit(np.array(X), np.array(Yraw), np.array(Ccon))

    return _record(problem, X, Yraw, feas, hv_hist)


# ==========================================================================
# General-purpose baselines on a benchmark problem.
# ==========================================================================
def run_random(problem, budget, seed, Xinit=None):
    _set_frame(problem)
    rng = np.random.default_rng(seed + 1000)
    Xinit = sobol_feasible(problem, C.HP["n0"], rng) if Xinit is None else np.asarray(Xinit)
    X, Yraw, Ccon, feas, hv = _init_state(problem, Xinit)
    sob = qmc.Sobol(d=DIM, seed=int(rng.integers(1 << 30)))
    while len(X) < budget:
        z = sob.random(1)[0]
        X.append(z); Yraw.append(problem.objectives(z[None])[0])
        feas.append(bool(problem.is_feasible(z[None])[0]))
        hv.append(_hv_norm(np.array(Yraw)[np.array(feas)]))
    return _record(problem, X, Yraw, feas, hv)


def run_nsga2(problem, budget, seed, pop=20, Xinit=None):
    _set_frame(problem)
    rng = np.random.default_rng(seed + 2000)
    Xinit = sobol_feasible(problem, C.HP["n0"], rng) if Xinit is None else np.asarray(Xinit)
    X, Yraw, Ccon, feas, hv = _init_state(problem, Xinit)
    Z = rng.random((pop, DIM))
    Z[:min(pop, len(Xinit))] = Xinit[:pop]

    def evaluate(Zb, skip=0):
        # the first `skip` rows are the shared initial designs, already evaluated
        # and logged in X: reuse them without spending budget again.
        Yn = C.normalize_y(problem.objectives(Zb))
        viol = problem.total_violation(Zb)
        for k in range(skip, len(Zb)):
            if len(X) >= budget:
                break
            X.append(Zb[k]); Yraw.append(problem.objectives(Zb[k][None])[0])
            feas.append(bool(viol[k] <= 0))
            hv.append(_hv_norm(np.array(Yraw)[np.array(feas)]))
        return Yn, viol

    Fmax, viol = evaluate(Z, skip=min(pop, len(Xinit)))
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
            p1 = nsga2._tournament(rank, crowd, rng); p2 = nsga2._tournament(rank, crowd, rng)
            c1, c2 = nsga2._sbx_crossover(Z[p1], Z[p2], rng)
            off[i] = nsga2._poly_mutation(c1, rng)
            if i + 1 < pop:
                off[i + 1] = nsga2._poly_mutation(c2, rng)
        Foff_max, voff = evaluate(off)
        Zc = np.vstack([Z, off]); Fc = np.vstack([F, -Foff_max]); vc = np.concatenate([viol, voff])
        fronts = nsga2._fast_nondominated_sort(Fc, vc)
        newidx = []
        for fr in fronts:
            if len(newidx) + len(fr) <= pop:
                newidx.extend(fr)
            else:
                cd = nsga2._crowding_distance(Fc, fr)
                newidx.extend([fr[o] for o in np.argsort(-cd)[:pop - len(newidx)]])
                break
        newidx = np.array(newidx)
        Z, F, viol = Zc[newidx], Fc[newidx], vc[newidx]
    return _record(problem, X[:budget], Yraw[:budget], feas[:budget], hv[:budget])


def run_qehvi(problem, budget, seed, q=4, Xinit=None):
    _set_frame(problem)
    rng = np.random.default_rng(seed + 4000)
    Xinit = sobol_feasible(problem, C.HP["n0"], rng) if Xinit is None else np.asarray(Xinit)
    X, Yraw, Ccon, feas, hv = _init_state(problem, Xinit)
    while len(X) < budget:
        surr = G.Surrogate(E=C.HP["ensemble_size"], seed=seed).fit(
            np.array(X), np.array(Yraw), np.array(Ccon))
        Yn, f = C.normalize_y(np.array(Yraw)), np.array(feas)
        P = H.pareto_front(Yn[f]) if f.any() else np.zeros((0, M))
        best = np.array(X)[f] if f.any() else np.array(X)
        pool = _candidate_pool(rng, best, n_sobol=512, n_pert=128)
        mu, sg, _ = surr.predict_objectives(pool)
        pf, _, _ = surr.predict_feasibility(pool)
        for j in H.greedy_batch(mu, sg, pf, P, REF, UPPER, q):
            if len(X) >= budget:
                break
            z = pool[j]
            X.append(z); Yraw.append(problem.objectives(z[None])[0])
            Ccon.append(problem.constraints(z[None])[0]); feas.append(bool(problem.is_feasible(z[None])[0]))
            hv.append(_hv_norm(np.array(Yraw)[np.array(feas)]))
    return _record(problem, X, Yraw, feas, hv)


def run_qparego(problem, budget, seed, Xinit=None):
    from scipy.special import ndtr
    _set_frame(problem)
    rng = np.random.default_rng(seed + 3000)
    Xinit = sobol_feasible(problem, C.HP["n0"], rng) if Xinit is None else np.asarray(Xinit)
    X, Yraw, Ccon, feas, hv = _init_state(problem, Xinit)
    _SQRT2PI = np.sqrt(2 * np.pi)
    while len(X) < budget:
        Xn, Yn = np.array(X), C.normalize_y(np.array(Yraw))
        w = rng.random(M); w /= w.sum()
        scal = np.min(w * Yn, axis=1) + 0.05 * np.sum(w * Yn, axis=1)
        gp = G._GPSingle(seed=0).fit(Xn, scal)
        con = G.Surrogate(E=1, seed=seed)
        con.con_models = [G._GPSingle(seed + l).fit(Xn, np.array(Ccon)[:, l]) for l in range(problem.n_con)]
        best = Xn[np.array(feas)] if np.array(feas).any() else Xn
        pool = _candidate_pool(rng, best)
        mu, sg = gp.predict(pool)
        pf, _, _ = con.predict_feasibility(pool)
        fbest = scal.max()
        z_ = (mu - fbest - 0.01) / np.maximum(sg, 1e-9)
        ei = np.maximum((mu - fbest - 0.01) * ndtr(z_) + sg * np.exp(-0.5 * z_ * z_) / _SQRT2PI, 0.0) * pf
        z = pool[int(np.argmax(ei))]
        X.append(z); Yraw.append(problem.objectives(z[None])[0])
        Ccon.append(problem.constraints(z[None])[0]); feas.append(bool(problem.is_feasible(z[None])[0]))
        hv.append(_hv_norm(np.array(Yraw)[np.array(feas)]))
    return _record(problem, X, Yraw, feas, hv)


def run_qehvi_lbfgs(problem, budget, seed, q=4, Xinit=None, n_raw=512, n_restarts=10, maxiter=50):
    """q-EHVI with a multi-start L-BFGS-B acquisition optimiser (see baselines.qehvi_lbfgs)."""
    from .baselines import lbfgs_greedy_batch
    _set_frame(problem)
    rng = np.random.default_rng(seed + 6000)
    Xinit = sobol_feasible(problem, C.HP["n0"], rng) if Xinit is None else np.asarray(Xinit)
    X, Yraw, Ccon, feas, hv = _init_state(problem, Xinit)
    while len(X) < budget:
        surr = G.Surrogate(E=C.HP["ensemble_size"], seed=seed).fit(
            np.array(X), np.array(Yraw), np.array(Ccon))
        Yn, f = C.normalize_y(np.array(Yraw)), np.array(feas)
        P = H.pareto_front(Yn[f]) if f.any() else np.zeros((0, M))
        best = np.array(X)[f] if f.any() else np.array(X)
        raw = _candidate_pool(rng, best, n_sobol=n_raw, n_pert=128)
        for z in lbfgs_greedy_batch(surr, P, raw, np.array(X), q, n_restarts, maxiter):
            if len(X) >= budget:
                break
            X.append(z); Yraw.append(problem.objectives(z[None])[0])
            Ccon.append(problem.constraints(z[None])[0]); feas.append(bool(problem.is_feasible(z[None])[0]))
            hv.append(_hv_norm(np.array(Yraw)[np.array(feas)]))
    return _record(problem, X, Yraw, feas, hv)


BENCH_METHODS = {
    "qEHVI-LBFGS": run_qehvi_lbfgs,
    "HERA": run_hera, "qEHVI": run_qehvi, "qParEGO": run_qparego,
    "NSGA-II": run_nsga2, "Random": run_random,
}
