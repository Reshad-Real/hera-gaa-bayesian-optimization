"""HERA: Hierarchical Expected-hypervolume Reinforcement-guided Acquisition.

Assembles Algorithm 1 (and the GreedyBatch of Algorithm 2) from the spec:
  Phase 1  feasible Sobol initialisation (n0 designs)
  Phase 2  fit GP ensemble surrogate + constraint GPs, init RL policy
  Phase 3  main loop:
             RL warm-start proposals  (Eq. 17)
             -> NSGA-II inner optimisation on surrogate means (Sec. 7)
             -> greedy submodular batch by single-point cEHVI (Algorithm 2)
             -> cost-aware active-learning gate (Eqs. 36-37)
             -> expensive evaluation, Pareto + reward + PPO update, refit
             -> 3-signal convergence check (Eqs. 38-40)
  Phase 4  return Pareto set, surrogate, policy.

The expensive evaluation E(x) is the TCAD-calibrated GAA digital-twin oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import qmc

from . import config as C
from . import finfet_oracle as O
from . import hypervolume as H
from . import gp_surrogate as G
from . import nsga2
from . import rl_controller as RL

REF = C.REF_POINT_NORM            # r = 0 in normalised objective space
UPPER = np.full(C.M_OBJ, 1.15)    # slight headroom above the ideal for EHVI cells


def sobol_feasible(n0: int, rng: np.random.Generator, max_tries: int = 200000):
    """Phase 1: n0 Sobol designs that satisfy the *analytic* constraints c1-c3.

    Only the geometric/process constraints (EOT floor, aspect ratio, scale-length
    rule) are screened, because they cost nothing to evaluate.  The simulation-
    derived constraints c4-c5 (DIBL, leakage) are NOT screened, so no expensive
    oracle call is spent outside the budget: every initial design is evaluated
    once and counted as one of the B expensive evaluations.
    """
    seed = int(rng.integers(1 << 30))
    sob = qmc.Sobol(d=6, seed=seed)
    X, found = [], 0
    batch = 256
    tries = 0
    while found < n0 and tries < max_tries:
        Z = sob.random(batch)
        Xp = C.denormalize_x(Z)
        feas = np.all(O.geometric_constraints(Xp) <= 0, axis=1)
        for k in np.where(feas)[0]:
            X.append(Xp[k])
            found += 1
            if found >= n0:
                break
        tries += batch
    return np.array(X[:n0])


@dataclass
class OptResult:
    name: str
    X: np.ndarray                 # evaluated physical designs (n,6)
    Y: np.ndarray                 # raw objectives (n,3)
    feas: np.ndarray              # feasibility mask (n,)
    hv_history: np.ndarray        # HV(normalised) after each expensive eval
    n_init: int                   # number of initial designs
    pareto_X: np.ndarray = None   # feasible Pareto designs (physical)
    pareto_Y: np.ndarray = None   # feasible Pareto objectives (raw)
    init_X: np.ndarray = None     # the n0 initial feasible designs
    init_Y: np.ndarray = None
    extra: dict = field(default_factory=dict)


class HERA:
    def __init__(self, budget: int = 100, seed: int = 0, use_convergence: bool = False,
                 verbose: bool = False, init_X: np.ndarray = None,
                 use_rl: bool = True, use_hierarchy: bool = True, use_gate: bool = True):
        self.B = budget
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_convergence = use_convergence
        self.verbose = verbose
        self.init_X = init_X
        # ablation switches (all True -> the full HERA of Algorithm 1).
        #   use_rl        : PPO controller warm-starts the inner search (Eq. 17);
        #                   off -> NSGA-II is seeded from Sobol/perturbations only.
        #   use_hierarchy : tier-factored policy (Eq. 15); off -> a single flat
        #                   policy over all six variables jointly.
        #   use_gate      : cost-aware active-learning evaluation gate (Eq. 37);
        #                   off -> every batch member is evaluated.
        self.use_rl = use_rl
        self.use_hierarchy = use_hierarchy
        self.use_gate = use_gate
        self.hp = C.HP

    # -- helpers ----------------------------------------------------------
    def _hv(self, Yn_feas):
        if len(Yn_feas) == 0:
            return 0.0
        return H.hypervolume(Yn_feas, REF)

    def run(self) -> OptResult:
        hp = self.hp
        q = hp["batch_size"]
        Tmax = max(self.B, 1)

        # ----- Phase 1: feasible init -----
        n0 = hp["n0"]
        Xinit = self.init_X if self.init_X is not None else sobol_feasible(n0, self.rng)
        Xinit = np.asarray(Xinit, dtype=float)
        n0 = len(Xinit)
        X = list(Xinit)
        Yraw = list(O.objectives(Xinit))
        Ccon = list(O.constraints(Xinit))
        feas = list(O.is_feasible(Xinit))

        def Xn_all():
            return C.normalize_x(np.array(X))

        def Yn_feas_arr():
            Y = np.array(Yraw)
            f = np.array(feas)
            return C.normalize_y(Y[f]) if f.any() else np.zeros((0, 3))

        hv_hist = []
        for i in range(len(X)):
            Y = np.array(Yraw[:i + 1]); f = np.array(feas[:i + 1])
            hv_hist.append(self._hv(C.normalize_y(Y[f]) if f.any() else np.zeros((0, 3))))

        # ----- Phase 2: surrogate + policy -----
        surr = G.Surrogate(E=hp["ensemble_size"], seed=self.seed).fit(
            Xn_all(), np.array(Yraw), np.array(Ccon))
        tiers = None if self.use_hierarchy else [tuple(range(6))]
        policy = RL.PPOController(seed=self.seed, tiers=tiers)
        P = H.pareto_front(Yn_feas_arr()) if np.array(feas).any() else np.zeros((0, 3))
        acq_history = []          # realised cEHVI/cost for the adaptive gate
        hv_window = []            # for convergence

        # ----- Phase 3: main loop -----
        t = 0
        while len(X) < self.B:
            t += 1
            budget_frac = len(X) / self.B
            mean_obj = (Yn_feas_arr().mean(axis=0) if len(Yn_feas_arr())
                        else np.full(3, 0.5))
            s = RL.state_features(self._hv(Yn_feas_arr()), len(P), budget_frac, mean_obj)

            # RL warm-start proposals (Eq. 17); ablation -> Sobol/uniform seeds
            if self.use_rl:
                Zrl, _ = policy.propose(s, n=hp["nsga_pop"] // 2)
            else:
                Zrl = self.rng.random((hp["nsga_pop"] // 2, 6))

            # inner NSGA-II on surrogate posterior means with constraint-domination
            def inner_eval(Z):
                mu, _, _ = surr.predict_objectives(Z)
                _, mc, _ = surr.predict_feasibility(Z)     # constraint means
                viol = np.sum(np.clip(mc, 0, None), axis=1)
                return mu, viol

            # seed = RL proposals + perturbations of current front designs
            seeds = [Zrl]
            if P.shape[0] > 0:
                # perturb best evaluated designs
                Xn = Xn_all(); ff = np.array(feas)
                if ff.any():
                    best = Xn[ff][:min(20, ff.sum())]
                    pert = np.clip(best + 0.03 * self.rng.standard_normal(best.shape), 0, 1)
                    seeds.append(pert)
            init_pop = np.vstack(seeds)
            res = nsga2.nsga2(inner_eval, d=6, pop_size=hp["nsga_pop"],
                              n_gen=hp["nsga_gen"], rng=self.rng,
                              init_pop=init_pop, return_pool=hp["nsga_pop"])
            pool = res["pool_X"]

            # surrogate posterior for the pool
            mu_p, sg_p, _ = surr.predict_objectives(pool)
            pf_p, _, _ = surr.predict_feasibility(pool)

            # greedy submodular batch (Algorithm 2)
            sel = H.greedy_batch(mu_p, sg_p, pf_p, P, REF, UPPER, q)
            batch_Z = pool[sel]

            # adaptive gate threshold tau_t (Eq. 37)
            tau0 = 0.02
            med = np.median(acq_history) if acq_history else 0.0
            tau_t = tau0 * (1.0 - t / (Tmax + 1)) * med

            transitions = []
            spent_this_iter = 0
            for bi, z in enumerate(batch_Z):
                if len(X) >= self.B:
                    break
                mu1 = mu_p[sel[bi]:sel[bi] + 1]
                sg1 = sg_p[sel[bi]:sel[bi] + 1]
                pf1 = pf_p[sel[bi]]
                acq = H.single_cehvi(mu1, sg1, pf1, P, REF, UPPER)[0]
                U = surr.epistemic_U(z[None])[0]
                cost = 1.0
                if self.use_gate:
                    gate = (acq / cost >= tau_t) or (U > hp["eps_sigma"]) or (spent_this_iter == 0 and bi == len(batch_Z) - 1)
                else:
                    gate = True          # ablation: evaluate every batch member
                if not gate:
                    continue
                # expensive evaluation E(x)
                xp = C.denormalize_x(z[None])[0]
                yraw = O.objectives(xp[None])[0]
                ccon = O.constraints(xp[None])[0]
                fz = bool(np.all(ccon <= 0))
                X.append(xp); Yraw.append(yraw); Ccon.append(ccon); feas.append(fz)
                spent_this_iter += 1
                acq_history.append(acq / cost)

                # update Pareto front + reward (Eq. 33)
                hv_before = self._hv(P)
                P = H.pareto_front(Yn_feas_arr()) if np.array(feas).any() else P
                hv_after = self._hv(P)
                hv_gain = max(hv_after - hv_before, 0.0)
                reward = (hv_gain / max(hv_after, 1e-6)
                          + hp["reward_feas"] * (1.0 if fz else 0.0)
                          - hp["reward_viol"] * float(np.sum(np.clip(ccon, 0, None))))
                hv_hist.append(hv_after)
                if self.use_rl:
                    transitions.append(dict(
                        s=s, means=policy._means(s),
                        z_tier=[z[list(tier)][None] for tier in policy.tiers],
                        reward=reward))

            # PPO update (Eqs. 34-35)
            if self.use_rl:
                policy.update(transitions)

            # refit surrogate
            surr.fit(Xn_all(), np.array(Yraw), np.array(Ccon))

            # convergence (Eqs. 38-40)
            hv_window.append(self._hv(Yn_feas_arr()))
            if self.use_convergence and len(hv_window) > hp["conv_window"]:
                recent = np.diff(hv_window[-hp["conv_window"]:])
                if recent.mean() < hp["eps_hv"]:
                    if self.verbose:
                        print(f"[HERA seed {self.seed}] converged at {len(X)} evals")
                    break
            if self.verbose and t % 5 == 0:
                print(f"[HERA seed {self.seed}] evals={len(X)} HV={self._hv(Yn_feas_arr()):.4f}")

        # ----- Phase 4: extract -----
        Y = np.array(Yraw); f = np.array(feas); Xa = np.array(X)
        pmask = H.pareto_front(C.normalize_y(Y[f]), return_mask=True) if f.any() else np.zeros(0, bool)
        pareto_X = Xa[f][pmask] if f.any() else np.zeros((0, 6))
        pareto_Y = Y[f][pmask] if f.any() else np.zeros((0, 3))

        return OptResult(
            name="HERA", X=Xa, Y=Y, feas=f, hv_history=np.array(hv_hist),
            n_init=n0, pareto_X=pareto_X, pareto_Y=pareto_Y,
            init_X=Xinit, init_Y=O.objectives(Xinit),
            extra=dict(surrogate=surr, policy=policy),
        )
