"""Multi-output Gaussian-process ensemble surrogate (Section 4 of the spec).

* One ensemble of E GPs per objective (Matern-5/2 ARD), giving the ensemble
  predictive mean (Eq. 20) and the aleatoric+epistemic variance decomposition
  (Eq. 21).
* One GP per constraint, giving the probabilistic feasibility estimate
  P_feas(x) = prod_l Phi(-mu_cl/sigma_cl)  (Eq. 22).
* Scalar epistemic-uncertainty summary U(x) = sqrt(sum_m sigma_bar_m^2) (Eq. 23).

Efficiency: kernel hyper-parameters are optimised once per objective/constraint
per refit (marginal-likelihood, a couple of restarts) using scikit-learn; the E
ensemble members share those hyper-parameters and differ only by bootstrap
resampling.  Prediction in the hot NSGA-II inner loop is done by a hand-rolled,
fully-vectorised Matern-5/2 predictor (FastGP) that reuses the fitted Cholesky
factors, avoiding scikit-learn per-call overhead.  Objectives are modelled in the
normalised objective space (config.normalize_y) so EHVI integrates over [r=0, 1].
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from . import config as C

warnings.filterwarnings("ignore")

_SQRT5 = np.sqrt(5.0)


def _make_kernel(d: int):
    ls = np.ones(d)
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1)))


class FastGP:
    """Vectorised Matern-5/2 ARD predictor built from a fitted sklearn GP."""

    def __init__(self, Xtr, alpha, L, length_scale, const, noise):
        self.Xtr = Xtr
        self.alpha = alpha.ravel()
        self.L = L
        self.l = length_scale
        self.const = const
        self.noise = noise
        self.kdiag = const + noise

    @classmethod
    def from_sklearn(cls, gp: GaussianProcessRegressor):
        p = gp.kernel_.get_params()
        const = float(p["k1__k1__constant_value"])
        ls = p["k1__k2__length_scale"]
        ls = np.atleast_1d(np.asarray(ls, dtype=float))
        if ls.shape[0] == 1:
            ls = np.full(gp.X_train_.shape[1], ls[0])
        noise = float(p["k2__noise_level"])
        return cls(gp.X_train_, gp.alpha_, gp.L_, ls, const, noise)

    def _k_trans(self, Xq):
        # scaled squared distance r^2 between Xq (B,d) and Xtr (n,d)
        Xs = Xq / self.l
        Ts = self.Xtr / self.l
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
        aa = np.sum(Xs * Xs, axis=1)[:, None]
        bb = np.sum(Ts * Ts, axis=1)[None, :]
        r2 = np.maximum(aa + bb - 2.0 * Xs @ Ts.T, 0.0)
        r = np.sqrt(r2)
        return self.const * (1.0 + _SQRT5 * r + (5.0 / 3.0) * r2) * np.exp(-_SQRT5 * r)

    def predict(self, Xq):
        Xq = np.atleast_2d(Xq)
        Kt = self._k_trans(Xq)                       # (B,n)
        mean = Kt @ self.alpha
        V = solve_triangular(self.L, Kt.T, lower=True)
        var = self.kdiag - np.sum(V * V, axis=0)
        return mean, np.maximum(var, 1e-12)


class _GPEnsemble:
    """Ensemble of E GPs sharing optimised hyper-parameters, bootstrap data."""

    def __init__(self, E: int, seed: int):
        self.E = E
        self.rng = np.random.default_rng(seed)
        self.fast: list[FastGP] = []
        self.ymean = 0.0
        self.ystd = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.ymean = float(np.mean(y))
        self.ystd = float(np.std(y)) or 1.0
        ys = (y - self.ymean) / self.ystd
        opt = GaussianProcessRegressor(
            kernel=_make_kernel(d), n_restarts_optimizer=2, normalize_y=False,
            alpha=1e-6, random_state=int(self.rng.integers(1 << 30)))
        opt.fit(X, ys)
        frozen = opt.kernel_
        self.fast = [FastGP.from_sklearn(opt)]
        for _ in range(self.E - 1):
            idx = self.rng.integers(0, n, size=n)
            gp = GaussianProcessRegressor(kernel=frozen, optimizer=None,
                                          normalize_y=False, alpha=1e-6)
            try:
                gp.fit(X[idx], ys[idx])
                self.fast.append(FastGP.from_sklearn(gp))
            except Exception:
                pass
        return self

    def predict(self, X):
        mus, vars_ = [], []
        for gp in self.fast:
            m, v = gp.predict(X)
            mus.append(m)
            vars_.append(v)
        mus = np.array(mus)
        vars_ = np.array(vars_)
        mu_bar = mus.mean(axis=0)
        aleatoric = vars_.mean(axis=0)
        epistemic = ((mus - mu_bar) ** 2).mean(axis=0)
        var_bar = aleatoric + epistemic
        mu_o = mu_bar * self.ystd + self.ymean
        std_o = np.sqrt(np.maximum(var_bar, 1e-18)) * self.ystd
        epi_o = np.sqrt(np.maximum(epistemic, 1e-18)) * self.ystd
        return mu_o, std_o, epi_o


class _GPSingle:
    """Single GP for a constraint, predicting mean/std in original units."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.fast = None
        self.ymean = 0.0
        self.ystd = 1.0

    def fit(self, X, y):
        self.ymean = float(np.mean(y))
        self.ystd = float(np.std(y)) or 1.0
        ys = (y - self.ymean) / self.ystd
        gp = GaussianProcessRegressor(
            kernel=_make_kernel(X.shape[1]), n_restarts_optimizer=1,
            normalize_y=False, alpha=1e-6,
            random_state=int(self.rng.integers(1 << 30)))
        gp.fit(X, ys)
        self.fast = FastGP.from_sklearn(gp)
        return self

    def predict(self, X):
        m, v = self.fast.predict(X)
        return m * self.ystd + self.ymean, np.maximum(np.sqrt(v) * self.ystd, 1e-9)


# Optional physics prior (twin-informed surrogate). When set to a callable
# PRIOR(Xn) -> (Yn_prior (n, M) in normalised objective space, C_prior (n, p)),
# the GPs model the residual between the expensive oracle and the prior, and the
# prior is added back in every prediction:  f(x) = prior(x) + delta(x),
# delta ~ GP. Default None: plain GPs (zero-mean after standardisation).
PRIOR = None
# Optional extra training data (Xn (k, d), Yobj_raw (k, M), Ccon (k, p)) appended to
# every fit, e.g. TCAD simulations available before the optimisation starts. They
# inform the surrogates only; fronts and hypervolumes use the optimiser's own data.
EXTRA = None


class Surrogate:
    """Objective ensembles + constraint GPs in normalised x / objective space."""

    def __init__(self, E: int = None, seed: int = 0):
        self.E = E or C.HP["ensemble_size"]
        self.seed = seed
        self.obj_models: list[_GPEnsemble] = []
        self.con_models: list[_GPSingle] = []
        self.prior = PRIOR

    def fit(self, Xn, Yobj_raw, Ccon):
        if EXTRA is not None:
            Xe, Ye, Ce = EXTRA
            Xn = np.vstack([Xn, Xe])
            Yobj_raw = np.vstack([Yobj_raw, Ye])
            Ccon = np.vstack([Ccon, Ce])
        Yn = C.normalize_y(Yobj_raw)
        Ccon = np.asarray(Ccon, dtype=float)
        if self.prior is not None:
            Yp, Cp = self.prior(Xn)
            Yn = Yn - Yp
            Ccon = Ccon - Cp
        self.obj_models = [
            _GPEnsemble(self.E, self.seed + 17 * m + 1).fit(Xn, Yn[:, m])
            for m in range(C.M_OBJ)]
        self.con_models = [
            _GPSingle(self.seed + 101 * l + 3).fit(Xn, Ccon[:, l])
            for l in range(Ccon.shape[1])]
        return self

    def predict_objectives(self, Xn):
        Xn = np.atleast_2d(Xn)
        mu = np.empty((Xn.shape[0], C.M_OBJ))
        sg = np.empty_like(mu)
        ep = np.empty_like(mu)
        for m, model in enumerate(self.obj_models):
            mu[:, m], sg[:, m], ep[:, m] = model.predict(Xn)
        if self.prior is not None:
            mu = mu + self.prior(Xn)[0]
        return mu, sg, ep

    def predict_feasibility(self, Xn):
        Xn = np.atleast_2d(Xn)
        pfeas = np.ones(Xn.shape[0])
        mus, sgs = [], []
        Cp = self.prior(Xn)[1] if self.prior is not None else None
        for l, model in enumerate(self.con_models):
            mc, sc = model.predict(Xn)
            if Cp is not None:
                mc = mc + Cp[:, l]
            pfeas *= ndtr(-mc / sc)
            mus.append(mc)
            sgs.append(sc)
        return pfeas, np.array(mus).T, np.array(sgs).T

    def epistemic_U(self, Xn):
        _, _, ep = self.predict_objectives(Xn)
        return np.sqrt(np.sum(ep ** 2, axis=1))
