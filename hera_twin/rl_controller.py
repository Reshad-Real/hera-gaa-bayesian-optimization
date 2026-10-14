"""Reinforcement-learning controller (Section 6 of the spec).

A PPO-trained policy pi_theta that learns *where* in the feasible set the
hypervolume-improving designs lie and warm-starts the NSGA-II inner optimiser.
It never replaces the acquisition; it only biases candidate generation.

Faithful to the spec structure:
* Tier-factored proposal distribution (Eq. 17): the policy factorises over the
  physical tiers, each a Gaussian over its (normalised) variables whose mean is a
  state-conditioned squashed map and whose log-std is learnable.  Passing a single
  all-variable tier recovers a non-hierarchical (flat) policy, used for the
  hierarchy ablation.
* Reward (Eq. 33): realised normalised hypervolume gain + feasibility bonus
  - total-violation penalty.
* Clipped PPO objective with entropy bonus and a (linear) value baseline
  (Eqs. 34-35).

Implementation is a compact, self-contained numpy PPO (manual gradients) - the
controller's role is candidate biasing, so robustness is favoured over scale.
"""
from __future__ import annotations

import numpy as np

from . import config as C

_STATE_DIM = 6               # [HV, |P|/Pmax, budget_frac, mean_obj(3)]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class PPOController:
    def __init__(self, seed: int = 0, lr: float = 0.02, tiers=None):
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        # tier layout: list of index-tuples partitioning the design variables.
        # Default = the physical hierarchy C.TIERS; a single all-variable tuple
        # gives a flat (non-hierarchical) policy for the ablation.
        self.tiers = [tuple(t) for t in (tiers if tiers is not None else C.TIERS)]
        self.nt = len(self.tiers)
        self.dim = sum(len(t) for t in self.tiers)
        w = [len(t) for t in self.tiers]                 # per-tier width
        # policy: per-tier linear map state->mean, learnable log-std
        self.W = [0.01 * self.rng.standard_normal((w[t], _STATE_DIM)) for t in range(self.nt)]
        self.b = [np.zeros(w[t]) for t in range(self.nt)]
        # initialise biases so the mean starts near 0.5 (centre of the cube)
        self.log_std = [np.full(w[t], np.log(0.25)) for t in range(self.nt)]
        # value baseline (linear)
        self.Wv = np.zeros(_STATE_DIM)
        self.bv = 0.0
        self.clip = C.HP["ppo_clip"]
        self.ent_w = C.HP["ppo_entropy"]

    # -- policy forward -----------------------------------------------------
    def _means(self, s: np.ndarray):
        return [_sigmoid(self.W[t] @ s + self.b[t]) for t in range(self.nt)]

    def propose(self, s: np.ndarray, n: int):
        """Sample n candidate designs (normalised x in [0,1]^dim) from pi(.|s).

        Returns (Z (n,dim), info) where info caches the raw tier samples and the
        state for the PPO update.
        """
        means = self._means(s)
        Z = np.empty((n, self.dim))
        z_tier = [np.empty((n, len(t))) for t in self.tiers]
        for t, tier in enumerate(self.tiers):
            std = np.exp(self.log_std[t])
            eps = self.rng.standard_normal((n, len(tier)))
            zt = means[t][None, :] + std[None, :] * eps
            z_tier[t] = zt
            Z[:, list(tier)] = np.clip(zt, 0.0, 1.0)
        return Z, dict(s=s, means=means, z_tier=z_tier)

    def _logprob(self, means, log_std, z_tier):
        """Sum of per-tier Gaussian log-probs for stored samples (n,)."""
        lp = np.zeros(z_tier[0].shape[0])
        for t in range(self.nt):
            std = np.exp(log_std[t])
            d = (z_tier[t] - means[t][None, :]) / std
            lp += np.sum(-0.5 * d * d - log_std[t][None, :] - 0.5 * np.log(2 * np.pi), axis=1)
        return lp

    # -- PPO update ---------------------------------------------------------
    def update(self, batch, epochs: int = 4):
        """One PPO update from a list of transition dicts.

        Each transition: dict(s, means, z_tier (1,width each), reward).
        """
        if not batch:
            return
        S = np.array([tr["s"] for tr in batch])               # (B, state)
        R = np.array([tr["reward"] for tr in batch])          # (B,)
        # stack tier samples
        Zt = [np.array([tr["z_tier"][t][0] for tr in batch]) for t in range(self.nt)]  # (B,width)
        means_old = [np.array([tr["means"][t] for tr in batch]) for t in range(self.nt)]
        log_std_old = [ls.copy() for ls in self.log_std]

        # value baseline (closed-form ridge fit to returns), advantages
        Vpred = S @ self.Wv + self.bv
        adv = R - Vpred
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # fit value head to returns (ridge)
        A = np.hstack([S, np.ones((len(S), 1))])
        lam = 1.0
        sol = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ R)
        self.Wv, self.bv = sol[:-1], sol[-1]

        # old log-probs (per-sample) with the means stored per sample
        def logprob_with(meansB, logstd, Zt):
            lp = np.zeros(len(S))
            for t in range(self.nt):
                std = np.exp(logstd[t])
                d = (Zt[t] - meansB[t]) / std
                lp += np.sum(-0.5 * d * d - logstd[t][None, :]
                             - 0.5 * np.log(2 * np.pi), axis=1)
            return lp

        lp_old = logprob_with(means_old, log_std_old, Zt)

        for _ in range(epochs):
            # current means per sample
            meansB = []
            pre = []
            for t in range(self.nt):
                z = S @ self.W[t].T + self.b[t]               # (B,width)
                pre.append(z)
                meansB.append(_sigmoid(z))
            lp_new = logprob_with(meansB, self.log_std, Zt)
            ratio = np.exp(np.clip(lp_new - lp_old, -10, 10))
            unclipped = ratio * adv
            clipped = np.clip(ratio, 1 - self.clip, 1 + self.clip) * adv
            use_unclipped = unclipped <= clipped       # PPO takes the min

            # gradient of -L wrt params; accumulate
            # d logpi / d mean_t = (z - mean)/std^2 ; d mean/d pre = mean(1-mean)
            for t in range(self.nt):
                std2 = np.exp(self.log_std[t]) ** 2
                dlp_dmean = (Zt[t] - meansB[t]) / std2       # (B,width)
                dmean_dpre = meansB[t] * (1 - meansB[t])      # (B,width)
                dlp_dpre = dlp_dmean * dmean_dpre             # (B,width)
                # objective coeff per sample: ratio*adv where unclipped active
                coeff = np.where(use_unclipped, ratio * adv, 0.0)  # (B,)
                g_pre = coeff[:, None] * dlp_dpre             # (B,width)
                gW = g_pre.T @ S / len(S)                     # (width,state)
                gb = g_pre.mean(axis=0)                       # (width,)
                # ascent (maximise objective) + small entropy already in log_std
                self.W[t] += self.lr * gW
                self.b[t] += self.lr * gb
                # log_std gradient: d logpi/d log_std = ((z-mean)/std)^2 - 1
                d = (Zt[t] - meansB[t]) / np.exp(self.log_std[t])
                dlp_dls = (d * d - 1.0)                       # (B,width)
                g_ls = (coeff[:, None] * dlp_dls).mean(axis=0)
                g_ls += self.ent_w                            # entropy bonus
                self.log_std[t] = np.clip(self.log_std[t] + self.lr * g_ls,
                                          np.log(0.03), np.log(0.5))


def state_features(hv, n_front, budget_frac, mean_obj_norm):
    """Build the controller state vector s (Eq. 32 summary phi(P), budget)."""
    s = np.array([
        np.clip(hv, 0, 2),
        np.clip(n_front / 40.0, 0, 2),
        np.clip(budget_frac, 0, 1),
        *np.clip(mean_obj_norm, -0.5, 1.5),
    ], dtype=float)
    return s
