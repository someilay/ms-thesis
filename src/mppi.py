"""MPPI (Model Predictive Path Integral) controller implemented in JAX."""

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass
class MPPIConfig:
    num_samples: int = 512
    horizon: int = 30
    lambda_: float = 1.0
    noise_sigma: list[float] = field(default_factory=lambda: [1.0])
    u_min: list[float] | None = None
    u_max: list[float] | None = None
    rollout_var_discount: float = 0.99

    @property
    def nu(self) -> int:
        return len(self.noise_sigma)


class MPPI:
    """
    Model Predictive Path Integral controller (simple variant).

    Reference: Williams et al., 2017 "Information Theoretic MPC for
    Model-Based Reinforcement Learning".

    Args:
        cfg: hyperparameters
        rollout_fn: (perturbed_actions [K, T, nu]) -> costs [T, K]
            Raw undiscounted per-step costs. Discounting is applied internally.
    """

    def __init__(
        self,
        cfg: MPPIConfig,
        rollout_fn: Callable[[jax.Array], jax.Array],
    ):
        self.cfg = cfg
        self.rollout_fn = rollout_fn

        self.k = cfg.num_samples
        self.t = cfg.horizon
        self.lambda_ = cfg.lambda_

        sigma = jnp.array(cfg.noise_sigma)
        self.nu = cfg.nu
        self.noise_sigma = jnp.diag(sigma)
        self.noise_sigma_inv = jnp.diag(1.0 / sigma)

        self.u_min = jnp.array(cfg.u_min) if cfg.u_min is not None else None
        self.u_max = jnp.array(cfg.u_max) if cfg.u_max is not None else None

        self.gamma_seq = jnp.cumprod(
            jnp.array([1.0] + [cfg.rollout_var_discount] * (self.t - 1)),
        )  # [T]

        self.u_seq = jnp.zeros((self.t, self.nu))
        self.key = jax.random.PRNGKey(0)

    def _importance_weights(self, costs: jax.Array) -> jax.Array:
        """Softmin importance weights. costs [K] -> weights [K]."""
        shifted = costs - costs.min()
        weights = jnp.exp(-shifted / self.lambda_)
        return weights / weights.sum()

    def _action_cost(self, noise: jax.Array) -> jax.Array:
        """Control perturbation cost: lambda * u^T Sigma^{-1} eps. noise [K, T, nu] -> [K]."""
        return self.lambda_ * jnp.einsum(
            "ti,ij,ktj->k",
            self.u_seq,
            self.noise_sigma_inv,
            noise,
        )

    def command(self) -> jax.Array:
        """
        Compute the optimal action.

        Returns:
            action [nu]
        """
        self.key, subkey = jax.random.split(self.key)
        noise = jax.random.multivariate_normal(
            subkey, jnp.zeros(self.nu), self.noise_sigma, shape=(self.k, self.t)
        )

        perturbed = self.u_seq[None] + noise  # [K, T, nu]
        if self.u_min is not None and self.u_max is not None:
            perturbed = jnp.clip(perturbed, self.u_min, self.u_max)

        costs_tk = self.rollout_fn(perturbed)  # [T, K]
        costs_k = (costs_tk * self.gamma_seq[:, None]).sum(axis=0)  # [K]
        costs_k = costs_k + self._action_cost(noise)

        weights = self._importance_weights(costs_k)
        self.u_seq = self.u_seq + jnp.einsum("k,kti->ti", weights, noise)

        if self.u_min is not None and self.u_max is not None:
            self.u_seq = jnp.clip(self.u_seq, self.u_min, self.u_max)

        action = self.u_seq[0]
        self.u_seq = jnp.roll(self.u_seq, -1, axis=0)
        self.u_seq = self.u_seq.at[-1].set(jnp.zeros(self.nu))
        return action
