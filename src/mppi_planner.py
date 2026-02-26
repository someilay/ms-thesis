from dataclasses import dataclass
import jax
import jax.numpy as jnp
from mjx_wrapper import MjxWrapper, MujocoConfig
from mppi import MPPI, MPPIConfig
from objective import Objective


@dataclass
class PlannerConfig:
    mppi_cfg: MPPIConfig
    muj_cfg: MujocoConfig


class MPPIMujocoPlanner:
    def __init__(
        self,
        cfg: PlannerConfig,
        objective: Objective,
    ):
        self.cfg = cfg
        self.objective = objective
        self.sim = MjxWrapper(
            cfg=cfg.muj_cfg,
            num_envs=cfg.mppi_cfg.num_samples,
        )

        self._cost_fn = self.objective.build_cost_fn(self.sim)
        self._applied_u_single = jnp.zeros((1, self.sim.nv), dtype=jnp.float32)

        step_batch_costed = self.sim.build_step_batch_costed(self._cost_fn)
        nv, nu = self.sim.nv, cfg.mppi_cfg.nu
        k, t = cfg.mppi_cfg.num_samples, cfg.mppi_cfg.horizon

        def rollout_fn(perturbed_actions: jax.Array) -> jax.Array:
            # perturbed_actions: [K, T, nu] -> costs [T, K]
            u_batch = (
                jnp.zeros((k, t, nv), dtype=jnp.float32)
                .at[:, :, -nu:]
                .set(perturbed_actions)
                .transpose(1, 0, 2)  # [T, K, nv]
            )
            costs, _ = step_batch_costed(u_batch)
            return costs  # [T, K]

        self.mppi = MPPI(cfg.mppi_cfg, rollout_fn=rollout_fn)

    def _to_applied_u_single(self, u: jax.Array) -> jax.Array:
        return self._applied_u_single.at[0, -self.cfg.mppi_cfg.nu :].set(u)

    def reset_rollout_sim(self, dof_state_tensor: jax.Array) -> None:
        self.sim.reset_visualize_link_buffer()
        self.sim.set_dof_state(dof_state_tensor)

    def compute_action_tensor(self, dof_state_tensor: jax.Array) -> jax.Array:
        self.reset_rollout_sim(dof_state_tensor)
        u = self._to_applied_u_single(self.command())
        u = self.sim.constrained_force(u)
        return u

    def command(self) -> jax.Array:
        action = self.mppi.command()
        self.sim.raise_if_nan()
        return action

    def get_rollouts(self) -> jax.Array | None:
        if not self.sim.visualize_link_present:
            return None
        return self.sim.visualize_link_buffer
