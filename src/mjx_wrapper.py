from pathlib import Path
from typing import Callable, cast
import numpy as np
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
from constraints import ConstraintsConfig
from load_free_flyer import load_urdf


ROOT_FOLDER = Path(__file__).parent.parent
jax.config.update("jax_compilation_cache_dir", (ROOT_FOLDER / "jax_cache").as_posix())
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)
# jax.config.update("jax_logging_level", "DEBUG")


@dataclass
class MujocoConfig:
    dt: float = 0.05
    init_pos: list[float] = field(default_factory=lambda: [0, 0, 0])
    init_ori: list[float] = field(default_factory=lambda: [0, 0, 0, 1])
    urdf_path: str = "assets/urdf/boxer/boxer.urdf"
    package_dirs: list[str] = field(default_factory=lambda: ["assets/urdf/boxer"])
    constraints_cfg: list[ConstraintsConfig] = field(default_factory=list)
    fixed: bool = False
    numerical_opt: bool = True
    visualize_link: str | None = "robot_root"


class MjxWrapper:
    """MuJoCo MJX wrapper with parallel environments and holonomic constraints.

    Implements constrained dynamics using Gauss least action principle:
    minimize: 0.5 * (a - a_free)^T mass (a - a_free)
    subject to: constraint_jac @ a = constraint_bias

    where a is constrained acceleration, a_free is unconstrained acceleration.
    """

    def __init__(
        self,
        cfg: MujocoConfig,
        num_envs: int = 1,
        device: jax.Device | None = None,
        actor_root_body: dict[str, str] | None = None,
        warmup: bool = True,
    ) -> None:
        model_folder = ROOT_FOLDER / Path(cfg.urdf_path).parent
        filename = Path(cfg.urdf_path).name
        model, data = load_urdf(
            model_folder, filename, fixed=cfg.fixed, add_floor=False
        )
        model.opt.timestep = cfg.dt
        mujoco.mj_forward(model, data)

        self._device = cast(jax.Device, device or jax.devices()[0])
        self._model_cpu = model
        self._model = mjx.put_model(model, device=self._device)
        self._num_envs = num_envs
        self._constraints = [
            c.build_constraint(model, data) for c in cfg.constraints_cfg
        ]
        self._dt = cfg.dt
        self._actor_root_body = actor_root_body or {
            "robot": "robot_root",
            "goal": "goal_root",
        }

        data_cpu = mujoco.MjData(model)
        mujoco.mj_forward(model, data_cpu)

        self._data = jax.vmap(
            lambda _: mjx.put_data(model, data_cpu, device=self._device),
        )(jnp.arange(num_envs))

        self._ctrl = jnp.zeros((num_envs, model.nv))
        self._visualize_link_buffer: jax.Array | list[jax.Array] = []
        self._visualize_link_id = -1
        if cfg.visualize_link is not None:
            self._visualize_link_id = mujoco.mj_name2id(
                self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, cfg.visualize_link
            )
            if self._visualize_link_id < 0:
                raise ValueError(f"Visualize link '{cfg.visualize_link}' not found")

        self._step_fn = jax.jit(jax.vmap(self._step_single, in_axes=(None, 0, 0)))
        if warmup:
            self._step_fn(self._model, self._data, self._ctrl)

        # setup goal positions and orientations
        self._goal_pos = jnp.zeros((self._num_envs, 3), dtype=np.float32)
        self._goal_ori = jnp.zeros((self._num_envs, 4), dtype=np.float32)
        self._goal_ori = self._goal_ori.at[:, -1].set(1)

    def _step_single(
        self,
        model: mjx.Model,
        data: mjx.Data,
        ctrl: jax.Array,
    ) -> mjx.Data:
        """Step single environment with constrained dynamics."""
        # Apply control
        data = data.replace(qfrc_applied=ctrl)

        if not self._constraints:
            return mjx.step(model, data)
        return self._constrained_step(model, data)

    def _constrained_step(self, model: mjx.Model, data: mjx.Data) -> mjx.Data:
        """Constrained step via Udwadia-Kalaba: Q_c = M^{1/2}(A M^{-1/2})^+ (b - A a_free).

        No Schur complement; (A M^{-1/2})^+ handles rank-deficient A and inconsistent b.
        """
        data = mjx.forward(model, data)
        a_free = data.qacc

        constraint_jac_blocks = []
        constraint_bias_blocks = []

        for constraint in self._constraints:
            jac, bias = constraint.compute(
                data.qpos,
                data.qvel,
                model,
                data,
            )
            if jac.shape[0] > 0:
                constraint_jac_blocks.append(jac)
                constraint_bias_blocks.append(bias)

        constraint_jac = jnp.vstack(constraint_jac_blocks)
        constraint_bias = jnp.concatenate(constraint_bias_blocks)

        mass = self._get_mass_matrix(model, data)
        mass_inv_sqrt = self._mass_inv_sqrt(mass)
        mass_sqrt = self._mass_sqrt(mass)

        a_m_inv_sqrt = constraint_jac @ mass_inv_sqrt
        rhs = constraint_bias - constraint_jac @ a_free
        qfrc_constraint = mass_sqrt @ (jnp.linalg.pinv(a_m_inv_sqrt) @ rhs)

        data = data.replace(qfrc_applied=data.qfrc_applied + qfrc_constraint)

        return mjx.step(model, data)

    def _get_mass_matrix(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        """Full dense mass matrix M (nv x nv). Call after mjx.forward so data has qM."""
        return mjx_support.full_m(model, data)

    def _mass_sqrt(self, mass: jax.Array) -> jax.Array:
        """Symmetric M^{1/2} so that M^{1/2} M^{1/2} = mass."""
        s, u = jnp.linalg.eigh(mass)
        return (u * jnp.sqrt(jnp.maximum(s, 0.0))) @ u.T

    def _mass_inv_sqrt(self, mass: jax.Array) -> jax.Array:
        """Symmetric M^{-1/2}."""
        s, u = jnp.linalg.eigh(mass)
        safe_s = jnp.maximum(s, jnp.finfo(mass.dtype).eps)
        return (u / jnp.sqrt(safe_s)) @ u.T

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> jax.Device:
        return self._device

    @property
    def nv(self) -> int:
        return self._model.nv

    @property
    def nq(self) -> int:
        return self._model_cpu.nq

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def dof_state(self) -> jax.Array:
        """
        State of the single controllable actor (robot).
        Shape: [num_envs, num_dofs, 2] with interleaved pos/vel, or
        shape: [num_envs, num_dofs * 2].
        """
        qpos = self._data.qpos
        qvel = self._data.qvel
        res = jnp.zeros(
            (self._num_envs, max(self._model_cpu.nq, self._model_cpu.nv), 2)
        )
        res = res.at[:, : self._model_cpu.nq, 0].set(qpos)
        res = res.at[:, : self._model_cpu.nv, 1].set(qvel)
        return res

    def set_dof_state(self, v: jax.Array) -> None:
        """
        Set state of the single controllable actor (robot).
        Shape: [num_envs, num_dofs, 2] with interleaved pos/vel, or
        shape: [num_envs, num_dofs * 2].
        """
        if v.ndim == 3:
            qpos = v[:, : self.nq, 0]
            qvel = v[:, : self.nv, 1]
        else:
            nq = self._model_cpu.nq
            qpos = v[:, :nq]
            qvel = v[:, nq:]

        if qpos.shape[0] == 1:
            qpos = jnp.broadcast_to(qpos, (self._num_envs, self.nq))
        if qvel.shape[0] == 1:
            qvel = jnp.broadcast_to(qvel, (self._num_envs, self.nv))

        self._data = jax.vmap(
            lambda d, q, v: d.replace(qpos=q, qvel=v),
        )(self._data, qpos, qvel)

    def apply_robot_cmd(self, u: jax.Array) -> None:
        """Apply generalized forces (qfrc_applied). u: [num_envs, nv] or [nv]."""
        if u.ndim == 1:
            u = jnp.broadcast_to(u, (self._num_envs, u.shape[0]))
        self._ctrl = u

    def constrained_force(self, u: jax.Array) -> jax.Array:
        if u.ndim == 1:
            u = jnp.broadcast_to(u, (self._num_envs, self.nv))
        if u.shape[0] != self._num_envs:
            u = jnp.broadcast_to(u, (self._num_envs, self.nv))
        data = self._step_fn(self._model, self._data, u)
        return data.qfrc_applied[0]

    def step(self) -> bool:
        """Step simulation."""
        self._data = self._step_fn(self._model, self._data, self._ctrl)
        if isinstance(self._visualize_link_buffer, jax.Array):
            self._visualize_link_buffer = []
        if self._visualize_link_id >= 0:
            self._visualize_link_buffer.append(
                self._data.xpos[:, self._visualize_link_id, :].copy()
            )
        return True

    def build_step_batch_costed(
        self,
        cost_fn: Callable[[mjx.Data, jax.Array], jax.Array],
    ) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
        @jax.jit
        def _step_batch_costed(
            u_batch: jax.Array,
            initial_data: mjx.Data,
            goal_pos: jax.Array,
        ) -> tuple[mjx.Data, jax.Array, jax.Array]:
            def _step_single(
                data: mjx.Data,
                u: jax.Array,
            ) -> tuple[mjx.Data, tuple[jax.Array, jax.Array]]:
                data = self._step_fn(self._model, data, u)
                cost = cost_fn(data, goal_pos)
                pos = jnp.empty(0)
                if self._visualize_link_id >= 0:
                    pos = data.xpos[:, self._visualize_link_id, :]
                return data, (cost, pos)

            initial_data, (costs, link_buffer) = jax.lax.scan(
                _step_single, initial_data, u_batch
            )
            return initial_data, costs, link_buffer

        def step_batch_costed(u_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
            _, costs, self._visualize_link_buffer = _step_batch_costed(
                u_batch, self._data, self._goal_pos
            )
            return costs, cast(jax.Array, self._visualize_link_buffer)

        return step_batch_costed

    def raise_if_nan(self) -> None:
        """Check current state for nan/inf with a single device-host sync.

        NaN propagates through dynamics, so calling this once after a rollout
        is sufficient to detect any nan that occurred during it.
        """
        for arr in (self._data.qacc, self._data.qvel, self._data.qpos):
            if bool(jnp.any(~jnp.isfinite(arr))):
                raise ValueError("nan/inf detected in simulation state")

    def actor_name_to_root_body_id(self, name: str) -> int:
        """Resolve actor name to MuJoCo body id of that actor's root link."""
        body_name = self._actor_root_body.get(name, name)
        body_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise ValueError(f"Actor '{name}' root body '{body_name}' not found")
        return body_id

    def actor_link_name_to_body_id(self, actor_name: str, link_name: str) -> int:
        """Resolve actor link name to MuJoCo body id."""
        body_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, link_name
        )
        if body_id == -1:
            raise ValueError(f"Link '{link_name}' not found")
        return body_id

    def get_actor_position_by_name(self, name: str) -> jax.Array:
        """Root link position of the actor with the given name. Returns [num_envs, 3]."""
        if name == "goal":
            return self._goal_pos
        body_id = self.actor_name_to_root_body_id(name)
        return self._data.xpos[:, body_id, :]

    def get_actor_velocity_by_name(self, name: str) -> jax.Array:
        """Root link linear velocity of the actor with the given name. Returns [num_envs, 3]."""
        if name == "goal":
            return jnp.zeros((self._num_envs, 3), dtype=np.float32)
        body_id = self.actor_name_to_root_body_id(name)
        return self._data.cvel[:, body_id, :3]

    def get_actor_orientation_by_name(self, name: str) -> jax.Array:
        """Root link orientation (quat w, x, y, z) of the actor with the given name. Returns [num_envs, 4]."""
        if name == "goal":
            return self._goal_ori
        body_id = self.actor_name_to_root_body_id(name)
        return self._data.xquat[:, body_id, :]

    def get_actor_link_by_name(self, actor_name: str, link_name: str) -> jax.Array:
        """Returns [num_envs, 3]."""
        link_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, link_name
        )
        if link_id == -1:
            raise ValueError(f"Link '{link_name}' not found")
        return self._data.xpos[:, link_id, :]

    def set_goal_position(self, pos: jax.Array) -> None:
        """Set goal position. pos: [num_envs, 3]."""
        self._goal_pos = self._goal_pos.at[:].set(pos)

    def set_goal_orientation(self, ori: jax.Array) -> None:
        """Set goal orientation. ori: [num_envs, 4]."""
        self._goal_ori = self._goal_ori.at[:].set(ori)

    @property
    def visualize_link_present(self) -> bool:
        return len(self._visualize_link_buffer) > 0

    @property
    def visualize_link_buffer(self) -> jax.Array:
        if isinstance(self._visualize_link_buffer, list):
            return jnp.stack(self._visualize_link_buffer)
        return self._visualize_link_buffer

    def reset_visualize_link_buffer(self) -> None:
        if isinstance(self._visualize_link_buffer, jax.Array):
            self._visualize_link_buffer = []
        self._visualize_link_buffer.clear()

    @property
    def robot_positions(self) -> jax.Array:
        """Positions of the single controllable actor (robot). [num_envs, nq]."""
        return self.get_actor_position_by_name("robot")

    @property
    def robot_velocities(self) -> jax.Array:
        """Velocities of the single controllable actor (robot). [num_envs, nv]."""
        return self.get_actor_velocity_by_name("robot")
