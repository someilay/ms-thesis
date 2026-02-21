from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
from dynamics_wrapper import DynamicsWrapper


@dataclass
class ConstraintsConfig:
    pass


@dataclass
class HorizontalPlaneConstraintConfig(ConstraintsConfig):
    kp: float = 100
    kd: float = float(2 * np.sqrt(100))
    anchor_name: str = "ee_link"
    z: float | None = None


@dataclass
class WheelsConstraintConfig(ConstraintsConfig):
    kd: float = float(2 * np.sqrt(100))
    wheel_joints: list[str] = field(
        default_factory=lambda: ["wheel_left_joint", "wheel_right_joint"],
    )
    wheel_radiuses: list[float] = field(default_factory=lambda: [0.08, 0.08])


@dataclass
class PinocchioConfig:
    dt: float = 0.05
    init_pos: list[float] = field(default_factory=lambda: [0, 0, 0])
    init_ori: list[float] = field(default_factory=lambda: [0, 0, 0, 1])
    urdf_path: str = "assets/urdf/boxer/boxer.urdf"
    package_dirs: list[str] = field(default_factory=lambda: ["assets/urdf/boxer"])
    constraints_cfg: list[ConstraintsConfig] = field(
        default_factory=lambda: [
            HorizontalPlaneConstraintConfig(),
            WheelsConstraintConfig(),
        ],
    )
    fixed: bool = False
    numerical_opt: bool = True
    visualize_link: str | None = "chassis_link"


class Constraint(ABC):
    """Abstract base class for holonomic constraints. Form: A @ qddot = b."""

    @abstractmethod
    def compute(
        self,
        q: jax.Array,
        v: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
        model_cpu: mujoco.MjModel,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    @property
    @abstractmethod
    def m(self) -> int:
        raise NotImplementedError


def skew(v: jax.Array) -> jax.Array:
    """Skew-symmetric matrix (3,3)."""
    return jnp.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
    )


def quat_mult(q0: jax.Array, q1: jax.Array) -> jax.Array:
    """Quaternion product (w, x, y, z)."""
    w0, x0, y0, z0 = q0[0], q0[1], q0[2], q0[3]
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    return jnp.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
    )


def quat_deriv(quat: jax.Array, omega: jax.Array) -> jax.Array:
    """d(quat)/dt = 0.5 * quat * [0, omega] for body-frame angular velocity omega."""
    return 0.5 * quat_mult(quat, jnp.concatenate([jnp.array([0.0]), omega]))


JNT_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
JNT_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
JNT_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
JNT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


def qvel_to_qpos_deriv(model: mjx.Model, q: jax.Array, v: jax.Array) -> jax.Array:
    """dq/dt from (q, v). Maps each joint's v to dq/dt via model.jnt_type, jnt_qposadr, jnt_dofadr."""
    nq, njnt = model.nq, model.njnt
    dq_dt = jnp.zeros(nq)

    def seg_free(
        q: jax.Array, v: jax.Array, qadr: jax.Array, dofadr: jax.Array
    ) -> jax.Array:
        return jnp.concatenate(
            [
                jax.lax.dynamic_slice(v, (dofadr,), (3,)),
                quat_deriv(
                    jax.lax.dynamic_slice(q, (qadr + 3,), (4,)),
                    jax.lax.dynamic_slice(v, (dofadr + 3,), (3,)),
                ),
            ]
        )

    def seg_ball(
        q: jax.Array, v: jax.Array, qadr: jax.Array, dofadr: jax.Array
    ) -> jax.Array:
        return quat_deriv(
            jax.lax.dynamic_slice(q, (qadr,), (4,)),
            jax.lax.dynamic_slice(v, (dofadr,), (3,)),
        )

    def seg_slide(
        q: jax.Array, v: jax.Array, qadr: jax.Array, dofadr: jax.Array
    ) -> jax.Array:
        return jax.lax.dynamic_slice(v, (dofadr,), (1,))

    def seg_hinge(
        q: jax.Array, v: jax.Array, qadr: jax.Array, dofadr: jax.Array
    ) -> jax.Array:
        return jax.lax.dynamic_slice(v, (dofadr,), (1,))

    def body(i: int, dq_dt_val: jax.Array) -> jax.Array:
        jnt_type = jax.lax.dynamic_slice(model.jnt_type, (i,), (1,))[0]
        qadr = jax.lax.dynamic_slice(model.jnt_qposadr, (i,), (1,))[0]
        dofadr = jax.lax.dynamic_slice(model.jnt_dofadr, (i,), (1,))[0]
        return jax.lax.switch(
            jnt_type,
            [
                lambda: jax.lax.dynamic_update_slice(
                    dq_dt_val, seg_free(q, v, qadr, dofadr), (qadr,)
                ),
                lambda: jax.lax.dynamic_update_slice(
                    dq_dt_val, seg_ball(q, v, qadr, dofadr), (qadr,)
                ),
                lambda: jax.lax.dynamic_update_slice(
                    dq_dt_val, seg_slide(q, v, qadr, dofadr), (qadr,)
                ),
                lambda: jax.lax.dynamic_update_slice(
                    dq_dt_val, seg_hinge(q, v, qadr, dofadr), (qadr,)
                ),
            ],
        )

    return jax.lax.fori_loop(0, njnt, body, dq_dt)


def body_jacobian_time_derivative(
    model: mjx.Model,
    data: mjx.Data,
    body_id: int,
    q: jax.Array,
    v: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Time derivative of body Jacobians at body origin: dJ/dt = (dJ/dq) @ dq_dt.
    Returns (dJ_pos_dt, dJ_rot_dt) each (3, nv). v_point_dot = J_pos @ qacc + dJ_pos_dt @ v.
    """
    body_id_jax = jnp.int32(body_id)
    dq_dt = qvel_to_qpos_deriv(model, q, v)

    def j_pos_fn(q_arg: jax.Array) -> jax.Array:
        d = mjx.forward(model, data.replace(qpos=q_arg, qvel=v))
        jacp, _ = mjx_support.jac(model, d, d.xpos[body_id], body_id_jax)
        return jacp.T

    def j_rot_fn(q_arg: jax.Array) -> jax.Array:
        d = mjx.forward(model, data.replace(qpos=q_arg, qvel=v))
        _, jacr = mjx_support.jac(model, d, d.xpos[body_id], body_id_jax)
        return jacr.T

    djac_pos_dq = jax.jacfwd(j_pos_fn)(q)
    djac_rot_dq = jax.jacfwd(j_rot_fn)(q)
    djac_pos_dt = jnp.einsum("ijn,n->ij", djac_pos_dq, dq_dt)
    djac_rot_dt = jnp.einsum("ijn,n->ij", djac_rot_dq, dq_dt)
    return djac_pos_dt, djac_rot_dt


class HorizontalPlaneConstraint(Constraint):
    """Holonomic constraint: anchor body z = z_ref, anchor z-axis horizontal. c_z = anchor.z - z_ref, c_orient = (xmat[anchor] @ r)[:2]."""

    def __init__(
        self,
        model_cpu: mujoco.MjModel,
        data_cpu: mujoco.MjData,
        cfg: HorizontalPlaneConstraintConfig,
    ):
        mujoco.mj_forward(model_cpu, data_cpu)
        self._anchor_name = cfg.anchor_name
        self._anchor_id = mujoco.mj_name2id(
            model_cpu, mujoco.mjtObj.mjOBJ_BODY, self._anchor_name
        )
        if self._anchor_id < 0:
            raise ValueError(f"Anchor body '{self._anchor_name}' not found")
        xpos = data_cpu.xpos[self._anchor_id]
        ori = data_cpu.xmat[self._anchor_id].reshape(3, 3)
        self._z_ref = jnp.float32(cfg.z if cfg.z is not None else xpos[2])
        self._kp = cfg.kp
        self._kd = cfg.kd
        self._r = jnp.array(ori[2, :].copy(), dtype=jnp.float32)
        self._anchor_id_jax = jnp.int32(self._anchor_id)

    def compute(
        self,
        q: jax.Array,
        v: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
        model_cpu: mujoco.MjModel,
    ) -> tuple[jax.Array, jax.Array]:
        bid = self._anchor_id
        anchor_pos = data.xpos[bid]
        c_z = anchor_pos[2] - self._z_ref
        jacp, jacr = mjx_support.jac(model, data, anchor_pos, self._anchor_id_jax)
        jac_pos_z = jacp.T[2:3, :]
        dc_z = (jac_pos_z @ v)[0]

        djac_pos_dt, djac_rot_dt = body_jacobian_time_derivative(model, data, bid, q, v)
        drift_z = jnp.dot(djac_pos_dt[2, :], v)

        rot_cur = data.xmat[bid].reshape(3, 3)
        v_vec = rot_cur @ self._r
        c_orient = v_vec[:2]
        jac_ang = jacr.T
        jac_orient = (-skew(v_vec) @ jac_ang)[:2, :]
        omega = jacr.T @ v
        dv_vec = jnp.cross(omega, v_vec)
        dc_orient = dv_vec[:2]
        drift_orient_1 = skew(v_vec) @ djac_rot_dt @ v
        drift_orient_2 = skew(dv_vec) @ jac_ang @ v
        drift_orient = drift_orient_1[:2] + drift_orient_2[:2]

        a_mat = jnp.vstack([jac_pos_z, jac_orient])
        c_vec = jnp.concatenate([jnp.array([c_z]), c_orient])
        dc_vec = jnp.concatenate([jnp.array([dc_z]), dc_orient])
        drift_vec = jnp.concatenate([jnp.array([drift_z]), drift_orient])
        b_vec = -drift_vec - self._kp * c_vec - self._kd * dc_vec
        return a_mat, b_vec

    @property
    def m(self) -> int:
        return 3


class WheelsConstraint(Constraint):
    """No-slip rolling: c_w = (v_lin_w)^xy - theta_dot*r*(a x e_z)^xy. d(c_w)/dt = A_w @ a + drift_w,
    drift_w = (d(c_w)/dq) @ dq_dt. Stabilization: A_w @ a = b_w with b_w = -k_d*c_w - drift_w."""

    def __init__(self, model_cpu: mujoco.MjModel, cfg: WheelsConstraintConfig):
        self._wheel_joint_names = cfg.wheel_joints
        self._wheel_radiuses = cfg.wheel_radiuses
        self._kd = cfg.kd
        self._wheel_indices = [
            mujoco.mj_name2id(model_cpu, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in cfg.wheel_joints
        ]
        if any(wheel_id < 0 for wheel_id in self._wheel_indices):
            raise ValueError(f"Wheel joints '{cfg.wheel_joints}' not found")
        self._axes = [
            jnp.array(model_cpu.jnt_axis[wheel_id]) for wheel_id in self._wheel_indices
        ]
        self._parent_ids = [
            model_cpu.body_parentid[wheel_id] for wheel_id in self._wheel_indices
        ]

    def compute(
        self,
        q: jax.Array,
        v: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
        model_cpu: mujoco.MjModel,
    ) -> tuple[jax.Array, jax.Array]:
        nv = model.nv
        base_axes = jnp.eye(3)
        jac_blocks = []
        c_blocks = []
        drift_blocks = []
        dq_dt = qvel_to_qpos_deriv(model, q, v)

        for wheel_id, parent_id, radius, axis in zip(
            self._wheel_indices,
            self._parent_ids,
            self._wheel_radiuses,
            self._axes,
        ):
            dof_id = model_cpu.jnt_dofadr[wheel_id]
            body_id = model_cpu.jnt_bodyid[wheel_id]

            def c_w_fn(q_arg: jax.Array) -> jax.Array:
                d = mjx.forward(model, data.replace(qpos=q_arg, qvel=v))
                R = d.xmat[parent_id].reshape(3, 3) if parent_id >= 0 else d.xmat[body_id].reshape(3, 3)
                jac_p, _ = mjx_support.jac(model, d, d.xpos[body_id], body_id)
                jac_lin = R.T @ jac_p.T
                vel_lin = jac_lin @ v
                exp_vel = v[dof_id] * radius * jnp.cross(axis, base_axes[2])
                return (vel_lin - exp_vel)[:2]

            if parent_id < 0:
                parent_to_world_rot = data.xmat[body_id].reshape(3, 3)
            else:
                parent_to_world_rot = data.xmat[parent_id].reshape(3, 3)

            body_pos = data.xpos[body_id]
            jac_p, _ = mjx_support.jac(model, data, body_pos, body_id)
            jac_lin_parent = parent_to_world_rot.T @ jac_p.T
            vel_lin_parent = jac_lin_parent @ v
            exp_vel_parent = v[dof_id] * radius * jnp.cross(axis, base_axes[2])
            c_blocks.append((vel_lin_parent - exp_vel_parent)[:2])

            dc_w_dq = jax.jacfwd(c_w_fn)(q)
            drift_blocks.append(dc_w_dq @ dq_dt)

            d = jnp.cross(base_axes[2], axis)
            a_lin_parent = jac_lin_parent.at[:, dof_id].add(d * radius)
            jac_blocks.append(a_lin_parent[:2, :])

        a_mat = jnp.vstack(jac_blocks)
        c = jnp.concatenate(c_blocks)
        drift = jnp.concatenate(drift_blocks)
        b = -self._kd * c - drift
        return a_mat, b

    @property
    def m(self) -> int:
        return 2 * len(self._wheel_joint_names)


class MjxWrapper(DynamicsWrapper):
    """MuJoCo MJX wrapper with parallel environments and holonomic constraints.

    Implements constrained dynamics using Gauss least action principle:
    minimize: 0.5 * (a - a_free)^T mass (a - a_free)
    subject to: constraint_jac @ a = constraint_bias

    where a is constrained acceleration, a_free is unconstrained acceleration.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        num_envs: int = 1,
        constraints: list[Constraint] | None = None,
        device: str = "cpu",
        actor_root_body: dict[str, str] | None = None,
    ):
        """
        actor_root_body: map actor name -> root body (link) name.
        E.g. {"robot": "base_link"} so get_actor_position_by_name("robot") returns base_link position.
        If an actor name is not in the map, it is used as the body name (lookup by body name).
        """
        self._model_cpu = model
        self._model = mjx.put_model(model)
        self._num_envs = num_envs
        self._constraints = constraints or []
        self._device = device
        self._dt = model.opt.timestep
        self._actor_root_body = actor_root_body or {}

        data_cpu = mujoco.MjData(model)
        mujoco.mj_forward(model, data_cpu)

        self._data = jax.vmap(
            lambda _: mjx.put_data(model, data_cpu),
        )(jnp.arange(num_envs))

        self._ctrl = jnp.zeros((num_envs, model.nu))
        self._visualize_link_buffer: list[jax.Array] = []

        self._step_fn = jax.jit(jax.vmap(self._step_single, in_axes=(None, 0, 0)))

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
                data.qpos, data.qvel, model, data, self._model_cpu
            )
            if jac.shape[0] > 0:
                constraint_jac_blocks.append(jac)
                constraint_bias_blocks.append(bias)

        if not constraint_jac_blocks or not constraint_bias_blocks:
            return mjx.step(model, data)

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
    def device(self) -> str:
        return self._device

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
        return jnp.stack([qpos, qvel], axis=-1)

    def set_dof_state(self, v: jax.Array) -> None:
        """
        Set state of the single controllable actor (robot).
        Shape: [num_envs, num_dofs, 2] with interleaved pos/vel, or
        shape: [num_envs, num_dofs * 2].
        """
        if v.ndim == 3:
            qpos = v[:, :, 0]
            qvel = v[:, :, 1]
        else:
            nq = self._model_cpu.nq
            qpos = v[:, :nq]
            qvel = v[:, nq:]

        self._data = jax.vmap(
            lambda d, q, v: d.replace(qpos=q, qvel=v),
        )(self._data, qpos, qvel)

    def apply_robot_cmd(self, u: jax.Array) -> None:
        """Apply control to the single controllable actor (robot). u: [num_envs, nu] or [nu]."""
        if u.ndim == 1:
            u = jnp.broadcast_to(u, (self._num_envs, u.shape[0]))
        self._ctrl = u

    def step(self) -> bool:
        """Step simulation. Returns False to exit."""
        self._data = self._step_fn(self._model, self._data, self._ctrl)
        return True

    def _actor_name_to_root_body_id(self, name: str) -> int:
        """Resolve actor name to MuJoCo body id of that actor's root link."""
        body_name = self._actor_root_body.get(name, name)
        body_id = mujoco.mj_name2id(
            self._model_cpu, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise ValueError(f"Actor '{name}' root body '{body_name}' not found")
        return body_id

    def get_actor_position_by_name(self, name: str) -> jax.Array:
        """Root link position of the actor with the given name. Returns [num_envs, 3]."""
        body_id = self._actor_name_to_root_body_id(name)
        return self._data.xpos[:, body_id, :]

    def get_actor_velocity_by_name(self, name: str) -> jax.Array:
        """Root link linear velocity of the actor with the given name. Returns [num_envs, 3]."""
        body_id = self._actor_name_to_root_body_id(name)
        return self._data.cvel[:, body_id, :3]

    def get_actor_orientation_by_name(self, name: str) -> jax.Array:
        """Root link orientation (quat w, x, y, z) of the actor with the given name. Returns [num_envs, 4]."""
        body_id = self._actor_name_to_root_body_id(name)
        return self._data.xquat[:, body_id, :]

    @property
    def visualize_link_present(self) -> bool:
        return len(self._visualize_link_buffer) > 0

    @property
    def visualize_link_buffer(self) -> list[jax.Array]:
        return self._visualize_link_buffer

    def reset_visualize_link_buffer(self) -> None:
        self._visualize_link_buffer.clear()

    @property
    def robot_positions(self) -> jax.Array:
        """Positions of the single controllable actor (robot). [num_envs, nq]."""
        return self._data.qpos

    @property
    def robot_velocities(self) -> jax.Array:
        """Velocities of the single controllable actor (robot). [num_envs, nv]."""
        return self._data.qvel
