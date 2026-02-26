import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support


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
