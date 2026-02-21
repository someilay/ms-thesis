"""Abstract base class for dynamics simulation wrappers."""

from abc import ABC, abstractmethod
import jax


class DynamicsWrapper(ABC):
    """
    An environment may have several actors; only one is controllable (robot).
    dof_state, set_dof_state, apply_robot_cmd, robot_positions, robot_velocities
    refer to that single controllable actor. Each actor has a name and a root
    link (the link with no parent). get_actor_*_by_name(name) return the root
    link position/velocity/orientation of the actor with that name.
    """

    @property
    @abstractmethod
    def num_envs(self) -> int:
        pass

    @property
    @abstractmethod
    def device(self) -> jax.Device:
        pass

    @property
    @abstractmethod
    def nv(self) -> int:
        pass

    @property
    @abstractmethod
    def nq(self) -> int:
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        pass

    @property
    @abstractmethod
    def dof_state(self) -> jax.Array:
        """
        Shape: [num_envs, num_dofs, 2] with interleaved pos/vel, or
        shape: [num_envs, num_dofs * 2].
        """
        pass

    @abstractmethod
    def set_dof_state(self, v: jax.Array) -> None:
        """
        Shape: [num_envs, num_dofs, 2] with interleaved pos/vel, or
        shape: [num_envs, num_dofs * 2].
        """
        pass

    @abstractmethod
    def apply_robot_cmd(self, u: jax.Array) -> None:
        """Apply control to the single controllable actor (robot). u: [num_envs, nu] or [nu]."""
        pass

    @abstractmethod
    def step(self) -> bool:
        """Step simulation. Returns False to exit."""
        pass

    @abstractmethod
    def get_actor_position_by_name(self, name: str) -> jax.Array:
        """Root link position of the actor with the given name. Returns [num_envs, 3]."""
        pass

    @abstractmethod
    def get_actor_velocity_by_name(self, name: str) -> jax.Array:
        """Root link linear velocity of the actor with the given name. Returns [num_envs, 3]."""
        pass

    @abstractmethod
    def get_actor_orientation_by_name(self, name: str) -> jax.Array:
        """Root link orientation (quat w, x, y, z) of the actor with the given name. Returns [num_envs, 4]."""
        pass

    @property
    @abstractmethod
    def visualize_link_present(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def visualize_link_buffer(self) -> list[jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def reset_visualize_link_buffer(self) -> None:
        raise NotImplementedError

    @property
    def robot_positions(self) -> jax.Array:
        raise NotImplementedError

    @property
    def robot_velocities(self) -> jax.Array:
        raise NotImplementedError
