
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import mujoco
from mujoco import mjx


class Constraint(ABC):
    """Abstract base class for holonomic constraints. Form: A @ v_dot = b."""

    @abstractmethod
    def compute(
        self,
        q: jax.Array,
        v: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
    ) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    @property
    @abstractmethod
    def m(self) -> int:
        raise NotImplementedError


@dataclass
class ConstraintsConfig(ABC):
    @abstractmethod
    def build_constraint(self, model: mujoco.MjModel, data: mujoco.MjData) -> Constraint:
        raise NotImplementedError
