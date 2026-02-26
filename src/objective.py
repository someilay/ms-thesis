from abc import ABC, abstractmethod
from typing import Callable

import jax
from mujoco import mjx

from mjx_wrapper import MjxWrapper


class Objective(ABC):
    @abstractmethod
    def build_cost_fn(
        self,
        sim: MjxWrapper,
    ) -> Callable[[mjx.Data, jax.Array], jax.Array]:
        raise NotImplementedError
