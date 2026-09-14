from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DynamicModel(ABC):
    state_names: tuple[str, ...]

    @property
    def nx(self) -> int:
        return len(self.state_names)

    @abstractmethod
    def discrete_dynamics(
        self, x: np.ndarray, *, delta: float, vx: float, dt: float
    ) -> np.ndarray:
        pass

    @abstractmethod
    def transition_matrices(
        self, *, delta: float, vx: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return F, B for x[k+1] = F x[k] + B delta[k]."""
        pass
