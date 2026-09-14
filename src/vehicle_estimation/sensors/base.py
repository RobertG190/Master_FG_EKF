from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MeasurementModel(ABC):
    source: str

    @abstractmethod
    def predict(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian_x(self, x: np.ndarray, *, delta: float, vx: float) -> np.ndarray:
        pass

    @abstractmethod
    def default_covariance(self) -> np.ndarray:
        pass
