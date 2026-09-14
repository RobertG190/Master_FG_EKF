from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from vehicle_estimation.core.types import Event


class Estimator(ABC):
    @abstractmethod
    def process(self, event: Event) -> None:
        pass

    @abstractmethod
    def estimates_frame(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def diagnostics(self) -> dict:
        pass
