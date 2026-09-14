from __future__ import annotations

from abc import ABC, abstractmethod

from vehicle_estimation.core.types import Event


class ScenarioTransform(ABC):
    """Pure transform from canonical events to modified experiment events."""

    @abstractmethod
    def apply(self, events: list[Event]) -> list[Event]:
        pass
