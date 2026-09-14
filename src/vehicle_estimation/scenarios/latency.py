from __future__ import annotations

from dataclasses import replace

from vehicle_estimation.core.types import Event
from .base import ScenarioTransform


class FixedLatency(ScenarioTransform):
    """Delay arrival time while preserving physical measurement time."""

    def __init__(self, source: str, latency_s: float):
        self.source = source
        self.latency_s = float(latency_s)

    def apply(self, events: list[Event]) -> list[Event]:
        return [
            replace(e, arrival_time=e.arrival_time + self.latency_s)
            if e.source == self.source
            else e
            for e in events
        ]
