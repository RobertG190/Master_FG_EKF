from __future__ import annotations

from abc import ABC, abstractmethod


class InformationMetric(ABC):
    """Future contract for excitation/observability-aware parameter policies."""

    @abstractmethod
    def score(self, *args, **kwargs) -> float:
        pass
