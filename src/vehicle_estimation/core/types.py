from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class EventKind(str, Enum):
    INPUT = "input"
    MEASUREMENT = "measurement"


@dataclass(frozen=True)
class Event:
    """Canonical estimator-facing event.

    measurement_time: physical timestamp at which the signal applies.
    arrival_time: timestamp at which the estimator receives it.

    Ground truth deliberately does not use this type.
    """

    measurement_time: float
    arrival_time: float
    kind: EventKind
    source: str
    value: np.ndarray
    covariance: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Estimate:
    time: float
    state: np.ndarray
    covariance: np.ndarray
    trigger: str
