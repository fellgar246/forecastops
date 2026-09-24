"""Tree SHAP drivers for one prediction row.

Drivers are associations from the tree model. Field names stay
``positive_drivers`` and ``negative_drivers``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Driver:
    """One feature's SHAP value on a single prediction."""

    feature: str
    shap_value: float


@dataclass(frozen=True)
class RowAttribution:
    """Top positive and top negative SHAP drivers for one row."""

    positive_drivers: tuple[Driver, ...]
    negative_drivers: tuple[Driver, ...]


def shap_drivers(
    contributions: Sequence[float],
    names: Sequence[str],
    *,
    top_n: int = 5,
) -> RowAttribution:
    """Return the largest positive and negative SHAP values.

    ``contributions`` aligns with ``names`` and omits the model bias term.
    Zero contributions are left out. Each side keeps at most ``top_n``
    drivers, ordered by magnitude.
    """

    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if len(contributions) != len(names):
        raise ValueError("SHAP contributions must have one value per feature.")
    if any(not isinstance(name, str) or name == "" for name in names):
        raise ValueError("SHAP feature names must be non-empty strings.")
    pairs: list[tuple[str, float]] = []
    for name, value in zip(names, contributions, strict=True):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"SHAP value for {name} is not finite.")
        pairs.append((name, number))
    positive = sorted(
        (pair for pair in pairs if pair[1] > 0.0),
        key=lambda pair: (-pair[1], pair[0]),
    )
    negative = sorted(
        (pair for pair in pairs if pair[1] < 0.0),
        key=lambda pair: (pair[1], pair[0]),
    )
    return RowAttribution(
        positive_drivers=tuple(Driver(feature, value) for feature, value in positive[:top_n]),
        negative_drivers=tuple(Driver(feature, value) for feature, value in negative[:top_n]),
    )
