"""Rolling-origin folds.

Folds are anchored on the latest dates in the frame. Each validation window
is ``horizon`` distinct dates long, windows do not share dates, and the next
window starts strictly after the previous origin. Training for a fold is every
row strictly before that origin, so earlier validation rows become history for
later folds.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

MIN_PUBLISHED_FOLDS = 3
PREFERRED_BENCHMARK_FOLDS = 5


@dataclass(frozen=True)
class Fold:
    """One validation window and the origin that opens it.

    ``origin`` is the first validation date. ``index`` starts at 1.
    """

    index: int
    origin: date
    validation_dates: tuple[date, ...]

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("Fold index starts at 1.")
        if not self.validation_dates:
            raise ValueError("A fold needs at least one validation date.")
        if tuple(sorted(self.validation_dates)) != self.validation_dates:
            raise ValueError("Validation dates must be sorted.")
        if self.validation_dates[0] != self.origin:
            raise ValueError("A fold origin is the first validation date.")


def build_folds(dates: Sequence[date], *, folds: int, horizon: int) -> tuple[Fold, ...]:
    """Build expanding-window origins from the distinct dates in ``dates``.

    A period is one distinct calendar date already present in the frame, at
    that frame's grain. The last ``folds * horizon`` dates become validation.
    At least one earlier date is required so every fold has training history.
    """

    if type(folds) is not int or folds < 1:
        raise ValueError("Fold count must be a positive integer.")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Horizon must be a positive number of periods.")
    if any(type(value) is not date for value in dates):
        raise ValueError("Fold dates must be calendar dates.")

    unique = tuple(sorted(set(dates)))
    if not unique:
        raise ValueError("Cannot build folds from an empty date list.")
    needed = folds * horizon
    if len(unique) < needed + 1:
        raise ValueError(
            f"The series has {len(unique)} distinct dates, which is too short for "
            f"{folds} folds of horizon {horizon}. At least {needed + 1} dates are "
            "required so each fold keeps training history."
        )

    start = len(unique) - needed
    built: list[Fold] = []
    previous_end: date | None = None
    for index in range(folds):
        window = unique[start + index * horizon : start + (index + 1) * horizon]
        origin = window[0]
        if previous_end is not None and origin <= previous_end:
            raise ValueError("Each validation window must start after the previous origin.")
        built.append(Fold(index=index + 1, origin=origin, validation_dates=window))
        previous_end = window[-1]
    return tuple(built)
