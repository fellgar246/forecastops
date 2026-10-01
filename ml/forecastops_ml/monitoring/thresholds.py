"""Limits for a drift monitoring report.

PSI above ``psi_retrain`` recommends retraining. PSI above ``psi_warning``
and at or below ``psi_retrain`` is a warning. The default retrain threshold
is 0.2. A recent WAPE more than ``wape_degradation_limit`` worse than the
approved model, or a dataset older than ``max_dataset_age_days``, also
recommends retraining. Coverage and other frequency moves above
``frequency_warning_delta`` are warnings.
"""

from dataclasses import dataclass

DEFAULT_PSI_WARNING = 0.1
DEFAULT_PSI_RETRAIN = 0.2
DEFAULT_WAPE_DEGRADATION_LIMIT = 0.2
DEFAULT_MAX_DATASET_AGE_DAYS = 30
DEFAULT_FREQUENCY_WARNING_DELTA = 0.05
DEFAULT_RECENT_WINDOW_DAYS = 28
DEFAULT_PSI_BIN_COUNT = 10


@dataclass(frozen=True)
class DriftThresholds:
    """Snapshot of the limits applied to one monitoring report."""

    psi_warning: float = DEFAULT_PSI_WARNING
    psi_retrain: float = DEFAULT_PSI_RETRAIN
    wape_degradation_limit: float = DEFAULT_WAPE_DEGRADATION_LIMIT
    max_dataset_age_days: int = DEFAULT_MAX_DATASET_AGE_DAYS
    frequency_warning_delta: float = DEFAULT_FREQUENCY_WARNING_DELTA
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS
    psi_bin_count: int = DEFAULT_PSI_BIN_COUNT

    def __post_init__(self) -> None:
        warning = _positive(self.psi_warning, "psi_warning")
        retrain = _positive(self.psi_retrain, "psi_retrain")
        if warning >= retrain:
            raise ValueError("psi_warning must be below psi_retrain.")
        degradation = _positive(self.wape_degradation_limit, "wape_degradation_limit")
        age = _whole(self.max_dataset_age_days, "max_dataset_age_days")
        if age < 0:
            raise ValueError("max_dataset_age_days must be zero or greater.")
        frequency = _non_negative(self.frequency_warning_delta, "frequency_warning_delta")
        window = _whole(self.recent_window_days, "recent_window_days")
        if window < 1:
            raise ValueError("recent_window_days must be at least 1.")
        bins = _whole(self.psi_bin_count, "psi_bin_count")
        if bins < 2:
            raise ValueError("psi_bin_count must be at least 2.")
        object.__setattr__(self, "psi_warning", warning)
        object.__setattr__(self, "psi_retrain", retrain)
        object.__setattr__(self, "wape_degradation_limit", degradation)
        object.__setattr__(self, "max_dataset_age_days", age)
        object.__setattr__(self, "frequency_warning_delta", frequency)
        object.__setattr__(self, "recent_window_days", window)
        object.__setattr__(self, "psi_bin_count", bins)

    def to_dict(self) -> dict[str, float | int]:
        """Return the JSON object for this snapshot."""

        return {
            "frequency_warning_delta": self.frequency_warning_delta,
            "max_dataset_age_days": self.max_dataset_age_days,
            "psi_bin_count": self.psi_bin_count,
            "psi_retrain": self.psi_retrain,
            "psi_warning": self.psi_warning,
            "recent_window_days": self.recent_window_days,
            "wape_degradation_limit": self.wape_degradation_limit,
        }


def _positive(value: float, name: str) -> float:
    if type(value) is not float and type(value) is not int:
        raise ValueError(f"{name} must be a number.")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return float(value)


def _non_negative(value: float, name: str) -> float:
    if type(value) is not float and type(value) is not int:
        raise ValueError(f"{name} must be a number.")
    if float(value) < 0.0:
        raise ValueError(f"{name} must be zero or greater.")
    return float(value)


def _whole(value: int, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer.")
    return value
