"""Thresholds for the promotion quality gate.

A candidate reaches pending approval only when it beats the reference on WAPE,
stays inside the bias limit, covers the P90 band when it emits that quantile,
and has no critical category regression. These defaults are the initial limits.
"""

from dataclasses import dataclass

DEFAULT_MAX_BIAS = 0.05
DEFAULT_MIN_P90_COVERAGE = 0.85
DEFAULT_MAX_SEGMENT_WAPE_REGRESSION = 0.02


@dataclass(frozen=True)
class PromotionThresholds:
    """Snapshot of the limits applied to one gate decision.

    ``max_bias`` is a strict upper bound on ``abs(bias)``. ``min_p90_coverage``
    is the minimum empirical share of actuals at or below ``p90``.
    ``max_segment_wape_regression`` is how far a category WAPE may worsen
    before the regression is critical. A worsening equal to this value is
    still allowed.
    """

    max_bias: float = DEFAULT_MAX_BIAS
    min_p90_coverage: float = DEFAULT_MIN_P90_COVERAGE
    max_segment_wape_regression: float = DEFAULT_MAX_SEGMENT_WAPE_REGRESSION

    def __post_init__(self) -> None:
        if type(self.max_bias) is not float and type(self.max_bias) is not int:
            raise ValueError("max_bias must be a number.")
        if float(self.max_bias) <= 0.0:
            raise ValueError("max_bias must be greater than zero.")
        if not 0.0 <= float(self.min_p90_coverage) <= 1.0:
            raise ValueError("min_p90_coverage must be between 0 and 1.")
        if float(self.max_segment_wape_regression) < 0.0:
            raise ValueError("max_segment_wape_regression must be zero or greater.")
        object.__setattr__(self, "max_bias", float(self.max_bias))
        object.__setattr__(self, "min_p90_coverage", float(self.min_p90_coverage))
        object.__setattr__(
            self,
            "max_segment_wape_regression",
            float(self.max_segment_wape_regression),
        )

    def to_dict(self) -> dict[str, float]:
        """Return the JSON object for this snapshot."""

        return {
            "max_bias": self.max_bias,
            "max_segment_wape_regression": self.max_segment_wape_regression,
            "min_p90_coverage": self.min_p90_coverage,
        }
