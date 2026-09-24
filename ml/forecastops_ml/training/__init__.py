"""Model training entry points.

:class:`GradientBoostingForecaster` fits one global LightGBM model. ``fit``
and ``predict`` both call :func:`forecastops_ml.features.build_features`.
"""

from forecastops_ml.training.gradient_boosting import GradientBoostingForecaster

__all__ = ["GradientBoostingForecaster"]
