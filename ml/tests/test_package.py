"""Import smoke test for the forecasting library layout."""

import forecastops_ml.baselines
import forecastops_ml.data
import forecastops_ml.evaluation
import forecastops_ml.explainability
import forecastops_ml.features
import forecastops_ml.inference
import forecastops_ml.pipelines
import forecastops_ml.training


def test_library_modules_import() -> None:
    modules = [
        forecastops_ml.data,
        forecastops_ml.features,
        forecastops_ml.baselines,
        forecastops_ml.training,
        forecastops_ml.evaluation,
        forecastops_ml.inference,
        forecastops_ml.explainability,
        forecastops_ml.pipelines,
    ]
    assert all(module.__doc__ for module in modules)
