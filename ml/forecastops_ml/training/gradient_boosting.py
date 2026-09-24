"""Global gradient-boosted demand model.

The public family name is ``gradient_boosting``. The library name lives on
``training_config``. The default grain is store-SKU week. Daily grain is
accepted only when ``profile`` is ``full``. Hyperparameters are fixed.
``fit`` and ``predict`` both call :func:`build_features`.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from forecastops_ml.data.synthetic import DEFAULT_SEED
from forecastops_ml.evaluation.backtest import ForecastFrame, SkippedSeries, make_series_id
from forecastops_ml.explainability.shap import RowAttribution, shap_drivers
from forecastops_ml.features.build import MODEL_FEATURES, FeatureFrame, build_features

F64 = NDArray[np.float64]
_CATEGORICAL = ("store_id", "sku_id", "category_id")
_INPUTS = (
    "date",
    "store_id",
    "sku_id",
    "category_id",
    "units_sold",
    "price",
    "discount_pct",
    "promotion",
    "holiday",
)


class GradientBoostingForecaster:
    """Fit one LightGBM model on store-SKU history.

    ``skipped_series`` lists series left out of the latest fit. A daily frame
    is skipped unless this instance was built with ``grain="day"`` and
    ``profile="full"``. ``attribute`` returns SHAP drivers for one row.
    """

    model_family = "gradient_boosting"

    def __init__(
        self,
        *,
        grain: str = "week",
        profile: str = "test",
        seed: int = DEFAULT_SEED,
    ) -> None:
        if grain not in {"day", "week"}:
            raise ValueError("Gradient boosting grain must be day or week.")
        if grain == "day" and profile != "full":
            raise ValueError("Daily grain is allowed only when profile is full.")
        if type(seed) is not int:
            raise ValueError("random seed must be an integer.")
        self.grain = grain
        self.profile = profile
        self.seed = seed
        self.training_config: dict[str, object] = {
            "bagging_fraction": 1.0,
            "deterministic": True,
            "feature_fraction": 1.0,
            "grain": grain,
            "learning_rate": 0.1,
            "library": "lightgbm",
            "min_data_in_leaf": 1,
            "num_boost_round": 40,
            "num_leaves": 15,
            "profile": profile,
            "random_seed": seed,
        }
        self.fit_runtime_ms = 0
        self.skipped_series: tuple[SkippedSeries, ...] = ()
        self._booster: Any = None
        self._history: pa.Table | None = None
        self._cutoff: date | None = None
        self._encoders: dict[str, dict[str, int]] = {}

    def fit(self, history: pa.Table, config: Mapping[str, object] | None) -> None:
        """Fit on rows in ``history``. Encoders see only those rows."""

        _ = config
        self._booster = None
        self._history = None
        self._cutoff = None
        self._encoders = {}
        self.skipped_series = ()
        if not _grain_matches(history, self.grain):
            self.skipped_series = _skip_all(
                history,
                "Gradient boosting trains on store-SKU weeks. "
                "This frame is not a Monday-start weekly aggregate.",
            )
            return
        prepared = _model_frame(history, with_target=True)
        cutoff = max(_dates(prepared))
        features = build_features(prepared, cutoff, {"grain": self.grain})
        labels = _targets(features)
        self._encoders = _fit_encoders(features.frame)
        matrix = _design_matrix(features.frame, self._encoders)
        self._booster = _train(matrix, labels, self.training_config)
        self._history = prepared
        self._cutoff = cutoff

    def predict(self, horizon: int, known_future: pa.Table) -> ForecastFrame:
        """Forecast ``p50`` for each known-future key. Quantiles stay null."""

        if type(horizon) is not int or horizon < 1:
            raise ValueError("Horizon must be a positive number of periods.")
        booster = self._require_model()
        future = _model_frame(known_future, with_target=False)
        features = self._features_for(future)
        matrix = _design_matrix(features.frame, self._encoders)
        predicted = np.asarray(booster.predict(matrix), dtype=np.float64)
        if predicted.shape != (features.frame.num_rows,):
            raise ValueError("Gradient boosting returned an unexpected number of forecasts.")
        if not np.all(np.isfinite(predicted)):
            raise ValueError("Gradient boosting returned a non-finite forecast.")
        return ForecastFrame(
            series_id=tuple(_series_ids(features.frame)),
            date=tuple(_dates(features.frame)),
            p50=tuple(float(value) for value in predicted.tolist()),
        )

    def attribute(self, known_future: pa.Table, *, top_n: int = 5) -> RowAttribution:
        """Return the top positive and negative SHAP drivers for one row."""

        if known_future.num_rows != 1:
            raise ValueError("Attribution takes one prediction row.")
        booster = self._require_model()
        future = _model_frame(known_future, with_target=False)
        features = self._features_for(future)
        matrix = _design_matrix(features.frame, self._encoders)
        contributions = np.asarray(booster.predict(matrix, pred_contrib=True), dtype=np.float64)
        if contributions.ndim != 2 or contributions.shape[1] != len(MODEL_FEATURES) + 1:
            raise ValueError("SHAP contributions do not match the model features.")
        return shap_drivers(contributions[0, :-1].tolist(), MODEL_FEATURES, top_n=top_n)

    def _features_for(self, future: pa.Table) -> FeatureFrame:
        history = self._history
        cutoff = self._cutoff
        if history is None or cutoff is None:
            raise ValueError("Gradient boosting must be fit before it can forecast.")
        combined = _concat_history(history, future)
        features = build_features(combined, cutoff, {"grain": self.grain})
        return _select_keys(features, future)

    def _require_model(self) -> Any:
        if self._booster is None:
            raise ValueError("Gradient boosting must be fit before it can forecast.")
        return self._booster


def _train(matrix: F64, labels: F64, training_config: Mapping[str, object]) -> Any:
    import lightgbm as lgb

    seed = int(cast(int, training_config["random_seed"]))
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": training_config["learning_rate"],
        "num_leaves": training_config["num_leaves"],
        "min_data_in_leaf": training_config["min_data_in_leaf"],
        "feature_fraction": training_config["feature_fraction"],
        "bagging_fraction": training_config["bagging_fraction"],
        "bagging_freq": 0,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
        "verbosity": -1,
        "feature_pre_filter": False,
    }
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        feature_name=list(MODEL_FEATURES),
        categorical_feature=list(_CATEGORICAL),
        free_raw_data=False,
    )
    rounds = int(cast(int, training_config["num_boost_round"]))
    return lgb.train(params, dataset, num_boost_round=rounds)


def _fit_encoders(frame: pa.Table) -> dict[str, dict[str, int]]:
    encoders: dict[str, dict[str, int]] = {}
    for name in _CATEGORICAL:
        labels = sorted({str(value) for value in frame.column(name).to_pylist()})
        encoders[name] = {label: index for index, label in enumerate(labels)}
    return encoders


def _design_matrix(frame: pa.Table, encoders: Mapping[str, Mapping[str, int]]) -> F64:
    matrix = np.full((frame.num_rows, len(MODEL_FEATURES)), np.nan, dtype=np.float64)
    for column, name in enumerate(MODEL_FEATURES):
        if name in _CATEGORICAL:
            codes = encoders[name]
            for row, value in enumerate(frame.column(name).to_pylist()):
                matrix[row, column] = float(codes.get(str(value), -1))
            continue
        if name in {"promotion", "holiday"}:
            for row, value in enumerate(frame.column(name).to_pylist()):
                if value is not None:
                    matrix[row, column] = 1.0 if value else 0.0
            continue
        for row, value in enumerate(frame.column(name).to_pylist()):
            if value is not None:
                matrix[row, column] = float(value)
    return matrix


def _targets(features: FeatureFrame) -> F64:
    values = features.frame.column("units_sold").to_pylist()
    if any(value is None for value in values):
        raise ValueError("Training features include a row without units_sold.")
    labels = np.asarray([float(value) for value in values], dtype=np.float64)
    if not np.all(np.isfinite(labels)):
        raise ValueError("Training targets must be finite.")
    return labels


def _grain_matches(frame: pa.Table, grain: str) -> bool:
    if grain == "day":
        return True
    if "date" not in frame.column_names:
        return False
    try:
        days = _dates(frame)
    except ValueError:
        return False
    return all(day.weekday() == 0 for day in days)


def _skip_all(frame: pa.Table, reason: str) -> tuple[SkippedSeries, ...]:
    series_ids = sorted(set(_series_ids(frame)))
    return tuple(SkippedSeries(series_id=series_id, reason=reason) for series_id in series_ids)


def _model_frame(frame: pa.Table, *, with_target: bool) -> pa.Table:
    if not isinstance(frame, pa.Table):
        raise ValueError("Observation frame must be a pyarrow table.")
    if frame.num_rows == 0:
        raise ValueError("Observation frame has no rows.")
    columns: dict[str, pa.Array] = {}
    for name in _INPUTS:
        if name == "units_sold" and not with_target:
            columns[name] = pa.array([None] * frame.num_rows, type=pa.float64())
            continue
        if name not in frame.column_names:
            if name in {"discount_pct", "promotion", "holiday"}:
                continue
            raise ValueError(f"Observation frame is missing {name}.")
        column = frame.column(name).combine_chunks()
        numeric = name in {"price", "discount_pct"} and pa.types.is_integer(column.type)
        if name == "units_sold" or numeric:
            column = column.cast(pa.float64())
        columns[name] = column
    return pa.table(columns)


def _concat_history(history: pa.Table, future: pa.Table) -> pa.Table:
    future_keys = set(zip(_series_ids(future), _dates(future), strict=True))
    history_keys = list(zip(_series_ids(history), _dates(history), strict=True))
    keep = [key not in future_keys for key in history_keys]
    trimmed = history.filter(pa.array(keep))
    names = [
        name for name in _INPUTS if name in trimmed.column_names or name in future.column_names
    ]
    return pa.concat_tables(
        [_aligned(trimmed, names), _aligned(future, names)],
        promote_options="default",
    )


def _aligned(frame: pa.Table, names: Sequence[str]) -> pa.Table:
    columns: dict[str, pa.Array] = {}
    for name in names:
        if name in frame.column_names:
            columns[name] = frame.column(name).combine_chunks()
        elif name == "units_sold" or name == "discount_pct":
            columns[name] = pa.array([None] * frame.num_rows, type=pa.float64())
        elif name in {"promotion", "holiday"}:
            columns[name] = pa.array([None] * frame.num_rows, type=pa.bool_())
        else:
            raise ValueError(f"Observation frame is missing {name}.")
    return pa.table(columns)


def _select_keys(features: FeatureFrame, future: pa.Table) -> FeatureFrame:
    wanted = list(zip(_series_ids(future), _dates(future), strict=True))
    keys = zip(_series_ids(features.frame), _dates(features.frame), strict=True)
    index = {key: position for position, key in enumerate(keys)}
    positions: list[int] = []
    for key in wanted:
        found = index.get(key)
        if found is None:
            series_id, day = key
            raise ValueError(f"Features are missing series {series_id} on {day.isoformat()}.")
        positions.append(found)
    return FeatureFrame(
        frame=features.frame.take(pa.array(positions, type=pa.int64())),
        metadata=features.metadata,
        cutoff_date=features.cutoff_date,
    )


def _series_ids(frame: pa.Table) -> list[str]:
    if "series_id" in frame.column_names and "store_id" not in frame.column_names:
        values = frame.column("series_id").to_pylist()
        return [str(value) for value in values]
    stores = frame.column("store_id").to_pylist()
    skus = frame.column("sku_id").to_pylist()
    return [
        make_series_id(str(store_id), str(sku_id))
        for store_id, sku_id in zip(stores, skus, strict=True)
    ]


def _dates(frame: pa.Table) -> list[date]:
    column = frame.column("date")
    if not pa.types.is_date32(column.type):
        column = column.cast(pa.date32())
    values = column.to_pylist()
    if any(type(value) is not date for value in values):
        raise ValueError("Column date must contain calendar dates.")
    return cast(list[date], values)
