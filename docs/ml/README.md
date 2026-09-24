# Forecasting library

`forecastops_ml` is the importable library for dataset access, features, baselines, training, evaluation, batch inference, explanations, and pipelines.

Evaluation scores a forecaster on rolling-origin folds. Training, evaluation, and inference share `build_features`, so a lag or rolling window used at prediction time was knowable at the cutoff.

## Synthetic history

`forecastops_ml.data` builds a deterministic daily retail history for schema `retail-demand-v1`. The supported entry point is `make seed-data`. The default profile is the full demo catalog. Pass `PROFILE=test` for the small catalog used by unit tests. Regenerate the full dataset locally with:

```bash
make seed-data
```

## Dataset validation

`validate_dataset` checks an observation table against the store, SKU, and category dimensions. It returns a quality report with status `valid` or `invalid`.

Blocking findings set the status to `invalid`:

- duplicate `(date, store_id, sku_id)` keys
- missing values in required columns
- negative `units_sold` or `price`
- `promotion` set without `discount_pct`, or `discount_pct` outside 0-100
- `store_id` or `sku_id` absent from its dimension table
- a gap longer than one day inside a daily series
- `date_max` more than the staleness window before `as_of` (default 14 days)
- one day's `units_sold` above the outlier multiple of that series' median positive demand (default 20)

A `date_max` exactly on the staleness boundary is still fresh. A demand value equal to the outlier threshold is still allowed.

Advisory findings do not change the status. They record the stock-out rate, how many categories are absent from the category dimension, and the fraction of rows with zero demand. The report also includes rates for missing values, duplicate rows, and stock-outs.

Pass a directory to write `quality_report.json` next to the dataset. The file contains the status, finding codes, counts, and rates.

## Profiling

`build_profile` summarizes a daily observation table. `make profile-data` reads a generated dataset directory and writes `profile.json` next to it. Running it again on the same tables produces the same file.

The profile includes:

- demand distribution: count, mean, median, 90th percentile, and the share of zero-demand rows
- units sold by category
- weekday and month seasonality indexes, each a group's mean demand divided by the overall mean
- promotion lift by category: mean demand on promoted rows, mean demand on other rows, and their ratio
- stock-out prevalence, the fraction of rows with `stockout` set
- sparsity, the share of store-SKU-days whose demand is zero
- series length: minimum, median, and maximum observations per store and SKU

Weekday indexes start on Monday. A seasonality index is null when every observed day has zero demand. Promotion lift is null when a category has no promoted rows, no unpromoted rows, or the unpromoted mean is zero.

Weekly rollups are `aggregate_sku_week`, `aggregate_category_week`, `aggregate_store_week`, and `aggregate_region_week`. Each week starts on Monday. `units_sold` is the sum of the daily values, so the weekly total matches the daily total over the same dates. `price` is the demand-weighted average and is null when the units sum to zero. `promotion`, `holiday`, and `stockout` are true when any day in the week is true. Region-week uses `store_region` on the observations when that column is filled in; otherwise it uses the store dimension.

## Evaluation

`Backtester.evaluate` scores a `Forecaster` on a daily store-SKU observation table. The table needs `date`, `store_id`, `sku_id`, `category_id`, and `units_sold`.

Folds are rolling origins anchored on the latest dates. Each validation window is `horizon` distinct dates long, and those windows do not share dates. The next window starts after the previous origin. Training rows for a fold are every observation strictly before that origin, so an earlier window becomes history for the next fold. `predict` receives the validation keys and covariates, and it does not receive `units_sold`. A published score uses at least 3 folds. A final benchmark uses 5.

`ForecastFrame` carries `series_id`, `date`, and `p50`. `series_id` is `{store_id}|{sku_id}`. `p10` and `p90` are optional.

The report includes MAE, RMSE, WAPE, sMAPE, bias, and pinball loss. WAPE is `sum(|actual - forecast|) / sum(actual)` and is the primary business metric. sMAPE averages `2 * |actual - forecast| / (|actual| + |forecast|)`, and a term is 0 when both values are 0. Bias is `sum(forecast - actual) / sum(actual)`. Pinball loss at 0.1 uses `p10`, and pinball loss at 0.9 uses `p90`. Those fields are null when the model omits the quantile. If actual demand in a scored group sums to zero, evaluation stops with an error.

`runtime_ms` is the total fit time in milliseconds. `wape_by_horizon` repeats WAPE at each horizon step. `skipped_series` lists series left out of the score and the English reason for each one. A model that skips every series raises an error from `evaluate`. A catalog comparison omits that model and still returns the families that scored.

The same metrics are reported globally and by category, store, horizon step, and SKU demand quartile. A quartile is `q1` through `q4` from total units sold per SKU in that fold's training rows. A SKU on a percentile edge stays in the lower quartile.

## Baselines

`naive`, weekly `seasonal_naive`, `holt_winters`, and `gradient_boosting` implement `Forecaster` and register in `local_catalog` under `model_family`.

Naive forecasts `value(t)` at every horizon step. The point forecast is `p50`. `p10` and `p90` stay null.

Weekly seasonal naive is for daily history. The forecast for a day is `units_sold` seven days earlier (`season_length=7`, `period="day"`).

Yearly seasonal naive is for weekly history. The forecast for a week is the value 52 weeks earlier (`season_length=52`, `period="week"`). It uses the same `seasonal_naive` family. The default catalog does not register it. A frame with fewer than 52 distinct weeks is too short for that lag, so the default comparison scores only naive and weekly seasonal naive.

If the seasonal lag is missing, the step uses the series' last observation and `fallback_count` on the model increases by one. The model does not substitute zero unless that last observation is zero. A series with no history is an error. Fold 1 starts `fallback_count` over. Later folds on the same instance add to it.

Additive Holt-Winters (`holt_winters`) fits one model per category on a category-week aggregate. The seasonal period is 52. A category with fewer than 104 weeks is skipped. A missing week, more than one row for the same category and week, or a fit that does not converge is also skipped, with an English reason on `skipped_series`. The point forecast is `p50`. `p10` and `p90` stay null. The model does not fit one series per store and SKU.

`ModelCatalog.compare` calls the backtester once per registered model on the same rolling-origin folds. The comparison table has one row per scored family with WAPE, sMAPE, bias, and `runtime_ms`. A family that skips every series is omitted. `write_comparison` writes that table as `baseline_comparison.json`. Holt-Winters appears in that table when the frame is a category-week aggregate long enough to fit.

Reported scores use those rolling-origin folds. A random split would put later observations in training and earlier ones in the score, so it is not part of the evaluation API and it is not used to compare these models.

## Features

`build_features` is the only lag and rolling-window implementation. Training, backtesting, and inference call it. The cutoff is the last date whose target may enter a lag or a rolling window. A window uses observations strictly before the row date and on or before that cutoff. Calendar fields, planned `price`, planned `discount_pct`, planned `promotion`, and planned `holiday` come from the row being forecast. `units_sold`, `stock_available`, and `stockout` are not model inputs. Requesting one of them raises an error.

Feature metadata has `name`, `description`, `dtype`, `source`, and `available_at_prediction_time`. Every column the model uses has `available_at_prediction_time` set.

`grain` is `day` or `week`. Week is the default, and one period is seven days. `day_of_week` uses Monday as 0. `week_of_year` is the ISO week. `rolling_std_28` is the population standard deviation and is null when the window has fewer than two observations.

`aggregate_store_sku_week` rolls daily store-SKU rows to Monday-start weeks for this model. `units_sold` is a sum. `price` is the demand-weighted average and is null when the week's units sum to zero. `discount_pct` uses that same weight when units are positive, and the unweighted mean of known discounts when they are not. `promotion` and `holiday` are true when any day in the week is true.

## Gradient boosting

`gradient_boosting` fits one global LightGBM model. The library name is `lightgbm` on `training_config`. Hyperparameters are fixed. The default grain is store-SKU week, so the comparison frame is the weekly aggregate. Daily grain is available only when the forecaster is built with `profile="full"`. On a daily frame the default model skips every series and drops out of the comparison table.

Identifiers are categorical. Their codes are fit on the training rows of that fold. An unseen identifier at prediction time uses its own code.

Forecasts are `p50`. `p10` and `p90` stay null. The same fixture, config, and `RANDOM_SEED` (`20260921`) reproduce `p50` within a relative tolerance of 1e-6.

`attribute` returns SHAP values for one prediction row. `positive_drivers` and `negative_drivers` list at most five features each, ordered by magnitude. They are associations from the trees.
