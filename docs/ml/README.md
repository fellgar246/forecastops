# Forecasting library

`forecastops_ml` is the importable library for dataset access, features, baselines, training, evaluation, batch inference, explanations, drift monitoring, and pipelines.

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

## Global probabilistic model

`deepar` trains one model on many related series and returns P10, P50, and P90. The demo grain is week. Use the small test dataset, which contains more than one series. Leave the full daily fact table for local profiling.

```bash
make seed-data OUT=data/synthetic PROFILE=test
make train-deepar DATASET=data/synthetic
```

`make train-deepar` submits one training job when `SAGEMAKER_ENABLED`, `TRAINING_ENABLED`, and `AWS_ML_ENABLED` are all true, `ALLOW_GPU_TRAINING` is false, and the runtime and daily job values stay within the ceilings below. A closed flag prints an English reason and the process stops before the training API. Set those flags in the environment for the run you intend to start. CI does not run this command, and merging a pull request does not start a job.

The job uses one `ml.c5.xlarge` CPU instance. The runtime cap is 45 minutes (`MAX_TRAINING_RUNTIME_MINUTES`). At most 2 jobs can start in a day (`MAX_TRAINING_JOBS_PER_DAY`). Hyperparameters are fixed, including quantiles 0.1, 0.5, and 0.9. There is no hyperparameter search. The job does not create a real-time endpoint.

Each series is JSON lines with `start`, `target`, `cat` (store and category), and `dynamic_feat` (promotion and holiday). Week frequency is `1W` and day frequency is `1D`. The train target stops at the forecast horizon. Promotion and holiday for that horizon stay on the train channel when those dates are in the dataset. A horizon plan can add the same two features for periods after the last observation.

Channel files are written under `training/{job}/`. Model output is `models/{job}/`. The bucket is `ARTIFACTS_BUCKET`, the role is `DEEPAR_ROLE_ARN`, and the training image is the DeepAR algorithm image in `us-east-1`.

A forecast point is kept when `p10`, `p50`, and `p90` are finite and `p10 <= p50 <= p90`. Local tests use `FakeDeepARRunner`, which returns a fixture forecast with those quantiles and does not call a training account.

## Batch inference

A forecast run is `QUEUED`, then `RUNNING`, then `SUCCEEDED` or `FAILED`. The worker writes `forecasts/{id}/forecast.json` and loads those points into metadata. P10 and P90 are stored when the model emits them. A point forecast leaves them null and keeps the value in P50.

Local mode runs that same sequence in process. Catalog families are fit on the dataset and scored for the requested horizon. `deepar` without a cloud job uses a fake quantile model: P50 is the series' latest `units_sold`, P10 is 80 percent of that value, and P90 is 120 percent. The same series and horizon produce the same points.

Cloud mode, when `SAGEMAKER_ENABLED` is true and execution is AWS, submits one SageMaker batch transform for `deepar`. The output prefix is `forecasts/{id}/`. The instance type is `ml.c5.xlarge`. The adapter does not create a real-time or serverless endpoint. `ONLINE_INFERENCE` stays false, and a process with that flag set is refused before any client call. Tests use a fake transform client.

`MAX_BATCH_INFERENCE_JOBS_PER_DAY` limits how many forecast runs can be requested in a UTC day. The next request receives HTTP 429. `MAX_FORECAST_HORIZON_DAYS` limits the horizon. A client may send `idempotency_key`. Repeating that key returns the original run and does not start a second job. A failed run stores an English message. The message does not include dataset rows.

## Explanations

An explanation is written from a small forecast package: scope, quantile totals, recent history, named signals, and model metrics. Numerical demand stays on the forecast. The language model does not change it.

`BEDROCK_ENABLED=false` selects `MockExplanationClient`. It builds a schema-valid explanation from the package fields and does not open a remote client. `BEDROCK_ENABLED=true` selects `BedrockExplanationClient`. The model id comes from `BEDROCK_MODEL_ID`. The prompt version is `v1`. It forbids changing numerical predictions, inventing promotions, inventing causal claims, hiding uncertainty, and claiming certainty.

Tree-model signals may list `positive_drivers` and `negative_drivers`. Those names are associations from the trees. A global probabilistic model receives historical context, recent trend, known future covariates, seasonal pattern, uncertainty, and baseline comparison. Those names are context, not causes.

A draft is rejected when a forecast total in the text is not in the package, a named signal is not in the package, or `uncertainty_note` is empty. Rejected drafts increment `explanation_validation_failures` and are not returned as explanations. The same forecast, scope, and prompt version returns the stored explanation on the next request. `MAX_BEDROCK_CALLS_PER_DAY`, `MAX_BEDROCK_INPUT_TOKENS`, and `MAX_BEDROCK_OUTPUT_TOKENS` stop another provider call. Token counts and latency are logged. The full dataset and the raw prompt are not.

## Training pipeline

One manual command runs Validate, Process, Train, Evaluate, Quality Gate, and Register. Register runs only when the quality gate leaves the candidate pending approval. A rejected candidate is not registered, and the gate does not approve a model by itself.

```bash
make start-pipeline DATASET_VERSION=2026-09-21 GIT_SHA=$(git rev-parse --short HEAD) CONFIG=config.json
```

`config.json` is a JSON object. Set `grain`, `prediction_length`, `store_cardinality`, `category_cardinality`, and `series_count` there when the dataset is not the demo default (week, horizon 2, cardinalities 1, and 32 series). The command merges that file with the training limits: one `ml.c5.xlarge`, at most 45 minutes, fixed DeepAR hyperparameters, and no hyperparameter search. `make start-pipeline` starts one execution when `SAGEMAKER_ENABLED`, `TRAINING_ENABLED`, and `AWS_ML_ENABLED` are all true, `ALLOW_GPU_TRAINING` is false, and the runtime and daily job values stay within those ceilings. A closed flag prints an English reason and the process stops before the pipeline API. Set those flags in the environment for the run you intend to start. CI does not run this command. A pull request or a push does not start an execution.

Validate checks the dataset. Process writes the DeepAR channels. Both read the dataset directory mounted for that step. Train uses the same CPU job settings as the manual DeepAR job. Evaluate scores the model with rolling-origin folds and the shared metrics (at least 3 folds). The quality gate compares that score with the reference. Register records the model as `PendingManualApproval` only when the gate is waiting for a person. Evaluate and the quality gate do not run from the step command on their own; they run once training has produced a model and an evaluation report.

The bucket is `ARTIFACTS_BUCKET`. The pipeline role is `PIPELINE_ROLE_ARN`. The training step uses `DEEPAR_ROLE_ARN`. The region is `us-east-1`.

A finished execution stores the git SHA, dataset version, resolved configuration, metric document, and artifact locations on the training run. Those locations are the quality report, channel prefix, model prefix, and metrics file for that dataset version.

`enable_schedules` stays false in the default environment, so no rule starts this pipeline on a timer. Turning the flag on adds a monthly retrain request. That request stays `PENDING` until a person confirms it. Confirmation starts a training run and does not set the model to `PRODUCTION`.

## Promotion

A finished training job is registered as `TRAINED`. Recording its evaluation moves it to `EVALUATED`. Neither step sets `APPROVED`.

`gate` compares the candidate evaluation with a reference evaluation. The candidate must beat the reference on WAPE (`candidate < reference`), keep `abs(bias)` strictly below `0.05`, and avoid a category WAPE that worsens by more than `0.02`. A worsening of exactly `0.02` is still allowed. Categories present on only one report are not compared.

When the candidate report has a P90 pinball loss, pass empirical P90 coverage: the share of actuals at or below `p90`. That share must be at least `0.85`. `empirical_p90_coverage` computes it. When the pinball loss is null, the coverage clause is skipped and the decision stores `p90_coverage` as null. Gradient boosting takes that path.

A passing gate sets `PENDING_APPROVAL` and records no rejection reason. A failing gate sets `REJECTED` with reason `quality_gate` and does not ask for an actor. Segment findings name each regressed category. The decision stores the candidate id, reference id, threshold snapshot, boolean checks, status, reason, and timestamp.

`approve` moves `PENDING_APPROVAL` to `APPROVED` and requires an actor id. `reject` moves `PENDING_APPROVAL` to `REJECTED` with that actor id and reason `human`. `promote` moves `APPROVED` to `PRODUCTION` with an actor id and returns the previous production model of the same family to `APPROVED`. No other status enters production. Approval does not promote a model; production stays a separate human step.

When a model registry is enabled, the gate records the candidate there. A model waiting for a person is `PendingManualApproval`. A candidate the gate rejects is `Rejected`, so that outcome stays visible. `approve` sets the internal status to `APPROVED` and the registry status to `Approved` in the same operation, and stores the registry ARN, the registry version, and the actor id on the model version. `reject` does the same for `Rejected`. With no registry, `approve` and `reject` update only the internal row. `promote` leaves the registry status at `Approved`, which already covers a model in `PRODUCTION`.

A cloud forecast loads a model only when the registry status is `Approved` and the internal status is `APPROVED` or `PRODUCTION`. Pending and rejected registry versions are refused. Local forecasts keep using the internal status alone.

When no production evaluation exists, `reference_report` scores seasonal naive on the same frame and dataset version. Daily frames use a 7-day lag. Weekly frames use a 52-week lag. A production report is returned unchanged when one is supplied.

## Drift monitoring

Monitoring compares the most recent 28 days with the earlier rows of the same dataset. Those earlier rows are the training baseline. The recent window ends on the dataset's latest date and includes that date.

Price and `units_sold` use the population stability index and a Kolmogorov-Smirnov statistic. PSI bins the baseline at 10 quantile intervals, drops duplicate edges, and compares bin shares:

```text
sum((recent_share - baseline_share) * ln(recent_share / baseline_share))
```

A bin share below `0.0001` is raised to `0.0001`, then both share vectors are scaled so they sum to 1. Values outside the outer edges stay in the first or last bin. A constant baseline uses one bin for that value and a second bin for every other value. The Kolmogorov-Smirnov statistic is the largest absolute gap between the two empirical distribution functions.

Promotion frequency, zero-demand frequency, and stock-out frequency use the absolute difference of the two rates. Store and category coverage use the absolute difference of each label's share. A missing promotion or stock-out column is stored as null and does not change the status.

The status is `HEALTHY`, `WARNING`, or `RETRAIN_RECOMMENDED`. Retraining is recommended when any of these hold:

```text
recent WAPE is more than 20% worse than the approved model's WAPE
OR any PSI is above 0.2
OR dataset age is greater than 30 days
```

Dataset age is `as_of - date_max` in days. An age of 30 days is still healthy. A WAPE that is exactly 20% worse is still healthy. PSI of exactly 0.2 is a warning, not a retrain recommendation.

`WARNING` is a moderate shift that does not meet a retrain rule: PSI above 0.1 and at or below 0.2, or a frequency or coverage move above 0.05.

`PSI_RETRAIN_THRESHOLD` defaults to `0.2`. `PSI_WARNING_THRESHOLD` defaults to `0.1`. `WAPE_DEGRADATION_LIMIT` defaults to `0.2`. `MAX_DATASET_AGE_DAYS` defaults to `30`. `FREQUENCY_WARNING_DELTA` defaults to `0.05`. `RECENT_WINDOW_DAYS` defaults to `28`. Those values are in `.env.example`.

`POST /admin/monitoring` stores the report. `GET /metrics/model-performance` returns it with the model scores. When the status is `RETRAIN_RECOMMENDED`, the run opens a retrain request and stops. It does not start training and it does not approve a model. Confirm that request separately to train. The new model is still not promoted.

```bash
make monitor-drift
```
