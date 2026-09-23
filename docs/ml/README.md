# Forecasting library

`forecastops_ml` is the importable library for dataset access, features, baselines, training, evaluation, batch inference, explanations, and pipelines.

Training, evaluation, and inference will share one feature path so a value used at prediction time is a value that was knowable at the cutoff. The modules are in place and do not train a model yet.

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
