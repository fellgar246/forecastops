# Forecasting library

`forecastops_ml` is the importable library for dataset access, features, baselines, training, evaluation, batch inference, explanations, and pipelines.

Training, evaluation, and inference will share one feature path so a value used at prediction time is a value that was knowable at the cutoff. The modules are in place and do not train a model yet.

## Synthetic history

`forecastops_ml.data` builds a deterministic daily retail history for schema `retail-demand-v1`. The supported entry point is `make seed-data`. The default profile is the full demo catalog. Pass `PROFILE=test` for the small catalog used by unit tests. Regenerate the full dataset locally with:

```bash
make seed-data
```
