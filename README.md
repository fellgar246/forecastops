# ForecastOps

ForecastOps is a retail demand forecasting platform. It produces batch forecasts with an uncertainty range, stores them as artifacts, and serves them from an API. A person approves a model before it is used in production. A language model may explain a finished forecast. It does not invent the numbers.

Cloud resources are optional. The default profile runs entirely on your machine.

## Local profile

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 20, Docker, and Terraform 1.5 or newer.

```bash
cp .env.example .env
uv sync
npm --prefix apps/web ci
docker compose up --build
```

- Web app: http://localhost:3000
- API health: http://localhost:8000/health

The web app opens on Overview. From there you can register a dataset, validate it, train a local model, approve it, and open a forecast. Cost reports local mode and does not track cloud spend. Local mode hides the sign-in form. Set `NEXT_PUBLIC_AUTH_ENABLED=true` and the Cognito client id only when the API is checking access tokens.

`GET /health` returns `healthy` with `execution_mode=local` and the AWS, Bedrock, and SageMaker flags set to false. Each response carries `X-Correlation-Id`. `GET /metrics` returns the same metric names the process writes to its JSON logs.

Quality checks:

```bash
make test
make lint
make aws-plan
```

## Synthetic data

Generate the demo retail history locally. The default profile writes a daily SKU-store fact table for 20 stores, 150 SKUs, 8 categories, and 24 months, plus store, SKU, and category dimensions. It reads `RANDOM_SEED` (default `20260921`) and refuses to exceed `MAX_DATASET_ROWS_DEMO`.

```bash
make seed-data
make seed-data OUT=data/synthetic PROFILE=test
```

`PROFILE=test` writes a tiny dataset (2 stores, 4 SKUs, 8 weeks) for tests. Unit tests use that profile and do not build the full history. Run `make seed-data` when you want the full catalog locally. Output lands in `data/synthetic` unless you set `OUT`:

- `observations.csv` and `observations.parquet` — one row per SKU, store, and day. `units_sold` is the target and `date` is a calendar date.
- `stores.csv` and `stores.parquet`
- `skus.csv` and `skus.parquet`
- `categories.csv` and `categories.parquet`
- `summary.json` — seed, row count, date range, and catalog counts

Profile that dataset with:

```bash
make profile-data OUT=data/synthetic
```

`make profile-data` reads the directory in `OUT` and writes `profile.json` beside the tables. The file describes the demand distribution, category volume, weekday and month seasonality, promotion lift by category, stock-out prevalence, sparsity, and series length. To profile the small test dataset:

```bash
make seed-data OUT=data/synthetic PROFILE=test
make profile-data OUT=data/synthetic
```

`make train-deepar` submits one CPU training job for a weekly small slice. It runs when `SAGEMAKER_ENABLED`, `TRAINING_ENABLED`, and `AWS_ML_ENABLED` are all true. With the default flags it prints a reason and does not call the training API. CI does not run it. The instance type, 45 minute cap, and manual steps are in `docs/ml/README.md`.

`make start-pipeline` starts one training pipeline for a dataset version, git SHA, and configuration file. It uses the same flags. With the defaults it prints a reason and does not call the pipeline API. CI does not run it. Schedules stay off unless `enable_schedules` is changed from its default of false.

`make refresh-forecast`, `make evaluate-forecasts`, and `make request-retrain` run the daily, weekly, and monthly operations on this machine. They do not call a cloud scheduler. The monthly command records a retrain request and waits. Confirming that request starts training and does not promote the model. How to turn the cloud clocks on for a showcase, and off again, is in `docs/architecture/overview.md`.

`make monitor-drift` compares the latest 28 days with the training baseline and stores a report. `GET /metrics/model-performance` returns that report with the model scores. The status is `HEALTHY`, `WARNING`, or `RETRAIN_RECOMMENDED`. A recommendation opens a retrain request and does not start training or approve a model. PSI above `0.2`, WAPE more than 20% worse than the approved model, or a dataset whose `date_max` is more than 30 days before the report date recommends retraining. The thresholds are in `.env.example`.

`make aws-deploy` packages the API and applies the `dev` environment. Set `ENV=demo` to apply demo instead. `make aws-status` prints that environment's outputs. `make aws-cost-check` reports the monthly budget alarms and does not start training. `make aws-destroy` destroys the disposable resources. `make aws-clean-artifacts` deletes leftover object versions in the data, artifacts, and forecasts buckets, then records the cleanup time for `GET /cost`. Run it before destroying again when those buckets still hold objects. `NAME_PREFIX` selects the bucket names and defaults to `forecastops`. The cost page shows the monthly budget, training runs this month, explanation usage today, which cloud capabilities are on, and that cleanup time. Estimated spend stays empty until billing is configured.

## Cloud profile

Terraform under `infra/` describes an optional cloud layout. The default `dev` and `demo` environments leave serverless endpoints, Bedrock, and schedules off, and they declare no training job, real-time endpoint, or notebook. Cloud mode adds a Cognito user pool. The API checks access tokens and keeps each organization's rows and objects under `tenant/{tenant_id}/`. A cloud `deepar` forecast uses batch transform rather than an endpoint. `ONLINE_INFERENCE` stays false. Object storage in that layout blocks public access, encrypts objects, and expires non-current versions. A monthly budget of $5 alerts at $1, $3, and $5, plus when forecasted spend exceeds $5.

`make aws-plan` formats and validates that configuration. It does not create resources. `make aws-deploy` does apply it. Replace the budget alert email, the GitHub repository, and the bucket name prefix before you deploy.

## Delivery

Pull requests run CI. That workflow lints and typechecks the backend, runs the backend tests, runs the ML unit tests and the leakage tests, lints and typechecks the web app, runs the web tests, checks Terraform formatting, validates the infrastructure, and scans the tree with Trivy.

A push to `main` builds the API package, applies `dev`, and calls `GET /health` and `GET /health/aws`. It does not start training or batch inference. `make aws-smoke` is that health check. It stops after those two routes.

Training starts only when someone runs the Train model workflow by hand. A push cannot start it. Run it from `main`, with a dataset version and the path of a training configuration file. The pipeline starts only when `SAGEMAKER_ENABLED`, `TRAINING_ENABLED`, and `AWS_ML_ENABLED` are true. Destroy demo is also manual and destroys the `demo` environment.

Those cloud workflows assume `GitHubDeployRole` with GitHub OpenID Connect. The repository does not store `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY`. The `dev` environment creates that role. Apply `dev` once from this machine before the first GitHub deploy, and set `github_repository` to `org/name`. Set the GitHub variable `AWS_ACCOUNT_ID` to the account that owns the role, and `AWS_REGION` when it is not `us-east-1`. For a cloud training run, also set `ARTIFACTS_BUCKET`, `PIPELINE_ROLE_ARN`, and `DEEPAR_ROLE_ARN`. With those flags left unset, the training command stops and does not call the training service.

Decisions are recorded in `docs/adr/`. The runtime layout is described in `docs/architecture/overview.md`.
