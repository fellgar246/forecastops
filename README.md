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

The web app opens on Overview. From there you can register a dataset, validate it, train a local model, approve it, and open a forecast. Cost reports local mode and does not track cloud spend.

`GET /health` returns `healthy` with `execution_mode=local` and the AWS, Bedrock, and SageMaker flags set to false.

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

`make aws-deploy`, `make aws-status`, `make aws-cost-check`, `make aws-destroy`, and `make aws-clean-artifacts` are reserved and do not call cloud APIs.

## Cloud profile

Terraform under `infra/` describes an optional cloud layout. The default `dev` and `demo` environments leave serverless endpoints, Bedrock, and schedules off, and they declare no training job, real-time endpoint, or notebook. Object storage in that layout blocks public access, encrypts objects, and expires non-current versions. A monthly budget of $5 alerts at $1, $3, and $5, plus when forecasted spend exceeds $5.

`make aws-plan` formats and validates that configuration. It does not create resources. Do not apply it until the budget email, GitHub repository, and bucket name prefix in the environment variables are yours.

Decisions are recorded in `docs/adr/`. The runtime layout is described in `docs/architecture/overview.md`.
