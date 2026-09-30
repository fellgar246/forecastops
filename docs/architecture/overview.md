# Architecture

ForecastOps produces retail demand forecasts as batch artifacts and serves them from an API. A person approves a model before it is used for production forecasts. A language model may explain a finished forecast. It does not produce the numbers.

## Local profile

This is the default. Copy `.env.example` and leave the cloud flags off.

```text
Browser → Next.js → FastAPI → PostgreSQL
                      ↓
                   local files and local models
```

`EXECUTION_MODE=local`. `AWS_ENABLED`, `BEDROCK_ENABLED`, and `SAGEMAKER_ENABLED` are false. Online inference is false. The web app, API, database, and tests run on the developer machine. No cloud credentials are required.

`make seed-data` writes the demo retail history as local files. The default run is the full daily catalog. Tests use a smaller profile and do not build that file.

The API reads one settings object. Missing required values stop startup with an English error. `GET /health` reports `healthy` and `execution_mode=local`. `GET /health/aws` reports each cloud integration. With the cloud flags off, training and forecasting stay in this process and write artifacts through a local directory store. Allowed locations are the prefixes `raw/`, `processed/`, `features/`, `training/`, `models/`, `forecasts/`, and `evaluations/`. Dataset files can be posted as a multipart upload to `POST /datasets/uploads`. The process does not construct a remote storage client and does not log object bodies.

Datasets, training runs, model versions, forecast runs, and promotion decisions are stored in PostgreSQL. A client registers a dataset, validates it, starts training, approves the model, and reads forecast points from `GET /forecasts/{id}/series`. Training and forecast requests return `202` with a job id. A forecast is stored as `QUEUED`, then the same worker resolves the approved model, runs inference, writes `forecasts/{id}/forecast.json`, and loads those points into the database before setting `SUCCEEDED`. A failed job is `FAILED` with an English message that does not include dataset rows. `MAX_BATCH_INFERENCE_JOBS_PER_DAY` rejects another request with `429`. An optional idempotency key returns the original run. `MAX_FORECAST_HORIZON_DAYS` still limits the horizon. Only a model in `APPROVED` or `PRODUCTION` can be used to forecast. In the cloud profile the model registry status must also be `Approved`. Local mode uses the local model and the directory store. `deepar` without a cloud job uses a fake quantile model that emits P10, P50, and P90. `POST /forecasts/{id}/explanation` writes a business explanation of that stored forecast. `BEDROCK_ENABLED=false` uses a local mock and does not call a remote model. The draft is checked against the forecast package: a forecast total that is not in the package, an unknown signal, or an empty uncertainty note is rejected and is not stored as a successful explanation. A repeated request for the same forecast, scope, and prompt version returns the stored row. Daily call and token ceilings reject another provider call with `429`. `GET /forecasts/{id}/explanation` returns the stored explanation. `AI_ENABLED=false` leaves the routes off.

## Cloud profile

Cloud resources are optional. They are described in Terraform and are not created by `make test` or `make lint`. `make aws-plan` checks formatting and validates the configuration. It does not apply it. Deploy, status, cost, destroy, and artifact cleanup commands are not wired yet and do not call cloud APIs.

The default `dev` and `demo` environments keep these switches off:

```text
enable_serverless_endpoint = false
enable_bedrock = false
enable_schedules = false
```

The configuration declares no training job, real-time endpoint, notebook, or schedule. When execution is in AWS and SageMaker is enabled, a `deepar` forecast submits one batch transform on `ml.c5.xlarge` and reads the artifact under `forecasts/`. That path does not create a real-time or serverless endpoint. `ONLINE_INFERENCE` stays false. If it is turned on, the batch adapter refuses the job before it opens a client. A separate manual command can submit one DeepAR training job on an `ml.c5.xlarge` for at most 45 minutes when `SAGEMAKER_ENABLED`, `TRAINING_ENABLED`, and `AWS_ML_ENABLED` are true. CI does not run that command. Another manual command starts one training pipeline with those same flags. Its steps are Validate, Process, Train, Evaluate, Quality Gate, and Register. Register is skipped when the quality gate rejects the candidate. CI does not run that command either. When execution is in AWS and SageMaker is enabled, the API uses the model registry: a gated candidate is recorded as `PendingManualApproval` or `Rejected`, and a person's approval sets both the internal status and the registry status to approved. A cloud forecast is refused unless the registry status is `Approved` and the internal status is `APPROVED` or `PRODUCTION`. Local mode does not open that client. `enable_schedules` stays false, so no rule starts the pipeline on a timer. Object storage blocks public access, encrypts objects with SSE-S3, keeps versions, and expires non-current versions after 30 days. Those buckets are described for a later deploy and are not created by local tests. When `AWS_ENABLED=true`, the API selects one artifacts bucket and `POST /datasets/upload-url` issues a pre-signed upload URL instead of accepting the file body. Metadata sits in DynamoDB with encryption. A small Lambda and HTTP API expose health for a future control plane. Separate roles cover API execution, pipeline execution, training, inference, explanation, and deployment. Deployment assumes a role through GitHub's workload identity provider, not a long-lived access key.

A monthly budget of $5 alerts at $1, $3, and $5, and when forecasted spend exceeds $5. Log retention is 7 days in dev and 14 days in demo. Every taggable resource is tagged `Project=ml-demand-forecasting`, `Environment` set to the active environment, `Owner=portfolio`, `ManagedBy=terraform`, `CostCenter=learning`, and `AutoCleanup=true`.

Apply is intentionally not part of the local workflow. Before any future apply, replace the budget alert email, the GitHub repository name, and the bucket name prefix.
