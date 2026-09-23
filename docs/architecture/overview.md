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

The API reads one settings object. Missing required values stop startup with an English error. `GET /health` reports `healthy` and the cloud flags.

## Cloud profile

Cloud resources are optional. They are described in Terraform and are not created by `make test` or `make lint`. `make aws-plan` checks formatting and validates the configuration. It does not apply it. Deploy, status, cost, destroy, and artifact cleanup commands are not wired yet and do not call cloud APIs.

The default `dev` and `demo` environments keep these switches off:

```text
enable_serverless_endpoint = false
enable_bedrock = false
enable_schedules = false
```

The configuration declares no training job, real-time endpoint, notebook, or schedule. Object storage blocks public access and encrypts objects. Metadata sits in DynamoDB with encryption. A small Lambda and HTTP API expose health for a future control plane. Separate roles cover API execution, pipeline execution, training, inference, explanation, and deployment. Deployment assumes a role through GitHub's workload identity provider, not a long-lived access key.

A monthly budget of $5 alerts at $1, $3, and $5, and when forecasted spend exceeds $5. Log retention is 7 days in dev and 14 days in demo. Every taggable resource is tagged `Project=ml-demand-forecasting`, `Environment` set to the active environment, `Owner=portfolio`, `ManagedBy=terraform`, `CostCenter=learning`, and `AutoCleanup=true`.

Apply is intentionally not part of the local workflow. Before any future apply, replace the budget alert email, the GitHub repository name, and the bucket name prefix.
