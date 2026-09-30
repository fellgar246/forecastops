# ADR 0006: Cloud spend has ceilings and kill switches

## Context

Training jobs, endpoints, notebooks, logs, and language-model calls can accumulate cost after a short demo. Price lists are not a project budget.

## Decision

Project ceilings, which are internal targets and not vendor price guarantees:

- Local development aims for about $0 of cloud spend.
- Occasional demonstration use targets at most $5 USD in a month.
- A temporary showcase has a hard internal target of at most $10 USD in a month.

The infrastructure defines a monthly budget of $5 with alerts at $1, $3, and $5 of actual spend, plus an alert when forecasted spend goes above $5. Training defaults are 2 jobs a day, 45 minutes each, on CPU only. Language-model calls, batch inference jobs, dataset size, and forecast horizon have their own ceilings. `AWS_ML_ENABLED`, `BEDROCK_ENABLED`, and `TRAINING_ENABLED` can be turned off independently. Training is started by an explicit manual workflow, never by an ordinary push to the main branch. Log retention is 7 days in dev and 14 days in demo.

## Consequences

The default environment declares no training job, real-time endpoint, or notebook. Schedules are created only when `enable_schedules` is true, and that variable defaults to false. Spend requires a deliberate flag change. Alerts fire before the monthly ceiling. Local development keeps working when the cloud switches are off.
