# ADR 0008: Each organization is one tenant

## Context

Datasets, models, forecasts, and the objects behind them belong to one organization. A cloud deployment serves more than one organization from the same API and the same bucket. Local development still needs to run without a sign-in step.

## Decision

Cloud mode requires a Cognito access token. The tenant id is the `tenant_id` claim. A missing token is rejected. A valid token without that claim is rejected. Local mode can leave authentication off, and then every row is stamped with tenant `local`.

Every domain row stores `tenant_id`. Queries filter on the caller's tenant. Asking for another tenant's id returns the same not-found response as an unknown id.

Artifact keys start with `tenant/{tenant_id}/` and then one of `raw/`, `processed/`, `features/`, `training/`, `models/`, `forecasts/`, or `evaluations/`. The store refuses a key outside the caller's prefix.

The web app shows a sign-in form only when cloud auth is enabled, and then sends the access token. One user role is enough. Operators create users and set the tenant attribute. There is no self-service signup.

## Consequences

Existing local rows default to tenant `local`. Cloud deploys a user pool, a public app client, and a pre-token function that copies the tenant attribute onto the access token. Replacing the disk with object storage still uses the same store interface, with the tenant prefix added to every key.
