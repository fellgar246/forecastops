# ADR 0002: Local execution is the default

## Context

Day-to-day development should not need cloud credentials or cloud spend. A profile that fails closed to the network is easier to test and safer to leave running.

## Decision

The default profile runs the web app, the API, PostgreSQL, files, and models on the developer machine. `AWS_ENABLED`, `BEDROCK_ENABLED`, and `SAGEMAKER_ENABLED` default to false. Turning every cloud integration off must leave the local product usable.

## Consequences

Lint, tests, and the local stack run without an account. Cloud behavior is a separate profile and is never assumed by the default commands. Missing required settings stop the API at startup instead of silently picking a cloud path.
